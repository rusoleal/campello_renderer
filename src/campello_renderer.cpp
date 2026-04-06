#include <campello_renderer/campello_renderer.hpp>
#include "campello_renderer_config.h"
#include <campello_gpu/constants/buffer_usage.hpp>
#include <campello_gpu/constants/compare_op.hpp>
#include <campello_gpu/constants/pixel_format.hpp>
#include <campello_gpu/constants/texture_usage.hpp>
#include <campello_gpu/constants/aspect.hpp>
#include <campello_gpu/constants/filter_mode.hpp>
#include <campello_gpu/constants/wrap_mode.hpp>
#include <campello_gpu/constants/shader_stage.hpp>
#include <campello_gpu/descriptors/begin_render_pass_descriptor.hpp>
#include <campello_gpu/descriptors/render_pipeline_descriptor.hpp>
#include <campello_gpu/descriptors/sampler_descriptor.hpp>
#include <campello_gpu/descriptors/bind_group_layout_descriptor.hpp>
#include <campello_gpu/descriptors/bind_group_descriptor.hpp>
#include <campello_gpu/bind_group.hpp>
#include <cmath>

#if defined(__APPLE__)
#include "shaders/metal_default.h"
#elif defined(ANDROID)
#include "shaders/vulkan_default.h"
#elif defined(_WIN32)
#include "shaders/directx_default.h"
#endif

#include <campello_image/image.hpp>

using namespace systems::leal::campello_renderer;

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

Renderer::Renderer(std::shared_ptr<systems::leal::campello_gpu::Device> device) {
    this->device = device;
}

// ---------------------------------------------------------------------------
// Asset / scene / camera
// ---------------------------------------------------------------------------

void Renderer::setAsset(std::shared_ptr<systems::leal::gltf::GLTF> asset) {
    this->asset = asset;
    if (asset == nullptr) {
        images.clear();
        nodeTransforms.clear();
        transformBuffer       = nullptr;
        materialUniformBuffer = nullptr;
        boundsRadius          = 1.0f;
        gpuSamplers.clear();
        gpuBindGroups.clear();
        return;
    }

    size_t imageCount = asset->images ? asset->images->size() : 0;
    size_t bufferCount = asset->buffers ? asset->buffers->size() : 0;
    images      = std::vector<std::shared_ptr<Image>>(imageCount, nullptr);
    gpuBuffers  = std::vector<std::shared_ptr<systems::leal::campello_gpu::Buffer>>(bufferCount, nullptr);
    gpuTextures = std::vector<std::shared_ptr<systems::leal::campello_gpu::Texture>>(imageCount, nullptr);

    if (asset->scenes->size() > 0) {
        if (asset->scene == -1) {
            setScene(0);
        } else {
            setScene(asset->scene);
        }
    }
}

std::shared_ptr<systems::leal::gltf::GLTF> Renderer::getAsset() {
    return asset;
}

void Renderer::setCamera(uint32_t index) {
    cameraIndex = index;
}

void Renderer::setScene(uint32_t index) {
    if (asset == nullptr) return;
    if (device == nullptr) return;

    auto info = asset->getRuntimeInfo(index);
    if (info == nullptr) return;

    sceneIndex = index;

    // Upload GLTF binary buffers referenced by this scene.
    for (int b = 0; b < (int)info->buffers.size(); b++) {
        if (info->buffers[b] && gpuBuffers[b] == nullptr) {
            auto &buf = (*asset->buffers)[b];
            if (!buf.data.empty()) {
                using BU = systems::leal::campello_gpu::BufferUsage;
                auto usage = (BU)((uint32_t)BU::vertex | (uint32_t)BU::index);
                gpuBuffers[b] = device->createBuffer(
                    buf.data.size(), usage,
                    const_cast<uint8_t *>(buf.data.data()));
            }
        }
    }

    // Decode and upload images referenced by this scene.
    for (int a = 0; a < (int)info->images.size(); a++) {
        if (info->images[a]) {
            if (gpuTextures[a] == nullptr) {
                auto &image = (*asset->images)[a];
                if (image.data.size() > 0) {
                    // TODO: handle data:uri images
                } else if (image.bufferView != -1) {
                    auto &bufferView = (*asset->bufferViews)[image.bufferView];
                    auto &buffer     = (*asset->buffers)[bufferView.buffer];
                    if (!buffer.data.empty()) {
                        const uint8_t *src = buffer.data.data() + bufferView.byteOffset;
                        auto img = systems::leal::campello_image::Image::fromMemory(src, bufferView.byteLength);
                        if (img != nullptr) {
                            auto texture = device->createTexture(
                                systems::leal::campello_gpu::TextureType::tt2d,
                                systems::leal::campello_gpu::PixelFormat::rgba8unorm,
                                img->getWidth(), img->getHeight(), 1, 1, 1,
                                systems::leal::campello_gpu::TextureUsage::textureBinding);
                            if (texture != nullptr) {
                                texture->upload(0, img->getDataSize(), const_cast<uint8_t*>(img->getData()));
                                gpuTextures[a] = texture;
                            }
                        }
                    }
                }
            }
        } else {
            gpuTextures[a] = nullptr;
        }
    }

    // Allocate transform buffer — one float4x4 (64 bytes) per node.
    if (asset->nodes && !asset->nodes->empty()) {
        size_t nodeCount = asset->nodes->size();
        nodeTransforms.assign(nodeCount * 16, 0.0f);
        using BU = systems::leal::campello_gpu::BufferUsage;
        transformBuffer = device->createBuffer(nodeCount * 64, BU::vertex);
    }

    // ------------------------------------------------------------------
    // Lazy-initialize shared texture resources (once per device lifetime).
    // ------------------------------------------------------------------
    namespace GPU = systems::leal::campello_gpu;

    if (!bindGroupLayout) {
        GPU::BindGroupLayoutDescriptor bglDesc{};

        GPU::EntryObject texEntry{};
        texEntry.binding    = 0;
        texEntry.visibility = GPU::ShaderStage::fragment;
        texEntry.type       = GPU::EntryObjectType::texture;
        texEntry.data.texture.multisampled = false;
        texEntry.data.texture.sampleType   = GPU::EntryObjectTextureType::ttFloat;
        texEntry.data.texture.viewDimension = GPU::TextureType::tt2d;
        bglDesc.entries.push_back(texEntry);

        GPU::EntryObject sampEntry{};
        sampEntry.binding    = 1;
        sampEntry.visibility = GPU::ShaderStage::fragment;
        sampEntry.type       = GPU::EntryObjectType::sampler;
        sampEntry.data.sampler.type = GPU::EntryObjectSamplerType::filtering;
        bglDesc.entries.push_back(sampEntry);

        bindGroupLayout = device->createBindGroupLayout(bglDesc);
    }

    if (!defaultSampler) {
        GPU::SamplerDescriptor sd{};
        sd.addressModeU  = GPU::WrapMode::repeat;
        sd.addressModeV  = GPU::WrapMode::repeat;
        sd.addressModeW  = GPU::WrapMode::repeat;
        sd.magFilter     = GPU::FilterMode::fmLinear;
        sd.minFilter     = GPU::FilterMode::fmLinear;
        sd.lodMinClamp   = 0.0;
        sd.lodMaxClamp   = 1000.0;
        sd.maxAnisotropy = 1.0;
        defaultSampler = device->createSampler(sd);
    }

    if (!defaultTexture) {
        uint8_t white[4] = {255, 255, 255, 255};
        defaultTexture = device->createTexture(
            GPU::TextureType::tt2d, GPU::PixelFormat::rgba8unorm,
            1, 1, 1, 1, 1, GPU::TextureUsage::textureBinding);
        if (defaultTexture) defaultTexture->upload(0, 4, white);
    }

    if (!defaultBindGroup && bindGroupLayout && defaultTexture && defaultSampler) {
        GPU::BindGroupDescriptor bgDesc{};
        bgDesc.layout  = bindGroupLayout;
        bgDesc.entries = { {0, defaultTexture}, {1, defaultSampler} };
        defaultBindGroup = device->createBindGroup(bgDesc);
    }

    // ------------------------------------------------------------------
    // Build GPU samplers from GLTF samplers.
    // ------------------------------------------------------------------
    gpuSamplers.clear();
    if (asset->samplers) {
        gpuSamplers.resize(asset->samplers->size());
        for (size_t s = 0; s < asset->samplers->size(); ++s) {
            auto &gs = (*asset->samplers)[s];
            GPU::SamplerDescriptor sd{};
            sd.addressModeU  = static_cast<GPU::WrapMode>(gs.wrapS);
            sd.addressModeV  = static_cast<GPU::WrapMode>(gs.wrapT);
            sd.addressModeW  = GPU::WrapMode::repeat;
            sd.magFilter     = (gs.magFilter == systems::leal::gltf::FilterMode::fmNearest)
                                   ? GPU::FilterMode::fmNearest : GPU::FilterMode::fmLinear;
            sd.minFilter     = (gs.minFilter == systems::leal::gltf::FilterMode::fmNearest ||
                                gs.minFilter == systems::leal::gltf::FilterMode::fmNearestMipmapNearest ||
                                gs.minFilter == systems::leal::gltf::FilterMode::fmNearestMipmapLinear)
                                   ? GPU::FilterMode::fmNearest : GPU::FilterMode::fmLinear;
            sd.lodMinClamp   = 0.0;
            sd.lodMaxClamp   = 1000.0;
            sd.maxAnisotropy = 1.0;
            gpuSamplers[s]   = device->createSampler(sd);
        }
    }

    // ------------------------------------------------------------------
    // Build one bind group per GLTF texture, pairing image + sampler.
    // ------------------------------------------------------------------
    gpuBindGroups.clear();
    if (asset->textures && bindGroupLayout) {
        gpuBindGroups.resize(asset->textures->size());
        for (size_t t = 0; t < asset->textures->size(); ++t) {
            auto &gt = (*asset->textures)[t];

            // Prefer WebP source when available and uploaded.
            int64_t imgIdx = (gt.ext_texture_webp >= 0) ? gt.ext_texture_webp : gt.source;
            std::shared_ptr<GPU::Texture> tex =
                (imgIdx >= 0 && imgIdx < (int64_t)gpuTextures.size() && gpuTextures[imgIdx])
                    ? gpuTextures[imgIdx] : defaultTexture;

            std::shared_ptr<GPU::Sampler> samp =
                (gt.sampler >= 0 && (size_t)gt.sampler < gpuSamplers.size() && gpuSamplers[gt.sampler])
                    ? gpuSamplers[gt.sampler] : defaultSampler;

            if (tex && samp) {
                GPU::BindGroupDescriptor bgDesc{};
                bgDesc.layout  = bindGroupLayout;
                bgDesc.entries = { {0, tex}, {1, samp} };
                gpuBindGroups[t] = device->createBindGroup(bgDesc);
            }
        }
    }

    // ------------------------------------------------------------------
    // Compute approximate scene bounding radius from node positions.
    // ------------------------------------------------------------------
    boundsRadius = 1.0f;
    auto &scene0 = (*asset->scenes)[sceneIndex];
    if (scene0.nodes) {
        namespace VM = systems::leal::vector_math;
        for (auto rootIdx : *scene0.nodes) {
            computeSceneBounds(rootIdx, VM::Matrix4<double>::identity());
        }
    }

    // ------------------------------------------------------------------
    // Create material uniform buffer: slot 0 = default [1,1,1,1],
    // slots 1..N = per-material baseColorFactor (16 bytes each).
    // ------------------------------------------------------------------
    {
        size_t matCount = asset->materials ? asset->materials->size() : 0;
        uint64_t bufSize = (uint64_t)(matCount + 1) * 16;
        materialUniformBuffer = device->createBuffer(bufSize, GPU::BufferUsage::vertex);
    }
}

// ---------------------------------------------------------------------------
// Pipeline / resize
// ---------------------------------------------------------------------------

void Renderer::createDefaultPipelines(systems::leal::campello_gpu::PixelFormat colorFormat) {
    namespace GPU = systems::leal::campello_gpu;

#if defined(__APPLE__)
    using namespace systems::leal::campello_renderer::shaders;

    // Load the embedded .metallib into a ShaderModule.
    auto shaderModule = device->createShaderModule(kDefaultMetalShader, kDefaultMetalShaderSize);
    if (!shaderModule) return;

    // --- Base pipeline descriptor (vertex stage + depth/stencil + rasterization,
    //     shared between both variants). ---
    GPU::RenderPipelineDescriptor base{};

    base.vertex.module     = shaderModule;
    base.vertex.entryPoint = "vertexMain";

    // Slots 0–3: stage_in attributes. Slots 16/17 are raw [[buffer(N)]] in the
    // shader and are not part of the vertex descriptor.
    auto makeLayout = [](GPU::ComponentType ct, GPU::AccessorType at,
                         double stride, GPU::StepMode sm, uint32_t location) {
        GPU::VertexLayout layout{};
        layout.arrayStride = stride;
        layout.stepMode    = sm;
        GPU::VertexAttribute attr{};
        attr.componentType  = ct;
        attr.accessorType   = at;
        attr.offset         = 0;
        attr.shaderLocation = location;
        layout.attributes.push_back(attr);
        return layout;
    };

    base.vertex.buffers.push_back(makeLayout(
        GPU::ComponentType::ctFloat, GPU::AccessorType::acVec3,
        12.0, GPU::StepMode::vertex, VERTEX_SLOT_POSITION));
    base.vertex.buffers.push_back(makeLayout(
        GPU::ComponentType::ctFloat, GPU::AccessorType::acVec3,
        12.0, GPU::StepMode::vertex, VERTEX_SLOT_NORMAL));
    base.vertex.buffers.push_back(makeLayout(
        GPU::ComponentType::ctFloat, GPU::AccessorType::acVec2,
        8.0, GPU::StepMode::vertex, VERTEX_SLOT_TEXCOORD0));
    base.vertex.buffers.push_back(makeLayout(
        GPU::ComponentType::ctFloat, GPU::AccessorType::acVec4,
        16.0, GPU::StepMode::vertex, VERTEX_SLOT_TANGENT));

    GPU::DepthStencilDescriptor ds{};
    ds.format              = GPU::PixelFormat::depth32float;
    ds.depthWriteEnabled   = true;
    ds.depthCompare        = GPU::CompareOp::less;
    ds.depthBias           = 0.0;
    ds.depthBiasClamp      = 0.0;
    ds.depthBiasSlopeScale = 0.0;
    ds.stencilReadMask     = 0xFFFFFFFF;
    ds.stencilWriteMask    = 0xFFFFFFFF;
    base.depthStencil      = ds;

    base.topology  = GPU::PrimitiveTopology::triangleList;
    base.cullMode  = GPU::CullMode::back;
    base.frontFace = GPU::FrontFace::ccw;

    GPU::ColorState cs{};
    cs.format    = colorFormat;
    cs.writeMask = GPU::ColorWrite::all;

    // --- Variant: flat (Phong + baseColorFactor, no texture) ---
    {
        GPU::RenderPipelineDescriptor d = base;
        GPU::FragmentDescriptor frag{};
        frag.module     = shaderModule;
        frag.entryPoint = "fragmentMain_flat";
        frag.targets.push_back(cs);
        d.fragment = frag;
        pipelineFlat = device->createRenderPipeline(d);
    }

    // --- Variant: textured (Phong + baseColorTexture × baseColorFactor) ---
    {
        GPU::RenderPipelineDescriptor d = base;
        GPU::FragmentDescriptor frag{};
        frag.module     = shaderModule;
        frag.entryPoint = "fragmentMain_textured";
        frag.targets.push_back(cs);
        d.fragment = frag;
        pipelineTextured = device->createRenderPipeline(d);
    }

#elif defined(ANDROID)
    using namespace systems::leal::campello_renderer::shaders;

    // Load separate vertex and fragment SPIR-V modules.
    auto vertModule = device->createShaderModule(kDefaultVulkanVertShader, kDefaultVulkanVertShaderSize);
    auto fragModule = device->createShaderModule(kDefaultVulkanFragShader, kDefaultVulkanFragShaderSize);
    if (!vertModule || !fragModule) return;

    GPU::RenderPipelineDescriptor desc{};

    // --- Vertex stage ---
    // Slots 0–3: per-vertex attributes (POSITION, NORMAL, TEXCOORD_0, TANGENT).
    // Slot 16:   per-instance MVP mat4 (locations 16–19 in GLSL).
    //
    // NOTE: campello_gpu Vulkan backend currently hardcodes
    // vertexBindingDescriptionCount = 0, so the vertex input descriptors below
    // are passed to the API but not yet applied. The pipeline will work correctly
    // once that upstream gap is resolved.
    desc.vertex.module     = vertModule;
    desc.vertex.entryPoint = "main";

    auto makeLayout = [](GPU::ComponentType ct, GPU::AccessorType at,
                         double stride, GPU::StepMode sm, uint32_t location) {
        GPU::VertexLayout layout{};
        layout.arrayStride = stride;
        layout.stepMode    = sm;
        GPU::VertexAttribute attr{};
        attr.componentType  = ct;
        attr.accessorType   = at;
        attr.offset         = 0;
        attr.shaderLocation = location;
        layout.attributes.push_back(attr);
        return layout;
    };

    // Slot 0 — POSITION  vec3  layout(location = 0)
    desc.vertex.buffers.push_back(makeLayout(
        GPU::ComponentType::ctFloat, GPU::AccessorType::acVec3,
        12.0, GPU::StepMode::vertex, VERTEX_SLOT_POSITION));
    // Slot 1 — NORMAL    vec3  layout(location = 1)
    desc.vertex.buffers.push_back(makeLayout(
        GPU::ComponentType::ctFloat, GPU::AccessorType::acVec3,
        12.0, GPU::StepMode::vertex, VERTEX_SLOT_NORMAL));
    // Slot 2 — TEXCOORD_0 vec2  layout(location = 2)
    desc.vertex.buffers.push_back(makeLayout(
        GPU::ComponentType::ctFloat, GPU::AccessorType::acVec2,
        8.0, GPU::StepMode::vertex, VERTEX_SLOT_TEXCOORD0));
    // Slot 3 — TANGENT   vec4  layout(location = 3)
    desc.vertex.buffers.push_back(makeLayout(
        GPU::ComponentType::ctFloat, GPU::AccessorType::acVec4,
        16.0, GPU::StepMode::vertex, VERTEX_SLOT_TANGENT));

    // Slots 4–15: unused, filled with empty placeholder layouts so that
    // the vector index lines up with the buffer slot number.
    for (int i = 4; i < (int)VERTEX_SLOT_MVP; i++) {
        GPU::VertexLayout empty{};
        empty.arrayStride = 0;
        empty.stepMode    = GPU::StepMode::vertex;
        desc.vertex.buffers.push_back(empty);
    }

    // Slot 16 — MVP mat4, per-instance (4 consecutive locations: 16–19).
    // Each column of the mat4 is declared as a separate vec4 attribute.
    {
        GPU::VertexLayout mvpLayout{};
        mvpLayout.arrayStride = 64; // sizeof(float4x4)
        mvpLayout.stepMode    = GPU::StepMode::instance;
        for (uint32_t col = 0; col < 4; col++) {
            GPU::VertexAttribute attr{};
            attr.componentType  = GPU::ComponentType::ctFloat;
            attr.accessorType   = GPU::AccessorType::acVec4;
            attr.offset         = col * 16; // 4 floats * 4 bytes per column
            attr.shaderLocation = VERTEX_SLOT_MVP + col; // locations 16, 17, 18, 19
            mvpLayout.attributes.push_back(attr);
        }
        desc.vertex.buffers.push_back(mvpLayout);
    }

    // --- Fragment stage ---
    GPU::FragmentDescriptor frag{};
    frag.module     = fragModule;
    frag.entryPoint = "main";
    GPU::ColorState cs{};
    cs.format    = colorFormat;
    cs.writeMask = GPU::ColorWrite::all;
    frag.targets.push_back(cs);
    desc.fragment = frag;

    // --- Depth/stencil ---
    GPU::DepthStencilDescriptor ds{};
    ds.format              = GPU::PixelFormat::depth32float;
    ds.depthWriteEnabled   = true;
    ds.depthCompare        = GPU::CompareOp::less;
    ds.depthBias           = 0.0;
    ds.depthBiasClamp      = 0.0;
    ds.depthBiasSlopeScale = 0.0;
    ds.stencilReadMask     = 0xFFFFFFFF;
    ds.stencilWriteMask    = 0xFFFFFFFF;
    desc.depthStencil      = ds;

    // --- Rasterization ---
    desc.topology  = GPU::PrimitiveTopology::triangleList;
    desc.cullMode  = GPU::CullMode::back;
    desc.frontFace = GPU::FrontFace::ccw;

    // Vulkan: assign same pipeline to both variants until separate SPIR-V
    // fragment shaders are compiled (flat variant TODO).
    pipelineTextured = device->createRenderPipeline(desc);
    pipelineFlat     = pipelineTextured;

#elif defined(_WIN32)
    using namespace systems::leal::campello_renderer::shaders;

    // DXIL binaries not yet compiled — pipelines remain null.
    // See shaders/directx/default.hlsl and src/shaders/directx_default.h.
    if (kDefaultDirectXVertShaderSize == 0 || kDefaultDirectXPixelShaderSize == 0)
        return;

    auto vertModule = device->createShaderModule(kDefaultDirectXVertShader, kDefaultDirectXVertShaderSize);
    auto pixelModule = device->createShaderModule(kDefaultDirectXPixelShader, kDefaultDirectXPixelShaderSize);
    if (!vertModule || !pixelModule) return;

    GPU::RenderPipelineDescriptor desc{};

    // --- Vertex stage ---
    // campello_gpu DirectX backend maps attrs to SemanticName="TEXCOORD",
    // SemanticIndex=shaderLocation — the HLSL uses TEXCOORD semantics throughout.
    desc.vertex.module     = vertModule;
    desc.vertex.entryPoint = "vertexMain";

    auto makeLayout = [](GPU::ComponentType ct, GPU::AccessorType at,
                         double stride, GPU::StepMode sm, uint32_t location) {
        GPU::VertexLayout layout{};
        layout.arrayStride = stride;
        layout.stepMode    = sm;
        GPU::VertexAttribute attr{};
        attr.componentType  = ct;
        attr.accessorType   = at;
        attr.offset         = 0;
        attr.shaderLocation = location;
        layout.attributes.push_back(attr);
        return layout;
    };

    // Slot 0 — POSITION  float3  TEXCOORD0
    desc.vertex.buffers.push_back(makeLayout(
        GPU::ComponentType::ctFloat, GPU::AccessorType::acVec3,
        12.0, GPU::StepMode::vertex, VERTEX_SLOT_POSITION));
    // Slot 1 — NORMAL    float3  TEXCOORD1
    desc.vertex.buffers.push_back(makeLayout(
        GPU::ComponentType::ctFloat, GPU::AccessorType::acVec3,
        12.0, GPU::StepMode::vertex, VERTEX_SLOT_NORMAL));
    // Slot 2 — TEXCOORD_0 float2  TEXCOORD2
    desc.vertex.buffers.push_back(makeLayout(
        GPU::ComponentType::ctFloat, GPU::AccessorType::acVec2,
        8.0, GPU::StepMode::vertex, VERTEX_SLOT_TEXCOORD0));
    // Slot 3 — TANGENT   float4  TEXCOORD3
    desc.vertex.buffers.push_back(makeLayout(
        GPU::ComponentType::ctFloat, GPU::AccessorType::acVec4,
        16.0, GPU::StepMode::vertex, VERTEX_SLOT_TANGENT));

    // Slots 4–15: empty placeholder layouts to align slot index with buffer index.
    for (int i = 4; i < (int)VERTEX_SLOT_MVP; i++) {
        GPU::VertexLayout empty{};
        empty.arrayStride = 0;
        empty.stepMode    = GPU::StepMode::vertex;
        desc.vertex.buffers.push_back(empty);
    }

    // Slot 16 — MVP mat4 split into 4 float4 rows, per-instance (TEXCOORD16–19).
    {
        GPU::VertexLayout mvpLayout{};
        mvpLayout.arrayStride = 64; // sizeof(float4x4)
        mvpLayout.stepMode    = GPU::StepMode::instance;
        for (uint32_t row = 0; row < 4; row++) {
            GPU::VertexAttribute attr{};
            attr.componentType  = GPU::ComponentType::ctFloat;
            attr.accessorType   = GPU::AccessorType::acVec4;
            attr.offset         = row * 16; // 4 floats * 4 bytes per row
            attr.shaderLocation = VERTEX_SLOT_MVP + row; // TEXCOORD16–19
            mvpLayout.attributes.push_back(attr);
        }
        desc.vertex.buffers.push_back(mvpLayout);
    }

    // --- Fragment stage ---
    GPU::FragmentDescriptor frag{};
    frag.module     = pixelModule;
    frag.entryPoint = "pixelMain";
    GPU::ColorState cs{};
    cs.format    = colorFormat;
    cs.writeMask = GPU::ColorWrite::all;
    frag.targets.push_back(cs);
    desc.fragment = frag;

    // --- Depth/stencil ---
    GPU::DepthStencilDescriptor ds{};
    ds.format              = GPU::PixelFormat::depth32float;
    ds.depthWriteEnabled   = true;
    ds.depthCompare        = GPU::CompareOp::less;
    ds.depthBias           = 0.0;
    ds.depthBiasClamp      = 0.0;
    ds.depthBiasSlopeScale = 0.0;
    ds.stencilReadMask     = 0xFFFFFFFF;
    ds.stencilWriteMask    = 0xFFFFFFFF;
    desc.depthStencil      = ds;

    // --- Rasterization ---
    desc.topology  = GPU::PrimitiveTopology::triangleList;
    desc.cullMode  = GPU::CullMode::back;
    desc.frontFace = GPU::FrontFace::ccw;

    // DirectX: assign same pipeline to both variants (flat TODO).
    pipelineTextured = device->createRenderPipeline(desc);
    pipelineFlat     = pipelineTextured;

#else
    (void)colorFormat;
#endif

    // Fallback UV buffer — all zeros, used for primitives without TEXCOORD_0.
    if (!fallbackUVBuffer) {
        constexpr uint64_t kFallbackUVSize = 256 * 1024; // 32 768 float2 values
        std::vector<uint8_t> zeros(kFallbackUVSize, 0);
        fallbackUVBuffer = device->createBuffer(
            kFallbackUVSize, GPU::BufferUsage::vertex, zeros.data());
    }
}

void Renderer::resize(uint32_t width, uint32_t height) {
    renderWidth  = width;
    renderHeight = height;

    if (width == 0 || height == 0) {
        depthTexture = nullptr;
        depthView    = nullptr;
        return;
    }

    namespace GPU = systems::leal::campello_gpu;
    using TU = GPU::TextureUsage;

    depthTexture = device->createTexture(
        GPU::TextureType::tt2d,
        GPU::PixelFormat::depth32float,
        width, height, 1, 1, 1,
        (TU)(uint32_t(TU::renderTarget)));

    if (depthTexture) {
        depthView = depthTexture->createView(
            GPU::PixelFormat::depth32float,
            1,
            GPU::Aspect::depthOnly,
            0, 0,
            GPU::TextureType::tt2d);
    }
}

// ---------------------------------------------------------------------------
// Transform helpers
// ---------------------------------------------------------------------------

systems::leal::vector_math::Matrix4<double>
Renderer::nodeLocalMatrix(const systems::leal::gltf::Node &node) {
    // If the stored matrix is not identity it was explicitly authored — use it.
    const auto &m = node.matrix;
    bool isIdentity =
        m.data[0]  == 1.0 && m.data[1]  == 0.0 && m.data[2]  == 0.0 && m.data[3]  == 0.0 &&
        m.data[4]  == 0.0 && m.data[5]  == 1.0 && m.data[6]  == 0.0 && m.data[7]  == 0.0 &&
        m.data[8]  == 0.0 && m.data[9]  == 0.0 && m.data[10] == 1.0 && m.data[11] == 0.0 &&
        m.data[12] == 0.0 && m.data[13] == 0.0 && m.data[14] == 0.0 && m.data[15] == 1.0;

    if (!isIdentity) return m;

    return systems::leal::vector_math::Matrix4<double>::compose(
        node.translation, node.rotation, node.scale);
}

bool Renderer::findCameraNode(
    uint64_t nodeIndex,
    const systems::leal::vector_math::Matrix4<double> &parentWorld,
    uint32_t camIndex,
    systems::leal::vector_math::Matrix4<double> &outWorld)
{
    if (!asset->nodes || nodeIndex >= asset->nodes->size()) return false;
    auto &node  = (*asset->nodes)[nodeIndex];
    auto  world = parentWorld * nodeLocalMatrix(node);

    if (node.camera == (int64_t)camIndex) {
        outWorld = world;
        return true;
    }
    for (auto childIndex : node.children) {
        if (findCameraNode(childIndex, world, camIndex, outWorld)) return true;
    }
    return false;
}

void Renderer::computeNodeTransform(
    uint64_t nodeIndex,
    const systems::leal::vector_math::Matrix4<double> &parentWorld)
{
    if (!asset->nodes || nodeIndex >= asset->nodes->size()) return;
    auto &node  = (*asset->nodes)[nodeIndex];
    auto  world = parentWorld * nodeLocalMatrix(node);

    if (nodeIndex * 16 + 15 < nodeTransforms.size()) {
        auto mvp = vpMatrix * world;
        for (int i = 0; i < 16; i++) {
            nodeTransforms[nodeIndex * 16 + i] = static_cast<float>(mvp.data[i]);
        }
    }
    for (auto childIndex : node.children) {
        computeNodeTransform(childIndex, world);
    }
}

void Renderer::computeSceneBounds(
    uint64_t nodeIndex,
    const systems::leal::vector_math::Matrix4<double> &parentWorld)
{
    if (!asset->nodes || nodeIndex >= asset->nodes->size()) return;
    auto &node  = (*asset->nodes)[nodeIndex];
    auto  world = parentWorld * nodeLocalMatrix(node);

    // Translation is in column 3 of the row-major Matrix4: data[row*4+3].
    float tx = (float)world.data[3];
    float ty = (float)world.data[7];
    float tz = (float)world.data[11];
    float dist = sqrtf(tx*tx + ty*ty + tz*tz) + 1.0f;
    if (dist > boundsRadius) boundsRadius = dist;

    for (auto childIndex : node.children) {
        computeSceneBounds(childIndex, world);
    }
}

// ---------------------------------------------------------------------------
// render()
// ---------------------------------------------------------------------------

void Renderer::render() {
    renderToTarget(device->getSwapchainTextureView());
}

void Renderer::render(std::shared_ptr<systems::leal::campello_gpu::TextureView> colorView) {
    renderToTarget(colorView);
}

void Renderer::renderToTarget(
    std::shared_ptr<systems::leal::campello_gpu::TextureView> colorView)
{
    if (!asset || (!pipelineFlat && !pipelineTextured)) return;
    if (!asset->scenes || sceneIndex >= asset->scenes->size()) return;
    if (!colorView) return;

    auto encoder = device->createCommandEncoder();
    if (!encoder) return;

    namespace GPU = systems::leal::campello_gpu;
    namespace VM  = systems::leal::vector_math;
    using M4 = VM::Matrix4<double>;

    // ------------------------------------------------------------------
    // 1. Compute view-projection from camera override or GLTF camera.
    // ------------------------------------------------------------------
    if (hasCameraOverride) {
        vpMatrix = overrideProj * overrideView;
    } else {
        double aspect = (renderHeight > 0)
            ? static_cast<double>(renderWidth) / renderHeight
            : 1.0;

        auto view = M4::lookAt(
            VM::Vector3<double>(0.0, 0.0, 5.0),
            VM::Vector3<double>(0.0, 0.0, 0.0),
            VM::Vector3<double>(0.0, 1.0, 0.0));

        static const double kDefaultFov = 60.0 * acos(-1.0) / 180.0;
        auto proj = M4::perspective(kDefaultFov, aspect, 0.1, 1000.0);

        if (asset->cameras && cameraIndex < asset->cameras->size()) {
            auto &scn = (*asset->scenes)[sceneIndex];
            if (scn.nodes) {
                M4 camWorld;
                bool found = false;
                for (auto rootIdx : *scn.nodes) {
                    if (findCameraNode(rootIdx, M4::identity(), cameraIndex, camWorld)) {
                        found = true; break;
                    }
                }
                if (found) {
                    view = camWorld.inverted();
                    auto *cam   = (*asset->cameras)[cameraIndex].get();
                    auto *persp = dynamic_cast<systems::leal::gltf::PerspectiveCamera *>(cam);
                    if (persp) {
                        double ar = persp->aspectRatio.value_or(aspect);
                        double zf = persp->zFar.value_or(1000.0);
                        proj = M4::perspective(persp->yFov, ar, persp->zNear, zf);
                    } else {
                        auto *ortho = dynamic_cast<systems::leal::gltf::OrthographicCamera *>(cam);
                        if (ortho)
                            proj = M4::ortho(ortho->xMag * 2.0, ortho->yMag * 2.0,
                                             ortho->zNear, ortho->zFar);
                    }
                }
            }
        }

        vpMatrix = proj * view;
    }

    // ------------------------------------------------------------------
    // 2. Upload material uniforms (baseColorFactor per material slot).
    // ------------------------------------------------------------------
    if (materialUniformBuffer) {
        float white[4] = {1.f, 1.f, 1.f, 1.f};
        materialUniformBuffer->upload(0, 16, white);
        if (asset->materials) {
            for (size_t i = 0; i < asset->materials->size(); ++i) {
                auto &mat = (*asset->materials)[i];
                float bc[4] = {1.f, 1.f, 1.f, 1.f};
                if (mat.pbrMetallicRoughness) {
                    bc[0] = (float)mat.pbrMetallicRoughness->baseColorFactor.x();
                    bc[1] = (float)mat.pbrMetallicRoughness->baseColorFactor.y();
                    bc[2] = (float)mat.pbrMetallicRoughness->baseColorFactor.z();
                    bc[3] = (float)mat.pbrMetallicRoughness->baseColorFactor.w();
                }
                materialUniformBuffer->upload((uint64_t)(i + 1) * 16, 16, bc);
            }
        }
    }

    // ------------------------------------------------------------------
    // 3. Compute MVP for every node and upload to the transform buffer.
    // ------------------------------------------------------------------
    auto &scene = (*asset->scenes)[sceneIndex];
    if (scene.nodes) {
        for (auto rootIdx : *scene.nodes) {
            computeNodeTransform(rootIdx, M4::identity());
        }
    }
    if (transformBuffer && !nodeTransforms.empty()) {
        transformBuffer->upload(
            0, nodeTransforms.size() * sizeof(float), nodeTransforms.data());
    }

    // ------------------------------------------------------------------
    // 4. Record render pass and draw calls.
    // ------------------------------------------------------------------
    GPU::ColorAttachment ca{};
    ca.view          = colorView;
    ca.clearValue[0] = 0.0f;
    ca.clearValue[1] = 0.0f;
    ca.clearValue[2] = 0.0f;
    ca.clearValue[3] = 1.0f;
    ca.loadOp        = GPU::LoadOp::clear;
    ca.storeOp       = GPU::StoreOp::store;
    ca.depthSlice    = 0;

    GPU::BeginRenderPassDescriptor rpDesc{};
    rpDesc.colorAttachments = { ca };

    if (depthView) {
        GPU::DepthStencilAttachment ds{};
        ds.view              = depthView;
        ds.depthClearValue   = 1.0f;
        ds.depthLoadOp       = GPU::LoadOp::clear;
        ds.depthStoreOp      = GPU::StoreOp::discard;
        ds.depthReadOnly     = false;
        ds.stencilClearValue = 0;
        ds.stencilLoadOp     = GPU::LoadOp::clear;
        ds.stencilStoreOp    = GPU::StoreOp::discard;
        ds.stencilRadOnly    = false;
        rpDesc.depthStencilAttachment = ds;
    }

    auto rpe = encoder->beginRenderPass(rpDesc);
    if (!rpe) return;

    currentPipelineVariant = 0; // reset — renderPrimitive() will set it per draw

    if (renderWidth > 0 && renderHeight > 0) {
        rpe->setViewport(0.0f, 0.0f, (float)renderWidth, (float)renderHeight, 0.0f, 1.0f);
        rpe->setScissorRect(0.0f, 0.0f, (float)renderWidth, (float)renderHeight);
    }

    if (scene.nodes) {
        for (auto nodeIndex : *scene.nodes) {
            renderNode(rpe, nodeIndex);
        }
    }

    rpe->end();
    device->submit(encoder->finish());
}

void Renderer::update(double dt) {
    // Reserved for animation updates.
    (void)dt;
}

// ---------------------------------------------------------------------------
// Scene-graph traversal
// ---------------------------------------------------------------------------

void Renderer::renderNode(
    const std::shared_ptr<systems::leal::campello_gpu::RenderPassEncoder> &rpe,
    uint64_t nodeIndex)
{
    if (!asset->nodes || nodeIndex >= asset->nodes->size()) return;
    auto &node = (*asset->nodes)[nodeIndex];

    if (node.mesh >= 0) {
        auto &mesh = (*asset->meshes)[(size_t)node.mesh];
        for (auto &primitive : mesh.primitives) {
            renderPrimitive(rpe, primitive, nodeIndex);
        }
    }
    for (auto childIndex : node.children) {
        renderNode(rpe, childIndex);
    }
}

void Renderer::renderPrimitive(
    const std::shared_ptr<systems::leal::campello_gpu::RenderPassEncoder> &rpe,
    const systems::leal::gltf::Primitive &primitive,
    uint64_t nodeIndex)
{
    // --- 1. Determine pipeline variant (flat vs textured) ---
    bool hasTexcoord = primitive.attributes.count("TEXCOORD_0") > 0;
    bool hasTexture  = false;
    int64_t matIdx   = primitive.material;

    if (hasTexcoord && matIdx >= 0 && asset->materials &&
        (size_t)matIdx < asset->materials->size())
    {
        auto &mat = (*asset->materials)[(size_t)matIdx];
        if (mat.pbrMetallicRoughness && mat.pbrMetallicRoughness->baseColorTexture)
            hasTexture = true;
    }

    int wantedVariant = hasTexture ? 2 : 1;
    auto &pipeline    = hasTexture ? pipelineTextured : pipelineFlat;

    if (pipeline && wantedVariant != currentPipelineVariant) {
        rpe->setPipeline(pipeline);
        currentPipelineVariant = wantedVariant;
    }

    // --- 2. Bind texture bind group at index 0 ---
    {
        std::shared_ptr<systems::leal::campello_gpu::BindGroup> bg = defaultBindGroup;

        if (hasTexture) {
            uint64_t texIdx = (*asset->materials)[(size_t)matIdx]
                .pbrMetallicRoughness->baseColorTexture->index;
            if (texIdx < gpuBindGroups.size() && gpuBindGroups[texIdx])
                bg = gpuBindGroups[texIdx];
        }

        if (bg) rpe->setBindGroup(0, bg);
    }

    // --- 3. Bind material uniforms at slot 17 ---
    if (materialUniformBuffer) {
        uint64_t matOffset = (matIdx >= 0) ? (uint64_t)(matIdx + 1) * 16 : 0;
        rpe->setVertexBuffer(VERTEX_SLOT_MATERIAL, materialUniformBuffer, matOffset);
    }

    // --- 4. Bind MVP transform for this node ---
    if (transformBuffer) {
        uint64_t mvpOffset = nodeIndex * 64; // 16 floats * 4 bytes
        if (mvpOffset + 64 <= transformBuffer->getLength())
            rpe->setVertexBuffer(VERTEX_SLOT_MVP, transformBuffer, mvpOffset);
    }

    // --- 5. Bind vertex attributes ---
    auto bindAttribute = [&](const std::string &semantic, uint32_t slot) {
        auto it = primitive.attributes.find(semantic);
        if (it == primitive.attributes.end()) return;
        auto &acc = (*asset->accessors)[it->second];
        if (acc.bufferView < 0) return;
        auto &bv  = (*asset->bufferViews)[(size_t)acc.bufferView];
        auto  buf = gpuBuffers[bv.buffer];
        if (buf) rpe->setVertexBuffer(slot, buf, bv.byteOffset + acc.byteOffset);
    };

    bindAttribute("POSITION", VERTEX_SLOT_POSITION);
    bindAttribute("NORMAL",   VERTEX_SLOT_NORMAL);
    bindAttribute("TANGENT",  VERTEX_SLOT_TANGENT);

    // TEXCOORD_0: bind real data or fallback zero buffer
    if (primitive.attributes.count("TEXCOORD_0")) {
        bindAttribute("TEXCOORD_0", VERTEX_SLOT_TEXCOORD0);
    } else if (fallbackUVBuffer) {
        rpe->setVertexBuffer(VERTEX_SLOT_TEXCOORD0, fallbackUVBuffer, 0);
    }

    // --- 6. POSITION must be present to know the vertex count ---
    auto posIt = primitive.attributes.find("POSITION");
    if (posIt == primitive.attributes.end()) return;
    auto &posAcc = (*asset->accessors)[posIt->second];
    if (posAcc.bufferView < 0) return;

    // --- 7. Draw indexed or non-indexed ---
    if (primitive.indices >= 0) {
        auto &idxAcc = (*asset->accessors)[(size_t)primitive.indices];
        if (idxAcc.bufferView >= 0) {
            auto &idxBV  = (*asset->bufferViews)[(size_t)idxAcc.bufferView];
            auto  idxBuf = gpuBuffers[idxBV.buffer];
            if (idxBuf) {
                using CT = systems::leal::gltf::ComponentType;
                auto fmt = (idxAcc.componentType == CT::ctUnsignedShort)
                    ? systems::leal::campello_gpu::IndexFormat::uint16
                    : systems::leal::campello_gpu::IndexFormat::uint32;
                rpe->setIndexBuffer(idxBuf, fmt, idxBV.byteOffset + idxAcc.byteOffset);
                rpe->drawIndexed((uint32_t)idxAcc.count);
                return;
            }
        }
    }

    rpe->draw((uint32_t)posAcc.count);
}

// ---------------------------------------------------------------------------
// Camera override
// ---------------------------------------------------------------------------

void Renderer::setCameraMatrices(const float *viewColMajor16, const float *projColMajor16) {
    namespace VM = systems::leal::vector_math;
    // Convert column-major float[16] (simd_float4x4 layout) to row-major Matrix4<double>.
    auto colToRow = [](const float *src, VM::Matrix4<double> &dst) {
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++)
                dst.data[r * 4 + c] = static_cast<double>(src[c * 4 + r]);
    };
    colToRow(viewColMajor16, overrideView);
    colToRow(projColMajor16, overrideProj);
    hasCameraOverride = true;
}

void Renderer::clearCameraOverride() {
    hasCameraOverride = false;
}

float Renderer::getBoundsRadius() const {
    return boundsRadius;
}

// ---------------------------------------------------------------------------
// GPU resource accessors
// ---------------------------------------------------------------------------

std::shared_ptr<systems::leal::campello_gpu::Buffer>
Renderer::getGpuBuffer(uint32_t index) const {
    if (index < gpuBuffers.size()) return gpuBuffers[index];
    return nullptr;
}

std::shared_ptr<systems::leal::campello_gpu::Texture>
Renderer::getGpuTexture(uint32_t index) const {
    if (index < gpuTextures.size()) return gpuTextures[index];
    return nullptr;
}

uint32_t Renderer::getGpuBufferCount()  const { return (uint32_t)gpuBuffers.size(); }
uint32_t Renderer::getGpuTextureCount() const { return (uint32_t)gpuTextures.size(); }

std::shared_ptr<systems::leal::campello_gpu::BindGroup>
Renderer::getBindGroup(uint32_t index) const {
    if (index < gpuBindGroups.size()) return gpuBindGroups[index];
    return nullptr;
}

uint32_t Renderer::getBindGroupCount() const { return (uint32_t)gpuBindGroups.size(); }

std::shared_ptr<systems::leal::campello_gpu::BindGroup>
Renderer::getDefaultBindGroup() const { return defaultBindGroup; }

std::string systems::leal::campello_renderer::getVersion() {
    return std::to_string(campello_renderer_VERSION_MAJOR) + "." +
           std::to_string(campello_renderer_VERSION_MINOR) + "." +
           std::to_string(campello_renderer_VERSION_PATCH);
}
