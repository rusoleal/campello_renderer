#include <campello_renderer/campello_renderer.hpp>
#include "campello_renderer_config.h"
#include <iostream>
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
#include <algorithm>
#include <cmath>
#include <unordered_set>

#if defined(__APPLE__)
#include "shaders/metal_default.h"
#elif defined(ANDROID)
#include "shaders/vulkan_default.h"
#elif defined(_WIN32)
#include "shaders/directx_default.h"
#endif

#include <campello_image/image.hpp>

using namespace systems::leal::campello_renderer;

// Byte size of one material slot in materialUniformBuffer.
// Layout: float4 baseColorFactor (16 B) + float4 uvTransformRow0 (16 B) + float4 uvTransformRow1 (16 B).
// Must be >= 256 for Metal vertex buffer offset alignment (setVertexBuffer offset must be 256-byte aligned).
static constexpr uint64_t kMaterialUniformStride = 256;

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
    activeVariant = -1;
    if (asset == nullptr) {
        images.clear();
        nodeTransforms.clear();
        transformBuffer       = nullptr;
        materialUniformBuffer = nullptr;
        boundsRadius          = 1.0f;
        gpuSamplers.clear();
        materialBindGroups.clear();
        flatMaterialBindGroups.clear();
        defaultFlatBindGroup = nullptr;
        quantizedPipelines.clear();
        dracoPrimitiveBuffers.clear();
        deinterleavedBuffers.clear();
        nodeInstanceData.clear();
        primitiveBounds.clear();
        nodeMeshLocalBounds.clear();
        nodeLocalBounds.clear();
        nodeWorldBounds.clear();
        nodeWorldMatrices.clear();
        visibleNodeMask.clear();
        opaqueQueue.clear();
        transparentQueue.clear();
        hasFrustumPlanes = false;
        animationStates.clear();
        animatedNodes.clear();
        return;
    }

    // New asset loaded — reset animation state.
    animationStates.clear();
    animatedNodes.clear();

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

    // Force rebuild of defaultBindGroup each setScene() so it references the
    // fresh materialUniformBuffer and cameraPositionBuffer for this asset.
    defaultBindGroup = nullptr;

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

    // Note: decompression is already done in GLTF::loadGLB() for GLB files.
    // For .gltf files with external buffers, decompressDraco/meshopt should be
    // called after loading the external buffer data (not implemented here).
    // Upload any Draco-decompressed buffers to GPU.
    uploadDracoBuffers(info);

    // Deinterleave vertex attributes from buffer views with byteStride > 0.
    // The pipeline vertex layouts assume tightly packed data (stride = attribute
    // size). Interleaved data shares a buffer view with a common stride, so we
    // extract each accessor into its own contiguous GPU buffer.
    deinterleavedBuffers.clear();
    if (asset->meshes && asset->accessors && asset->bufferViews) {
        auto getComponentSize = [](systems::leal::gltf::ComponentType ct) -> size_t {
            using CT = systems::leal::gltf::ComponentType;
            switch (ct) {
                case CT::ctByte: return 1;
                case CT::ctUnsignedByte: return 1;
                case CT::ctShort: return 2;
                case CT::ctUnsignedShort: return 2;
                case CT::ctUnsignedInt: return 4;
                case CT::ctFloat: return 4;
            }
            return 1;
        };
        auto getTypeCount = [](systems::leal::gltf::AccessorType at) -> size_t {
            using AT = systems::leal::gltf::AccessorType;
            switch (at) {
                case AT::acScalar: return 1;
                case AT::acVec2:   return 2;
                case AT::acVec3:   return 3;
                case AT::acVec4:   return 4;
                case AT::acMat2:   return 4;
                case AT::acMat3:   return 9;
                case AT::acMat4:   return 16;
            }
            return 1;
        };

        for (auto &mesh : *asset->meshes) {
            for (auto &primitive : mesh.primitives) {
                for (auto &[semantic, accIdx] : primitive.attributes) {
                    if (accIdx < 0 || (size_t)accIdx >= asset->accessors->size()) continue;
                    auto &acc = (*asset->accessors)[(size_t)accIdx];
                    if (acc.bufferView < 0 || (size_t)acc.bufferView >= asset->bufferViews->size()) continue;
                    auto &bv = (*asset->bufferViews)[(size_t)acc.bufferView];
                    if (bv.byteStride <= 0) continue;
                    if (deinterleavedBuffers.count(accIdx)) continue;

                    size_t compSize = getComponentSize(acc.componentType);
                    size_t typeCount = getTypeCount(acc.type);
                    size_t elementSize = compSize * typeCount;
                    size_t totalSize = elementSize * acc.count;
                    if (totalSize == 0) continue;

                    auto &buf = (*asset->buffers)[bv.buffer];
                    if (buf.data.empty()) continue;

                    size_t paddedElementSize = (elementSize + 3) & ~size_t(3); // round up to 4
                    std::vector<uint8_t> deinterleaved(paddedElementSize * acc.count, 0);
                    const uint8_t *src = buf.data.data() + bv.byteOffset + acc.byteOffset;
                    for (size_t i = 0; i < acc.count; ++i) {
                        std::memcpy(deinterleaved.data() + i * paddedElementSize,
                                    src + i * bv.byteStride,
                                    elementSize);
                    }

                    using BU = systems::leal::campello_gpu::BufferUsage;
                    auto gpuBuf = device->createBuffer(
                        totalSize, BU::vertex,
                        deinterleaved.data());
                    if (gpuBuf) {
                        deinterleavedBuffers[accIdx] = gpuBuf;
                    }
                }
            }
        }
    }

    // Scan ALL materials in the asset to determine which image indices carry
    // colour data (baseColor, emissive) and therefore need sRGB sampling.
    // Linear data textures (metallicRoughness, normal, occlusion) must stay
    // rgba8unorm so the GPU does not gamma-decode them.
    // This scan covers all materials (not just the active scene) because
    // gpuTextures persists across setScene() calls.
    std::unordered_set<int64_t> srgbImageIndices;
    if (asset->textures && asset->materials) {
        auto imageIndexForTex = [&](int64_t texIdx) -> int64_t {
            if (texIdx < 0 || (size_t)texIdx >= asset->textures->size()) return -1;
            auto &gt = (*asset->textures)[(size_t)texIdx];
            return (gt.ext_texture_webp >= 0) ? gt.ext_texture_webp : gt.source;
        };
        for (auto &mat : *asset->materials) {
            if (mat.pbrMetallicRoughness && mat.pbrMetallicRoughness->baseColorTexture)
                srgbImageIndices.insert(imageIndexForTex(mat.pbrMetallicRoughness->baseColorTexture->index));
            if (mat.emissiveTexture)
                srgbImageIndices.insert(imageIndexForTex(mat.emissiveTexture->index));
            // KHR_materials_specular: specularColorTexture is sRGB-encoded
            if (mat.khrMaterialsSpecular && mat.khrMaterialsSpecular->specularColorTexture)
                srgbImageIndices.insert(imageIndexForTex(mat.khrMaterialsSpecular->specularColorTexture->index));
            // KHR_materials_sheen: sheenColorTexture is sRGB-encoded; roughness texture is linear
            if (mat.khrMaterialsSheen && mat.khrMaterialsSheen->sheenColorTexture)
                srgbImageIndices.insert(imageIndexForTex(mat.khrMaterialsSheen->sheenColorTexture->index));
        }
        srgbImageIndices.erase(-1); // remove sentinel from any unresolved lookups
    }

    namespace GPU = systems::leal::campello_gpu;

    // Helper: map campello_image format to campello_gpu pixel format.
    auto imageFormatToPixelFormat = [](systems::leal::campello_image::ImageFormat imgFmt,
                                        bool srgb) -> GPU::PixelFormat {
        switch (imgFmt) {
            case systems::leal::campello_image::ImageFormat::rgba8:
                return srgb ? GPU::PixelFormat::rgba8unorm_srgb : GPU::PixelFormat::rgba8unorm;
            case systems::leal::campello_image::ImageFormat::rgba16f:
                return GPU::PixelFormat::rgba16float;
            case systems::leal::campello_image::ImageFormat::rgba32f:
                return GPU::PixelFormat::rgba32float;
        }
        return GPU::PixelFormat::rgba8unorm;
    };

    // Decode and upload images referenced by this scene.
    for (int a = 0; a < (int)info->images.size(); a++) {
        if (info->images[a]) {
            if (gpuTextures[a] == nullptr) {
                auto &image = (*asset->images)[a];
                bool wantsSrgb = srgbImageIndices.count(a) > 0;

                if (image.data.size() > 0) {
                    // Data:uri images are already decoded by the gltf library.
                    auto img = systems::leal::campello_image::Image::fromMemory(
                        image.data.data(), image.data.size());
                    if (img != nullptr) {
                        auto fmt = imageFormatToPixelFormat(img->getFormat(), wantsSrgb);
                        auto texture = device->createTexture(
                            GPU::TextureType::tt2d, fmt,
                            img->getWidth(), img->getHeight(), 1, 1, 1,
                            GPU::TextureUsage::textureBinding);
                        if (texture != nullptr) {
                            texture->upload(0, img->getDataSize(), const_cast<void*>(img->getData()));
                            gpuTextures[a] = texture;
                        }
                    }
                } else if (image.bufferView != -1) {
                    auto &bufferView = (*asset->bufferViews)[image.bufferView];
                    auto &buffer     = (*asset->buffers)[bufferView.buffer];
                    if (!buffer.data.empty()) {
                        const uint8_t *src = buffer.data.data() + bufferView.byteOffset;
                        auto img = systems::leal::campello_image::Image::fromMemory(src, bufferView.byteLength);
                        if (img != nullptr) {
                            auto fmt = imageFormatToPixelFormat(img->getFormat(), wantsSrgb);
                            auto texture = device->createTexture(
                                GPU::TextureType::tt2d, fmt,
                                img->getWidth(), img->getHeight(), 1, 1, 1,
                                GPU::TextureUsage::textureBinding);
                            if (texture != nullptr) {
                                texture->upload(0, img->getDataSize(), const_cast<void*>(img->getData()));
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

    // Allocate transform buffer — two float4x4 per node: MVP (clip) + Model (world).
    // Layout per node: 16 floats MVP, 16 floats Model = 32 floats = 128 bytes.
    if (asset->nodes && !asset->nodes->empty()) {
        size_t nodeCount = asset->nodes->size();
        nodeTransforms.assign(nodeCount * 32, 0.0f); // 32 floats per node
        nodeMeshLocalBounds.assign(nodeCount, Bounds{});
        nodeLocalBounds.assign(nodeCount, Bounds{});
        nodeWorldBounds.assign(nodeCount, Bounds{});
        nodeWorldMatrices.assign(nodeCount, systems::leal::vector_math::Matrix4<double>::identity());
        visibleNodeMask.assign(nodeCount, 0);
        using BU = systems::leal::campello_gpu::BufferUsage;
        for (uint32_t f = 0; f < kMaxFramesInFlight; ++f) {
            frameResources[f].transformBuffer = device->createBuffer(nodeCount * 128, BU::vertex);
        }
        transformBuffer = frameResources[0].transformBuffer;
    }

    // ------------------------------------------------------------------
    // Lazy-initialize shared texture resources (once per device lifetime).
    // ------------------------------------------------------------------
    if (!bindGroupLayout) {
        GPU::BindGroupLayoutDescriptor bglDesc{};

        // Binding 0: baseColorTexture
        GPU::EntryObject texEntry0{};
        texEntry0.binding    = 0;
        texEntry0.visibility = GPU::ShaderStage::fragment;
        texEntry0.type       = GPU::EntryObjectType::texture;
        texEntry0.data.texture.multisampled = false;
        texEntry0.data.texture.sampleType   = GPU::EntryObjectTextureType::ttFloat;
        texEntry0.data.texture.viewDimension = GPU::TextureType::tt2d;
        bglDesc.entries.push_back(texEntry0);

        // Binding 1: baseColorSampler
        GPU::EntryObject sampEntry0{};
        sampEntry0.binding    = 1;
        sampEntry0.visibility = GPU::ShaderStage::fragment;
        sampEntry0.type       = GPU::EntryObjectType::sampler;
        sampEntry0.data.sampler.type = GPU::EntryObjectSamplerType::filtering;
        bglDesc.entries.push_back(sampEntry0);

        // Binding 2: metallicRoughnessTexture
        GPU::EntryObject texEntry1{};
        texEntry1.binding    = 2;
        texEntry1.visibility = GPU::ShaderStage::fragment;
        texEntry1.type       = GPU::EntryObjectType::texture;
        texEntry1.data.texture.multisampled = false;
        texEntry1.data.texture.sampleType   = GPU::EntryObjectTextureType::ttFloat;
        texEntry1.data.texture.viewDimension = GPU::TextureType::tt2d;
        bglDesc.entries.push_back(texEntry1);

        // Binding 3: metallicRoughnessSampler
        GPU::EntryObject sampEntry1{};
        sampEntry1.binding    = 3;
        sampEntry1.visibility = GPU::ShaderStage::fragment;
        sampEntry1.type       = GPU::EntryObjectType::sampler;
        sampEntry1.data.sampler.type = GPU::EntryObjectSamplerType::filtering;
        bglDesc.entries.push_back(sampEntry1);

        // Binding 4: normalTexture
        GPU::EntryObject texEntry2{};
        texEntry2.binding    = 4;
        texEntry2.visibility = GPU::ShaderStage::fragment;
        texEntry2.type       = GPU::EntryObjectType::texture;
        texEntry2.data.texture.multisampled = false;
        texEntry2.data.texture.sampleType   = GPU::EntryObjectTextureType::ttFloat;
        texEntry2.data.texture.viewDimension = GPU::TextureType::tt2d;
        bglDesc.entries.push_back(texEntry2);

        // Binding 5: normalSampler
        GPU::EntryObject sampEntry2{};
        sampEntry2.binding    = 5;
        sampEntry2.visibility = GPU::ShaderStage::fragment;
        sampEntry2.type       = GPU::EntryObjectType::sampler;
        sampEntry2.data.sampler.type = GPU::EntryObjectSamplerType::filtering;
        bglDesc.entries.push_back(sampEntry2);

        // Binding 6: emissiveTexture
        GPU::EntryObject texEntry3{};
        texEntry3.binding    = 6;
        texEntry3.visibility = GPU::ShaderStage::fragment;
        texEntry3.type       = GPU::EntryObjectType::texture;
        texEntry3.data.texture.multisampled = false;
        texEntry3.data.texture.sampleType   = GPU::EntryObjectTextureType::ttFloat;
        texEntry3.data.texture.viewDimension = GPU::TextureType::tt2d;
        bglDesc.entries.push_back(texEntry3);

        // Binding 7: emissiveSampler
        GPU::EntryObject sampEntry3{};
        sampEntry3.binding    = 7;
        sampEntry3.visibility = GPU::ShaderStage::fragment;
        sampEntry3.type       = GPU::EntryObjectType::sampler;
        sampEntry3.data.sampler.type = GPU::EntryObjectSamplerType::filtering;
        bglDesc.entries.push_back(sampEntry3);

        // Binding 8: occlusionTexture
        GPU::EntryObject texEntry4{};
        texEntry4.binding    = 8;
        texEntry4.visibility = GPU::ShaderStage::fragment;
        texEntry4.type       = GPU::EntryObjectType::texture;
        texEntry4.data.texture.multisampled = false;
        texEntry4.data.texture.sampleType   = GPU::EntryObjectTextureType::ttFloat;
        texEntry4.data.texture.viewDimension = GPU::TextureType::tt2d;
        bglDesc.entries.push_back(texEntry4);

        // Binding 9: occlusionSampler
        GPU::EntryObject sampEntry4{};
        sampEntry4.binding    = 9;
        sampEntry4.visibility = GPU::ShaderStage::fragment;
        sampEntry4.type       = GPU::EntryObjectType::sampler;
        sampEntry4.data.sampler.type = GPU::EntryObjectSamplerType::filtering;
        bglDesc.entries.push_back(sampEntry4);

        // Binding 10: lightsUniformBuffer (KHR_lights_punctual)
        GPU::EntryObject lightsEntry{};
        lightsEntry.binding    = 10;
        lightsEntry.visibility = GPU::ShaderStage::fragment;
        lightsEntry.type       = GPU::EntryObjectType::buffer;
        lightsEntry.data.buffer.hasDinamicOffaset = false;
        lightsEntry.data.buffer.minBindingSize    = 272; // 16-byte header + 4 lights * 64 bytes
        lightsEntry.data.buffer.type              = GPU::EntryObjectBufferType::uniform;
        bglDesc.entries.push_back(lightsEntry);

        // Binding 11: specularTexture (KHR_materials_specular — A channel = specular factor)
        GPU::EntryObject texEntry5{};
        texEntry5.binding    = 11;
        texEntry5.visibility = GPU::ShaderStage::fragment;
        texEntry5.type       = GPU::EntryObjectType::texture;
        texEntry5.data.texture.multisampled  = false;
        texEntry5.data.texture.sampleType    = GPU::EntryObjectTextureType::ttFloat;
        texEntry5.data.texture.viewDimension = GPU::TextureType::tt2d;
        bglDesc.entries.push_back(texEntry5);

        // Binding 12: specularSampler
        GPU::EntryObject sampEntry5{};
        sampEntry5.binding    = 12;
        sampEntry5.visibility = GPU::ShaderStage::fragment;
        sampEntry5.type       = GPU::EntryObjectType::sampler;
        sampEntry5.data.sampler.type = GPU::EntryObjectSamplerType::filtering;
        bglDesc.entries.push_back(sampEntry5);

        // Binding 13: specularColorTexture (KHR_materials_specular — RGB = F0 color tint, sRGB)
        GPU::EntryObject texEntry6{};
        texEntry6.binding    = 13;
        texEntry6.visibility = GPU::ShaderStage::fragment;
        texEntry6.type       = GPU::EntryObjectType::texture;
        texEntry6.data.texture.multisampled  = false;
        texEntry6.data.texture.sampleType    = GPU::EntryObjectTextureType::ttFloat;
        texEntry6.data.texture.viewDimension = GPU::TextureType::tt2d;
        bglDesc.entries.push_back(texEntry6);

        // Binding 14: specularColorSampler
        GPU::EntryObject sampEntry6{};
        sampEntry6.binding    = 14;
        sampEntry6.visibility = GPU::ShaderStage::fragment;
        sampEntry6.type       = GPU::EntryObjectType::sampler;
        sampEntry6.data.sampler.type = GPU::EntryObjectSamplerType::filtering;
        bglDesc.entries.push_back(sampEntry6);

        // Binding 15: sheenColorTexture (KHR_materials_sheen — RGB sRGB = sheen color)
        // Note: sampler reused from baseColorSampler (binding 1) in the shader — Metal only
        // allows 16 sampler slots (0–15) and all are already claimed.
        GPU::EntryObject texEntry7{};
        texEntry7.binding    = 15;
        texEntry7.visibility = GPU::ShaderStage::fragment;
        texEntry7.type       = GPU::EntryObjectType::texture;
        texEntry7.data.texture.multisampled  = false;
        texEntry7.data.texture.sampleType    = GPU::EntryObjectTextureType::ttFloat;
        texEntry7.data.texture.viewDimension = GPU::TextureType::tt2d;
        bglDesc.entries.push_back(texEntry7);

        // Binding 16: sheenRoughnessTexture (KHR_materials_sheen — R = roughness factor)
        GPU::EntryObject texEntry8{};
        texEntry8.binding    = 16;
        texEntry8.visibility = GPU::ShaderStage::fragment;
        texEntry8.type       = GPU::EntryObjectType::texture;
        texEntry8.data.texture.multisampled  = false;
        texEntry8.data.texture.sampleType    = GPU::EntryObjectTextureType::ttFloat;
        texEntry8.data.texture.viewDimension = GPU::TextureType::tt2d;
        bglDesc.entries.push_back(texEntry8);

        // Binding 17: clearcoatTexture (KHR_materials_clearcoat — R = intensity)
        GPU::EntryObject texEntry9{};
        texEntry9.binding    = 17;
        texEntry9.visibility = GPU::ShaderStage::fragment;
        texEntry9.type       = GPU::EntryObjectType::texture;
        texEntry9.data.texture.multisampled  = false;
        texEntry9.data.texture.sampleType    = GPU::EntryObjectTextureType::ttFloat;
        texEntry9.data.texture.viewDimension = GPU::TextureType::tt2d;
        bglDesc.entries.push_back(texEntry9);

        // Binding 18: clearcoatRoughnessTexture (KHR_materials_clearcoat — G = roughness)
        GPU::EntryObject texEntry10{};
        texEntry10.binding    = 18;
        texEntry10.visibility = GPU::ShaderStage::fragment;
        texEntry10.type       = GPU::EntryObjectType::texture;
        texEntry10.data.texture.multisampled  = false;
        texEntry10.data.texture.sampleType    = GPU::EntryObjectTextureType::ttFloat;
        texEntry10.data.texture.viewDimension = GPU::TextureType::tt2d;
        bglDesc.entries.push_back(texEntry10);

        // Binding 19: clearcoatNormalTexture (KHR_materials_clearcoat — tangent-space normal)
        GPU::EntryObject texEntry11{};
        texEntry11.binding    = 19;
        texEntry11.visibility = GPU::ShaderStage::fragment;
        texEntry11.type       = GPU::EntryObjectType::texture;
        texEntry11.data.texture.multisampled  = false;
        texEntry11.data.texture.sampleType    = GPU::EntryObjectTextureType::ttFloat;
        texEntry11.data.texture.viewDimension = GPU::TextureType::tt2d;
        bglDesc.entries.push_back(texEntry11);

        // Binding 20: transmissionTexture (KHR_materials_transmission — R=transmission factor)
        // Reuses baseColorSampler (Metal has 16 sampler limit)
        GPU::EntryObject texEntry12{};
        texEntry12.binding    = 20;
        texEntry12.visibility = GPU::ShaderStage::fragment;
        texEntry12.type       = GPU::EntryObjectType::texture;
        texEntry12.data.texture.multisampled  = false;
        texEntry12.data.texture.sampleType    = GPU::EntryObjectTextureType::ttFloat;
        texEntry12.data.texture.viewDimension = GPU::TextureType::tt2d;
        bglDesc.entries.push_back(texEntry12);

        // Binding 21: environmentMap (cube texture for IBL / skybox)
        GPU::EntryObject texEntryEnv{};
        texEntryEnv.binding    = 21;
        texEntryEnv.visibility = GPU::ShaderStage::fragment;
        texEntryEnv.type       = GPU::EntryObjectType::texture;
        texEntryEnv.data.texture.multisampled  = false;
        texEntryEnv.data.texture.sampleType    = GPU::EntryObjectTextureType::ttFloat;
        texEntryEnv.data.texture.viewDimension = GPU::TextureType::ttCube;
        bglDesc.entries.push_back(texEntryEnv);

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
            GPU::TextureType::tt2d, GPU::PixelFormat::rgba8unorm_srgb,
            1, 1, 1, 1, 1, GPU::TextureUsage::textureBinding);
        if (defaultTexture) defaultTexture->upload(0, 4, white);
    }

    // Default metallic-roughness: G=roughness=1.0, B=metallic=1.0 (factors default to 1.0)
    if (!defaultMetallicRoughnessTexture) {
        uint8_t metalRough[4] = {0, 255, 255, 255}; // R=0, G=1.0 (roughness), B=1.0 (metallic), A=1.0
        defaultMetallicRoughnessTexture = device->createTexture(
            GPU::TextureType::tt2d, GPU::PixelFormat::rgba8unorm,
            1, 1, 1, 1, 1, GPU::TextureUsage::textureBinding);
        if (defaultMetallicRoughnessTexture) defaultMetallicRoughnessTexture->upload(0, 4, metalRough);
    }

    // Default normal: (0.5, 0.5, 1.0) represents flat normal (0,0,1) in tangent space
    if (!defaultNormalTexture) {
        uint8_t normal[4] = {128, 128, 255, 255}; // RGB=(0.5,0.5,1.0), A=1.0
        defaultNormalTexture = device->createTexture(
            GPU::TextureType::tt2d, GPU::PixelFormat::rgba8unorm,
            1, 1, 1, 1, 1, GPU::TextureUsage::textureBinding);
        if (defaultNormalTexture) defaultNormalTexture->upload(0, 4, normal);
    }

    // Default emissive: black (no emission)
    if (!defaultEmissiveTexture) {
        uint8_t black[4] = {0, 0, 0, 255}; // RGB=(0,0,0), A=1.0
        defaultEmissiveTexture = device->createTexture(
            GPU::TextureType::tt2d, GPU::PixelFormat::rgba8unorm_srgb,
            1, 1, 1, 1, 1, GPU::TextureUsage::textureBinding);
        if (defaultEmissiveTexture) defaultEmissiveTexture->upload(0, 4, black);
    }

    // Default occlusion: white (no occlusion - multiply by 1.0)
    if (!defaultOcclusionTexture) {
        uint8_t white[4] = {255, 255, 255, 255}; // RGB=(1,1,1), A=1.0
        defaultOcclusionTexture = device->createTexture(
            GPU::TextureType::tt2d, GPU::PixelFormat::rgba8unorm,
            1, 1, 1, 1, 1, GPU::TextureUsage::textureBinding);
        if (defaultOcclusionTexture) defaultOcclusionTexture->upload(0, 4, white);
    }

    // Default specular texture: white (1,1,1,1) — A=1.0 passes specularFactor through unchanged
    if (!defaultSpecularTexture) {
        uint8_t white[4] = {255, 255, 255, 255};
        defaultSpecularTexture = device->createTexture(
            GPU::TextureType::tt2d, GPU::PixelFormat::rgba8unorm,
            1, 1, 1, 1, 1, GPU::TextureUsage::textureBinding);
        if (defaultSpecularTexture) defaultSpecularTexture->upload(0, 4, white);
    }

    // Default specular color texture: white sRGB (1,1,1,1) — no F0 color tint
    if (!defaultSpecularColorTexture) {
        uint8_t white[4] = {255, 255, 255, 255};
        defaultSpecularColorTexture = device->createTexture(
            GPU::TextureType::tt2d, GPU::PixelFormat::rgba8unorm_srgb,
            1, 1, 1, 1, 1, GPU::TextureUsage::textureBinding);
        if (defaultSpecularColorTexture) defaultSpecularColorTexture->upload(0, 4, white);
    }

    // Default sheen color texture: black sRGB (0,0,0,1) — sheenColor=[0,0,0] means no sheen by default
    if (!defaultSheenColorTexture) {
        uint8_t black[4] = {0, 0, 0, 255};
        defaultSheenColorTexture = device->createTexture(
            GPU::TextureType::tt2d, GPU::PixelFormat::rgba8unorm_srgb,
            1, 1, 1, 1, 1, GPU::TextureUsage::textureBinding);
        if (defaultSheenColorTexture) defaultSheenColorTexture->upload(0, 4, black);
    }

    // Default sheen roughness texture: white linear (1,1,1,1) — R=1.0 passes sheenRoughnessFactor through
    if (!defaultSheenRoughnessTexture) {
        uint8_t white[4] = {255, 255, 255, 255};
        defaultSheenRoughnessTexture = device->createTexture(
            GPU::TextureType::tt2d, GPU::PixelFormat::rgba8unorm,
            1, 1, 1, 1, 1, GPU::TextureUsage::textureBinding);
        if (defaultSheenRoughnessTexture) defaultSheenRoughnessTexture->upload(0, 4, white);
    }

    // Default clearcoat intensity texture: white linear — R=1.0 passes clearcoatFactor through (default factor=0)
    if (!defaultClearcoatTexture) {
        uint8_t white[4] = {255, 255, 255, 255};
        defaultClearcoatTexture = device->createTexture(
            GPU::TextureType::tt2d, GPU::PixelFormat::rgba8unorm,
            1, 1, 1, 1, 1, GPU::TextureUsage::textureBinding);
        if (defaultClearcoatTexture) defaultClearcoatTexture->upload(0, 4, white);
    }

    // Default clearcoat roughness texture: white linear — G=1.0 passes clearcoatRoughnessFactor through
    if (!defaultClearcoatRoughnessTexture) {
        uint8_t white[4] = {255, 255, 255, 255};
        defaultClearcoatRoughnessTexture = device->createTexture(
            GPU::TextureType::tt2d, GPU::PixelFormat::rgba8unorm,
            1, 1, 1, 1, 1, GPU::TextureUsage::textureBinding);
        if (defaultClearcoatRoughnessTexture) defaultClearcoatRoughnessTexture->upload(0, 4, white);
    }

    // Default clearcoat normal texture: flat normal (128,128,255,255) — identity tangent-space normal
    if (!defaultClearcoatNormalTexture) {
        uint8_t flatNormal[4] = {128, 128, 255, 255};
        defaultClearcoatNormalTexture = device->createTexture(
            GPU::TextureType::tt2d, GPU::PixelFormat::rgba8unorm,
            1, 1, 1, 1, 1, GPU::TextureUsage::textureBinding);
        if (defaultClearcoatNormalTexture) defaultClearcoatNormalTexture->upload(0, 4, flatNormal);
    }

    // Default environment map: 1x1x6 dark gray cube — used when no environment is set.
    if (!environmentMap) {
        uint8_t darkGray[4] = {26, 26, 26, 255}; // ~0.1 linear
        auto defaultEnvTex = device->createTexture(
            GPU::TextureType::ttCube, GPU::PixelFormat::rgba8unorm,
            1, 1, 1, 1, 1, GPU::TextureUsage::textureBinding);
        if (defaultEnvTex) {
            uint8_t faces[24];
            for (int f = 0; f < 6; ++f) memcpy(faces + f * 4, darkGray, 4);
            defaultEnvTex->upload(0, 24, faces);
            environmentMap = defaultEnvTex;
        }
    }
    if (!environmentSampler) {
        GPU::SamplerDescriptor esd{};
        esd.addressModeU = GPU::WrapMode::clampToEdge;
        esd.addressModeV = GPU::WrapMode::clampToEdge;
        esd.addressModeW = GPU::WrapMode::clampToEdge;
        esd.magFilter    = GPU::FilterMode::fmLinear;
        esd.minFilter    = GPU::FilterMode::fmLinear;
        environmentSampler = device->createSampler(esd);
    }

    // Default instance matrix: identity matrix for non-instanced rendering.
    // Column-major float4x4 (64 bytes) — bound to slot 19 when EXT_mesh_gpu_instancing is not used.
    if (!defaultInstanceMatrixBuffer) {
        float identity[16] = {
            1, 0, 0, 0,  // column 0
            0, 1, 0, 0,  // column 1
            0, 0, 1, 0,  // column 2
            0, 0, 0, 1   // column 3
        };
        defaultInstanceMatrixBuffer = device->createBuffer(
            64, GPU::BufferUsage::vertex, reinterpret_cast<uint8_t*>(identity));
    }

    // Frame-in-flight buffers: lights, camera, and fences.
    for (uint32_t f = 0; f < kMaxFramesInFlight; ++f) {
        if (!frameResources[f].lightsUniformBuffer) {
            frameResources[f].lightsUniformBuffer = device->createBuffer(272, GPU::BufferUsage::uniform);
            uint8_t zeros[272] = {0};
            frameResources[f].lightsUniformBuffer->upload(0, 272, zeros);
        }
        if (!frameResources[f].cameraPositionBuffer) {
            frameResources[f].cameraPositionBuffer = device->createBuffer(16, GPU::BufferUsage::vertex);
            float defaultCam[4] = {0.f, 0.f, 3.f, 0.f};
            frameResources[f].cameraPositionBuffer->upload(0, 16, defaultCam);
        }
        if (!frameResources[f].fence) {
            frameResources[f].fence = device->createFence();
        }
    }
    lightsUniformBuffer = frameResources[0].lightsUniformBuffer;
    cameraPositionBuffer = frameResources[0].cameraPositionBuffer;

    // Material uniform buffer — recreated each setScene() (size depends on material count).
    // Slot 0 = default; slots 1..N = per-material.  Static after upload, so single buffer.
    {
        size_t matCount = asset->materials ? asset->materials->size() : 0;
        uint64_t bufSize = (uint64_t)(matCount + 1) * kMaterialUniformStride;
        materialUniformBuffer = device->createBuffer(bufSize, GPU::BufferUsage::vertex);
    }

    if (!defaultBindGroup && bindGroupLayout && defaultTexture && defaultSampler &&
        defaultMetallicRoughnessTexture && defaultNormalTexture &&
        defaultEmissiveTexture && defaultOcclusionTexture &&
        defaultSpecularTexture && defaultSpecularColorTexture &&
        defaultSheenColorTexture && defaultSheenRoughnessTexture &&
        defaultClearcoatTexture && defaultClearcoatRoughnessTexture &&
        defaultClearcoatNormalTexture &&
        materialUniformBuffer) {
        GPU::BindGroupDescriptor bgDesc{};
        bgDesc.layout  = bindGroupLayout;
        bgDesc.entries = {
            {0,  defaultTexture},
            {1,  defaultSampler},
            {2,  defaultMetallicRoughnessTexture},
            {3,  defaultSampler},
            {4,  defaultNormalTexture},
            {5,  defaultSampler},
            {6,  defaultEmissiveTexture},
            {7,  defaultSampler},
            {8,  defaultOcclusionTexture},
            {9,  defaultSampler},
            {11, defaultSpecularTexture},
            {12, defaultSampler},
            {13, defaultSpecularColorTexture},
            {14, defaultSampler},
            {15, defaultSheenColorTexture},
            {16, defaultSheenRoughnessTexture},
            {17, defaultClearcoatTexture},
            {18, defaultClearcoatRoughnessTexture},
            {19, defaultClearcoatNormalTexture},
            {21, environmentMap},
            // Buffer(17): material uniforms — static after scene load.
            {17, GPU::BufferBinding{materialUniformBuffer, 0, kMaterialUniformStride}},
        };
        defaultBindGroup = device->createBindGroup(bgDesc);

        // Flat-variant bind group: only material buffer, no textures/samplers.
        GPU::BindGroupDescriptor flatBgDesc{};
        flatBgDesc.layout  = bindGroupLayout;
        flatBgDesc.entries = {
            {17, GPU::BufferBinding{materialUniformBuffer, 0, kMaterialUniformStride}},
        };
        defaultFlatBindGroup = device->createBindGroup(flatBgDesc);
    }

    // Per-frame bind groups for lights (10) and camera (18).
    for (uint32_t f = 0; f < kMaxFramesInFlight; ++f) {
        if (!frameBindGroup[f] && bindGroupLayout &&
            frameResources[f].lightsUniformBuffer &&
            frameResources[f].cameraPositionBuffer) {
            GPU::BindGroupDescriptor bgDesc{};
            bgDesc.layout  = bindGroupLayout;
            bgDesc.entries = {
                {10, GPU::BufferBinding{frameResources[f].lightsUniformBuffer, 0, 272}},
                {18, GPU::BufferBinding{frameResources[f].cameraPositionBuffer, 0, 16}},
            };
            frameBindGroup[f] = device->createBindGroup(bgDesc);
        }
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
    // Build one bind group per GLTF material with all material textures.
    // ------------------------------------------------------------------
    materialBindGroups.clear();
    flatMaterialBindGroups.clear();
    if (asset->materials && bindGroupLayout) {
        materialBindGroups.resize(asset->materials->size());
        flatMaterialBindGroups.resize(asset->materials->size());
        for (size_t m = 0; m < asset->materials->size(); ++m) {
            auto &mat = (*asset->materials)[m];

            // Helper to get texture and sampler from a TextureInfo
            auto getTextureAndSampler = [&](const std::shared_ptr<systems::leal::gltf::TextureInfo> &texInfo,
                                            std::shared_ptr<GPU::Texture> &outTex,
                                            std::shared_ptr<GPU::Sampler> &outSamp) {
                if (!texInfo || texInfo->index < 0 || !asset->textures) {
                    return;
                }
                size_t texIdx = (size_t)texInfo->index;
                if (texIdx >= asset->textures->size()) return;

                auto &gt = (*asset->textures)[texIdx];

                // Get image (prefer WebP)
                int64_t imgIdx = (gt.ext_texture_webp >= 0) ? gt.ext_texture_webp : gt.source;
                if (imgIdx >= 0 && imgIdx < (int64_t)gpuTextures.size() && gpuTextures[imgIdx]) {
                    outTex = gpuTextures[imgIdx];
                }

                // Get sampler
                if (gt.sampler >= 0 && (size_t)gt.sampler < gpuSamplers.size() && gpuSamplers[gt.sampler]) {
                    outSamp = gpuSamplers[gt.sampler];
                }
            };

            // Base color texture (binding 0, 1)
            std::shared_ptr<GPU::Texture> baseColorTex = defaultTexture;
            std::shared_ptr<GPU::Sampler> baseColorSamp = defaultSampler;
            if (mat.pbrMetallicRoughness && mat.pbrMetallicRoughness->baseColorTexture) {
                getTextureAndSampler(mat.pbrMetallicRoughness->baseColorTexture, baseColorTex, baseColorSamp);
            }

            // Metallic-roughness texture (binding 2, 3)
            std::shared_ptr<GPU::Texture> mrTex = defaultMetallicRoughnessTexture;
            std::shared_ptr<GPU::Sampler> mrSamp = defaultSampler;
            if (mat.pbrMetallicRoughness && mat.pbrMetallicRoughness->metallicRoughnessTexture) {
                getTextureAndSampler(mat.pbrMetallicRoughness->metallicRoughnessTexture, mrTex, mrSamp);
            }

            // Normal texture (binding 4, 5)
            std::shared_ptr<GPU::Texture> normalTex = defaultNormalTexture;
            std::shared_ptr<GPU::Sampler> normalSamp = defaultSampler;
            if (mat.normalTexture) {
                getTextureAndSampler(mat.normalTexture, normalTex, normalSamp);
            }

            // Emissive texture (binding 6, 7)
            std::shared_ptr<GPU::Texture> emissiveTex = defaultEmissiveTexture;
            std::shared_ptr<GPU::Sampler> emissiveSamp = defaultSampler;
            if (mat.emissiveTexture) {
                getTextureAndSampler(mat.emissiveTexture, emissiveTex, emissiveSamp);
            }

            // Occlusion texture (binding 8, 9)
            std::shared_ptr<GPU::Texture> occlusionTex = defaultOcclusionTexture;
            std::shared_ptr<GPU::Sampler> occlusionSamp = defaultSampler;
            if (mat.occlusionTexture) {
                getTextureAndSampler(mat.occlusionTexture, occlusionTex, occlusionSamp);
            }

            // Specular texture (binding 11, 12) — KHR_materials_specular: A channel
            std::shared_ptr<GPU::Texture> specularTex = defaultSpecularTexture;
            std::shared_ptr<GPU::Sampler> specularSamp = defaultSampler;
            if (mat.khrMaterialsSpecular && mat.khrMaterialsSpecular->specularTexture) {
                getTextureAndSampler(mat.khrMaterialsSpecular->specularTexture, specularTex, specularSamp);
            }

            // Specular color texture (binding 13, 14) — KHR_materials_specular: RGB F0 color tint
            std::shared_ptr<GPU::Texture> specularColorTex = defaultSpecularColorTexture;
            std::shared_ptr<GPU::Sampler> specularColorSamp = defaultSampler;
            if (mat.khrMaterialsSpecular && mat.khrMaterialsSpecular->specularColorTexture) {
                getTextureAndSampler(mat.khrMaterialsSpecular->specularColorTexture, specularColorTex, specularColorSamp);
            }

            // Sheen color texture (binding 15) — KHR_materials_sheen: RGB sRGB sheen color
            // Note: no dedicated sampler; shader reuses baseColorSampler (Metal sampler slot limit).
            std::shared_ptr<GPU::Texture> sheenColorTex = defaultSheenColorTexture;
            {
                std::shared_ptr<GPU::Sampler> unused = defaultSampler;
                if (mat.khrMaterialsSheen && mat.khrMaterialsSheen->sheenColorTexture)
                    getTextureAndSampler(mat.khrMaterialsSheen->sheenColorTexture, sheenColorTex, unused);
            }

            // Sheen roughness texture (binding 16) — KHR_materials_sheen: R channel = roughness
            std::shared_ptr<GPU::Texture> sheenRoughnessTex = defaultSheenRoughnessTexture;
            {
                std::shared_ptr<GPU::Sampler> unused = defaultSampler;
                if (mat.khrMaterialsSheen && mat.khrMaterialsSheen->sheenRoughnessTexture)
                    getTextureAndSampler(mat.khrMaterialsSheen->sheenRoughnessTexture, sheenRoughnessTex, unused);
            }

            // Clearcoat intensity texture (binding 17) — KHR_materials_clearcoat: R channel
            std::shared_ptr<GPU::Texture> clearcoatTex = defaultClearcoatTexture;
            {
                std::shared_ptr<GPU::Sampler> unused = defaultSampler;
                if (mat.khrMaterialsClearcoat && mat.khrMaterialsClearcoat->clearcoatTexture)
                    getTextureAndSampler(mat.khrMaterialsClearcoat->clearcoatTexture, clearcoatTex, unused);
            }

            // Clearcoat roughness texture (binding 18) — KHR_materials_clearcoat: G channel
            std::shared_ptr<GPU::Texture> clearcoatRoughnessTex = defaultClearcoatRoughnessTexture;
            {
                std::shared_ptr<GPU::Sampler> unused = defaultSampler;
                if (mat.khrMaterialsClearcoat && mat.khrMaterialsClearcoat->clearcoatRoughnessTexture)
                    getTextureAndSampler(mat.khrMaterialsClearcoat->clearcoatRoughnessTexture, clearcoatRoughnessTex, unused);
            }

            // Clearcoat normal texture (binding 19) — KHR_materials_clearcoat: tangent-space normal
            std::shared_ptr<GPU::Texture> clearcoatNormalTex = defaultClearcoatNormalTexture;
            {
                std::shared_ptr<GPU::Sampler> unused = defaultSampler;
                if (mat.khrMaterialsClearcoat && mat.khrMaterialsClearcoat->clearcoatNormalTexture)
                    getTextureAndSampler(mat.khrMaterialsClearcoat->clearcoatNormalTexture, clearcoatNormalTex, unused);
            }

            // Transmission texture (binding 20) — KHR_materials_transmission: R channel scales transmissionFactor
            std::shared_ptr<GPU::Texture> transmissionTex = defaultTexture;  // White = full transmission
            std::shared_ptr<GPU::Sampler> transmissionSamp = defaultSampler;
            if (mat.khrMaterialsTransmission && mat.khrMaterialsTransmission->transmissionTexture) {
                getTextureAndSampler(mat.khrMaterialsTransmission->transmissionTexture, transmissionTex, transmissionSamp);
            }

            // Create bind group with all textures and static material buffer.
            // Lights (10) and camera (18) are bound separately via frameBindGroup.
            GPU::BindGroupDescriptor bgDesc{};
            bgDesc.layout  = bindGroupLayout;
            bgDesc.entries = {
                {0,  baseColorTex},
                {1,  baseColorSamp},
                {2,  mrTex},
                {3,  mrSamp},
                {4,  normalTex},
                {5,  normalSamp},
                {6,  emissiveTex},
                {7,  emissiveSamp},
                {8,  occlusionTex},
                {9,  occlusionSamp},
                {11, specularTex},
                {12, specularSamp},
                {13, specularColorTex},
                {14, specularColorSamp},
                {15, sheenColorTex},
                {16, sheenRoughnessTex},
                {17, clearcoatTex},
                {18, clearcoatRoughnessTex},
                {19, clearcoatNormalTex},
                {20, transmissionTex},
                {21, environmentMap},
                // Buffer(17): material uniforms at the per-material offset.
                {17, GPU::BufferBinding{materialUniformBuffer,
                                        (uint64_t)(m + 1) * kMaterialUniformStride,
                                        kMaterialUniformStride}},
            };
            materialBindGroups[m] = device->createBindGroup(bgDesc);

            // Flat-variant bind group: only material buffer, no textures/samplers.
            GPU::BindGroupDescriptor flatBgDesc{};
            flatBgDesc.layout  = bindGroupLayout;
            flatBgDesc.entries = {
                {17, GPU::BufferBinding{materialUniformBuffer,
                                        (uint64_t)(m + 1) * kMaterialUniformStride,
                                        kMaterialUniformStride}},
            };
            flatMaterialBindGroups[m] = device->createBindGroup(flatBgDesc);
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
    // Load EXT_mesh_gpu_instancing data.
    //
    // For each node with extMeshGpuInstancing, read the TRANSLATION, ROTATION,
    // and SCALE accessors and build per-instance transform matrices.
    // The matrices are uploaded to a GPU buffer (column-major float4x4).
    // ------------------------------------------------------------------
    nodeInstanceData.clear();
    if (asset->nodes) {
        for (uint64_t nodeIdx = 0; nodeIdx < asset->nodes->size(); ++nodeIdx) {
            auto &node = (*asset->nodes)[nodeIdx];
            if (!node.extMeshGpuInstancing) continue;

            // Get accessor indices for translation, rotation, scale
            int64_t transIdx = -1, rotIdx = -1, scaleIdx = -1;
            auto &attrs = node.extMeshGpuInstancing->attributes;
            auto it = attrs.find("TRANSLATION");
            if (it != attrs.end()) transIdx = (int64_t)it->second;
            it = attrs.find("ROTATION");
            if (it != attrs.end()) rotIdx = (int64_t)it->second;
            it = attrs.find("SCALE");
            if (it != attrs.end()) scaleIdx = (int64_t)it->second;

            // Determine instance count from the first available accessor
            uint32_t instanceCount = 0;
            if (transIdx >= 0 && asset->accessors) instanceCount = (uint32_t)(*asset->accessors)[(size_t)transIdx].count;
            else if (rotIdx >= 0 && asset->accessors) instanceCount = (uint32_t)(*asset->accessors)[(size_t)rotIdx].count;
            else if (scaleIdx >= 0 && asset->accessors) instanceCount = (uint32_t)(*asset->accessors)[(size_t)scaleIdx].count;

            if (instanceCount == 0) continue;

            // Read accessor data
            namespace VM = systems::leal::vector_math;
            std::vector<VM::Vector3<float>> translations(instanceCount, VM::Vector3<float>(0, 0, 0));
            std::vector<VM::Quaternion<float>> rotations(instanceCount, VM::Quaternion<float>(0, 0, 0, 1));
            std::vector<VM::Vector3<float>> scales(instanceCount, VM::Vector3<float>(1, 1, 1));

            auto readFloatData = [&](int64_t accIdx, std::vector<float> &outData) {
                if (accIdx < 0 || !asset->accessors) return;
                auto &acc = (*asset->accessors)[(size_t)accIdx];
                if (acc.bufferView < 0 || !asset->bufferViews) return;
                auto &bv = (*asset->bufferViews)[(size_t)acc.bufferView];
                if (bv.buffer < 0 || !asset->buffers) return;
                auto &buf = (*asset->buffers)[(size_t)bv.buffer];

                size_t numComponents = 0;
                switch (acc.type) {
                    case systems::leal::gltf::AccessorType::acScalar: numComponents = 1; break;
                    case systems::leal::gltf::AccessorType::acVec2:   numComponents = 2; break;
                    case systems::leal::gltf::AccessorType::acVec3:   numComponents = 3; break;
                    case systems::leal::gltf::AccessorType::acVec4:   numComponents = 4; break;
                    case systems::leal::gltf::AccessorType::acMat2:   numComponents = 4; break;
                    case systems::leal::gltf::AccessorType::acMat3:   numComponents = 9; break;
                    case systems::leal::gltf::AccessorType::acMat4:   numComponents = 16; break;
                }

                outData.resize(acc.count * numComponents);
                const uint8_t *src = buf.data.data() + bv.byteOffset + acc.byteOffset;
                // Handle component type
                switch (acc.componentType) {
                    case systems::leal::gltf::ComponentType::ctFloat:
                        memcpy(outData.data(), src, acc.count * numComponents * sizeof(float));
                        break;
                    default:
                        // Other component types not yet supported for instancing
                        break;
                }
            };

            std::vector<float> transData, rotData, scaleData;
            readFloatData(transIdx, transData);
            readFloatData(rotIdx, rotData);
            readFloatData(scaleIdx, scaleData);

            for (uint32_t i = 0; i < instanceCount; ++i) {
                if (transIdx >= 0 && !transData.empty()) {
                    translations[i] = VM::Vector3<float>(
                        transData[i * 3 + 0],
                        transData[i * 3 + 1],
                        transData[i * 3 + 2]);
                }
                if (rotIdx >= 0 && !rotData.empty()) {
                    rotations[i] = VM::Quaternion<float>(
                        rotData[i * 4 + 0],
                        rotData[i * 4 + 1],
                        rotData[i * 4 + 2],
                        rotData[i * 4 + 3]);
                }
                if (scaleIdx >= 0 && !scaleData.empty()) {
                    scales[i] = VM::Vector3<float>(
                        scaleData[i * 3 + 0],
                        scaleData[i * 3 + 1],
                        scaleData[i * 3 + 2]);
                }
            }

            // Build instance matrices (column-major for Metal)
            std::vector<float> instanceMatrices(instanceCount * 16);
            std::vector<VM::Matrix4<double>> cpuMatrices(instanceCount);
            for (uint32_t i = 0; i < instanceCount; ++i) {
                VM::Matrix4<float> m = VM::Matrix4<float>::compose(translations[i], rotations[i], scales[i]);
                cpuMatrices[i] = VM::Matrix4<double>::compose(
                    VM::Vector3<double>(translations[i].x(), translations[i].y(), translations[i].z()),
                    VM::Quaternion<double>(rotations[i].x(), rotations[i].y(), rotations[i].z(), rotations[i].w()),
                    VM::Vector3<double>(scales[i].x(), scales[i].y(), scales[i].z()));
                // Transpose to column-major for Metal
                for (int row = 0; row < 4; ++row) {
                    for (int col = 0; col < 4; ++col) {
                        instanceMatrices[i * 16 + col * 4 + row] = m.data[row * 4 + col];
                    }
                }
            }

            // Upload to GPU buffer
            using BU = systems::leal::campello_gpu::BufferUsage;
            auto matrixBuffer = device->createBuffer(
                instanceMatrices.size() * sizeof(float),
                BU::vertex,
                reinterpret_cast<uint8_t*>(instanceMatrices.data()));

            if (matrixBuffer) {
                InstanceData data;
                data.matrixBuffer = matrixBuffer;
                data.instanceCount = instanceCount;
                data.cpuMatrices = std::move(cpuMatrices);
                data.visibleMatrices.resize((size_t)instanceCount * 16);
                data.visibleCount = instanceCount;
                nodeInstanceData[nodeIdx] = std::move(data);
            }
        }
    }

    primitiveBounds.clear();
    if (asset->meshes) {
        for (auto &mesh : *asset->meshes) {
            for (auto &primitive : mesh.primitives) {
                primitiveBounds[&primitive] = computePrimitiveBounds(primitive);
            }
        }
    }

    if (asset->nodes) {
        for (uint64_t nodeIdx = 0; nodeIdx < asset->nodes->size(); ++nodeIdx) {
            Bounds merged;
            auto &node = (*asset->nodes)[nodeIdx];
            if (node.mesh >= 0 && asset->meshes && (size_t)node.mesh < asset->meshes->size()) {
                auto &mesh = (*asset->meshes)[(size_t)node.mesh];
                for (auto &primitive : mesh.primitives) {
                    auto it = primitiveBounds.find(&primitive);
                    if (it != primitiveBounds.end()) {
                        merged = mergeBounds(merged, it->second);
                    }
                }
            }
            nodeMeshLocalBounds[nodeIdx] = merged;
            nodeLocalBounds[nodeIdx] = merged;
        }

        for (auto &[nodeIdx, instanceData] : nodeInstanceData) {
            if (nodeIdx >= nodeLocalBounds.size()) continue;
            if (!nodeLocalBounds[nodeIdx].valid || instanceData.cpuMatrices.empty()) continue;

            Bounds instancedBounds;
            for (auto &instanceMatrix : instanceData.cpuMatrices) {
                instancedBounds = mergeBounds(
                    instancedBounds,
                    transformBounds(nodeLocalBounds[nodeIdx], instanceMatrix));
            }
            nodeLocalBounds[nodeIdx] = instancedBounds;
        }
    }

    // ------------------------------------------------------------------
    // Load skinning data (skeletal meshes).
    // ------------------------------------------------------------------
    skinData.clear();
    nodeSkinIndex.clear();
    totalJointMatrixBytes = 0;
    if (asset->nodes) {
        nodeSkinIndex.assign(asset->nodes->size(), -1);
        for (uint64_t nodeIdx = 0; nodeIdx < asset->nodes->size(); ++nodeIdx) {
            auto &node = (*asset->nodes)[nodeIdx];
            if (node.skin >= 0) {
                nodeSkinIndex[nodeIdx] = node.skin;
            }
        }
    }
    if (asset->skins && asset->accessors && asset->bufferViews && asset->buffers) {
        uint64_t gpuOffset = 0;
        for (size_t skinIdx = 0; skinIdx < asset->skins->size(); ++skinIdx) {
            auto &gltfSkin = (*asset->skins)[skinIdx];
            SkinData sd;
            sd.jointCount = gltfSkin.joints ? gltfSkin.joints->size() : 0;
            if (sd.jointCount == 0) {
                skinData.push_back(std::move(sd));
                continue;
            }
            // Read inverse bind matrices accessor (float4x4 per joint).
            if (gltfSkin.inverseBindMatrices >= 0 &&
                (size_t)gltfSkin.inverseBindMatrices < asset->accessors->size()) {
                auto &ibmAcc = (*asset->accessors)[(size_t)gltfSkin.inverseBindMatrices];
                if (ibmAcc.bufferView >= 0 && (size_t)ibmAcc.bufferView < asset->bufferViews->size()) {
                    auto &ibmBV = (*asset->bufferViews)[(size_t)ibmAcc.bufferView];
                    if (ibmBV.buffer >= 0 && (size_t)ibmBV.buffer < asset->buffers->size()) {
                        auto &buf = (*asset->buffers)[(size_t)ibmBV.buffer];
                        const uint8_t *src = buf.data.data() + ibmBV.byteOffset + ibmAcc.byteOffset;
                        sd.inverseBindMatrices.resize(sd.jointCount * 16);
                        // GLTF stores matrices as 16 floats (column-major).
                        // Our Matrix4 uses row-major, so we need to transpose.
                        const float *fSrc = reinterpret_cast<const float*>(src);
                        for (uint64_t j = 0; j < sd.jointCount; ++j) {
                            for (int row = 0; row < 4; ++row) {
                                for (int col = 0; col < 4; ++col) {
                                    // src is column-major: src[col*4 + row]
                                    // Store in row-major for our Matrix4
                                    float val = fSrc[j * 16 + col * 4 + row];
                                    sd.inverseBindMatrices[j * 16 + row * 4 + col] = val;
                                }
                            }
                        }
                    }
                }
            }
            // Cache joint node indices.
            if (gltfSkin.joints) {
                for (auto jn : *gltfSkin.joints) {
                    sd.jointNodeIndices.push_back(jn);
                }
            }
            // 256-byte aligned offset into joint matrix buffer.
            sd.gpuOffset = gpuOffset;
            uint64_t bytesNeeded = sd.jointCount * 64;
            uint64_t paddedBytes = ((bytesNeeded + 255) / 256) * 256;
            gpuOffset += paddedBytes;
            skinData.push_back(std::move(sd));
        }
        totalJointMatrixBytes = gpuOffset;
    }

    // Fallback buffers for primitives without JOINTS_0 / WEIGHTS_0.
    if (!fallbackJointBuffer) {
        constexpr uint64_t kFallbackJointSize = 256 * 1024; // enough zero uint4s
        std::vector<uint8_t> zeros(kFallbackJointSize, 0);
        fallbackJointBuffer = device->createBuffer(
            kFallbackJointSize, GPU::BufferUsage::vertex, zeros.data());
    }
    if (!fallbackWeightBuffer) {
        constexpr uint64_t kFallbackWeightSize = 256 * 1024; // enough zero float4s
        std::vector<uint8_t> zeros(kFallbackWeightSize, 0);
        fallbackWeightBuffer = device->createBuffer(
            kFallbackWeightSize, GPU::BufferUsage::vertex, zeros.data());
    }
    // Default identity joint matrix for non-skinned draws.
    if (!defaultJointMatrixBuffer) {
        float identity[16] = {
            1, 0, 0, 0,  // column 0
            0, 1, 0, 0,  // column 1
            0, 0, 1, 0,  // column 2
            0, 0, 0, 1   // column 3
        };
        defaultJointMatrixBuffer = device->createBuffer(
            64, GPU::BufferUsage::vertex, reinterpret_cast<uint8_t*>(identity));
    }

    // Per-frame joint matrix buffers.
    if (totalJointMatrixBytes > 0) {
        jointMatrixData.resize(totalJointMatrixBytes / sizeof(float));
        for (uint32_t f = 0; f < kMaxFramesInFlight; ++f) {
            frameResources[f].jointMatrixBuffer = device->createBuffer(
                totalJointMatrixBytes, GPU::BufferUsage::vertex);
        }
    }

    // Detect actual JOINTS_0 / WEIGHTS_0 component types and recreate pipelines if needed.
    {
        GPU::ComponentType detectedJointsType = jointsComponentType;
        GPU::ComponentType detectedWeightsType = weightsComponentType;
        bool detectedWeightsNorm = weightsNormalized;
        bool foundJoints = false, foundWeights = false;
        if (asset->meshes && asset->accessors) {
            for (auto &mesh : *asset->meshes) {
                for (auto &primitive : mesh.primitives) {
                    if (!foundJoints) {
                        auto jit = primitive.attributes.find("JOINTS_0");
                        if (jit != primitive.attributes.end() && jit->second >= 0 &&
                            (size_t)jit->second < asset->accessors->size()) {
                            detectedJointsType = static_cast<GPU::ComponentType>(
                                static_cast<int>((*asset->accessors)[(size_t)jit->second].componentType));
                            foundJoints = true;
                        }
                    }
                    if (!foundWeights) {
                        auto wit = primitive.attributes.find("WEIGHTS_0");
                        if (wit != primitive.attributes.end() && wit->second >= 0 &&
                            (size_t)wit->second < asset->accessors->size()) {
                            auto &acc = (*asset->accessors)[(size_t)wit->second];
                            detectedWeightsType = static_cast<GPU::ComponentType>(
                                static_cast<int>(acc.componentType));
                            detectedWeightsNorm = acc.normalized;
                            foundWeights = true;
                        }
                    }
                    if (foundJoints && foundWeights) break;
                }
                if (foundJoints && foundWeights) break;
            }
        }
        if (detectedJointsType != jointsComponentType ||
            detectedWeightsType != weightsComponentType ||
            detectedWeightsNorm != weightsNormalized) {
            jointsComponentType = detectedJointsType;
            weightsComponentType = detectedWeightsType;
            weightsNormalized = detectedWeightsNorm;
            // Recreate default pipelines with the correct vertex descriptor.
            if (cachedColorFormat != GPU::PixelFormat::invalid) {
                createDefaultPipelines(cachedColorFormat);
            }
            // Clear quantized pipelines so they get recreated with the new types.
            quantizedPipelines.clear();
        }
    }

    // Reset animation state when switching scenes — ensures clean slate.
    stopAllAnimations();

    // Create quantized pipeline variants if this asset uses KHR_mesh_quantization.
    createQuantizedPipelinesIfNeeded();
}

// ---------------------------------------------------------------------------
// Pipeline / resize
// ---------------------------------------------------------------------------

void Renderer::createQuantizedPipelinesIfNeeded() {
    namespace GPU = systems::leal::campello_gpu;

    if (!quantizedPipelines.empty()) return;
    if (!asset || !asset->meshes || !asset->accessors) return;
    if (cachedColorFormat == GPU::PixelFormat::invalid) return;

    // Scan the scene to find the first non-float accessor and determine
    // which component types are used for each semantic.
    GPU::ComponentType posType = GPU::ComponentType::ctFloat;
    GPU::ComponentType normType = GPU::ComponentType::ctFloat;
    GPU::ComponentType texType = GPU::ComponentType::ctFloat;
    GPU::ComponentType tanType = GPU::ComponentType::ctFloat;
    bool hasQuantized = false;

    auto detectType = [&](const std::string &semantic, GPU::ComponentType &outType) {
        for (auto &mesh : *asset->meshes) {
            for (auto &primitive : mesh.primitives) {
                auto it = primitive.attributes.find(semantic);
                if (it == primitive.attributes.end()) continue;
                int64_t accIdx = it->second;
                if (accIdx < 0 || (size_t)accIdx >= asset->accessors->size()) continue;
                auto &acc = (*asset->accessors)[(size_t)accIdx];
                if (acc.componentType != systems::leal::gltf::ComponentType::ctFloat) {
                    outType = static_cast<GPU::ComponentType>(static_cast<int>(acc.componentType));
                    hasQuantized = true;
                }
            }
        }
    };

    detectType("POSITION", posType);
    detectType("NORMAL", normType);
    detectType("TEXCOORD_0", texType);
    detectType("TANGENT", tanType);

    if (!hasQuantized) return;

#if defined(__APPLE__)
    using namespace systems::leal::campello_renderer::shaders;
    auto shaderModule = device->createShaderModule(kDefaultMetalShader, kDefaultMetalShaderSize);
    if (!shaderModule) return;

    auto makeLayout = [](GPU::ComponentType ct, GPU::AccessorType at,
                         double stride, GPU::StepMode sm, uint32_t location,
                         bool normalized) {
        GPU::VertexLayout layout{};
        layout.arrayStride = stride;
        layout.stepMode    = sm;
        GPU::VertexAttribute attr{};
        attr.componentType  = ct;
        attr.accessorType   = at;
        attr.offset         = 0;
        attr.shaderLocation = location;
        attr.normalized     = normalized;
        layout.attributes.push_back(attr);
        return layout;
    };

    // Strides depend on component size, padded to 4-byte boundary for Metal.
    auto compSize = [](GPU::ComponentType ct) -> double {
        switch (ct) {
            case GPU::ComponentType::ctByte: return 1;
            case GPU::ComponentType::ctUnsignedByte: return 1;
            case GPU::ComponentType::ctShort: return 2;
            case GPU::ComponentType::ctUnsignedShort: return 2;
            case GPU::ComponentType::ctUnsignedInt: return 4;
            case GPU::ComponentType::ctFloat: return 4;
        }
        return 4;
    };
    auto paddedStride = [&](double raw) -> double {
        uint32_t u = static_cast<uint32_t>(raw);
        return static_cast<double>((u + 3) & ~uint32_t(3));
    };

    // Only integer types are normalized for KHR_mesh_quantization.
    bool posNorm = (posType != GPU::ComponentType::ctFloat && posType != GPU::ComponentType::ctUnsignedInt);
    bool normNorm = (normType != GPU::ComponentType::ctFloat && normType != GPU::ComponentType::ctUnsignedInt);
    bool texNorm = (texType != GPU::ComponentType::ctFloat && texType != GPU::ComponentType::ctUnsignedInt);
    bool tanNorm = (tanType != GPU::ComponentType::ctFloat && tanType != GPU::ComponentType::ctUnsignedInt);

    GPU::RenderPipelineDescriptor base{};
    base.vertex.module     = shaderModule;
    base.vertex.entryPoint = "vertexMain";
    base.vertex.buffers.push_back(makeLayout(
        posType, GPU::AccessorType::acVec3,
        paddedStride(3.0 * compSize(posType)), GPU::StepMode::vertex, VERTEX_SLOT_POSITION, posNorm));
    base.vertex.buffers.push_back(makeLayout(
        normType, GPU::AccessorType::acVec3,
        paddedStride(3.0 * compSize(normType)), GPU::StepMode::vertex, VERTEX_SLOT_NORMAL, normNorm));
    base.vertex.buffers.push_back(makeLayout(
        texType, GPU::AccessorType::acVec2,
        paddedStride(2.0 * compSize(texType)), GPU::StepMode::vertex, VERTEX_SLOT_TEXCOORD0, texNorm));
    base.vertex.buffers.push_back(makeLayout(
        tanType, GPU::AccessorType::acVec4,
        paddedStride(4.0 * compSize(tanType)), GPU::StepMode::vertex, VERTEX_SLOT_TANGENT, tanNorm));
    // JOINTS_0 / WEIGHTS_0 for skinning — use detected component types.
    double qJointsStride = (jointsComponentType == GPU::ComponentType::ctUnsignedShort) ? 8.0 : 4.0;
    base.vertex.buffers.push_back(makeLayout(
        jointsComponentType, GPU::AccessorType::acVec4,
        qJointsStride, GPU::StepMode::vertex, VERTEX_SLOT_JOINTS, false));
    double qWeightsStride = (weightsComponentType == GPU::ComponentType::ctFloat) ? 16.0 :
                            (weightsComponentType == GPU::ComponentType::ctUnsignedShort) ? 8.0 : 4.0;
    base.vertex.buffers.push_back(makeLayout(
        weightsComponentType, GPU::AccessorType::acVec4,
        qWeightsStride, GPU::StepMode::vertex, VERTEX_SLOT_WEIGHTS, weightsNormalized));

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
    cs.format    = cachedColorFormat;
    cs.writeMask = GPU::ColorWrite::all;

    // Helper to create a variant and store it in the map.
    auto storeVariant = [&](int variant, const GPU::RenderPipelineDescriptor &d) {
        auto pipe = device->createRenderPipeline(d);
        if (pipe) quantizedPipelines[variant] = pipe;
    };

    // Variant 1: flat
    {
        GPU::RenderPipelineDescriptor d = base;
        GPU::FragmentDescriptor frag{};
        frag.module     = shaderModule;
        frag.entryPoint = "fragmentMain_flat";
        frag.targets.push_back(cs);
        d.fragment = frag;
        storeVariant(1, d);
    }
    // Variant 2: textured
    {
        GPU::RenderPipelineDescriptor d = base;
        GPU::FragmentDescriptor frag{};
        frag.module     = shaderModule;
        frag.entryPoint = "fragmentMain_textured";
        frag.targets.push_back(cs);
        d.fragment = frag;
        storeVariant(2, d);
    }
    // Variant 3: debug
    {
        GPU::RenderPipelineDescriptor d = base;
        GPU::FragmentDescriptor frag{};
        frag.module     = shaderModule;
        frag.entryPoint = "fragmentMain_debug";
        frag.targets.push_back(cs);
        d.fragment = frag;
        storeVariant(3, d);
    }
    // Variant 4: flat double-sided
    {
        GPU::RenderPipelineDescriptor d = base;
        d.cullMode = GPU::CullMode::none;
        GPU::FragmentDescriptor frag{};
        frag.module     = shaderModule;
        frag.entryPoint = "fragmentMain_flat";
        frag.targets.push_back(cs);
        d.fragment = frag;
        storeVariant(4, d);
    }
    // Variant 5: textured double-sided
    {
        GPU::RenderPipelineDescriptor d = base;
        d.cullMode = GPU::CullMode::none;
        GPU::FragmentDescriptor frag{};
        frag.module     = shaderModule;
        frag.entryPoint = "fragmentMain_textured";
        frag.targets.push_back(cs);
        d.fragment = frag;
        storeVariant(5, d);
    }
    // Blend state
    GPU::BlendState alphaBlend{};
    alphaBlend.color = { GPU::BlendFactor::srcAlpha, GPU::BlendFactor::oneMinusSrcAlpha, GPU::BlendOperation::add };
    alphaBlend.alpha = { GPU::BlendFactor::one, GPU::BlendFactor::oneMinusSrcAlpha, GPU::BlendOperation::add };
    GPU::DepthStencilDescriptor blendDs = ds;
    blendDs.depthWriteEnabled = false;
    // Variant 6: flat blend
    {
        GPU::RenderPipelineDescriptor d = base;
        d.depthStencil = blendDs;
        GPU::ColorState blendCs = cs;
        blendCs.blend = alphaBlend;
        GPU::FragmentDescriptor frag{};
        frag.module     = shaderModule;
        frag.entryPoint = "fragmentMain_flat";
        frag.targets.push_back(blendCs);
        d.fragment = frag;
        storeVariant(6, d);
    }
    // Variant 7: textured blend
    {
        GPU::RenderPipelineDescriptor d = base;
        d.depthStencil = blendDs;
        GPU::ColorState blendCs = cs;
        blendCs.blend = alphaBlend;
        GPU::FragmentDescriptor frag{};
        frag.module     = shaderModule;
        frag.entryPoint = "fragmentMain_textured";
        frag.targets.push_back(blendCs);
        d.fragment = frag;
        storeVariant(7, d);
    }
    // Variant 8: flat blend double-sided
    {
        GPU::RenderPipelineDescriptor d = base;
        d.cullMode = GPU::CullMode::none;
        d.depthStencil = blendDs;
        GPU::ColorState blendCs = cs;
        blendCs.blend = alphaBlend;
        GPU::FragmentDescriptor frag{};
        frag.module     = shaderModule;
        frag.entryPoint = "fragmentMain_flat";
        frag.targets.push_back(blendCs);
        d.fragment = frag;
        storeVariant(8, d);
    }
    // Variant 9: textured blend double-sided
    {
        GPU::RenderPipelineDescriptor d = base;
        d.cullMode = GPU::CullMode::none;
        d.depthStencil = blendDs;
        GPU::ColorState blendCs = cs;
        blendCs.blend = alphaBlend;
        GPU::FragmentDescriptor frag{};
        frag.module     = shaderModule;
        frag.entryPoint = "fragmentMain_textured";
        frag.targets.push_back(blendCs);
        d.fragment = frag;
        storeVariant(9, d);
    }
#endif
}

void Renderer::createDefaultPipelines(systems::leal::campello_gpu::PixelFormat colorFormat) {
    namespace GPU = systems::leal::campello_gpu;
    cachedColorFormat = colorFormat;

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

    // Slots 0–5: stage_in attributes. Slots 16/17 are raw [[buffer(N)]] in the
    // shader and are not part of the vertex descriptor.
    auto makeLayout = [](GPU::ComponentType ct, GPU::AccessorType at,
                         double stride, GPU::StepMode sm, uint32_t location,
                         bool normalized = false) {
        GPU::VertexLayout layout{};
        layout.arrayStride = stride;
        layout.stepMode    = sm;
        GPU::VertexAttribute attr{};
        attr.componentType  = ct;
        attr.accessorType   = at;
        attr.offset         = 0;
        attr.shaderLocation = location;
        attr.normalized     = normalized;
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
    // JOINTS_0 / WEIGHTS_0 — use detected component types (may be recreated in setScene()).
    double jointsStride = (jointsComponentType == GPU::ComponentType::ctUnsignedShort) ? 8.0 : 4.0;
    base.vertex.buffers.push_back(makeLayout(
        jointsComponentType, GPU::AccessorType::acVec4,
        jointsStride, GPU::StepMode::vertex, VERTEX_SLOT_JOINTS));
    double weightsStride = (weightsComponentType == GPU::ComponentType::ctFloat) ? 16.0 :
                           (weightsComponentType == GPU::ComponentType::ctUnsignedShort) ? 8.0 : 4.0;
    base.vertex.buffers.push_back(makeLayout(
        weightsComponentType, GPU::AccessorType::acVec4,
        weightsStride, GPU::StepMode::vertex, VERTEX_SLOT_WEIGHTS, weightsNormalized));

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

    // --- Variant: debug (flat normal visualization) ---
    {
        GPU::RenderPipelineDescriptor d = base;
        GPU::FragmentDescriptor frag{};
        frag.module     = shaderModule;
        frag.entryPoint = "fragmentMain_debug";
        frag.targets.push_back(cs);
        d.fragment = frag;
        pipelineDebug = device->createRenderPipeline(d);
    }

    // --- Double-sided variants (no back-face culling) ---
    {
        GPU::RenderPipelineDescriptor d = base;
        d.cullMode = GPU::CullMode::none;
        GPU::FragmentDescriptor frag{};
        frag.module     = shaderModule;
        frag.entryPoint = "fragmentMain_flat";
        frag.targets.push_back(cs);
        d.fragment = frag;
        pipelineFlatDoubleSided = device->createRenderPipeline(d);
    }
    {
        GPU::RenderPipelineDescriptor d = base;
        d.cullMode = GPU::CullMode::none;
        GPU::FragmentDescriptor frag{};
        frag.module     = shaderModule;
        frag.entryPoint = "fragmentMain_textured";
        frag.targets.push_back(cs);
        d.fragment = frag;
        pipelineTexturedDoubleSided = device->createRenderPipeline(d);
    }

    // --- Alpha-blend variants (transparency) ---
    // Standard alpha blending: src * srcAlpha + dst * (1 - srcAlpha)
    // Depth write disabled for transparent materials to avoid artifacts
    GPU::BlendState alphaBlend{};
    alphaBlend.color = { GPU::BlendFactor::srcAlpha, GPU::BlendFactor::oneMinusSrcAlpha, GPU::BlendOperation::add };
    alphaBlend.alpha = { GPU::BlendFactor::one, GPU::BlendFactor::oneMinusSrcAlpha, GPU::BlendOperation::add };
    
    GPU::DepthStencilDescriptor blendDs = ds;
    blendDs.depthWriteEnabled = false;  // Don't write depth for transparent objects
    
    {
        GPU::RenderPipelineDescriptor d = base;
        d.depthStencil = blendDs;
        GPU::ColorState blendCs = cs;
        blendCs.blend = alphaBlend;
        GPU::FragmentDescriptor frag{};
        frag.module     = shaderModule;
        frag.entryPoint = "fragmentMain_flat";
        frag.targets.push_back(blendCs);
        d.fragment = frag;
        pipelineFlatBlend = device->createRenderPipeline(d);
    }
    {
        GPU::RenderPipelineDescriptor d = base;
        d.depthStencil = blendDs;
        GPU::ColorState blendCs = cs;
        blendCs.blend = alphaBlend;
        GPU::FragmentDescriptor frag{};
        frag.module     = shaderModule;
        frag.entryPoint = "fragmentMain_textured";
        frag.targets.push_back(blendCs);
        d.fragment = frag;
        pipelineTexturedBlend = device->createRenderPipeline(d);
    }
    {
        GPU::RenderPipelineDescriptor d = base;
        d.cullMode = GPU::CullMode::none;
        d.depthStencil = blendDs;
        GPU::ColorState blendCs = cs;
        blendCs.blend = alphaBlend;
        GPU::FragmentDescriptor frag{};
        frag.module     = shaderModule;
        frag.entryPoint = "fragmentMain_flat";
        frag.targets.push_back(blendCs);
        d.fragment = frag;
        pipelineFlatBlendDoubleSided = device->createRenderPipeline(d);
    }
    {
        GPU::RenderPipelineDescriptor d = base;
        d.cullMode = GPU::CullMode::none;
        d.depthStencil = blendDs;
        GPU::ColorState blendCs = cs;
        blendCs.blend = alphaBlend;
        GPU::FragmentDescriptor frag{};
        frag.module     = shaderModule;
        frag.entryPoint = "fragmentMain_textured";
        frag.targets.push_back(blendCs);
        d.fragment = frag;
        pipelineTexturedBlendDoubleSided = device->createRenderPipeline(d);
    }

    // --- Skybox pipeline ---
    {
        // Skybox bind group layout: cube texture (0), sampler (1), uniform buffer (2).
        if (!skyboxBindGroupLayout) {
            GPU::BindGroupLayoutDescriptor sbglDesc{};
            GPU::EntryObject sbTex{};
            sbTex.binding = 0;
            sbTex.visibility = GPU::ShaderStage::fragment;
            sbTex.type = GPU::EntryObjectType::texture;
            sbTex.data.texture.multisampled = false;
            sbTex.data.texture.sampleType = GPU::EntryObjectTextureType::ttFloat;
            sbTex.data.texture.viewDimension = GPU::TextureType::ttCube;
            sbglDesc.entries.push_back(sbTex);

            GPU::EntryObject sbSamp{};
            sbSamp.binding = 1;
            sbSamp.visibility = GPU::ShaderStage::fragment;
            sbSamp.type = GPU::EntryObjectType::sampler;
            sbSamp.data.sampler.type = GPU::EntryObjectSamplerType::filtering;
            sbglDesc.entries.push_back(sbSamp);

            GPU::EntryObject sbBuf{};
            sbBuf.binding = 2;
            sbBuf.visibility = GPU::ShaderStage::fragment;
            sbBuf.type = GPU::EntryObjectType::buffer;
            sbBuf.data.buffer.hasDinamicOffaset = false;
            sbBuf.data.buffer.minBindingSize = 96;
            sbBuf.data.buffer.type = GPU::EntryObjectBufferType::uniform;
            sbglDesc.entries.push_back(sbBuf);

            skyboxBindGroupLayout = device->createBindGroupLayout(sbglDesc);
        }

        GPU::RenderPipelineDescriptor skyDesc{};
        skyDesc.vertex.module = shaderModule;
        skyDesc.vertex.entryPoint = "skyboxVertex";
        // No vertex buffers — fullscreen triangle generated from vertex_id.

        GPU::DepthStencilDescriptor skyDs{};
        skyDs.format = GPU::PixelFormat::depth32float;
        skyDs.depthWriteEnabled = false;   // Don't write depth
        skyDs.depthCompare = GPU::CompareOp::lessEqual;
        skyDs.depthBias = 0.0;
        skyDs.depthBiasClamp = 0.0;
        skyDs.depthBiasSlopeScale = 0.0;
        skyDs.stencilReadMask = 0xFFFFFFFF;
        skyDs.stencilWriteMask = 0xFFFFFFFF;
        skyDesc.depthStencil = skyDs;

        skyDesc.topology = GPU::PrimitiveTopology::triangleList;
        skyDesc.cullMode = GPU::CullMode::none;
        skyDesc.frontFace = GPU::FrontFace::ccw;

        GPU::FragmentDescriptor skyFrag{};
        skyFrag.module = shaderModule;
        skyFrag.entryPoint = "skyboxFragment";
        GPU::ColorState skyCs{};
        skyCs.format = colorFormat;
        skyCs.writeMask = GPU::ColorWrite::all;
        skyFrag.targets.push_back(skyCs);
        skyDesc.fragment = skyFrag;

        pipelineSkybox = device->createRenderPipeline(skyDesc);
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

    // Slot 17 — MaterialUniforms per-instance (stride = 256 bytes).
    // Attributes at locations 20-23 follow the MVP (16-19) to avoid collision.
    {
        GPU::VertexLayout matLayout{};
        matLayout.arrayStride = kMaterialUniformStride;
        matLayout.stepMode    = GPU::StepMode::instance;

        GPU::VertexAttribute bcAttr{};
        bcAttr.componentType  = GPU::ComponentType::ctFloat;
        bcAttr.accessorType   = GPU::AccessorType::acVec4;
        bcAttr.offset         = 0;
        bcAttr.shaderLocation = 20; // baseColorFactor
        matLayout.attributes.push_back(bcAttr);

        GPU::VertexAttribute r0Attr{};
        r0Attr.componentType  = GPU::ComponentType::ctFloat;
        r0Attr.accessorType   = GPU::AccessorType::acVec4;
        r0Attr.offset         = 16;
        r0Attr.shaderLocation = 21; // uvTransformRow0
        matLayout.attributes.push_back(r0Attr);

        GPU::VertexAttribute r1Attr{};
        r1Attr.componentType  = GPU::ComponentType::ctFloat;
        r1Attr.accessorType   = GPU::AccessorType::acVec4;
        r1Attr.offset         = 32;
        r1Attr.shaderLocation = 22; // uvTransformRow1
        matLayout.attributes.push_back(r1Attr);

        GPU::VertexAttribute flagsAttr{};
        flagsAttr.componentType  = GPU::ComponentType::ctFloat;
        flagsAttr.accessorType   = GPU::AccessorType::acVec4;
        flagsAttr.offset         = 48;
        flagsAttr.shaderLocation = 23; // materialFlags [alphaMode, alphaCutoff, 0, 0]
        matLayout.attributes.push_back(flagsAttr);

        desc.vertex.buffers.push_back(matLayout);
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
    pipelineDebug    = pipelineFlat;  // TODO: compile debug SPIR-V variant
    
    // Double-sided variants (TODO: proper pipeline creation when Vulkan backend is fixed)
    pipelineFlatDoubleSided     = pipelineFlat;
    pipelineTexturedDoubleSided = pipelineTextured;
    
    // Alpha-blend variants (TODO: proper blend state when Vulkan backend supports it)
    pipelineFlatBlend             = pipelineFlat;
    pipelineTexturedBlend         = pipelineTextured;
    pipelineFlatBlendDoubleSided  = pipelineFlat;
    pipelineTexturedBlendDoubleSided = pipelineTextured;

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

    // Slot 17 — MaterialUniforms per-instance (stride = 256 bytes, TEXCOORD20–22).
    {
        GPU::VertexLayout matLayout{};
        matLayout.arrayStride = kMaterialUniformStride;
        matLayout.stepMode    = GPU::StepMode::instance;

        GPU::VertexAttribute bcAttr{};
        bcAttr.componentType  = GPU::ComponentType::ctFloat;
        bcAttr.accessorType   = GPU::AccessorType::acVec4;
        bcAttr.offset         = 0;
        bcAttr.shaderLocation = 20; // TEXCOORD20 — baseColorFactor
        matLayout.attributes.push_back(bcAttr);

        GPU::VertexAttribute r0Attr{};
        r0Attr.componentType  = GPU::ComponentType::ctFloat;
        r0Attr.accessorType   = GPU::AccessorType::acVec4;
        r0Attr.offset         = 16;
        r0Attr.shaderLocation = 21; // TEXCOORD21 — uvTransformRow0
        matLayout.attributes.push_back(r0Attr);

        GPU::VertexAttribute r1Attr{};
        r1Attr.componentType  = GPU::ComponentType::ctFloat;
        r1Attr.accessorType   = GPU::AccessorType::acVec4;
        r1Attr.offset         = 32;
        r1Attr.shaderLocation = 22; // TEXCOORD22 — uvTransformRow1
        matLayout.attributes.push_back(r1Attr);

        desc.vertex.buffers.push_back(matLayout);
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
    pipelineDebug    = pipelineFlat;  // TODO: compile debug DXIL variant
    
    // Double-sided variants (TODO: proper pipeline creation when DXIL shaders are ready)
    pipelineFlatDoubleSided     = pipelineFlat;
    pipelineTexturedDoubleSided = pipelineTextured;
    
    // Alpha-blend variants (TODO: proper blend state when DXIL shaders are ready)
    pipelineFlatBlend             = pipelineFlat;
    pipelineTexturedBlend         = pipelineTextured;
    pipelineFlatBlendDoubleSided  = pipelineFlat;
    pipelineTexturedBlendDoubleSided = pipelineTextured;

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

    // Fallback tangent buffer — all zeros, used for primitives without TANGENT.
    // Zero-length tangents trigger the shader's auto-generated fallback.
    if (!fallbackTangentBuffer) {
        constexpr uint64_t kFallbackTangentSize = 256 * 1024; // 16 384 float4 values
        std::vector<uint8_t> zeros(kFallbackTangentSize, 0);
        fallbackTangentBuffer = device->createBuffer(
            kFallbackTangentSize, GPU::BufferUsage::vertex, zeros.data());
    }
}

void Renderer::resize(uint32_t width, uint32_t height) {
    std::cout << "[Renderer::resize] " << renderWidth << "x" << renderHeight
              << " → " << width << "x" << height << std::endl;
    renderWidth  = width;
    renderHeight = height;

    if (width == 0 || height == 0) {
        depthTexture = nullptr;
        depthView    = nullptr;
        std::cout << "[Renderer::resize] zero size, cleared depth" << std::endl;
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
    // GLTF v0.4.0+ transposes matrices during loading, so node.matrix is
    // already in row-major vector_math format.
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

Renderer::Bounds Renderer::mergeBounds(const Bounds &a, const Bounds &b) {
    if (!a.valid) return b;
    if (!b.valid) return a;

    Bounds out;
    out.valid = true;
    out.min = systems::leal::vector_math::Vector3<double>(
        std::min(a.min.x(), b.min.x()),
        std::min(a.min.y(), b.min.y()),
        std::min(a.min.z(), b.min.z()));
    out.max = systems::leal::vector_math::Vector3<double>(
        std::max(a.max.x(), b.max.x()),
        std::max(a.max.y(), b.max.y()),
        std::max(a.max.z(), b.max.z()));
    return out;
}

Renderer::Bounds Renderer::transformBounds(
    const Bounds &bounds,
    const systems::leal::vector_math::Matrix4<double> &world) const
{
    if (!bounds.valid) return {};

    auto transformPoint = [&](double x, double y, double z) {
        return systems::leal::vector_math::Vector3<double>(
            world.data[0] * x + world.data[1] * y + world.data[2]  * z + world.data[3],
            world.data[4] * x + world.data[5] * y + world.data[6]  * z + world.data[7],
            world.data[8] * x + world.data[9] * y + world.data[10] * z + world.data[11]);
    };

    Bounds out;
    out.valid = true;
    bool first = true;
    for (int ix = 0; ix < 2; ++ix) {
        for (int iy = 0; iy < 2; ++iy) {
            for (int iz = 0; iz < 2; ++iz) {
                auto p = transformPoint(
                    ix ? bounds.max.x() : bounds.min.x(),
                    iy ? bounds.max.y() : bounds.min.y(),
                    iz ? bounds.max.z() : bounds.min.z());
                if (first) {
                    out.min = out.max = p;
                    first = false;
                } else {
                    out.min = systems::leal::vector_math::Vector3<double>(
                        std::min(out.min.x(), p.x()),
                        std::min(out.min.y(), p.y()),
                        std::min(out.min.z(), p.z()));
                    out.max = systems::leal::vector_math::Vector3<double>(
                        std::max(out.max.x(), p.x()),
                        std::max(out.max.y(), p.y()),
                        std::max(out.max.z(), p.z()));
                }
            }
        }
    }
    return out;
}

Renderer::Bounds Renderer::computePrimitiveBounds(
    const systems::leal::gltf::Primitive &primitive) const
{
    Bounds out;
    if (!asset || !asset->accessors || !asset->bufferViews || !asset->buffers) return out;

    auto posIt = primitive.attributes.find("POSITION");
    if (posIt == primitive.attributes.end()) return out;
    if ((size_t)posIt->second >= asset->accessors->size()) return out;

    auto &acc = (*asset->accessors)[posIt->second];
    if (acc.type != systems::leal::gltf::AccessorType::acVec3) return out;

    if (acc.min && acc.max && acc.min->size() >= 3 && acc.max->size() >= 3) {
        out.valid = true;
        out.min = systems::leal::vector_math::Vector3<double>((*acc.min)[0], (*acc.min)[1], (*acc.min)[2]);
        out.max = systems::leal::vector_math::Vector3<double>((*acc.max)[0], (*acc.max)[1], (*acc.max)[2]);
        return out;
    }

    if (acc.componentType != systems::leal::gltf::ComponentType::ctFloat) return out;

    if (acc.bufferView < 0 || (size_t)acc.bufferView >= asset->bufferViews->size()) return out;
    auto &bv = (*asset->bufferViews)[(size_t)acc.bufferView];
    if ((size_t)bv.buffer >= asset->buffers->size()) return out;
    auto &buffer = (*asset->buffers)[(size_t)bv.buffer];
    if (buffer.data.empty()) return out;

    size_t stride = bv.byteStride > 0 ? (size_t)bv.byteStride : 12;
    size_t start = (size_t)bv.byteOffset + (size_t)acc.byteOffset;
    if (start + 12 > buffer.data.size()) return out;

    bool first = true;
    for (size_t i = 0; i < acc.count; ++i) {
        size_t offset = start + i * stride;
        if (offset + 12 > buffer.data.size()) break;
        const float *v = reinterpret_cast<const float *>(buffer.data.data() + offset);
        systems::leal::vector_math::Vector3<double> p(v[0], v[1], v[2]);
        if (first) {
            out.min = out.max = p;
            out.valid = true;
            first = false;
        } else {
            out.min = systems::leal::vector_math::Vector3<double>(
                std::min(out.min.x(), p.x()),
                std::min(out.min.y(), p.y()),
                std::min(out.min.z(), p.z()));
            out.max = systems::leal::vector_math::Vector3<double>(
                std::max(out.max.x(), p.x()),
                std::max(out.max.y(), p.y()),
                std::max(out.max.z(), p.z()));
        }
    }

    return out;
}

int64_t Renderer::resolvePrimitiveMaterial(const systems::leal::gltf::Primitive &primitive) const {
    int64_t matIdx = primitive.material;
    if (activeVariant >= 0 && !primitive.khrMaterialsVariantsMappings.empty()) {
        for (auto &mapping : primitive.khrMaterialsVariantsMappings) {
            for (auto v : mapping.variants) {
                if ((int64_t)v == activeVariant) {
                    return mapping.material;
                }
            }
        }
    }
    return matIdx;
}

bool Renderer::isTransparentMaterial(int64_t materialIndex) const {
    if (materialIndex < 0 || !asset->materials || (size_t)materialIndex >= asset->materials->size()) {
        return false;
    }

    auto &mat = (*asset->materials)[(size_t)materialIndex];
    if (mat.alphaMode == systems::leal::gltf::AlphaMode::blend) return true;
    return mat.khrMaterialsTransmission && mat.khrMaterialsTransmission->transmissionFactor > 0.0f;
}

void Renderer::updateFrustumPlanes() {
    auto makePlane = [](double a, double b, double c, double d) {
        Plane plane;
        plane.normal = systems::leal::vector_math::Vector3<double>(a, b, c);
        double len = plane.normal.length();
        if (len > 1e-8) {
            plane.normal = plane.normal / len;
            plane.distance = d / len;
        } else {
            plane.distance = d;
        }
        return plane;
    };

    const auto &m = vpMatrix.data;
    // vpMatrix is stored row-major and uses column-vector convention (M * v).
    // Row i is at m[i*4 + 0..3].  Clip coordinates are dot(row_i, v).
    // OpenGL/Metal clip space: -w <= x,y,z <= w.
    frustumPlanes[0] = makePlane(m[0] + m[12],  m[1] + m[13],  m[2] + m[14],  m[3] + m[15]);   // left   (row0 + row3)
    frustumPlanes[1] = makePlane(m[12] - m[0],  m[13] - m[1],  m[14] - m[2],  m[15] - m[3]);   // right  (row3 - row0)
    frustumPlanes[2] = makePlane(m[4] + m[12],  m[5] + m[13],  m[6] + m[14],  m[7] + m[15]);   // bottom (row1 + row3)
    frustumPlanes[3] = makePlane(m[12] - m[4],  m[13] - m[5],  m[14] - m[6],  m[15] - m[7]);   // top    (row3 - row1)
    frustumPlanes[4] = makePlane(m[8] + m[12],  m[9] + m[13],  m[10] + m[14], m[11] + m[15]);  // near   (row2 + row3)
    frustumPlanes[5] = makePlane(m[12] - m[8],  m[13] - m[9],  m[14] - m[10], m[15] - m[11]);  // far    (row3 - row2)
    hasFrustumPlanes = true;
}

bool Renderer::isBoundsVisible(const Bounds &bounds) const {
    if (!bounds.valid || !hasFrustumPlanes) return true;

    for (const auto &plane : frustumPlanes) {
        // p-vertex: the AABB corner most inside along the plane normal
        double px = (plane.normal.x() >= 0.0) ? bounds.max.x() : bounds.min.x();
        double py = (plane.normal.y() >= 0.0) ? bounds.max.y() : bounds.min.y();
        double pz = (plane.normal.z() >= 0.0) ? bounds.max.z() : bounds.min.z();

        double dist = plane.normal.x() * px
                    + plane.normal.y() * py
                    + plane.normal.z() * pz
                    + plane.distance;
        if (dist < 0.0) {
            // Even the most-inside corner is behind this plane → cull
            return false;
        }
    }
    return true;
}

void Renderer::uploadVisibleNodeTransforms() {
    if (!transformBuffer || nodeTransforms.empty()) return;

    // Find the highest visible node index to reduce upload size.
    size_t maxVisibleIndex = 0;
    for (size_t i = visibleNodeMask.size(); i-- > 0;) {
        if (visibleNodeMask[i]) {
            maxVisibleIndex = i;
            break;
        }
    }

    size_t uploadFloats = (maxVisibleIndex + 1) * 32; // 32 floats per node
    if (uploadFloats > nodeTransforms.size()) uploadFloats = nodeTransforms.size();

    transformBuffer->upload(
        0,
        uploadFloats * sizeof(float),
        reinterpret_cast<uint8_t *>(nodeTransforms.data()));
}

void Renderer::updateVisibleInstances(uint64_t nodeIndex) {
    auto instIt = nodeInstanceData.find(nodeIndex);
    if (instIt == nodeInstanceData.end()) return;

    auto &instanceData = instIt->second;
    instanceData.visibleCount = 0;

    if (!instanceData.matrixBuffer || instanceData.cpuMatrices.empty()) return;

    size_t writeOffset = 0;
    for (const auto &instanceMatrix : instanceData.cpuMatrices) {
        // Quick point-in-frustum test using the instance translation.
        double px = instanceMatrix.data[3];   // column 3, row 0
        double py = instanceMatrix.data[7];   // column 3, row 1
        double pz = instanceMatrix.data[11];  // column 3, row 2

        bool visible = true;
        if (hasFrustumPlanes) {
            for (const auto &plane : frustumPlanes) {
                double dist = plane.normal.x() * px
                            + plane.normal.y() * py
                            + plane.normal.z() * pz
                            + plane.distance;
                if (dist < 0.0) { visible = false; break; }
            }
        }

        if (!visible) continue;

        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                instanceData.visibleMatrices[writeOffset + col * 4 + row] =
                    static_cast<float>(instanceMatrix.data[row * 4 + col]);
            }
        }
        writeOffset += 16;
        instanceData.visibleCount++;
    }

    if (instanceData.visibleCount == 0) return;

    instanceData.matrixBuffer->upload(
        0,
        static_cast<uint64_t>(instanceData.visibleCount) * 64,
        reinterpret_cast<uint8_t *>(instanceData.visibleMatrices.data()));
}

// ---------------------------------------------------------------------------
// Skeletal mesh skinning
// ---------------------------------------------------------------------------

void Renderer::computeSkinningTransforms() {
    if (skinData.empty() || totalJointMatrixBytes == 0) return;
    if (jointMatrixData.empty()) return;

    namespace VM = systems::leal::vector_math;

    for (size_t skinIdx = 0; skinIdx < skinData.size(); ++skinIdx) {
        auto &sd = skinData[skinIdx];
        if (sd.jointCount == 0) continue;

        // Find the world matrix of the skinned mesh node so we can cancel it out.
        // Per glTF spec: jointMatrix = inverse(skinNodeWorld) * jointWorld * inverseBindMatrix
        VM::Matrix4<double> skinNodeWorld = VM::Matrix4<double>::identity();
        for (uint64_t nodeIdx = 0; nodeIdx < nodeSkinIndex.size(); ++nodeIdx) {
            if (nodeSkinIndex[nodeIdx] == (int64_t)skinIdx && nodeIdx < nodeWorldMatrices.size()) {
                skinNodeWorld = nodeWorldMatrices[nodeIdx];
                break;
            }
        }
        VM::Matrix4<double> invSkinNodeWorld = skinNodeWorld.inverted();

        for (uint64_t j = 0; j < sd.jointCount; ++j) {
            uint64_t jointNodeIdx = sd.jointNodeIndices[j];
            VM::Matrix4<double> jointWorld = VM::Matrix4<double>::identity();
            if (jointNodeIdx < nodeWorldMatrices.size()) {
                jointWorld = nodeWorldMatrices[jointNodeIdx];
            }

            // Read inverse bind matrix for this joint (stored row-major in sd.inverseBindMatrices).
            VM::Matrix4<double> ibm;
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    ibm.data[row * 4 + col] = sd.inverseBindMatrices[j * 16 + row * 4 + col];
                }
            }

            // finalMatrix = inv(skinNodeWorld) * jointWorld * inverseBindMatrix
            VM::Matrix4<double> finalMatrix = invSkinNodeWorld * jointWorld * ibm;

            // Write to jointMatrixData in column-major for Metal (transposed from row-major).
            size_t baseIdx = (sd.gpuOffset / sizeof(float)) + j * 16;
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    jointMatrixData[baseIdx + col * 4 + row] = static_cast<float>(finalMatrix.data[row * 4 + col]);
                }
            }
        }
    }
}

void Renderer::uploadJointMatrices() {
    if (skinData.empty() || totalJointMatrixBytes == 0) return;
    if (jointMatrixData.empty()) return;
    auto &frame = frameResources[currentFrameIndex];
    if (!frame.jointMatrixBuffer) return;
    frame.jointMatrixBuffer->upload(
        0, totalJointMatrixBytes,
        reinterpret_cast<uint8_t *>(jointMatrixData.data()));
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

    // Apply animation if active — modifies node TRS before computing matrix.
    applyAnimatedTRS(nodeIndex);

    auto local = nodeLocalMatrix(node);
    auto  world = parentWorld * local;
    if (nodeIndex < nodeWorldMatrices.size()) {
        nodeWorldMatrices[nodeIndex] = world;
    }
    Bounds worldBounds;
    if (nodeIndex < nodeLocalBounds.size()) {
        worldBounds = transformBounds(nodeLocalBounds[nodeIndex], world);
    }

    // Buffer layout per node: 32 floats total, stored as two contiguous float4x4 matrices
    // [0..15]  = MVP matrix (clip space) - matrices[0] in shader
    // [16..31] = Model matrix (world space) - matrices[1] in shader
    size_t baseIdx = nodeIndex * 32;
    if (baseIdx + 31 < nodeTransforms.size()) {
        auto mvp = vpMatrix * world;
        
        // Transpose from row-major (vector_math) to column-major (Metal) format.
        // Metal's float4x4 interprets memory as 4 columns of 4 floats each.
        // We store MVP first, then Model - matching the shader's matrices[0] and matrices[1].
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                float mvpVal    = static_cast<float>(mvp.data[row * 4 + col]);
                float worldVal  = static_cast<float>(world.data[row * 4 + col]);
                // MVP at indices 0-15
                nodeTransforms[baseIdx + col * 4 + row] = mvpVal;
                // Model at indices 16-31
                nodeTransforms[baseIdx + 16 + col * 4 + row] = worldVal;
            }
        }
        
        // (matrix debug removed)
    }
    for (auto childIndex : node.children) {
        computeNodeTransform(childIndex, world);
        if (childIndex < nodeWorldBounds.size()) {
            worldBounds = mergeBounds(worldBounds, nodeWorldBounds[childIndex]);
        }
    }
    if (nodeIndex < nodeWorldBounds.size()) {
        nodeWorldBounds[nodeIndex] = worldBounds;
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
    if (!asset || (!pipelineFlat && !pipelineTextured && !pipelineDebug &&
                   !pipelineFlatDoubleSided && !pipelineTexturedDoubleSided &&
                   !pipelineFlatBlend && !pipelineTexturedBlend &&
                   !pipelineFlatBlendDoubleSided && !pipelineTexturedBlendDoubleSided)) return;
    if (!asset->scenes || sceneIndex >= asset->scenes->size()) return;
    if (!colorView) return;

    // ------------------------------------------------------------------
    // Frame-in-flight synchronization.
    // Wait until the GPU has finished with the frame slot we're about to overwrite.
    // ------------------------------------------------------------------
    auto &frame = frameResources[currentFrameIndex];
    if (frame.fence) {
        frame.fence->wait();
    }

    // Update aliases so existing code references the current frame's buffers.
    transformBuffer       = frame.transformBuffer;
    cameraPositionBuffer  = frame.cameraPositionBuffer;
    lightsUniformBuffer   = frame.lightsUniformBuffer;
    // jointMatrixBuffer is accessed directly via frameResources[currentFrameIndex] during render.

    auto encoder = device->createCommandEncoder();
    if (!encoder) return;

    namespace GPU = systems::leal::campello_gpu;
    namespace VM  = systems::leal::vector_math;
    using M4 = VM::Matrix4<double>;

    // ------------------------------------------------------------------
    // 1. Compute view-projection from camera override or GLTF camera.
    // ------------------------------------------------------------------
    M4 view;
    if (hasCameraOverride) {
        view = overrideView;
        vpMatrix = overrideProj * view;
    } else {
        double aspect = (renderHeight > 0)
            ? static_cast<double>(renderWidth) / renderHeight
            : 1.0;

        view = M4::lookAt(
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

    updateFrustumPlanes();

    // ------------------------------------------------------------------
    // 2. Upload camera position for specular lighting.
    //
    // The view matrix is [R | -R*eye] where R is rotation and eye is camera position.
    // We extract eye = -R^T * view[:3, 3]
    // ------------------------------------------------------------------
    {
        if (cameraPositionBuffer) {
            // Extract camera position from view matrix.
            // view matrix stores: [R | t] where t = -R * eye
            // So eye = -R^T * t
            // For row-major matrix, the translation is at indices 3, 7, 11
            float camPos[4] = {0, 0, 3, 0}; // default
            
            // Use the view matrix we computed above
            
            // Extract rotation part (3x3) and translation (3) from view matrix
            // Row-major: indices [0,1,2] = first row of R, [3] = t.x
            //            indices [4,5,6] = second row of R, [7] = t.y
            //            indices [8,9,10] = third row of R, [11] = t.z
            double R[3][3] = {
                {view.data[0], view.data[1], view.data[2]},
                {view.data[4], view.data[5], view.data[6]},
                {view.data[8], view.data[9], view.data[10]}
            };
            double t[3] = {view.data[3], view.data[7], view.data[11]};
            
            // eye = -R^T * t
            camPos[0] = -(float)(R[0][0] * t[0] + R[1][0] * t[1] + R[2][0] * t[2]);
            camPos[1] = -(float)(R[0][1] * t[0] + R[1][1] * t[1] + R[2][1] * t[2]);
            camPos[2] = -(float)(R[0][2] * t[0] + R[1][2] * t[1] + R[2][2] * t[2]);

            cameraWorldPos[0] = camPos[0];
            cameraWorldPos[1] = camPos[1];
            cameraWorldPos[2] = camPos[2];

            cameraPositionBuffer->upload(0, 16, camPos);
        }
    }

    // ------------------------------------------------------------------
    // 3a. Upload skybox uniforms (invVP + screenSize + cameraPos).
    // ------------------------------------------------------------------
    if (pipelineSkybox && environmentMap) {
        if (!skyboxUniformBuffer[currentFrameIndex]) {
            skyboxUniformBuffer[currentFrameIndex] = device->createBuffer(96, GPU::BufferUsage::uniform);
        }
        if (skyboxUniformBuffer[currentFrameIndex]) {
            auto invVP = vpMatrix.inverted();
            float skyboxData[24] = {0}; // 96 bytes
            // invVP in column-major for Metal (transposed from row-major)
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    skyboxData[col * 4 + row] = (float)invVP.data[row * 4 + col];
                }
            }
            // screenSize at offset 64 (float2)
            skyboxData[16] = (float)renderWidth;
            skyboxData[17] = (float)renderHeight;
            // cameraPos at offset 80 (float3, 16-byte aligned)
            skyboxData[20] = cameraWorldPos[0];
            skyboxData[21] = cameraWorldPos[1];
            skyboxData[22] = cameraWorldPos[2];
            skyboxUniformBuffer[currentFrameIndex]->upload(
                0, 96, reinterpret_cast<uint8_t*>(skyboxData));
        }
    }

    // ------------------------------------------------------------------
    // 3. Upload KHR_lights_punctual lights to uniform buffer (binding 10).
    //
    // Uniform buffer layout (256 bytes):
    //   [0]:      uint32_t lightCount
    //   [4-15]:   padding (12 bytes)
    //   [16-79]:  Light 0 (64 bytes: position, color, direction, spot angles)
    //   [80-143]: Light 1
    //   [144-207]: Light 2
    //   [208-271]: Light 3
    // ------------------------------------------------------------------
    {
        if (lightsUniformBuffer) {
            struct LightData {
                float position[4];    // xyz + type (0=dir, 1=point, 2=spot)
                float color[4];       // rgb + intensity
                float direction[4];   // xyz + range
                float spotAngles[4];  // inner/outer cone angles + padding
            };
            
            struct LightsUniform {
                uint32_t count;
                float padding[3];
                LightData lights[4];
            };
            
            LightsUniform lightsData = {};
            lightsData.count = 0;
            
            if (punctualLightsEnabled && asset && asset->khrLightsPunctual && !asset->khrLightsPunctual->empty()) {
                // Count and collect lights from scene nodes
                int lightCount = 0;
                
                auto countLights = [&](auto&& self, uint64_t nodeIndex, 
                                       const VM::Matrix4<double>& parentWorld) -> void {
                    if (!asset->nodes || nodeIndex >= asset->nodes->size()) return;
                    auto &node = (*asset->nodes)[nodeIndex];
                    VM::Matrix4<double> world = parentWorld * nodeLocalMatrix(node);
                    
                    if (node.light >= 0 && node.light < (int64_t)asset->khrLightsPunctual->size()) {
                        if (lightCount < 4) {
                            auto &light = (*asset->khrLightsPunctual)[(size_t)node.light];
                            LightData &ld = lightsData.lights[lightCount];
                            
                            // Type: 0=directional, 1=point, 2=spot
                            float typeVal = 0.0f;
                            if (light.type == systems::leal::gltf::KHRLightPunctualType::point) typeVal = 1.0f;
                            else if (light.type == systems::leal::gltf::KHRLightPunctualType::spot) typeVal = 2.0f;
                            
                            // Position (point/spot) or direction (directional)
                            if (light.type == systems::leal::gltf::KHRLightPunctualType::directional) {
                                // Z-axis (column 2) in row-major VM::Matrix4: data[row*4+2]
                                double dirX = world.data[2];
                                double dirY = world.data[6];
                                double dirZ = world.data[10];
                                double len = std::sqrt(dirX*dirX + dirY*dirY + dirZ*dirZ);
                                if (len > 0.0001) {
                                    ld.position[0] = (float)(-dirX / len);
                                    ld.position[1] = (float)(-dirY / len);
                                    ld.position[2] = (float)(-dirZ / len);
                                } else {
                                    ld.position[0] = 0.0f; ld.position[1] = 0.0f; ld.position[2] = -1.0f;
                                }
                            } else {
                                ld.position[0] = (float)world.data[3];
                                ld.position[1] = (float)world.data[7];
                                ld.position[2] = (float)world.data[11];
                            }
                            ld.position[3] = typeVal;
                            
                            // Color and intensity
                            ld.color[0] = (float)light.color.x();
                            ld.color[1] = (float)light.color.y();
                            ld.color[2] = (float)light.color.z();
                            ld.color[3] = (float)light.intensity;
                            
                            // Spot direction and range
                            if (light.type == systems::leal::gltf::KHRLightPunctualType::spot) {
                                // Z-axis (column 2) in row-major VM::Matrix4: data[row*4+2]
                                double dirX = world.data[2];
                                double dirY = world.data[6];
                                double dirZ = world.data[10];
                                double len = std::sqrt(dirX*dirX + dirY*dirY + dirZ*dirZ);
                                if (len > 0.0001) {
                                    ld.direction[0] = (float)(-dirX / len);
                                    ld.direction[1] = (float)(-dirY / len);
                                    ld.direction[2] = (float)(-dirZ / len);
                                } else {
                                    ld.direction[0] = 0.0f; ld.direction[1] = 0.0f; ld.direction[2] = -1.0f;
                                }
                            } else {
                                ld.direction[0] = 0.0f;
                                ld.direction[1] = 0.0f;
                                ld.direction[2] = 0.0f;
                            }
                            ld.direction[3] = (float)light.range;
                            
                            // Spot cone angles
                            if (light.type == systems::leal::gltf::KHRLightPunctualType::spot) {
                                ld.spotAngles[0] = (float)light.innerConeAngle;
                                ld.spotAngles[1] = (float)light.outerConeAngle;
                            }
                            
                            lightCount++;
                        }
                    }
                    
                    for (auto childIdx : node.children) {
                        self(self, childIdx, world);
                    }
                };
                
                auto &scene = (*asset->scenes)[sceneIndex];
                if (scene.nodes) {
                    for (auto rootIdx : *scene.nodes) {
                        countLights(countLights, rootIdx, M4::identity());
                    }
                }
                lightsData.count = (uint32_t)lightCount;
            }

            // If the asset has no lights, optionally synthesize a default directional light.
            if (lightsData.count == 0 && defaultLightEnabled) {
                lightsData.count = 1;
                LightData &ld = lightsData.lights[0];
                // Direction: normalize(0.5, 1.0, 0.5) — front-top-right
                constexpr float dx = 0.5f, dy = 1.0f, dz = 0.5f;
                constexpr float len = 1.2247448f; // sqrt(0.25 + 1.0 + 0.25)
                ld.position[0] = dx / len;
                ld.position[1] = dy / len;
                ld.position[2] = dz / len;
                ld.position[3] = 0.0f;  // type = directional
                ld.color[0] = 1.0f; ld.color[1] = 1.0f; ld.color[2] = 1.0f;
                ld.color[3] = 5.0f;  // intensity = 5 lux (compensates for Reinhard tone mapping)
                ld.direction[0] = ld.direction[1] = ld.direction[2] = 0.0f;
                ld.direction[3] = 0.0f;  // range (unused for directional)
                ld.spotAngles[0] = ld.spotAngles[1] = ld.spotAngles[2] = ld.spotAngles[3] = 0.0f;
            }

            lightsUniformBuffer->upload(0, sizeof(LightsUniform), &lightsData);
            // (lights debug removed)
        }
    }

    // ------------------------------------------------------------------
    // 4. Upload material uniforms (PBR params + UV transform + alpha + emissive + occlusion per slot).
    //
    // Each slot is kMaterialUniformStride (256) bytes. Used layout (140 bytes):
    //   [0..15]   float4 baseColorFactor
    //   [16..31]  float4 uvTransformRow0  (KHR_texture_transform row 0: [a, b, tx, hasTransform])
    //   [32..47]  float4 uvTransformRow1  (KHR_texture_transform row 1: [c, d, ty, 0])
    //   [48..51]  float  metallicFactor
    //   [52..55]  float  roughnessFactor
    //   [56..59]  float  normalScale
    //   [60..63]  float  alphaMode        (0=opaque, 1=mask, 2=blend)
    //   [64..67]  float  alphaCutoff
    //   [68..71]  float  unlit            (0=lit, 1=unlit)
    //   [72..75]  float  hasNormalTexture (0=no, 1=yes)
    //   [76..79]  float  hasEmissiveTexture (0=no, 1=yes)
    //   [80..83]  float  hasOcclusionTexture (0=no, 1=yes)
    //   [84..87]  float  occlusionStrength
    //   [88..95]  float2 pad (Metal float3 alignment: emissiveFactor lands at offset 96)
    //   [96..107] float3 emissiveFactor   (16-byte aligned in Metal)
    //   [108..111] float ior              (KHR_materials_ior, default 1.5)
    //   [112..115] float specularFactor   (KHR_materials_specular, default 1.0)
    //   [116..119] float hasSpecularTexture
    //   [120..123] float hasSpecularColorTexture
    //   [124..127] float pad2             (Metal float3 alignment: specularColorFactor at offset 128)
    //   [128..139] float3 specularColorFactor (KHR_materials_specular F0 tint, default [1,1,1])
    //
    // Slot 0 is the default (white, identity UV transform, metallic=1, roughness=1, no textures).
    // Slots 1..N correspond to asset->materials indices 0..N-1.
    // ------------------------------------------------------------------
    if (materialUniformBuffer) {
        // Helper: build a material slot from material data.
        // Matches Metal struct layout EXACTLY.
        // Metal packs floats at 4-byte intervals, but float3 is 16-byte aligned.
        // Correct offsets (verified from shader struct):
        //   transmissionFactor = offset 228 (index 57)
        //   hasTransmissionTexture = offset 232 (index 58)
        //   viewMode = offset 236 (index 59)
        auto buildSlot = [](float bc[4], float r0[4], float r1[4],
                            float metallic, float roughness, float normalScale,
                            float alphaMode, float alphaCutoff, float unlit,
                            float hasNormal, float hasEmissive, float hasOcclusion,
                            float occlusionStrength, float emissiveFactor[3],
                            float ior,
                            float specularFactor, float hasSpecularTex, float hasSpecularColorTex,
                            float specularColorFactor[3],
                            float sheenColorFactor[3], float sheenRoughnessFactor,
                            float hasSheenColorTex, float hasSheenRoughnessTex,
                            float clearcoatFactor, float clearcoatRoughnessFactor,
                            float hasClearcoatTex, float hasClearcoatRoughnessTex,
                            float hasClearcoatNormalTex, float clearcoatNormalScale,
                            float transmissionFactor, float hasTransmissionTex,
                            float viewModeValue,
                            float environmentIntensityValue, float iblEnabledValue,
                            float out[64]) {
            // Zero initialize
            for (int i = 0; i < 64; i++) out[i] = 0.f;
            
            // [0-2] float4s at offsets 0, 16, 32
            out[0]  = bc[0]; out[1]  = bc[1]; out[2]  = bc[2];  out[3]  = bc[3];
            out[4]  = r0[0]; out[5]  = r0[1]; out[6]  = r0[2];  out[7]  = r0[3];
            out[8]  = r1[0]; out[9]  = r1[1]; out[10] = r1[2];  out[11] = r1[3];
            
            // [12-22] floats at offsets 48-88
            out[12] = metallic;           // 48
            out[13] = roughness;          // 52
            out[14] = normalScale;        // 56
            out[15] = alphaMode;          // 60
            out[16] = alphaCutoff;        // 64
            out[17] = unlit;              // 68
            out[18] = hasNormal;          // 72
            out[19] = hasEmissive;        // 76
            out[20] = hasOcclusion;       // 80
            out[21] = occlusionStrength;  // 84
            out[22] = 0.f;                // 88 - _padding
            
            // emissiveFactor float3 at offset 96 (16-byte aligned)
            out[24] = emissiveFactor[0];  // 96
            out[25] = emissiveFactor[1];  // 100
            out[26] = emissiveFactor[2];  // 104
            out[27] = ior;                // 108
            
            // [28-31] at offsets 112-124
            out[28] = specularFactor;         // 112
            out[29] = hasSpecularTex;         // 116
            out[30] = hasSpecularColorTex;    // 120
            out[31] = 0.f;                    // 124 - _pad2
            
            // specularColorFactor float3 at offset 128? No wait, 124+4=128 which IS 16-aligned
            // But struct says _pad2 at 124, then specularColorFactor... let me check
            // After _pad2 at 124, next is 128. 128 is 16-aligned. Good.
            out[32] = specularColorFactor[0]; // 128
            out[33] = specularColorFactor[1]; // 132
            out[34] = specularColorFactor[2]; // 136
            // _pad3 at 140? No wait, 136+4=140, not 16-aligned
            // Next float3 needs 16-align, so next is 144
            // So _pad3 at 140, then sheenColorFactor at 144
            out[35] = 0.f;                    // 140 - _pad3
            
            // sheenColorFactor float3 at offset 144
            out[36] = sheenColorFactor[0];    // 144
            out[37] = sheenColorFactor[1];    // 148
            out[38] = sheenColorFactor[2];    // 152
            out[39] = sheenRoughnessFactor;   // 156
            
            // [40-43] at offsets 160-172
            out[40] = hasSheenColorTex;       // 160
            out[41] = hasSheenRoughnessTex;   // 164
            out[42] = clearcoatFactor;        // 168
            out[43] = clearcoatRoughnessFactor; // 172
            
            // [44-47] at offsets 176-188
            out[44] = hasClearcoatTex;        // 176
            out[45] = hasClearcoatRoughnessTex; // 180
            out[46] = hasClearcoatNormalTex;  // 184
            out[47] = clearcoatNormalScale;   // 188
            
            // transmissionFactor at offset 228 (index 57)
            // hasTransmissionTexture at offset 232 (index 58)
            out[57] = transmissionFactor;     // 228
            out[58] = hasTransmissionTex;     // 232
            out[59] = viewModeValue;          // 236
            out[60] = environmentIntensityValue; // 240
            out[61] = iblEnabledValue;           // 244
        };

        // Default slot — white, identity UV transform, metallic=1, roughness=1, no textures, no sheen, no clearcoat, no transmission.
        {
            float bc[4]            = {1.f, 1.f, 1.f, 1.f};
            float row0[4]          = {1.f, 0.f, 0.f, 0.f}; // w=0 → identity fast path
            float row1[4]          = {0.f, 1.f, 0.f, 0.f};
            float emissive[3]      = {0.f, 0.f, 0.f};
            float specularColor[3] = {1.f, 1.f, 1.f};
            float sheenColor[3]    = {0.f, 0.f, 0.f};
            float slot[64];
            buildSlot(bc, row0, row1, 1.f, 1.f, 1.f, 0.f, 0.5f, 0.f, 0.f, 0.f, 0.f, 1.f,
                      emissive, 1.5f, 1.f, 0.f, 0.f, specularColor,
                      sheenColor, 0.f, 0.f, 0.f,
                      0.f, 0.f, 0.f, 0.f, 0.f, 1.f,
                      0.f, 0.f, (float)viewMode,
                      environmentIntensity, iblEnabled ? 1.f : 0.f, slot);
            materialUniformBuffer->upload(0, (uint64_t)(64 * sizeof(float)), slot);
        }

        if (asset->materials) {
            for (size_t i = 0; i < asset->materials->size(); ++i) {
                auto &mat = (*asset->materials)[i];

                // Base color factor.
                float bc[4] = {1.f, 1.f, 1.f, 1.f};
                float metallic = 1.f;
                float roughness = 1.f;
                if (mat.pbrMetallicRoughness) {
                    bc[0] = (float)mat.pbrMetallicRoughness->baseColorFactor.x();
                    bc[1] = (float)mat.pbrMetallicRoughness->baseColorFactor.y();
                    bc[2] = (float)mat.pbrMetallicRoughness->baseColorFactor.z();
                    bc[3] = (float)mat.pbrMetallicRoughness->baseColorFactor.w();
                    metallic = (float)mat.pbrMetallicRoughness->metallicFactor;
                    roughness = (float)mat.pbrMetallicRoughness->roughnessFactor;
                }

                // KHR_texture_transform for baseColorTexture.
                float row0[4] = {1.f, 0.f, 0.f, 0.f}; // w=0 → identity fast path
                float row1[4] = {0.f, 1.f, 0.f, 0.f};
                if (mat.pbrMetallicRoughness && mat.pbrMetallicRoughness->baseColorTexture) {
                    auto &xfPtr = mat.pbrMetallicRoughness->baseColorTexture->khrTextureTransform;
                    if (xfPtr) {
                        float c  = (float)std::cos(xfPtr->rotation);
                        float s  = (float)std::sin(xfPtr->rotation);
                        float sx = (float)xfPtr->scale.x();
                        float sy = (float)xfPtr->scale.y();
                        float ox = (float)xfPtr->offset.x();
                        float oy = (float)xfPtr->offset.y();
                        row0[0] = sx * c;  row0[1] = -sy * s;  row0[2] = ox;  row0[3] = 1.f;
                        row1[0] = sx * s;  row1[1] =  sy * c;  row1[2] = oy;
                    }
                }

                // Normal scale from normalTexture info.
                float normalScale = 1.f;
                float hasNormal = 0.f;
                if (mat.normalTexture) {
                    normalScale = (float)mat.normalTexture->scale;
                    hasNormal = 1.f;
                }

                // Emissive factor and texture.
                float emissiveFactor[3] = {0.f, 0.f, 0.f};
                float hasEmissive = 0.f;
                if (mat.emissiveTexture) {
                    hasEmissive = 1.f;
                }
                emissiveFactor[0] = (float)mat.emissiveFactor.x();
                emissiveFactor[1] = (float)mat.emissiveFactor.y();
                emissiveFactor[2] = (float)mat.emissiveFactor.z();

                // Occlusion texture and strength.
                float hasOcclusion = 0.f;
                float occlusionStrength = 1.f;
                if (mat.occlusionTexture) {
                    hasOcclusion = 1.f;
                    occlusionStrength = (float)mat.occlusionTexture->strength;
                }

                // Alpha mode and flags.
                float alphaMode   = (float)mat.alphaMode;
                float alphaCutoff = (float)mat.alphaCutoff;
                float unlit       = mat.khrMaterialsUnlit ? 1.0f : 0.0f;

                // KHR_materials_ior: index of refraction (default 1.5 → F0 = 0.04).
                float ior = (float)mat.khrMaterialsIor;

                // KHR_materials_specular: specular layer weight and F0 color tint.
                float specularFactor       = 1.f;
                float hasSpecularTex       = 0.f;
                float hasSpecularColorTex  = 0.f;
                float specularColorFactor[3] = {1.f, 1.f, 1.f};
                if (mat.khrMaterialsSpecular) {
                    auto &spec = *mat.khrMaterialsSpecular;
                    specularFactor          = (float)spec.specularFactor;
                    specularColorFactor[0]  = (float)spec.specularColorFactor.x();
                    specularColorFactor[1]  = (float)spec.specularColorFactor.y();
                    specularColorFactor[2]  = (float)spec.specularColorFactor.z();
                    if (spec.specularTexture)      hasSpecularTex      = 1.f;
                    if (spec.specularColorTexture)  hasSpecularColorTex = 1.f;
                }

                // KHR_materials_sheen: sheen color and roughness.
                float sheenColorFactor[3]  = {0.f, 0.f, 0.f};
                float sheenRoughnessFactor = 0.f;
                float hasSheenColorTex     = 0.f;
                float hasSheenRoughnessTex = 0.f;
                if (mat.khrMaterialsSheen) {
                    auto &sheen = *mat.khrMaterialsSheen;
                    sheenColorFactor[0]  = (float)sheen.sheenColorFactor.x();
                    sheenColorFactor[1]  = (float)sheen.sheenColorFactor.y();
                    sheenColorFactor[2]  = (float)sheen.sheenColorFactor.z();
                    sheenRoughnessFactor = (float)sheen.sheenRoughnessFactor;
                    if (sheen.sheenColorTexture)     hasSheenColorTex     = 1.f;
                    if (sheen.sheenRoughnessTexture) hasSheenRoughnessTex = 1.f;
                }

                // KHR_materials_clearcoat: layer intensity, roughness, and optional textures.
                float clearcoatFactor          = 0.f;
                float clearcoatRoughnessFactor = 0.f;
                float hasClearcoatTex          = 0.f;
                float hasClearcoatRoughnessTex = 0.f;
                float hasClearcoatNormalTex    = 0.f;
                float clearcoatNormalScale     = 1.f;
                if (mat.khrMaterialsClearcoat) {
                    auto &cc = *mat.khrMaterialsClearcoat;
                    clearcoatFactor          = (float)cc.clearcoatFactor;
                    clearcoatRoughnessFactor = (float)cc.clearcoatRoughnessFactor;
                    if (cc.clearcoatTexture)          hasClearcoatTex          = 1.f;
                    if (cc.clearcoatRoughnessTexture) hasClearcoatRoughnessTex = 1.f;
                    if (cc.clearcoatNormalTexture) {
                        hasClearcoatNormalTex = 1.f;
                        clearcoatNormalScale  = (float)cc.clearcoatNormalTexture->scale;
                    }
                }

                // KHR_materials_transmission
                float transmissionFactor = 0.f;
                float hasTransmissionTex = 0.f;
                if (mat.khrMaterialsTransmission) {
                    transmissionFactor = (float)mat.khrMaterialsTransmission->transmissionFactor;
                    if (mat.khrMaterialsTransmission->transmissionTexture) {
                        hasTransmissionTex = 1.f;
                    }
                }

                float slot[64];
                buildSlot(bc, row0, row1, metallic, roughness, normalScale,
                          alphaMode, alphaCutoff, unlit, hasNormal, hasEmissive, hasOcclusion,
                          occlusionStrength, emissiveFactor, ior,
                          specularFactor, hasSpecularTex, hasSpecularColorTex, specularColorFactor,
                          sheenColorFactor, sheenRoughnessFactor, hasSheenColorTex, hasSheenRoughnessTex,
                          clearcoatFactor, clearcoatRoughnessFactor,
                          hasClearcoatTex, hasClearcoatRoughnessTex, hasClearcoatNormalTex, clearcoatNormalScale,
                          transmissionFactor, hasTransmissionTex, (float)viewMode,
                          environmentIntensity, iblEnabled ? 1.f : 0.f, slot);
                
                materialUniformBuffer->upload((uint64_t)(i + 1) * kMaterialUniformStride,
                                              (uint64_t)(64 * sizeof(float)), slot);
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
    opaqueQueue.clear();
    transparentQueue.clear();
    std::fill(visibleNodeMask.begin(), visibleNodeMask.end(), 0);
    if (scene.nodes) {
        for (auto nodeIndex : *scene.nodes) {
            gatherVisibleDraws(nodeIndex);
        }
    }

    // Compute joint matrices for skeletal meshes after all node world matrices are ready.
    computeSkinningTransforms();
    uploadJointMatrices();

    uploadVisibleNodeTransforms();

    // ------------------------------------------------------------------
    // 4. Record render pass and draw calls.
    // ------------------------------------------------------------------
    GPU::ColorAttachment ca{};
    ca.view          = colorView;
    ca.clearValue[0] = clearColor[0];
    ca.clearValue[1] = clearColor[1];
    ca.clearValue[2] = clearColor[2];
    ca.clearValue[3] = clearColor[3];
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
        ds.stencilReadOnly   = false;
        rpDesc.depthStencilAttachment = ds;
    }

    auto rpe = encoder->beginRenderPass(rpDesc);
    if (!rpe) return;

    currentPipelineVariant = 0; // reset — renderPrimitive() will set it per draw
    lastBoundVertexBuffers.fill({}); // reset vertex buffer binding state

    if (renderWidth > 0 && renderHeight > 0) {
        rpe->setViewport(0.0f, 0.0f, (float)renderWidth, (float)renderHeight, 0.0f, 1.0f);
        rpe->setScissorRect(0.0f, 0.0f, (float)renderWidth, (float)renderHeight);
    }
    // (render debug removed)

    // ------------------------------------------------------------------
    // Render skybox first (before opaque geometry).
    // ------------------------------------------------------------------
    if (skyboxEnabled && pipelineSkybox && environmentMap && skyboxUniformBuffer[currentFrameIndex]) {
        // Create/update skybox bind group for the current frame.
        if (!skyboxBindGroup[currentFrameIndex] && skyboxBindGroupLayout) {
            GPU::BindGroupDescriptor sbDesc{};
            sbDesc.layout = skyboxBindGroupLayout;
            sbDesc.entries = {
                {0, environmentMap},
                {1, environmentSampler},
                {2, GPU::BufferBinding{skyboxUniformBuffer[currentFrameIndex], 0, 96}},
            };
            skyboxBindGroup[currentFrameIndex] = device->createBindGroup(sbDesc);
        }
        if (skyboxBindGroup[currentFrameIndex]) {
            rpe->setPipeline(pipelineSkybox);
            rpe->setBindGroup(0, skyboxBindGroup[currentFrameIndex]);
            // Reset pipeline tracking so renderPrimitive() will rebind its pipeline.
            currentPipelineVariant = 0;
            lastBoundVertexBuffers.fill({});
            rpe->draw(3);
        }
    }

    for (auto &draw : opaqueQueue) {
        renderPrimitive(rpe, *draw.primitive, draw.nodeIndex);
    }

    // Sort transparent draws back-to-front and draw them after all opaque geometry.
    // nodeTransforms stores the model matrix column-major at [nodeIndex*32 + 16..31];
    // the world translation is at column 3: offsets 28 (X), 29 (Y), 30 (Z).
    if (!transparentQueue.empty()) {
        std::sort(transparentQueue.begin(), transparentQueue.end(),
            [&](const DrawItem &a, const DrawItem &b) {
                auto squaredDist = [&](uint64_t ni) -> float {
                    size_t base = ni * 32;
                    if (base + 31 >= nodeTransforms.size()) return 0.0f;
                    float dx = nodeTransforms[base + 28] - cameraWorldPos[0];
                    float dy = nodeTransforms[base + 29] - cameraWorldPos[1];
                    float dz = nodeTransforms[base + 30] - cameraWorldPos[2];
                    return dx*dx + dy*dy + dz*dz;
                };
                return squaredDist(a.nodeIndex) > squaredDist(b.nodeIndex); // farther first
            });
        for (auto &draw : transparentQueue) {
            renderPrimitive(rpe, *draw.primitive, draw.nodeIndex);
        }
    }

    rpe->end();
    device->submit(encoder->finish(), frame.fence);
    currentFrameIndex = (currentFrameIndex + 1) % kMaxFramesInFlight;
}

void Renderer::update(double dt) {
    if (!asset || !asset->animations) return;
    if (animationStates.empty()) return;

    // Clear previous animated values — will be rebuilt from all playing animations.
    animatedNodes.clear();

    // Process each animation that has a state.
    for (auto &pair : animationStates) {
        int32_t animIndex = pair.first;
        AnimationState &state = pair.second;

        if (animIndex < 0 || (size_t)animIndex >= asset->animations->size()) continue;
        if (!state.playing) continue;

        // Advance time.
        state.time += dt;

        // Handle looping or clamping.
        if (state.time > state.duration) {
            if (state.loop) {
                state.time = fmod(state.time, state.duration);
            } else {
                state.time = state.duration;
                state.playing = false;
            }
        }

        // Sample this animation at its current time.
        sampleAnimation(animIndex, (float)state.time);
    }
}

// ---------------------------------------------------------------------------
// Animation sampling
// ---------------------------------------------------------------------------

void Renderer::sampleAnimation(int32_t animIndex, float time) {
    if (!asset || !asset->animations) return;
    if (animIndex < 0 || (size_t)animIndex >= asset->animations->size()) return;

    auto &animation = (*asset->animations)[(size_t)animIndex];

    // Process each channel.
    for (auto &channel : animation.channels) {
        if (channel.target.node < 0) continue;
        if (!asset->nodes || (size_t)channel.target.node >= asset->nodes->size()) continue;
        if (channel.sampler >= animation.samplers.size()) continue;

        auto &sampler = animation.samplers[channel.sampler];
        if (!asset->accessors) continue;
        if (sampler.input >= asset->accessors->size()) continue;
        if (sampler.output >= asset->accessors->size()) continue;

        auto &inputAcc = (*asset->accessors)[sampler.input];
        auto &outputAcc = (*asset->accessors)[sampler.output];
        if (inputAcc.bufferView < 0 || outputAcc.bufferView < 0) continue;
        if (!asset->bufferViews) continue;
        if ((size_t)inputAcc.bufferView >= asset->bufferViews->size()) continue;
        if ((size_t)outputAcc.bufferView >= asset->bufferViews->size()) continue;

        auto &inputBV = (*asset->bufferViews)[(size_t)inputAcc.bufferView];
        auto &outputBV = (*asset->bufferViews)[(size_t)outputAcc.bufferView];
        if (!asset->buffers) continue;
        if ((size_t)inputBV.buffer >= asset->buffers->size()) continue;
        if ((size_t)outputBV.buffer >= asset->buffers->size()) continue;

        auto &inputBuf = (*asset->buffers)[(size_t)inputBV.buffer];
        auto &outputBuf = (*asset->buffers)[(size_t)outputBV.buffer];

        // Get keyframe times (input).
        const float *times = reinterpret_cast<const float*>(
            inputBuf.data.data() + inputBV.byteOffset + inputAcc.byteOffset);
        uint32_t keyframeCount = (uint32_t)inputAcc.count;
        if (keyframeCount == 0) continue;

        // Get keyframe values (output).
        const uint8_t *values = outputBuf.data.data() + outputBV.byteOffset + outputAcc.byteOffset;

        // Find keyframe interval.
        uint32_t kf0 = 0, kf1 = 0;
        float t = 0.0f;

        if (time <= times[0]) {
            kf0 = kf1 = 0;
            t = 0.0f;
        } else if (time >= times[keyframeCount - 1]) {
            kf0 = kf1 = keyframeCount - 1;
            t = 0.0f;
        } else {
            for (uint32_t i = 0; i < keyframeCount - 1; ++i) {
                if (time >= times[i] && time < times[i + 1]) {
                    kf0 = i;
                    kf1 = i + 1;
                    float span = times[kf1] - times[kf0];
                    t = (span > 0.0f) ? (time - times[kf0]) / span : 0.0f;
                    break;
                }
            }
        }

        // Apply interpolation based on type and path.
        namespace VM = systems::leal::vector_math;
        uint64_t nodeIdx = (uint64_t)channel.target.node;

        if (channel.target.path == "translation") {
            auto &trs = animatedNodes[nodeIdx];
            trs.hasTranslation = true;

            const float *v0 = reinterpret_cast<const float*>(values) + kf0 * 3;
            const float *v1 = reinterpret_cast<const float*>(values) + kf1 * 3;

            if (sampler.interpolation == systems::leal::gltf::AnimationInterpolation::aiStep) {
                trs.translation = VM::Vector3<double>(v0[0], v0[1], v0[2]);
            } else {
                // Linear interpolation.
                double x = v0[0] + (v1[0] - v0[0]) * t;
                double y = v0[1] + (v1[1] - v0[1]) * t;
                double z = v0[2] + (v1[2] - v0[2]) * t;
                trs.translation = VM::Vector3<double>(x, y, z);
            }
        } else if (channel.target.path == "rotation") {
            auto &trs = animatedNodes[nodeIdx];
            trs.hasRotation = true;

            const float *v0 = reinterpret_cast<const float*>(values) + kf0 * 4;
            const float *v1 = reinterpret_cast<const float*>(values) + kf1 * 4;

            VM::Quaternion<double> q0(v0[0], v0[1], v0[2], v0[3]);
            VM::Quaternion<double> q1(v1[0], v1[1], v1[2], v1[3]);

            if (sampler.interpolation == systems::leal::gltf::AnimationInterpolation::aiStep) {
                trs.rotation = q0;
            } else {
                // Spherical linear interpolation for quaternions.
                trs.rotation = VM::Quaternion<double>::slerp(q0, q1, (double)t);
            }
        } else if (channel.target.path == "scale") {
            auto &trs = animatedNodes[nodeIdx];
            trs.hasScale = true;

            const float *v0 = reinterpret_cast<const float*>(values) + kf0 * 3;
            const float *v1 = reinterpret_cast<const float*>(values) + kf1 * 3;

            if (sampler.interpolation == systems::leal::gltf::AnimationInterpolation::aiStep) {
                trs.scale = VM::Vector3<double>(v0[0], v0[1], v0[2]);
            } else {
                // Linear interpolation.
                double x = v0[0] + (v1[0] - v0[0]) * t;
                double y = v0[1] + (v1[1] - v0[1]) * t;
                double z = v0[2] + (v1[2] - v0[2]) * t;
                trs.scale = VM::Vector3<double>(x, y, z);
            }
        }
        // TODO: "weights" for morph targets (not implemented).
    }
}

// Apply animated TRS values to node transforms before computing world matrices.
void Renderer::applyAnimatedTRS(uint64_t nodeIndex) {
    if (!asset || !asset->nodes || nodeIndex >= asset->nodes->size()) return;

    auto it = animatedNodes.find(nodeIndex);
    if (it == animatedNodes.end()) return;

    auto &node = (*asset->nodes)[nodeIndex];
    auto &trs = it->second;

    if (trs.hasTranslation) {
        node.translation = trs.translation;
    }
    if (trs.hasRotation) {
        node.rotation = trs.rotation;
    }
    if (trs.hasScale) {
        node.scale = trs.scale;
    }
}

void Renderer::setViewMode(ViewMode mode) {
    viewMode = mode;
}

ViewMode Renderer::getViewMode() const {
    return viewMode;
}

void Renderer::setDebugMode(bool enabled) {
    viewMode = enabled ? ViewMode::worldNormal : ViewMode::normal;
}

bool Renderer::isDebugModeEnabled() const {
    return viewMode == ViewMode::worldNormal;
}

// ---------------------------------------------------------------------------
// Scene-graph traversal
// ---------------------------------------------------------------------------

void Renderer::gatherVisibleDraws(uint64_t nodeIndex)
{
    if (!asset->nodes || nodeIndex >= asset->nodes->size()) return;

    // Frustum cull: skip this entire subtree if the node's world bounds are outside.
    if (nodeIndex < nodeWorldBounds.size() && !isBoundsVisible(nodeWorldBounds[nodeIndex])) {
        return;
    }

    if (nodeIndex < visibleNodeMask.size()) {
        visibleNodeMask[nodeIndex] = 1;
    }
    auto &node = (*asset->nodes)[nodeIndex];

    if (node.mesh >= 0) {
        updateVisibleInstances(nodeIndex);
        auto &mesh = (*asset->meshes)[(size_t)node.mesh];
        for (auto &primitive : mesh.primitives) {
            DrawItem draw;
            draw.primitive = &primitive;
            draw.nodeIndex = nodeIndex;
            draw.materialIndex = resolvePrimitiveMaterial(primitive);
            draw.transparent = (viewMode == ViewMode::normal) && isTransparentMaterial(draw.materialIndex);
            if (draw.transparent) {
                transparentQueue.push_back(draw);
            } else {
                opaqueQueue.push_back(draw);
            }
        }
    }
    for (auto childIndex : node.children) {
        gatherVisibleDraws(childIndex);
    }
}

void Renderer::setVertexBufferIfChanged(
    const std::shared_ptr<systems::leal::campello_gpu::RenderPassEncoder> &rpe,
    uint32_t slot,
    const std::shared_ptr<systems::leal::campello_gpu::Buffer> &buffer,
    uint64_t offset)
{
    if (slot >= lastBoundVertexBuffers.size()) return;
    auto &last = lastBoundVertexBuffers[slot];
    if (last.buffer == buffer.get() && last.offset == offset) return;
    rpe->setVertexBuffer(slot, buffer, offset);
    last.buffer = buffer.get();
    last.offset = offset;
}

void Renderer::renderPrimitive(
    const std::shared_ptr<systems::leal::campello_gpu::RenderPassEncoder> &rpe,
    const systems::leal::gltf::Primitive &primitive,
    uint64_t nodeIndex)
{
    using namespace systems::leal::gltf;
    
    // --- 1. Determine pipeline variant (flat vs textured), cull mode, and alpha mode ---
    bool hasTexcoord = primitive.attributes.count("TEXCOORD_0") > 0;
    bool hasTexture  = false;
    bool doubleSided = false;
    bool useBlend    = false;  // true for BLEND alpha mode
    int64_t matIdx   = resolvePrimitiveMaterial(primitive);
    bool needsTexturedView =
        viewMode == ViewMode::baseColor ||
        viewMode == ViewMode::metallic ||
        viewMode == ViewMode::roughness ||
        viewMode == ViewMode::occlusion ||
        viewMode == ViewMode::emissive ||
        viewMode == ViewMode::alpha ||
        viewMode == ViewMode::uv0 ||
        viewMode == ViewMode::specularFactor ||
        viewMode == ViewMode::specularColor ||
        viewMode == ViewMode::sheenColor ||
        viewMode == ViewMode::sheenRoughness ||
        viewMode == ViewMode::clearcoat ||
        viewMode == ViewMode::clearcoatRoughness ||
        viewMode == ViewMode::clearcoatNormal ||
        viewMode == ViewMode::transmission;

    // Get material properties (even without texcoords for transmission/blend mode)
    if (matIdx >= 0 && asset->materials && (size_t)matIdx < asset->materials->size()) {
        auto &mat = (*asset->materials)[(size_t)matIdx];
        doubleSided = mat.doubleSided;
        useBlend    = (mat.alphaMode == AlphaMode::blend);
        
        // KHR_materials_transmission: force blend mode if transmission is active
        if (mat.khrMaterialsTransmission && mat.khrMaterialsTransmission->transmissionFactor > 0.0f) {
            useBlend = true;
        }

        if (hasTexcoord) {
            if ((mat.pbrMetallicRoughness && mat.pbrMetallicRoughness->baseColorTexture) ||
                mat.normalTexture ||
                mat.occlusionTexture ||
                mat.emissiveTexture ||
                (mat.khrMaterialsSpecular &&
                 (mat.khrMaterialsSpecular->specularTexture || mat.khrMaterialsSpecular->specularColorTexture)) ||
                mat.khrMaterialsSheen ||
                mat.khrMaterialsClearcoat ||
                mat.khrMaterialsTransmission ||
                needsTexturedView) {
                hasTexture = true;
            }
        } else if (needsTexturedView) {
            hasTexture = true;
        }
    }

    int wantedVariant = hasTexture ? 2 : 1;
    std::shared_ptr<systems::leal::campello_gpu::RenderPipeline> pipeline =
        hasTexture ? pipelineTextured : pipelineFlat;

    // Double-sided materials use no culling.
    if (doubleSided) {
        pipeline = hasTexture ? pipelineTexturedDoubleSided : pipelineFlatDoubleSided;
        wantedVariant = hasTexture ? 5 : 4;
    }
    
    // Alpha-blend materials use blend pipelines (depth write disabled for transparency).
    if (useBlend) {
        if (doubleSided) {
            pipeline = hasTexture ? pipelineTexturedBlendDoubleSided : pipelineFlatBlendDoubleSided;
            wantedVariant = hasTexture ? 9 : 8;
        } else {
            pipeline = hasTexture ? pipelineTexturedBlend : pipelineFlatBlend;
            wantedVariant = hasTexture ? 7 : 6;
        }
    }

    // Debug mode overrides pipeline selection.
    if (viewMode == ViewMode::worldNormal) {
        wantedVariant = 3;
        pipeline      = pipelineDebug;
    }

    // KHR_mesh_quantization: if this primitive uses non-float accessors,
    // switch to the quantized pipeline variant for the current wantedVariant.
    if (asset->accessors && !quantizedPipelines.empty()) {
        bool usesQuantized = false;
        for (auto &[semantic, accIdx] : primitive.attributes) {
            if (accIdx < 0 || (size_t)accIdx >= asset->accessors->size()) continue;
            auto &acc = (*asset->accessors)[(size_t)accIdx];
            if (acc.componentType != systems::leal::gltf::ComponentType::ctFloat) {
                usesQuantized = true;
                break;
            }
        }
        if (usesQuantized) {
            auto it = quantizedPipelines.find(wantedVariant);
            if (it != quantizedPipelines.end() && it->second) {
                pipeline = it->second;
            }
        }
    }

    if (pipeline && wantedVariant != currentPipelineVariant) {
        rpe->setPipeline(pipeline);
        currentPipelineVariant = wantedVariant;
    }

    // --- 2. Bind bind groups ---
    // Index 0: textures + samplers.
    // Index 1: frame-varying buffers — lights (10) and camera position (18).
    // Only bind what the active shader variant actually references to avoid
    // Metal debug-layer "unused binding" assertions.
    bool needsTextures = (wantedVariant == 2 || wantedVariant == 5 ||
                          wantedVariant == 7 || wantedVariant == 9);
    if (needsTextures) {
        std::shared_ptr<systems::leal::campello_gpu::BindGroup> bg = defaultBindGroup;
        if (matIdx >= 0 && (size_t)matIdx < materialBindGroups.size() && materialBindGroups[matIdx]) {
            bg = materialBindGroups[matIdx];
        }
        if (bg) rpe->setBindGroup(0, bg);
        if (frameBindGroup[currentFrameIndex]) {
            rpe->setBindGroup(1, frameBindGroup[currentFrameIndex]);
        }
    } else {
        // Flat/debug variants: bind only the material buffer (no textures/samplers)
        // so the fragment shader can read mat.baseColorFactor without triggering
        // Metal debug-layer unused-binding asserts.
        std::shared_ptr<systems::leal::campello_gpu::BindGroup> bg = defaultFlatBindGroup;
        if (matIdx >= 0 && (size_t)matIdx < flatMaterialBindGroups.size() && flatMaterialBindGroups[matIdx]) {
            bg = flatMaterialBindGroups[matIdx];
        }
        if (bg) rpe->setBindGroup(0, bg);
    }
    
    // --- 5. Bind transform matrices for this node ---
    // Buffer contains: MVP (64 bytes) + Model (64 bytes) = 128 bytes per node.
    if (transformBuffer) {
        uint64_t offset = nodeIndex * 128; // 32 floats * 4 bytes
        if (offset + 128 <= transformBuffer->getLength()) {
            // (bind debug removed)
            setVertexBufferIfChanged(rpe, VERTEX_SLOT_MVP, transformBuffer, offset);
        }
    }

    // Bind material uniforms to vertex stage (shader reads [[buffer(17)]]).
    // matIndex = -1 → slot 0 (default); matIndex >= 0 → slot matIndex+1.
    if (materialUniformBuffer) {
        uint64_t matOffset = (uint64_t)(matIdx + 1) * kMaterialUniformStride;
        if (matOffset + kMaterialUniformStride <= materialUniformBuffer->getLength()) {
            setVertexBufferIfChanged(rpe, VERTEX_SLOT_MATERIAL, materialUniformBuffer, matOffset);
        }
    }

    // --- 5a. Bind instance matrix buffer for EXT_mesh_gpu_instancing ---
    uint32_t instanceCount = 1;
    auto instIt = nodeInstanceData.find(nodeIndex);
    if (instIt != nodeInstanceData.end() && instIt->second.matrixBuffer) {
        if (instIt->second.visibleCount == 0) return;
        setVertexBufferIfChanged(rpe, VERTEX_SLOT_INSTANCE_MATRIX, instIt->second.matrixBuffer, 0);
        instanceCount = instIt->second.visibleCount;
    } else if (defaultInstanceMatrixBuffer) {
        // Bind identity matrix for non-instanced objects
        setVertexBufferIfChanged(rpe, VERTEX_SLOT_INSTANCE_MATRIX, defaultInstanceMatrixBuffer, 0);
    }

    // --- 5b. Bind joint matrix palette for skeletal meshes ---
    int64_t skinIdx = (nodeIndex < nodeSkinIndex.size()) ? nodeSkinIndex[nodeIndex] : -1;
    if (skinIdx >= 0 && (size_t)skinIdx < skinData.size() &&
        frameResources[currentFrameIndex].jointMatrixBuffer) {
        auto &sd = skinData[(size_t)skinIdx];
        setVertexBufferIfChanged(
            rpe, VERTEX_SLOT_JOINT_MATRICES,
            frameResources[currentFrameIndex].jointMatrixBuffer, sd.gpuOffset);
    } else if (defaultJointMatrixBuffer) {
        setVertexBufferIfChanged(
            rpe, VERTEX_SLOT_JOINT_MATRICES, defaultJointMatrixBuffer, 0);
    }

    // --- 6. Check for Draco-compressed buffers ---
    auto dracoIt = dracoPrimitiveBuffers.find(&primitive);
    bool hasDracoGPUBuffer = (dracoIt != dracoPrimitiveBuffers.end() && 
                               dracoIt->second.attributeBuffers.count("POSITION") &&
                               dracoIt->second.attributeBuffers["POSITION"] != nullptr);

    // --- 7. Bind vertex attributes ---
    bool positionBound = false;
    
    if (hasDracoGPUBuffer) {
        // Use Draco-decompressed attribute buffers.
        auto &dracoBufs = dracoIt->second;
        auto bindDracoAttribute = [&](const std::string &semantic, uint32_t slot) -> bool {
            auto it = dracoBufs.attributeBuffers.find(semantic);
            if (it != dracoBufs.attributeBuffers.end() && it->second) {
                setVertexBufferIfChanged(rpe, slot, it->second, 0);
                return true;
            }
            return false;
        };

        positionBound = bindDracoAttribute("POSITION", VERTEX_SLOT_POSITION);
        bindDracoAttribute("NORMAL",   VERTEX_SLOT_NORMAL);
        if (!bindDracoAttribute("TANGENT", VERTEX_SLOT_TANGENT)) {
            if (fallbackTangentBuffer) {
                setVertexBufferIfChanged(rpe, VERTEX_SLOT_TANGENT, fallbackTangentBuffer, 0);
            }
        }

        // TEXCOORD_0: bind real data or fallback zero buffer
        if (!bindDracoAttribute("TEXCOORD_0", VERTEX_SLOT_TEXCOORD0)) {
            if (fallbackUVBuffer) {
                setVertexBufferIfChanged(rpe, VERTEX_SLOT_TEXCOORD0, fallbackUVBuffer, 0);
            }
        }

        // JOINTS_0 / WEIGHTS_0 for skeletal meshes
        if (!bindDracoAttribute("JOINTS_0", VERTEX_SLOT_JOINTS)) {
            if (fallbackJointBuffer) {
                setVertexBufferIfChanged(rpe, VERTEX_SLOT_JOINTS, fallbackJointBuffer, 0);
            }
        }
        if (!bindDracoAttribute("WEIGHTS_0", VERTEX_SLOT_WEIGHTS)) {
            if (fallbackWeightBuffer) {
                setVertexBufferIfChanged(rpe, VERTEX_SLOT_WEIGHTS, fallbackWeightBuffer, 0);
            }
        }
    } else {
        // Use standard GLTF buffer views.
        auto bindAttribute = [&](const std::string &semantic, uint32_t slot) -> bool {
            auto it = primitive.attributes.find(semantic);
            if (it == primitive.attributes.end()) return false;
            int64_t accIdx = it->second;
            auto &acc = (*asset->accessors)[accIdx];
            if (acc.bufferView < 0) return false;
            auto &bv  = (*asset->bufferViews)[(size_t)acc.bufferView];

            // Prefer deinterleaved buffer if this accessor came from an interleaved
            // buffer view (byteStride > 0).
            auto deinterIt = deinterleavedBuffers.find(accIdx);
            if (deinterIt != deinterleavedBuffers.end() && deinterIt->second) {
                setVertexBufferIfChanged(rpe, slot, deinterIt->second, 0);
                return true;
            }

            auto  buf = gpuBuffers[bv.buffer];
            if (buf) {
                setVertexBufferIfChanged(rpe, slot, buf, bv.byteOffset + acc.byteOffset);
                return true;
            }
            return false;
        };

        positionBound = bindAttribute("POSITION", VERTEX_SLOT_POSITION);
        bool normalBound = bindAttribute("NORMAL",   VERTEX_SLOT_NORMAL);
        bool tangentBound = bindAttribute("TANGENT",  VERTEX_SLOT_TANGENT);
        if (!tangentBound && fallbackTangentBuffer) {
            setVertexBufferIfChanged(rpe, VERTEX_SLOT_TANGENT, fallbackTangentBuffer, 0);
        }

        // TEXCOORD_0: bind real data or fallback zero buffer
        if (!bindAttribute("TEXCOORD_0", VERTEX_SLOT_TEXCOORD0)) {
            if (fallbackUVBuffer) {
                setVertexBufferIfChanged(rpe, VERTEX_SLOT_TEXCOORD0, fallbackUVBuffer, 0);
            }
        }

        // JOINTS_0 / WEIGHTS_0 for skeletal meshes
        if (!bindAttribute("JOINTS_0", VERTEX_SLOT_JOINTS)) {
            if (fallbackJointBuffer) {
                setVertexBufferIfChanged(rpe, VERTEX_SLOT_JOINTS, fallbackJointBuffer, 0);
            }
        }
        if (!bindAttribute("WEIGHTS_0", VERTEX_SLOT_WEIGHTS)) {
            if (fallbackWeightBuffer) {
                setVertexBufferIfChanged(rpe, VERTEX_SLOT_WEIGHTS, fallbackWeightBuffer, 0);
            }
        }
    }
    
    // Skip drawing if we couldn't bind a position buffer
    if (!positionBound) return;

    // --- 8. Draw indexed or non-indexed ---
    // EXT_mesh_gpu_instancing: use instanceCount if the node has instance data.
    if (hasDracoGPUBuffer && dracoIt->second.indexBuffer) {
        // Use Draco-decompressed index buffer.
        rpe->setIndexBuffer(dracoIt->second.indexBuffer,
                           systems::leal::campello_gpu::IndexFormat::uint32, 0);
        rpe->drawIndexed(dracoIt->second.indexCount, instanceCount);
    } else if (primitive.indices >= 0) {
        // Use standard GLTF index buffer.
        auto &idxAcc = (*asset->accessors)[(size_t)primitive.indices];
        if (idxAcc.bufferView >= 0) {
            auto &idxBV  = (*asset->bufferViews)[(size_t)idxAcc.bufferView];
            auto  idxBuf = gpuBuffers[idxBV.buffer];
            if (idxBuf) {
                uint64_t idxOffset = idxBV.byteOffset + idxAcc.byteOffset;
                using CT = systems::leal::gltf::ComponentType;
                auto fmt = (idxAcc.componentType == CT::ctUnsignedShort)
                    ? systems::leal::campello_gpu::IndexFormat::uint16
                    : systems::leal::campello_gpu::IndexFormat::uint32;
                rpe->setIndexBuffer(idxBuf, fmt, idxOffset);
                rpe->drawIndexed((uint32_t)idxAcc.count, instanceCount);
                return;
            }
        }
        rpe->draw((uint32_t)idxAcc.count, instanceCount);
    } else {
        // Non-indexed draw - need vertex count from POSITION accessor.
        auto posIt = primitive.attributes.find("POSITION");
        if (posIt != primitive.attributes.end()) {
            auto &posAcc = (*asset->accessors)[posIt->second];
            rpe->draw((uint32_t)posAcc.count, instanceCount);
        }
    }
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
// Scene presentation controls
// ---------------------------------------------------------------------------

void Renderer::setClearColor(float r, float g, float b, float a) {
    clearColor[0] = r;
    clearColor[1] = g;
    clearColor[2] = b;
    clearColor[3] = a;
}

void Renderer::getClearColor(float *outR, float *outG, float *outB, float *outA) const {
    if (outR) *outR = clearColor[0];
    if (outG) *outG = clearColor[1];
    if (outB) *outB = clearColor[2];
    if (outA) *outA = clearColor[3];
}

void Renderer::setPunctualLightsEnabled(bool enabled) {
    punctualLightsEnabled = enabled;
}

bool Renderer::isPunctualLightsEnabled() const {
    return punctualLightsEnabled;
}

void Renderer::setDefaultLightEnabled(bool enabled) {
    defaultLightEnabled = enabled;
}

bool Renderer::isDefaultLightEnabled() const {
    return defaultLightEnabled;
}

// ---------------------------------------------------------------------------
// Environment map / IBL controls
// ---------------------------------------------------------------------------

void Renderer::setEnvironmentMap(std::shared_ptr<systems::leal::campello_gpu::Texture> cubemap) {
    environmentMap = cubemap;
    for (uint32_t f = 0; f < kMaxFramesInFlight; ++f) {
        skyboxBindGroup[f] = nullptr; // Force recreation with new texture
    }
}

std::shared_ptr<systems::leal::campello_gpu::Texture>
Renderer::loadEnvironmentMap(
    const std::string &px, const std::string &nx,
    const std::string &py, const std::string &ny,
    const std::string &pz, const std::string &nz)
{
    namespace GPU = systems::leal::campello_gpu;
    namespace Img = systems::leal::campello_image;

    std::vector<std::shared_ptr<Img::Image>> faces;
    faces.push_back(Img::Image::fromFile(px.c_str()));
    faces.push_back(Img::Image::fromFile(nx.c_str()));
    faces.push_back(Img::Image::fromFile(py.c_str()));
    faces.push_back(Img::Image::fromFile(ny.c_str()));
    faces.push_back(Img::Image::fromFile(pz.c_str()));
    faces.push_back(Img::Image::fromFile(nz.c_str()));

    for (auto &f : faces) {
        if (!f) return nullptr;
    }

    uint32_t w = faces[0]->getWidth();
    uint32_t h = faces[0]->getHeight();
    for (auto &f : faces) {
        if (f->getWidth() != w || f->getHeight() != h) return nullptr;
    }

    GPU::PixelFormat fmt = GPU::PixelFormat::rgba8unorm;
    switch (faces[0]->getFormat()) {
        case Img::ImageFormat::rgba8:   fmt = GPU::PixelFormat::rgba8unorm; break;
        case Img::ImageFormat::rgba16f: fmt = GPU::PixelFormat::rgba16float; break;
        case Img::ImageFormat::rgba32f: fmt = GPU::PixelFormat::rgba32float; break;
    }

    auto tex = device->createTexture(
        GPU::TextureType::ttCube, fmt,
        w, h, 1, 1, 1,
        (GPU::TextureUsage)(uint32_t(GPU::TextureUsage::textureBinding) |
                            uint32_t(GPU::TextureUsage::copyDst)));
    if (!tex) return nullptr;

    size_t bytesPerFace = faces[0]->getDataSize();
    size_t totalBytes = bytesPerFace * 6;
    auto staging = device->createBuffer(totalBytes, GPU::BufferUsage::copySrc);
    if (!staging) return nullptr;

    for (int i = 0; i < 6; ++i) {
        staging->upload(i * bytesPerFace, bytesPerFace, const_cast<void*>(faces[i]->getData()));
    }

    uint64_t bytesPerRow = (h > 0) ? (bytesPerFace / h) : bytesPerFace;
    auto encoder = device->createCommandEncoder();
    if (encoder) {
        for (int i = 0; i < 6; ++i) {
            encoder->copyBufferToTexture(staging, i * bytesPerFace, bytesPerRow, tex, 0, i);
        }
        auto fence = device->createFence();
        device->submit(encoder->finish(), fence);
        if (fence) fence->wait();
    }

    return tex;
}

void Renderer::setSkyboxEnabled(bool enabled) {
    skyboxEnabled = enabled;
}

void Renderer::setIBLEnabled(bool enabled) {
    iblEnabled = enabled;
}

void Renderer::setEnvironmentIntensity(float intensity) {
    environmentIntensity = intensity;
}

bool Renderer::isSkyboxEnabled() const {
    return skyboxEnabled;
}

bool Renderer::isIBLEnabled() const {
    return iblEnabled;
}

float Renderer::getEnvironmentIntensity() const {
    return environmentIntensity;
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

// ---------------------------------------------------------------------------
// KHR_materials_variants
// ---------------------------------------------------------------------------

void Renderer::setMaterialVariant(int32_t variantIndex) {
    activeVariant = variantIndex;
}

uint32_t Renderer::getMaterialVariantCount() const {
    if (!asset || !asset->khrMaterialsVariants) return 0;
    return (uint32_t)asset->khrMaterialsVariants->size();
}

std::string Renderer::getMaterialVariantName(uint32_t variantIndex) const {
    if (!asset || !asset->khrMaterialsVariants) return {};
    if (variantIndex >= asset->khrMaterialsVariants->size()) return {};
    return (*asset->khrMaterialsVariants)[variantIndex].name;
}

std::shared_ptr<systems::leal::campello_gpu::BindGroup>
Renderer::getBindGroup(uint32_t index) const {
    if (index < materialBindGroups.size()) return materialBindGroups[index];
    return nullptr;
}

uint32_t Renderer::getBindGroupCount() const { return (uint32_t)materialBindGroups.size(); }

std::shared_ptr<systems::leal::campello_gpu::BindGroup>
Renderer::getDefaultBindGroup() const { return defaultBindGroup; }

// ---------------------------------------------------------------------------
// Animation control — multi-animation support
// ---------------------------------------------------------------------------

uint32_t Renderer::getAnimationCount() const {
    if (!asset || !asset->animations) return 0;
    return (uint32_t)asset->animations->size();
}

std::string Renderer::getAnimationName(uint32_t animationIndex) const {
    if (!asset || !asset->animations) return {};
    if (animationIndex >= asset->animations->size()) return {};
    return (*asset->animations)[animationIndex].name;
}

double Renderer::getAnimationDuration(uint32_t animationIndex) const {
    if (!asset || !asset->animations || animationIndex >= asset->animations->size()) return 0.0;

    auto &anim = (*asset->animations)[animationIndex];
    double maxTime = 0.0;
    for (auto &sampler : anim.samplers) {
        if (!asset->accessors || sampler.input >= asset->accessors->size()) continue;
        auto &inputAcc = (*asset->accessors)[sampler.input];
        if (inputAcc.bufferView < 0 || !asset->bufferViews) continue;
        auto &inputBV = (*asset->bufferViews)[(size_t)inputAcc.bufferView];
        if ((size_t)inputBV.buffer >= asset->buffers->size()) continue;
        auto &inputBuf = (*asset->buffers)[(size_t)inputBV.buffer];

        const float *times = reinterpret_cast<const float*>(
            inputBuf.data.data() + inputBV.byteOffset + inputAcc.byteOffset);
        if (inputAcc.count > 0) {
            maxTime = std::max(maxTime, (double)times[inputAcc.count - 1]);
        }
    }
    return maxTime;
}

void Renderer::playAnimation(uint32_t animationIndex) {
    if (!asset || !asset->animations || animationIndex >= asset->animations->size()) return;

    auto &state = animationStates[(int32_t)animationIndex];
    state.playing = true;
    // Initialize duration if not set.
    if (state.duration <= 0.0) {
        state.duration = getAnimationDuration(animationIndex);
    }
}

void Renderer::pauseAnimation(uint32_t animationIndex) {
    auto it = animationStates.find((int32_t)animationIndex);
    if (it != animationStates.end()) {
        it->second.playing = false;
    }
}

void Renderer::stopAnimation(uint32_t animationIndex) {
    auto it = animationStates.find((int32_t)animationIndex);
    if (it != animationStates.end()) {
        it->second.playing = false;
        it->second.time = 0.0;
    }
    // Clear animated nodes if no animations are playing.
    bool anyPlaying = false;
    for (auto &pair : animationStates) {
        if (pair.second.playing) {
            anyPlaying = true;
            break;
        }
    }
    if (!anyPlaying) {
        animatedNodes.clear();
    }
}

void Renderer::stopAllAnimations() {
    for (auto &pair : animationStates) {
        pair.second.playing = false;
        pair.second.time = 0.0;
    }
    animatedNodes.clear();
}

bool Renderer::isAnimationPlaying(uint32_t animationIndex) const {
    auto it = animationStates.find((int32_t)animationIndex);
    if (it != animationStates.end()) {
        return it->second.playing;
    }
    return false;
}

void Renderer::setAnimationLoop(uint32_t animationIndex, bool loop) {
    if (!asset || !asset->animations || animationIndex >= asset->animations->size()) return;
    animationStates[(int32_t)animationIndex].loop = loop;
}

bool Renderer::isAnimationLooping(uint32_t animationIndex) const {
    auto it = animationStates.find((int32_t)animationIndex);
    if (it != animationStates.end()) {
        return it->second.loop;
    }
    return true; // Default to looping.
}

void Renderer::setAnimationTime(uint32_t animationIndex, double time) {
    if (!asset || !asset->animations || animationIndex >= asset->animations->size()) return;
    auto &state = animationStates[(int32_t)animationIndex];
    state.time = std::max(0.0, std::min(time, state.duration));
}

double Renderer::getAnimationTime(uint32_t animationIndex) const {
    auto it = animationStates.find((int32_t)animationIndex);
    if (it != animationStates.end()) {
        return it->second.time;
    }
    return 0.0;
}

// ---------------------------------------------------------------------------
// Draco decompressed buffer upload
// ---------------------------------------------------------------------------

void Renderer::uploadDracoBuffers(std::shared_ptr<gltf::RuntimeInfo> &info) {
    (void)info;  // Scene filtering happens at render time
    if (!asset->meshes) return;

    namespace GPU = systems::leal::campello_gpu;
    
    // Find all Draco-compressed primitives and upload their decoded data.
    for (auto &mesh : *asset->meshes) {
        for (auto &primitive : mesh.primitives) {
            if (primitive.khrDracoMeshCompression) {
                auto &draco = primitive.khrDracoMeshCompression;
                DracoBuffers buffers;

                // Upload decoded index buffer.
                if (!draco->decodedIndices.empty()) {
                    using BU = GPU::BufferUsage;
                    auto usage = BU::index;
                    buffers.indexCount = static_cast<uint32_t>(draco->decodedIndices.size());
                    size_t indexSize = draco->decodedIndices.size() * sizeof(uint32_t);
                    buffers.indexBuffer = device->createBuffer(
                        indexSize, usage,
                        reinterpret_cast<uint8_t *>(draco->decodedIndices.data()));
                }

                // Upload decoded attribute buffers.
                for (auto &attr : draco->decodedAttributes) {
                    if (!attr.second.empty()) {
                        using BU = GPU::BufferUsage;
                        auto usage = BU::vertex;
                        buffers.attributeBuffers[attr.first] = device->createBuffer(
                            attr.second.size(), usage,
                            const_cast<uint8_t *>(attr.second.data()));
                    }
                }

                // Validate: Check that POSITION buffer exists and has reasonable size
                auto posIt = buffers.attributeBuffers.find("POSITION");
                if (posIt != buffers.attributeBuffers.end() && posIt->second) {
                    // Draco stores positions as float3 (12 bytes per vertex)
                    size_t posBufferSize = posIt->second->getLength();
                    if (posBufferSize % 12 != 0) continue;
                    
                    // Validate index count is divisible by 3 (triangles)
                    if (buffers.indexCount > 0 && buffers.indexCount % 3 != 0) continue;
                    
                    dracoPrimitiveBuffers[&primitive] = std::move(buffers);
                }
            }
        }
    }
}

std::string systems::leal::campello_renderer::getVersion() {
    return std::to_string(campello_renderer_VERSION_MAJOR) + "." +
           std::to_string(campello_renderer_VERSION_MINOR) + "." +
           std::to_string(campello_renderer_VERSION_PATCH);
}
