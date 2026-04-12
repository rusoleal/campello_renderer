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
        dracoPrimitiveBuffers.clear();
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

    // Decode and upload images referenced by this scene.
    for (int a = 0; a < (int)info->images.size(); a++) {
        if (info->images[a]) {
            if (gpuTextures[a] == nullptr) {
                auto &image = (*asset->images)[a];
                // Colour images (baseColor, emissive) are sRGB-encoded — the GPU
                // linearises them automatically on sample.  Data images (normal,
                // metallicRoughness, occlusion) must remain linear.
                auto fmt = srgbImageIndices.count(a)
                    ? GPU::PixelFormat::rgba8unorm_srgb
                    : GPU::PixelFormat::rgba8unorm;

                if (image.data.size() > 0) {
                    // Data:uri images are already decoded by the gltf library.
                    auto img = systems::leal::campello_image::Image::fromMemory(
                        image.data.data(), image.data.size());
                    if (img != nullptr) {
                        auto texture = device->createTexture(
                            GPU::TextureType::tt2d, fmt,
                            img->getWidth(), img->getHeight(), 1, 1, 1,
                            GPU::TextureUsage::textureBinding);
                        if (texture != nullptr) {
                            texture->upload(0, img->getDataSize(), const_cast<uint8_t*>(img->getData()));
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
                            auto texture = device->createTexture(
                                GPU::TextureType::tt2d, fmt,
                                img->getWidth(), img->getHeight(), 1, 1, 1,
                                GPU::TextureUsage::textureBinding);
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

    // Allocate transform buffer — two float4x4 per node: MVP (clip) + Model (world).
    // Layout per node: 16 floats MVP, 16 floats Model = 32 floats = 128 bytes.
    if (asset->nodes && !asset->nodes->empty()) {
        size_t nodeCount = asset->nodes->size();
        std::cout << "Scene has " << nodeCount << " nodes" << std::endl;
        nodeTransforms.assign(nodeCount * 32, 0.0f); // 32 floats per node
        using BU = systems::leal::campello_gpu::BufferUsage;
        transformBuffer = device->createBuffer(nodeCount * 128, BU::vertex); // 128 bytes per node
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

    // Lights uniform buffer (KHR_lights_punctual) - created once, updated each frame
    if (!lightsUniformBuffer) {
        lightsUniformBuffer = device->createBuffer(272, GPU::BufferUsage::uniform);
        // Initialize to zero (no lights)
        uint8_t zeros[272] = {0};
        lightsUniformBuffer->upload(0, 272, zeros);
    }

    // Camera position buffer — created once, uploaded each frame.
    // 16 bytes for float4 alignment (float3 cameraPos + 1 pad).
    if (!cameraPositionBuffer) {
        cameraPositionBuffer = device->createBuffer(16, GPU::BufferUsage::vertex);
        float defaultCam[4] = {0.f, 0.f, 3.f, 0.f};
        cameraPositionBuffer->upload(0, 16, defaultCam);
    }

    // Material uniform buffer — recreated each setScene() (size depends on material count).
    // Slot 0 = default; slots 1..N = per-material.
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
        defaultClearcoatNormalTexture && lightsUniformBuffer &&
        materialUniformBuffer && cameraPositionBuffer) {
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
            {10, GPU::BufferBinding{lightsUniformBuffer, 0, 272}},
            {11, defaultSpecularTexture},
            {12, defaultSampler},
            {13, defaultSpecularColorTexture},
            {14, defaultSampler},
            {15, defaultSheenColorTexture},
            {16, defaultSheenRoughnessTexture},
            {17, defaultClearcoatTexture},
            {18, defaultClearcoatRoughnessTexture},
            {19, defaultClearcoatNormalTexture},
            // Buffer(17): material uniforms — bound to both vertex and fragment stages via setBindGroup.
            // This makes constant MaterialUniforms &mat [[buffer(17)]] visible to fragment shaders.
            {17, GPU::BufferBinding{materialUniformBuffer, 0, kMaterialUniformStride}},
            // Buffer(18): camera world position — read by fragmentMain_textured [[buffer(18)]].
            {18, GPU::BufferBinding{cameraPositionBuffer, 0, 16}},
        };
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
    // Build one bind group per GLTF material with all material textures.
    // ------------------------------------------------------------------
    materialBindGroups.clear();
    if (asset->materials && bindGroupLayout) {
        materialBindGroups.resize(asset->materials->size());
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

            // Create bind group with all textures and lights buffer
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
                {10, GPU::BufferBinding{lightsUniformBuffer, 0, 272}},
                {11, specularTex},
                {12, specularSamp},
                {13, specularColorTex},
                {14, specularColorSamp},
                {15, sheenColorTex},
                {16, sheenRoughnessTex},
                {17, clearcoatTex},
                {18, clearcoatRoughnessTex},
                {19, clearcoatNormalTex},
                // Buffer(17): material uniforms at the per-material offset — bound to both
                // vertex and fragment stages via setBindGroup, making [[buffer(17)]] visible
                // in all fragment shader variants.
                {17, GPU::BufferBinding{materialUniformBuffer,
                                        (uint64_t)(m + 1) * kMaterialUniformStride,
                                        kMaterialUniformStride}},
                // Buffer(18): camera world position for specular lighting in fragment shader.
                {18, GPU::BufferBinding{cameraPositionBuffer, 0, 16}},
            };
            materialBindGroups[m] = device->createBindGroup(bgDesc);
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
    auto local = nodeLocalMatrix(node);
    auto  world = parentWorld * local;

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
        
        // Debug: print first matrix values for first node
        static int matrixDebugCount = 0;
        if (matrixDebugCount < 3 && nodeIndex == 0) {
            std::cout << "Node 0 Model matrix translation: " 
                      << world.data[3] << ", "
                      << world.data[7] << ", "
                      << world.data[11] << std::endl;
            std::cout << "  First few buffer values: "
                      << nodeTransforms[16] << ", "
                      << nodeTransforms[17] << ", "
                      << nodeTransforms[18] << ", "
                      << nodeTransforms[19] << std::endl;
            matrixDebugCount++;
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
    if (!asset || (!pipelineFlat && !pipelineTextured && !pipelineDebug &&
                   !pipelineFlatDoubleSided && !pipelineTexturedDoubleSided &&
                   !pipelineFlatBlend && !pipelineTexturedBlend &&
                   !pipelineFlatBlendDoubleSided && !pipelineTexturedBlendDoubleSided)) return;
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
            
            if (asset && asset->khrLightsPunctual && !asset->khrLightsPunctual->empty()) {
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

            // If the asset has no lights, synthesize a default directional light
            // so the shader always iterates at least one light with no special cases.
            if (lightsData.count == 0) {
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
            
            // DEBUG: Print light data
            if (lightsData.count > 0) {
                std::cout << "=== LIGHTS UPLOADED: count=" << lightsData.count << " ===" << std::endl;
                for (uint32_t i = 0; i < lightsData.count; i++) {
                    std::cout << "  Light " << i << ": pos=" 
                              << lightsData.lights[i].position[0] << ","
                              << lightsData.lights[i].position[1] << ","
                              << lightsData.lights[i].position[2]
                              << " color="
                              << lightsData.lights[i].color[0] << ","
                              << lightsData.lights[i].color[1] << ","
                              << lightsData.lights[i].color[2]
                              << " intensity=" << lightsData.lights[i].color[3]
                              << std::endl;
                }
            }
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
                            float out[48]) {
            out[0]  = bc[0]; out[1]  = bc[1]; out[2]  = bc[2];  out[3]  = bc[3];
            out[4]  = r0[0]; out[5]  = r0[1]; out[6]  = r0[2];  out[7]  = r0[3];
            out[8]  = r1[0]; out[9]  = r1[1]; out[10] = r1[2];  out[11] = r1[3];
            out[12] = metallic;
            out[13] = roughness;
            out[14] = normalScale;
            out[15] = alphaMode;
            out[16] = alphaCutoff;
            out[17] = unlit;
            out[18] = hasNormal;
            out[19] = hasEmissive;
            out[20] = hasOcclusion;
            out[21] = occlusionStrength;
            out[22] = 0.f; // explicit pad (offset 88)
            out[23] = 0.f; // implicit Metal pad (offset 92) — float3 aligns to 16 bytes → offset 96
            out[24] = emissiveFactor[0]; // offset 96
            out[25] = emissiveFactor[1]; // offset 100
            out[26] = emissiveFactor[2]; // offset 104
            out[27] = ior;               // offset 108 — KHR_materials_ior (default 1.5)
            out[28] = specularFactor;    // offset 112 — KHR_materials_specular scalar weight
            out[29] = hasSpecularTex;    // offset 116
            out[30] = hasSpecularColorTex; // offset 120
            out[31] = 0.f;               // offset 124 — pad before float3
            out[32] = specularColorFactor[0]; // offset 128
            out[33] = specularColorFactor[1]; // offset 132
            out[34] = specularColorFactor[2]; // offset 136
            out[35] = 0.f;               // offset 140 — pad before float3
            out[36] = sheenColorFactor[0]; // offset 144
            out[37] = sheenColorFactor[1]; // offset 148
            out[38] = sheenColorFactor[2]; // offset 152
            out[39] = sheenRoughnessFactor;   // offset 156
            out[40] = hasSheenColorTex;       // offset 160
            out[41] = hasSheenRoughnessTex;   // offset 164
            out[42] = clearcoatFactor;        // offset 168
            out[43] = clearcoatRoughnessFactor; // offset 172
            out[44] = hasClearcoatTex;        // offset 176
            out[45] = hasClearcoatRoughnessTex; // offset 180
            out[46] = hasClearcoatNormalTex;  // offset 184
            out[47] = clearcoatNormalScale;   // offset 188
        };

        // Default slot — white, identity UV transform, metallic=1, roughness=1, no textures, no sheen, no clearcoat.
        {
            float bc[4]            = {1.f, 1.f, 1.f, 1.f};
            float row0[4]          = {1.f, 0.f, 0.f, 0.f}; // w=0 → identity fast path
            float row1[4]          = {0.f, 1.f, 0.f, 0.f};
            float emissive[3]      = {0.f, 0.f, 0.f};
            float specularColor[3] = {1.f, 1.f, 1.f};
            float sheenColor[3]    = {0.f, 0.f, 0.f};
            float slot[48];
            buildSlot(bc, row0, row1, 1.f, 1.f, 1.f, 0.f, 0.5f, 0.f, 0.f, 0.f, 0.f, 1.f,
                      emissive, 1.5f, 1.f, 0.f, 0.f, specularColor,
                      sheenColor, 0.f, 0.f, 0.f,
                      0.f, 0.f, 0.f, 0.f, 0.f, 1.f, slot);
            materialUniformBuffer->upload(0, (uint64_t)sizeof(slot), slot);
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

                float slot[48];
                buildSlot(bc, row0, row1, metallic, roughness, normalScale,
                          alphaMode, alphaCutoff, unlit, hasNormal, hasEmissive, hasOcclusion,
                          occlusionStrength, emissiveFactor, ior,
                          specularFactor, hasSpecularTex, hasSpecularColorTex, specularColorFactor,
                          sheenColorFactor, sheenRoughnessFactor, hasSheenColorTex, hasSheenRoughnessTex,
                          clearcoatFactor, clearcoatRoughnessFactor,
                          hasClearcoatTex, hasClearcoatRoughnessTex, hasClearcoatNormalTex, clearcoatNormalScale,
                          slot);
                
                // Debug: print first material's values
                if (i == 0) {
                    std::cout << "Material 0 upload:" << std::endl;
                    std::cout << "  baseColor: " << bc[0] << ", " << bc[1] << ", " << bc[2] << ", " << bc[3] << std::endl;
                    std::cout << "  metallic: " << metallic << ", roughness: " << roughness << std::endl;
                    std::cout << "  alphaMode: " << alphaMode << ", unlit: " << unlit << std::endl;
                    std::cout << "  hasNormal: " << hasNormal << std::endl;
                    std::cout << "  emissiveFactor: " << emissiveFactor[0] << ", " << emissiveFactor[1] << ", " << emissiveFactor[2] << std::endl;
                    std::cout << "  hasEmissive: " << hasEmissive << std::endl;
                }
                
                materialUniformBuffer->upload((uint64_t)(i + 1) * kMaterialUniformStride,
                                              (uint64_t)sizeof(slot), slot);
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
    // Match the MTKView clear color (dark blue-gray) to avoid blinking
    ca.clearValue[0] = 0.08f;
    ca.clearValue[1] = 0.08f;
    ca.clearValue[2] = 0.10f;
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
        ds.stencilReadOnly   = false;
        rpDesc.depthStencilAttachment = ds;
    }

    auto rpe = encoder->beginRenderPass(rpDesc);
    if (!rpe) return;

    currentPipelineVariant = 0; // reset — renderPrimitive() will set it per draw

    if (renderWidth > 0 && renderHeight > 0) {
        rpe->setViewport(0.0f, 0.0f, (float)renderWidth, (float)renderHeight, 0.0f, 1.0f);
        rpe->setScissorRect(0.0f, 0.0f, (float)renderWidth, (float)renderHeight);
    }

    transparentQueue.clear();

    if (scene.nodes) {
        for (auto nodeIndex : *scene.nodes) {
            renderNode(rpe, nodeIndex);
        }
    }

    // Sort transparent draws back-to-front and draw them after all opaque geometry.
    // nodeTransforms stores the model matrix column-major at [nodeIndex*32 + 16..31];
    // the world translation is at column 3: offsets 28 (X), 29 (Y), 30 (Z).
    if (!transparentQueue.empty()) {
        std::sort(transparentQueue.begin(), transparentQueue.end(),
            [&](const TransparentDraw &a, const TransparentDraw &b) {
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
    device->submit(encoder->finish());
}

void Renderer::update(double dt) {
    // Reserved for animation updates.
    (void)dt;
}

void Renderer::setDebugMode(bool enabled) {
    debugModeEnabled = enabled;
}

bool Renderer::isDebugModeEnabled() const {
    return debugModeEnabled;
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
            // Defer BLEND primitives to the back-to-front transparent pass.
            // Debug mode skips alpha logic entirely — draw immediately.
            bool isBlend = false;
            if (!debugModeEnabled) {
                int64_t matIdx = primitive.material;
                if (activeVariant >= 0 && !primitive.khrMaterialsVariantsMappings.empty()) {
                    for (auto &mapping : primitive.khrMaterialsVariantsMappings) {
                        for (auto v : mapping.variants) {
                            if ((int64_t)v == activeVariant) { matIdx = mapping.material; goto blendCheck; }
                        }
                    }
                }
                blendCheck:
                if (matIdx >= 0 && asset->materials && (size_t)matIdx < asset->materials->size()) {
                    isBlend = ((*asset->materials)[(size_t)matIdx].alphaMode == gltf::AlphaMode::blend);
                }
            }
            if (isBlend) {
                transparentQueue.push_back({&primitive, nodeIndex});
            } else {
                renderPrimitive(rpe, primitive, nodeIndex);
            }
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
    using namespace systems::leal::gltf;
    
    // --- 1. Determine pipeline variant (flat vs textured), cull mode, and alpha mode ---
    bool hasTexcoord = primitive.attributes.count("TEXCOORD_0") > 0;
    bool hasTexture  = false;
    bool doubleSided = false;
    bool useBlend    = false;  // true for BLEND alpha mode
    int64_t matIdx   = primitive.material;

    // KHR_materials_variants: override matIdx when a variant is active.
    if (activeVariant >= 0 && !primitive.khrMaterialsVariantsMappings.empty()) {
        for (auto &mapping : primitive.khrMaterialsVariantsMappings) {
            for (auto v : mapping.variants) {
                if ((int64_t)v == activeVariant) {
                    matIdx = mapping.material;
                    goto variantResolved;
                }
            }
        }
        variantResolved:;
    }

    if (hasTexcoord && matIdx >= 0 && asset->materials &&
        (size_t)matIdx < asset->materials->size())
    {
        auto &mat = (*asset->materials)[(size_t)matIdx];
        if (mat.pbrMetallicRoughness && mat.pbrMetallicRoughness->baseColorTexture)
            hasTexture = true;
        doubleSided = mat.doubleSided;
        useBlend    = (mat.alphaMode == AlphaMode::blend);
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
    if (debugModeEnabled) {
        wantedVariant = 3;
        pipeline      = pipelineDebug;
    }

    if (pipeline && wantedVariant != currentPipelineVariant) {
        rpe->setPipeline(pipeline);
        currentPipelineVariant = wantedVariant;
    }

    // --- 2. Bind material bind group at index 0 ---
    // The bind group contains textures (fragment stage), samplers (fragment stage),
    // lightsUniformBuffer (fragment [[buffer(10)]]),
    // materialUniformBuffer at the per-material offset (both vertex [[buffer(17)]] and
    // fragment [[buffer(17)]]), and cameraPositionBuffer (fragment [[buffer(18)]]).
    // Always bound — even in debug mode, unused bindings are harmless on Metal.
    {
        std::shared_ptr<systems::leal::campello_gpu::BindGroup> bg = defaultBindGroup;

        if (matIdx >= 0 && (size_t)matIdx < materialBindGroups.size() && materialBindGroups[matIdx]) {
            bg = materialBindGroups[matIdx];
        }

        if (bg) rpe->setBindGroup(0, bg);
    }
    
    // --- 5. Bind transform matrices for this node ---
    // Buffer contains: MVP (64 bytes) + Model (64 bytes) = 128 bytes per node.
    if (transformBuffer) {
        uint64_t offset = nodeIndex * 128; // 32 floats * 4 bytes
        if (offset + 128 <= transformBuffer->getLength()) {
            // Debug: print first few binding operations
            static int matrixBindCount = 0;
            if (matrixBindCount < 5) {
                std::cout << "Binding matrices for node " << nodeIndex 
                          << " at offset " << offset 
                          << " buffer size " << transformBuffer->getLength() << std::endl;
                matrixBindCount++;
            }
            rpe->setVertexBuffer(VERTEX_SLOT_MVP, transformBuffer, offset);
        }
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
                rpe->setVertexBuffer(slot, it->second, 0);
                return true;
            }
            return false;
        };

        positionBound = bindDracoAttribute("POSITION", VERTEX_SLOT_POSITION);
        bindDracoAttribute("NORMAL",   VERTEX_SLOT_NORMAL);
        bindDracoAttribute("TANGENT",  VERTEX_SLOT_TANGENT);

        // TEXCOORD_0: bind real data or fallback zero buffer
        if (!bindDracoAttribute("TEXCOORD_0", VERTEX_SLOT_TEXCOORD0)) {
            if (fallbackUVBuffer) {
                rpe->setVertexBuffer(VERTEX_SLOT_TEXCOORD0, fallbackUVBuffer, 0);
            }
        }
    } else {
        // Use standard GLTF buffer views.
        auto bindAttribute = [&](const std::string &semantic, uint32_t slot) -> bool {
            auto it = primitive.attributes.find(semantic);
            if (it == primitive.attributes.end()) return false;
            auto &acc = (*asset->accessors)[it->second];
            if (acc.bufferView < 0) return false;
            auto &bv  = (*asset->bufferViews)[(size_t)acc.bufferView];
            auto  buf = gpuBuffers[bv.buffer];
            if (buf) {
                rpe->setVertexBuffer(slot, buf, bv.byteOffset + acc.byteOffset);
                return true;
            }
            return false;
        };

        positionBound = bindAttribute("POSITION", VERTEX_SLOT_POSITION);
        bool normalBound = bindAttribute("NORMAL",   VERTEX_SLOT_NORMAL);
        bool tangentBound = bindAttribute("TANGENT",  VERTEX_SLOT_TANGENT);
        
        // Debug: check if attributes were bound
        static int attrDebugCount = 0;
        if (attrDebugCount < 10) {
            std::cout << "Attributes bound: POS=" << positionBound 
                      << " NORM=" << normalBound 
                      << " TAN=" << tangentBound << std::endl;
            std::cout << "  Primitive has NORMAL attr: " 
                      << (primitive.attributes.count("NORMAL") > 0 ? "YES" : "NO") << std::endl;
            attrDebugCount++;
        }

        // TEXCOORD_0: bind real data or fallback zero buffer
        if (!bindAttribute("TEXCOORD_0", VERTEX_SLOT_TEXCOORD0)) {
            if (fallbackUVBuffer) {
                rpe->setVertexBuffer(VERTEX_SLOT_TEXCOORD0, fallbackUVBuffer, 0);
            }
        }
    }
    
    // Skip drawing if we couldn't bind a position buffer
    if (!positionBound) return;

    // --- 8. Draw indexed or non-indexed ---
    if (hasDracoGPUBuffer && dracoIt->second.indexBuffer) {
        // Use Draco-decompressed index buffer.
        rpe->setIndexBuffer(dracoIt->second.indexBuffer,
                           systems::leal::campello_gpu::IndexFormat::uint32, 0);
        rpe->drawIndexed(dracoIt->second.indexCount);
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
                rpe->drawIndexed((uint32_t)idxAcc.count);
                return;
            }
        }
        rpe->draw((uint32_t)idxAcc.count);
    } else {
        // Non-indexed draw - need vertex count from POSITION accessor.
        auto posIt = primitive.attributes.find("POSITION");
        if (posIt != primitive.attributes.end()) {
            auto &posAcc = (*asset->accessors)[posIt->second];
            rpe->draw((uint32_t)posAcc.count);
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
