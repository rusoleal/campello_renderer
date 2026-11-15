#include <campello_renderer/path_tracer.hpp>

#include <campello_gpu/acceleration_structure.hpp>
#include <campello_gpu/bind_group.hpp>
#include <campello_gpu/bind_group_layout.hpp>
#include <campello_gpu/buffer.hpp>
#include <campello_gpu/command_buffer.hpp>
#include <campello_gpu/command_encoder.hpp>
#include <campello_gpu/compute_pass_encoder.hpp>
#include <campello_gpu/compute_pipeline.hpp>
#include <campello_gpu/constants/feature.hpp>
#include <campello_gpu/descriptors/bind_group_descriptor.hpp>
#include <campello_gpu/descriptors/bind_group_layout_descriptor.hpp>
#include <campello_gpu/descriptors/bottom_level_acceleration_structure_descriptor.hpp>
#include <campello_gpu/descriptors/compute_pipeline_descriptor.hpp>
#include <campello_gpu/descriptors/pipeline_layout_descriptor.hpp>
#include <campello_gpu/descriptors/ray_tracing_pipeline_descriptor.hpp>
#include <campello_gpu/descriptors/top_level_acceleration_structure_descriptor.hpp>
#include <campello_gpu/device.hpp>
#include <campello_gpu/fence.hpp>
#include <campello_gpu/ray_tracing_pass_encoder.hpp>
#include <campello_gpu/ray_tracing_pipeline.hpp>
#include <campello_gpu/texture.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

#if defined(__APPLE__)
#include "shaders/metal_path_tracer.h"
#endif

#if defined(CAMPELLO_RENDERER_DENOISER)
#include <OpenImageDenoise/oidn.hpp>
#endif

namespace systems::leal::campello_renderer {

namespace GPU = systems::leal::campello_gpu;
namespace VM = systems::leal::vector_math;

namespace {

// Layouts shared with shaders/metal/path_tracer.metal.
struct NormalParams {
    uint32_t triangleCount;
    uint32_t outOffset;
    uint32_t indexFormat;
    uint32_t vertexStride;
    uint32_t uvFormat;  // 0 none, 1 float2, 2 unorm16x2, 3 unorm8x2
    uint32_t uvStride;
    uint32_t normalFormat;  // 0 none (face normals), 1 float3
    uint32_t normalStride;
};

constexpr uint32_t MAX_TEXTURES = 64;  // as in path_tracer.metal
constexpr uint32_t FIRST_TEXTURE = 2;  // texture bindings 2..65

struct TraceUniforms {
    float invViewProj[16];
    float cameraPos[4];
    float sunDirection[4][4];
    float sunRadiance[4][4];
    float params[4];
    uint32_t size[4];
    float clearColor[4];
    float cameraForward[4];
    float cameraRight[4];
    float cameraUp[4];
    float tuning[4];         // x indirect clamp (0 off), y sampler (0 random, 1 Sobol)
    uint32_t envSampling[4]; // x environment table resolution per face (0: none)
    uint32_t lights[4];      // x emissive triangles sampled as lights
    float camera[4];         // x lens aperture radius, y focus distance, z tone mapping
};

struct EmissiveParams {
    uint32_t triangleCount;
    uint32_t outOffset;
    uint32_t indexFormat;
    uint32_t vertexStride;
    float model[3][4];
    float radiance[4];
};

struct EmissiveTri {
    float v0[4];
    float e1[4];
    float e2[4];
    float radiance[4];  // w: area
};

constexpr uint32_t MAX_EMISSIVE_TRIANGLES = 1u << 20;  // sampled as lights (beyond: found by bounces only)

struct PortalInfo {
    float corner[4];
    float axisX[4];  // unit, w length
    float axisY[4];
};
static_assert(PathTracer::MAX_PORTALS == 32, "MAX_PORTALS must match path_tracer.metal");

struct EnvParams {
    uint32_t resolution;
    uint32_t pad[3];
};

struct EnvTexel {
    float cdf;
    float pdf;
};

constexpr uint32_t ENV_RESOLUTION = 64;  // environment table texels per cube face edge

/// rgb (3 floats per pixel) scaled up from w x h to W x H: nearest, or
/// bilinear (pixel centres).
void upscale(const std::vector<float>& src, uint32_t w, uint32_t h, std::vector<float>& dst, uint32_t W, uint32_t H,
             bool bilinear) {
    dst.resize((size_t)W * H * 3);
    const float sx = float(w) / float(W), sy = float(h) / float(H);
    for (uint32_t y = 0; y < H; ++y) {
        for (uint32_t x = 0; x < W; ++x) {
            float* out = &dst[((size_t)y * W + x) * 3];
            if (!bilinear) {
                const uint32_t ix = std::min(uint32_t(float(x) * sx), w - 1), iy = std::min(uint32_t(float(y) * sy), h - 1);
                for (int c = 0; c < 3; ++c) out[c] = src[((size_t)iy * w + ix) * 3 + c];
                continue;
            }
            const float fx = std::clamp((float(x) + 0.5f) * sx - 0.5f, 0.0f, float(w - 1));
            const float fy = std::clamp((float(y) + 0.5f) * sy - 0.5f, 0.0f, float(h - 1));
            const uint32_t x0 = uint32_t(fx), y0 = uint32_t(fy);
            const uint32_t x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
            const float tx = fx - float(x0), ty = fy - float(y0);
            for (int c = 0; c < 3; ++c) {
                const float a = src[((size_t)y0 * w + x0) * 3 + c], b = src[((size_t)y0 * w + x1) * 3 + c];
                const float d = src[((size_t)y1 * w + x0) * 3 + c], e = src[((size_t)y1 * w + x1) * 3 + c];
                out[c] = (a + (b - a) * tx) * (1.0f - ty) + (d + (e - d) * tx) * ty;
            }
        }
    }
}

double millisecondsSince(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

struct InstanceInfo {
    float normalMatrix[3][4];
    uint32_t normalOffset;
    uint32_t material;
    uint32_t lightOffset;  // first entry in the emissive light list + 1 (0: none)
    uint32_t pad1;
    float model[3][4];  // the model's 3x3 rows (tangents)
};

struct MaterialInfo {
    float baseColor[4];
    float emissive[4];
    float roughness;
    float metallic;
    float transmission;
    float ior;
    float unlit;
    float worldMapping;
    float worldMappingUp;
    int32_t texture;  // -1 none
    float uvRow0[4];
    float uvRow1[4];
    float average[4];
    int32_t normalTexture;  // -1 none
    int32_t mrTexture;      // -1 none
    float normalScale;
    float doubleSided;
    float normalUvRow0[4];
    float normalUvRow1[4];
};
static_assert(sizeof(MaterialInfo) == 160, "MaterialInfo must match path_tracer.metal");

/// How a mesh's NORMAL is stored, for the faceNormals kernel (0: none or not float3).
uint32_t normalFormat(const GpuMesh& mesh, uint32_t& stride) {
    if (!mesh.vertexBuffers[Renderer::VERTEX_SLOT_NORMAL]) return 0;
    VertexAttributeFormat format{VertexSemantic::normal, GPU::ComponentType::ctFloat, GPU::AccessorType::acVec3, false};
    if (mesh.layout) {
        const VertexAttributeFormat* found = mesh.layout->find(VertexSemantic::normal);
        if (!found) return 0;
        format = *found;
    }
    if (format.componentType != GPU::ComponentType::ctFloat || format.accessorType != GPU::AccessorType::acVec3) return 0;
    stride = format.stride();
    return 1;
}

/// How a mesh's TEXCOORD_0 is stored, for the faceNormals kernel (0: none or unreadable).
uint32_t uvFormat(const GpuMesh& mesh, uint32_t& stride) {
    if (!mesh.vertexBuffers[Renderer::VERTEX_SLOT_TEXCOORD0]) return 0;
    VertexAttributeFormat format{VertexSemantic::texcoord0, GPU::ComponentType::ctFloat, GPU::AccessorType::acVec2, false};
    if (mesh.layout) {
        const VertexAttributeFormat* found = mesh.layout->find(VertexSemantic::texcoord0);
        if (!found) return 0;
        format = *found;
    }
    if (format.accessorType != GPU::AccessorType::acVec2) return 0;
    stride = format.stride();
    if (format.componentType == GPU::ComponentType::ctFloat) return 1;
    if (format.normalized && format.componentType == GPU::ComponentType::ctUnsignedShort) return 2;
    if (format.normalized && format.componentType == GPU::ComponentType::ctUnsignedByte) return 3;
    return 0;
}

void mix(uint64_t& key, const void* data, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i) {
        key ^= bytes[i];
        key *= 1099511628211ull;
    }
}

bool floatPositions(const GpuMesh& mesh) {
    if (!mesh.layout) return true;
    const VertexAttributeFormat* format = mesh.layout->find(VertexSemantic::position);
    return format && format->componentType == GPU::ComponentType::ctFloat && format->accessorType == GPU::AccessorType::acVec3;
}

} // namespace

PathTracer::PathTracer(std::shared_ptr<GPU::Device> device)
    : device_(std::move(device)) {
#if defined(__APPLE__) && !TARGET_OS_SIMULATOR
    if (!device_ || !device_->getFeatures().count(GPU::Feature::raytracing)) return;
    auto module = device_->createShaderModule(shaders::kPathTracerMetalShader, shaders::kPathTracerMetalShaderSize);
    if (!module) return;

    // Trace: the scene (0), accumulation (1), uniforms (2), instances (3),
    // materials (4), normals (5), denoiser guides (6), texture coordinates
    // (7), environment sampling table (8), light portals (9), second moments
    // (10), converged pixel counter (11), vertex normals (12), tangents (13),
    // emissive triangles (14) and their distribution (15); textures: output (0),
    // environment (1), base colour textures (2..65).
    {
        GPU::BindGroupLayoutDescriptor desc{};
        GPU::EntryObject scene{};
        scene.binding = 0;
        scene.type = GPU::EntryObjectType::accelerationStructure;
        scene.visibility = GPU::ShaderStage::rayGeneration;
        desc.entries.push_back(scene);
        for (uint32_t binding = 1; binding <= 15; ++binding) {
            GPU::EntryObject buffer{};
            buffer.binding = binding;
            buffer.type = GPU::EntryObjectType::buffer;
            buffer.visibility = GPU::ShaderStage::rayGeneration;
            buffer.data.buffer.hasDinamicOffaset = false;
            buffer.data.buffer.minBindingSize = 0;
            buffer.data.buffer.type = binding == 2 ? GPU::EntryObjectBufferType::uniform : GPU::EntryObjectBufferType::storage;
            desc.entries.push_back(buffer);
        }
        for (uint32_t binding = 0; binding < FIRST_TEXTURE + MAX_TEXTURES; ++binding) {
            GPU::EntryObject texture{};
            texture.binding = binding;
            texture.type = GPU::EntryObjectType::texture;
            texture.visibility = GPU::ShaderStage::rayGeneration;
            texture.data.texture = {false, GPU::EntryObjectTextureType::ttFloat,
                                    binding == 1 ? GPU::TextureType::ttCube : GPU::TextureType::tt2d};
            desc.entries.push_back(texture);
        }
        traceLayout_ = device_->createBindGroupLayout(desc);
    }
    {
        GPU::BindGroupLayoutDescriptor desc{};
        for (uint32_t binding = 0; binding <= 8; ++binding) {
            GPU::EntryObject buffer{};
            buffer.binding = binding;
            buffer.type = GPU::EntryObjectType::buffer;
            buffer.visibility = GPU::ShaderStage::compute;
            buffer.data.buffer.hasDinamicOffaset = false;
            buffer.data.buffer.minBindingSize = 0;
            buffer.data.buffer.type = binding == 3 ? GPU::EntryObjectBufferType::uniform : GPU::EntryObjectBufferType::storage;
            desc.entries.push_back(buffer);
        }
        normalsLayout_ = device_->createBindGroupLayout(desc);
    }
    {
        // Environment luminance: weights (buffer 0), parameters (buffer 1),
        // the environment cube (texture 0). A ray generation stage: the
        // Metal backend's compute passes bind buffers only.
        GPU::BindGroupLayoutDescriptor desc{};
        for (uint32_t binding = 0; binding <= 1; ++binding) {
            GPU::EntryObject buffer{};
            buffer.binding = binding;
            buffer.type = GPU::EntryObjectType::buffer;
            buffer.visibility = GPU::ShaderStage::rayGeneration;
            buffer.data.buffer.hasDinamicOffaset = false;
            buffer.data.buffer.minBindingSize = 0;
            buffer.data.buffer.type = binding == 1 ? GPU::EntryObjectBufferType::uniform : GPU::EntryObjectBufferType::storage;
            desc.entries.push_back(buffer);
        }
        GPU::EntryObject texture{};
        texture.binding = 0;
        texture.type = GPU::EntryObjectType::texture;
        texture.visibility = GPU::ShaderStage::rayGeneration;
        texture.data.texture = {false, GPU::EntryObjectTextureType::ttFloat, GPU::TextureType::ttCube};
        desc.entries.push_back(texture);
        envLayout_ = device_->createBindGroupLayout(desc);
    }
    if (!traceLayout_ || !normalsLayout_) return;

    if (envLayout_) {
        // Optional: without it the sky is found by bounces only.
        GPU::PipelineLayoutDescriptor envPl{};
        envPl.bindGroupLayouts = {envLayout_};
        GPU::RayTracingPipelineDescriptor envDesc{};
        envDesc.rayGeneration = {module, "environmentLuminance"};
        envDesc.layout = device_->createPipelineLayout(envPl);
        envDesc.maxRecursionDepth = 1;
        envPipeline_ = device_->createRayTracingPipeline(envDesc);
    }

    GPU::PipelineLayoutDescriptor normalsPl{};
    normalsPl.bindGroupLayouts = {normalsLayout_};
    GPU::ComputePipelineDescriptor normalsDesc{};
    normalsDesc.compute = {module, "faceNormals"};
    normalsDesc.layout = device_->createPipelineLayout(normalsPl);
    normalsPipeline_ = device_->createComputePipeline(normalsDesc);
    {
        // Emissive triangles: positions (0), indices (1), out (2), parameters (3).
        GPU::BindGroupLayoutDescriptor desc{};
        for (uint32_t binding = 0; binding <= 3; ++binding) {
            GPU::EntryObject buffer{};
            buffer.binding = binding;
            buffer.type = GPU::EntryObjectType::buffer;
            buffer.visibility = GPU::ShaderStage::compute;
            buffer.data.buffer.hasDinamicOffaset = false;
            buffer.data.buffer.minBindingSize = 0;
            buffer.data.buffer.type = binding == 3 ? GPU::EntryObjectBufferType::uniform : GPU::EntryObjectBufferType::storage;
            desc.entries.push_back(buffer);
        }
        emissiveLayout_ = device_->createBindGroupLayout(desc);
        if (emissiveLayout_) {
            GPU::PipelineLayoutDescriptor pl{};
            pl.bindGroupLayouts = {emissiveLayout_};
            GPU::ComputePipelineDescriptor cd{};
            cd.compute = {module, "emissiveTriangles"};
            cd.layout = device_->createPipelineLayout(pl);
            emissivePipeline_ = device_->createComputePipeline(cd);  // optional: emitters found by bounces without it
        }
    }

    GPU::PipelineLayoutDescriptor tracePl{};
    tracePl.bindGroupLayouts = {traceLayout_};
    GPU::RayTracingPipelineDescriptor traceDesc{};
    traceDesc.rayGeneration = {module, "pathTrace"};
    traceDesc.layout = device_->createPipelineLayout(tracePl);
    traceDesc.maxRecursionDepth = 1;
    if (normalsPipeline_) {
        pipeline_ = device_->createRayTracingPipeline(traceDesc);
    }
#endif
}

PathTracer::~PathTracer() {
    waitForSample();  // the GPU may still be using the buffers
}

std::vector<DrawCall> PathTracer::traced(const RenderScene& scene) const {
    std::vector<DrawCall> out;
    const auto& opaque = scene.shadowCasters.empty() ? scene.opaque : scene.shadowCasters;
    for (const auto* list : {&opaque, &scene.transparent}) {
        for (const auto& draw : *list) {
            if (!draw.mesh || !draw.material || draw.mesh->topology != GPU::PrimitiveTopology::triangleList ||
                !draw.mesh->vertexBuffers[Renderer::VERTEX_SLOT_POSITION] || !floatPositions(*draw.mesh) ||
                draw.mesh->vertexCount == 0) {
                continue;
            }
            out.push_back(draw);
        }
    }
    return out;
}

bool PathTracer::rebuild(const RenderScene& scene, const Renderer& renderer, bool transformsOnly) {
    const auto draws = traced(scene);
    if (draws.empty()) return false;

    // Acceleration structures of meshes not seen before (kept while the
    // mesh is in the scene).
    std::map<const GpuMesh*, MeshEntry> meshes;
    std::vector<std::pair<std::shared_ptr<GPU::AccelerationStructure>, GPU::BottomLevelAccelerationStructureDescriptor>> builds;
    for (const auto& draw : draws) {
        if (meshes.count(draw.mesh)) continue;
        const void* positions = draw.mesh->vertexBuffers[Renderer::VERTEX_SLOT_POSITION].get();
        const void* indices = draw.mesh->indexBuffer.get();
        if (auto it = meshes_.find(draw.mesh);
            it != meshes_.end() && it->second.positions == positions && it->second.indices == indices) {
            meshes.emplace(draw.mesh, it->second);
            continue;
        }
        GPU::AccelerationStructureGeometryDescriptor geometry{};
        geometry.type = GPU::AccelerationStructureGeometryType::triangles;
        geometry.opaque = true;
        geometry.vertexBuffer = draw.mesh->vertexBuffers[Renderer::VERTEX_SLOT_POSITION];
        geometry.vertexStride = 12;
        geometry.vertexCount = draw.mesh->vertexCount;
        geometry.componentType = GPU::ComponentType::ctFloat;
        uint32_t triangles = draw.mesh->vertexCount / 3;
        if (draw.mesh->indexBuffer && draw.mesh->indexCount > 0) {
            geometry.indexBuffer = draw.mesh->indexBuffer;
            geometry.indexFormat = draw.mesh->indexFormat;
            geometry.indexCount = draw.mesh->indexCount;
            triangles = draw.mesh->indexCount / 3;
        }
        if (triangles == 0) continue;
        GPU::BottomLevelAccelerationStructureDescriptor desc{};
        desc.geometries = {geometry};
        desc.buildFlags = GPU::AccelerationStructureBuildFlag::preferFastTrace;
        auto blas = device_->createBottomLevelAccelerationStructure(desc);
        if (!blas) continue;
        meshes.emplace(draw.mesh, MeshEntry{blas, triangles, positions, indices});
        builds.emplace_back(blas, desc);
    }
    meshes_ = std::move(meshes);

    // Normal offsets per mesh, materials and instances.
    std::map<const GpuMesh*, uint32_t> normalOffsets;
    uint32_t triangleTotal = 0;
    for (const auto& [mesh, entry] : meshes_) {
        normalOffsets[mesh] = triangleTotal;
        triangleTotal += entry.triangleCount;
    }
    std::map<const GpuMaterial*, uint32_t> materialIndex;
    std::vector<MaterialInfo> materials;
    std::map<const GPU::Texture*, int32_t> textureIndex;
    textures_.clear();
    std::vector<InstanceInfo> instanceInfo;
    GPU::TopLevelAccelerationStructureDescriptor tlasDesc{};
    tlasDesc.buildFlags = GPU::AccelerationStructureBuildFlag::preferFastTrace;
    for (const auto& draw : draws) {
        const auto meshIt = meshes_.find(draw.mesh);
        if (meshIt == meshes_.end()) continue;
        auto [matIt, added] = materialIndex.emplace(draw.material, static_cast<uint32_t>(materials.size()));
        if (added) {
            MaterialInfo info{};
            info.baseColor[0] = info.baseColor[1] = info.baseColor[2] = info.baseColor[3] = 1.0f;
            info.roughness = 0.8f;
            info.ior = 1.5f;
            info.texture = -1;
            const auto& average = draw.material->baseColorTextureAverage;
            for (int c = 0; c < 3; ++c) info.average[c] = average[(size_t)c];
            info.average[3] = 1.0f;
            info.normalTexture = -1;
            info.mrTexture = -1;
            info.normalScale = 1.0f;
            // Textures, while there are bindings for them (beyond that: the
            // base colour's average colour, no normal or roughness detail).
            const auto bind = [&](const std::shared_ptr<GPU::Texture>& texture) {
                if (!texture) return int32_t(-1);
                auto [texIt, newTexture] = textureIndex.emplace(texture.get(), static_cast<int32_t>(textures_.size()));
                if (newTexture) {
                    if (textures_.size() < MAX_TEXTURES) {
                        textures_.push_back(texture);
                    } else {
                        texIt->second = -1;
                    }
                }
                return texIt->second;
            };
            info.texture = bind(draw.material->baseColorTexture);
            const float* uniforms = renderer.materialUniforms(draw.material);
            // (The tracer has TEXCOORD_0 only: textures on TEXCOORD_1 are left out.)
            const uint32_t uv1 = uniforms ? static_cast<uint32_t>(uniforms[92]) : 0u;
            if (!(uv1 & (1u << 1))) info.mrTexture = bind(draw.material->metallicRoughnessTexture);
            if (!(uv1 & (1u << 2))) info.normalTexture = bind(draw.material->normalTexture);
            if (const float* slot = renderer.materialUniforms(draw.material)) {
                for (int c = 0; c < 4; ++c) info.baseColor[c] = slot[c];
                if (info.texture < 0) {
                    for (int c = 0; c < 3; ++c) info.baseColor[c] *= average[(size_t)c];
                }
                for (int c = 0; c < 4; ++c) {
                    info.uvRow0[c] = slot[4 + c];
                    info.uvRow1[c] = slot[8 + c];
                }
                info.normalScale = slot[14];
                for (int c = 0; c < 4; ++c) {
                    info.normalUvRow0[c] = slot[84 + c];
                    info.normalUvRow1[c] = slot[88 + c];
                }
                info.worldMapping = slot[81];
                info.worldMappingUp = slot[82];
                info.metallic = slot[12];
                info.roughness = slot[13];
                info.unlit = slot[17];
                info.doubleSided = draw.material->doubleSided ? 1.0f : 0.0f;
                for (int c = 0; c < 3; ++c) info.emissive[c] = slot[24 + c];
                info.ior = slot[28] > 1.0f ? slot[28] : 1.5f;
                info.transmission = draw.material->alphaBlend ? slot[57] : 0.0f;
            }
            materials.push_back(info);
        }
        const double* m = draw.worldTransform.data;  // row-major
        GPU::AccelerationStructureInstance instance{};
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 4; ++c) instance.transform[r][c] = static_cast<float>(m[r * 4 + c]);
        }
        instance.instanceId = static_cast<uint32_t>(instanceInfo.size());
        const bool glass = materials[matIt->second].transmission > 0.0f;
        instance.mask = glass ? 2 : 1;
        instance.opaque = true;
        instance.blas = meshIt->second.blas;
        tlasDesc.instances.push_back(instance);

        // Normal matrix: the inverse transpose of the 3x3.
        const double a = m[0], b = m[1], c = m[2], d = m[4], e = m[5], f = m[6], g = m[8], h = m[9], k = m[10];
        const double det = a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g);
        const double inv = std::abs(det) > 1e-30 ? 1.0 / det : 0.0;
        const double cof[3][3] = {{(e * k - f * h), -(d * k - f * g), (d * h - e * g)},
                                  {-(b * k - c * h), (a * k - c * g), -(a * h - b * g)},
                                  {(b * f - c * e), -(a * f - c * d), (a * e - b * d)}};
        InstanceInfo info{};
        for (int r = 0; r < 3; ++r) {
            for (int cc = 0; cc < 3; ++cc) info.normalMatrix[r][cc] = static_cast<float>(cof[r][cc] * inv);
        }
        info.normalOffset = normalOffsets[draw.mesh];
        for (int r = 0; r < 3; ++r) {
            for (int cc = 0; cc < 3; ++cc) info.model[r][cc] = static_cast<float>(m[r * 4 + cc]);
        }
        info.material = matIt->second;
        instanceInfo.push_back(info);
    }
    if (tlasDesc.instances.empty()) return false;

    // Emissive instances' triangles: the light list (also when only
    // transforms changed: it's in world space).
    struct EmissiveDraw {
        const GpuMesh* mesh;
        uint32_t offset;
        uint32_t triangles;
        const double* transform;
        float radiance[3];
        bool doubleSided;
    };
    std::vector<EmissiveDraw> emissiveDraws;
    uint32_t emissiveTotal = 0;
    if (emissivePipeline_) {
        size_t instance = 0;
        for (const auto& draw : draws) {
            const auto meshIt = meshes_.find(draw.mesh);
            if (meshIt == meshes_.end()) continue;
            InstanceInfo& info = instanceInfo[instance++];
            const MaterialInfo& material = materials[info.material];
            const float strength = std::max(material.emissive[0], std::max(material.emissive[1], material.emissive[2]));
            if (strength <= 0.0f || material.unlit > 0.5f || material.transmission > 0.0f) continue;
            const uint32_t triangles = meshIt->second.triangleCount;
            if (emissiveTotal + triangles > MAX_EMISSIVE_TRIANGLES) continue;
            info.lightOffset = emissiveTotal + 1;
            emissiveDraws.push_back({draw.mesh, emissiveTotal, triangles, draw.worldTransform.data,
                                     {material.emissive[0], material.emissive[1], material.emissive[2]},
                                     material.doubleSided > 0.5f});
            emissiveTotal += triangles;
        }
    }
    triangles_ = triangleTotal;
    instanceCount_ = static_cast<uint32_t>(tlasDesc.instances.size());

    if (!transformsOnly) {
        normals_ = device_->createBuffer(std::max<uint64_t>(triangleTotal, 1) * 16, GPU::BufferUsage::storage);
        uvs_ = device_->createBuffer(std::max<uint64_t>(triangleTotal, 1) * 32, GPU::BufferUsage::storage);
        vertexNormals_ = device_->createBuffer(std::max<uint64_t>(triangleTotal, 1) * 48, GPU::BufferUsage::storage);
        tangents_ = device_->createBuffer(std::max<uint64_t>(triangleTotal, 1) * 16, GPU::BufferUsage::storage);
        if (!vertexNormals_ || !tangents_) return false;
        materials_ = device_->createBuffer(materials.size() * sizeof(MaterialInfo), GPU::BufferUsage::storage, materials.data());
    }
    instances_ = device_->createBuffer(instanceInfo.size() * sizeof(InstanceInfo), GPU::BufferUsage::storage, instanceInfo.data());
    tlas_ = device_->createTopLevelAccelerationStructure(tlasDesc);
    if (!normals_ || !uvs_ || !materials_ || !instances_ || !tlas_) return false;

    // Build: new mesh structures, then normals of every mesh, then the scene.
    auto encoder = device_->createCommandEncoder();
    if (!encoder) return false;
    std::vector<std::shared_ptr<GPU::Buffer>> keepAlive;
    for (auto& [blas, desc] : builds) {
        auto scratch = device_->createBuffer(std::max<uint64_t>(blas->getBuildScratchSize(), 16), GPU::BufferUsage::storage);
        keepAlive.push_back(scratch);
        encoder->buildAccelerationStructure(blas, desc, scratch);
    }
    std::vector<std::shared_ptr<GPU::BindGroup>> groups;
    // (The same meshes as before when only transforms changed: their
    // normals and coordinates are still in the buffers.)
    if (auto pass = transformsOnly ? nullptr : encoder->beginComputePass()) {
        pass->setPipeline(normalsPipeline_);
        for (const auto& [mesh, entry] : meshes_) {
            uint32_t uvStride = 8;
            const uint32_t uvKind = uvFormat(*mesh, uvStride);
            uint32_t normalStride = 12;
            const uint32_t normalKind = normalFormat(*mesh, normalStride);
            NormalParams params{entry.triangleCount, normalOffsets[mesh],
                                !mesh->indexBuffer || mesh->indexCount == 0
                                    ? 0u
                                    : (mesh->indexFormat == GPU::IndexFormat::uint16 ? 1u : 2u),
                                12u, uvKind, uvStride, normalKind, normalStride};
            auto paramsBuffer = device_->createBuffer(sizeof(params), GPU::BufferUsage::uniform, &params);
            keepAlive.push_back(paramsBuffer);
            const auto& positions = mesh->vertexBuffers[Renderer::VERTEX_SLOT_POSITION];
            const auto& indices = mesh->indexBuffer ? mesh->indexBuffer : positions;
            const auto& texcoords = uvKind != 0 ? mesh->vertexBuffers[Renderer::VERTEX_SLOT_TEXCOORD0] : positions;
            const auto& vertexNormalsIn = normalKind != 0 ? mesh->vertexBuffers[Renderer::VERTEX_SLOT_NORMAL] : positions;
            GPU::BindGroupDescriptor desc{};
            desc.layout = normalsLayout_;
            desc.entries = {{0, GPU::BufferBinding{positions, 0, positions->getLength()}},
                            {1, GPU::BufferBinding{indices, 0, indices->getLength()}},
                            {2, GPU::BufferBinding{normals_, 0, normals_->getLength()}},
                            {3, GPU::BufferBinding{paramsBuffer, 0, sizeof(params)}},
                            {4, GPU::BufferBinding{texcoords, 0, texcoords->getLength()}},
                            {5, GPU::BufferBinding{uvs_, 0, uvs_->getLength()}},
                            {6, GPU::BufferBinding{vertexNormalsIn, 0, vertexNormalsIn->getLength()}},
                            {7, GPU::BufferBinding{vertexNormals_, 0, vertexNormals_->getLength()}},
                            {8, GPU::BufferBinding{tangents_, 0, tangents_->getLength()}}};
            auto group = device_->createBindGroup(desc);
            if (!group) continue;
            groups.push_back(group);
            pass->setBindGroup(0, group, {}, 0, 0);
            // Threadgroups are the pipeline's SIMD width (8 or more threads).
            pass->dispatchWorkgroups((entry.triangleCount + 7) / 8, 1, 1);
        }
        pass->end();
    }
    {
        auto scratch = device_->createBuffer(std::max<uint64_t>(tlas_->getBuildScratchSize(), 16), GPU::BufferUsage::storage);
        keepAlive.push_back(scratch);
        encoder->buildAccelerationStructure(tlas_, tlasDesc, scratch);
    }
    std::shared_ptr<GPU::Buffer> emissives;
    if (emissiveTotal > 0) {
        emissives = device_->createBuffer(uint64_t(emissiveTotal) * sizeof(EmissiveTri),
                                          (GPU::BufferUsage)(uint32_t(GPU::BufferUsage::storage) | uint32_t(GPU::BufferUsage::copySrc)));
        if (emissives) {
            if (auto pass = encoder->beginComputePass()) {
                pass->setPipeline(emissivePipeline_);
                for (const auto& e : emissiveDraws) {
                    EmissiveParams params{};
                    params.triangleCount = e.triangles;
                    params.outOffset = e.offset;
                    params.indexFormat = !e.mesh->indexBuffer || e.mesh->indexCount == 0
                                             ? 0u
                                             : (e.mesh->indexFormat == GPU::IndexFormat::uint16 ? 1u : 2u);
                    params.vertexStride = 12;
                    for (int r = 0; r < 3; ++r) {
                        for (int c = 0; c < 4; ++c) params.model[r][c] = static_cast<float>(e.transform[r * 4 + c]);
                    }
                    for (int c = 0; c < 3; ++c) params.radiance[c] = e.radiance[c];
                    params.radiance[3] = e.doubleSided ? 1.0f : 0.0f;
                    auto paramsBuffer = device_->createBuffer(sizeof(params), GPU::BufferUsage::uniform, &params);
                    keepAlive.push_back(paramsBuffer);
                    const auto& positions = e.mesh->vertexBuffers[Renderer::VERTEX_SLOT_POSITION];
                    const auto& indices = e.mesh->indexBuffer ? e.mesh->indexBuffer : positions;
                    GPU::BindGroupDescriptor desc{};
                    desc.layout = emissiveLayout_;
                    desc.entries = {{0, GPU::BufferBinding{positions, 0, positions->getLength()}},
                                    {1, GPU::BufferBinding{indices, 0, indices->getLength()}},
                                    {2, GPU::BufferBinding{emissives, 0, emissives->getLength()}},
                                    {3, GPU::BufferBinding{paramsBuffer, 0, sizeof(params)}}};
                    auto group = device_->createBindGroup(desc);
                    if (!group) continue;
                    groups.push_back(group);
                    pass->setBindGroup(0, group, {}, 0, 0);
                    pass->dispatchWorkgroups((e.triangles + 7) / 8, 1, 1);
                }
                pass->end();
            }
        }
    }
    auto fence = device_->createFence();
    device_->submit(encoder->finish(), fence);
    if (fence) fence->wait();

    // The lights' distribution by power (luminance x area), each entry's
    // probability the step of the stored float distribution (as the sky's).
    emissiveCount_ = 0;
    emissives_.reset();
    lightTable_.reset();
    if (emissives) {
        std::vector<EmissiveTri> tris(emissiveTotal);
        if (emissives->download(0, tris.size() * sizeof(EmissiveTri), tris.data())) {
            std::vector<double> power(emissiveTotal);
            double total = 0.0;
            for (uint32_t i = 0; i < emissiveTotal; ++i) {
                const double luminance = 0.2126 * tris[i].radiance[0] + 0.7152 * tris[i].radiance[1] + 0.0722 * tris[i].radiance[2];
                power[i] = std::isfinite(luminance) && luminance > 0.0 && tris[i].radiance[3] > 0.0f ? luminance * tris[i].radiance[3] : 0.0;
                total += power[i];
            }
            if (total > 0.0) {
                std::vector<float> table(size_t(emissiveTotal) * 2);
                double running = 0.0;
                for (uint32_t i = 0; i < emissiveTotal; ++i) {
                    running += power[i];
                    table[i * 2] = static_cast<float>(running / total);
                }
                table[size_t(emissiveTotal - 1) * 2] = 1.0f;
                for (uint32_t i = 0; i < emissiveTotal; ++i) {
                    table[i * 2 + 1] = static_cast<float>(double(table[i * 2]) - (i > 0 ? double(table[(i - 1) * 2]) : 0.0));
                }
                lightTable_ = device_->createBuffer(table.size() * sizeof(float), GPU::BufferUsage::storage, table.data());
                if (lightTable_) {
                    emissives_ = emissives;
                    emissiveCount_ = emissiveTotal;
                }
            }
        }
    }
    return true;
}

bool PathTracer::sampleInFlight() const {
    return pending_ && !pending_->isSignaled();
}

void PathTracer::waitForSample() const {
    if (pending_) {
        pending_->wait();
        pending_.reset();
    }
    collectTiming();
}

void PathTracer::collectTiming() const {
    if (!pendingCommands_ || (pending_ && !pending_->isSignaled())) return;
    const uint64_t ns = pendingCommands_->getGPUExecutionTime();
    pendingCommands_.reset();
    if (ns == 0) return;  // not measured by this backend
    const double samples = std::max<uint32_t>(pendingSamples_, 1);
    lastSampleGpuMs_ = static_cast<double>(ns) * 1e-6 / samples;
    if (pendingPixels_ > 0) nsPerPixelSample_ = static_cast<double>(ns) / samples / static_cast<double>(pendingPixels_);
    imageGpuMs_ += static_cast<double>(ns) * 1e-6;
    imageTimedSamples_ += pendingSamples_;
    uint32_t count[4] = {0, 0, 0, 0};
    if (converged_ && converged_->download(0, sizeof(count), count)) {
        convergedPixels_ = count[0];
        convergedOf_ = static_cast<uint32_t>(pendingPixels_);
    }
}

PathTracer::Stats PathTracer::stats() const {
    collectTiming();
    Stats out;
    out.sampleGpuMs = lastSampleGpuMs_;
    out.averageSampleGpuMs = imageTimedSamples_ > 0 ? imageGpuMs_ / imageTimedSamples_ : 0.0;
    out.rebuildMs = rebuildMs_;
    out.denoiseMs = denoiseMs_;
    out.firstDenoisedMs = firstDenoisedMs_;
    out.triangles = triangles_;
    out.instances = instanceCount_;
    out.samplesPerRender = lastRenderSamples_;
    out.convergedPixels = convergedPixels_;
    out.tracedPixels = convergedOf_;
    out.fullRebuilds = fullRebuilds_;
    out.transformRebuilds = transformRebuilds_;
    return out;
}

void PathTracer::setPortals(std::vector<Portal> portals) {
    portals_ = std::move(portals);
    portalsDirty_ = true;
    samples_ = 0;
}

std::vector<PathTracer::Portal> PathTracer::portalsFromGlass(const RenderScene& scene, const Renderer& renderer) {
    std::vector<Portal> out;
    for (const auto& draw : scene.transparent) {
        if (!draw.mesh || !draw.material || !draw.mesh->hasBounds || !draw.material->alphaBlend) continue;
        const float* slot = renderer.materialUniforms(draw.material);
        if (!slot || slot[57] <= 0.0f) continue;  // not transmissive: not glass to the tracer
        const auto& lo = draw.mesh->boundsMin;
        const auto& hi = draw.mesh->boundsMax;
        const double extent[3] = {double(hi[0]) - lo[0], double(hi[1]) - lo[1], double(hi[2]) - lo[2]};
        int thin = 0;
        for (int a = 1; a < 3; ++a) {
            if (extent[a] < extent[thin]) thin = a;
        }
        const int a1 = (thin + 1) % 3, a2 = (thin + 2) % 3;
        if (extent[a1] <= 0.0 || extent[a2] <= 0.0 || extent[thin] > 0.1 * std::min(extent[a1], extent[a2])) continue;
        // The rectangle through the middle of the bounds, to world space
        // (row-major: world = M * p).
        double corner[3] = {lo[0], lo[1], lo[2]};
        corner[thin] = 0.5 * (double(lo[thin]) + hi[thin]);
        double e1[3] = {0, 0, 0}, e2[3] = {0, 0, 0};
        e1[a1] = extent[a1];
        e2[a2] = extent[a2];
        const double* m = draw.worldTransform.data;
        const auto point = [&](const double* p) {
            return VM::Vector3<double>(m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[3],
                                       m[4] * p[0] + m[5] * p[1] + m[6] * p[2] + m[7],
                                       m[8] * p[0] + m[9] * p[1] + m[10] * p[2] + m[11]);
        };
        const auto vector = [&](const double* v) {
            return VM::Vector3<double>(m[0] * v[0] + m[1] * v[1] + m[2] * v[2], m[4] * v[0] + m[5] * v[1] + m[6] * v[2],
                                       m[8] * v[0] + m[9] * v[1] + m[10] * v[2]);
        };
        out.push_back(Portal{point(corner), vector(e1), vector(e2)});
    }
    return out;
}

bool PathTracer::buildEnvironmentTable(const std::shared_ptr<GPU::Texture>& environment) {
    envTableSource_ = environment.get();
    envResolution_ = 0;
    if (!envPipeline_ || !environment) return false;
    constexpr uint32_t R = ENV_RESOLUTION;
    constexpr uint32_t count = 6 * R * R;
    auto weights = device_->createBuffer(count * sizeof(float), (GPU::BufferUsage)(uint32_t(GPU::BufferUsage::storage) |
                                                                                    uint32_t(GPU::BufferUsage::copySrc)));
    EnvParams params{R, {0, 0, 0}};
    auto paramsBuffer = device_->createBuffer(sizeof(params), GPU::BufferUsage::uniform, &params);
    if (!weights || !paramsBuffer) return false;
    GPU::BindGroupDescriptor desc{};
    desc.layout = envLayout_;
    desc.entries = {{0, GPU::BufferBinding{weights, 0, weights->getLength()}},
                    {1, GPU::BufferBinding{paramsBuffer, 0, sizeof(params)}},
                    {0, environment}};
    auto group = device_->createBindGroup(desc);
    auto encoder = device_->createCommandEncoder();
    if (!group || !encoder) return false;
    if (auto pass = encoder->beginRayTracingPass()) {
        pass->setPipeline(envPipeline_);
        pass->setBindGroup(0, group);
        pass->traceRays(6 * R, R, 1);
        pass->end();
    }
    auto fence = device_->createFence();
    device_->submit(encoder->finish(), fence);
    if (fence) fence->wait();
    std::vector<float> w(count);
    if (!weights->download(0, count * sizeof(float), w.data())) return false;

    // Probabilities, with a floor so no texel is impossible (the BSDF's
    // directions cover the sky regardless: this only keeps the light
    // samples' densities finite).
    double total = 0.0;
    for (float& x : w) {
        if (!std::isfinite(x) || x < 0.0f) x = 0.0f;
        total += x;
    }
    if (total <= 0.0) return false;  // a black sky: nothing to sample
    const double floor = total / count * 1e-3;
    total = 0.0;
    for (float& x : w) {
        x = static_cast<float>(std::max<double>(x, floor));
        total += x;
    }

    // The cumulative distribution, inverted by a binary search in the
    // shader (sampleEnvironment()). Each texel's probability is the step
    // of the stored (float) distribution, so the densities the shader
    // reports are exactly those it samples with, even where rounding
    // leaves a texel no step (then never picked, density 0: the BSDF's
    // directions cover it alone).
    std::vector<EnvTexel> table(count);
    double running = 0.0;
    for (uint32_t i = 0; i < count; ++i) {
        running += w[i];
        table[i].cdf = static_cast<float>(running / total);
    }
    table[count - 1].cdf = 1.0f;  // whatever the rounding: every u < 1 finds a texel
    for (uint32_t i = 0; i < count; ++i) {
        table[i].pdf = static_cast<float>(double(table[i].cdf) - (i > 0 ? double(table[i - 1].cdf) : 0.0));
    }
    envTable_ = device_->createBuffer(table.size() * sizeof(EnvTexel), GPU::BufferUsage::storage, table.data());
    if (!envTable_) return false;
    envResolution_ = R;
    return true;
}

bool PathTracer::render(const RenderScene& scene, const Renderer& renderer, uint32_t width, uint32_t height) {
    if (!supported() || width == 0 || height == 0) return false;
    // The previous sample (usually done by now, a frame later): its
    // uniforms and buffers are rewritten below.
    waitForSample();

    // What the image depends on besides the camera, by the work a change
    // needs: the draws' meshes and materials (a full rebuild), their
    // transforms (the instances and the scene's structure), and the
    // lights and lighting settings (only a new image). Exposure isn't
    // among them: it applies to the average when it's shown.
    uint64_t geometryKey = 1469598103934665603ull;
    uint64_t transformKey = 1469598103934665603ull;
    for (const auto& draw : traced(scene)) {
        mix(geometryKey, &draw.mesh, sizeof(draw.mesh));
        const void* positions = draw.mesh->vertexBuffers[Renderer::VERTEX_SLOT_POSITION].get();
        mix(geometryKey, &positions, sizeof(positions));
        const void* indices = draw.mesh->indexBuffer.get();
        mix(geometryKey, &indices, sizeof(indices));
        mix(geometryKey, &draw.mesh->indexCount, sizeof(draw.mesh->indexCount));
        mix(geometryKey, &draw.material, sizeof(draw.material));
        const void* textures[3] = {draw.material->baseColorTexture.get(), draw.material->normalTexture.get(),
                                   draw.material->metallicRoughnessTexture.get()};
        mix(geometryKey, textures, sizeof(textures));
        if (const float* slot = renderer.materialUniforms(draw.material)) mix(geometryKey, slot, 93 * sizeof(float));
        mix(transformKey, draw.worldTransform.data, sizeof(double) * 16);
    }
    uint64_t lightingKey = 1469598103934665603ull;
    for (const auto& light : scene.lights) {
        const double values[7] = {light.direction.x(), light.direction.y(), light.direction.z(),
                                  light.color.x(),     light.color.y(),     light.color.z(), light.intensity};
        mix(lightingKey, values, sizeof(values));
    }
    float clear[4];
    renderer.getClearColor(&clear[0], &clear[1], &clear[2], &clear[3]);
    const float settings[5] = {renderer.getEnvironmentIntensity(), clear[0], clear[1], clear[2],
                               renderer.getSkyboxEnabled() ? 1.0f : 0.0f};
    mix(lightingKey, settings, sizeof(settings));
    const auto environment = renderer.getEnvironmentMap();
    const void* envPtr = environment.get();
    mix(lightingKey, &envPtr, sizeof(envPtr));
    mix(lightingKey, &maxBounces_, sizeof(maxBounces_));

    uint64_t viewKey = 1469598103934665603ull;
    mix(viewKey, scene.camera.view.data, sizeof(double) * 16);
    mix(viewKey, scene.camera.projection.data, sizeof(double) * 16);
    mix(viewKey, &width, sizeof(width));
    mix(viewKey, &height, sizeof(height));
    mix(viewKey, &projection_, sizeof(projection_));

    if (!tlas_ || geometryKey != geometryKey_ || transformKey != transformKey_) {
        const bool transformsOnly = tlas_ && geometryKey == geometryKey_;
        const auto start = std::chrono::steady_clock::now();
        if (!rebuild(scene, renderer, transformsOnly)) {
            tlas_.reset();  // a full rebuild next time
            return false;
        }
        rebuildMs_ = millisecondsSince(start);
        geometryKey_ = geometryKey;
        transformKey_ = transformKey;
        ++(transformsOnly ? transformRebuilds_ : fullRebuilds_);
        samples_ = 0;
    }
    if (lightingKey != lightingKey_) {
        lightingKey_ = lightingKey;
        samples_ = 0;
    }
    if (viewKey != viewKey_) {
        viewKey_ = viewKey;
        samples_ = 0;
    }
    if (!environment) return false;
    if (environmentSampling_ && environment.get() != envTableSource_) {
        const auto start = std::chrono::steady_clock::now();
        buildEnvironmentTable(environment);
        rebuildMs_ += millisecondsSince(start);
        samples_ = 0;
    }
    if (width != width_ || height != height_ || !accum_ || !output_) {
        const auto readable = (GPU::BufferUsage)(uint32_t(GPU::BufferUsage::storage) | uint32_t(GPU::BufferUsage::copyDst) |
                                                 uint32_t(GPU::BufferUsage::copySrc));
        accum_ = device_->createBuffer((uint64_t)width * height * 16, readable);
        aux_ = device_->createBuffer((uint64_t)width * height * 32, readable);
        moments_ = device_->createBuffer((uint64_t)width * height * 4, readable);
        output_ = device_->createTexture(GPU::TextureType::tt2d, GPU::PixelFormat::rgba8unorm, width, height, 1, 1, 1,
                                         (GPU::TextureUsage)(uint32_t(GPU::TextureUsage::storageBinding) |
                                                             uint32_t(GPU::TextureUsage::textureBinding) |
                                                             uint32_t(GPU::TextureUsage::copySrc) |
                                                             uint32_t(GPU::TextureUsage::copyDst)));
        width_ = width;
        height_ = height;
        samples_ = 0;
        if (!accum_ || !aux_ || !moments_ || !output_) return false;
    }

    // The preview at reduced resolution (setPreviewResolution()): an
    // image's first samples, then it starts again at full resolution.
    const bool newImage = samples_ == 0;
    if (newImage) {
        previewActive_ = previewSamples_ > 0 && previewDivisor_ > 1;
    } else if (previewActive_ && samples_ >= previewSamples_) {
        previewActive_ = false;
        samples_ = 0;
    }
    const uint32_t divisor = previewActive_ ? previewDivisor_ : 1;
    traceWidth_ = (width + divisor - 1) / divisor;
    traceHeight_ = (height + divisor - 1) / divisor;

    // Samples in this dispatch: the fixed count, or as many as the budget
    // holds at the last measured cost; never past the end of the preview
    // or of the first samples' fewer bounces (they change per phase).
    uint32_t samplesNow = samplesPerRender_;
    if (frameBudgetMs_ > 0.0 && nsPerPixelSample_ > 0.0) {
        const double perSample = nsPerPixelSample_ * double(traceWidth_) * double(traceHeight_) * 1e-6;  // ms
        samplesNow = static_cast<uint32_t>(std::clamp(std::floor(frameBudgetMs_ / perSample), 1.0, 64.0));
    }
    if (previewActive_) samplesNow = std::min(samplesNow, previewSamples_ - samples_);
    if (samples_ < progressiveSamples_) samplesNow = std::min(samplesNow, progressiveSamples_ - samples_);
    samplesNow = std::max(samplesNow, 1u);

    TraceUniforms u{};
    const auto invViewProj = (scene.camera.projection * scene.camera.view).inverted();
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) u.invViewProj[c * 4 + r] = static_cast<float>(invViewProj.data[r * 4 + c]);
    }
    {
        // The eye from the view matrix (not CameraData::position, which a
        // caller may leave unset): camera rays start there.
        const auto eye = scene.camera.view.inverted() * VM::Vector4<double>(0.0, 0.0, 0.0, 1.0);
        u.cameraPos[0] = static_cast<float>(eye.x() / eye.w());
        u.cameraPos[1] = static_cast<float>(eye.y() / eye.w());
        u.cameraPos[2] = static_cast<float>(eye.z() / eye.w());
    }
    uint32_t lights = 0;
    for (const auto& light : scene.lights) {
        if (light.type != 0 || lights == 4) continue;  // directional lights, the first four
        const double len = std::sqrt(light.direction.x() * light.direction.x() + light.direction.y() * light.direction.y() +
                                     light.direction.z() * light.direction.z());
        if (len < 1e-12 || light.intensity <= 0.0f) continue;
        u.sunDirection[lights][0] = static_cast<float>(-light.direction.x() / len);
        u.sunDirection[lights][1] = static_cast<float>(-light.direction.y() / len);
        u.sunDirection[lights][2] = static_cast<float>(-light.direction.z() / len);
        u.sunDirection[lights][3] = std::cos(0.265f * 3.14159265f / 180.0f);  // the sun's disc
        u.sunRadiance[lights][0] = static_cast<float>(light.color.x()) * light.intensity;
        u.sunRadiance[lights][1] = static_cast<float>(light.color.y()) * light.intensity;
        u.sunRadiance[lights][2] = static_cast<float>(light.color.z()) * light.intensity;
        u.sunRadiance[lights][3] = 1.0f;
        ++lights;
    }
    u.params[0] = renderer.getEnvironmentIntensity();
    u.params[1] = renderer.getExposure();
    // Fewer bounces for the first samples (setProgressiveBounces()).
    const uint32_t bounces = samples_ < progressiveSamples_ ? std::min(progressiveBounces_, maxBounces_) : maxBounces_;
    u.params[2] = static_cast<float>(bounces);
    u.params[3] = static_cast<float>(samples_);
    u.size[0] = traceWidth_;
    u.size[1] = traceHeight_;
    u.size[2] = renderer.getSkyboxEnabled() ? 1u : 0u;
    u.size[3] = projection_ == Projection::panorama ? 1u : 0u;
    {
        // The camera's directions from its projection: straight ahead, and
        // towards the image's right and top edges (no handedness assumed).
        const auto unproject = [&](double x, double y) {
            const auto p = invViewProj * VM::Vector4<double>(x, y, 1.0, 1.0);
            const auto q = invViewProj * VM::Vector4<double>(x, y, 0.0, 1.0);
            return VM::Vector3<double>(p.x() / p.w() - q.x() / q.w(), p.y() / p.w() - q.y() / q.w(),
                                       p.z() / p.w() - q.z() / q.w());
        };
        const auto unit = [](VM::Vector3<double> v) {
            const double len = std::sqrt(v.x() * v.x() + v.y() * v.y() + v.z() * v.z());
            return len > 1e-30 ? VM::Vector3<double>(v.x() / len, v.y() / len, v.z() / len) : v;
        };
        const auto dot = [](const VM::Vector3<double>& a, const VM::Vector3<double>& b) {
            return a.x() * b.x() + a.y() * b.y() + a.z() * b.z();
        };
        const auto forward = unit(unproject(0.0, 0.0));
        auto up = unit(unproject(0.0, 0.5));
        up = unit(VM::Vector3<double>(up.x() - forward.x() * dot(up, forward), up.y() - forward.y() * dot(up, forward),
                                      up.z() - forward.z() * dot(up, forward)));
        auto right = unit(unproject(0.5, 0.0));
        right = unit(VM::Vector3<double>(right.x() - forward.x() * dot(right, forward) - up.x() * dot(right, up),
                                         right.y() - forward.y() * dot(right, forward) - up.y() * dot(right, up),
                                         right.z() - forward.z() * dot(right, forward) - up.z() * dot(right, up)));
        const auto store = [](float* out, const VM::Vector3<double>& v) {
            out[0] = static_cast<float>(v.x());
            out[1] = static_cast<float>(v.y());
            out[2] = static_cast<float>(v.z());
        };
        store(u.cameraForward, forward);
        store(u.cameraRight, right);
        store(u.cameraUp, up);
    }
    for (int c = 0; c < 3; ++c) u.clearColor[c] = std::pow(std::max(clear[c], 0.0f), 2.2f);  // display -> linear
    u.tuning[0] = indirectClamp_;
    u.tuning[2] = noiseTarget_;
    u.tuning[3] = static_cast<float>(noiseMinSamples_);
    if (!converged_) {
        converged_ = device_->createBuffer(16, (GPU::BufferUsage)(uint32_t(GPU::BufferUsage::storage) |
                                                                  uint32_t(GPU::BufferUsage::copySrc) |
                                                                  uint32_t(GPU::BufferUsage::copyDst)));
        if (!converged_) return false;
    }
    {
        uint32_t zero[4] = {0, 0, 0, 0};  // (the last sample is done: waited for above)
        converged_->upload(0, sizeof(zero), zero);
    }
    u.tuning[1] = sampler_ == Sampler::sobol ? 1.0f : 0.0f;
    const bool envSampled = environmentSampling_ && envResolution_ > 0 && envTable_;
    u.envSampling[0] = envSampled ? envResolution_ : 0u;
    if (portalsDirty_) {
        // Rectangles: unit, perpendicular axes and their lengths.
        std::vector<PortalInfo> infos;
        for (const auto& portal : portals_) {
            if (infos.size() == MAX_PORTALS) break;
            const double e1[3] = {portal.edge1.x(), portal.edge1.y(), portal.edge1.z()};
            double e2[3] = {portal.edge2.x(), portal.edge2.y(), portal.edge2.z()};
            const double l1 = std::sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]);
            if (l1 < 1e-9) continue;
            const double x[3] = {e1[0] / l1, e1[1] / l1, e1[2] / l1};
            const double along = e2[0] * x[0] + e2[1] * x[1] + e2[2] * x[2];
            for (int c = 0; c < 3; ++c) e2[c] -= x[c] * along;
            const double l2 = std::sqrt(e2[0] * e2[0] + e2[1] * e2[1] + e2[2] * e2[2]);
            if (l2 < 1e-9) continue;
            PortalInfo info{};
            info.corner[0] = static_cast<float>(portal.corner.x());
            info.corner[1] = static_cast<float>(portal.corner.y());
            info.corner[2] = static_cast<float>(portal.corner.z());
            for (int c = 0; c < 3; ++c) {
                info.axisX[c] = static_cast<float>(x[c]);
                info.axisY[c] = static_cast<float>(e2[c] / l2);
            }
            info.axisX[3] = static_cast<float>(l1);
            info.axisY[3] = static_cast<float>(l2);
            infos.push_back(info);
        }
        portalBuffer_ = infos.empty() ? nullptr
                                      : device_->createBuffer(infos.size() * sizeof(PortalInfo), GPU::BufferUsage::storage,
                                                              infos.data());
        portalCount_ = portalBuffer_ ? static_cast<uint32_t>(infos.size()) : 0u;
        portalsDirty_ = false;
    }
    // Portals aim the sky's light samples: only with them on.
    const uint32_t portalCount = environmentSampling_ ? portalCount_ : 0u;
    u.envSampling[1] = portalCount;
    u.envSampling[2] = divisor;
    u.envSampling[3] = samplesNow;
    u.lights[0] = emissiveSampling_ ? emissiveCount_ : 0u;
    u.camera[0] = projection_ == Projection::perspective ? apertureRadius_ : 0.0f;
    u.camera[1] = focusDistance_;
    u.camera[2] = static_cast<float>(toneMapping_);
    if (!uniforms_) uniforms_ = device_->createBuffer(sizeof(TraceUniforms), GPU::BufferUsage::uniform);
    if (!uniforms_) return false;
    uniforms_->upload(0, sizeof(u), &u);

    GPU::BindGroupDescriptor desc{};
    desc.layout = traceLayout_;
    desc.entries = {{0, tlas_},
                    {1, GPU::BufferBinding{accum_, 0, accum_->getLength()}},
                    {2, GPU::BufferBinding{uniforms_, 0, sizeof(TraceUniforms)}},
                    {3, GPU::BufferBinding{instances_, 0, instances_->getLength()}},
                    {4, GPU::BufferBinding{materials_, 0, materials_->getLength()}},
                    {5, GPU::BufferBinding{normals_, 0, normals_->getLength()}},
                    {6, GPU::BufferBinding{aux_, 0, aux_->getLength()}},
                    {7, GPU::BufferBinding{uvs_, 0, uvs_->getLength()}},
                    {8, envSampled ? GPU::BufferBinding{envTable_, 0, envTable_->getLength()}
                                   : GPU::BufferBinding{uvs_, 0, uvs_->getLength()}},  // unused without a table
                    {9, portalCount > 0 ? GPU::BufferBinding{portalBuffer_, 0, portalBuffer_->getLength()}
                                        : GPU::BufferBinding{uvs_, 0, uvs_->getLength()}},  // unused without portals
                    {10, GPU::BufferBinding{moments_, 0, moments_->getLength()}},
                    {11, GPU::BufferBinding{converged_, 0, converged_->getLength()}},
                    {12, GPU::BufferBinding{vertexNormals_, 0, vertexNormals_->getLength()}},
                    {13, GPU::BufferBinding{tangents_, 0, tangents_->getLength()}},
                    {14, emissiveCount_ > 0 ? GPU::BufferBinding{emissives_, 0, emissives_->getLength()}
                                            : GPU::BufferBinding{uvs_, 0, uvs_->getLength()}},  // unused without emitters
                    {15, emissiveCount_ > 0 ? GPU::BufferBinding{lightTable_, 0, lightTable_->getLength()}
                                            : GPU::BufferBinding{uvs_, 0, uvs_->getLength()}},
                    {0, output_},
                    {1, environment}};
    // Base colour textures; the rest of the bindings a 1x1 white texture.
    if (!blank_) {
        uint8_t white[4] = {255, 255, 255, 255};
        blank_ = device_->createTexture(GPU::TextureType::tt2d, GPU::PixelFormat::rgba8unorm, 1, 1, 1, 1, 1,
                                        (GPU::TextureUsage)(uint32_t(GPU::TextureUsage::textureBinding) |
                                                            uint32_t(GPU::TextureUsage::copyDst)));
        if (blank_) blank_->upload(0, 4, white);
    }
    for (uint32_t i = 0; i < MAX_TEXTURES; ++i) {
        const auto& texture = i < textures_.size() ? textures_[i] : blank_;
        if (texture) desc.entries.push_back({FIRST_TEXTURE + i, texture});
    }
    auto group = device_->createBindGroup(desc);
    auto encoder = device_->createCommandEncoder();
    if (!group || !encoder) return false;
    if (auto pass = encoder->beginRayTracingPass()) {
        pass->setPipeline(pipeline_);
        pass->setBindGroup(0, group);
        pass->traceRays(traceWidth_, traceHeight_, 1);
        pass->end();
    }
    // Not waited for here: the GPU traces while the caller goes on (the next
    // render() and the readers wait). Work submitted later on the same queue
    // (e.g. copying output()) runs after it.
    if (newImage) {
        // A new image: its timings start here.
        imageGpuMs_ = 0.0;
        imageTimedSamples_ = 0;
        firstDenoisedMs_ = 0.0;
        imageStartNs_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count();
    }
    auto fence = device_->createFence();
    auto commands = encoder->finish();
    device_->submit(commands, fence);
    pending_ = fence;
    pendingCommands_ = commands;
    pendingSamples_ = samplesNow;
    pendingPixels_ = uint64_t(traceWidth_) * traceHeight_;
    lastRenderSamples_ = samplesNow;
    samples_ += samplesNow;
    exposure_ = renderer.getExposure();
    denoised_ = false;
    return true;
}

bool PathTracer::readOutput(std::vector<uint8_t>& rgba) const {
    if (!output_ || samples_ == 0) return false;
    waitForSample();
    rgba.resize((size_t)width_ * height_ * 4);
    device_->waitForIdle();
    return output_->download(0, 0, rgba.data(), rgba.size());
}

bool PathTracer::readLinear(std::vector<float>& rgb, std::vector<float>* albedo, std::vector<float>* normal) const {
    if (!readTraced(rgb, albedo, normal)) return false;
    if (traceWidth_ != width_ || traceHeight_ != height_) {
        // A preview: each traced pixel covers a block of the image.
        std::vector<float> small;
        for (std::vector<float>* image : {&rgb, albedo, normal}) {
            if (!image) continue;
            small.swap(*image);
            upscale(small, traceWidth_, traceHeight_, *image, width_, height_, false);
        }
    }
    return true;
}

bool PathTracer::readTraced(std::vector<float>& rgb, std::vector<float>* albedo, std::vector<float>* normal) const {
    if (!accum_ || !aux_ || samples_ == 0) return false;
    waitForSample();
    const size_t pixels = (size_t)traceWidth_ * traceHeight_;
    std::vector<float> sums(pixels * 4);
    if (!accum_->download(0, sums.size() * 4, sums.data())) return false;
    const float scale = std::exp2(exposure_);
    rgb.resize(pixels * 3);
    for (size_t i = 0; i < pixels; ++i) {
        const float n = std::max(sums[i * 4 + 3], 1.0f);
        for (int c = 0; c < 3; ++c) rgb[i * 3 + c] = sums[i * 4 + c] / n * scale;
    }
    if (albedo || normal) {
        std::vector<float> aux(pixels * 8);
        if (!aux_->download(0, aux.size() * 4, aux.data())) return false;
        // Each pixel's own count (w): adaptive sampling stops pixels early.
        if (albedo) {
            albedo->resize(pixels * 3);
            for (size_t i = 0; i < pixels; ++i) {
                const float n = std::max(aux[i * 4 + 3], 1.0f);
                for (int c = 0; c < 3; ++c) (*albedo)[i * 3 + c] = aux[i * 4 + c] / n;
            }
        }
        if (normal) {
            normal->resize(pixels * 3);
            for (size_t i = 0; i < pixels; ++i) {
                const float n = std::max(aux[(pixels + i) * 4 + 3], 1.0f);
                for (int c = 0; c < 3; ++c) (*normal)[i * 3 + c] = aux[(pixels + i) * 4 + c] / n;
            }
        }
    }
    return true;
}

float PathTracer::exposureForCamera(float fNumber, float shutterSeconds, float iso) noexcept {
    if (fNumber <= 0.0f || shutterSeconds <= 0.0f || iso <= 0.0f) return 0.0f;
    const float ev100 = std::log2(fNumber * fNumber / shutterSeconds * 100.0f / iso);
    return -ev100 - std::log2(1.2f);  // scale 1 / (1.2 * 2^EV100)
}

namespace {
// As toneMapAces() and toneMapAgx() in path_tracer.metal (linear in, linear out).
void toneMapAces(float* c) {
    const float i[3][3] = {{0.59719f, 0.35458f, 0.04823f}, {0.07600f, 0.90834f, 0.01566f}, {0.02840f, 0.13383f, 0.83777f}};
    const float o[3][3] = {{1.60475f, -0.53108f, -0.07367f}, {-0.10208f, 1.10813f, -0.00605f}, {-0.00327f, -0.07276f, 1.07602f}};
    float v[3];
    for (int r = 0; r < 3; ++r) {
        const float x = i[r][0] * c[0] + i[r][1] * c[1] + i[r][2] * c[2];
        v[r] = (x * (x + 0.0245786f) - 0.000090537f) / (x * (0.983729f * x + 0.4329510f) + 0.238081f);
    }
    for (int r = 0; r < 3; ++r) c[r] = std::clamp(o[r][0] * v[0] + o[r][1] * v[1] + o[r][2] * v[2], 0.0f, 1.0f);
}
void toneMapAgx(float* c) {
    const float i[3][3] = {{0.842479062253094f, 0.0784335999999992f, 0.0792237451477643f},
                           {0.0423282422610123f, 0.878468636469772f, 0.0791661274605434f},
                           {0.0423756549057051f, 0.0784336f, 0.879142973793104f}};
    const float o[3][3] = {{1.19687900512017f, -0.0980208811401368f, -0.0990297440797205f},
                           {-0.0528968517574562f, 1.15190312990417f, -0.0989611768448433f},
                           {-0.0529716355144438f, -0.0980434501171241f, 1.15107367264116f}};
    const float minEv = -12.47393f, maxEv = 4.026069f;
    float v[3];
    for (int r = 0; r < 3; ++r) {
        float x = i[r][0] * c[0] + i[r][1] * c[1] + i[r][2] * c[2];
        x = (std::clamp(std::log2(std::max(x, 1e-10f)), minEv, maxEv) - minEv) / (maxEv - minEv);
        const float x2 = x * x, x4 = x2 * x2;
        v[r] = 15.5f * x4 * x2 - 40.14f * x4 * x + 31.96f * x4 - 6.868f * x2 * x + 0.4298f * x2 + 0.1191f * x - 0.00232f;
    }
    for (int r = 0; r < 3; ++r) {
        c[r] = std::pow(std::clamp(o[r][0] * v[0] + o[r][1] * v[1] + o[r][2] * v[2], 0.0f, 1.0f), 2.2f);
    }
}
} // namespace

void PathTracer::toneMap(const float* rgb, size_t pixels, uint8_t* rgba, ToneMapping toneMapping) {
    // As the shader's tone mapping (path_tracer.metal).
    for (size_t i = 0; i < pixels; ++i) {
        float c[3] = {std::max(rgb[i * 3], 0.0f), std::max(rgb[i * 3 + 1], 0.0f), std::max(rgb[i * 3 + 2], 0.0f)};
        if (toneMapping != ToneMapping::pbrNeutral) {
            if (toneMapping == ToneMapping::aces) toneMapAces(c);
            else toneMapAgx(c);
            for (int k = 0; k < 3; ++k) {
                const float display = std::pow(std::max(c[k], 0.0f), 1.0f / 2.2f);
                rgba[i * 4 + k] = static_cast<uint8_t>(std::lround(std::clamp(display, 0.0f, 1.0f) * 255.0f));
            }
            rgba[i * 4 + 3] = 255;
            continue;
        }
        const float startCompression = 0.8f - 0.04f;
        const float desaturation = 0.15f;
        const float x = std::min(c[0], std::min(c[1], c[2]));
        const float offset = x < 0.08f ? x - 6.25f * x * x : 0.04f;
        for (float& v : c) v -= offset;
        const float peak = std::max(c[0], std::max(c[1], c[2]));
        if (peak >= startCompression) {
            const float d = 1.0f - startCompression;
            const float newPeak = 1.0f - d * d / (peak + d - startCompression);
            const float g = 1.0f - 1.0f / (desaturation * (peak - newPeak) + 1.0f);
            for (float& v : c) {
                v *= newPeak / peak;
                v = v + (newPeak - v) * g;
            }
        }
        for (int k = 0; k < 3; ++k) {
            const float display = std::pow(std::max(c[k], 0.0f), 1.0f / 2.2f);
            rgba[i * 4 + k] = static_cast<uint8_t>(std::lround(std::clamp(display, 0.0f, 1.0f) * 255.0f));
        }
        rgba[i * 4 + 3] = 255;
    }
}

#if defined(CAMPELLO_RENDERER_DENOISER)
namespace {
struct DenoiserState {
    std::string type;  // "cpu", "metal", ...
    oidn::DeviceRef device;
    oidn::FilterRef filter;
    oidn::FilterRef albedoFilter, normalFilter;  // the guides' prefilters (setDenoiserPrefilter())
    oidn::BufferRef color, albedo, normal, output;
    uint32_t width = 0;
    uint32_t height = 0;
    int quality = -1;  // the filters' settings
    bool prefilter = false;
};
const char* deviceTypeName(int type) {
    switch (type) {
        case OIDN_DEVICE_TYPE_CPU: return "cpu";
        case OIDN_DEVICE_TYPE_SYCL: return "sycl";
        case OIDN_DEVICE_TYPE_CUDA: return "cuda";
        case OIDN_DEVICE_TYPE_HIP: return "hip";
        case OIDN_DEVICE_TYPE_METAL: return "metal";
        default: return "unknown";
    }
}

/// A committed device: the first GPU one the library has (unless `want`
/// is cpu), else the CPU (unless `want` is gpu). Null when there's none.
oidn::DeviceRef newDenoiserDevice(PathTracer::DenoiserDevice want, std::string& type) {
    if (want != PathTracer::DenoiserDevice::cpu) {
        const int count = oidn::getNumPhysicalDevices();
        for (int i = 0; i < count; ++i) {
            oidn::PhysicalDeviceRef physical(i);
            const int kind = physical.get<int>("type");
            if (kind == OIDN_DEVICE_TYPE_CPU) continue;
            oidn::DeviceRef device = physical.newDevice();
            if (!device) continue;
            device.commit();
            const char* message = nullptr;
            if (device.getError(message) != oidn::Error::None) continue;
            type = deviceTypeName(kind);
            return device;
        }
        if (want == PathTracer::DenoiserDevice::gpu) return {};
    }
    oidn::DeviceRef device = oidn::newDevice(oidn::DeviceType::CPU);
    if (!device) return {};
    device.commit();
    const char* message = nullptr;
    if (device.getError(message) != oidn::Error::None) return {};
    type = "cpu";
    return device;
}
} // namespace
#endif

void PathTracer::setDenoiserDevice(DenoiserDevice device) {
    if (device != denoiserDevice_) denoiser_.reset();  // made again on the next denoise()
    denoiserDevice_ = device;
}

std::string PathTracer::denoiserDeviceType() const {
#if defined(CAMPELLO_RENDERER_DENOISER)
    if (const auto* state = static_cast<const DenoiserState*>(denoiser_.get())) return state->type;
#endif
    return {};
}

bool PathTracer::denoiserAvailable() noexcept {
#if defined(CAMPELLO_RENDERER_DENOISER)
    return true;
#else
    return false;
#endif
}

bool PathTracer::denoise() {
#if defined(CAMPELLO_RENDERER_DENOISER)
    const auto start = std::chrono::steady_clock::now();
    // At the traced size (a preview's is smaller), scaled up after.
    std::vector<float> color, albedo, normal;
    if (!readTraced(color, &albedo, &normal)) return false;
    const uint32_t w = traceWidth_, h = traceHeight_;
    try {
        // A GPU device where the library has one, else the CPU
        // (setDenoiserDevice()); no sharing with campello_gpu's device. Its
        // filter and buffers are kept for the next image of the same size:
        // committing a filter loads its weights.
        auto* state = static_cast<DenoiserState*>(denoiser_.get());
        if (!state) {
            auto created = std::make_shared<DenoiserState>();
            created->device = newDenoiserDevice(denoiserDevice_, created->type);
            if (!created->device) return false;
            denoiser_ = created;
            state = created.get();
        }
        const size_t bytes = color.size() * sizeof(float);
        const int quality = denoiserQuality_ == DenoiserQuality::fast       ? int(oidn::Quality::Fast)
                            : denoiserQuality_ == DenoiserQuality::balanced ? int(oidn::Quality::Balanced)
                                                                            : int(oidn::Quality::High);
        if (state->width != w || state->height != h || state->quality != quality || state->prefilter != denoiserPrefilter_) {
            if (state->width != w || state->height != h || !state->color) {
                state->color = state->device.newBuffer(bytes);
                state->albedo = state->device.newBuffer(bytes);
                state->normal = state->device.newBuffer(bytes);
                state->output = state->device.newBuffer(bytes);
            }
            state->albedoFilter = {};
            state->normalFilter = {};
            if (denoiserPrefilter_) {
                // The guides denoised in place, each by itself.
                state->albedoFilter = state->device.newFilter("RT");
                state->albedoFilter.setImage("albedo", state->albedo, oidn::Format::Float3, w, h);
                state->albedoFilter.setImage("output", state->albedo, oidn::Format::Float3, w, h);
                state->albedoFilter.set("quality", quality);
                state->albedoFilter.commit();
                state->normalFilter = state->device.newFilter("RT");
                state->normalFilter.setImage("normal", state->normal, oidn::Format::Float3, w, h);
                state->normalFilter.setImage("output", state->normal, oidn::Format::Float3, w, h);
                state->normalFilter.set("quality", quality);
                state->normalFilter.commit();
            }
            state->filter = state->device.newFilter("RT");
            state->filter.setImage("color", state->color, oidn::Format::Float3, w, h);
            state->filter.setImage("albedo", state->albedo, oidn::Format::Float3, w, h);
            state->filter.setImage("normal", state->normal, oidn::Format::Float3, w, h);
            state->filter.setImage("output", state->output, oidn::Format::Float3, w, h);
            state->filter.set("hdr", true);
            state->filter.set("quality", quality);
            state->filter.set("cleanAux", denoiserPrefilter_);
            state->filter.commit();
            state->width = w;
            state->height = h;
            state->quality = quality;
            state->prefilter = denoiserPrefilter_;
        }
        state->color.write(0, bytes, color.data());
        state->albedo.write(0, bytes, albedo.data());
        state->normal.write(0, bytes, normal.data());
        if (state->albedoFilter) state->albedoFilter.execute();
        if (state->normalFilter) state->normalFilter.execute();
        state->filter.execute();
        const char* message = nullptr;
        if (state->device.getError(message) != oidn::Error::None) {
            denoiser_.reset();  // start afresh next time
            return false;
        }
        state->output.read(0, bytes, color.data());
    } catch (...) {
        denoiser_.reset();
        return false;
    }
    if (w != width_ || h != height_) {
        std::vector<float> small;
        small.swap(color);
        upscale(small, w, h, color, width_, height_, true);
    }
    std::vector<uint8_t> rgba((size_t)width_ * height_ * 4);
    toneMap(color.data(), (size_t)width_ * height_, rgba.data(), toneMapping_);
    device_->waitForIdle();
    if (!output_->upload(0, rgba.size(), rgba.data())) return false;
    denoised_ = true;
    denoiseMs_ = millisecondsSince(start);
    if (firstDenoisedMs_ == 0.0) {
        const int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count();
        firstDenoisedMs_ = static_cast<double>(now - imageStartNs_) * 1e-6;
    }
    return true;
#else
    return false;
#endif
}

} // namespace systems::leal::campello_renderer
