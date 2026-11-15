#include <campello_renderer/rt_effects.hpp>

#include <campello_gpu/acceleration_structure.hpp>
#include <campello_gpu/bind_group.hpp>
#include <campello_gpu/bind_group_layout.hpp>
#include <campello_gpu/buffer.hpp>
#include <campello_gpu/command_buffer.hpp>
#include <campello_gpu/command_encoder.hpp>
#include <campello_gpu/constants/feature.hpp>
#include <campello_gpu/descriptors/bind_group_descriptor.hpp>
#include <campello_gpu/descriptors/bind_group_layout_descriptor.hpp>
#include <campello_gpu/descriptors/bottom_level_acceleration_structure_descriptor.hpp>
#include <campello_gpu/descriptors/pipeline_layout_descriptor.hpp>
#include <campello_gpu/descriptors/ray_tracing_pipeline_descriptor.hpp>
#include <campello_gpu/descriptors/top_level_acceleration_structure_descriptor.hpp>
#include <campello_gpu/device.hpp>
#include <campello_gpu/fence.hpp>
#include <campello_gpu/ray_tracing_pass_encoder.hpp>
#include <campello_gpu/ray_tracing_pipeline.hpp>
#include <campello_gpu/texture.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

#if defined(__APPLE__)
#include "shaders/metal_rt_effects.h"
#endif

namespace systems::leal::campello_renderer {

namespace GPU = systems::leal::campello_gpu;
namespace VM = systems::leal::vector_math;

// (A named namespace: the unity build joins this file with path_tracer.cpp,
// whose anonymous namespace has helpers of the same names.)
namespace rt_effects_detail {

// As ShadowUniforms in rt_effects.metal.
struct ShadowUniforms {
    float invViewProj[16];  // column-major
    float eye[4];
    float sun[4];
    uint32_t size[4];
};

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

std::vector<DrawCall> casters(const RenderScene& scene) {
    std::vector<DrawCall> out;
    for (const auto& draw : scene.shadowCasters.empty() ? scene.opaque : scene.shadowCasters) {
        if (!draw.mesh || draw.mesh->topology != GPU::PrimitiveTopology::triangleList ||
            !draw.mesh->vertexBuffers[Renderer::VERTEX_SLOT_POSITION] || !floatPositions(*draw.mesh) || draw.mesh->vertexCount == 0) {
            continue;
        }
        out.push_back(draw);
    }
    return out;
}

} // namespace rt_effects_detail


RayTracedShadows::RayTracedShadows(std::shared_ptr<GPU::Device> device)
    : device_(std::move(device)) {
#if defined(__APPLE__) && !TARGET_OS_SIMULATOR
    if (!device_ || !device_->getFeatures().count(GPU::Feature::raytracing)) return;
    auto module = device_->createShaderModule(shaders::kRtEffectsMetalShader, shaders::kRtEffectsMetalShaderSize);
    if (!module) return;
    // The scene (buffer 0), uniforms (buffer 1); normal + roughness (texture
    // 0), depth (texture 1), visibility out (texture 2).
    GPU::BindGroupLayoutDescriptor desc{};
    GPU::EntryObject sceneEntry{};
    sceneEntry.binding = 0;
    sceneEntry.type = GPU::EntryObjectType::accelerationStructure;
    sceneEntry.visibility = GPU::ShaderStage::rayGeneration;
    desc.entries.push_back(sceneEntry);
    GPU::EntryObject uniforms{};
    uniforms.binding = 1;
    uniforms.type = GPU::EntryObjectType::buffer;
    uniforms.visibility = GPU::ShaderStage::rayGeneration;
    uniforms.data.buffer.hasDinamicOffaset = false;
    uniforms.data.buffer.minBindingSize = 0;
    uniforms.data.buffer.type = GPU::EntryObjectBufferType::uniform;
    desc.entries.push_back(uniforms);
    for (uint32_t binding = 0; binding < 3; ++binding) {
        GPU::EntryObject texture{};
        texture.binding = binding;
        texture.type = GPU::EntryObjectType::texture;
        texture.visibility = GPU::ShaderStage::rayGeneration;
        texture.data.texture = {false, binding == 1 ? GPU::EntryObjectTextureType::ttDepth : GPU::EntryObjectTextureType::ttFloat,
                                GPU::TextureType::tt2d};
        desc.entries.push_back(texture);
    }
    layout_ = device_->createBindGroupLayout(desc);
    if (!layout_) return;
    GPU::PipelineLayoutDescriptor pl{};
    pl.bindGroupLayouts = {layout_};
    GPU::RayTracingPipelineDescriptor rt{};
    rt.rayGeneration = {module, "rayTracedShadows"};
    rt.layout = device_->createPipelineLayout(pl);
    rt.maxRecursionDepth = 1;
    pipeline_ = device_->createRayTracingPipeline(rt);
#endif
}

RayTracedShadows::~RayTracedShadows() {
    if (pending_) pending_->wait();
}

bool RayTracedShadows::rebuild(const RenderScene& scene) {
    const auto draws = rt_effects_detail::casters(scene);
    if (draws.empty()) return false;
    // Mesh structures: reused while the mesh's buffers are the same.
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
        if (draw.mesh->indexBuffer && draw.mesh->indexCount > 0) {
            geometry.indexBuffer = draw.mesh->indexBuffer;
            geometry.indexFormat = draw.mesh->indexFormat;
            geometry.indexCount = draw.mesh->indexCount;
        }
        GPU::BottomLevelAccelerationStructureDescriptor blasDesc{};
        blasDesc.geometries = {geometry};
        blasDesc.buildFlags = GPU::AccelerationStructureBuildFlag::preferFastTrace;
        auto blas = device_->createBottomLevelAccelerationStructure(blasDesc);
        if (!blas) continue;
        meshes.emplace(draw.mesh, MeshEntry{blas, positions, indices});
        builds.emplace_back(blas, blasDesc);
    }
    meshes_ = std::move(meshes);

    GPU::TopLevelAccelerationStructureDescriptor tlasDesc{};
    tlasDesc.buildFlags = GPU::AccelerationStructureBuildFlag::preferFastTrace;
    for (const auto& draw : draws) {
        const auto it = meshes_.find(draw.mesh);
        if (it == meshes_.end()) continue;
        GPU::AccelerationStructureInstance instance{};
        const double* m = draw.worldTransform.data;  // row-major
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 4; ++c) instance.transform[r][c] = static_cast<float>(m[r * 4 + c]);
        }
        instance.instanceId = static_cast<uint32_t>(tlasDesc.instances.size());
        instance.mask = 1;  // opaque (as the path tracer's MASK_OPAQUE)
        instance.opaque = true;
        instance.blas = it->second.blas;
        tlasDesc.instances.push_back(instance);
    }
    if (tlasDesc.instances.empty()) return false;
    tlas_ = device_->createTopLevelAccelerationStructure(tlasDesc);
    if (!tlas_) return false;
    auto encoder = device_->createCommandEncoder();
    if (!encoder) return false;
    std::vector<std::shared_ptr<GPU::Buffer>> keepAlive;
    for (auto& [blas, desc] : builds) {
        auto scratch = device_->createBuffer(std::max<uint64_t>(blas->getBuildScratchSize(), 16), GPU::BufferUsage::storage);
        keepAlive.push_back(scratch);
        encoder->buildAccelerationStructure(blas, desc, scratch);
    }
    auto scratch = device_->createBuffer(std::max<uint64_t>(tlas_->getBuildScratchSize(), 16), GPU::BufferUsage::storage);
    keepAlive.push_back(scratch);
    encoder->buildAccelerationStructure(tlas_, tlasDesc, scratch);
    auto fence = device_->createFence();
    device_->submit(encoder->finish(), fence);
    if (fence) fence->wait();
    if (builds.empty()) {
        ++transformRebuilds_;
    } else {
        ++fullRebuilds_;
    }
    return true;
}

bool RayTracedShadows::trace(const RenderScene& scene, const GBufferTargets& gbuffer) {
    if (!supported() || !gbuffer.normalRoughness || !gbuffer.depth || gbuffer.width == 0 || gbuffer.height == 0) return false;
    // The sun: the first shadow-casting directional light, else the first directional one.
    const LightData* sun = nullptr;
    for (const auto& light : scene.lights) {
        if (light.type != 0) continue;
        if (!sun || (light.castShadows && !sun->castShadows)) sun = &light;
    }
    if (!sun) return false;
    const double len = std::sqrt(sun->direction.x() * sun->direction.x() + sun->direction.y() * sun->direction.y() +
                                 sun->direction.z() * sun->direction.z());
    if (len < 1e-12) return false;
    if (pending_) {
        pending_->wait();  // its uniforms are rewritten below
        pending_.reset();
    }

    // The scene structure: rebuilt when the casters' meshes or transforms change.
    uint64_t geometryKey = 1469598103934665603ull, transformKey = 1469598103934665603ull;
    for (const auto& draw : rt_effects_detail::casters(scene)) {
        rt_effects_detail::mix(geometryKey, &draw.mesh, sizeof(draw.mesh));
        const void* buffers[2] = {draw.mesh->vertexBuffers[Renderer::VERTEX_SLOT_POSITION].get(), draw.mesh->indexBuffer.get()};
        rt_effects_detail::mix(geometryKey, buffers, sizeof(buffers));
        rt_effects_detail::mix(geometryKey, &draw.mesh->indexCount, sizeof(draw.mesh->indexCount));
        rt_effects_detail::mix(transformKey, draw.worldTransform.data, sizeof(double) * 16);
    }
    if (!tlas_ || geometryKey != geometryKey_ || transformKey != transformKey_) {
        if (!rebuild(scene)) {
            tlas_.reset();
            return false;
        }
        geometryKey_ = geometryKey;
        transformKey_ = transformKey;
    }

    const uint32_t w = gbuffer.width, h = gbuffer.height;
    if (!output_ || output_->getWidth() != w || output_->getHeight() != h) {
        output_ = device_->createTexture(GPU::TextureType::tt2d, GPU::PixelFormat::rgba16float, w, h, 1, 1, 1,
                                         (GPU::TextureUsage)(uint32_t(GPU::TextureUsage::storageBinding) |
                                                             uint32_t(GPU::TextureUsage::textureBinding) |
                                                             uint32_t(GPU::TextureUsage::copySrc)));
        if (!output_) return false;
    }
    rt_effects_detail::ShadowUniforms u{};
    const auto invViewProj = (scene.camera.projection * scene.camera.view).inverted();
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) u.invViewProj[c * 4 + r] = static_cast<float>(invViewProj.data[r * 4 + c]);
    }
    const auto eye = scene.camera.view.inverted() * VM::Vector4<double>(0.0, 0.0, 0.0, 1.0);
    u.eye[0] = static_cast<float>(eye.x() / eye.w());
    u.eye[1] = static_cast<float>(eye.y() / eye.w());
    u.eye[2] = static_cast<float>(eye.z() / eye.w());
    u.sun[0] = static_cast<float>(-sun->direction.x() / len);
    u.sun[1] = static_cast<float>(-sun->direction.y() / len);
    u.sun[2] = static_cast<float>(-sun->direction.z() / len);
    u.sun[3] = std::cos(sunDiameterDegrees_ * 0.5f * 3.14159265f / 180.0f);
    u.size[0] = w;
    u.size[1] = h;
    u.size[2] = frame_++;
    if (!uniforms_) uniforms_ = device_->createBuffer(sizeof(rt_effects_detail::ShadowUniforms), GPU::BufferUsage::uniform);
    if (!uniforms_) return false;
    uniforms_->upload(0, sizeof(u), &u);

    GPU::BindGroupDescriptor desc{};
    desc.layout = layout_;
    desc.entries = {{0, tlas_},
                    {1, GPU::BufferBinding{uniforms_, 0, sizeof(rt_effects_detail::ShadowUniforms)}},
                    {0, gbuffer.normalRoughness},
                    {1, gbuffer.depth},
                    {2, output_}};
    auto group = device_->createBindGroup(desc);
    auto encoder = device_->createCommandEncoder();
    if (!group || !encoder) return false;
    if (auto pass = encoder->beginRayTracingPass()) {
        pass->setPipeline(pipeline_);
        pass->setBindGroup(0, group);
        pass->traceRays(w, h, 1);
        pass->end();
    }
    auto fence = device_->createFence();
    device_->submit(encoder->finish(), fence);
    pending_ = fence;
    return true;
}

} // namespace systems::leal::campello_renderer
