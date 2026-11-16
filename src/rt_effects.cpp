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

// As EffectUniforms in rt_effects.metal.
struct EffectUniforms {
    float invViewProj[16];  // column-major
    float eye[4];
    float params[4];
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

// ---------------------------------------------------------------------------
// RayTracingScene
// ---------------------------------------------------------------------------

RayTracingScene::RayTracingScene(std::shared_ptr<GPU::Device> device)
    : device_(std::move(device)) {}

bool RayTracingScene::update(const RenderScene& scene) {
    if (!device_) return false;
    uint64_t geometryKey = 1469598103934665603ull, transformKey = 1469598103934665603ull;
    for (const auto& draw : rt_effects_detail::casters(scene)) {
        rt_effects_detail::mix(geometryKey, &draw.mesh, sizeof(draw.mesh));
        const void* buffers[2] = {draw.mesh->vertexBuffers[Renderer::VERTEX_SLOT_POSITION].get(), draw.mesh->indexBuffer.get()};
        rt_effects_detail::mix(geometryKey, buffers, sizeof(buffers));
        rt_effects_detail::mix(geometryKey, &draw.mesh->indexCount, sizeof(draw.mesh->indexCount));
        rt_effects_detail::mix(transformKey, draw.worldTransform.data, sizeof(double) * 16);
    }
    if (tlas_ && geometryKey == geometryKey_ && transformKey == transformKey_) return true;
    if (!rebuild(scene)) {
        tlas_.reset();
        return false;
    }
    geometryKey_ = geometryKey;
    transformKey_ = transformKey;
    return true;
}

bool RayTracingScene::rebuild(const RenderScene& scene) {
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
    ++(builds.empty() ? transformRebuilds_ : fullRebuilds_);
    return true;
}

// ---------------------------------------------------------------------------
// RayTracedEffect
// ---------------------------------------------------------------------------

RayTracedEffect::RayTracedEffect(std::shared_ptr<GPU::Device> device, std::shared_ptr<RayTracingScene> scene,
                                 const char* entryPoint)
    : device_(std::move(device)), scene_(std::move(scene)) {
    if (!scene_ && device_) scene_ = std::make_shared<RayTracingScene>(device_);
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
    rt.rayGeneration = {module, entryPoint};
    rt.layout = device_->createPipelineLayout(pl);
    rt.maxRecursionDepth = 1;
    pipeline_ = device_->createRayTracingPipeline(rt);
#else
    (void)entryPoint;
#endif
}

RayTracedEffect::~RayTracedEffect() {
    if (pending_) pending_->wait();
}

bool RayTracedEffect::dispatch(const RenderScene& scene, const GBufferTargets& gbuffer, const float (&parameters)[4]) {
    if (!supported() || !scene_ || !gbuffer.normalRoughness || !gbuffer.depth || gbuffer.width == 0 || gbuffer.height == 0) {
        return false;
    }
    if (pending_) {
        pending_->wait();  // its uniforms are rewritten below
        pending_.reset();
    }
    if (!scene_->update(scene) || !scene_->tlas()) return false;
    const uint32_t w = gbuffer.width, h = gbuffer.height;
    if (!output_ || output_->getWidth() != w || output_->getHeight() != h) {
        output_ = device_->createTexture(GPU::TextureType::tt2d, GPU::PixelFormat::rgba16float, w, h, 1, 1, 1,
                                         (GPU::TextureUsage)(uint32_t(GPU::TextureUsage::storageBinding) |
                                                             uint32_t(GPU::TextureUsage::textureBinding) |
                                                             uint32_t(GPU::TextureUsage::copySrc)));
        if (!output_) return false;
    }
    rt_effects_detail::EffectUniforms u{};
    const auto invViewProj = (scene.camera.projection * scene.camera.view).inverted();
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) u.invViewProj[c * 4 + r] = static_cast<float>(invViewProj.data[r * 4 + c]);
    }
    const auto eye = scene.camera.view.inverted() * VM::Vector4<double>(0.0, 0.0, 0.0, 1.0);
    u.eye[0] = static_cast<float>(eye.x() / eye.w());
    u.eye[1] = static_cast<float>(eye.y() / eye.w());
    u.eye[2] = static_cast<float>(eye.z() / eye.w());
    for (int k = 0; k < 4; ++k) u.params[k] = parameters[k];
    u.size[0] = w;
    u.size[1] = h;
    u.size[2] = frame_++;
    if (!uniforms_) uniforms_ = device_->createBuffer(sizeof(u), GPU::BufferUsage::uniform);
    if (!uniforms_) return false;
    uniforms_->upload(0, sizeof(u), &u);

    GPU::BindGroupDescriptor desc{};
    desc.layout = layout_;
    desc.entries = {{0, scene_->tlas()},
                    {1, GPU::BufferBinding{uniforms_, 0, sizeof(u)}},
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

// ---------------------------------------------------------------------------
// RayTracedShadows
// ---------------------------------------------------------------------------

RayTracedShadows::RayTracedShadows(std::shared_ptr<GPU::Device> device, std::shared_ptr<RayTracingScene> scene)
    : RayTracedEffect(std::move(device), std::move(scene), "rayTracedShadows") {}

bool RayTracedShadows::trace(const RenderScene& scene, const GBufferTargets& gbuffer) {
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
    const float parameters[4] = {static_cast<float>(-sun->direction.x() / len), static_cast<float>(-sun->direction.y() / len),
                                 static_cast<float>(-sun->direction.z() / len),
                                 std::cos(sunDiameterDegrees_ * 0.5f * 3.14159265f / 180.0f)};
    return dispatch(scene, gbuffer, parameters);
}

// ---------------------------------------------------------------------------
// RayTracedAmbientOcclusion
// ---------------------------------------------------------------------------

RayTracedAmbientOcclusion::RayTracedAmbientOcclusion(std::shared_ptr<GPU::Device> device, std::shared_ptr<RayTracingScene> scene)
    : RayTracedEffect(std::move(device), std::move(scene), "rayTracedAmbientOcclusion") {}

bool RayTracedAmbientOcclusion::trace(const RenderScene& scene, const GBufferTargets& gbuffer) {
    const float parameters[4] = {radius_, 0.0f, 0.0f, 0.0f};
    return dispatch(scene, gbuffer, parameters);
}

// ---------------------------------------------------------------------------
// RestirDirectLighting
// ---------------------------------------------------------------------------

namespace rt_effects_detail {

// As RestirUniforms / RestirLight in rt_effects.metal.
struct RestirUniforms {
    float invViewProj[16];  // column-major
    float eye[4];
    uint32_t size[4];    // width, height, frame, light count
    uint32_t counts[4];  // candidates, temporal cap, spatial samples, history valid
    float params[4];     // spatial radius, temporal on, spatial on, spatial below this fraction of the cap
};
struct RestirLight {
    float position[4];
    float color[4];
    float direction[4];
    float extra[4];
};

} // namespace rt_effects_detail

RestirDirectLighting::RestirDirectLighting(std::shared_ptr<GPU::Device> device, std::shared_ptr<RayTracingScene> scene)
    : device_(std::move(device)), scene_(std::move(scene)) {
    if (!scene_ && device_) scene_ = std::make_shared<RayTracingScene>(device_);
#if defined(__APPLE__) && !TARGET_OS_SIMULATOR
    if (!device_ || !device_->getFeatures().count(GPU::Feature::raytracing)) return;
    auto module = device_->createShaderModule(shaders::kRtEffectsMetalShader, shaders::kRtEffectsMetalShaderSize);
    if (!module) return;
    // The scene (buffer 0), uniforms (1), lights (2); textures: normal +
    // roughness, depth, albedo + metallic, motion, reservoirs in, surfaces
    // in, reservoirs out, surfaces out, radiance out.
    GPU::BindGroupLayoutDescriptor desc{};
    GPU::EntryObject sceneEntry{};
    sceneEntry.binding = 0;
    sceneEntry.type = GPU::EntryObjectType::accelerationStructure;
    sceneEntry.visibility = GPU::ShaderStage::rayGeneration;
    desc.entries.push_back(sceneEntry);
    for (uint32_t binding = 1; binding <= 2; ++binding) {
        GPU::EntryObject buffer{};
        buffer.binding = binding;
        buffer.type = GPU::EntryObjectType::buffer;
        buffer.visibility = GPU::ShaderStage::rayGeneration;
        buffer.data.buffer.hasDinamicOffaset = false;
        buffer.data.buffer.minBindingSize = 0;
        buffer.data.buffer.type = binding == 1 ? GPU::EntryObjectBufferType::uniform : GPU::EntryObjectBufferType::readOnlyStorage;
        desc.entries.push_back(buffer);
    }
    for (uint32_t binding = 0; binding < 9; ++binding) {
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
    auto pipelineLayout = device_->createPipelineLayout(pl);
    for (auto [pipeline, entry] :
         {std::pair{&initial_, "restirInitial"}, {&spatial_, "restirSpatial"}, {&reference_, "restirReference"}}) {
        GPU::RayTracingPipelineDescriptor rt{};
        rt.rayGeneration = {module, entry};
        rt.layout = pipelineLayout;
        rt.maxRecursionDepth = 1;
        *pipeline = device_->createRayTracingPipeline(rt);
    }
    if (!initial_ || !spatial_ || !reference_) {
        initial_ = nullptr;
        spatial_ = nullptr;
        reference_ = nullptr;
    }
#endif
}

RestirDirectLighting::~RestirDirectLighting() {
    if (pending_) pending_->wait();
}

bool RestirDirectLighting::ensureTargets(uint32_t width, uint32_t height) {
    if (radiance_ && radiance_->getWidth() == width && radiance_->getHeight() == height) return true;
    const auto make = [&](GPU::PixelFormat format) {
        return device_->createTexture(GPU::TextureType::tt2d, format, width, height, 1, 1, 1,
                                      (GPU::TextureUsage)(uint32_t(GPU::TextureUsage::storageBinding) |
                                                          uint32_t(GPU::TextureUsage::textureBinding) |
                                                          uint32_t(GPU::TextureUsage::copySrc)));
    };
    reservoirs_[0] = make(GPU::PixelFormat::rgba32float);
    reservoirs_[1] = make(GPU::PixelFormat::rgba32float);
    final_ = make(GPU::PixelFormat::rgba32float);
    surfaces_[0] = make(GPU::PixelFormat::rgba32float);
    surfaces_[1] = make(GPU::PixelFormat::rgba32float);
    radiance_ = make(GPU::PixelFormat::rgba16float);
    historyValid_ = false;
    return reservoirs_[0] && reservoirs_[1] && final_ && surfaces_[0] && surfaces_[1] && radiance_;
}

bool RestirDirectLighting::render(const RenderScene& scene, const GBufferTargets& gbuffer) {
    return frame(scene, gbuffer, false);
}

bool RestirDirectLighting::renderReference(const RenderScene& scene, const GBufferTargets& gbuffer) {
    return frame(scene, gbuffer, true);
}

bool RestirDirectLighting::frame(const RenderScene& scene, const GBufferTargets& gbuffer, bool reference) {
    namespace D = rt_effects_detail;
    if (!supported() || !scene_ || !gbuffer.normalRoughness || !gbuffer.depth || !gbuffer.albedoMetallic || !gbuffer.motion ||
        gbuffer.width == 0 || gbuffer.height == 0 || scene.lights.empty()) {
        return false;
    }
    if (pending_) {
        pending_->wait();  // its uniforms and lights are rewritten below
        pending_.reset();
    }
    if (!scene_->update(scene) || !scene_->tlas()) return false;
    const uint32_t w = gbuffer.width, h = gbuffer.height;
    if (!ensureTargets(w, h)) return false;

    // The lights, with a CDF by power (luminance x intensity).
    std::vector<D::RestirLight> lights;
    lights.reserve(scene.lights.size());
    double total = 0.0;
    std::vector<double> power;
    for (const auto& light : scene.lights) {
        D::RestirLight l{};
        if (light.type == 0) {
            const double len = std::sqrt(light.direction.x() * light.direction.x() + light.direction.y() * light.direction.y() +
                                         light.direction.z() * light.direction.z());
            if (len < 1e-12) continue;
            l.position[0] = static_cast<float>(-light.direction.x() / len);
            l.position[1] = static_cast<float>(-light.direction.y() / len);
            l.position[2] = static_cast<float>(-light.direction.z() / len);
            l.position[3] = 0.0f;
        } else {
            l.position[0] = static_cast<float>(light.position.x());
            l.position[1] = static_cast<float>(light.position.y());
            l.position[2] = static_cast<float>(light.position.z());
            l.position[3] = static_cast<float>(light.type == 2 ? 2 : 1);
            const double len = std::sqrt(light.direction.x() * light.direction.x() + light.direction.y() * light.direction.y() +
                                         light.direction.z() * light.direction.z());
            if (len > 1e-12) {
                l.direction[0] = static_cast<float>(light.direction.x() / len);
                l.direction[1] = static_cast<float>(light.direction.y() / len);
                l.direction[2] = static_cast<float>(light.direction.z() / len);
            }
            l.direction[3] = std::cos(light.outerConeAngle);
            l.extra[0] = std::cos(light.innerConeAngle);
        }
        l.color[0] = static_cast<float>(light.color.x()) * light.intensity;
        l.color[1] = static_cast<float>(light.color.y()) * light.intensity;
        l.color[2] = static_cast<float>(light.color.z()) * light.intensity;
        l.color[3] = light.range;
        const double p = std::max(0.0, 0.2126 * l.color[0] + 0.7152 * l.color[1] + 0.0722 * l.color[2]);
        power.push_back(p);
        total += p;
        lights.push_back(l);
    }
    if (lights.empty() || total <= 0.0) return false;
    double running = 0.0;
    for (size_t i = 0; i < lights.size(); ++i) {
        running += power[i];
        lights[i].extra[1] = i + 1 == lights.size() ? 1.0f : static_cast<float>(running / total);
        lights[i].extra[2] = static_cast<float>(power[i] / total);
    }
    const uint64_t lightBytes = lights.size() * sizeof(D::RestirLight);
    if (!lights_ || lights_->getLength() < lightBytes) {
        lights_ = device_->createBuffer(lightBytes, GPU::BufferUsage::storage);
        if (!lights_) return false;
    }
    lights_->upload(0, lightBytes, lights.data());

    D::RestirUniforms u{};
    const auto invViewProj = (scene.camera.projection * scene.camera.view).inverted();
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) u.invViewProj[c * 4 + r] = static_cast<float>(invViewProj.data[r * 4 + c]);
    }
    const auto eye = scene.camera.view.inverted() * VM::Vector4<double>(0.0, 0.0, 0.0, 1.0);
    u.eye[0] = static_cast<float>(eye.x() / eye.w());
    u.eye[1] = static_cast<float>(eye.y() / eye.w());
    u.eye[2] = static_cast<float>(eye.z() / eye.w());
    u.size[0] = w;
    u.size[1] = h;
    u.size[2] = frame_++;
    u.size[3] = static_cast<uint32_t>(lights.size());
    u.counts[0] = std::max(1u, settings_.candidates);
    u.counts[1] = std::max(1u, settings_.temporalCap);
    u.counts[2] = settings_.spatialSamples;
    u.counts[3] = historyValid_ ? 1u : 0u;
    u.params[0] = settings_.spatialRadius > 0.0f ? std::max(1.0f, settings_.spatialRadius)
                                                  : std::max(2.0f, 0.01f * static_cast<float>(w));
    u.params[1] = settings_.temporal ? 1.0f : 0.0f;
    u.params[2] = settings_.spatial && settings_.spatialSamples > 0 ? 1.0f : 0.0f;
    u.params[3] = std::clamp(settings_.spatialHistory, 0.0f, 1.0f);
    if (!uniforms_) uniforms_ = device_->createBuffer(sizeof(u), GPU::BufferUsage::uniform);
    if (!uniforms_) return false;
    uniforms_->upload(0, sizeof(u), &u);

    const auto& current = surfaces_[surfaceIndex_];
    const auto& previous = surfaces_[1 - surfaceIndex_];
    const auto& reservoirs = reservoirs_[surfaceIndex_];
    const auto& previousReservoirs = reservoirs_[1 - surfaceIndex_];
    const auto group = [&](const std::shared_ptr<GPU::Texture>& reservoirIn, const std::shared_ptr<GPU::Texture>& surfaceIn,
                           const std::shared_ptr<GPU::Texture>& reservoirOut, const std::shared_ptr<GPU::Texture>& surfaceOut) {
        GPU::BindGroupDescriptor desc{};
        desc.layout = layout_;
        desc.entries = {{0, scene_->tlas()},
                        {1, GPU::BufferBinding{uniforms_, 0, sizeof(u)}},
                        {2, GPU::BufferBinding{lights_, 0, lightBytes}},
                        {0, gbuffer.normalRoughness},
                        {1, gbuffer.depth},
                        {2, gbuffer.albedoMetallic},
                        {3, gbuffer.motion},
                        {4, reservoirIn},
                        {5, surfaceIn},
                        {6, reservoirOut},
                        {7, surfaceOut},
                        {8, radiance_}};
        return device_->createBindGroup(desc);
    };
    // Initial + temporal: last frame's reservoirs and surfaces in, this
    // frame's out. Spatial: those in, the radiance out. The history is the
    // temporal pass's reservoirs: fed back, the spatial pass's would carry
    // its bias (neighbours' samples without their visibility) from frame to
    // frame (the error grew with it in the test). Its own reservoir output
    // (final_) and surface output aren't read.
    auto initialGroup = group(previousReservoirs, previous, reservoirs, current);
    auto spatialGroup = group(reservoirs, current, final_, previous);
    auto encoder = device_->createCommandEncoder();
    if (!initialGroup || !spatialGroup || !encoder) return false;
    std::vector<std::pair<std::shared_ptr<GPU::RayTracingPipeline>, std::shared_ptr<GPU::BindGroup>>> passes;
    if (reference) {
        passes = {{reference_, spatialGroup}};
    } else {
        passes = {{initial_, initialGroup}, {spatial_, spatialGroup}};
    }
    for (const auto& [pipeline, bindGroup] : passes) {
        if (auto pass = encoder->beginRayTracingPass()) {
            pass->setPipeline(pipeline);
            pass->setBindGroup(0, bindGroup);
            pass->traceRays(w, h, 1);
            pass->end();
        }
    }
    auto fence = device_->createFence();
    device_->submit(encoder->finish(), fence);
    pending_ = fence;
    if (!reference) {
        surfaceIndex_ = 1 - surfaceIndex_;
        historyValid_ = true;
    }
    return true;
}

} // namespace systems::leal::campello_renderer
