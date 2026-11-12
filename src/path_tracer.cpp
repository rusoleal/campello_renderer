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
};

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
};

struct InstanceInfo {
    float normalMatrix[3][4];
    uint32_t normalOffset;
    uint32_t material;
    uint32_t pad0;
    uint32_t pad1;
};

struct MaterialInfo {
    float baseColor[4];
    float emissive[4];
    float roughness;
    float metallic;
    float transmission;
    float ior;
    float unlit;
    float pad0;
    float pad1;
    float pad2;
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

} // namespace

PathTracer::PathTracer(std::shared_ptr<GPU::Device> device)
    : device_(std::move(device)) {
#if defined(__APPLE__) && !TARGET_OS_SIMULATOR
    if (!device_ || !device_->getFeatures().count(GPU::Feature::raytracing)) return;
    auto module = device_->createShaderModule(shaders::kPathTracerMetalShader, shaders::kPathTracerMetalShaderSize);
    if (!module) return;

    // Trace: the scene (0), accumulation (1), uniforms (2), instances (3),
    // materials (4), normals (5), denoiser guides (6); textures: output (0),
    // environment (1).
    {
        GPU::BindGroupLayoutDescriptor desc{};
        GPU::EntryObject scene{};
        scene.binding = 0;
        scene.type = GPU::EntryObjectType::accelerationStructure;
        scene.visibility = GPU::ShaderStage::rayGeneration;
        desc.entries.push_back(scene);
        for (uint32_t binding = 1; binding <= 6; ++binding) {
            GPU::EntryObject buffer{};
            buffer.binding = binding;
            buffer.type = GPU::EntryObjectType::buffer;
            buffer.visibility = GPU::ShaderStage::rayGeneration;
            buffer.data.buffer.hasDinamicOffaset = false;
            buffer.data.buffer.minBindingSize = 0;
            buffer.data.buffer.type = binding == 2 ? GPU::EntryObjectBufferType::uniform : GPU::EntryObjectBufferType::storage;
            desc.entries.push_back(buffer);
        }
        for (uint32_t binding = 0; binding <= 1; ++binding) {
            GPU::EntryObject texture{};
            texture.binding = binding;
            texture.type = GPU::EntryObjectType::texture;
            texture.visibility = GPU::ShaderStage::rayGeneration;
            texture.data.texture = {false, GPU::EntryObjectTextureType::ttFloat,
                                    binding == 0 ? GPU::TextureType::tt2d : GPU::TextureType::ttCube};
            desc.entries.push_back(texture);
        }
        traceLayout_ = device_->createBindGroupLayout(desc);
    }
    {
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
        normalsLayout_ = device_->createBindGroupLayout(desc);
    }
    if (!traceLayout_ || !normalsLayout_) return;

    GPU::PipelineLayoutDescriptor normalsPl{};
    normalsPl.bindGroupLayouts = {normalsLayout_};
    GPU::ComputePipelineDescriptor normalsDesc{};
    normalsDesc.compute = {module, "faceNormals"};
    normalsDesc.layout = device_->createPipelineLayout(normalsPl);
    normalsPipeline_ = device_->createComputePipeline(normalsDesc);

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

bool PathTracer::rebuild(const RenderScene& scene, const Renderer& renderer, uint64_t sceneKey) {
    const auto draws = traced(scene);
    if (draws.empty()) return false;

    // Acceleration structures of meshes not seen before (kept while the
    // mesh is in the scene).
    std::map<const GpuMesh*, MeshEntry> meshes;
    std::vector<std::pair<std::shared_ptr<GPU::AccelerationStructure>, GPU::BottomLevelAccelerationStructureDescriptor>> builds;
    for (const auto& draw : draws) {
        if (meshes.count(draw.mesh)) continue;
        if (auto it = meshes_.find(draw.mesh); it != meshes_.end()) {
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
        meshes.emplace(draw.mesh, MeshEntry{blas, triangles});
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
            if (const float* slot = renderer.materialUniforms(draw.material)) {
                for (int c = 0; c < 4; ++c) info.baseColor[c] = slot[c];
                for (int c = 0; c < 3; ++c) info.baseColor[c] *= draw.material->baseColorTextureAverage[(size_t)c];
                info.metallic = slot[12];
                info.roughness = slot[13];
                info.unlit = slot[17];
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
        info.material = matIt->second;
        instanceInfo.push_back(info);
    }
    if (tlasDesc.instances.empty()) return false;

    normals_ = device_->createBuffer(std::max<uint64_t>(triangleTotal, 1) * 16, GPU::BufferUsage::storage);
    materials_ = device_->createBuffer(materials.size() * sizeof(MaterialInfo), GPU::BufferUsage::storage, materials.data());
    instances_ = device_->createBuffer(instanceInfo.size() * sizeof(InstanceInfo), GPU::BufferUsage::storage, instanceInfo.data());
    tlas_ = device_->createTopLevelAccelerationStructure(tlasDesc);
    if (!normals_ || !materials_ || !instances_ || !tlas_) return false;

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
    if (auto pass = encoder->beginComputePass()) {
        pass->setPipeline(normalsPipeline_);
        for (const auto& [mesh, entry] : meshes_) {
            NormalParams params{entry.triangleCount, normalOffsets[mesh],
                                !mesh->indexBuffer || mesh->indexCount == 0
                                    ? 0u
                                    : (mesh->indexFormat == GPU::IndexFormat::uint16 ? 1u : 2u),
                                12u};
            auto paramsBuffer = device_->createBuffer(sizeof(params), GPU::BufferUsage::uniform, &params);
            keepAlive.push_back(paramsBuffer);
            const auto& positions = mesh->vertexBuffers[Renderer::VERTEX_SLOT_POSITION];
            const auto& indices = mesh->indexBuffer ? mesh->indexBuffer : positions;
            GPU::BindGroupDescriptor desc{};
            desc.layout = normalsLayout_;
            desc.entries = {{0, GPU::BufferBinding{positions, 0, positions->getLength()}},
                            {1, GPU::BufferBinding{indices, 0, indices->getLength()}},
                            {2, GPU::BufferBinding{normals_, 0, normals_->getLength()}},
                            {3, GPU::BufferBinding{paramsBuffer, 0, sizeof(params)}}};
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
    auto fence = device_->createFence();
    device_->submit(encoder->finish(), fence);
    if (fence) fence->wait();
    sceneKey_ = sceneKey;
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
}

bool PathTracer::render(const RenderScene& scene, const Renderer& renderer, uint32_t width, uint32_t height) {
    if (!supported() || width == 0 || height == 0) return false;
    // The previous sample (usually done by now, a frame later): its
    // uniforms and buffers are rewritten below.
    waitForSample();

    // What the image depends on besides the camera: geometry, materials,
    // lights and the renderer's lighting settings.
    uint64_t sceneKey = 1469598103934665603ull;
    for (const auto& draw : traced(scene)) {
        mix(sceneKey, &draw.mesh, sizeof(draw.mesh));
        const void* positions = draw.mesh->vertexBuffers[Renderer::VERTEX_SLOT_POSITION].get();
        mix(sceneKey, &positions, sizeof(positions));
        mix(sceneKey, &draw.mesh->indexCount, sizeof(draw.mesh->indexCount));
        mix(sceneKey, &draw.material, sizeof(draw.material));
        if (const float* slot = renderer.materialUniforms(draw.material)) mix(sceneKey, slot, 93 * sizeof(float));
        mix(sceneKey, draw.worldTransform.data, sizeof(double) * 16);
    }
    for (const auto& light : scene.lights) {
        const double values[7] = {light.direction.x(), light.direction.y(), light.direction.z(),
                                  light.color.x(),     light.color.y(),     light.color.z(), light.intensity};
        mix(sceneKey, values, sizeof(values));
    }
    float clear[4];
    renderer.getClearColor(&clear[0], &clear[1], &clear[2], &clear[3]);
    const float settings[6] = {renderer.getEnvironmentIntensity(), renderer.getExposure(), clear[0], clear[1], clear[2],
                               renderer.getSkyboxEnabled() ? 1.0f : 0.0f};
    mix(sceneKey, settings, sizeof(settings));
    const auto environment = renderer.getEnvironmentMap();
    const void* envPtr = environment.get();
    mix(sceneKey, &envPtr, sizeof(envPtr));
    mix(sceneKey, &maxBounces_, sizeof(maxBounces_));

    uint64_t viewKey = 1469598103934665603ull;
    mix(viewKey, scene.camera.view.data, sizeof(double) * 16);
    mix(viewKey, scene.camera.projection.data, sizeof(double) * 16);
    mix(viewKey, &width, sizeof(width));
    mix(viewKey, &height, sizeof(height));
    mix(viewKey, &projection_, sizeof(projection_));

    if (!tlas_ || sceneKey != sceneKey_) {
        if (!rebuild(scene, renderer, sceneKey)) return false;
        samples_ = 0;
    }
    if (viewKey != viewKey_) {
        viewKey_ = viewKey;
        samples_ = 0;
    }
    if (!environment) return false;
    if (width != width_ || height != height_ || !accum_ || !output_) {
        const auto readable = (GPU::BufferUsage)(uint32_t(GPU::BufferUsage::storage) | uint32_t(GPU::BufferUsage::copyDst) |
                                                 uint32_t(GPU::BufferUsage::copySrc));
        accum_ = device_->createBuffer((uint64_t)width * height * 16, readable);
        aux_ = device_->createBuffer((uint64_t)width * height * 32, readable);
        output_ = device_->createTexture(GPU::TextureType::tt2d, GPU::PixelFormat::rgba8unorm, width, height, 1, 1, 1,
                                         (GPU::TextureUsage)(uint32_t(GPU::TextureUsage::storageBinding) |
                                                             uint32_t(GPU::TextureUsage::textureBinding) |
                                                             uint32_t(GPU::TextureUsage::copySrc) |
                                                             uint32_t(GPU::TextureUsage::copyDst)));
        width_ = width;
        height_ = height;
        samples_ = 0;
        if (!accum_ || !aux_ || !output_) return false;
    }

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
    u.params[2] = static_cast<float>(maxBounces_);
    u.params[3] = static_cast<float>(samples_);
    u.size[0] = width;
    u.size[1] = height;
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
                    {0, output_},
                    {1, environment}};
    auto group = device_->createBindGroup(desc);
    auto encoder = device_->createCommandEncoder();
    if (!group || !encoder) return false;
    if (auto pass = encoder->beginRayTracingPass()) {
        pass->setPipeline(pipeline_);
        pass->setBindGroup(0, group);
        pass->traceRays(width, height, 1);
        pass->end();
    }
    // Not waited for here: the GPU traces while the caller goes on (the next
    // render() and the readers wait). Work submitted later on the same queue
    // (e.g. copying output()) runs after it.
    auto fence = device_->createFence();
    device_->submit(encoder->finish(), fence);
    pending_ = fence;
    ++samples_;
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
    if (!accum_ || !aux_ || samples_ == 0) return false;
    waitForSample();
    const size_t pixels = (size_t)width_ * height_;
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
        const float n = static_cast<float>(samples_);
        if (albedo) {
            albedo->resize(pixels * 3);
            for (size_t i = 0; i < pixels; ++i) {
                for (int c = 0; c < 3; ++c) (*albedo)[i * 3 + c] = aux[i * 4 + c] / n;
            }
        }
        if (normal) {
            normal->resize(pixels * 3);
            for (size_t i = 0; i < pixels; ++i) {
                for (int c = 0; c < 3; ++c) (*normal)[i * 3 + c] = aux[(pixels + i) * 4 + c] / n;
            }
        }
    }
    return true;
}

void PathTracer::toneMap(const float* rgb, size_t pixels, uint8_t* rgba) {
    // As toneMapKhronosPbrNeutral() in path_tracer.metal.
    for (size_t i = 0; i < pixels; ++i) {
        float c[3] = {std::max(rgb[i * 3], 0.0f), std::max(rgb[i * 3 + 1], 0.0f), std::max(rgb[i * 3 + 2], 0.0f)};
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
    oidn::DeviceRef device;
    oidn::FilterRef filter;
    oidn::BufferRef color, albedo, normal, output;
    uint32_t width = 0;
    uint32_t height = 0;
};
} // namespace
#endif

bool PathTracer::denoiserAvailable() noexcept {
#if defined(CAMPELLO_RENDERER_DENOISER)
    return true;
#else
    return false;
#endif
}

bool PathTracer::denoise() {
#if defined(CAMPELLO_RENDERER_DENOISER)
    std::vector<float> color, albedo, normal;
    if (!readLinear(color, &albedo, &normal)) return false;
    try {
        // A CPU device (works everywhere the library does, no sharing with
        // campello_gpu's device), its filter and buffers kept for the next
        // image of the same size: committing a filter loads its weights.
        auto* state = static_cast<DenoiserState*>(denoiser_.get());
        if (!state) {
            auto created = std::make_shared<DenoiserState>();
            created->device = oidn::newDevice(oidn::DeviceType::CPU);
            created->device.commit();
            denoiser_ = created;
            state = created.get();
        }
        const size_t bytes = color.size() * sizeof(float);
        if (state->width != width_ || state->height != height_) {
            state->color = state->device.newBuffer(bytes);
            state->albedo = state->device.newBuffer(bytes);
            state->normal = state->device.newBuffer(bytes);
            state->output = state->device.newBuffer(bytes);
            state->filter = state->device.newFilter("RT");
            state->filter.setImage("color", state->color, oidn::Format::Float3, width_, height_);
            state->filter.setImage("albedo", state->albedo, oidn::Format::Float3, width_, height_);
            state->filter.setImage("normal", state->normal, oidn::Format::Float3, width_, height_);
            state->filter.setImage("output", state->output, oidn::Format::Float3, width_, height_);
            state->filter.set("hdr", true);
            state->filter.commit();
            state->width = width_;
            state->height = height_;
        }
        state->color.write(0, bytes, color.data());
        state->albedo.write(0, bytes, albedo.data());
        state->normal.write(0, bytes, normal.data());
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
    std::vector<uint8_t> rgba((size_t)width_ * height_ * 4);
    toneMap(color.data(), (size_t)width_ * height_, rgba.data());
    device_->waitForIdle();
    if (!output_->upload(0, rgba.size(), rgba.data())) return false;
    denoised_ = true;
    return true;
#else
    return false;
#endif
}

} // namespace systems::leal::campello_renderer
