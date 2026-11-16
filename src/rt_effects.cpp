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

// ---------------------------------------------------------------------------
// SkinnedMeshes
// ---------------------------------------------------------------------------

namespace rt_effects_detail {

// As SkinParams in rt_effects.metal.
struct SkinParams {
    uint32_t vertexCount;
    uint32_t jointsFormat;
    uint32_t weightsFormat;
    uint32_t hasNormals;
    uint32_t jointCount;
    uint32_t pad[3];
};

// The joints' and weights' formats (kernel codes): the layout's, else by
// the buffers' bytes per vertex. False if unreadable.
bool skinFormats(const GpuMesh& mesh, uint32_t& joints, uint32_t& weights) {
    const auto& jb = mesh.vertexBuffers[Renderer::VERTEX_SLOT_JOINTS];
    const auto& wb = mesh.vertexBuffers[Renderer::VERTEX_SLOT_WEIGHTS];
    if (!jb || !wb || mesh.vertexCount == 0) return false;
    if (mesh.layout) {
        joints = weights = 99;
        for (const auto& a : mesh.layout->attributes) {
            if (a.semantic == VertexSemantic::joints) {
                joints = a.componentType == GPU::ComponentType::ctUnsignedByte    ? 0u
                         : a.componentType == GPU::ComponentType::ctUnsignedShort ? 1u
                         : a.componentType == GPU::ComponentType::ctUnsignedInt   ? 2u
                                                                                  : 99u;
            } else if (a.semantic == VertexSemantic::weights) {
                weights = a.componentType == GPU::ComponentType::ctFloat                         ? 0u
                          : a.componentType == GPU::ComponentType::ctUnsignedByte && a.normalized  ? 1u
                          : a.componentType == GPU::ComponentType::ctUnsignedShort && a.normalized ? 2u
                                                                                                   : 99u;
            }
        }
        return joints != 99 && weights != 99;
    }
    const uint64_t jointBytes = jb->getLength() / mesh.vertexCount, weightBytes = wb->getLength() / mesh.vertexCount;
    joints = jointBytes >= 16 ? 2u : jointBytes >= 8 ? 1u : 0u;
    weights = weightBytes >= 16 ? 0u : weightBytes >= 8 ? 2u : 1u;
    return jointBytes >= 4 && weightBytes >= 4;
}

constexpr uint32_t RT_ALPHA_TEXTURES = 16;  // as rt_effects.metal

// As AlphaInstance / UvGatherParams in rt_effects.metal.
struct AlphaInstance {
    float cutoff;
    float baseAlpha;
    int32_t texture;
    uint32_t uvOffset;
};
struct UvGatherParams {
    uint32_t triangleCount;
    uint32_t outOffset;
    uint32_t indexFormat;
    uint32_t pad;
};

// A mesh's TEXCOORD_0 as float2 (what the alpha test reads), or null.
std::shared_ptr<GPU::Buffer> floatTexcoords(const GpuMesh& mesh) {
    const auto& t = mesh.vertexBuffers[Renderer::VERTEX_SLOT_TEXCOORD0];
    if (!t || t->getLength() < uint64_t(mesh.vertexCount) * 8) return nullptr;
    if (mesh.layout) {
        for (const auto& a : mesh.layout->attributes) {
            if (a.semantic == VertexSemantic::texcoord0) {
                return a.componentType == GPU::ComponentType::ctFloat && a.accessorType == GPU::AccessorType::acVec2 ? t : nullptr;
            }
        }
        return nullptr;
    }
    return t;  // the standard layout: float2
}

// The effects' alpha test bindings: buffers 10 (instances), 11 (UVs),
// textures 16 .. 31.
void appendAlphaLayout(GPU::BindGroupLayoutDescriptor& desc, GPU::ShaderStage stage) {
    for (uint32_t binding = 10; binding <= 11; ++binding) {
        GPU::EntryObject buffer{};
        buffer.binding = binding;
        buffer.type = GPU::EntryObjectType::buffer;
        buffer.visibility = stage;
        buffer.data.buffer.hasDinamicOffaset = false;
        buffer.data.buffer.minBindingSize = 0;
        buffer.data.buffer.type = GPU::EntryObjectBufferType::readOnlyStorage;
        desc.entries.push_back(buffer);
    }
    for (uint32_t i = 0; i < RT_ALPHA_TEXTURES; ++i) {
        GPU::EntryObject texture{};
        texture.binding = 16 + i;
        texture.type = GPU::EntryObjectType::texture;
        texture.visibility = stage;
        texture.data.texture = {false, GPU::EntryObjectTextureType::ttFloat, GPU::TextureType::tt2d};
        desc.entries.push_back(texture);
    }
}

void appendAlphaEntries(std::vector<GPU::BindGroupEntryDescriptor>& entries, const RayTracingScene& scene) {
    const auto& instances = scene.alphaInstances();
    const auto& uvs = scene.alphaUvs();
    entries.push_back({10, GPU::BufferBinding{instances, 0, instances->getLength()}});
    entries.push_back({11, GPU::BufferBinding{uvs, 0, uvs->getLength()}});
    const auto& textures = scene.alphaTextures();
    for (uint32_t i = 0; i < RT_ALPHA_TEXTURES; ++i) entries.push_back({16 + i, textures[i]});
}

} // namespace rt_effects_detail

SkinnedMeshes::SkinnedMeshes(std::shared_ptr<GPU::Device> device) : device_(std::move(device)) {
#if defined(__APPLE__) && !TARGET_OS_SIMULATOR
    if (!device_) return;
    auto module = device_->createShaderModule(shaders::kRtEffectsMetalShader, shaders::kRtEffectsMetalShaderSize);
    if (!module) return;
    // positions, normals, joints, weights, joint matrices, out positions,
    // out normals (storage), parameters (uniform).
    GPU::BindGroupLayoutDescriptor desc{};
    for (uint32_t binding = 0; binding <= 7; ++binding) {
        GPU::EntryObject buffer{};
        buffer.binding = binding;
        buffer.type = GPU::EntryObjectType::buffer;
        buffer.visibility = GPU::ShaderStage::compute;
        buffer.data.buffer.hasDinamicOffaset = false;
        buffer.data.buffer.minBindingSize = 0;
        buffer.data.buffer.type = binding == 7 ? GPU::EntryObjectBufferType::uniform : GPU::EntryObjectBufferType::storage;
        desc.entries.push_back(buffer);
    }
    layout_ = device_->createBindGroupLayout(desc);
    if (!layout_) return;
    GPU::PipelineLayoutDescriptor pl{};
    pl.bindGroupLayouts = {layout_};
    GPU::ComputePipelineDescriptor cd{};
    cd.compute = {module, "skinVertices"};
    cd.layout = device_->createPipelineLayout(pl);
    pipeline_ = device_->createComputePipeline(cd);
#endif
}

SkinnedMeshes::~SkinnedMeshes() {
    if (pending_) pending_->wait();
}

bool SkinnedMeshes::isPosed(const GpuMesh* mesh) const {
    for (const auto& [key, posed] : posed_) {
        if (posed.mesh.get() == mesh) return true;
    }
    return false;
}

RenderScene SkinnedMeshes::pose(const RenderScene& scene) {
    namespace D = rt_effects_detail;
    posedDraws_ = 0;
    const auto skinned = [](const DrawCall& d) {
        return d.mesh && d.jointMatrixBuffer && d.jointCount > 0 && d.mesh->vertexBuffers[Renderer::VERTEX_SLOT_JOINTS] &&
               d.mesh->vertexBuffers[Renderer::VERTEX_SLOT_WEIGHTS] && d.mesh->vertexBuffers[Renderer::VERTEX_SLOT_POSITION];
    };
    bool any = false;
    for (const auto* list : {&scene.opaque, &scene.transparent, &scene.shadowCasters}) {
        for (const auto& d : *list) any = any || skinned(d);
    }
    if (!any || !supported()) {
        posed_.clear();
        return scene;
    }
    if (pending_) {
        pending_->wait();  // its parameter buffers are released below
        pending_.reset();
    }
    keepAlive_.clear();
    for (auto& [key, p] : posed_) p.used = false;
    RenderScene out = scene;
    auto encoder = device_->createCommandEncoder();
    if (!encoder) return scene;
    std::vector<std::shared_ptr<GPU::BindGroup>> groups;
    auto pass = encoder->beginComputePass();
    if (!pass) return scene;
    pass->setPipeline(pipeline_);
    // Each (mesh, joints) once, however many lists it's in.
    std::map<std::pair<const GpuMesh*, const void*>, GpuMesh*> done;
    for (auto* list : {&out.opaque, &out.transparent, &out.shadowCasters}) {
        for (auto& d : *list) {
            if (!skinned(d)) continue;
            const auto key = std::make_pair(static_cast<const GpuMesh*>(d.mesh), static_cast<const void*>(d.jointMatrixBuffer.get()));
            if (auto it = done.find(key); it != done.end()) {
                d.mesh = it->second;
                d.jointMatrixBuffer = nullptr;
                d.jointCount = 0;
                continue;
            }
            uint32_t jointsFormat = 0, weightsFormat = 0;
            if (!D::skinFormats(*d.mesh, jointsFormat, weightsFormat)) continue;
            const GpuMesh& source = *d.mesh;
            Posed& p = posed_[key];
            const uint64_t bytes = uint64_t(source.vertexCount) * 12;
            const bool hasNormals = source.vertexBuffers[Renderer::VERTEX_SLOT_NORMAL] &&
                                    source.vertexBuffers[Renderer::VERTEX_SLOT_NORMAL]->getLength() >= bytes;
            if (!p.mesh || p.mesh->vertexCount != source.vertexCount || p.mesh->indexBuffer != source.indexBuffer) {
                p.mesh = std::make_unique<GpuMesh>(source);
                const auto usage = (GPU::BufferUsage)(uint32_t(GPU::BufferUsage::storage) | uint32_t(GPU::BufferUsage::vertex));
                p.mesh->vertexBuffers[Renderer::VERTEX_SLOT_POSITION] = device_->createBuffer(bytes, usage);
                p.mesh->vertexBuffers[Renderer::VERTEX_SLOT_NORMAL] = hasNormals ? device_->createBuffer(bytes, usage) : nullptr;
                p.mesh->vertexBuffers[Renderer::VERTEX_SLOT_JOINTS] = nullptr;
                p.mesh->vertexBuffers[Renderer::VERTEX_SLOT_WEIGHTS] = nullptr;
            }
            for (size_t s = 0; s < kVertexSemanticCount; ++s) {
                if (s != Renderer::VERTEX_SLOT_POSITION && s != Renderer::VERTEX_SLOT_NORMAL && s != Renderer::VERTEX_SLOT_JOINTS &&
                    s != Renderer::VERTEX_SLOT_WEIGHTS) {
                    p.mesh->vertexBuffers[s] = source.vertexBuffers[s];
                }
            }
            p.used = true;
            const auto& outPositions = p.mesh->vertexBuffers[Renderer::VERTEX_SLOT_POSITION];
            const auto& outNormals = hasNormals ? p.mesh->vertexBuffers[Renderer::VERTEX_SLOT_NORMAL] : outPositions;
            if (!outPositions) continue;
            D::SkinParams params{source.vertexCount, jointsFormat, weightsFormat, hasNormals ? 1u : 0u, d.jointCount, {0, 0, 0}};
            auto paramsBuffer = device_->createBuffer(sizeof(params), GPU::BufferUsage::uniform, &params);
            keepAlive_.push_back(paramsBuffer);
            const auto& positions = source.vertexBuffers[Renderer::VERTEX_SLOT_POSITION];
            const auto& normals = hasNormals ? source.vertexBuffers[Renderer::VERTEX_SLOT_NORMAL] : positions;
            const auto& joints = source.vertexBuffers[Renderer::VERTEX_SLOT_JOINTS];
            const auto& weights = source.vertexBuffers[Renderer::VERTEX_SLOT_WEIGHTS];
            GPU::BindGroupDescriptor bg{};
            bg.layout = layout_;
            bg.entries = {{0, GPU::BufferBinding{positions, 0, positions->getLength()}},
                          {1, GPU::BufferBinding{normals, 0, normals->getLength()}},
                          {2, GPU::BufferBinding{joints, 0, joints->getLength()}},
                          {3, GPU::BufferBinding{weights, 0, weights->getLength()}},
                          {4, GPU::BufferBinding{d.jointMatrixBuffer, 0, d.jointMatrixBuffer->getLength()}},
                          {5, GPU::BufferBinding{outPositions, 0, outPositions->getLength()}},
                          {6, GPU::BufferBinding{outNormals, 0, outNormals->getLength()}},
                          {7, GPU::BufferBinding{paramsBuffer, 0, sizeof(params)}}};
            auto group = device_->createBindGroup(bg);
            if (!group) continue;
            groups.push_back(group);
            pass->setBindGroup(0, group, {}, 0, 0);
            pass->dispatchWorkgroups((source.vertexCount + 63) / 64, 1, 1);
            done[key] = p.mesh.get();
            d.mesh = p.mesh.get();
            d.jointMatrixBuffer = nullptr;
            d.jointCount = 0;
            ++posedDraws_;
        }
    }
    pass->end();
    auto fence = device_->createFence();
    device_->submit(encoder->finish(), fence);
    pending_ = fence;
    for (auto it = posed_.begin(); it != posed_.end();) {
        it = it->second.used ? std::next(it) : posed_.erase(it);
    }
    return out;
}

// ---------------------------------------------------------------------------
// RayTracingScene
// ---------------------------------------------------------------------------

RayTracingScene::RayTracingScene(std::shared_ptr<GPU::Device> device)
    : device_(device), skinned_(device) {
#if defined(__APPLE__) && !TARGET_OS_SIMULATOR
    if (!device_) return;
    auto module = device_->createShaderModule(shaders::kRtEffectsMetalShader, shaders::kRtEffectsMetalShaderSize);
    if (!module) return;
    // Triangle UVs of alpha-masked meshes: indices, texcoords, out (storage), parameters.
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
    uvLayout_ = device_->createBindGroupLayout(desc);
    if (!uvLayout_) return;
    GPU::PipelineLayoutDescriptor pl{};
    pl.bindGroupLayouts = {uvLayout_};
    GPU::ComputePipelineDescriptor cd{};
    cd.compute = {module, "gatherTriangleUVs"};
    cd.layout = device_->createPipelineLayout(pl);
    uvPipeline_ = device_->createComputePipeline(cd);
    const uint8_t white[4] = {255, 255, 255, 255};
    alphaPlaceholder_ = device_->createTexture(GPU::TextureType::tt2d, GPU::PixelFormat::rgba8unorm, 1, 1, 1, 1, 1,
                                               (GPU::TextureUsage)(uint32_t(GPU::TextureUsage::textureBinding) |
                                                                   uint32_t(GPU::TextureUsage::copyDst)));
    if (alphaPlaceholder_) alphaPlaceholder_->upload(0, sizeof(white), const_cast<uint8_t*>(white));
#endif
}

bool RayTracingScene::update(const RenderScene& sceneIn) {
    if (!device_) return false;
    // Skinned draws posed this frame: their structures are rebuilt.
    const RenderScene scene = skinned_.pose(sceneIn);
    const bool posed = skinned_.posedDraws() > 0;
    uint64_t geometryKey = 1469598103934665603ull, transformKey = 1469598103934665603ull;
    for (const auto& draw : rt_effects_detail::casters(scene)) {
        rt_effects_detail::mix(geometryKey, &draw.mesh, sizeof(draw.mesh));
        const void* buffers[2] = {draw.mesh->vertexBuffers[Renderer::VERTEX_SLOT_POSITION].get(), draw.mesh->indexBuffer.get()};
        rt_effects_detail::mix(geometryKey, buffers, sizeof(buffers));
        rt_effects_detail::mix(geometryKey, &draw.mesh->indexCount, sizeof(draw.mesh->indexCount));
        // Alpha-masked materials: their tests change with them.
        if (draw.material && draw.material->alphaMask) {
            const float alpha[2] = {draw.material->alphaCutoff, draw.material->baseColorAlpha};
            const void* texture = draw.material->baseColorTexture.get();
            rt_effects_detail::mix(geometryKey, alpha, sizeof(alpha));
            rt_effects_detail::mix(geometryKey, &texture, sizeof(texture));
        }
        rt_effects_detail::mix(transformKey, draw.worldTransform.data, sizeof(double) * 16);
    }
    if (tlas_ && !posed && geometryKey == geometryKey_ && transformKey == transformKey_) return true;
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
        // (Posed copies change every frame: always rebuilt.)
        const bool posedMesh = skinned_.isPosed(draw.mesh);
        if (auto it = meshes_.find(draw.mesh);
            !posedMesh && it != meshes_.end() && it->second.positions == positions && it->second.indices == indices) {
            meshes.emplace(draw.mesh, it->second);
            continue;
        }
        GPU::AccelerationStructureGeometryDescriptor geometry{};
        geometry.type = GPU::AccelerationStructureGeometryType::triangles;
        // Not opaque at the geometry level: the instance decides (opaque
        // instances are forced opaque; alpha-masked ones get candidate
        // hits to test). On Metal a non-opaque instance has no option of
        // its own, so an opaque geometry would win.
        geometry.opaque = false;
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
        blasDesc.buildFlags = posedMesh ? GPU::AccelerationStructureBuildFlag::preferFastBuild
                                        : GPU::AccelerationStructureBuildFlag::preferFastTrace;
        auto blas = device_->createBottomLevelAccelerationStructure(blasDesc);
        if (!blas) continue;
        meshes.emplace(draw.mesh, MeshEntry{blas, positions, indices});
        builds.emplace_back(blas, blasDesc);
    }
    meshes_ = std::move(meshes);

    GPU::TopLevelAccelerationStructureDescriptor tlasDesc{};
    tlasDesc.buildFlags = GPU::AccelerationStructureBuildFlag::preferFastTrace;
    // Alpha tests: per instance, the textures, each masked mesh's first triangle in the UVs.
    std::vector<rt_effects_detail::AlphaInstance> alphaInstances;
    std::vector<std::shared_ptr<GPU::Texture>> textures;
    std::map<const GpuMesh*, uint32_t> uvOffsets;
    uint32_t uvTriangles = 0;
    uint32_t alphaTested = 0;
    for (const auto& draw : draws) {
        const auto it = meshes_.find(draw.mesh);
        if (it == meshes_.end()) continue;
        GPU::AccelerationStructureInstance instance{};
        const double* m = draw.worldTransform.data;  // row-major
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 4; ++c) instance.transform[r][c] = static_cast<float>(m[r * 4 + c]);
        }
        instance.instanceId = static_cast<uint32_t>(tlasDesc.instances.size());
        instance.mask = 1;  // (as the path tracer's MASK_OPAQUE)
        // Alpha-masked: candidate hits alpha tested (rtVisible()).
        const bool masked = draw.material && draw.material->alphaMask;
        instance.opaque = !masked;
        rt_effects_detail::AlphaInstance alpha{0.0f, 1.0f, -1, 0xFFFFFFFFu};
        if (masked) {
            alpha.cutoff = std::max(draw.material->alphaCutoff, 1e-4f);
            alpha.baseAlpha = draw.material->baseColorAlpha;
            if (const auto& texture = draw.material->baseColorTexture) {
                const auto found = std::find(textures.begin(), textures.end(), texture);
                if (found != textures.end()) {
                    alpha.texture = static_cast<int32_t>(found - textures.begin());
                } else if (textures.size() < rt_effects_detail::RT_ALPHA_TEXTURES) {
                    alpha.texture = static_cast<int32_t>(textures.size());
                    textures.push_back(texture);
                }
            }
            if (alpha.texture >= 0 && rt_effects_detail::floatTexcoords(*draw.mesh)) {
                auto [uvIt, added] = uvOffsets.emplace(draw.mesh, uvTriangles);
                if (added) uvTriangles += (draw.mesh->indexBuffer && draw.mesh->indexCount > 0 ? draw.mesh->indexCount
                                                                                                 : draw.mesh->vertexCount) / 3;
                alpha.uvOffset = uvIt->second;
            }
            ++alphaTested;
        }
        alphaInstances.push_back(alpha);
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
    // The alpha tests' data (placeholders where there's none).
    alphaTestedInstances_ = alphaTested;
    alphaInstances_ = device_->createBuffer(alphaInstances.size() * sizeof(rt_effects_detail::AlphaInstance),
                                            GPU::BufferUsage::storage, alphaInstances.data());
    alphaUvs_ = device_->createBuffer(std::max<uint64_t>(uint64_t(uvTriangles) * 32, 32), GPU::BufferUsage::storage);
    alphaTextures_ = textures;
    while (alphaTextures_.size() < rt_effects_detail::RT_ALPHA_TEXTURES) alphaTextures_.push_back(alphaPlaceholder_);
    if (!alphaInstances_ || !alphaUvs_ || !alphaPlaceholder_) return false;
    std::vector<std::shared_ptr<GPU::BindGroup>> uvGroups;
    if (!uvOffsets.empty() && uvPipeline_) {
        if (auto pass = encoder->beginComputePass()) {
            pass->setPipeline(uvPipeline_);
            for (const auto& [mesh, offset] : uvOffsets) {
                const bool indexed = mesh->indexBuffer && mesh->indexCount > 0;
                rt_effects_detail::UvGatherParams params{(indexed ? mesh->indexCount : mesh->vertexCount) / 3, offset,
                                                         !indexed ? 0u : (mesh->indexFormat == GPU::IndexFormat::uint16 ? 1u : 2u), 0};
                auto paramsBuffer = device_->createBuffer(sizeof(params), GPU::BufferUsage::uniform, &params);
                keepAlive.push_back(paramsBuffer);
                const auto texcoords = rt_effects_detail::floatTexcoords(*mesh);
                const auto& indices = indexed ? mesh->indexBuffer : texcoords;
                GPU::BindGroupDescriptor bg{};
                bg.layout = uvLayout_;
                bg.entries = {{0, GPU::BufferBinding{indices, 0, indices->getLength()}},
                              {1, GPU::BufferBinding{texcoords, 0, texcoords->getLength()}},
                              {2, GPU::BufferBinding{alphaUvs_, 0, alphaUvs_->getLength()}},
                              {3, GPU::BufferBinding{paramsBuffer, 0, sizeof(params)}}};
                auto group = device_->createBindGroup(bg);
                if (!group) continue;
                uvGroups.push_back(group);
                pass->setBindGroup(0, group, {}, 0, 0);
                pass->dispatchWorkgroups((params.triangleCount + 63) / 64, 1, 1);
            }
            pass->end();
        }
    }
    auto fence = device_->createFence();
    device_->submit(encoder->finish(), fence);
    if (fence) fence->wait();
    bool onlyPosed = !builds.empty();
    for (const auto& [blas, desc] : builds) {
        bool isPosedBuild = false;
        for (const auto& [mesh, entry] : meshes_) {
            if (entry.blas == blas) isPosedBuild = skinned_.isPosed(mesh);
        }
        onlyPosed = onlyPosed && isPosedBuild;
    }
    if (onlyPosed) {
        ++skinnedRebuilds_;
    } else {
        ++(builds.empty() ? transformRebuilds_ : fullRebuilds_);
    }
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
    rt_effects_detail::appendAlphaLayout(desc, GPU::ShaderStage::rayGeneration);
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
    rt_effects_detail::appendAlphaEntries(desc.entries, *scene_);
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
    auto commands = encoder->finish();
    device_->submit(commands, fence);
    timing_.begin();
    timing_.submitted(commands, fence);
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
    rt_effects_detail::appendAlphaLayout(desc, GPU::ShaderStage::rayGeneration);
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
        rt_effects_detail::appendAlphaEntries(desc.entries, *scene_);
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
    auto commands = encoder->finish();
    device_->submit(commands, fence);
    timing_.begin();
    timing_.submitted(commands, fence);
    pending_ = fence;
    if (!reference) {
        surfaceIndex_ = 1 - surfaceIndex_;
        historyValid_ = true;
    }
    return true;
}

// ---------------------------------------------------------------------------
// RestirGlobalIllumination
// ---------------------------------------------------------------------------

namespace rt_effects_detail {

// As GiUniforms in rt_effects.metal.
struct GiUniforms {
    float invViewProj[16];
    float eye[4];
    uint32_t size[4];    // width, height, frame, history valid
    uint32_t counts[4];  // temporal cap, spatial samples
    float params[4];     // spatial radius, temporal on, spatial on, spatial below this M
};

} // namespace rt_effects_detail

RestirGlobalIllumination::RestirGlobalIllumination(std::shared_ptr<GPU::Device> device, std::shared_ptr<RayTracingScene> scene)
    : device_(std::move(device)), scene_(std::move(scene)) {
    if (!scene_ && device_) scene_ = std::make_shared<RayTracingScene>(device_);
    if (device_) tracer_ = std::make_unique<PathTracer>(device_);
#if defined(__APPLE__) && !TARGET_OS_SIMULATOR
    if (!device_ || !device_->getFeatures().count(GPU::Feature::raytracing)) return;
    auto module = device_->createShaderModule(shaders::kRtEffectsMetalShader, shaders::kRtEffectsMetalShaderSize);
    if (!module) return;
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
    for (uint32_t binding = 0; binding < 13; ++binding) {
        GPU::EntryObject texture{};
        texture.binding = binding;
        texture.type = GPU::EntryObjectType::texture;
        texture.visibility = GPU::ShaderStage::rayGeneration;
        texture.data.texture = {false, binding == 1 ? GPU::EntryObjectTextureType::ttDepth : GPU::EntryObjectTextureType::ttFloat,
                                GPU::TextureType::tt2d};
        desc.entries.push_back(texture);
    }
    rt_effects_detail::appendAlphaLayout(desc, GPU::ShaderStage::rayGeneration);
    layout_ = device_->createBindGroupLayout(desc);
    if (!layout_) return;
    GPU::PipelineLayoutDescriptor pl{};
    pl.bindGroupLayouts = {layout_};
    auto pipelineLayout = device_->createPipelineLayout(pl);
    for (auto [pipeline, entry] : {std::pair{&initial_, "restirGiInitial"}, {&spatial_, "restirGiSpatial"}}) {
        GPU::RayTracingPipelineDescriptor rt{};
        rt.rayGeneration = {module, entry};
        rt.layout = pipelineLayout;
        rt.maxRecursionDepth = 1;
        *pipeline = device_->createRayTracingPipeline(rt);
    }
    if (!initial_ || !spatial_) {
        initial_ = nullptr;
        spatial_ = nullptr;
    }
#endif
}

RestirGlobalIllumination::~RestirGlobalIllumination() {
    if (pending_) pending_->wait();
}

bool RestirGlobalIllumination::supported() const noexcept {
    return initial_ != nullptr && spatial_ != nullptr && tracer_ && tracer_->supported();
}

bool RestirGlobalIllumination::ensureTargets(uint32_t width, uint32_t height) {
    if (irradiance_ && irradiance_->getWidth() == width && irradiance_->getHeight() == height) return true;
    const auto make = [&](GPU::PixelFormat format) {
        return device_->createTexture(GPU::TextureType::tt2d, format, width, height, 1, 1, 1,
                                      (GPU::TextureUsage)(uint32_t(GPU::TextureUsage::storageBinding) |
                                                          uint32_t(GPU::TextureUsage::textureBinding) |
                                                          uint32_t(GPU::TextureUsage::copySrc)));
    };
    bool ok = true;
    for (auto& set : reservoirs_) {
        for (auto& t : set) ok = (t = make(GPU::PixelFormat::rgba32float)) && ok;
    }
    for (auto& t : scratch_) ok = (t = make(GPU::PixelFormat::rgba32float)) && ok;
    for (auto& t : surfaces_) ok = (t = make(GPU::PixelFormat::rgba32float)) && ok;
    ok = (irradiance_ = make(GPU::PixelFormat::rgba16float)) && ok;
    historyValid_ = false;
    return ok;
}

bool RestirGlobalIllumination::render(const RenderScene& scene, const Renderer& renderer, const GBufferTargets& gbuffer) {
    namespace D = rt_effects_detail;
    if (!supported() || !scene_ || !gbuffer.normalRoughness || !gbuffer.depth || !gbuffer.motion || gbuffer.width == 0 ||
        gbuffer.height == 0) {
        return false;
    }
    if (pending_) {
        pending_->wait();
        pending_.reset();
    }
    // This frame's samples (submitted first: the same queue runs them first).
    tracer_->setGlobalIlluminationBounces(settings_.bounces);
    if (!tracer_->traceGlobalIllumination(scene, renderer, gbuffer) || !tracer_->globalIlluminationSamples()) return false;
    if (!scene_->update(scene) || !scene_->tlas()) return false;
    const uint32_t w = gbuffer.width, h = gbuffer.height;
    if (!ensureTargets(w, h)) return false;

    D::GiUniforms u{};
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
    u.size[3] = historyValid_ ? 1u : 0u;
    u.counts[0] = std::max(1u, settings_.temporalCap);
    u.counts[1] = settings_.spatialSamples;
    u.params[0] = settings_.spatialRadius > 0.0f ? std::max(1.0f, settings_.spatialRadius)
                                                  : std::max(2.0f, 0.01f * static_cast<float>(w));
    u.params[1] = settings_.temporal ? 1.0f : 0.0f;
    u.params[2] = settings_.spatial && settings_.spatialSamples > 0 ? 1.0f : 0.0f;
    u.params[3] = std::clamp(settings_.spatialHistory, 0.0f, 1.0f) * static_cast<float>(u.counts[0]);
    if (!uniforms_) uniforms_ = device_->createBuffer(sizeof(u), GPU::BufferUsage::uniform);
    if (!uniforms_) return false;
    uniforms_->upload(0, sizeof(u), &u);

    const auto& now = reservoirs_[index_];
    const auto& before = reservoirs_[1 - index_];
    const auto& surface = surfaces_[index_];
    const auto& previousSurface = surfaces_[1 - index_];
    using Set = std::shared_ptr<GPU::Texture>[3];
    const auto group = [&](const Set& in, const std::shared_ptr<GPU::Texture>& surfaceIn, const Set& out,
                           const std::shared_ptr<GPU::Texture>& surfaceOut) {
        GPU::BindGroupDescriptor desc{};
        desc.layout = layout_;
        desc.entries = {{0, scene_->tlas()},
                        {1, GPU::BufferBinding{uniforms_, 0, sizeof(u)}},
                        {0, gbuffer.normalRoughness},
                        {1, gbuffer.depth},
                        {2, gbuffer.motion},
                        {3, tracer_->globalIlluminationSamples()},
                        {4, in[0]},
                        {5, in[1]},
                        {6, in[2]},
                        {7, surfaceIn},
                        {8, out[0]},
                        {9, out[1]},
                        {10, out[2]},
                        {11, surfaceOut},
                        {12, irradiance_}};
        rt_effects_detail::appendAlphaEntries(desc.entries, *scene_);
        return device_->createBindGroup(desc);
    };
    // Initial + temporal: the history in, this frame's reservoirs out.
    // Spatial: those in, the irradiance out (its reservoirs aren't fed back).
    auto initialGroup = group(before, previousSurface, now, surface);
    auto spatialGroup = group(now, surface, scratch_, previousSurface);
    auto encoder = device_->createCommandEncoder();
    if (!initialGroup || !spatialGroup || !encoder) return false;
    for (const auto& [pipeline, bindGroup] :
         {std::pair{initial_, initialGroup}, std::pair{spatial_, spatialGroup}}) {
        if (auto pass = encoder->beginRayTracingPass()) {
            pass->setPipeline(pipeline);
            pass->setBindGroup(0, bindGroup);
            pass->traceRays(w, h, 1);
            pass->end();
        }
    }
    auto fence = device_->createFence();
    auto commands = encoder->finish();
    device_->submit(commands, fence);
    timing_.begin();
    timing_.submitted(commands, fence);
    pending_ = fence;
    index_ = 1 - index_;
    historyValid_ = true;
    return true;
}

} // namespace systems::leal::campello_renderer
