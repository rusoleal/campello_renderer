#pragma once

#include <campello/core/core.hpp>
#include <campello_renderer/campello_renderer.hpp>
#include <campello_renderer/animation.hpp>
#include <vector_math/vector_math.hpp>
#include <functional>
#include <unordered_map>

// ---------------------------------------------------------------------------
// Optional ECS bridge for campello_core + campello_renderer.
//
// This header is **header-only** and does not change the compiled
// campello_renderer shared library. Consumers who include it must have
// campello_core available in their include path.
// ---------------------------------------------------------------------------

namespace systems::leal::campello_renderer {

// ------------------------------------------------------------------
// Components
// ------------------------------------------------------------------

struct Transform {
    systems::leal::vector_math::Matrix4<double> worldMatrix;
    systems::leal::vector_math::Vector3<double> translation = {0.0, 0.0, 0.0};
    systems::leal::vector_math::Quaternion<double> rotation = {0.0, 0.0, 0.0, 1.0};
    systems::leal::vector_math::Vector3<double> scale = {1.0, 1.0, 1.0};
};

struct MeshRenderer {
    GpuMesh* mesh = nullptr;
    GpuMaterial* material = nullptr;
    // Optional skinning data. If jointMatrixBuffer is non-null and jointCount > 0,
    // the renderer will bind it to the joint matrix palette slot.
    std::shared_ptr<systems::leal::campello_gpu::Buffer> jointMatrixBuffer;
    uint32_t jointCount = 0;
};

struct Skin {
    std::vector<float> inverseBindMatrices; // float4x4 per joint, row-major
    std::vector<campello::core::Entity> joints; // entity handles for joint nodes
};

struct GltfAnimation {
    std::shared_ptr<systems::leal::gltf::GLTF> asset;
    std::shared_ptr<GltfAnimator> animator;
    std::unordered_map<uint64_t, campello::core::Entity> nodeIndexToEntity;
    std::unordered_map<uint64_t, GpuMaterial*> materialIndexToGpuMaterial;
    std::unordered_set<uint64_t> modifiedMaterialIndices;
};

struct Camera {
    float fovDegrees = 60.0f;
    float nearPlane = 0.1f;
    float farPlane = 1000.0f;
    bool isOrthographic = false;
    float orthoWidth = 10.0f;
    float orthoHeight = 10.0f;
};

struct DirectionalLight {
    systems::leal::vector_math::Vector3<double> color = {1.0, 1.0, 1.0};
    float intensity = 1.0f;
    bool castShadows = false;
};

struct PointLight {
    systems::leal::vector_math::Vector3<double> color = {1.0, 1.0, 1.0};
    float intensity = 1.0f;
    float radius = 10.0f;
};

struct SpotLight {
    systems::leal::vector_math::Vector3<double> color = {1.0, 1.0, 1.0};
    float intensity = 1.0f;
    float radius = 10.0f;
    float innerAngle = 0.0f;
    float outerAngle = 0.785398f; // 45 degrees
};

// ------------------------------------------------------------------
// Helper: build CameraData from ECS Transform + Camera
// ------------------------------------------------------------------
inline CameraData buildCameraData(const Transform& transform, const Camera& camera,
                                    double aspect = 1.0) {
    CameraData data;
    data.view = transform.worldMatrix.inverted();
    if (camera.isOrthographic) {
        data.projection = systems::leal::vector_math::Matrix4<double>::ortho(
            camera.orthoWidth, camera.orthoHeight, camera.nearPlane, camera.farPlane);
    } else {
        double fovRad = camera.fovDegrees * 3.14159265358979323846 / 180.0;
        data.projection = systems::leal::vector_math::Matrix4<double>::perspective(
            fovRad, aspect, camera.nearPlane, camera.farPlane);
    }
    // Extract position from world matrix (last column).
    data.position = systems::leal::vector_math::Vector3<double>(
        transform.worldMatrix.data[3],
        transform.worldMatrix.data[7],
        transform.worldMatrix.data[11]);
    return data;
}

// ------------------------------------------------------------------
// Helper: build LightData from ECS Transform + light components
// ------------------------------------------------------------------
inline LightData buildLightData(const Transform& transform,
                                 const DirectionalLight& light) {
    LightData data;
    data.position = systems::leal::vector_math::Vector3<double>(
        transform.worldMatrix.data[3],
        transform.worldMatrix.data[7],
        transform.worldMatrix.data[11]);
    // Direction from Z-axis of world matrix.
    double dirX = transform.worldMatrix.data[2];
    double dirY = transform.worldMatrix.data[6];
    double dirZ = transform.worldMatrix.data[10];
    double len = std::sqrt(dirX*dirX + dirY*dirY + dirZ*dirZ);
    if (len > 0.0001) {
        data.direction = systems::leal::vector_math::Vector3<double>(
            -dirX / len, -dirY / len, -dirZ / len);
    } else {
        data.direction = {0.0, 0.0, -1.0};
    }
    data.color = light.color;
    data.intensity = light.intensity;
    data.range = 0.0f;
    data.innerConeAngle = 0.0f;
    data.outerConeAngle = 0.0f;
    data.type = 0; // directional
    return data;
}

inline LightData buildLightData(const Transform& transform,
                                 const PointLight& light) {
    LightData data;
    data.position = systems::leal::vector_math::Vector3<double>(
        transform.worldMatrix.data[3],
        transform.worldMatrix.data[7],
        transform.worldMatrix.data[11]);
    data.direction = {0.0, 0.0, 0.0};
    data.color = light.color;
    data.intensity = light.intensity;
    data.range = light.radius;
    data.innerConeAngle = 0.0f;
    data.outerConeAngle = 0.0f;
    data.type = 1; // point
    return data;
}

inline LightData buildLightData(const Transform& transform,
                                 const SpotLight& light) {
    LightData data;
    data.position = systems::leal::vector_math::Vector3<double>(
        transform.worldMatrix.data[3],
        transform.worldMatrix.data[7],
        transform.worldMatrix.data[11]);
    double dirX = transform.worldMatrix.data[2];
    double dirY = transform.worldMatrix.data[6];
    double dirZ = transform.worldMatrix.data[10];
    double len = std::sqrt(dirX*dirX + dirY*dirY + dirZ*dirZ);
    if (len > 0.0001) {
        data.direction = systems::leal::vector_math::Vector3<double>(
            -dirX / len, -dirY / len, -dirZ / len);
    } else {
        data.direction = {0.0, 0.0, -1.0};
    }
    data.color = light.color;
    data.intensity = light.intensity;
    data.range = light.radius;
    data.innerConeAngle = light.innerAngle;
    data.outerConeAngle = light.outerAngle;
    data.type = 2; // spot
    return data;
}

// ------------------------------------------------------------------
// Depth-sort helper for transparent draws
// ------------------------------------------------------------------
inline void sortTransparentBackToFront(std::vector<DrawCall>& transparent,
                                       const systems::leal::vector_math::Vector3<double>& cameraPos) {
    std::sort(transparent.begin(), transparent.end(),
        [&](const DrawCall& a, const DrawCall& b) {
            auto distSq = [&](const DrawCall& d) -> double {
                double dx = d.worldTransform.data[3] - cameraPos.x();
                double dy = d.worldTransform.data[7] - cameraPos.y();
                double dz = d.worldTransform.data[11] - cameraPos.z();
                return dx*dx + dy*dy + dz*dz;
            };
            return distSq(a) > distSq(b);
        });
}

// ------------------------------------------------------------------
// Animation system
// ------------------------------------------------------------------
inline void animation_system(campello::core::World& world, float dt) {
    namespace VM = systems::leal::vector_math;
    using campello::core::Entity;

    for (auto [entity, anim] : world.query<GltfAnimation>()) {
        if (!anim.animator || !anim.asset) continue;
        anim.animator->update(dt);

        const auto& animatedNodes = anim.animator->getAnimatedNodes();
        for (const auto& [nodeIdx, trs] : animatedNodes) {
            auto it = anim.nodeIndexToEntity.find(nodeIdx);
            if (it == anim.nodeIndexToEntity.end()) continue;

            Entity targetEntity = it->second;
            if (!world.has<Transform>(targetEntity)) continue;
            auto* tx = world.get<Transform>(targetEntity);
            if (!tx) continue;

            if (trs.hasTranslation) tx->translation = trs.translation;
            if (trs.hasRotation)    tx->rotation    = trs.rotation;
            if (trs.hasScale)       tx->scale       = trs.scale;
        }

        // KHR_animation_pointer: apply animated material/light properties.
        anim.modifiedMaterialIndices = anim.animator->applyAnimatedPointers();
    }
}

// ------------------------------------------------------------------
// Material animation sync system — re-uploads animated material slots.
// Must run after animation_system and before render_system.
// ------------------------------------------------------------------
inline void material_animation_sync_system(campello::core::World& world, Renderer& renderer) {
    using campello::core::Entity;

    for (auto [entity, anim] : world.query<GltfAnimation>()) {
        if (!anim.asset || anim.modifiedMaterialIndices.empty()) continue;
        if (!anim.asset->materials) continue;

        for (uint64_t matIdx : anim.modifiedMaterialIndices) {
            if (matIdx >= anim.asset->materials->size()) continue;
            auto it = anim.materialIndexToGpuMaterial.find(matIdx);
            if (it == anim.materialIndexToGpuMaterial.end()) continue;
            GpuMaterial* gpuMat = it->second;
            if (!gpuMat) continue;
            renderer.reuploadMaterialSlot(gpuMat->uniformSlot,
                                          (*anim.asset->materials)[matIdx],
                                          *anim.asset);
        }
        anim.modifiedMaterialIndices.clear();
    }
}

// ------------------------------------------------------------------
// Transform hierarchy system — recomputes world matrices from local TRS.
// ------------------------------------------------------------------
inline void transform_hierarchy_system(campello::core::World& world) {
    namespace VM = systems::leal::vector_math;
    using campello::core::Entity;

    auto computeLocalMatrix = [](const Transform& tx) -> VM::Matrix4<double> {
        return VM::Matrix4<double>::compose(tx.translation, tx.rotation, tx.scale);
    };

    std::function<void(Entity, const VM::Matrix4<double>&)> updateChildren;
    updateChildren = [&](Entity parentEntity, const VM::Matrix4<double>& parentWorld) {
        if (!world.has<campello::core::Children>(parentEntity)) return;
        auto* children = world.get<campello::core::Children>(parentEntity);
        if (!children) return;

        for (Entity child : children->entities) {
            if (!world.has<Transform>(child)) continue;
            auto* tx = world.get<Transform>(child);
            if (!tx) continue;

            VM::Matrix4<double> localMatrix = computeLocalMatrix(*tx);
            tx->worldMatrix = parentWorld * localMatrix;

            updateChildren(child, tx->worldMatrix);
        }
    };

    world.query<Transform>().each_with_entity([&](Entity entity, Transform& tx) {
        if (world.has<campello::core::Parent>(entity)) return;

        VM::Matrix4<double> localMatrix = computeLocalMatrix(tx);
        tx.worldMatrix = localMatrix;

        updateChildren(entity, tx.worldMatrix);
    });
}

// ------------------------------------------------------------------
// Skinning system — recomputes joint matrices from current world transforms.
// ------------------------------------------------------------------
inline void skinning_system(campello::core::World& world, Renderer& renderer) {
    namespace VM = systems::leal::vector_math;
    using GPU = systems::leal::campello_gpu;
    using campello::core::Entity;

    auto device = renderer.getDevice();
    if (!device) return;

    world.query<Transform, MeshRenderer, Skin>().each_with_entity(
        [&](Entity entity, Transform& tx, MeshRenderer& mr, Skin& skin) {
            if (!mr.mesh || skin.joints.empty()) return;

            uint64_t jointCount = skin.joints.size();
            std::vector<float> jointMatrixData(jointCount * 16);

            VM::Matrix4<double> meshWorld = tx.worldMatrix;
            VM::Matrix4<double> invMeshWorld = meshWorld.inverted();

            for (uint64_t j = 0; j < jointCount; ++j) {
                Entity jointEntity = skin.joints[j];
                VM::Matrix4<double> jointWorld = VM::Matrix4<double>::identity();
                if (world.has<Transform>(jointEntity)) {
                    auto* jtx = world.get<Transform>(jointEntity);
                    if (jtx) jointWorld = jtx->worldMatrix;
                }

                VM::Matrix4<double> ibm;
                for (int row = 0; row < 4; ++row) {
                    for (int col = 0; col < 4; ++col) {
                        ibm.data[row * 4 + col] = skin.inverseBindMatrices[j * 16 + row * 4 + col];
                    }
                }

                VM::Matrix4<double> finalMatrix = invMeshWorld * jointWorld * ibm;
                for (int row = 0; row < 4; ++row) {
                    for (int col = 0; col < 4; ++col) {
                        jointMatrixData[j * 16 + col * 4 + row] = (float)finalMatrix.data[row * 4 + col];
                    }
                }
            }

            uint64_t bufferSize = jointCount * 64;
            if (!mr.jointMatrixBuffer || mr.jointMatrixBuffer->getSize() < bufferSize) {
                mr.jointMatrixBuffer = device->createBuffer(bufferSize, GPU::BufferUsage::vertex);
            }
            if (mr.jointMatrixBuffer) {
                mr.jointMatrixBuffer->upload(0, bufferSize,
                    reinterpret_cast<uint8_t*>(jointMatrixData.data()));
            }
            mr.jointCount = (uint32_t)jointCount;
        });
}

// ------------------------------------------------------------------
// Render system
// ------------------------------------------------------------------
inline void render_system(
    campello::core::World& world,
    Renderer& renderer,
    std::shared_ptr<systems::leal::campello_gpu::TextureView> target,
    uint32_t renderWidth = 0,
    uint32_t renderHeight = 0)
{
    RenderScene scene;

    // 1. Camera
    bool cameraFound = false;
    double aspect = (renderWidth > 0 && renderHeight > 0)
        ? static_cast<double>(renderWidth) / renderHeight
        : 1.0;
    for (auto [transform, camera] : world.query<Transform, Camera>()) {
        scene.camera = buildCameraData(transform, camera, aspect);
        cameraFound = true;
        break; // pick first camera
    }
    if (!cameraFound) {
        // Default camera if none exists.
        scene.camera.view = systems::leal::vector_math::Matrix4<double>::lookAt(
            {0.0, 0.0, 5.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0});
        scene.camera.projection = systems::leal::vector_math::Matrix4<double>::perspective(
            60.0 * 3.14159265358979323846 / 180.0, 1.0, 0.1, 1000.0);
        scene.camera.position = {0.0, 0.0, 5.0};
    }

    // 2. Lights
    for (auto [transform, light] : world.query<Transform, DirectionalLight>()) {
        scene.lights.push_back(buildLightData(transform, light));
    }
    for (auto [transform, light] : world.query<Transform, PointLight>()) {
        scene.lights.push_back(buildLightData(transform, light));
    }
    for (auto [transform, light] : world.query<Transform, SpotLight>()) {
        scene.lights.push_back(buildLightData(transform, light));
    }

    // 3. Draw calls
    for (auto [transform, meshRenderer] : world.query<Transform, MeshRenderer>()) {
        if (!meshRenderer.mesh || !meshRenderer.material) continue;
        DrawCall draw;
        draw.mesh = meshRenderer.mesh;
        draw.material = meshRenderer.material;
        draw.worldTransform = transform.worldMatrix;
        draw.instanceCount = 1;

        if (meshRenderer.jointMatrixBuffer && meshRenderer.jointCount > 0) {
            draw.jointMatrixBuffer = meshRenderer.jointMatrixBuffer;
            draw.jointCount = meshRenderer.jointCount;
        }

        if (draw.material->alphaBlend) {
            scene.transparent.push_back(draw);
        } else {
            scene.opaque.push_back(draw);
        }
    }

    // 4. Sort transparent back-to-front
    sortTransparentBackToFront(scene.transparent, scene.camera.position);

    // 5. Submit
    renderer.render(scene, target);
}

} // namespace systems::leal::campello_renderer

// ------------------------------------------------------------------
// campello_core component traits (optional, for editor/reflection)
// ------------------------------------------------------------------
namespace campello::core {

template<> struct ComponentTraits<systems::leal::campello_renderer::Transform>
    : ComponentTraitsBase<systems::leal::campello_renderer::Transform> {
    static constexpr std::string_view name = "Transform";
};

template<> struct ComponentTraits<systems::leal::campello_renderer::MeshRenderer>
    : ComponentTraitsBase<systems::leal::campello_renderer::MeshRenderer> {
    static constexpr std::string_view name = "MeshRenderer";
};

template<> struct ComponentTraits<systems::leal::campello_renderer::Camera>
    : ComponentTraitsBase<systems::leal::campello_renderer::Camera> {
    static constexpr std::string_view name = "Camera";
};

template<> struct ComponentTraits<systems::leal::campello_renderer::DirectionalLight>
    : ComponentTraitsBase<systems::leal::campello_renderer::DirectionalLight> {
    static constexpr std::string_view name = "DirectionalLight";
};

template<> struct ComponentTraits<systems::leal::campello_renderer::PointLight>
    : ComponentTraitsBase<systems::leal::campello_renderer::PointLight> {
    static constexpr std::string_view name = "PointLight";
};

template<> struct ComponentTraits<systems::leal::campello_renderer::SpotLight>
    : ComponentTraitsBase<systems::leal::campello_renderer::SpotLight> {
    static constexpr std::string_view name = "SpotLight";
};

template<> struct ComponentTraits<systems::leal::campello_renderer::Skin>
    : ComponentTraitsBase<systems::leal::campello_renderer::Skin> {
    static constexpr std::string_view name = "Skin";
};

template<> struct ComponentTraits<systems::leal::campello_renderer::GltfAnimation>
    : ComponentTraitsBase<systems::leal::campello_renderer::GltfAnimation> {
    static constexpr std::string_view name = "GltfAnimation";
};

} // namespace campello::core

// ------------------------------------------------------------------
// glTF importer — populates ECS World from a glTF file
// ------------------------------------------------------------------

#include <fstream>
#include <future>
#include <sstream>

namespace systems::leal::campello_renderer {

inline campello::core::Entity import_gltf(
    campello::core::World& world,
    Renderer& renderer,
    const std::string& path)
{
    namespace VM = systems::leal::vector_math;
    using campello::core::Entity;

    // 1. Load glTF asset.
    std::shared_ptr<systems::leal::gltf::GLTF> asset;
    {
        std::ifstream file(path, std::ios::binary);
        if (!file) return campello::core::null_entity;
        std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)),
                                   std::istreambuf_iterator<char>());
        if (data.size() < 4) return campello::core::null_entity;

        // Check magic for GLB.
        bool isGlb = (data[0] == 'g' && data[1] == 'l' && data[2] == 'T' && data[3] == 'F');
        if (isGlb) {
            asset = systems::leal::gltf::GLTF::loadGLB(data.data(), data.size());
        } else {
            std::string text(data.begin(), data.end());
            // Compute base directory for resolving external URIs in the callback.
            std::string baseDir;
            size_t lastSlash = path.find_last_of("/\\");
            if (lastSlash != std::string::npos) {
                baseDir = path.substr(0, lastSlash + 1);
            }
            auto loadCallback = [baseDir](const std::string& uri) -> std::future<std::vector<uint8_t>> {
                return std::async(std::launch::deferred, [baseDir, uri]() {
                    std::string fullPath = baseDir + uri;
                    std::ifstream file(fullPath, std::ios::binary);
                    if (!file) return std::vector<uint8_t>();
                    return std::vector<uint8_t>(
                        std::istreambuf_iterator<char>(file),
                        std::istreambuf_iterator<char>());
                });
            };
            asset = systems::leal::gltf::GLTF::loadGLTF(text, loadCallback);
        }
    }
    if (!asset) return campello::core::null_entity;

    // Compute base directory for resolving external image URIs.
    {
        size_t lastSlash = path.find_last_of("/\\");
        if (lastSlash != std::string::npos) {
            renderer.setAssetBasePath(path.substr(0, lastSlash + 1));
        } else {
            renderer.setAssetBasePath("");
        }
    }

    // 2. Upload materials.
    std::vector<GpuMaterial*> gpuMaterials;
    if (asset->materials) {
        gpuMaterials.reserve(asset->materials->size());
        for (auto& mat : *asset->materials) {
            gpuMaterials.push_back(renderer.uploadMaterial(mat, *asset));
        }
    }

    // 3. Upload meshes (primitives).
    // Map: mesh index -> vector of GpuMesh* per primitive.
    std::vector<std::vector<GpuMesh*>> gpuMeshes;
    if (asset->meshes) {
        gpuMeshes.resize(asset->meshes->size());
        for (size_t m = 0; m < asset->meshes->size(); ++m) {
            auto& mesh = (*asset->meshes)[m];
            gpuMeshes[m].reserve(mesh.primitives.size());
            for (auto& prim : mesh.primitives) {
                gpuMeshes[m].push_back(renderer.uploadMesh(prim, *asset));
            }
        }
    }

    // 4. Compute node local + world matrices (top-down traversal of default scene).
    std::vector<VM::Matrix4<double>> nodeLocalMatrices;
    std::vector<VM::Matrix4<double>> nodeWorldMatrices;
    if (asset->nodes) {
        nodeLocalMatrices.resize(asset->nodes->size());
        nodeWorldMatrices.resize(asset->nodes->size());
        for (size_t n = 0; n < asset->nodes->size(); ++n) {
            auto& node = (*asset->nodes)[n];
            VM::Vector3<double> t = node.translation;
            VM::Quaternion<double> r = node.rotation;
            VM::Vector3<double> s = node.scale;
            nodeLocalMatrices[n] = VM::Matrix4<double>::compose(t, r, s);
        }

        auto computeWorld = [&](auto&& self, uint64_t nodeIndex,
                                const VM::Matrix4<double>& parentWorld) -> void {
            if (nodeIndex >= nodeWorldMatrices.size()) return;
            nodeWorldMatrices[nodeIndex] = parentWorld * nodeLocalMatrices[nodeIndex];
            auto& node = (*asset->nodes)[nodeIndex];
            for (auto childIdx : node.children) {
                self(self, childIdx, nodeWorldMatrices[nodeIndex]);
            }
        };

        size_t sceneIdx = 0;
        if (asset->scene >= 0 && (size_t)asset->scene < asset->scenes->size()) {
            sceneIdx = (size_t)asset->scene;
        }
        if (asset->scenes && sceneIdx < asset->scenes->size()) {
            auto& scene = (*asset->scenes)[sceneIdx];
            if (scene.nodes) {
                for (auto rootIdx : *scene.nodes) {
                    computeWorld(computeWorld, rootIdx, VM::Matrix4<double>::identity());
                }
            }
        }
    }

    // 5. Spawn entities for nodes.
    std::vector<campello::core::Entity> nodeEntities;
    if (asset->nodes) {
        nodeEntities.reserve(asset->nodes->size());
        for (size_t n = 0; n < asset->nodes->size(); ++n) {
            auto& node = (*asset->nodes)[n];
            Entity e = world.spawn();
            nodeEntities.push_back(e);

            // Insert Transform with local TRS + pre-computed world matrix.
            Transform tx;
            tx.translation = node.translation;
            tx.rotation    = node.rotation;
            tx.scale       = node.scale;
            tx.worldMatrix = nodeWorldMatrices[n];
            world.insert<Transform>(e, tx);

            // Insert MeshRenderer(s) if node has mesh.
            if (node.mesh >= 0 && asset->meshes &&
                (size_t)node.mesh < asset->meshes->size() &&
                (size_t)node.mesh < gpuMeshes.size()) {
                auto& primList = gpuMeshes[(size_t)node.mesh];
                auto& meshObj = (*asset->meshes)[(size_t)node.mesh];

                if (primList.size() == 1) {
                    // Single primitive — attach directly to node entity.
                    GpuMesh* mesh = primList[0];
                    GpuMaterial* mat = nullptr;
                    int64_t matIdx = meshObj.primitives[0].material;
                    if (matIdx >= 0 && (size_t)matIdx < gpuMaterials.size()) {
                        mat = gpuMaterials[(size_t)matIdx];
                    }
                    if (mesh && mat) {
                        world.insert<MeshRenderer>(e, MeshRenderer{mesh, mat});
                    }
                } else if (primList.size() > 1) {
                    // Multiple primitives — spawn child entities.
                    for (size_t p = 0; p < primList.size(); ++p) {
                        GpuMesh* mesh = primList[p];
                        GpuMaterial* mat = nullptr;
                        int64_t matIdx = meshObj.primitives[p].material;
                        if (matIdx >= 0 && (size_t)matIdx < gpuMaterials.size()) {
                            mat = gpuMaterials[(size_t)matIdx];
                        }
                        if (mesh && mat) {
                            Entity child = world.spawn();
                            world.insert<Transform>(child, tx); // same world transform
                            world.insert<MeshRenderer>(child, MeshRenderer{mesh, mat});
                            world.set_parent(child, e);
                        }
                    }
                }
            }
        }

        // Set up hierarchy.
        for (size_t n = 0; n < asset->nodes->size(); ++n) {
            auto& node = (*asset->nodes)[n];
            for (auto childIdx : node.children) {
                if (childIdx < nodeEntities.size()) {
                    world.set_parent(nodeEntities[childIdx], nodeEntities[n]);
                }
            }
        }
    }

    // Determine root entity early so we can attach animation data.
    Entity rootEntity = campello::core::null_entity;
    if (asset->scenes && !asset->scenes->empty()) {
        size_t sceneIdx = 0;
        if (asset->scene >= 0 && (size_t)asset->scene < asset->scenes->size()) {
            sceneIdx = (size_t)asset->scene;
        }
        auto& scene = (*asset->scenes)[sceneIdx];
        if (scene.nodes && !scene.nodes->empty()) {
            size_t rootIdx = (size_t)(*scene.nodes)[0];
            if (rootIdx < nodeEntities.size()) {
                rootEntity = nodeEntities[rootIdx];
            }
        }
    }
    if (rootEntity == campello::core::null_entity && !nodeEntities.empty()) {
        rootEntity = nodeEntities[0];
    }

    // Attach GltfAnimation component on root entity if animations exist.
    if (rootEntity != campello::core::null_entity && asset->animations &&
        !asset->animations->empty() && asset->nodes) {
        GltfAnimation animComp;
        animComp.asset = asset;
        animComp.animator = std::make_shared<GltfAnimator>(asset);
        for (size_t n = 0; n < asset->nodes->size(); ++n) {
            animComp.nodeIndexToEntity[(uint64_t)n] = nodeEntities[n];
        }
        for (size_t m = 0; m < gpuMaterials.size(); ++m) {
            animComp.materialIndexToGpuMaterial[(uint64_t)m] = gpuMaterials[m];
        }
        world.insert<GltfAnimation>(rootEntity, std::move(animComp));
    }

    // 5. Process skins — attach Skin components for runtime joint matrix updates.
    if (asset->skins && asset->accessors && asset->bufferViews && asset->buffers && !nodeEntities.empty()) {
        for (size_t skinIdx = 0; skinIdx < asset->skins->size(); ++skinIdx) {
            auto& gltfSkin = (*asset->skins)[skinIdx];
            if (!gltfSkin.joints || gltfSkin.joints->empty()) continue;

            uint64_t jointCount = gltfSkin.joints->size();

            // Read inverse bind matrices (column-major in glTF).
            std::vector<float> inverseBindMatrices(jointCount * 16);
            bool hasIBM = false;
            if (gltfSkin.inverseBindMatrices >= 0 &&
                (size_t)gltfSkin.inverseBindMatrices < asset->accessors->size()) {
                auto& ibmAcc = (*asset->accessors)[(size_t)gltfSkin.inverseBindMatrices];
                if (ibmAcc.bufferView >= 0 && (size_t)ibmAcc.bufferView < asset->bufferViews->size()) {
                    auto& ibmBV = (*asset->bufferViews)[(size_t)ibmAcc.bufferView];
                    if (ibmBV.buffer >= 0 && (size_t)ibmBV.buffer < asset->buffers->size()) {
                        auto& buf = (*asset->buffers)[(size_t)ibmBV.buffer];
                        const uint8_t* src = buf.data.data() + ibmBV.byteOffset + ibmAcc.byteOffset;
                        const float* fSrc = reinterpret_cast<const float*>(src);
                        for (uint64_t j = 0; j < jointCount; ++j) {
                            for (int row = 0; row < 4; ++row) {
                                for (int col = 0; col < 4; ++col) {
                                    // glTF stores column-major; our Matrix4 is row-major.
                                    inverseBindMatrices[j * 16 + row * 4 + col] =
                                        fSrc[j * 16 + col * 4 + row];
                                }
                            }
                        }
                        hasIBM = true;
                    }
                }
            }
            if (!hasIBM) {
                // No inverse bind matrices provided — use identity.
                for (uint64_t j = 0; j < jointCount; ++j) {
                    for (int i = 0; i < 4; ++i) {
                        inverseBindMatrices[j * 16 + i * 4 + i] = 1.0f;
                    }
                }
            }

            // Find all nodes that use this skin and attach Skin components.
            for (size_t nodeIdx = 0; nodeIdx < asset->nodes->size(); ++nodeIdx) {
                auto& node = (*asset->nodes)[nodeIdx];
                if (node.skin != (int64_t)skinIdx) continue;
                if (nodeIdx >= nodeEntities.size()) continue;

                Entity nodeEntity = nodeEntities[nodeIdx];

                Skin skin;
                skin.inverseBindMatrices = inverseBindMatrices;
                skin.joints.reserve(jointCount);
                for (uint64_t j = 0; j < jointCount; ++j) {
                    uint64_t jointNodeIdx = (*gltfSkin.joints)[j];
                    if (jointNodeIdx < nodeEntities.size()) {
                        skin.joints.push_back(nodeEntities[jointNodeIdx]);
                    } else {
                        skin.joints.push_back(campello::core::null_entity);
                    }
                }

                auto attachSkin = [&](Entity targetEntity) {
                    if (world.has<MeshRenderer>(targetEntity)) {
                        world.insert<Skin>(targetEntity, skin);
                    }
                };

                attachSkin(nodeEntity);
                // Also attach to child entities (multi-primitive meshes).
                if (world.has<campello::core::Children>(nodeEntity)) {
                    auto* children = world.get<campello::core::Children>(nodeEntity);
                    if (children) {
                        for (Entity child : children->entities) {
                            attachSkin(child);
                        }
                    }
                }
            }
        }
    }

    // 6. Return root entity of the default scene.
    return rootEntity;
}

} // namespace systems::leal::campello_renderer
