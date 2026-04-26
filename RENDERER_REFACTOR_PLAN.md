# Campello Renderer Refactor Plan
## From glTF Viewer to ECS Scene Renderer

---

## Goal

Convert `campello_renderer` from a monolithic glTF viewer (single `Renderer` class that owns and traverses a `gltf::GLTF` asset) into a **general-purpose scene renderer** that consumes a frame description (`RenderScene`) produced by the ECS.

**The renderer keeps its valuable internals**: shaders, pipelines, PBR logic, skinning, IBL, post-processing, and multi-platform GPU backends. Only its **input boundary** changes.

---

## Dependency Decision: Should `campello_renderer` depend on `campello_core`?

**Answer: No — not at the compiled-library level.**

| Layer | Depends On | Responsibility |
|-------|-----------|----------------|
| **`campello_renderer`** (compiled shared lib) | `campello_gpu`, `gltf`, `campello_image` | GPU resource management, shader pipelines, and frame submission |
| **`campello_renderer/ecs.hpp`** (optional header-only bridge) | `campello_core` + `campello_renderer` | ECS component definitions and the `render_system` that queries the `World` and submits to the `Renderer` |
| **`campello_scene`** (future optional module) | `campello_core` + `campello_renderer` | Advanced scene services (culling, LOD, transform hierarchy caching) |

**Rationale:** `campello_renderer` should remain a standalone rendering API. A game should be able to use it without adopting your ECS if desired. The ECS integration lives in an **optional header** (`ecs.hpp`) that users include when they want the bridge. Because `campello_core` is header-only, there is no linking penalty — the dependency is purely at compile-time for consumers of that header.

---

## Phase 1 — Extract Engine-Native Render Structures

**Objective:** Replace the glTF-coupled `DrawItem` with renderer-native structs.

### 1.1 Public structs in `campello_renderer.hpp`

```cpp
struct GpuMesh {
    std::shared_ptr<campello_gpu::Buffer> indexBuffer;
    uint32_t indexCount;
    std::array<std::shared_ptr<campello_gpu::Buffer>, 6> vertexBuffers; // POSITION, NORMAL, UV, TANGENT, JOINTS, WEIGHTS
    uint32_t vertexCount;
    // ... topology, index format, bounds
};

struct GpuMaterial {
    std::shared_ptr<campello_gpu::BindGroup> bindGroup;     // textures + samplers
    std::shared_ptr<campello_gpu::BindGroup> flatBindGroup; // fallback for flat pipeline
    uint32_t uniformSlot; // index into the global material uniform buffer
    bool doubleSided;
    bool alphaBlend;
    bool alphaMask;
};

struct DrawCall {
    GpuMesh* mesh;
    GpuMaterial* material;
    Matrix4 worldTransform; // model matrix
    std::shared_ptr<campello_gpu::Buffer> jointMatrixBuffer; // nullptr if unskinned
    uint32_t jointCount = 0;
    uint32_t instanceCount = 1;
};

struct CameraData {
    Matrix4 view;
    Matrix4 projection;
    Vector3 position;
};

struct LightData {
    Vector3 direction;
    Vector3 position;
    Vector3 color;
    float intensity;
    float range;
    float innerConeAngle;
    float outerConeAngle;
    int type; // 0 = directional, 1 = point, 2 = spot
};

struct RenderScene {
    CameraData camera;
    std::vector<LightData> lights;
    std::vector<DrawCall> opaque;
    std::vector<DrawCall> transparent;
};
```

### 1.2 Remove glTF-specific fields from `DrawItem`

- Delete `const gltf::Primitive* primitive` and `uint64_t nodeIndex`.
- Replace `std::vector<DrawItem> opaqueQueue / transparentQueue` with `std::vector<DrawCall>`.

**Deliverable:** `campello_renderer.hpp` compiles with the new public API.

---

## Phase 2 — Decouple Resource Upload from Scene Ownership

**Objective:** The renderer must no longer own the glTF scene graph. It should only own GPU resources.

### 2.1 Split `setAsset()`

Current `setAsset()` does everything at once. Split into:

```cpp
// Upload a single primitive's geometry and return an engine-native handle.
GpuMesh* uploadMesh(const gltf::Primitive& primitive, const gltf::GLTF& asset);

// Upload a material's textures/uniforms and return an engine-native handle.
GpuMaterial* uploadMaterial(const gltf::Material& material, const gltf::GLTF& asset);

// Legacy compatibility wrapper.
void setAsset(std::shared_ptr<gltf::GLTF> asset);
```

### 2.2 Extract internal GPU resource caches

Current private members are indexed by glTF indices:
- `gpuBuffers[bufferIndex]`
- `gpuTextures[imageIndex]`
- `materialBindGroups[materialIndex]`

**Refactor:** Move these into an internal `GpuResourcePool` or keep them as `std::vector` but managed by the new upload functions. The key change is that `GpuMesh*` and `GpuMaterial*` are opaque handles returned to the caller — they are not tied to the lifetime of a specific `gltf::GLTF` object.

### 2.3 Deprecate glTF-specific traversal state

Remove or make private:
- `sceneIndex`, `cameraIndex` (camera will come from `RenderScene`)
- `nodeWorldMatrices`, `nodeWorldBounds`, `visibleNodeMask` (culling moves to the ECS layer or to a pre-submit pass)
- `animatedNodes` (animation moves out)
- `skinData`, `nodeSkinIndex` (skinning data travels with the `DrawCall`)

**Deliverable:** `Renderer` no longer stores `gltf::GLTF` as the source of truth for scene state. It only uses glTF data during the upload phase.

---

## Phase 3 — Build the `RenderScene` Submission API

**Objective:** The renderer consumes a fully-prepared `RenderScene` instead of traversing glTF nodes.

### 3.1 New public render entry point

```cpp
void render(const RenderScene& scene,
            std::shared_ptr<campello_gpu::TextureView> colorView);
```

### 3.2 Refactor `renderToTarget()`

Current flow:
1. Compute camera from glTF camera or override
2. Upload lights from `KHR_lights_punctual`
3. Compute node transforms
4. Frustum cull and fill queues
5. Upload transforms
6. Iterate queues → `renderPrimitive`

**New flow:**
1. Copy `scene.camera` into GPU uniform buffer (slot 18).
2. Copy `scene.lights` into GPU lights uniform buffer (slot 15/10).
3. Iterate `scene.opaque` and `scene.transparent`:
   - For each `DrawCall`, upload/bind `worldTransform` (slot 16) and `jointMatrixBuffer` (slot 20) if present.
   - Call refactored `renderPrimitive(const DrawCall&)`.

### 3.3 Refactor `renderPrimitive()`

Change signature:
```cpp
void renderPrimitive(
    const std::shared_ptr<RenderPassEncoder>& rpe,
    const DrawCall& draw);
```

Implementation changes:
- Pipeline selection: read `draw.material->doubleSided / alphaBlend` instead of looking up `asset->materials[materialIndex]`.
- Vertex buffer binding: read from `draw.mesh->vertexBuffers[]` instead of resolving glTF accessors.
- Material binding: bind `draw.material->bindGroup` (or `flatBindGroup`) instead of `materialBindGroups[matIndex]`.
- Transform binding: upload `draw.worldTransform` instead of `nodeTransforms[nodeIndex * 32]`.
- Skinning: bind `draw.jointMatrixBuffer` instead of the global `jointMatrixBuffer` sliced by skin index.

**Deliverable:** A working `render(const RenderScene&, colorView)` that can draw arbitrary geometry.

---

## Phase 4 — Refactor Per-Frame GPU Uploads

**Objective:** The current renderer pre-allocates transform buffers sized to glTF node count. This must become dynamic.

### 4.1 Transform ring buffer

Current: `nodeTransforms` is a `std::vector<float>` with `32 floats × nodeCount`, uploaded to `transformBuffer`.

New: `RenderScene` contains N draw calls. The renderer must upload N transforms (16 floats for Model matrix; optionally 16 for MVP if computed on CPU) to the GPU each frame.

**Approach:** Reuse the existing `FrameResources` ring buffer. Resize `transformBuffer` dynamically (or use a grow-only strategy) to hold `maxDrawCalls × matrixSize`. Copy draw call transforms sequentially before rendering.

### 4.2 Joint matrix buffer

Current: `jointMatrixBuffer` is a single large buffer holding all skinned joints for the asset, uploaded via `uploadJointMatrices()`.

New: Each `DrawCall` with skinning provides its own joint matrix data. The renderer copies these into the ring buffer sequentially, or the caller pre-allocates and owns the `Buffer`. The simplest path is:
- The ECS/glTF animation system computes joint matrices and stores them in a per-entity `Buffer` (or a CPU-side vector that gets uploaded).
- `DrawCall::jointMatrixBuffer` points to that buffer.
- `renderPrimitive()` binds it directly.

**Deliverable:** No glTF-specific upload logic remains in the render loop.

---

## Phase 5 — Extract Animation

**Objective:** Animation sampling must leave the renderer and become an ECS system (or a helper the ECS uses).

### 5.1 Create `campello_renderer/animation.hpp`

Extract from `Renderer`:
- `AnimationState`
- `sampleAnimation()`
- `applyAnimatedTRS()`
- `update()` (the glTF animation sampling part)

Into a standalone helper:

```cpp
class GltfAnimator {
public:
    void update(float dt);
    void sampleAnimation(int32_t animIndex, float time);
    // Returns animated local transforms for each node index
    const std::unordered_map<uint64_t, AnimatedTRS>& getAnimatedNodes() const;
    // ... playback controls
};
```

### 5.2 World matrix computation moves to ECS

Currently `computeNodeTransform()` recursively computes world matrices from glTF node hierarchy + animated TRS.

In the new architecture:
1. The ECS owns hierarchy via `Parent` / `Children` components.
2. A `transform_hierarchy_system` (or the glTF importer) computes world matrices and stores them in a `Transform` or `GlobalTransform` component.
3. The render system reads the final world matrix directly — no recursion needed inside the renderer.

**Deliverable:** `Renderer::update()` is deleted or reduced to internal timekeeping. Animation and hierarchy are ECS concerns.

---

## Phase 6 — Add Optional ECS Integration Header

**Objective:** Provide a drop-in bridge between `campello_core::World` and `campello_renderer::Renderer`.

### 6.1 Create `inc/campello_renderer/ecs.hpp`

```cpp
#pragma once
#include <campello_core/world.hpp>
#include <campello_renderer/campello_renderer.hpp>

namespace systems::leal::campello_renderer {

struct MeshRenderer {
    GpuMesh* mesh;
    GpuMaterial* material;
};

struct Camera {
    float fovDegrees;
    float nearPlane;
    float farPlane;
    bool isOrthographic;
    // ... ortho size if needed
};

struct DirectionalLight {
    Vector3 color;
    float intensity;
    bool castShadows;
};

struct PointLight {
    Vector3 color;
    float intensity;
    float radius;
};

struct SpotLight {
    Vector3 color;
    float intensity;
    float radius;
    float innerAngle;
    float outerAngle;
};

// ... and a system function:

void render_system(
    campello_core::World& world,
    Renderer& renderer,
    std::shared_ptr<campello_gpu::TextureView> target);

} // namespace
```

### 6.2 Implement `render_system()`

```cpp
void render_system(World& world, Renderer& renderer, TextureViewPtr target) {
    RenderScene scene;

    // 1. Camera
    for (auto [transform, camera] : world.query<Transform, Camera>()) {
        // Build view/proj from transform + camera params
        // (pick active camera based on priority or tag)
        scene.camera = buildCameraData(transform, camera);
        break;
    }

    // 2. Lights
    for (auto [transform, light] : world.query<Transform, DirectionalLight>()) {
        scene.lights.push_back(buildLightData(transform, light));
    }
    for (auto [transform, light] : world.query<Transform, PointLight>()) {
        scene.lights.push_back(buildLightData(transform, light));
    }
    // ... spot lights

    // 3. Draw calls
    for (auto [transform, meshRenderer] : world.query<Transform, MeshRenderer>()) {
        DrawCall draw;
        draw.mesh = meshRenderer.mesh;
        draw.material = meshRenderer.material;
        draw.worldTransform = transform.worldMatrix; // or compute from hierarchy
        // TODO: skinning — read SkinComponent if present
        
        if (draw.material->alphaBlend) {
            scene.transparent.push_back(draw);
        } else {
            scene.opaque.push_back(draw);
        }
    }

    // 4. Sort transparent back-to-front (using camera position)
    sortByDepth(scene.transparent, scene.camera.position);

    // 5. Submit
    renderer.render(scene, target);
}
```

**Notes:**
- `ecs.hpp` is **header-only**. It does not change the compiled `campello_renderer` shared library.
- It introduces a compile-time dependency on `campello_core` only for consumers who include this header.

**Deliverable:** A game can now render a level by writing:

```cpp
World world;
// ... spawn entities with MeshRenderer, Camera, Light components

Schedule schedule;
schedule.add_system(render_system)
    .in_stage(Stage::PreRender);

schedule.run(world);
```

---

## Phase 7 — glTF Import as ECS Population

**Objective:** Loading a glTF must spawn ECS entities, not just feed the renderer.

### 7.1 glTF import function

```cpp
// Returns the root entity of the imported scene subgraph
Entity import_gltf(World& world, Renderer& renderer, const std::string& path);
```

What it does:
1. Parse glTF (using your existing `gltf` library).
2. For each image/material: call `renderer.uploadMaterial()` → store `GpuMaterial*`.
3. For each mesh/primitive: call `renderer.uploadMesh()` → store `GpuMesh*`.
4. For each node:
   - Spawn `Entity`.
   - Insert `Transform` (local TRS).
   - If node has mesh, insert `MeshRenderer { gpuMesh, gpuMaterial }`.
   - If node has skin, insert `Skin { jointEntityRefs, inverseBindMatrices }`.
   - Call `world.set_parent(entity, parentEntity)` to preserve hierarchy.
5. Return the root entity(s).

### 7.2 glTF animation integration

After import, the glTF animation data (keyframes, channels) is stored somewhere accessible:
- Either as a `Resource<AnimationClips>` in the World.
- Or as a component on the root entity.

An ECS animation system reads this data, samples it each frame, and writes updated local transforms to the affected entities' `Transform` components.

**Deliverable:** A level can be built by importing multiple glTF files into the same `World`:

```cpp
World world;
auto hero = import_gltf(world, renderer, "assets/hero.gltf");
auto level = import_gltf(world, renderer, "assets/level.gltf");

// Add engine-native entities on top
auto sun = world.spawn();
world.insert<DirectionalLight>(sun, { {1,1,1}, 5.0f, true });
world.insert<Transform>(sun, /* ... */);
```

---

## Phase 8 — Build System & Backward Compatibility

### 8.1 CMake changes

`campello_renderer/CMakeLists.txt`:
- **No new required dependencies.** Keep `campello_gpu`, `gltf`, `campello_image`.
- Add a CMake option: `CAMPello_RENDERER_ECS_SUPPORT` (default `ON`).
- When ON, install `ecs.hpp` and document that consumers need `campello_core` in their include path.

### 8.2 Backward compatibility shim

Keep the old API working during transition:

```cpp
// In Renderer:
void setAsset(std::shared_ptr<gltf::GLTF> asset);
void update(double dt);
void render(std::shared_ptr<TextureView> colorView);
```

Internal implementation:
1. `setAsset()` uploads resources and caches the asset.
2. `update(dt)` runs the extracted `GltfAnimator` and writes to a temporary `RenderScene` cache.
3. `render(colorView)` builds a `RenderScene` from the cached glTF data and calls the new `render(scene, colorView)`.

This allows existing examples to keep compiling while you develop the ECS path.

**Deprecation timeline:** Mark old methods `[[deprecated("Use RenderScene API")]]` after the ECS path is stable. Remove in a future major version.

---

## Phase 9 — Testing Checklist

Verify the following features still work after each phase:

- [ ] Single glTF model rendering (backward compatibility)
- [ ] PBR metallic-roughness shading
- [ ] Normal mapping
- [ ] Alpha blend / alpha mask
- [ ] Double-sided materials
- [ ] Skeletal animation (skinning)
- [ ] `EXT_mesh_gpu_instancing`
- [ ] `KHR_lights_punctual` (via glTF compatibility path)
- [ ] IBL / environment maps
- [ ] Skybox
- [ ] FXAA and SSAA
- [ ] `KHR_materials_variants`
- [ ] Transmission / refraction
- [ ] Frustum culling performance

After ECS integration:
- [ ] Multiple glTF imports in one `World`
- [ ] Engine-native lights mixed with glTF meshes
- [ ] Custom camera with ECS `Camera` component
- [ ] Widget overlay on top of 3D scene
- [ ] Physics + rendering interaction (Transform updates)

---

## Timeline Estimate

| Phase | Duration | Key Deliverable |
|-------|----------|----------------|
| 1 — Extract structs | 2–3 days | `GpuMesh`, `GpuMaterial`, `DrawCall`, `RenderScene` defined |
| 2 — Decouple upload | 3–4 days | `uploadMesh/Material` split from `setAsset` |
| 3 — RenderScene API | 3–4 days | `render(const RenderScene&)` works |
| 4 — Refactor uploads | 2–3 days | Dynamic transform/joint buffer uploads |
| 5 — Extract animation | 2–3 days | `GltfAnimator` helper, `Renderer::update()` removed |
| 6 — ECS header | 2–3 days | `render_system()` functional |
| 7 — glTF importer | 3–4 days | `import_gltf()` spawns ECS entities |
| 8 — Build + compat | 2 days | CMake updated, old API shimmed |
| 9 — Testing | 3–5 days | All platforms and features verified |

**Total: ~3–4 weeks of focused work.**

---

## Migration Path for Existing Code

### Before (glTF only)
```cpp
auto renderer = std::make_shared<Renderer>(device);
auto asset = gltf::GLTF::loadGLB(data, size);
renderer->setAsset(asset);
renderer->render(colorView);
```

### After (ECS-driven)
```cpp
auto renderer = std::make_shared<Renderer>(device);
renderer->createDefaultPipelines(colorFormat);

World world;
auto hero = import_gltf(world, *renderer, "assets/hero.gltf");
auto level = import_gltf(world, *renderer, "assets/level.gltf");

// Add native engine entities
auto sun = world.spawn();
world.insert<Transform>(sun, ...);
world.insert<DirectionalLight>(sun, { {1,1,1}, 5.0f, true });

auto cam = world.spawn();
world.insert<Transform>(cam, ...);
world.insert<Camera>(cam, { 60.0f, 0.1f, 1000.0f });

// Run frame
Schedule schedule;
schedule.add_system(render_system).in_stage(Stage::PreRender);
schedule.run(world); // render_system queries World and calls renderer->render(scene, target)
```

---

## Summary

1. **`campello_renderer` stays a standalone compiled library.** The ECS bridge is an optional header.
2. **glTF becomes an import path**, not the runtime scene format. The runtime scene is the ECS `World`.
3. **The renderer's input becomes `RenderScene`** — a flat list of `DrawCall`s + camera + lights.
4. **Animation, hierarchy, and culling move out** of the renderer and into ECS systems.
5. **Backward compatibility** is maintained via a shim until the ECS path is fully validated.
