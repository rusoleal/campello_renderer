# campello_renderer TODO

## Current State (v0.5.0)

### What works
- `setAsset(shared_ptr<GLTF>)` — loads asset, allocates GPU buffer/texture/sampler/bind-group slot arrays
- `setScene(index)` — uploads GPU buffers, decodes + uploads textures via `campello_image`, builds per-texture bind groups from GLTF samplers, computes scene bounding radius, allocates material uniform buffer and per-node transform buffer
- `setCamera(index)` — selects active GLTF camera
- `resize(w, h)` — creates `depth32float` texture and view
- `createDefaultPipelines(colorFormat)` — compiles flat + textured pipeline variants; Metal (macOS) fully implemented; Vulkan (Android) one variant (flat = textured workaround); DirectX stubs (nonfunctional)
- `render()` / `render(colorView)` — full scene graph traversal, per-node MVP, camera from GLTF or override, depth pass, POSITION + NORMAL + TEXCOORD_0 + TANGENT binding, fallback UV buffer, per-primitive pipeline selection, indexed + non-indexed draw
- `setCameraMatrices(view, proj)` / `clearCameraOverride()` — external camera override
- `getBoundsRadius()` — scene bounds
- `getGpuBuffer/Texture/BindGroup` accessors
- `getVersion()` — works; test asserts exact match

---

## Immediate (pre-commit)

- [x] **Commit pending changes** — `campello_gpu` v0.9.0, `gltf` v0.4.1, `LICENSE` — PBR textures (metallicRoughness, normal, emissive, occlusion) implemented.
- [x] **Triple-buffer frame-in-flight ring** — 3 `FrameResources` with fences; prevents CPU/GPU desync.
- [x] **Build directory split** — separate `build/macos/debug` and `build/macos/release` to prevent mixed objects.
- [x] **Local `campello_gpu` override** — `dependencies/campello_gpu.cmake` points to `/Users/rubenleal/Projects/campello_gpu` for rapid iteration.
- [x] **Metal validation fixes** — removed `instLayout` from vertex descriptor (fixed missing binding 3/4 crash); added `fallbackTangentBuffer`; fixed vertex stage material buffer binding at slot 17.
- [x] **Depth/stencil format mismatch fix** — patched `campello_gpu` Metal backend to skip stencil attachment when depth format is `depth32float` (no stencil component).
- [x] **Redundant buffer binding deduplication** — `setVertexBufferIfChanged()` tracks last-bound `(buffer, offset)` per slot to avoid Metal debug-layer asserts.
- [x] **Frustum culling** — AABB-vs-frustum plane tests skip whole node subtrees outside the camera view. Fixed row-major matrix plane extraction bug.
- [x] **View mode hotkeys** — 17 debug visualization modes: normal, worldNormal, baseColor, metallic, roughness, occlusion, emissive, alpha, uv0, specularFactor, specularColor, sheenColor, sheenRoughness, clearcoat, clearcoatRoughness, clearcoatNormal, transmission.

---

## Bugs / Upstream Blockers

- [ ] **Vulkan vertex input not applied** — `campello_gpu` Vulkan backend hardcodes `vertexBindingDescriptionCount = 0`, so vertex buffer layouts passed via `createDefaultPipelines()` are ignored. Fix upstream in `campello_gpu`. Tracked in the Vulkan `createDefaultPipelines` path.
- [ ] **DirectX rendering nonfunctional** — DXIL binaries not compiled; `kDefaultDirectXVertShaderSize == 0` causes early return. `pipelineFlat` and `pipelineTextured` remain null on Windows.

---

## Asset Loading

- [x] **`data:uri` image support** — The gltf library already decodes data:uri images automatically. The renderer now uses `image.data` directly with `campello_image::Image::fromMemory()`.
- [x] **External image URIs** — `.gltf` files referencing external image files are loaded via `Renderer::imageBasePath` / `setAssetBasePath()`. Both `setAsset()` and `uploadMaterial()` resolve external image paths relative to the base path.
- [x] **Non-standard buffer view strides** — interleaved vertex data (`byteStride > 0` in buffer views) is now deinterleaved into contiguous GPU buffers per accessor at load time. The renderer scans all mesh primitives in `setScene()`, detects accessors in buffer views with `byteStride > 0`, and extracts each attribute into its own tightly-packed GPU buffer. `renderPrimitive()` prefers the deinterleaved buffer when binding standard GLTF attributes.

---

## Shaders

### Vulkan (Android)
- [ ] **Flat fragment shader** — compile a dedicated SPIR-V fragment shader for the flat (no-texture) variant. Currently `pipelineFlat = pipelineTextured` as workaround.
- [ ] **Material slot 17 in GLSL** — verify the `baseColorFactor` uniform at `VERTEX_SLOT_MATERIAL` (slot 17) is read correctly by the current SPIR-V; the Vulkan pipeline descriptor doesn't include slot 17 layout.
- [ ] **Add `glslc` compile step to CMake** — automate SPIR-V compilation for Vulkan targets instead of embedding pre-compiled binaries.

### DirectX (Windows)
- [ ] **Compile DXIL binaries** — HLSL source exists at `shaders/directx/default.hlsl`. Run:
  ```
  dxc -T vs_6_0 -E vertexMain default.hlsl -Fo default_vs.dxil
  dxc -T ps_6_0 -E pixelMain  default.hlsl -Fo default_ps.dxil
  ```
  Then embed as byte arrays in `src/shaders/directx_default.h`.
- [ ] **Add `dxc` compile step to CMake** — guard with `if(WIN32)`.

### Apple (macOS / iOS)
- [x] **iOS SDK compile** — `build_metal_shaders.sh` compiles for macOS, iOS simulator, and iOS device SDKs.
- [x] **Add Metal compile step to CMake** — `metal_shaders` custom target + `add_dependencies(campello_renderer metal_shaders)` auto-rebuilds embedded `.metallib` header when `default.metal` changes.

### Longer term
- [ ] **Single shader source** — evaluate HLSL → MSL + SPIR-V via DXC + SPIRV-Cross to eliminate per-platform divergence.

---

## Animation

- [x] **Implement `update(double dt)`** — GLTF animation channels fully implemented with multi-animation support:
  - Multiple animations can play simultaneously on different node sets
  - `playAnimation(index)` / `pauseAnimation(index)` / `stopAnimation(index)` for per-animation playback control
  - `stopAllAnimations()` to stop all animations at once
  - `setAnimationTime(index, t)` / `getAnimationTime(index)` for per-animation seeking
  - `setAnimationLoop(index, bool)` / `isAnimationLooping(index)` for per-animation loop control
  - `isAnimationPlaying(index)` to check individual animation state
  - `getAnimationCount()` / `getAnimationName(i)` / `getAnimationDuration(i)` for introspection
  - Keyframe sampling with LINEAR interpolation (translation, rotation via slerp, scale)
  - STEP interpolation supported
  - Animated TRS merged from all playing animations before `render()` computes world matrices
  - Last-animation-wins when multiple animations target the same node/property
- [x] **Skeletal meshes** — joint matrix palette implemented. `setScene()` reads `inverseBindMatrices` and caches per-skin joint data. Each frame, `computeSkinningTransforms()` computes `jointMatrix = inverse(skinNodeWorld) * jointWorld * inverseBindMatrix` per glTF spec and uploads to a triple-buffered GPU palette. `renderPrimitive()` binds `JOINTS_0` (slot 4), `WEIGHTS_0` (slot 5), and joint matrix buffer (slot 20). Metal shader blends up to 4 joint matrices in vertexMain before instance/node transforms. Fallback zero buffers and identity joint matrix handle non-skinned primitives cleanly. Pipeline vertex descriptors dynamically match detected `JOINTS_0` (`UNSIGNED_BYTE`/`UNSIGNED_SHORT`) and `WEIGHTS_0` (`FLOAT`/`UNSIGNED_BYTE` normalized/`UNSIGNED_SHORT` normalized) component types.

---

## Rendering Performance (large scenes)

### Priority 1 — prerequisites for visibility culling

- [x] **Add per-primitive bounds** — local AABB data is now cached per primitive from `POSITION` accessor min/max when available, with a CPU fallback scan for float position buffers.
- [x] **Cache per-node world bounds** — node-local bounds are derived from mesh primitives in `setScene()`, then world bounds are refreshed during transform updates and merged up the hierarchy each frame.
- [x] **Build a visible draw list before the render pass** — scene traversal now gathers opaque / transparent draw items before the render pass instead of drawing recursively during traversal.

### Priority 2 — primary large-scene optimization

- [x] **Implement camera frustum culling** — the renderer now extracts the six frustum planes from the active view-projection matrix and skips whole node subtrees whose cached world bounds are outside the frustum before issuing draw calls.
- [x] **Reduce transform buffer uploads** — `uploadVisibleNodeTransforms()` now finds the highest visible node index and uploads only `[0, (maxIndex+1) * 128]` bytes instead of the entire buffer.
- [x] **Cull `EXT_mesh_gpu_instancing` instances** — `updateVisibleInstances()` now tests each instance's world-space translation against the frustum planes. Visible instances are compacted into `visibleMatrices` and only the surviving count is uploaded/drawn.

### Priority 3 — draw-call and state-change reduction

- [x] **Add mesh / material batching and state sorting** — opaque draws sorted by `(pipelineVariant << 48) | (materialPtr << 16) | meshPtr`. Both legacy `renderToTarget()` and ECS `render(const RenderScene&)` paths sort before submission.
- [x] **Guard debug logging in hot paths** — `std::cout` diagnostics removed from `computeNodeTransform()` and `renderPrimitive()` hot paths.

### Priority 4 — longer-term scalability

- [x] **Add renderer performance counters** — `RenderStats` exposes `opaqueDrawCount`, `transparentDrawCount`, `totalDrawCount`, `culledNodeCount`, `visibleNodeCount`, `cpuFrameTimeMs`. Populated in both legacy and ECS render paths.
- [ ] **Add distance-based LOD hooks** — define per-mesh LOD selection and optional far-distance impostor / low-poly fallback support so large environments do not submit full-resolution geometry at every distance.

---

## GLTF Extension Rendering Support

The gltf library (v0.4.1) parses **23 extensions** into structured data. The renderer currently uses a small subset. Below is what remains to implement, grouped by complexity.

### Currently rendered
- [x] Core PBR `baseColorFactor` — uploaded to `materialUniformBuffer` (slot 17) per material
- [x] Core PBR `baseColorTexture` — textured pipeline variant, bound via per-GLTF-texture bind groups
- [x] `EXT_texture_webp` — renderer prefers `ext_texture_webp` image index when building bind groups
- [x] GLTF samplers — `gpuSamplers` built from `asset->samplers` with wrap/filter mode mapping
- [x] Debug rendering mode — `setDebugMode(true)` renders geometry with normal visualization (flat shading, no textures/lighting)

### Core material properties (not yet rendered)

- [x] **`metallicRoughnessTexture`** — Full PBR metallic-roughness workflow implemented with Lambert diffuse + Blinn-Phong specular.
- [x] **`normalTexture`** — Tangent-space normal mapping with TBN matrix construction. Supports `normalScale` from `NormalTextureInfo`.
- [x] **`occlusionTexture`** — parsed as `Material::occlusionTexture` (`OcclusionTextureInfo` with `strength` field). Multiply lighting by R channel of the occlusion texture. Implemented in fragment shader with `occlusionStrength` uniform.
- [x] **`emissiveFactor` + `emissiveTexture`** — parsed as `Material::emissiveFactor` (Vector3) and `emissiveTexture` (TextureInfo). Add emissive contribution after lighting. `KHR_materials_emissive_strength` scalar multiplier is already parsed as `Material::khrMaterialsEmissiveStrength`. Fixed Metal `float3` alignment (16-byte aligned, starts at offset 96).
- [x] **Alpha modes: MASK** — `Material::alphaMode` (OPAQUE/MASK/BLEND) and `alphaCutoff`. MASK: `discard` in fragment shader when alpha < cutoff.
- [x] **Alpha modes: BLEND (macOS/Metal)** — alpha blending enabled in pipeline color state with `srcAlpha * oneMinusSrcAlpha`. Blend pipelines created for all variants (flat/textured × single/double-sided). Depth write disabled for transparent materials. Android/Windows: blend pipeline variants alias to opaque (TODO per-platform). Back-to-front sort: transparent primitives collected during opaque traversal, sorted by squared camera distance (descending), drawn after all opaque geometry.
- [x] **Double-sided materials** — `Material::doubleSided`. Pipeline variants with `CullMode::none` for double-sided materials.

### KHR_materials_unlit

- [x] **`KHR_materials_unlit`** — parsed as `Material::khrMaterialsUnlit` (bool). When true, render using only `baseColorFactor × baseColorTexture` with no lighting calculation. Implemented as shader branch using materialFlags.z.

### KHR_texture_transform

- [x] **`KHR_texture_transform`** — UV transform (T × R × S) baked into `materialUniformBuffer` as two float4 rows per material slot (48-byte stride). Applied in the vertex shader for Metal and Vulkan. DirectX HLSL also updated. Note: `texCoord` override field not yet respected (always applies to TEXCOORD_0).

### KHR_lights_punctual

- [x] **`KHR_lights_punctual`** — parsed into `GLTF::khrLightsPunctual` (vector of `KHRLightPunctual`). Each light has `type` (directional/point/spot), `color`, `intensity`, `range`, and spot cone angles. Node references via `Node::light` index.
  - Lights collected by traversing scene graph, finding nodes with `node.light >= 0`
  - Light transforms computed from node world matrices (directional: -Z axis, point/spot: position)
  - Up to 4 lights supported, uploaded to vertex buffer slot 15 (`VERTEX_SLOT_LIGHTS`)
  - Shader iterates lights, computes inverse-square attenuation (point/spot), spot cone factor (spot), Reinhard tone mapping at output
  - C++ injects a default directional light (5 lux) when no lights exist in the scene (no hardcoded shader fallback)

### KHR_materials_ior

- [x] **`KHR_materials_ior`** — uploaded to material uniform buffer slot 27 (byte 108). Fragment shader replaces hardcoded `F0 = 0.04` with `((ior-1)/(ior+1))²`. Default IOR 1.5 produces F0=0.04, so no regression on assets without the extension.

### KHR_materials_specular

- [x] **`KHR_materials_specular`** — `specularFactor` (scalar) and `specularColorFactor` (vec3) uploaded to `materialUniformBuffer` (offsets 112, 128). `specularTexture` (A channel) and `specularColorTexture` (RGB, sRGB) bound at slots 11–14. Fragment shader computes `F0_dielectric = min(IOR_f0 × specularColor, 1) × specular` before mixing with metallic F0. Default values (factor=1, color=[1,1,1]) are backward-compatible with KHR_materials_ior behaviour.

### KHR_materials_clearcoat

- [x] **`KHR_materials_clearcoat`** — GGX NDF + Smith-GGX visibility + Schlick Fresnel (fixed F0=0.04, IOR 1.5). Three textures: clearcoatTexture (R, binding 17), clearcoatRoughnessTexture (G, binding 18), clearcoatNormalTexture (binding 19). All reuse baseColorSampler. Clearcoat normal decoded with TBN and `clearcoatNormalScale`. Base layer attenuated by `(1 - ccFactor × F_Schlick(0.04, NdotV))`. Uniforms at material buffer offsets 168–191.

### KHR_materials_sheen

- [x] **`KHR_materials_sheen`** — Charlie NDF + Neubelt visibility term. `sheenColorFactor` × `sheenColorTexture` (RGB sRGB, binding 15) tints the sheen lobe; `sheenRoughnessFactor` × `sheenRoughnessTexture` (R linear, binding 16) controls lobe width. Uniforms packed at material buffer offsets 144–167. Both sheen textures reuse `baseColorSampler` (Metal sampler slot limit of 16 is reached).

### KHR_materials_transmission

- [x] **`KHR_materials_transmission`** (simplified) — parsed as `Material::khrMaterialsTransmission` (`transmissionFactor`, `transmissionTexture`). 
  - Simplified implementation: treats transmission as additional alpha transparency
  - `transmissionFactor` 0.0 = fully opaque, 1.0 = fully transparent
  - Multiplied with base color alpha in both `fragmentMain_flat` and `fragmentMain_textured`
  - No render-to-texture (simpler, good for thin glass/panels)
  - Uniform at material buffer offset 228 (index 57), correctly aligned for Metal
  
- [x] **`KHR_materials_transmission`** (sophisticated / physically based) — screen-space refraction implemented. Opaque scene rendered to offscreen `opaqueSceneTexture` (binding 22) with mipmap generation. Transparent pass samples background with refracted UVs using IOR. `transmissionTexture` (binding 20) scales factor. Beer-Lambert attenuation integrated with `KHR_materials_volume`.

### KHR_materials_volume

- [x] **`KHR_materials_volume`** — `thicknessTexture` sampled at binding 23. Beer-Lambert attenuation `exp(-thickness * attn)` applied in fragment shader. Uniforms: `thicknessFactor`, `hasThicknessTex`, `attenuationDistance`, `attenuationColor`.

### KHR_materials_iridescence

- [x] **`KHR_materials_iridescence`** — `iridescenceTexture` (binding 24) and `iridescenceThicknessTexture` (binding 25). `ThinFilmIridescence()` helper in Metal shader computes optical path difference for RGB wavelengths. Modulates `F0` in direct and IBL specular paths. Uniforms expanded to 78 floats.

### KHR_materials_anisotropy

- [x] **`KHR_materials_anisotropy`** — parsed as `Material::khrMaterialsAnisotropy` (`anisotropyStrength`, `anisotropyRotation`, `anisotropyTexture`). Directional specular highlights (brushed metal). Tangent-space anisotropic GGX BRDF implemented in Metal shader with `aspect = sqrt(1.0 - 0.9 * strength)`. Anisotropy texture (binding 26) samples R=strength and G=rotation (0–1 maps to 0–2π). View mode and macOS example hotkey (`p`) added.

### KHR_materials_dispersion

- [ ] **`KHR_materials_dispersion`** — parsed as `Material::khrMaterialsDispersion` (scalar, default 0.0). Chromatic aberration / prism effect for transmissive materials. Depends on `KHR_materials_transmission`.

### EXT_mesh_gpu_instancing

- [x] **`EXT_mesh_gpu_instancing`** — parsed into `Node::extMeshGpuInstancing` (`attributes` map: `TRANSLATION`, `ROTATION`, `SCALE` → accessor IDs). Enables efficient GPU instancing for large repeated meshes. 
  - Instance data loaded in `setScene()`: reads accessors for translation, rotation, scale
  - Per-instance transform matrices uploaded to GPU buffer (`nodeInstanceData`)
  - Vertex buffer slot 19 added for per-instance matrices (stepMode=instance)
  - Metal shader updated to apply instance transform before node transform
  - `renderPrimitive()` uses `drawIndexed(..., instanceCount)` / `draw(..., instanceCount)`

### KHR_materials_variants

- [x] **`KHR_materials_variants`** — `setMaterialVariant(int32_t)` API implemented. `getMaterialVariantCount()` / `getMaterialVariantName()` accessors added. `renderPrimitive()` overrides `matIdx` from `khrMaterialsVariantsMappings` when a variant is active. Pass -1 to restore asset defaults. `activeVariant` reset on `setAsset()`.

### KHR_texture_basisu (KTX2)

- [ ] **`KHR_texture_basisu`** — parsed as `Texture::khr_texture_basisu` (image index). KTX2 / Basis Universal compressed GPU textures. Requires `campello_gpu` to support compressed texture formats (BC7, ETC2, ASTC) and a KTX2 decoder (e.g., `basisu` transcoder).

### Compression (call before rendering)

- [x] **`KHR_draco_mesh_compression`** — `GLTF::loadGLB()` already calls `decompressDraco()` internally, so the renderer should NOT call it again in `setScene()`. Fixed double-decompression bug.
- [x] **`KHR_draco_mesh_compression`** for .gltf files — `import_gltf()` now uses the callback-based `GLTF::loadGLTF(text, callback)` overload for `.gltf` files. The callback loads external `.bin` buffers (and images) relative to the file's base directory. `decompressDraco()` and `decompressMeshopt()` are called automatically by the gltf library after all external buffers arrive.
- [x] **`EXT_meshopt_compression` / `KHR_meshopt_compression`** — `GLTF::decompressMeshopt()` called in `setScene()`, decoded buffer view data used automatically since it's stored in the standard buffer view structure.

### KHR_animation_pointer

- [x] **`KHR_animation_pointer`** — `GltfAnimator::sampleAnimation()` handles `target.path == "pointer"` channels. Node pointers (`/nodes/{N}/translation|rotation|scale`) route to the existing `AnimatedTRS` system. Material pointers (`/materials/{N}/...`) and light pointers (`/extensions/KHR_lights_punctual/lights/{N}/...`) are stored in `animatedPointers` and resolved by `applyAnimatedPointers()`, which mutates the glTF asset and returns modified material indices. Legacy `Renderer::update()` re-uploads affected material slots automatically. ECS path has `material_animation_sync_system()` for the same purpose.

### KHR_mesh_quantization

- [ ] **`KHR_mesh_quantization`** — pipeline variants and `normalized` flag support are implemented, but the test asset (`BoxQuantized.glb`) revealed geometry/normal issues. Needs revisiting to verify end-to-end correctness with proper Khronos sample assets. Deferred until higher-priority extensions are complete.

---

## Materials (PBR metallic-roughness)

- [x] Base color factor — `materialUniformBuffer` uploads `baseColorFactor` per material
- [x] Base color texture — textured pipeline variant binds `baseColorTexture` via bind groups
- [x] Metallic-roughness texture — G=roughness, B=metallic channels sampled and multiplied by factors
- [x] Normal map — tangent-space normal mapping with TBN matrix, `normalScale` support
- [x] Occlusion texture — R channel sampled, multiplied with ambient and diffuse lighting
- [x] Emissive factor + texture — RGB texture sampled, multiplied by `emissiveFactor`, added after lighting
- [x] Alpha modes: MASK (cutout) — implemented
- [x] Alpha modes: BLEND (macOS/Metal) — pipelines ready; Android/Windows alias to opaque; back-to-front sort not yet implemented
- [x] Double-sided materials (toggle `CullMode::none`)

---

## Dependencies

- [x] **`webp.cmake` removed** — `campello_image` already fetches and links its own `libwebp` internally (v1.6.0); the unused root-level `dependencies/webp.cmake` has been deleted.

---

## CI / Testing

- [ ] **Enable tests on Linux** — commented out; blocked by Vulkan vertex input gap upstream. Re-enable once `campello_gpu` Vulkan backend is fixed.
- [ ] **Enable tests on Windows** — blocked by DXIL shaders and DLL symbol export (.lib) gaps.
- [ ] **GPU-backed integration tests** — all current tests use `nullptr` device. Add macOS-only test suite using a real Metal device to verify render output (render to texture, compare pixel values).
- [ ] **Test with KhronosGroup sample assets** — DamagedHelmet, Sponza, BoxAnimated.

---

## Examples

### macOS (`examples/macos/`)
- [x] **Lighting controls menu** — `Lighting` menu with toggles for punctual lights (Cmd+Shift+L) and default light (Cmd+Option+L), plus `Background` submenu with Dark/Gray/Light presets (1/2/3). Scene presentation API: `setClearColor()`, `setPunctualLightsEnabled()`, `setDefaultLightEnabled()`.
- [x] **Drag-and-drop `.gltf` / `.glb` file loading**
- [ ] Node hierarchy / scene picker UI
- [x] **Load `.gltf` text format** — macOS example uses `GLTF::loadGLTF()` with a URI resolver callback for external buffers/images. Both `.glb` and `.gltf` are supported.

### Android (`examples/android/`)
- [ ] Verify render with campello_gpu v0.7.0 + gltf v0.3.6 after dependency bumps

---

## Documentation

- [ ] Update README with build instructions for all platforms
- [ ] Document the `createDefaultPipelines()` + `resize()` + `render()` required call sequence
- [ ] Document `VERTEX_SLOT_*` shader contract for consumers writing custom shaders
