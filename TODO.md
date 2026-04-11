# campello_renderer TODO

## Current State (v0.1.2)

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

- [ ] **Fix `actions/checkout@v6` in CI** — `.github/workflows/build.yml` uses `@v6` (does not exist). Must be `@v4`. All 5 jobs affected. Fix before committing the pending dependency bumps.
- [ ] **Commit pending changes** — `campello_gpu` v0.6.1→v0.7.0, `gltf` v0.3.5→v0.3.6, `LICENSE` — after fixing the checkout version.

---

## Bugs / Upstream Blockers

- [ ] **Vulkan vertex input not applied** — `campello_gpu` Vulkan backend hardcodes `vertexBindingDescriptionCount = 0`, so vertex buffer layouts passed via `createDefaultPipelines()` are ignored. Fix upstream in `campello_gpu`. Tracked in the Vulkan `createDefaultPipelines` path.
- [ ] **DirectX rendering nonfunctional** — DXIL binaries not compiled; `kDefaultDirectXVertShaderSize == 0` causes early return. `pipelineFlat` and `pipelineTextured` remain null on Windows.

---

## Asset Loading

- [ ] **`data:uri` image support** — `setScene()` has a `TODO: handle data:uri images` comment and skips images where `image.data` is non-empty (base64 data URIs). Implement base64 decode + `campello_image` decode path.
- [ ] **External image URIs** — `.gltf` files referencing external image files (not buffer views) are not loaded.
- [ ] **Non-standard buffer view strides** — interleaved vertex data (`byteStride > 0` in buffer views) is not handled; `byteOffset` is applied but stride is ignored.

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
- [ ] **iOS SDK compile** — current `.metallib` is compiled for macOS. Add `iphoneos` and `iphonesimulator` SDK variants.
- [ ] **Add Metal compile step to CMake** — automate `metal` → `metallib` compilation for Apple targets.

### Longer term
- [ ] **Single shader source** — evaluate HLSL → MSL + SPIR-V via DXC + SPIRV-Cross to eliminate per-platform divergence.

---

## Animation

- [ ] **Implement `update(double dt)`** — currently a no-op `(void)dt`. Evaluate GLTF animation channels (TRS keyframe interpolation: LINEAR, STEP, CUBICSPLINE) and apply to node transforms before `render()`.
- [ ] **Skeletal meshes** — joint matrix palette (skin / inverse bind matrices).

---

## GLTF Extension Rendering Support

The gltf library (v0.3.6) parses **23 extensions** into structured data. The renderer currently uses a small subset. Below is what remains to implement, grouped by complexity.

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
- [ ] **Alpha modes: BLEND** — requires sorted transparent draw pass with alpha blending enabled in the pipeline color state.
- [x] **Double-sided materials** — `Material::doubleSided`. Pipeline variants with `CullMode::none` for double-sided materials.

### KHR_materials_unlit

- [x] **`KHR_materials_unlit`** — parsed as `Material::khrMaterialsUnlit` (bool). When true, render using only `baseColorFactor × baseColorTexture` with no lighting calculation. Implemented as shader branch using materialFlags.z.

### KHR_texture_transform

- [x] **`KHR_texture_transform`** — UV transform (T × R × S) baked into `materialUniformBuffer` as two float4 rows per material slot (48-byte stride). Applied in the vertex shader for Metal and Vulkan. DirectX HLSL also updated. Note: `texCoord` override field not yet respected (always applies to TEXCOORD_0).

### KHR_lights_punctual

- [ ] **`KHR_lights_punctual`** — parsed into `GLTF::khrLightsPunctual` (vector of `KHRLightPunctual`). Each light has `type` (directional/point/spot), `color`, `intensity`, `range`, and spot cone angles. Node references via `Node::light` index. Currently no lighting model beyond a hardcoded view direction. Implementation needs:
  - Pass active lights to the shader (uniform buffer or push constants)
  - Implement point/directional/spot attenuation
  - Respect light world transform from the node's world matrix

### KHR_materials_ior

- [ ] **`KHR_materials_ior`** — parsed as `Material::khrMaterialsIor` (scalar, default 1.5). Affects Fresnel reflectance. Replace hardcoded F0 in specular calculations with `((ior - 1) / (ior + 1))^2`.

### KHR_materials_specular

- [ ] **`KHR_materials_specular`** — parsed as `Material::khrMaterialsSpecular` (`specularFactor`, `specularTexture`, `specularColorFactor` vec3, `specularColorTexture`). Modulates the specular reflection layer on top of PBR metallic-roughness.

### KHR_materials_clearcoat

- [ ] **`KHR_materials_clearcoat`** — parsed as `Material::khrMaterialsClearcoat` (`clearcoatFactor`, `clearcoatTexture`, `clearcoatRoughnessFactor`, `clearcoatRoughnessTexture`, `clearcoatNormalTexture`). Adds a thin, clear coating layer over the base material. Requires an additional BRDF lobe in shaders.

### KHR_materials_sheen

- [ ] **`KHR_materials_sheen`** — parsed as `Material::khrMaterialsSheen` (`sheenColorFactor` vec3, `sheenColorTexture`, `sheenRoughnessFactor`, `sheenRoughnessTexture`). Models cloth/fabric micro-fibre retroreflection.

### KHR_materials_transmission

- [ ] **`KHR_materials_transmission`** — parsed as `Material::khrMaterialsTransmission` (`transmissionFactor`, `transmissionTexture`). Physically-based glass / thin-sheet transparency. Requires a render-to-texture pass to capture the background for refraction sampling.

### KHR_materials_volume

- [ ] **`KHR_materials_volume`** — parsed as `Material::khrMaterialsVolume` (`thicknessFactor`, `thicknessTexture`, `attenuationDistance`, `attenuationColor`). Beer-Law absorption for transmissive volumes (e.g., colored glass). Depends on `KHR_materials_transmission`.

### KHR_materials_iridescence

- [ ] **`KHR_materials_iridescence`** — parsed as `Material::khrMaterialsIridescence` (`iridescenceFactor`, `iridescenceTexture`, `iridescenceIor`, `iridescenceThicknessMinimum/Maximum` nm, `iridescenceThicknessTexture`). Thin-film interference (soap bubble / oil slick effect). Complex shader math; low priority.

### KHR_materials_anisotropy

- [ ] **`KHR_materials_anisotropy`** — parsed as `Material::khrMaterialsAnisotropy` (`anisotropyStrength`, `anisotropyRotation`, `anisotropyTexture`). Directional specular highlights (brushed metal). Requires tangent-space anisotropic BRDF.

### KHR_materials_dispersion

- [ ] **`KHR_materials_dispersion`** — parsed as `Material::khrMaterialsDispersion` (scalar, default 0.0). Chromatic aberration / prism effect for transmissive materials. Depends on `KHR_materials_transmission`.

### EXT_mesh_gpu_instancing

- [ ] **`EXT_mesh_gpu_instancing`** — parsed into `Node::extMeshGpuInstancing` (`attributes` map: `TRANSLATION`, `ROTATION`, `SCALE` → accessor IDs). Enables efficient GPU instancing for large repeated meshes. Requires reading the instance accessor data and submitting multi-instance draw calls with per-instance transform buffers.

### KHR_materials_variants

- [ ] **`KHR_materials_variants`** — parsed into `GLTF::khrMaterialsVariants` (name list) and `Primitive::khrMaterialsVariantsMappings` (material index per variant). Add a `setMaterialVariant(uint32_t variantIndex)` API that swaps material indices before rendering.

### KHR_texture_basisu (KTX2)

- [ ] **`KHR_texture_basisu`** — parsed as `Texture::khr_texture_basisu` (image index). KTX2 / Basis Universal compressed GPU textures. Requires `campello_gpu` to support compressed texture formats (BC7, ETC2, ASTC) and a KTX2 decoder (e.g., `basisu` transcoder).

### Compression (call before rendering)

- [x] **`KHR_draco_mesh_compression`** — `GLTF::loadGLB()` already calls `decompressDraco()` internally, so the renderer should NOT call it again in `setScene()`. Fixed double-decompression bug.
- [ ] **`KHR_draco_mesh_compression`** for .gltf files — For .gltf files with external .bin buffers, the renderer needs to load the external buffer and then call `decompressDraco()`.
- [x] **`EXT_meshopt_compression` / `KHR_meshopt_compression`** — `GLTF::decompressMeshopt()` called in `setScene()`, decoded buffer view data used automatically since it's stored in the standard buffer view structure.

### KHR_animation_pointer

- [ ] **`KHR_animation_pointer`** — parsed into `AnimationChannelTarget::khrAnimationPointer` (RFC 6901 JSON pointer string). Extends animation to drive arbitrary material and extension properties (e.g., `emissiveFactor`, `transmission`). Depends on `update()` animation being implemented first.

### KHR_mesh_quantization

- [ ] **`KHR_mesh_quantization`** — recognized by the gltf library but no struct is generated. Quantized vertex attributes use normalized integer component types (BYTE, SHORT) instead of FLOAT. The renderer's `createDefaultPipelines()` vertex layouts currently hard-code `ctFloat`. Add per-accessor component type detection and map to the correct `GPU::ComponentType` when binding vertex buffers.

---

## Materials (PBR metallic-roughness)

- [x] Base color factor — `materialUniformBuffer` uploads `baseColorFactor` per material
- [x] Base color texture — textured pipeline variant binds `baseColorTexture` via bind groups
- [x] Metallic-roughness texture — G=roughness, B=metallic channels sampled and multiplied by factors
- [x] Normal map — tangent-space normal mapping with TBN matrix, `normalScale` support
- [x] Occlusion texture — R channel sampled, multiplied with ambient and diffuse lighting
- [x] Emissive factor + texture — RGB texture sampled, multiplied by `emissiveFactor`, added after lighting
- [x] Alpha modes: MASK (cutout) — implemented
- [ ] Alpha modes: BLEND (transparency) — requires sorted transparent draw pass
- [x] Double-sided materials (toggle `CullMode::none`)

---

## Dependencies

- [ ] **Wire up `webp.cmake`** — `dependencies/webp.cmake` (libwebp v1.5.0) is defined but not included in `CMakeLists.txt`. `campello_image` + GLTF `ext_texture_webp` already support WebP image indices; linking libwebp would enable actual decode.

---

## CI / Testing

- [ ] **Enable tests on Linux** — commented out; blocked by Vulkan vertex input gap upstream. Re-enable once `campello_gpu` Vulkan backend is fixed.
- [ ] **Enable tests on Windows** — blocked by DXIL shaders and DLL symbol export (.lib) gaps.
- [ ] **GPU-backed integration tests** — all current tests use `nullptr` device. Add macOS-only test suite using a real Metal device to verify render output (render to texture, compare pixel values).
- [ ] **Test with KhronosGroup sample assets** — DamagedHelmet, Sponza, BoxAnimated.

---

## Examples

### macOS (`examples/macos/`)
- [ ] Drag-and-drop `.gltf` / `.glb` file loading
- [ ] Node hierarchy / scene picker UI
- [ ] Load `.gltf` text format (not just `.glb`)

### Android (`examples/android/`)
- [ ] Verify render with campello_gpu v0.7.0 + gltf v0.3.6 after dependency bumps

---

## Documentation

- [ ] Update README with build instructions for all platforms
- [ ] Document the `createDefaultPipelines()` + `resize()` + `render()` required call sequence
- [ ] Document `VERTEX_SLOT_*` shader contract for consumers writing custom shaders
