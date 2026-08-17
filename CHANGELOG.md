# Changelog

## [0.10.0] - 2026-08-17

### Added
- **Full glTF PBR pipeline ported to DirectX 12 (Windows)** — metallic-roughness Cook-Torrance GGX BRDF, image-based lighting (GGX-prefiltered specular + Lambertian diffuse irradiance + BRDF LUT), normal mapping, punctual lights (`KHR_lights_punctual`), and `KHR_materials_specular`/`anisotropy`/`iridescence`/`clearcoat`/`sheen`/`transmission`, matching the existing Metal/Vulkan feature set. Windows rendering (previously non-functional — DXIL binaries weren't compiled) now fully works, including resize.
- **DirectX combined per-(material, frame-in-flight) bind group scheme** (`Renderer::ensureDirectXPbrBindGroupLayout()` / `buildDirectXCombinedBindGroup()` / `rebuildDirectXCombinedBindGroups()`) — works around a `campello_gpu` D3D12 root-signature bug where a second `setBindGroup()` call landed on the wrong descriptor table, by combining all per-material and per-frame PBR resources (textures, samplers, lights, camera, environment/IBL, scene-color) into one bind group per (material, frame-in-flight) pair instead of Vulkan/Metal's material+frame split.
- New Windows example app (`examples/windows/`, GLFW-based) — drag-and-drop glTF/HDR loading, orbit camera, animation playback, and the same view-mode/hotkey set as the macOS/Linux examples. Enabled via `-DCAMPELLO_RENDERER_BUILD_WINDOWS_EXAMPLE=ON`.
- `build_directx_shaders.ps1` — compiles `shaders/directx/default.hlsl` to DXIL via `dxc.exe` and regenerates the embedded `src/shaders/directx_default.h` header, mirroring `build_metal_shaders.sh` / `build_vulkan_shaders.sh`.
- `WINDOWS_EXPORT_ALL_SYMBOLS` enabled for the Windows DLL build — without it, MSVC's linker built the `.dll` but never emitted the companion import `.lib`, failing every consumer (tests, example apps) at link time with `LNK1104`.

### Changed
- Upgraded `campello_gpu` dependency from v0.23.0 to v0.23.1 (Windows/DirectX descriptor-heap contiguity, DSV/sampler-heap leaks, swapchain resize, and cube-texture fixes).
- `dependencies/campello_gpu.cmake` no longer supports a local-checkout override — always fetches from GitHub via `FetchContent`, removing a footgun where a developer's local `campello_gpu` checkout could silently diverge from the pinned released version.

### Fixed
- **Mirrored default camera** — `render(view)`'s and the ECS `render_system()`'s built-in fallback camera (used whenever no glTF/ECS camera exists) rendered a left-right-mirrored image and incorrectly backface-culled single-sided, CCW-front geometry near the origin. Root cause: `vector_math::Matrix4::lookAt()` builds its right/up axes with a cross-product argument order that's mirrored relative to what `Matrix4::perspective()`'s +Z-forward convention expects — confirmed empirically (geometry translated to world +X rendered on the left half of the frame instead of the right). Same root cause as the long-disabled `DISABLED_MeshRendersNonClearPixels` and sibling Vulkan tests. Worked around locally with a corrected `buildDefaultCameraView()` helper in both `campello_renderer.cpp` and `ecs.hpp`, scoped narrowly to the default-camera fallback only — real cameras (glTF-embedded, ECS `Transform`-driven, both of which use `Matrix4::inverted()` instead) are unaffected. The underlying bug lives upstream in the shared `vector_math` dependency.
- **DirectX offscreen-render test crashes** — `BasicOffscreenRenderDoesNotCrash` / `MultipleConsecutiveRenders` / `ResizeOffscreenTarget` / `DifferentPixelFormatBGRA8` released their local render-target texture before the GPU finished executing the render into it (`render()`'s `submit()` is async and doesn't block), which D3D12's debug layer treats as fatal resource corruption. Fixed by adding `device->waitForIdle()` before the texture goes out of scope, matching the pattern `readBackPixels()` already used.
- **`ECSPathRendersToOffscreen` backface culling** — same root cause as the mirrored-camera fix above; the test now builds its own corrected view matrix since it exercises the public ECS `render(scene, view)` API as an external caller would, supplying its own camera.

## [0.9.0] - 2026-08-13

### Added
- **Vulkan IBL precompute pipeline** (`Renderer::bakeIblResources()`) — bakes the BRDF LUT, a GGX-prefiltered specular cubemap, and a Lambertian diffuse irradiance cubemap from the current environment map via a new fullscreen-triangle shader (`shaders/vulkan/ibl_bake.frag`), matching the reference glTF-Sample-Renderer's IBL approach that was already implemented for Metal. The PBR fragment shader now samples these baked resources (frame bind group bindings 6/7/8) instead of the raw environment map, falling back to it when a bake hasn't run yet.
- **Vulkan skybox rendering** — new `shaders/vulkan/skybox.frag` plus `pipelineSkybox` wiring; `drawSkybox()` was already shared by both render paths and only needed the Vulkan pipeline to light up.
- `Renderer::getBoundsCenter()` — world-space center of the current scene's geometry bounding box (via new `computeSceneAABB()`), so camera code can orbit the scene's actual visual center instead of assuming the origin. Wired into the Linux and macOS example apps' `Camera::fitBounds()`.
- `Renderer::~Renderer()` — explicit destructor that waits for all in-flight GPU frames to finish before member buffers/textures/bind groups are released.

### Changed
- `skyboxEnabled` now defaults to `true` (previously `false`).
- Default embedded environment map replaced with "Cannon Exterior" by Greg Zaal — the Khronos glTF-Sample-Viewer's own default HDRI — instead of "Kiara 5 Noon", so out-of-the-box IBL brightness is directly comparable to the reference viewer.
- `sceneColorTexture`/`sceneColorView` (FXAA/SSAA) and `opaqueSceneTexture`/`opaqueSceneView` (screen-space refraction) are now per-frame-in-flight arrays instead of single shared instances, fixing a race where one frame's write could stomp another still-in-flight frame's read of the same texture — visible as corrupted/torn output on animated scenes using FXAA/SSAA or `KHR_materials_transmission`.
- Upgraded `campello_gpu` dependency from v0.21.1 to v0.23.0.

### Fixed
- `OffscreenRenderTest.ClearColorIsApplied` now explicitly disables the skybox, since `skyboxEnabled` defaulting to `true` would otherwise paint over the clear color under test.

## [0.8.0] - 2026-08-02

### Added
- **Sparse accessor support** — `resolveAccessorBytes()` resolves any vertex attribute accessor's base bufferView data (or an implicit zero base if absent) with `accessor.sparse` indices/values overlaid on top. Applied to the deinterleaving pass, COLOR_0 conversion, and morph target deltas. Scoped to vertex attributes only — sparse `primitive.indices` is not handled (rare in practice).
- **Morph target support** — POSITION and NORMAL blending (TANGENT not implemented) for up to 8 targets per primitive:
  - `GltfAnimator` gained an `animatedWeights` map + `getAnimatedWeights()`, and `sampleAnimation()` now handles the `"weights"` animation channel path (linear and step interpolation)
  - `Renderer::updateMorphWeights()` resolves weight precedence (`node.weights` > `mesh.weights` > animated override) into a per-node `MorphInfo` uniform each frame
  - New vertex buffer slots 21–23 (`MorphInfo`, position deltas, normal deltas), bound in both the primary glTF-driven `renderPrimitive()` path and the ECS `drawDrawCall()` path (the latter always binds the zero-weight fallback, since standalone `uploadMesh()` meshes don't carry morph data)
  - Metal shader blends targets before skinning, matching glTF's evaluation order
- **COLOR_0 and multi-UV (`texCoord`) support** — `COLOR_0` (VEC3/VEC4, float or normalized ubyte/ushort) is normalized to a canonical float4 buffer and multiplied into `baseColor` per spec; every texture reference now respects its own `texCoord` index (0 or 1) via a per-material bitmask and `selectUV()` in the shader, instead of always sampling `TEXCOORD_0`. New vertex slots 6 (COLOR_0) and 7 (TEXCOORD_1).
- **KHR_materials_iridescence rewritten** to the actual Khronos reference algorithm (multi-bounce Airy summation with CIE XYZ sensitivity curves), replacing a naive 3-wavelength cosine approximation.
- **KHR_materials_emissive_strength** — `emissiveFactor` is now multiplied by the extension's strength before upload (was previously ignored).
- **On-demand rendering** in the macOS example — MTKView's continuous CVDisplayLink drives a per-tick dirty-flag gate (`requestRedraw`/`_needsRedraw`) instead of manual `setNeedsDisplay:` invalidation, avoiding cross-display vsync races while still skipping GPU work for static scenes.
- Embedded default environment map replaced with a genuine 1024×512 Radiance HDR (was an 8-bit tonemapped JPEG that hard-capped highlights at 1.0), with exposure correction (`kBuiltinEnvironmentExposure`) calibrated against the source HDRI's real radiance values; built-in cubemap bake resolution raised from 128px to 512px per face.

### Changed
- Upgraded `gltf` dependency from v0.4.2 to v0.5.0.
- Upgraded `campello_gpu` dependency from v0.21.0 to v0.21.1.
- Fixed stale `campello_gpu.cmake` / `campello_image.cmake` local-path overrides pointing at nonexistent `/Users/rubenleal/Projects/...` directories (silently always falling back to GitHub fetch for local dev builds); now point at the actual local checkouts.

### Fixed
- **Animation delta-time bug (macOS example)** — `drawInMTKView:` was advancing the animation clock by a hardcoded 1/60s every call regardless of actual elapsed time, so any frame slower than 16.7ms (heavy scenes, the per-frame `waitForIdle()`, GPU contention) silently ran animations in slow motion. Now measures real elapsed time via `CACurrentMediaTime()`, captured every display-link tick (so idle/static periods don't accumulate into a catch-up jump when animation resumes) and clamped to 0.25s for genuine stalls.
- **MaterialUniforms iridescence offset bug** — a phantom padding float inserted before `iridescenceFactor` (based on a false assumption that Metal needed alignment padding between two plain floats) shifted every field from `iridescenceFactor` through `dispersion` one float late relative to the compiled Metal struct layout, silently disabling `KHR_materials_iridescence` for every asset using it.
- **Transparent draw-order sort** — used the node's own local translation as a distance proxy, which ties (and falls to undefined `std::sort` order) for coincident "shell" meshes sharing a parent with zero local offset; now uses `nodeWorldBounds[nodeIndex]`'s center.
- **Metallic materials routed to the flat pipeline** — materials with `metallicFactor > 0` but no textures (e.g. `metallicFactor` defaulting to 1.0 when omitted) were rendered with zero reflections; now routed to the textured/IBL pipeline like other reflective materials.
- **UNSIGNED_BYTE index buffers** — the index-format ternary defaulted any non-`UNSIGNED_SHORT` component type to `uint32`, including the spec-legal but uncommon `UNSIGNED_BYTE`, reading 4× too much data per index from a 1-byte-per-index buffer and producing corrupted geometry. Now widened to `uint16` at scene-load time.
- `fragmentMain_flat` was dropping `emissiveFactor` entirely (materials with near-black `baseColor` + `KHR_materials_emissive_strength` and no textures rendered invisible).
- Skybox fragment shader was missing tonemapping, clipping/saturating raw HDR background values.
- `__weak` local variable in `ViewController.mm` (compiles under MRC, not ARC) caused a hard compile error.

## [0.7.0] - 2026-04-28

### Added
- **KHR_texture_procedurals support** — Load-time procedural texture baking:
  - CPU baker `bakeProceduralTexture()` evaluates node graphs (constant, mix, noise2D, checkerboard, sin, place2D, image, swizzle, combine, extract, ifGreater, length, distance, crossProduct, normalize)
  - GPU baker `bakeProceduralTextureGPU()` dispatches a compute shader per pixel; falls back to CPU for unsupported nodes (e.g. `image`) or pipeline creation failures
  - `setProceduralBakeSize(int)` / `getProceduralBakeSize()` controls resolution (default 1024)
  - `setScene()` automatically bakes procedural textures referenced by materials and caches them keyed by `"graph:<index>:output:<name>"`
  - New public header: `inc/campello_renderer/procedural_texture_baker.hpp`
  - 23 procedural baker tests: 16 CPU (`ProceduralBakerTest`) + 7 GPU (`ProceduralBakerGPUTest`)
- **Offscreen rendering test suite** — GPU integration tests without window/swapchain:
  - `OffscreenRenderTest` fixture creates real GPU textures with `renderTarget | copySrc`, renders, and reads back pixels for verification
  - 7 tests covering: basic clear color, multiple consecutive renders (frame ring buffer), resize, mesh rendering, ECS path rendering, BGRA8 pixel format
- **ECS path matrix transpose fix** — `uploadOneTransform` now transposes row-major `vector_math::Matrix4` to column-major `float4x4` for Metal shaders, matching the glTF path behavior
- **`GpuMesh::indexFormat`** — New field defaults to `uint32`; `uploadMesh()` sets it from glTF accessor component type (`UNSIGNED_SHORT` → `uint16`). ECS `renderPrimitive()` uses the mesh's format instead of hardcoding `uint32`

### Changed
- Upgraded `campello_gpu` dependency from v0.13.0 to v0.13.1
- Upgraded `gltf` dependency from v0.4.1 to v0.4.2 (adds `KHR_texture_procedurals` extension support)
- Regenerated embedded shader headers (Metal, Vulkan, DirectX) from latest shader sources

### Fixed
- ECS offscreen rendering now correctly renders geometry (was rendering only clear color due to garbage MVP matrices)
- glTF meshes with `UNSIGNED_SHORT` indices now render correctly via the ECS path

## [0.6.0] - 2026-04-27

### Added
- **KHR_texture_basisu support** — GPU-compressed texture loading via `campello_image::TextureData`:
  - `setScene()` detects images referenced by `KHR_texture_basisu` and transcodes them to the optimal GPU format for the device
  - Format selection hierarchy: ASTC 4×4 (Apple/modern mobile) → BC7 (desktop) → ETC2 (older mobile) → RGBA8 (fallback)
  - Full mip chain upload via `copyBufferToTexture` for compressed and uncompressed formats
  - `imageIndexForTex` now resolves `khr_texture_basisu` before `ext_texture_webp` and `source`
  - New test: `SetAssetTest.SetAssetWithBasisuTextureDoesNotCrash`

### Changed
- Upgraded `campello_image` dependency from v0.4.0 to v0.5.0 (local override available at `/Users/rubenleal/Projects/campello_image`)
- `dependencies/campello_image.cmake` now supports local path override (matching `campello_gpu.cmake` pattern)

## [0.5.0] - 2026-04-25

### Added
- **KHR_materials_iridescence** — Thin-film interference for soap-bubble / oil-slick effects:
  - `iridescenceTexture` (binding 24) and `iridescenceThicknessTexture` (binding 25)
  - `ThinFilmIridescence()` helper in Metal shader computes optical path difference for RGB wavelengths
  - Modulates `F0` in direct and IBL specular paths
- **KHR_materials_anisotropy** — Directional specular highlights (brushed metal):
  - Tangent-space anisotropic GGX BRDF with `aspect = sqrt(1.0 - 0.9 * strength)`
  - `anisotropicTexture` (binding 26): R channel = strength, G channel = rotation (0–1 → 0–2π)
  - View mode hotkey `p` in macOS example
- **ECS refactor — engine-native render API** (Phase 1–3):
  - New public structs: `GpuMesh`, `GpuMaterial`, `DrawCall`, `CameraData`, `LightData`, `RenderScene`
  - `uploadMesh()` / `uploadMaterial()` — decouple GPU resource upload from glTF scene ownership
  - `render(const RenderScene& scene, colorView)` — flat draw-list submission instead of recursive node traversal
  - New optional header-only bridge: `inc/campello_renderer/ecs.hpp`
- **Animation extraction** — `GltfAnimator` moved to standalone `inc/campello_renderer/animation.hpp` + `src/animation.cpp`:
  - `KHR_animation_pointer` support: material and light properties can be animated via glTF pointer targets
  - `reuploadMaterialSlot()` syncs animated material properties back to GPU uniform buffer
- **Performance counters** — `RenderStats` exposes `opaqueDrawCount`, `transparentDrawCount`, `totalDrawCount`, `culledNodeCount`, `visibleNodeCount`, `cpuFrameTimeMs`
- **View mode expansions** — `iridescence` (`o`), `anisotropy` (`p`), `dispersion` (`z`) debug visualization modes

### Changed
- Upgraded `campello_gpu` dependency from v0.11.1 to v0.12.0
- `kMaterialUniformStride` bumped from 256 to 512 bytes to accommodate expanded `MaterialUniforms` struct (288 bytes)
- Metal shader: `MaterialUniforms` expanded to 82 floats (anisotropy, iridescence, dispersion fields)

## [0.4.0] - 2026-04-23

### Added
- **Equirectangular-to-cubemap environment map loading** — CPU-based conversion from single 2:1 equirectangular images to GPU cubemaps:
  - `loadEquirectangularEnvironmentMap(path, faceSize = 0)` — loads `.hdr`, `.exr`, `.png`, `.jpg`, `.webp`
  - Bilinear filtering during conversion for smooth face generation
  - Auto face-size: half the equirectangular height (capped at 2048)
  - Preserves source pixel format (`rgba8` → `rgba8unorm`, `rgba16f` → `rgba16float`, `rgba32f` → `rgba32float`)
  - Standard graphics convention: v=0 at zenith (+Y), v=1 at nadir (-Y)
- **macOS example environment map UI**:
  - `Lighting → Load Environment Map…` (Cmd+Shift+E) — file picker for equirectangular HDR/EXR/LDR images
  - `Lighting → Background Mode` submenu — one-click switch between:
    - Solid Color (dark clear + skybox/IBL off)
    - Skybox only (cubemap background, no IBL)
    - Skybox + IBL (cubemap background + image-based lighting)
  - Auto-enables skybox and IBL when a new environment map is loaded
- **KHR_materials_transmission + KHR_materials_volume** — Proper environment-based transmission:
  - Replaced simple alpha-reduction with physical refraction via `refract()` and environment cubemap sampling
  - Fresnel-mixed transmission: `(1 - F) * transmission * (1 - metallic)`
  - Diffuse/ambient/IBL-diffuse terms are scaled by `(1 - transmittance)`; specular/clearcoat remain
  - KHR_materials_volume attenuation via Beer-Lambert law: `exp(-thickness / attenuationDistance * (1 - attenuationColor))`
  - **Fixed long-standing `MaterialUniforms` layout bug** — CPU-side offsets for `transmissionFactor`, `hasTransmissionTexture`, `viewMode`, `environmentIntensity`, and `iblEnabled` were 36 bytes late due to an index gap, causing the shader to read zeros for these fields
  - Bumped `kMaterialUniformStride` from 256 to 512 bytes to accommodate the expanded struct (288 bytes) while maintaining Metal's 256-byte vertex buffer offset alignment

## [0.3.0] - 2026-04-23

### Added
- **FXAA post-process anti-aliasing** — Fullscreen post-process pass based on FXAA 3.11:
  - `setFxaaEnabled(bool)` / `isFxaaEnabled()` API
  - Intermediate `sceneColorTexture` render target when FXAA is active
  - `fxaaVertex` / `fxaaFragment` shaders in `default.metal`
  - Clamp-to-edge sampler to avoid edge artifacts
  - macOS example menu item: `Lighting → FXAA` (Cmd+Shift+A)
- **SSAA (Super-Sample Anti-Aliasing)** — Render scene at scaled resolution and bilinear downsample:
  - `setSsaaScale(float)` / `getSsaaScale()` API (1.0 = off, 1.5, 2.0)
  - `sceneColorTexture` created at `width * scale` × `height * scale`
  - `downsampleFragment` shader for all platforms (Metal, Vulkan, DirectX HLSL)
  - FXAA is automatically disabled when SSAA is active (SSAA supersedes FXAA)
  - macOS example submenu: `Lighting → SSAA` (Off / 1.5× / 2.0×)

## [0.2.1] - 2026-04-23

### Fixed
- **Linux CI build failure** — `campello_gpu` v0.11.0's `CMakeLists.txt` called `target_include_directories(... PUBLIC ...)` on an `INTERFACE` library when Vulkan SDK was missing. Fixed upstream in `campello_gpu` v0.11.1; dependency bumped accordingly.

## [0.2.0] - 2026-04-22

### Added
- **Skybox rendering** — Fullscreen-triangle skybox that samples an environment cubemap:
  - `pipelineSkybox` with depth-write disabled, rendered before opaque geometry
  - Inverse VP matrix unprojects screen pixels to world-space ray directions
  - `setSkyboxEnabled(bool)` / `isSkyboxEnabled()` API
- **Image-Based Lighting (IBL)** — Environment cubemap sampled in PBR fragment shader:
  - Diffuse: sample cubemap along normal direction, modulated by `baseColor × (1 − metallic)`
  - Specular: sample cubemap along reflection vector with Fresnel approximation
  - `setIBLEnabled(bool)` / `isIBLEnabled()` API
  - `setEnvironmentIntensity(float)` / `getEnvironmentIntensity()` API
  - `setEnvironmentMap(cubemap)` to bind a custom environment cubemap
  - `loadEnvironmentMap(px, nx, py, ny, pz, nz)` — load 6 face images into a `ttCube` texture
  - New `ViewMode::environment` (`i` key) visualizes IBL contribution
- **campello_gpu v0.11.0** — Adds `ttCube` / `ttCubeArray` texture support (required for skybox + IBL)

### Changed
- Upgraded `campello_image` dependency from v0.3.1 to v0.4.0
  - **BREAKING**: `Image::getData()` now returns `const void*` (was `const uint8_t*`)
  - All image upload calls updated to cast via `const_cast<void*>(img->getData())`
  - HDR formats (Radiance `.hdr`, OpenEXR `.exr`) now decode to `rgba32float`
  - Texture creation in `setScene()` now selects `PixelFormat` based on `Image::getFormat()`:
    - `rgba8` → `rgba8unorm` / `rgba8unorm_srgb`
    - `rgba16f` → `rgba16float`
    - `rgba32f` → `rgba32float`
- `MaterialUniforms` expanded with `environmentIntensity` and `iblEnabled` fields
- Metal shader `fragmentMain_textured` updated with IBL sampling
- New skybox vertex/fragment shaders added to `default.metal`

---

## [0.1.3] - 2026-04-12

### Added
- **Animation system** — Full GLTF animation support with multi-animation playback:
  - `update(double dt)` — advances all playing animations
  - `playAnimation(index)` / `pauseAnimation(index)` / `stopAnimation(index)` — per-animation control
  - `stopAllAnimations()` — stop all animations at once
  - `setAnimationTime(index, t)` / `getAnimationTime(index)` — per-animation seeking
  - `setAnimationLoop(index, bool)` / `isAnimationLooping(index)` — per-animation loop control
  - `isAnimationPlaying(index)` — check individual animation state
  - `getAnimationCount()` / `getAnimationName(i)` / `getAnimationDuration(i)` — animation introspection
  - LINEAR interpolation with slerp for rotations, STEP interpolation supported
  - Last-animation-wins when multiple animations target same node/property
- **EXT_mesh_gpu_instancing** — GPU instancing for repeated meshes via `EXT_mesh_gpu_instancing`:
  - Instance transforms (translation, rotation, scale) loaded from accessor data
  - Per-instance matrices uploaded to GPU, bound at vertex slot 19
  - `drawIndexed`/`draw` with instance count for efficient rendering
- **KHR_materials_variants** — Material variant switching:
  - `setMaterialVariant(index)` / `getMaterialVariantCount()` / `getMaterialVariantName(i)` API
  - `renderPrimitive()` applies variant material index when active
- **KHR_materials_ior** — Index of refraction for dielectric materials:
  - `ior` uploaded to material uniform buffer (offset 108)
  - Shader computes F0 from IOR: `((ior-1)/(ior+1))²` instead of hardcoded 0.04
- **KHR_materials_specular** — Specular layer for dielectric materials:
  - `specularFactor` (scalar) and `specularColorFactor` (vec3) uniforms (offsets 112, 128)
  - `specularTexture` (A channel) and `specularColorTexture` (RGB, sRGB) bindings
  - Shader mixes dielectric F0 with metallic F0 based on specular parameters
- **KHR_materials_clearcoat** — Clearcoat layer rendering:
  - GGX NDF + Smith-GGX visibility + Schlick Fresnel (F0=0.04)
  - `clearcoatTexture` (R, binding 17), `clearcoatRoughnessTexture` (G, binding 18), `clearcoatNormalTexture` (binding 19)
  - Base layer attenuated by `(1 - ccFactor × F_Schlick(0.04, NdotV))`
- **KHR_materials_sheen** — Sheen lobe for fabric-like materials:
  - Charlie NDF + Neubelt visibility term
  - `sheenColorTexture` (RGB sRGB, binding 15) and `sheenRoughnessTexture` (R linear, binding 16)
  - Uniforms at material buffer offsets 144-167
- **KHR_materials_transmission** (simplified) — Transmission for transparent materials:
  - `transmissionFactor` (scalar, offset 228) and `transmissionTexture` (R channel, binding 20)
  - Simplified implementation: modulates alpha (`alpha *= 1 - transmission`)
  - Forces blend pipeline when transmission is active
  - No render-to-texture (thin glass approximation)
- **KHR_materials_unlit** — Unlit shading model:
  - `khrMaterialsUnlit` flag in material buffer (offset 68)
  - Returns `baseColor × baseColorTexture` without lighting when enabled
- **Emissive + Occlusion textures**:
  - `emissiveTexture` (RGB, sRGB) with `emissiveFactor` (vec3, offset 96)
  - `KHR_materials_emissive_strength` scalar multiplier
  - `occlusionTexture` (R channel) with `occlusionStrength` — multiplies ambient and diffuse
- **Alpha blend mode** — Full transparency support:
  - Blend pipelines (`srcAlpha * oneMinusSrcAlpha`) for all variants
  - Depth write disabled for transparent materials
  - Back-to-front sort for transparent primitives (squared camera distance)
- **Double-sided materials** — No culling with `CullMode::none` for double-sided materials

### Changed
- Upgraded `campello_gpu` dependency from v0.6.0 to v0.8.0
- Upgraded `gltf` dependency from v0.3.5 to v0.4.0
  - Breaking: `GLTF::loadGLTF()` now requires callback for external resources
- Material uniform buffer expanded to 256-byte stride:
  - New fields: emissiveFactor, ior, specularFactor, specularColorFactor, sheen params, clearcoat params, transmission params
- Metal shader updated with all new PBR extensions

### Fixed
- Metal `float3` alignment — All float3 fields now at 16-byte aligned offsets (96, 128, 144, etc.)
- Transmission struct alignment — `transmissionFactor` now at correct offset 228 (index 57)

---

## [0.1.4] - 2026-04-12

### Added
- **PBR metallic-roughness rendering** — Full metallic-roughness workflow with:
  - `metallicRoughnessTexture` sampling (G=roughness, B=metallic)
  - `metallicFactor` and `roughnessFactor` scalar multipliers
  - Simple Lambert diffuse + Blinn-Phong specular approximation
- **Normal mapping** — `normalTexture` with tangent-space decoding and TBN matrix
  - `normalScale` intensity control from `NormalTextureInfo::scale`
  - Falls back to vertex normals when no normal texture present
- **Per-material bind groups** — Each material now has its own bind group with all three textures (baseColor, metallicRoughness, normal)
- **Dual matrix upload** — Both MVP (clip space) and Model (world space) matrices uploaded per node for correct world-space lighting
- **Camera uniform buffer** (slot 18) — Passes camera position and light direction to shaders for proper specular calculations
- **Fixed lighting calculations** — World-space normals, proper view direction from camera position, lower ambient (0.05) for higher contrast
- **Default textures** for missing material properties:
  - White for baseColor
  - (0,1,1,1) for metallicRoughness (roughness=1, metallic=1)
  - (0.5,0.5,1,1) for normal (flat normal in tangent space)

### Changed
- Upgraded `gltf` dependency from v0.3.6 to v0.4.0
  - **BREAKING**: `GLTF::loadGLTF()` now requires a callback for loading external resources
  - Matrix transposition is now handled internally by the gltf library — removed manual transpose workarounds
- Updated all `GLTF::loadGLTF()` calls in tests and examples to use the new callback-based API
- **Bind group layout expanded** — Now supports 6 bindings (3 textures + 3 samplers)
- **Material uniform buffer expanded** — New layout includes metallicFactor, roughnessFactor, normalScale, hasNormalTexture flag
- Updated Metal shaders with new PBR lighting model

### Fixed
- Dependency cmake files (`campello_gpu.cmake`, `campello_image.cmake`, `gltf.cmake`) now guard against re-fetching targets already defined by a parent project — prevents CMake target redefinition errors when `campello_renderer` is consumed as a sub-project

## [0.1.2] - 2026-04-06

### Added
- **CI/CD workflows** (`.github/workflows/`) — Automated builds for all platforms
  - `build.yml` — Build and test on Linux, Windows, macOS, iOS, Android
  - `release.yml` — Automated release packaging on version tags
  - `code-quality.yml` — Formatting and static analysis checks
- **Multi-platform CI support**:
  - Linux (Ubuntu) — `.so` library
  - Windows (MSVC x64) — `.dll` + `.lib`
  - macOS — `.dylib` library + example app bundle
  - iOS — static `.a` library (arm64)
  - Android — `.so` for arm64-v8a

### Changed
- Upgraded `campello_gpu` dependency from v0.5.1 to v0.6.0
- Restricted Android ABI targets to `arm64-v8a` only — `vector_math` (pulled in by campello_gpu v0.6.0) uses ARMv8 NEON intrinsics (`vfmaq_laneq_f32`, `vpaddq_f32`) unavailable on armeabi-v7a
- Upgraded `gltf` dependency from v0.3.2 to v0.3.5
- Upgraded `campello_image` dependency from v0.3.0 to v0.3.1
- Enabled Unity Build for faster compilation (main library and test executable)

### Fixed
- Fixed `stencilRadOnly` → `stencilReadOnly` in `src/campello_renderer.cpp` — field was renamed in campello_gpu v0.6.0 (`DepthStencilAttachment`)
- Fixed version string in `test/main.cpp` — test expected "0.1.1" but library returned "0.1.2", causing CI failures
- Fixed Windows CI configure step — changed from PowerShell backticks to `cmd` shell with `^` line continuation

### CI/CD
- **Linux CI partially enabled** — Linux/Vulkan backend is still placeholder only; library builds but tests are disabled
- **Windows CI build-only** — Tests disabled due to DLL import library (.lib) generation issue; library builds successfully

### Documentation
- Added versioning checklist to `CLAUDE.md` — documents all files that must be updated when bumping version (CMakeLists.txt, test/main.cpp, etc.)

## [0.1.1] - 2026-03-23

### Added
- **macOS example** (`examples/macos/`) — `ViewController`-based app demonstrating the renderer on macOS via `MTKView`
- **Shader system** (`shaders/`, `src/shaders/`) — Metal, Vulkan, and DirectX shader sources with embedded header generation; build scripts `build_metal_shaders.sh`, `build_vulkan_shaders.sh`
- **WebP image decoding** — embedded images in GLTF buffer views are now decoded via `libwebp` (detected by RIFF/WEBP magic bytes) in addition to stb_image formats
- **WebP CMake dependency** (`dependencies/webp.cmake`)
- **`resize(width, height)`** — notifies the renderer of swapchain dimensions; creates/recreates the depth buffer and depth view
- **`createDefaultPipelines(colorFormat)`** — builds flat and textured pipeline variants for a given color format; replaces the old single-pipeline approach
- **`render(colorView)`** — overload that renders to an externally provided `TextureView` (e.g., macOS `MTKView` drawable); uses the renderer's own depth buffer
- **`setCameraMatrices(viewColMajor16, projColMajor16)`** / **`clearCameraOverride()`** — inject an external view/projection matrix pair, overriding the active GLTF camera
- **`getBoundsRadius()`** — returns the approximate bounding radius of the current scene, computed from node world-space positions in `setScene()`
- **GPU resource accessors** — `getGpuBuffer(i)`, `getGpuTexture(i)`, `getGpuBufferCount()`, `getGpuTextureCount()`, `getBindGroup(i)`, `getBindGroupCount()`, `getDefaultBindGroup()`
- **Vertex slot constants** — `VERTEX_SLOT_POSITION`, `VERTEX_SLOT_NORMAL`, `VERTEX_SLOT_TEXCOORD0`, `VERTEX_SLOT_TANGENT`, `VERTEX_SLOT_MVP`, `VERTEX_SLOT_MATERIAL` matching pipeline and shader bindings
- **Per-node transform buffer** — `setScene()` allocates a GPU buffer (one `float4x4` per node) and recomputes MVP matrices each frame
- **Material uniform buffer** — per-material `baseColorFactor` uploaded to the GPU
- **Fallback UV buffer** — zero-UV buffer bound for primitives without `TEXCOORD_0`
- **Bind group infrastructure** — `setScene()` now creates a `BindGroupLayout`, per-GLTF-sampler GPU samplers, per-GLTF-texture bind groups, a 1×1 white default texture, and a default bind group
- **macOS build script** (`build_macos_example.sh`)
- **`.gitignore` additions** — intermediate shader compiler artifacts (`*.air`, `*.metallib`, `*.spv`, `*.dxil`, generated headers)

### Changed
- Upgraded `campello_gpu` dependency from v0.3.6 to v0.3.8
  - v0.3.7: Added alpha-blending support (`BlendFactor`, `BlendOperation`, `BlendComponent`, `BlendState`) across Metal, Vulkan, and DirectX 12 backends
  - v0.3.8: Added DirectX 12 indirect drawing (`drawIndirect`/`drawIndexedIndirect`), indirect compute dispatch, occlusion query methods, and dynamic swapchain resizing; fixed missing device data forwarding in compute passes and unreleased cached command signatures in Device destructor
- Dependency cmake files moved from the root to `dependencies/` (`campello_gpu.cmake` → `dependencies/campello_gpu.cmake`, `gltf.cmake` → `dependencies/gltf.cmake`)
- `setScene()` now fully uploads GLTF binary buffers to GPU, decodes and uploads all referenced images, builds GPU samplers and bind groups, and computes the scene bounding radius
- `setAsset(nullptr)` now also clears `nodeTransforms`, `transformBuffer`, `materialUniformBuffer`, `gpuSamplers`, and `gpuBindGroups`
- Texture pixel format changed from `rgba8uint` to `rgba8unorm` for correct color sampling
- `render()` and `update()` promoted from stubs to a full implementation

### Fixed
- `Image` destructor: replaced `delete data` with `free(data)` — pixel data is allocated by `stbi_load_from_memory` / `WebPDecodeRGBA` (both use `malloc`), so `delete` was undefined behavior

## [0.1.0] - 2026-03-14

### Changed
- Upgraded C++ standard from C++17 to C++20
- Upgraded `campello_gpu` dependency from v0.1.1 to v0.2.0
  - Adapted `TextureUsage::shaderRead` → `TextureUsage::textureBinding`
  - Adapted `Device::createTexture` to new signature (`TextureType`, removed `StorageMode`, added `depth`, `mipLevels`, `samples`)

### Added
- Expanded unit test suite from 9 to 37 tests covering version, construction, `setAsset`, `setScene`, `setCamera`, `render`, `update`, and multi-step sequences

## [0.0.3] - initial

- Initial release
