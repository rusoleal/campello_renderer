# Changelog

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
