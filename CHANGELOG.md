# Changelog

## Unreleased

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
