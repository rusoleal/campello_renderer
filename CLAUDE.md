# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`campello_renderer` (v0.2.1) is a C++20 shared library that provides a 3D rendering layer on top of custom dependencies:
- **campello_gpu** (v0.11.1) — low-level multiplatform GPU abstraction (Vulkan, Metal, DirectX)
- **gltf** (v0.4.1) — GLTF/GLB asset loader
- **campello_image** (v0.4.0) — image decoding (PNG, JPEG, WebP, HDR, OpenEXR)

The library is consumed as a CMake dependency and targets Android, macOS, iOS, Windows, and Linux. macOS is the primary development/test platform.

## Build System

Dependencies are fetched automatically via `FetchContent` from GitHub (see `dependencies/`). The main build entry points:

```bash
# macOS (tests enabled) — Debug
cmake -S . -B build/macos/debug -DCAMPELLO_RENDERER_BUILD_TEST=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build/macos/debug
ctest --test-dir build/macos/debug

# macOS example app — Debug or Release
./build_macos_example.sh Debug    # → build/macos/debug
./build_macos_example.sh Release  # → build/macos/release

# Then run with or without Metal API Validation:
./run_macos_example_debug.sh      # MTL_DEBUG_LAYER=1
./run_macos_example_release.sh    # no validation

# Android (primary runtime target)
cd examples/android
./gradlew build          # Debug + Release
./gradlew assembleDebug  # Debug APK only
```

CMake minimum version: 3.22.1.

## CI

Five jobs in `.github/workflows/build.yml`: Linux, Windows, macOS, iOS, Android.
- **Tests run on macOS only** — Linux and Windows test steps are commented out (Vulkan/DirectX backends are incomplete).
- `actions/checkout@v4` is the correct version to use — the workflow currently has `@v6` which does not exist (bug in uncommitted changes).

## Architecture

### Core Class: `Renderer`

Defined in `inc/campello_renderer/campello_renderer.hpp`, implemented in `src/campello_renderer.cpp`.

**Typical usage sequence:**
1. Construct with an injected `shared_ptr<Device>`.
2. Call `createDefaultPipelines(colorFormat)` once to compile shader pipelines.
3. Call `resize(width, height)` once (and on surface resize) to create the depth buffer.
4. Call `setAsset(gltf)` to load a GLTF/GLB and upload GPU buffers/textures.
5. Call `render()` or `render(colorView)` each frame.

**Public API summary:**
- `Renderer(device)` — GPU backend is injected, not created internally.
- `setAsset(shared_ptr<GLTF>)` / `getAsset()` — load/clear a GLTF asset; allocates GPU buffers, textures, samplers, bind groups.
- `setScene(index)` — select active scene within the asset.
- `setCamera(index)` — select active GLTF camera.
- `resize(width, height)` — create/recreate depth texture; must be called before `render()`.
- `createDefaultPipelines(colorFormat)` — compile built-in pipeline variants (flat + textured); must be called before `render()`.
- `render()` — render to the device's swapchain (Android / device-owns-surface platforms).
- `render(colorView)` — render to an external color target (e.g., MTKView drawable on macOS).
- `update(double dt)` — reserved for animation; currently a no-op.
- `setCameraMatrices(view16, proj16)` / `clearCameraOverride()` — override GLTF camera with externally supplied column-major float[16] matrices.
- `getBoundsRadius()` — approximate bounding radius of the current scene.
- `getGpuBuffer/Texture/BindGroup(index)` and count accessors.
- `getDefaultBindGroup()` — the 1×1 white texture bind group used as fallback.

**Vertex buffer slot convention (must match shaders):**

| Slot | Semantic | Type |
|------|----------|------|
| 0 | POSITION | float3 |
| 1 | NORMAL | float3 |
| 2 | TEXCOORD_0 | float2 |
| 3 | TANGENT | float4 |
| 16 | MVP matrix (per-instance) | float4×4 |
| 17 | MaterialUniforms (baseColorFactor) | float4 |

### Resource Lifecycle

GPU resources (`gpuBuffers`, `gpuTextures`, `gpuSamplers`, `gpuBindGroups`) are allocated in `setScene()`. The default sampler, default texture (1×1 white), bind group layout, and default bind group are lazy-initialized on the first `setScene()` call and reused across asset swaps.

Images are decoded from GLTF buffer views via `campello_image::Image::fromMemory()`. `data:uri` images are not yet supported (TODO).

### Pipeline Variants

Two variants are created by `createDefaultPipelines()`:
- `pipelineFlat` — Phong shading using `baseColorFactor` only (no texture lookup).
- `pipelineTextured` — Phong shading sampling `baseColorTexture × baseColorFactor`.

`renderPrimitive()` selects the variant per-primitive based on whether the material has a `baseColorTexture` and the primitive has `TEXCOORD_0`. Pipeline changes are minimized by tracking `currentPipelineVariant`.

**Platform shader status:**
- **macOS**: Embedded `.metallib` (`src/shaders/metal_default.h`). Both flat and textured variants are fully implemented.
- **Android/Vulkan**: Embedded SPIR-V (`src/shaders/vulkan_default.h`). `pipelineFlat = pipelineTextured` — separate flat fragment shader TODO. Also, `campello_gpu` Vulkan backend currently hardcodes `vertexBindingDescriptionCount = 0`, so vertex input descriptors are passed but not applied (upstream gap).
- **Windows/DirectX**: DXIL binaries not yet compiled — pipelines remain null, Windows rendering is nonfunctional. HLSL source at `src/shaders/directx/`.

### Key Files

- `inc/campello_renderer/campello_renderer.hpp` — public API
- `inc/campello_renderer/image.hpp` — legacy `Image` struct (not used by renderer internals; kept for API compatibility)
- `src/campello_renderer.cpp` — full implementation (~1050 lines)
- `src/shaders/metal_default.h` — embedded Metal shader library
- `src/shaders/vulkan_default.h` — embedded Vulkan SPIR-V
- `src/shaders/directx_default.h` — DirectX DXIL (currently empty stubs)
- `dependencies/campello_gpu.cmake` / `gltf.cmake` / `campello_image.cmake` — FetchContent declarations

### Examples

- **Android**: `examples/android/` — Kotlin `MainActivity`, native layer at `app/src/main/cpp/`.
- **macOS**: `examples/macos/` — Objective-C++ app using `ViewController.mm` and `AppDelegate.mm`. Enabled via `-DCAMPELLO_RENDERER_BUILD_MACOS_EXAMPLE=ON`.

### Tests

Located in `test/main.cpp`. Uses Google Test (fetched via `dependencies/googletest.cmake`). Enabled via `-DCAMPELLO_RENDERER_BUILD_TEST=ON` (macOS only in CI).

Tests cover: version string, construction, `setAsset`/`getAsset`, `setScene`, `setCamera`, `render`, `update`, and multi-step lifecycle sequences. All tests run without a real GPU device (`nullptr` device).

## Versioning

Version is defined in `CMakeLists.txt` (`project(campello_renderer VERSION 0.2.1)`) and propagated to `campello_renderer_config.h` via `configure_file`.

**When upgrading the package version, update the version string in ALL of these locations:**
1. `CMakeLists.txt` — `project(campello_renderer VERSION x.x.x)`
2. `test/main.cpp` — `ReturnsExpectedVersion` test assertion
3. `examples/android/app/build.gradle.kts` — `versionName` (if applicable)

**CI Note:** The `ReturnsExpectedVersion` test in `test/main.cpp` expects an exact version string match. If the versions are out of sync, macOS Debug CI will fail with:
```
Expected equality of these values:
  systems::leal::campello_renderer::getVersion()
    Which is: "0.2.0"
  "0.1.1"
```
Always verify tests pass after version bumps.
