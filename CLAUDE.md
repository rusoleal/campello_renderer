# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`campello_renderer` (v0.0.3) is a C++17 shared library that provides a 3D rendering layer on top of two custom dependencies:
- **campello_gpu** — low-level multiplatform GPU abstraction (Vulkan, Metal, DirectX)
- **gltf** — GLTF/GLB asset loader

The library is consumed as a CMake dependency and currently targets Android as the primary platform.

## Build System

The library itself is built via CMake as part of a host project. The included Android example is the primary way to build and test:

```bash
cd examples/android
./gradlew build          # Build debug + release
./gradlew assembleDebug  # Build debug APK only
```

The Android example compiles:
- `libcampello_renderer.so` — the renderer library
- `libtest.so` — the example native layer

CMake minimum version: 3.22.1. Dependencies are fetched automatically via `FetchContent` from GitHub (see `campello_gpu.cmake` and `gltf.cmake`).

## Architecture

### Core Class: `Renderer`

Defined in `inc/campello_renderer/campello_renderer.hpp`, implemented in `src/campello_renderer.cpp`.

- Constructor takes a `shared_ptr<systems::leal::campello_gpu::Device>` — the GPU backend is injected, not created internally.
- `setAsset(path)` — loads a GLTF/GLB file and allocates GPU buffers/textures.
- `setScene(index)` — activates a scene from the loaded asset; extracts images from GLTF buffer views using stb_image and uploads as GPU textures.
- `setCamera(index)` — selects active camera.
- `render()` and `update(double dt)` — currently stubs, not yet implemented.

### Resource Lifecycle

GPU resources (`gpuBuffers`, `gpuTextures`) are created in `setScene()`, not at construction. Images are decoded from GLTF buffer views via `stbi_load_from_memory` and stored as `Image` structs before upload.

### Key Files

- `inc/campello_renderer/campello_renderer.hpp` — public API
- `inc/campello_renderer/image.hpp` — `Image` struct (format, dimensions, pixel data)
- `src/campello_renderer.cpp` — full implementation
- `src/stb_image.h` — vendored STB image decoder
- `campello_gpu.cmake` / `gltf.cmake` — FetchContent dependency declarations

### Android Example

Located in `examples/android/`. The Kotlin `MainActivity` loads the native library; `app/src/main/cpp/main.cpp` contains `android_main()` which instantiates the renderer.

## Versioning

Version is defined in `CMakeLists.txt` (`project(campello_renderer VERSION 0.0.3)`) and propagated to `campello_renderer_config.h` via `configure_file`.

**When upgrading the package version, update the version string in ALL of these locations:**
1. `CMakeLists.txt` — `project(campello_renderer VERSION x.x.x)`
2. `test/main.cpp` — `ReturnsExpectedVersion` test assertion
3. `examples/android/app/build.gradle.kts` — `versionName` (if applicable)
4. Any other files that hardcode the version string

**CI Note:** The `ReturnsExpectedVersion` test in `test/main.cpp` expects an exact version string match. If the versions are out of sync, macOS Debug CI (and other test-enabled builds) will fail with:
```
Expected equality of these values:
  systems::leal::campello_renderer::getVersion()
    Which is: "0.1.2"
  "0.1.1"
```
Always verify tests pass after version bumps.
