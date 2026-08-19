# campello_renderer

A native, cross-platform 3D renderer for glTF 2.0 scenes, built on top of [campello_gpu](https://github.com/rusoleal/campello_gpu) (Vulkan/Metal/DirectX 12 abstraction) and [gltf](https://github.com/rusoleal/gltf) (C++ glTF/GLB loader). C++20, ships as a shared library.

It implements a full metallic-roughness PBR pipeline with image-based lighting, matching the [Khronos glTF-Sample-Viewer](https://github.com/KhronosGroup/glTF-Sample-Viewer) reference as closely as practical, across Metal, Vulkan, and DirectX 12 from a single shared C++ core.

[![Build](https://github.com/rusoleal/campello_renderer/actions/workflows/build.yml/badge.svg)](https://github.com/rusoleal/campello_renderer/actions/workflows/build.yml)
[![Release](https://github.com/rusoleal/campello_renderer/actions/workflows/release.yml/badge.svg)](https://github.com/rusoleal/campello_renderer/actions/workflows/release.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🚀 Part of the Campello Engine

This project is a module within the **Campello** ecosystem.

👉 Main repository: https://github.com/rusoleal/campello

Campello is a modular, composable game engine built as a collection of independent libraries.
Each module is designed to work standalone, but integrates seamlessly into the engine runtime.

## Features

- **Metallic-roughness PBR** — Cook-Torrance GGX BRDF, normal mapping, punctual lights (`KHR_lights_punctual`)
- **Image-based lighting** — GGX-prefiltered specular + Lambertian diffuse irradiance cubemaps and a BRDF LUT, baked in-engine from any equirectangular or cubemap environment
- **glTF extensions** — `KHR_materials_iridescence`, `KHR_materials_emissive_strength`, `KHR_materials_transmission` (screen-space refraction), `KHR_materials_clearcoat`/`sheen`/`specular`/`anisotropy`, `KHR_texture_basisu` (GPU-compressed textures), `KHR_texture_procedurals` (CPU/GPU procedural texture baking), `EXT_mesh_gpu_instancing`, sparse accessors
- **Animation & deformation** — skeletal animation, joint skinning, morph targets (POSITION/NORMAL blending, up to 8 targets per primitive)
- **Post-processing** — FXAA and SSAA anti-aliasing
- **Offscreen rendering** — render to an arbitrary texture and read pixels back, no window/swapchain required
- **Optional ECS bridge** (`ecs.hpp`) — drives the renderer from a [campello_core](https://github.com/rusoleal/campello_core) ECS world (`Transform`/`Camera`/`MeshRenderer`/light components) instead of a loaded glTF scene graph
- **Three GPU backends from one shared core** — Metal (macOS/iOS), Vulkan (Android/Linux), DirectX 12 (Windows)

## Supported Platforms

| Platform | Status | GPU Backend | Artifacts |
|----------|--------|--------------|-----------|
| macOS | ✅ | Metal | `.dylib` library + example app |
| iOS | ✅ | Metal | `.a` static library |
| Android | ✅ | Vulkan | `.so` library + APK |
| Windows | ✅ | DirectX 12 | `.dll` library + example app |
| Linux | ⚠️ (build only) | Vulkan | `.so` library — Vulkan backend incomplete, tests disabled in CI |

## Quick Start

```cpp
#include <campello_renderer/campello_renderer.hpp>
#include <gltf/gltf.hpp>

using namespace systems::leal::campello_renderer;
using namespace systems::leal::gltf;

// `device` is a campello_gpu::Device your platform's windowing code created
// (a Metal/Vulkan/DirectX 12 device wired to your window's surface).
auto renderer = std::make_shared<Renderer>(device);

renderer->createDefaultPipelines(colorFormat); // compile built-in shader pipelines, once
renderer->resize(width, height);               // create the depth buffer; call again on resize

auto asset = GLTF::loadGLB(fileBytes, fileSize);
renderer->setAsset(asset);                     // upload GPU buffers/textures for the scene

// Each frame:
renderer->update(deltaTimeSeconds);            // advance animations (real elapsed time, not a fixed step)
renderer->render(colorView);                   // render to your swapchain/offscreen target
// — or renderer->render() to target the device's own swapchain directly (Android and similar).
```

See `inc/campello_renderer/campello_renderer.hpp` for the full public API, and `examples/` for complete, platform-specific applications (drag-and-drop glTF/HDR loading, orbit camera, animation playback).

## Dependencies

* [systems::leal::gltf](https://github.com/rusoleal/gltf) v0.5.1 — C++ glTF/GLB asset loader
* [systems::leal::campello_gpu](https://github.com/rusoleal/campello_gpu) v0.23.1 — Low-level multiplatform GPU abstraction (Vulkan, Metal, DirectX 12)
* [systems::leal::campello_image](https://github.com/rusoleal/campello_image) v0.5.0 — Image decoding (PNG, JPEG, WebP, HDR, OpenEXR) + GPU-compressed texture transcoding (Basis Universal / KTX2)
* [systems::leal::vector_math](https://github.com/rusoleal/vector_math) v0.6.0 — Vector/matrix/quaternion math, SIMD-accelerated

All dependencies are fetched automatically via CMake `FetchContent` — no manual setup required.

## Building

The library is consumed as a CMake dependency. Minimum CMake version: 3.22.1.

### Android (primary runtime target)

```bash
cd examples/android
./gradlew assembleDebug   # debug APK
./gradlew build           # debug + release
```

### macOS

```bash
# Library + unit tests
cmake -S . -B build/macos/debug -DCAMPELLO_RENDERER_BUILD_TEST=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build/macos/debug
ctest --test-dir build/macos/debug

# Example app (Debug or Release), with or without Metal API Validation
./build_macos_example.sh Debug
./run_macos_example_debug.sh
```

### Windows

```powershell
cmake -S . -B build/windows/debug -DCAMPELLO_RENDERER_BUILD_TEST=ON -DCAMPELLO_RENDERER_BUILD_WINDOWS_EXAMPLE=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build/windows/debug --config Debug
.\build\windows\debug\Debug\campello_renderer_test.exe
```

### Linux

```bash
./build_linux_example.sh
./run_linux_example_debug.sh
```

### iOS

```bash
./build_ios_example.sh
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md).
