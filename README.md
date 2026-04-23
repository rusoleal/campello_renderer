# campello_renderer

Native 3D renderer library built on top of [campello_gpu](https://github.com/rusoleal/campello_gpu) and [gltf](https://github.com/rusoleal/gltf) (C++20, shared library)

[![Build](https://github.com/rusoleal/campello_renderer/actions/workflows/build.yml/badge.svg)](https://github.com/rusoleal/campello_renderer/actions/workflows/build.yml)
[![Release](https://github.com/rusoleal/campello_renderer/actions/workflows/release.yml/badge.svg)](https://github.com/rusoleal/campello_renderer/actions/workflows/release.yml)

## 🚀 Part of the Campello Engine

This project is a module within the **Campello** ecosystem.

👉 Main repository: https://github.com/rusoleal/campello

Campello is a modular, composable game engine built as a collection of independent libraries.
Each module is designed to work standalone, but integrates seamlessly into the engine runtime.

## Supported Platforms

| Platform | Status | Artifacts |
|----------|--------|-----------|
| Linux | ⚠️ (build only) | `.so` library — campello_gpu Vulkan backend is placeholder, tests disabled |
| Windows | ⚠️ (build only) | `.dll` library — tests disabled (DLL export issue) |
| macOS | ✅ | `.dylib` library + example app |
| iOS | ✅ | `.a` static library |
| Android | ✅ | `.so` library + APK |

## Dependencies

* [systems::leal::gltf](https://github.com/rusoleal/gltf) v0.4.1 — C++ glTF/GLB asset loader
* [systems::leal::campello_gpu](https://github.com/rusoleal/campello_gpu) v0.11.0 — Low-level multiplatform GPU abstraction (Vulkan, Metal, DirectX)
* [systems::leal::campello_image](https://github.com/rusoleal/campello_image) v0.4.0 — Image decoding (PNG, JPEG, WebP, HDR, OpenEXR)

## Building

The library is consumed as a CMake dependency. The primary build target is Android:

```bash
cd examples/android
./gradlew assembleDebug   # debug APK
./gradlew build           # debug + release
```

To run the unit test suite on the host machine:

```bash
cmake -B build -DCAMPELLO_RENDERER_BUILD_TEST=ON
cmake --build build
cd build && ctest --output-on-failure
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md).
