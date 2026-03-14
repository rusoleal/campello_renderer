# campello_renderer

**v0.1.0** — Native 3D renderer library (C++20, shared library)

## Dependencies

* [systems::leal::gltf](https://github.com/rusoleal/gltf) v0.2.1 — C++ glTF/GLB asset loader
* [systems::leal::campello_gpu](https://github.com/rusoleal/campello_gpu) v0.2.0 — Low-level multiplatform GPU abstraction (Vulkan, Metal, DirectX)

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
