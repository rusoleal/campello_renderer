# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`campello_renderer` (v0.29.0) is a C++20 shared library that provides a 3D rendering layer on top of custom dependencies:
- **campello_gpu** (v0.34.0) — low-level multiplatform GPU abstraction (Vulkan, Metal, DirectX)
- **gltf** (v0.7.0) — GLTF/GLB asset loader
- **campello_procedural** (v0.4.0) — procedural texture graphs (MaterialX nodes, KHR_texture_procedurals), baked on the CPU
- **campello_image** (v0.7.0) — image decoding (PNG, JPEG, WebP, HDR, OpenEXR)

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

# Frame benchmark (Release; scenarios: draws, shadows, gltf, gbuffer, denoise)
cmake -S . -B build/macos/release -DCAMPELLO_RENDERER_BUILD_BENCH=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/macos/release --target campello_renderer_bench
./build/macos/release/campello_renderer_bench [scenario] [frames]

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
- `update(double dt)` — advances playing animations (`GltfAnimator`), applying TRS/morph-weight/material-pointer channels; pass the real elapsed time since the last call, not a fixed timestep (a hardcoded 1/60 here previously caused animations to run in slow motion on any frame slower than 16.7ms — see the macOS example's `drawInMTKView:`).
- `playAnimation` / `pauseAnimation` / `stopAnimation` / `setAnimationTime` / `getAnimationCount` / `getAnimationName` / `getAnimationDuration` — animation playback controls, delegated to `GltfAnimator`.
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
| 4 | JOINTS_0 | uint4 |
| 5 | WEIGHTS_0 | float4 |
| 6 | COLOR_0 (normalized to float4 on upload) | float4 |
| 7 | TEXCOORD_1 | float2 |
| 16 | MVP matrix (per-instance) | float4×4 |
| 17 | MaterialUniforms (baseColorFactor) | float4 |
| 18 | CameraPosition | float3 |
| 19 | Instance transform (EXT_mesh_gpu_instancing; Metal: indexed by `[[instance_id]]`, Vulkan: per-instance locations 12-15) | float4×4 |
| 20 | Joint matrix palette (per-skin; Vulkan: storage buffer, set 2 binding 0) | float4×4[] |
| 21 | MorphInfo (targetCount, hasNormal, vertexCount, weights[8]; Vulkan: storage buffer, set 3 binding 0) | raw struct |
| 22 | Morph target POSITION deltas, indexed via `[[vertex_id]]` (Vulkan: set 3 binding 1, `gl_VertexIndex`) | float3[] |
| 23 | Morph target NORMAL deltas, indexed via `[[vertex_id]]` (Vulkan: set 3 binding 2) | float3[] |
| 26 | Wide line segments (`GpuMesh::wideLines`) | raw |
| 27 | G-buffer: previous frame's MVP per draw (same offset/stride as slot 16) | float4x4 + pad |
| 28 | G-buffer: previous frame's joint matrix palette (Vulkan: set 2 binding 1) | float4x4[] |

Slots 21–23 are raw `device const T*` buffers, not part of the `[[stage_in]]` vertex descriptor (target count varies per primitive) — bound directly in `renderPrimitive()`/`drawDrawCall()`, always with a valid (possibly zero-weight) fallback since `MorphInfo` is a non-pointer `constant` reference in the shader and Metal requires something bound there regardless of use.

### Resource Lifecycle

GPU resources (`gpuBuffers`, `gpuTextures`, `gpuSamplers`, `gpuBindGroups`) are allocated in `setScene()`. The default sampler, default texture (1×1 white), bind group layout, and default bind group are lazy-initialized on the first `setScene()` call and reused across asset swaps.

Images are decoded from GLTF buffer views via `campello_image::Image::fromMemory()`. `data:uri` images are not yet supported (TODO).

Sparse accessors (`accessor.sparse`) are resolved for vertex attributes (including morph target deltas and COLOR_0) via the shared `resolveAccessorBytes()` helper in `src/campello_renderer.cpp`, which reads the base bufferView data (or an implicit zero base if absent) and overlays the sparse indices/values on top. Sparse `primitive.indices` accessors are not handled (rare in practice; the existing UNSIGNED_BYTE index-widening logic doesn't call the sparse resolver).

Morph targets (`mesh.primitives[].targets`) are supported for POSITION and NORMAL (not TANGENT). Weight precedence is `node.weights` > `mesh.weights` > the `GltfAnimator`'s `"weights"` animation channel override, resolved per-node each frame in `updateMorphWeights()`.

### Pipeline Variants

Two variants are created by `createDefaultPipelines()`:
- `pipelineFlat` — Phong shading using `baseColorFactor` only (no texture lookup).
- `pipelineTextured` — Phong shading sampling `baseColorTexture × baseColorFactor`.

`renderPrimitive()` selects the variant per-primitive based on whether the material has a `baseColorTexture` and the primitive has `TEXCOORD_0`. Pipeline changes are minimized by tracking `currentPipelineVariant`.

**Platform shader status:**
- **macOS**: Embedded `.metallib` (`src/shaders/metal_default.h`). Both flat and textured variants are fully implemented.
- **Android/Vulkan**: Embedded SPIR-V (`src/shaders/vulkan_default.h`, `vulkan_denoiser.h`), generated by `build_vulkan_shaders.sh` from `shaders/vulkan/` (one line per shader there). One PBR fragment shader (no flat/debug variants, no blending). Skinning and morph targets: `default_skinned.vert`/`gbuffer_skinned.vert` (`skinning.glsl`), picked by `meshPipeline()` only for draws with a skin or morph targets; the joint palette is a storage buffer at set 2 (binding 1: the previous palette, for G-buffer motion), bound by `bindSkin()`, the morph targets set 3, bound by `bindMorph()`. The per-draw transforms have stride 0 (every instance reads the draw's), so EXT_mesh_gpu_instancing instances differ only by their instance matrix (locations 12-15). Material specialization: `default.frag` reads the optional PBR features' uniforms through `MAT_<field>` macros on specialization constant 0 (`kMaterialFeatures`, set per pipeline by `meshPipeline()` from `materialFeaturesOf()`); don't copy the `MaterialUniforms` block into a local struct (an Adreno 642L kept it in scratch memory, 7x slower). Sun shadow casters with instances or a skin use `shadow.vert`/`shadow_skinned.vert` (`shadow.glsl`; plain casters too, with an identity instance), Metal's `shadowVertexInstanced`/`shadowVertexSkinned` (`shadowDeformPipeline()`). Has the G-buffer (`gbuffer.vert/frag`; the previous transforms are a per-instance vertex buffer at locations 20-23/28), TAA and the built-in upscaler (`taa.frag`) and `RealtimeDenoiser` (`denoise_*.frag`); no ray tracing. Test on a device with `./run_android_tests.sh [gtest filter]` (`CAMPELLO_GPU_SOURCE=<checkout>` to build against a local campello_gpu).
- **Windows/DirectX**: Embedded DXIL (`src/shaders/directx_default.h`, `directx_denoiser.h`), generated by `build_directx_shaders.ps1` (dxc from the Windows SDK; one line per entry point) from `shaders/directx/default.hlsl` and `denoiser.hlsl`. Full PBR with Vulkan's bind groups and layouts (`CAMPELLO_RENDERER_PBR_BIND_GROUPS`): 0 material, 1 frame, 2 joint palettes, 3 morph targets; campello_gpu puts group N in register space N, so `default.hlsl` declares each group's Vulkan binding numbers in `spaceN`. Skinning and morph targets: `vertexSkinned`/`gbufferSkinnedVertex`, palettes and deltas as raw `ByteAddressBuffer`s, bound by the same `bindSkin()`/`bindMorph()` as Vulkan (`CAMPELLO_RENDERER_SPLIT_DEFORM`); the deform buffers stay CPU-written upload buffers on DirectX. Storage buffers are raw views in campello_gpu, so DirectX compute shaders use `(RW)ByteAddressBuffer` (e.g. `proceduralBakeKernel`). EXT_mesh_gpu_instancing (locations 12-15, transforms with stride 0), position-only meshes (`leanVertex`), lines (`lineVertex`), clip planes and section caps, sun shadow maps (`shadowLookup()`; depth-only `shadowVertex`, and `shadowSkinnedVertex` for instanced and skinned casters via `shadowDeformPipeline()`; the lean stages apply instance matrices), SSAO (`ssaoPixel`/`ssaoApplyPixel`), the G-buffer (`gbufferVertex`/`gbufferLeanVertex`/`gbufferPixel`), TAA and the built-in upscaler (`taaPixel`) and `RealtimeDenoiser`. No wide lines (Metal-only everywhere) and no ray tracing (the dev machine's Intel Iris 550 and the Xbox in UWP Developer Mode have no DXR). Test programs: `test/shaders/element_id.hlsl` (dxc is found in the Windows SDK by `CampelloRendererShaders.cmake`).

### Key Files

- `inc/campello_renderer/campello_renderer.hpp` — public API
- `inc/campello_renderer/shaders/program_interface.*` — the app shader program contract; the Metal one also holds the lighting inputs (camera/light/G-buffer structs, effect helpers) and is included by `shaders/metal/default.metal` (`build_metal_shaders.sh` passes `-I inc`)
- `docs/using-the-renderer.md` — usage levels, per-feature switches, custom shading
- `inc/campello_renderer/image.hpp` — legacy `Image` struct (not used by renderer internals; kept for API compatibility)
- `src/campello_renderer.cpp` — full implementation (~1050 lines)
- `src/shaders/metal_default.h` — embedded Metal shader library
- `src/path_tracer.cpp`, `shaders/metal/path_tracer.metal` — `PathTracer` (progressive GPU path tracing; `setFrameBudget()` traces a sample in bands of rows, up to two calls in flight; `denoiseAsync()` denoises on a worker thread)
- `src/realtime_denoiser.cpp`, `shaders/metal/denoiser.metal` — `RealtimeDenoiser` (SVGF, G-buffer guided)
- `src/rt_effects.cpp`, `shaders/metal/rt_effects.metal` — `RayTracingScene`, `RayTracedShadows`, `RayTracedAmbientOcclusion`, `RestirDirectLighting`, `RestirGlobalIllumination`, `SkinnedMeshes` (skinned draws posed for ray tracing; alpha-masked casters alpha tested via `intersection_query`) (ray traced effects on the raster renderer; ReSTIR DI feeds `Renderer::setDirectLighting()`, ReSTIR GI `setIndirectDiffuse()` with samples from `PathTracer::traceGlobalIllumination()`)
- `src/realtime_effects.cpp` — `RealtimeEffects`: the effect chain with quality presets (`EffectQuality`), per-effect GPU timings (`GpuTiming`, `gpu_timing.hpp`) and dynamic resolution
- `Renderer::compressedTexture()` (`setTextureCompression()`): RGBA8 pixels encoded with campello_image's `compressRGBA8()` to the device's format (`chooseBasisTargetFormat()`: ASTC, BC7, ETC2), with mips, uploaded by `uploadTextureDataWithMips()`. Null where it doesn't apply, and the caller uploads RGBA8. Used by `uploadMaterial()`'s images and `proceduralTexture()`.
- `src/procedural_texture_baker.cpp`, `src/procedural_bridge.hpp` — KHR_texture_procedurals: gltf's graphs converted to campello_procedural graphs (`procedural::toGraph()`), images for `image` nodes, `bakeProceduralTexture()`. `Renderer::proceduralTexture()` bakes a material slot: one bake per graph (`proceduralBakes`, released once `setScene()`/`uploadMaterial()` has made the textures), textures cached by graph content, output, encoding and size (`proceduralBakedTextures`), sRGB for colour slots, mipmapped, `proceduralSampler`. campello_procedural is linked privately (`dependencies/campello_procedural.cmake`; `CAMPELLO_PROCEDURAL_SOURCE=<checkout>` for a local one). The GPU bake kernels (`proceduralBakeKernel`, `procedural_bake.comp`) are unused.
- `src/ddgi.cpp`, `shaders/metal/ddgi.metal` — `DdgiVolume` (probe-based GI; rays traced by `PathTracer::traceProbeRays()`)
- `src/shaders/vulkan_default.h`, `vulkan_denoiser.h` — embedded Vulkan SPIR-V (`build_vulkan_shaders.sh`)
- `src/shaders/directx_default.h` — DirectX DXIL (currently empty stubs)
- `dependencies/campello_gpu.cmake` / `gltf.cmake` / `campello_image.cmake` — FetchContent declarations

### Examples

- **Android**: `examples/android/` — Kotlin `MainActivity`, native layer at `app/src/main/cpp/`.
- **macOS**: `examples/macos/` — Objective-C++ app using `ViewController.mm` and `AppDelegate.mm`. Enabled via `-DCAMPELLO_RENDERER_BUILD_MACOS_EXAMPLE=ON`.
- **UWP (Windows PC and Xbox Developer Mode)**: `examples/uwp/`, see its README. Enabled via `-DCMAKE_SYSTEM_NAME=WindowsStore -DCAMPELLO_RENDERER_BUILD_UWP_EXAMPLE=ON`.

### Tests

Located in `test/main.cpp`. Uses Google Test (fetched via `dependencies/googletest.cmake`). Enabled via `-DCAMPELLO_RENDERER_BUILD_TEST=ON` (macOS only in CI).

Tests cover: version string, construction, `setAsset`/`getAsset`, `setScene`, `setCamera`, `render`, `update`, and multi-step lifecycle sequences. All tests run without a real GPU device (`nullptr` device).

## Versioning

Version is defined in `CMakeLists.txt` (`project(campello_renderer VERSION 0.29.0)`) and propagated to `campello_renderer_config.h` via `configure_file`.

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

## Releasing

1. Add a new `## [x.y.z] - YYYY-MM-DD` entry to `CHANGELOG.md` (Added/Changed/Fixed sections), matching the style of existing entries.
2. Bump the version in all locations listed under Versioning above; rebuild and run `ctest` to confirm the version-string test passes.
3. Commit and push to `main`.
4. Create the GitHub release with `gh release create vX.Y.Z --title vX.Y.Z ...` — pushing the tag triggers `.github/workflows/release.yml` (`on: push: tags: 'v*'`), which builds Linux/Windows/macOS/iOS/Android packages and attaches them via `softprops/action-gh-release`.
5. **Wait for that workflow to finish** (`gh run watch <run-id>` or `gh run list --workflow release.yml`) before writing the final release notes — its `create-release` job runs with `generate_release_notes: true` and no explicit `body`, which overwrites whatever notes were set at creation time with GitHub's raw auto-generated commit list.
6. Once CI has finished, set the real release notes with `gh release edit vX.Y.Z --notes "..."`. The release description must contain:
   - A short (3–6 bullet) summary of the version's highlights, drawn from the new `CHANGELOG.md` entry — not the raw auto-generated commit list.
   - A link to the full changelog detail, e.g. `https://github.com/rusoleal/campello_renderer/blob/vX.Y.Z/CHANGELOG.md#xyz---yyyy-mm-dd`.
