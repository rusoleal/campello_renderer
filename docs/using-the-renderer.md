# Using the renderer

campello_renderer can be used at several levels, from "load a glTF file and draw it" to "my own shaders, my own effect chain". Every feature beyond basic drawing is opt-in: it's off by default, or it falls back to the classic path when its input isn't set. This page lists the ways to use the renderer, what each feature needs, and what custom shading gets.

## 1. Ways to use it

### glTF viewer

The renderer owns the asset, the scene graph, the cameras and the animations.

```cpp
Renderer renderer(device);
renderer.createDefaultPipelines(colorFormat);
renderer.resize(width, height);
renderer.setAsset(gltf);
// each frame
renderer.update(dt);
renderer.render(colorView);   // or render() on the device's swapchain
```

### Your own scene (games, ECS)

You keep the scene. Each frame you fill a `RenderScene` with the camera, the lights and the opaque and transparent `DrawCall`s, then draw it. The ECS (`ecs.hpp`) fills it for you, including the previous transforms for motion vectors.

```cpp
renderer.render(scene, colorView);
```

The games features below (TAA, upscaling, the G-buffer and the ray traced effects) work on this path. `render(colorView)` with an asset doesn't run TAA.

### Offline path tracing

`PathTracer` runs on its own, from the same `RenderScene` and the renderer's uploaded meshes and materials. It converges progressively; `denoise()` runs Open Image Denoise.

```cpp
PathTracer tracer(device);
tracer.render(scene, renderer, width, height);   // one or more samples per call
tracer.denoise();
use(tracer.output());
```

### Building blocks for your own passes

- `renderGBuffer(scene)` and `gBuffer()` provide normal, roughness, albedo, metallic, motion and depth. They're the input for your own screen-space or ray traced effects.
- `renderProgramPass(scene, view, options)` draws the whole scene with one program, for example element ids into an `r32uint` target for picking.
- `setOverlayCallback()` records your own campello_gpu work into the frame's command encoder after the scene.
- `getDevice()` gives access to everything else.

## 2. Features and how to turn them on

| Feature | Turn on | Default | Backends | Custom programs |
|---|---|---|---|---|
| Image-based lighting | `setEnvironmentMap()`, `setIBLEnabled()` | on (built-in sky) | every backend | read the environment textures (see 3) |
| Sun shadow maps | `setShadowsEnabled()`, `LightData::castShadows` | on | Metal | `campello::sunVisibility()` |
| SSAO | `setAmbientOcclusion()` | off | varies (see the setter) | automatic (applied to the frame) |
| FXAA / SSAA | `setFxaaEnabled()` / `setSsaaScale()` | off | varies (see the setter) | automatic |
| Clip planes | `setClipPlanes()` | off | Metal, Vulkan | clip in your fragment shader (`CameraUniforms::clipPlanes`) |
| TAA | `setTemporalAntiAliasing(true)` | off | Metal, Vulkan | automatic (with your G-buffer stages if you move geometry) |
| Render scale + upscaling | `setRenderScale()`, `setUpscaler()` | 1, built-in | Metal, Vulkan (built-in); MetalFX on macOS 13 / iOS 16 | automatic |
| G-buffer | `renderGBuffer()` | not run | Metal, Vulkan | your own G-buffer stages (see 3) |
| Ray traced shadows | `RayTracedShadows` → `setShadowMask()` | null (shadow maps) | Metal + `Feature::raytracing` | `campello::sunVisibility()` |
| Ray traced AO | `RayTracedAmbientOcclusion` → `setAmbientOcclusionMask()` | null (SSAO) | same | automatic |
| Ray traced reflections | `PathTracer::traceReflections()` → `setReflections()` | null (environment) | same | `campello::reflectedRadiance()` |
| Probe GI (DDGI) | `DdgiVolume` → `setIndirectDiffuse()` | null (irradiance map) | same | `campello::indirectIrradiance()` |
| Many lights (ReSTIR DI) | `RestirDirectLighting` → `setDirectLighting()` | null (4-light loop) | same | `campello::directRadiance()` |
| One-bounce GI (ReSTIR GI) | `RestirGlobalIllumination` → `setIndirectDiffuse()` | null (irradiance map) | same | `campello::indirectIrradiance()` |
| Real-time denoiser | `RealtimeDenoiser` | not run | Metal, Vulkan | — |

The ray traced effects are classes you create and run. The renderer never runs them itself. Each one reads the G-buffer and writes a texture, and you hand that texture to the renderer. A frame with all of them looks like this:

```cpp
renderer.renderGBuffer(scene);
const auto& g = renderer.gBuffer();
shadows.trace(scene, g);
shadowDenoiser.denoise(shadows.output(), g, scene.camera.projection);
renderer.setShadowMask(shadowDenoiser.output());
ao.trace(scene, g);
aoDenoiser.denoise(ao.output(), g, scene.camera.projection);
renderer.setAmbientOcclusionMask(aoDenoiser.output());
tracer.traceReflections(scene, renderer, g);
reflectionDenoiser.denoise(tracer.reflections(), g, scene.camera.projection);
renderer.setReflections(reflectionDenoiser.output());
ddgi.update(tracer, scene, renderer);
ddgi.resolve(g, scene.camera);
renderer.setIndirectDiffuse(ddgi.irradiance());
restir.render(scene, g);
directDenoiser.denoise(restir.output(), g, scene.camera.projection);
renderer.setDirectLighting(directDenoiser.output());
// Indirect diffuse: DDGI (above) or ReSTIR GI, one of the two.
// restirGi.render(scene, renderer, g);
// giDenoiser.denoise(restirGi.output(), g, scene.camera.projection);  // demodulate off
// renderer.setIndirectDiffuse(giDenoiser.output());
renderer.render(scene, colorView);
```

Leave any line out and that effect falls back to its classic version. Shadows, AO and the path tracer's effects can share acceleration structures through one `RayTracingScene`.

### The same chain with presets

`RealtimeEffects` (`realtime_effects.hpp`) runs this chain for you. `EffectQuality` chooses the effects and the render scale:

| Preset | Effects | Render scale |
|---|---|---|
| `off` | raster only (shadow maps, SSAO if on, the environment) | 1 |
| `low` | ray traced sun shadows | 0.5 |
| `medium` | + ray traced AO | 0.58 |
| `high` | AO, reflections, ReSTIR DI (every light shadowed) | 0.67 |
| `ultra` | + ReSTIR GI | 0.75 |

```cpp
RealtimeEffects effects(renderer);
effects.setQuality(EffectQuality::high);   // or setSettings() effect by effect
effects.setFrameBudget(16.0);              // dynamic resolution: render scale down to 0.5 to fit
// each frame
effects.render(scene, colorView);
auto t = effects.timings();                // GPU ms per effect, the last finished frame
```

Without `Feature::raytracing` the presets fall back to raster only. MetalFX is the upscaler wherever it's supported. While `RealtimeEffects` is in use it owns the renderer's effect inputs (`setShadowMask()` and the rest). Each effect also reports its own `gpuTimeMs()`.

## 3. Custom shading

`registerProgram(desc)` loads your shaders (`program.hpp`) and returns a handle. Set `GpuMaterial::program` to that handle to draw the material with them. You can write one source in Slang for every backend, or one file per backend, and compile it with `campello_renderer_add_shader_program()` (CMake). Programs draw in the opaque and transparent passes, follow the material's culling and blending, and can read the built-in material and frame bindings.

### Lighting inputs (Metal)

`program_interface.metal.h` exposes what the renderer's effects give the built-in PBR shader. App programs use the same structs and helpers; the built-in shader includes this header too, so it's exactly what PBR uses.

```metal
#include "campello_renderer/shaders/program_interface.metal.h"

fragment float4 fragmentMain(VertexOut in [[stage_in]], CAMPELLO_LIGHTING_PARAMETERS) {
    float sun = campello::sunVisibility(campelloCamera, campelloShadowMap, campelloShadowMapNear,
                                        in.worldPos, N, L, in.position);
    float3 indirect = campello::hasIndirectDiffuse(campelloCamera, /*opaqueDraw=*/true)
        ? campello::indirectIrradiance(campelloCamera, campelloScreenLighting, in.position)
        : yourIrradianceLookup(N);
    ...
    return float4(color * campello::exposureScale(campelloCamera), 1.0);
}
```

- `sunVisibility()` uses the ray traced mask when `setShadowMask()` is set, otherwise the shadow-map cascades, and returns 1 when shadows are off.
- `hasDirectLighting()` / `directRadiance()` give ReSTIR's direct light (every light, every shadow) to use in place of your own light loop.
- The reflection helpers (`hasReflections()`, `reflectedRadiance()`, `reflectionWeight(roughness)`) and the indirect diffuse helpers (`hasIndirectDiffuse()`, `indirectIrradiance()`) return the effects' textures. These exist for opaque draws only, because the transparent pass binds the scene colour in the same slot. Pass `opaqueDraw = false` from blended programs.
- `CAMPELLO_ENVIRONMENT_TEXTURE`, `CAMPELLO_IRRADIANCE_TEXTURE` and `CAMPELLO_BRDF_LUT_TEXTURE` are the image-based lighting textures, for your own fallback.
- `campello::Light` and `LightsUniform` give the frame's lights, at most 4.
- Ambient occlusion is applied to the finished frame, so custom programs get it with no code.

The helpers hide how the renderer packs its effects, and that packing may change between versions. Recompile your programs against the header of the version you link.

### Your own G-buffer stages (Metal)

`renderGBuffer()` produces the data that TAA, the upscalers, the denoisers and every ray traced effect read. By default it draws every opaque mesh with the built-in G-buffer stages, as plain PBR geometry. If your vertex shader moves geometry (wind, displacement, your own skinning) or your fragment shader computes its own normals, give the program its own G-buffer stages:

```cpp
desc.metalGBufferVertexEntryPoint = "gbufferVertexMain";
desc.metalGBufferFragmentEntryPoint = "gbufferFragmentMain";
```

- **Vertex stage:** reads the same attributes as your vertex stage, plus `campello::GBufferTransforms` at `[[buffer(CAMPELLO_PREVIOUS_TRANSFORMS_BUFFER)]]` (the previous frame's clip transform and this frame's jitter). Output the current clip position, `campello::unjitteredClip()` of it, and the previous clip position.
- **Fragment stage:** return `campello::gbufferOutput(normal, roughness, albedo, metallic, currentClip, previousClip)`.

`test/shaders/custom_lit.metal` is a complete example covering both the lighting inputs and the G-buffer stages.

### What doesn't carry over yet

- **Ray traced passes shade with the built-in material model.** The path tracer, reflections and probe GI see glTF PBR materials, so a custom material looks like its `GpuMaterial` PBR parameters there.
- **The frame order is fixed:** shadows, opaque, transmission copy, transparent, post-processing, overlay. You can't insert a pass between built-in passes. The overlay callback runs at the end, and the building blocks above run outside `render()`.
- **Backends:** the lighting inputs and app G-buffer stages are Metal only, like the ray traced effects. Vulkan has the built-in G-buffer, TAA, the built-in upscaler and the real-time denoiser, but not app G-buffer stages yet. The Slang, GLSL, HLSL and WGSL headers carry the vertex and transform contract only.
