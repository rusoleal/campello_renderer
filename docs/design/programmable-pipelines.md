# Programmable pipelines: app-defined vertex layouts and shaders

Status: **in progress** (2026-11-09). Phases 1 (vertex layouts, pipeline cache), 3 (app shader programs), 4 (clip planes, section caps) and 5 (Slang spike: go for app programs, PBR planned) and most of phase 2 (compact slots, binding contract) are done; the rest is proposed.

## Problem

The renderer has one hardcoded PBR pipeline per backend:

- **Fixed vertex slots.** POSITION 0, NORMAL 1, TEXCOORD_0 2, TANGENT 3, JOINTS 4, WEIGHTS 5, COLOR_0 6, TEXCOORD_1 7, plus per-draw data at 15 (lights), 16 (NodeTransforms), 17 (material), 18 (camera), 19 (instance matrices), 20 (joints) and 21–23 (morph targets). `GpuMesh::vertexBuffers` is a fixed `std::array` of six.
- **Embedded shaders.** `default.metal`, `default.vert/.frag` and `default.hlsl` are compiled offline by per-backend scripts into headers (`metal_default.h`, `vulkan_default.h`, ...).
- **No WebGPU path.** campello_gpu has a WebGPU backend (WGSL via `createShaderModule(const char*)`), but under Emscripten the renderer's `createDefaultPipelines()` falls through to an empty `#else`. campello_bim will need a web version, so WebGPU is a fourth backend to design for, not an afterthought.
- **Hand-made variants.** Each variant is a named member created in `createDefaultPipelines()`, once per backend: flat / textured / debug × single- or double-sided × opaque or blend, plus the lean position-only variants added for BIM (`pipelineLean*`, `supportsPositionOnlyMeshes()`).

Every new need multiplies the variants across three backends. The lean meshes show it: a second vertex layout meant a second vertex shader per backend, four more pipeline members, a capability flag, and a branch in `renderPrimitive`. Apps can't add their own rendering at all, and BIM needs several kinds soon:

| BIM need | Kind |
|---|---|
| Section planes (clip the model) | small change to the PBR shader |
| Selection / hover tint, x-ray / ghost mode | small change to the PBR shader |
| Position-only meshes (done, hardcoded) | different vertex layout |
| Element edges, 2D drawing views (lines) | custom program, line topology |
| GPU picking (element id buffer) | custom program, integer target |
| Selection outline | custom program + post pass |

## Goals

1. A mesh describes its own vertex layout. Pipelines are built on demand from that layout and cached, not declared as members.
2. Apps can register their own shader programs, with a documented contract for the data the renderer provides (camera, transforms, lights, environment).
3. The built-in PBR becomes the default program, registered through the same public API the apps use.
4. Common app needs (clip planes, tint, id output) are built-in **features** of the PBR program, switched by uniforms, so most apps never write a shader.
5. Four backends: Metal, Vulkan, DirectX and **WebGPU**. Every contract in this document has to fit WebGPU's limits (below), which are the tightest.
6. The existing API keeps working. glTF loading, `render_system`, materials and debug views don't change for current users.

## Non-goals

- A runtime shader compiler in the library. Shaders stay compiled offline (Vulkan and DirectX need binaries anyway).
- A render graph or user-defined passes. The pass list stays the renderer's: skybox, opaque, scene-color copy (transmission), transparent, FXAA. Programs take part in passes; they don't define them. A render graph can come later on top of this.
- Shader source injection (gluing app snippets into the PBR source). See "Hooks" below.

## WebGPU constraints

WebGPU's default limits are lower than what the renderer uses today, and a web build can't assume more:

| Limit (default) | Value | Renderer today | Consequence |
|---|---|---|---|
| `maxVertexBuffers` | 8 | slots 0–7 attributes, 16 transforms, 17–19 placeholders/instances | Vertex buffer slots must be **compact**: a layout's attributes take slots 0..n-1, then the per-draw transforms |
| `maxVertexAttributes` | 16 | 8 attributes + 8 matrix columns (MVP, model) | Fits once the shader reads 6 attributes (no skinning): 6 + 8 = 14 |
| `maxSampledTexturesPerShaderStage` | 16 | PBR fragment: 21 textures (glTF core 5, KHR extensions 12, IBL 3, scene color 1) | PBR is split into **variants by feature set**: core (5 + IBL 3 + scene color = 9) and extensions grouped so each variant stays ≤ 16 |
| `maxSamplersPerShaderStage` | 16 | 8 | fine |
| `maxBindGroups` | 4 | 2 (material, frame) | room for a per-draw group |
| Derivatives (`dpdx`) | uniform control flow only | lean meshes take them unconditionally | already compatible |
| Shading language | WGSL only | MSL, GLSL, HLSL | a fourth hand-written copy, or generated (Slang emits WGSL) |

These shape the design below: compact vertex slots on **all** backends (one contract, not a WebGPU special case), and PBR variants keyed by material features. Per-draw transforms can stay per-instance vertex attributes: a buffer's slot and a shader input's location are independent, so they keep locations 16–19/24–27 and only move to the slot after the attributes (7 buffers, 14 attributes for the standard shader).

## Design

### 1. Vertex layouts described by data

```cpp
enum class VertexSemantic : uint8_t { position, normal, tangent, texcoord0, texcoord1,
                                      color0, joints, weights, custom0, custom1, custom2, custom3 };

struct VertexAttributeDesc {
    VertexSemantic semantic;
    GPU::ComponentType component;   // ctFloat, ctUnsignedShort, ...
    GPU::AccessorType  type;        // acVec2, acVec3, ...
    uint32_t location;              // shader input location / [[attribute]]
};

struct VertexLayoutDesc {
    std::vector<VertexAttributeDesc> attributes;   // one buffer per attribute (non-interleaved, as today)
    uint64_t hash() const;
};
```

- `GpuMesh` gains `const VertexLayoutDesc* layout` (or a small interned layout id) and `std::vector<Buffer>` replaces the fixed array. `uploadMesh()` fills both from the glTF accessors, so glTF users see no change.
- The standard PBR layout is a predefined constant (`VertexLayouts::pbr()`), and so is `VertexLayouts::positionOnly()`. The `positionOnly` flag goes away: a lean mesh is just a mesh with that layout.
- **Compact slots:** a layout's attribute *i* is read from vertex buffer slot *i* (non-interleaved, as today), and the shader location comes from the semantic. A position-only mesh binds one buffer, a PBR mesh at most eight (WebGPU's limit).
- **Per-draw data.** Transforms stay per-instance vertex attributes on Vulkan, DirectX and (later) WebGPU, in the slot after the attributes, and a `[[buffer(16)]]` argument on Metal. Instance matrices (19), joints (20) and morph data (21–23) are Metal-only features today (raw `[[buffer(n)]]` arguments); when they come to the other backends they become storage buffers (WebGPU has no room for more vertex buffers).
- **Slot maps (implemented).** Each cached pipeline carries a map from logical slot (`VERTEX_SLOT_*`) to physical slot, or "not read". Draw code binds by logical slot and the map places or skips the buffer, so the same draw code serves every backend and layout.

### 2. Shader programs

```cpp
struct ShaderStageSource {           // one compiled stage for one backend
    const uint8_t* code; uint64_t size;
    std::string entryPoint;
};

struct ShaderProgramDesc {
    std::string name;
    // Compiled binaries per backend. A backend with no binary can't draw
    // the program; drawRenderables skips it and reports it once.
    std::optional<ShaderStageSource> metalLibrary;        // one .metallib, both entry points
    std::optional<ShaderStageSource> vulkanVertex, vulkanFragment;
    std::optional<ShaderStageSource> directxVertex, directxFragment;
    std::optional<std::string> wgsl;                       // WGSL source, both entry points

    std::vector<VertexLayoutDesc> acceptedLayouts;        // empty = any layout with a position
    FrameInputs   frameInputs;    // which renderer data it reads: camera, lights, environment, sceneColor
    DrawInputs    drawInputs;     // transforms, instances, joints, morph
    MaterialLayout materialLayout; // its own bind group: uniform block size + texture/sampler slots
    PassMask      passes;         // opaque | transparent (later: depthOnly, id)
    ColorTargets  targets;        // default: the scene color format; id programs: r32uint
};

ProgramHandle Renderer::registerProgram(const ShaderProgramDesc&);
ProgramHandle Renderer::pbrProgram() const;     // the built-in one
```

- `GpuMaterial` gains `ProgramHandle program` (defaults to the PBR program) and a program-owned bind group. The current PBR material fields become the PBR program's material data.
- Programs don't pick their render state. Cull mode and blending come from the material (`doubleSided`, `alphaBlend`), as today.

### 3. Pipeline cache

```
key = (program, layout hash, cullMode, blendMode, colorFormat, depthFormat, sampleCount, pass)
```

- Pipelines are created on first use and kept in a hash map. That replaces the 13 `pipeline*` members for meshes; skybox, FXAA, downsample and IBL bake stay as fixed internal pipelines.
- `prewarm(program, layout, ...)` lets apps create pipelines up front and avoid hitches on the first frame (Metal and D3D12 pipeline creation takes milliseconds).
- `supportsPositionOnlyMeshes()` generalises to `bool supports(ProgramHandle, const VertexLayoutDesc&)`.

### 4. Binding contract

This is the core of the proposal: what a custom shader may rely on. It's written down once per backend and covered by tests that compile a minimal custom program on every backend.

| Data | Metal | Vulkan | DirectX | WebGPU |
|---|---|---|---|---|
| Vertex attributes | `[[attribute(semantic)]]`, buffers 0..n-1 | `location = semantic`, bindings 0..n-1 | `TEXCOORD<semantic>`, input slots 0..n-1 | `@location(semantic)`, buffers 0..n-1 (n ≤ 7) |
| NodeTransforms {mvp, model} (128 B per draw) | `[[buffer(16)]]` | per-instance attributes, locations 16–19 (MVP), 24–27 (model), binding n | same as Vulkan, `TEXCOORD16–19/24–27`, slot n | same as Vulkan, buffer n |
| Instance matrices | `[[buffer(19)]]` | not supported yet | not supported yet | storage buffer, group 2 |
| Camera | buffer 18 (fragment) | set 1 | combined group (to split) | group 1 |
| Lights | buffer 15 | set 1 | combined group (to split) | group 1 |
| Environment (IBL, BRDF LUT) | fragment textures | set 1 | combined group (to split) | group 1 |
| Program material | buffer 17 + its textures | set 0 | (to define) | group 0 (≤ 16 textures with the frame's) |

Work implied by the table:

- **DirectX** uses one combined material + frame bind group per frame-in-flight (`GpuMaterial::directxBindGroup`). It must be split into frame and material groups, as on Vulkan, before custom programs can work there.
- **Vulkan** carries the transforms as per-instance vertex attributes and Metal as a device buffer. Both are fine for the contract; the table just has to say so.

The same structs (`NodeTransforms`, `CameraUniforms`, `LightsUniforms`) ship as headers that shaders include, one per language: `campello_renderer.metal.h`, `campello_renderer.glsl`, `campello_renderer.hlsli`, `campello_renderer.wgsl` (WGSL has no `#include`: the renderer prepends it to WGSL sources, which are compiled at runtime by the browser). Later this becomes one Slang module (see §6).

### 5. Hooks: built-in features, not source injection

Gluing app code into the PBR shader needs either a runtime compiler (not available for SPIR-V/DXIL without shipping one) or every combination compiled offline. Instead, the PBR program gets a few **features** switched by uniforms, covering the common needs:

- **Clip planes:** up to 6 world-space planes in the frame data; fragments behind any enabled plane are discarded. Optionally the cut is capped (back faces drawn in a cap colour). These are BIM section planes.
- **Tint / override:** a per-draw colour, mode (multiply, replace) and opacity. This gives selection highlight, ghost/x-ray, and colour-by-property.
- **Flat-normal fallback:** what the lean meshes do today, kept as a feature of the PBR fragment shader.
- **Debug views:** stay a PBR feature.

Per-draw feature data rides on `DrawCall` (for example `DrawCall::tint`) and goes in the transforms buffer or a small per-draw uniform. Anything beyond the features is a custom program.

### 6. Shader sources across three backends

With WebGPU, every program is written **four** times (MSL, GLSL, HLSL, WGSL); that is the main cost of this proposal, for the library's own PBR shader as much as for apps. The options:

| Option | Toolchain | Notes |
|---|---|---|
| A. Hand-written per backend (today) | xcrun metal, glslc / glslangValidator, dxc; WGSL needs no compiler | No new tool; four copies to keep in step |
| B. **Slang** → MSL, SPIR-V, DXIL, WGSL | `slangc` (one binary on macOS/Windows/Linux) | One source for four targets; modules fit the binding-contract headers; MSL and WGSL outputs are newer than the others |
| C. HLSL → DXIL and SPIR-V (dxc) → MSL and WGSL (SPIRV-Cross, Tint/Naga) | dxc + spirv-cross + tint | Mature parts; three tools |

Recommendation: A for phases 1–2 (the PBR shaders exist in three languages already). Before writing the WGSL PBR shader (phase 7), spike B: if Slang's output matches today's images (the existing image tests), move the built-in shaders to Slang first and get WGSL from it, instead of hand-writing a fourth 1,000-line copy. CMake gets a helper either way:

```cmake
campello_renderer_add_shaders(my_app_shaders
    PROGRAM bim_ids SOURCES shaders/bim_ids.slang   # or per-backend files
    OUTPUT  ${CMAKE_CURRENT_BINARY_DIR}/bim_ids_shaders.h)
```

The helper embeds the binaries in a header, as the library does for its own shaders today.

### 7. Taking part in passes

| Pass | Today | Custom programs |
|---|---|---|
| Skybox | internal | — |
| Opaque | PBR | yes, if `passes` has `opaque` |
| Scene-color copy (transmission) | internal | — |
| Transparent (sorted back to front) | PBR | yes, if `passes` has `transparent` |
| FXAA | internal | — |
| Depth-only (future: prepass, shadows) | — | needs a position-only version, which any position-carrying layout already allows |
| Id (future: GPU picking) | — | separate target, own program |

- Programs with a non-default color target (an id program) are drawn in their own render call (`renderIds(target, ...)`), not mixed into the lit frame.
- Culling (frustum, size), batching and transparency sorting stay program-agnostic. They only use bounds and the material's blend flag.

## Migration plan

Each phase ships on its own, keeps the existing tests and image comparisons passing, and changes no existing public signature unless noted.

1. **Layouts and the pipeline cache.** *Done.*
   - Implemented as `VertexLayout` (`vertex_layout.hpp`, attributes as semantic + format) and `GpuMesh::layout`. Each backend's recipe lists vertex stages with the semantics they read; a layout is drawn with the stage that reads the most attributes it has, so formats can vary (the vertex fetch converts them). Slots still follow the semantic (today's convention); compact slots move to phase 2.
   - Add `VertexLayoutDesc` and the cache, with the PBR program still internal.
   - Replace the `pipeline*` mesh members, the lean variants and `positionOnly` with layouts.
   - The capability check becomes `supports(...)`.
   - Absorbs the lean work. Medium-sized: most of it is `createDefaultPipelines()` and `renderPrimitive()`.
2. **Binding contract.** *Mostly done.*
   - Done: compact vertex slots on every backend with per-pipeline slot maps; transforms in the slot after the attributes (WebGPU-sized); `quantizedPipelines` (KHR_mesh_quantization, Metal) folded into layouts (`quantizedLayout`, the standard layout with integer formats); the table above.
   - Moved to phase 3: the per-language interface headers (they are only useful once a custom program includes them).
   - Open: split the DirectX combined bind group. It changes `default.hlsl`'s registers, so it needs `dxc` (not available on macOS) and a Windows run.
   - Test: a minimal "unlit colour" program written against the headers draws on Metal and Vulkan (DirectX when dxc is available in CI).
3. **Public program API.** *Done.*
   - `registerProgram()` (`program.hpp`), `GpuMaterial::program`, `renderProgramPass()`; the PBR program is handle 0 in the same table (its pipelines come from the same cache). Programs share the built-in pipeline layout, so they may read the material/frame resources at the built-in bindings.
   - Interface headers per language (`inc/campello_renderer/shaders/program_interface.*`) and `campello_renderer_add_shader_program()` (CMake) to compile and embed app programs.
   - `VertexSemantic::custom0`-`3` for app vertex data; first program: element ids into `r32uint` (renderer test). campello_gpu needed a fix for integer/normalized vertex formats on Vulkan and DirectX (released in 0.28.0).
   - Not done: BIM GPU picking (its CPU BVH picking already returns the hit point that snapping needs; ids would add 4 bytes per vertex). The built-in shaders don't include the interface headers yet (they would need regenerated binaries on every backend).
4. **PBR features.** *Clip planes done; tint deferred.*
   - Done: `setClipPlanes()` (6 world planes in the camera block, discard in the PBR fragment shader on Metal and Vulkan), section caps (back faces of single-sided materials in a cap colour, culling off while clipping), CPU culling of draws entirely clipped (`isClippedAway()` in `render_system`). BIM uses it for horizontal sections. DirectX: needs `default.hlsl` to read the clip block (dxc).
   - Deferred: per-draw tint/override. Materials already cover highlight and ghosting (BIM swaps materials), and colour-by-property on merged batches needs per-vertex colour (`COLOR_0`, which `GpuMesh` now carries) rather than a per-draw value; revisit with a concrete need.
5. **Single-source shaders.** *Spike done (Slang 2026.20): go for app programs; PBR move planned.* See "Slang spike findings" below.
   - Done: one transform contract on every backend (Metal also gets the transforms as per-instance attributes); `program_interface.slang`; `campello_renderer_add_shader_program(SLANG ...)` compiles one source to Metal, SPIR-V (`-fvk-invert-y`), HLSL (then dxc) and WGSL; the element-id test program is Slang and passes on Metal, with valid SPIR-V.
   - Next (PBR): one binding layout for all backends (material group + frame group, the DirectX split from phase 2), port the Metal PBR shader (the most complete) to Slang, Metal image tests pixel-identical, then retire the three hand-written copies; split PBR by feature set for WebGPU's 16-texture limit.
6. **Lines and topology.** *Done (1-pixel lines).*
   - `GpuMesh::topology` in the pipeline key; a line vertex stage in the built-in program (Metal, Vulkan) with a clip-space depth offset so edges win over their faces; unlit materials give flat colours. campello_bim draws feature edges with it.
   - Later, for drawing views: wide lines (screen-space quads expanded in the vertex shader), dashes, line weights by category.
7. **WebGPU backend.** *Mesh rendering done (hand-written WGSL); the test suite passes in headless Chrome (86/88, the other two are Metal-only features).*
   - Done: the web build shares Vulkan's bind-group code (`CAMPELLO_RENDERER_VK_BINDINGS`); `shaders/webgpu/default.wgsl` (core PBR feature set within 16 textures; material groups trimmed to bindings 0-9 + 24); transforms as a storage buffer (group 2) indexed by `firstInstance`, because **WebGPU vertex inputs stop at location 15** (the 16-19/24-27 attribute contract can't exist there); real alpha blending; app programs via `ShaderProgramDesc::wgsl`. `test/webgpu/run_wgsl_check.sh` checks the WGSL against the exact layouts and renders lean, line, glTF-layout, blended, clipped and capped draws in headless Chrome (Intel GPU via Chrome's Metal backend).
   - Built with Emscripten 4.0.10 and run in the browser (`test/webgpu/run_web_tests.sh`). Getting there fixed: vector_math (AVX2 flags under Emscripten), gltf (draco's JS glue and environment check), campello_gpu (WebGPU buffers whose size isn't a multiple of 4 couldn't be written: every uint16 glTF index buffer).
   - Next: WGSL versions of skybox, FXAA, downsample and the IBL bakes; feature variants for the KHR extension textures; then campello_bim's web version (campello_widgets on the web).

## Slang spike findings (2026-11-09, slangc 2026.20)

**Vertex input locations.** Each target follows a different annotation: Metal and SPIR-V take `[[vk::location(n)]]`, HLSL and WGSL take the semantic. With both (`[[vk::location(n)]] float3 p : TEXCOORD<n>`) all four keep location *n*, and `TEXCOORD<n>` is DirectX's existing contract. Named semantics (`: POSITION`) make WGSL renumber inputs. `program_interface.slang` wraps it as `CAMPELLO_INPUT(type, name, n)` (n literal: it is pasted into the semantic; a macro parameter named `location` would also replace the `vk::location` attribute name).

**Clip-space Y.** `-fvk-invert-y` negates `SV_Position.y` for SPIR-V, so sources contain no per-target code. Entry points: Metal and HLSL keep their names; SPIR-V renames to `main` (the CMake helper and `ShaderProgramDesc` defaults handle both).

**Per-draw transforms.** One source needs one contract: Metal now also gets the transforms as per-instance attributes (16-19, 24-27), as Vulkan/DirectX do (the built-in Metal shaders keep `[[buffer(16)]]`).

**Element-id program.** One `.slang` file; Metal output passes the r32uint id test; SPIR-V passes `spirv-val`; WGSL has the right `@location`s (not drawable until phase 7); HLSL has the right `TEXCOORD<n>` semantics (DXIL not built here: no dxc on macOS).

**PBR.** The existing `default.hlsl` compiles through Slang to Metal (60 KB), SPIR-V (valid) and WGSL with warnings only (implicit conversions, D3D registers without Vulkan bindings). Two things block using that output directly:
- *Bindings.* Slang maps D3D registers to the same Metal indices (`t0` -> `texture(0)`, `b24` -> `buffer(24)`) and needs explicit `[[vk::binding]]` for Vulkan, but the three backends' bind group layouts differ today (Metal shared layout, Vulkan material/frame sets, DirectX combined group). Moving PBR needs one layout, declared once in the Slang source, that every backend's bind group layouts follow.
- *Feature parity.* The three hand-written PBR copies have drifted (lean meshes and clip planes are Metal/Vulkan only; the debug and flat variants are Metal only). The port should start from the Metal shader and be judged by pixel-identical Metal images.

**WebGPU.** The PBR fragment shader in WGSL declares 21 textures (limit 16), confirming the feature-set split.

**Tooling.** slangc is one 30 MB download per platform (macOS, Windows, Linux), no install. The built-in shaders would keep committed generated headers (as `metal_default.h` / `vulkan_default.h` today), so building the renderer never needs slangc, only editing shaders does. DXIL still needs dxc (or slangc with dxcompiler on Windows).

**PBR port (branch `wip/slang-pbr`, 2026-11-09).** `shaders/slang/pbr.slang` (1,312 lines, ported from the Metal shader; dual Metal/Vulkan bindings, so no C++ binding changes besides sampler 15) generates Metal that renders the campello_bim office model **pixel-identical** to the hand-written MSL and passes all tests under Metal API validation; its SPIR-V validates and matches the Vulkan set/binding layout. But on an Intel UHD 630 the generated fragment shader is **~10x slower** (office model zoomed in: GPU 10 ms -> 105 ms). Ruled out: branch flattening (branches and samples preserved), Slang's resource `KernelContext` (removing it by hand changed nothing), slangc `-O3` / `-fp-mode fast`, the vertex stage (fast with a trivial fragment shader). The cost is spread through the whole shader (an early return right after the clip test already costs 18 ms against 5 ms for a trivial shader), which points at register pressure / occupancy in Apple's compiler for this code shape. Next: an Xcode GPU frame capture (register count, occupancy) of both versions, and a run on Apple silicon (different compiler) before deciding; until then the built-in Metal shaders stay hand-written. Side finding: the hand-written Metal shader reads morph-target deltas with a 16-byte stride while the CPU packs them at 12 bytes (morph targets broken on Metal); the Slang version reads them correctly.

**Recommendation.** Go: new shader work (app programs, the WebGPU path) is written in Slang from now on. The PBR move is its own step (estimate: several days), after the binding-layout unification, and it is what removes the remaining DirectX and WGSL gaps (clip planes, lean meshes, a fourth hand-written copy).

## Risks and open questions

- **Public API surface.** The binding contract becomes something apps depend on. Keep it small and versioned (`kBindingContractVersion`), and change it only with a major renderer version.
- **Pipeline creation hitches.** On-demand creation costs a few ms the first time a combination is drawn; `prewarm` plus an async creation path (if campello_gpu gets one) handles it.
- **WebGPU texture limit.** Splitting PBR by feature set means a material using many KHR extensions may need a variant per combination; group extensions so the common ones (clearcoat, transmission, sheen) fit, and fall back to dropping textures (keeping factors) beyond the limit, with a one-time warning.
- **DirectX coverage.** CI is Windows-only but can't run the GPU tests today, and the DXIL headers aren't regenerated from macOS. Phase 2 needs dxc in CI to be credible.
- **Skinning and morph targets** stay PBR-only at first; custom programs declare `drawInputs` without them until someone needs it.
- **Where ECS integration lives.** `render_system` keeps building `DrawCall`s from components. A material component that names a program is enough; no new ECS concepts.
- **Open: per-draw data for features.** Grow `NodeTransforms` (128 → 160 bytes for tint + flags) or add a separate per-draw buffer. Measure on the BIM office model (≈10k draws) before choosing.
