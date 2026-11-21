# Changelog

## [0.31.1] - 2026-11-21

### Fixed
- **Supersampling (`setSsaaScale()` > 1) showed only part of the scene, and WebGPU dropped the frame.** The SSAA downsample and FXAA passes draw into the output but took the supersampled scene's viewport and scissor (`setupViewport()`): twice the output's size at 2x. WebGPU rejected the scissor ("not contained in the render area") and dropped every supersampled frame; on D3D12 the scene was drawn at the wrong scale (the test's quad, on the right, was missing). They now use the output's size (`setupOutputViewport()`). WebGPU also rejected the opaque pass, which bound the scene colour texture (its attachment when supersampling) as a texture: it now binds the placeholder there, as DirectX does (`kSceneColorPlaceholder`). New test `SupersampledFrameShowsTheWholeScene` (fails without the fix on D3D12).

## [0.31.0] - 2026-11-21

### Changed
- **8 punctual lights** (was 4): `LightsUniform` holds `kMaxLights` = 8 lights (528 bytes), strongest first. The DirectX and WebGPU shaders read all 8; the Metal and Vulkan shaders still read the first 4 (the strongest) until they're updated. A larger uniform buffer than a shader declares is valid on every API.

### Fixed
- **DirectX: the thickness texture is read from its G channel**, as KHR_materials_volume says (it read R). Same as WebGPU since v0.30.0; Metal and Vulkan still read R.

## [0.30.0] - 2026-11-20

### Added
- **Volume, iridescence and the transmission texture on WebGPU.** The web build drew transmissive materials thin-walled (the scene straight behind, unbent) and ignored KHR_materials_volume, KHR_materials_iridescence and the transmission texture, so glass didn't refract (IridescentDishWithOlives) and a transmission texture's opaque parts were see-through (LightsPunctualLamp's shade). `shaders/webgpu/default.wgsl` now ports the Vulkan shader's code:
  - KHR_materials_volume: the scene is sampled where the refracted ray leaves the volume (per channel with KHR_materials_dispersion), attenuated by Beer's law; off screen, the environment along the refracted ray.
  - KHR_materials_iridescence: thin-film interference on the IBL and punctual-light F0.
  - The transmission, thickness, iridescence and iridescence-thickness textures (material bindings 19-22, sampled with the base colour sampler, as on Vulkan): 15 of WebGPU's 16 sampled textures per stage. `trimWebGpuMaterialEntries()` keeps them.
  - The thickness texture is read from its G channel, as the glTF spec says (the native shaders read R; see TODO.md).

### Changed
- **More lights than the shaders take: the strongest are kept.** The shaders take 4 punctual lights; the renderer used the first 4 it found in the scene graph. It now keeps directional lights first, then point and spot lights by intensity times their brightest channel (LightsPunctualLamp has 5: its dimmest, 1.5 cd, is the one left out). Every backend.
- **WebAssembly builds use SIMD** (`-msimd128`), on by default: every browser with WebGPU supports it. The option `CAMPELLO_RENDERER_WASM_SIMD` turns it off. It applies to the library and is passed on to consumers (compile and link, so wasm-opt accepts the SIMD instructions).
- campello_gpu v0.34.0, campello_image v0.7.0, campello_procedural v0.4.0, gltf v0.7.0, vector_math v0.7.0 (WebAssembly SIMD).

## [0.29.0] - 2026-11-20

### Added
- **Skinning and morph targets on WebGPU.** The web build drew skinned meshes in their rest pose and ignored morph weights: `shaders/webgpu/default.wgsl` had no deforming vertex stage. New entry point `vs_skinned` (a port of `skinning.glsl`: morph targets, then up to four joints per vertex; weights summing to zero leave the vertex unskinned), which `meshPipeline()` picks only for draws with a skin or morph targets, as on Vulkan and DirectX (`CAMPELLO_RENDERER_SPLIT_DEFORM` now includes WebGPU).
  - WebGPU allows four bind groups and group 2 already holds the per-draw transforms, so the joint palette and the morph targets share **group 3** (binding 0: palette, 1: MorphInfo, 2/3: POSITION/NORMAL deltas), instead of Vulkan's sets 2 and 3. Only `vs_skinned`'s pipelines have group 3 in their layout (`webgpuSkinnedPipelineLayout`), since WebGPU requires every group of a pipeline's layout to be bound. `bindSkin()` keeps the palette for the `bindMorph()` call that follows; that one builds and caches the combined group.
  - The joint palettes, morph buffers and the per-vertex fallback buffers are created with storage usage on WebGPU too (`kDeformBufferUsage`): the fallback tangent buffer is the "no morph deltas" binding, and WebGPU rejected it as vertex-only.
  - Not yet on WebGPU: skinned and instanced shadow casters (still drawn at rest), and the G-buffer's previous palette (no G-buffer on WebGPU).
- **Instancing on WebGPU** (`EXT_mesh_gpu_instancing` and `DrawCall::instanceTransformBuffer`). The web build drew an instanced mesh once, at its own transform: WebGPU draws find their transform through `firstInstance`, which instancing needs for the instance index, and the WGSL stages never applied instance matrices. New entry points `vs_instanced` and `vs_lean_instanced` (position-only meshes), which `meshPipeline()` picks for draws with instance matrices (new `instanced` argument; ignored on the other backends, whose stages always apply instances). Their pipeline layout's group 2 holds the draw's own transform as a uniform: `bindWebGpuInstanced()` copies it from the CPU staging to a 256-byte slot of `webgpuInstanceTransformBuffer` (slots reset per `render()`), and the draw starts at instance 0. The instance matrices stay a per-instance vertex buffer (locations 8-11), so `instanceTransformBuffer` still only needs vertex usage.
  - Limits on WebGPU: a draw both skinned and instanced keeps its skin and loses the instance placement (the skinned stage already uses WebGPU's 8 vertex buffers); line meshes don't take instances.
- `test/webgpu/run_wgsl_check.sh`: 18 checks, including the skinned and instanced pipelines and renders (a joint, a morph target and a zero weight; two instances at their matrices).
- Web test suite (`test/webgpu/run_web_tests.sh`, Edge on an Intel UHD 630, with v0.28.0 merged): **100 passed, 0 failed** (v0.27.0: 6 failures). Fixed: `GltfSkinPosesTheMesh`, `DrawCallJointsPoseTheMesh`, `MorphTargetMovesEveryVertex`, `GltfInstancesDrawAtTheirTransforms`, `DrawCallInstancesDrawAtTheirTransforms` and `FramesDrawAfterTheEnvironmentBake` (it draws the instanced quad).

### Fixed
- **A model loaded after another could stay at the old one's transforms on WebGPU.** `bindWebGpuTransforms()` cached each transform buffer's bind group by the buffer's address and kept the cache past `setScene()`. A WebGPU `BindGroup` doesn't keep its buffer alive, so a new transform buffer could get a freed one's address and be drawn with the old bind group. That group's buffer no longer got uploads, so the geometry froze while the camera and background moved. Each cache entry now holds its buffer and is matched on it, and `setScene()` clears the cache.
  - The same function cleared the map past 8 entries while holding a reference into it, then assigned through that reference. It no longer does.
  - Found in campello_web_viewer: Cameras.gltf, loaded after CesiumMan and MaterialsVariantsShoe, didn't move with the camera. With the fix it does.
  - New web test `NewAssetDrawsAtItsOwnTransforms` (five loads alternating a quad's position, three frames each). It can pass without the fix too, since the address reuse depends on the allocator.

## [0.28.1] - 2026-11-19

### Changed
- **campello_gpu v0.33.2**, whose WebGPU backend copies compressed mips smaller than a block (2×2, 1×1) as whole blocks. Compressed textures have their whole mip chain again (v0.28.0 stopped them at 4×4 to avoid that bug).

## [0.28.0] - 2026-11-19

### Added
- **`canCompressTextures(srgb)`**: whether this device keeps KTX2 (Basis) textures of a kind compressed on the GPU: colour data, or linear data such as normal maps. An app encoding its own KTX2 textures (campello_bim, a frame at a time) can skip the linear ones where they would become RGBA8.

### Fixed
- **KTX2 normal maps on Apple and most mobile GPUs** (and other linear KTX2 data): they were transcoded to ASTC, which campello_gpu has only as sRGB, so the GPU decoded them as sRGB and skewed them. Linear KTX2 data now goes to BC7 where the GPU has it, else RGBA8 (`basisTargetFormatFor()`, used by `setScene()` and `uploadMaterial()`).
- **KTX2 base colour textures had no average colour**, so `uploadMaterial()` didn't record them as `GpuMaterial::baseColorTexture`, and the path tracer shaded them with the factor alone. The average now comes from the texture's smallest mip, transcoded to RGBA8.
- **Compressed textures on WebGPU** (KTX2, and `setTextureCompression()`'s): their uploads failed validation, so they never reached the GPU. A mip's rows went to the GPU packed, and WebGPU copies rows from a buffer only at a 256-byte pitch, which compressed mips under 64 texels across don't have; and mips under a 4x4 block were copied with their own size, where WebGPU takes whole blocks. `uploadTextureDataWithMips()` now pads rows to 256 bytes and mips to 512 (which D3D12 asks too), and compressed textures stop at their last whole-block mip (`uploadedMipCount()`; a 512 one has 8 mips, not 10).
- New test `Ktx2TexturesTranscodeForTheirData`: a KTX2 colour map is uploaded compressed sRGB and recorded; a KTX2 normal map gets a linear format, compressed where `canCompressTextures(false)` says so. Passes on macOS (BC7) and on an Adreno 642L (ASTC colour, RGBA8 normal).

## [0.27.0] - 2026-11-19

### Added
- **Compressed textures** (`setTextureCompression()`, off by default). `uploadMaterial()`'s 8-bit images and baked procedural textures are encoded once (KTX2 UASTC, campello_image v0.6.0) and uploaded in the device's compressed format, with their mip chain: BC7 on desktop GPUs, ASTC 4x4 on Apple and most mobile GPUs, ETC2 on older ones. That's about a quarter of the GPU memory of RGBA8.
  - **Cost:** encoding takes about 0.1 s per 512x512 texture on a desktop CPU at the fastest setting (more on one thread, as on the web), once per texture.
  - **Where it stays RGBA8:** a device without a compressed format; linear data such as normal maps where the only format is sRGB (campello_gpu's ASTC); alpha on ETC2 RGB; sizes that aren't whole 4x4 blocks.
  - **Not yet:** `setScene()`'s images (the scene asset's own glTF textures).
  - New test `CompressedTexturesDrawAsTheirPixels`: the baked checkerboard is uploaded compressed (BC7 sRGB here) with its mips and draws its squares; with compression off it stays RGBA8.
- **campello_image v0.6.0** (KTX2 encoding).
- **campello_gpu v0.33.1**: compressed formats on Intel Macs (ASTC was reported, and its textures failed Metal validation) and compressed uploads on Vulkan (rows were skewed). Compressed textures need both.
- `run_android_tests.sh`: `CAMPELLO_IMAGE_SOURCE` builds against a local campello_image, as `CAMPELLO_GPU_SOURCE` does for campello_gpu.

### Changed
- **Procedural textures (KHR_texture_procedurals) are baked by campello_procedural v0.1.0**, a new library: MaterialX 1.39 standard library nodes (about 70, against about 20 before), with noise ported from MaterialX, so graphs bake as MaterialX renders them. Linked privately; the public API is unchanged.
  - **Standalone materials bake their graphs too.** `uploadMaterial()` resolves procedural texture slots, as `setScene()` does; before, only the scene asset's materials did, so materials built at run time (BIM's) showed their fallback texture.
  - **One bake per graph.** A material's slots fed by one graph's outputs (base colour, normal, roughness from the same pattern) come from a single evaluation. Textures are cached by the graph's content, so materials with the same graph and values share them, whatever asset they come from.
  - **Encoding and sampling.** Colour slots (base colour, emissive, specular colour, sheen colour) get sRGB textures; the rest stay linear. Before, everything was 8-bit linear, which lost dark detail. Baked textures have a mip chain (before, one level, which shimmered in the distance) and their own repeat, trilinear sampler, since the draft says the glTF texture's sampler is ignored. They also record an average colour for the path tracer.
  - New test `ProceduralMaterialsBakeInUploadMaterial`: an sRGB, mipmapped texture, shared by two assets with the same graph, and a quad showing the checkerboard the right way up.
- **The bake follows MaterialX semantics** (`bakeProceduralTexture()`):
  - Texture coordinates have v up, so row 0 of the image is v = 1 and the image is a glTF texture as it is.
  - `place2d` subtracts its offset.
  - Float outputs are grey, (v, v, v, 1).
  - Values are rounded to 8 bits, not truncated.
  - Only MaterialX names are accepted: the old baker's `abs`, `combine` and `noise2d`'s `position` input are now `absval`, `combine2/3/4` and `texcoord`.
  - A graph that doesn't compile logs why and bakes black.

- **gltf v0.5.3**: procedural graphs without `inputs` (no parameters) no longer abort the load.

### Deprecated
- **`bakeProceduralTextureGPU()`** is now the CPU bake (the device is unused). The GPU baker evaluated its own subset of nodes with its own noise, and its results differed from the CPU's. Its compute kernels are still in the shader sources, unused, until they're removed or rebuilt on campello_procedural's programs. Its 7 GPU-vs-CPU tests are removed.

## [0.26.0] - 2026-11-19

### Added
- **`PathTracer::denoiseAsync()`**: denoises the image so far without blocking the caller. The samples are copied on the GPU (submitted, not waited for); a worker thread reads the copy back, runs Open Image Denoise as `denoise()` does and tone maps it, while `render()` goes on tracing. `collectDenoised()` picks the result up without waiting into `denoisedOutput()` (a new texture each time, with `denoisedSamples()`), and drops the result of an image that started again since (another view, size or setting). `denoising()` tells whether one is running; `denoise()` waits for a running one first. An interactive viewer's denoise at 8, 32, 128 and 512 samples blocked its UI thread for 380-480 ms each (CPU denoiser, Intel UHD 630, 470x800); now it spends under a millisecond there.
- **`PathTracer::imageNumber()`**: a new number each time `render()` starts an image, so a caller can tell whether what it kept is of the current one.
- New tests `PathTracerTracesASampleInBandsWithinTheBudget` and `PathTracerDenoisesOnAWorkerThread`.

### Changed
- **`setFrameBudget()` splits a sample into bands of rows** when one sample of the whole image takes longer than the budget: each `render()` traces as many rows as fit, sized from each row's GPU time measured in the previous sample (rows cost very different amounts: sky, glass, interiors), at least 4k pixels and 8 rows; until a band is measured, an eighth of the image. `sampleCount()` counts a sample once its last band is submitted, and the bands add up to exactly the image whole samples make. A sample that keeps the GPU busy for 100 ms then never holds back the caller's other GPU work (its UI) for longer than about the budget. Shader: `pathTrace` offsets its rows by `TraceUniforms::lights.y`.
- **Up to two `render()` calls' work on the GPU at once**, so the GPU doesn't idle between calls (the uniforms and the converged-pixel counter are per slot). A third call waits for the first; rebuilds, a new size, a new environment table, new portals and the read and denoise calls wait for all of it. `sampleInFlight()` now tells whether the next `render()` would wait.
- **The converged-pixel count is read back only with a noise target** (`setNoiseTarget()`): reading it waits for the GPU's queue, and every finished sample read it.
- Measured in Campello BIM on a house model (Intel UHD 630, 470x800, 10 ms budget): frame intervals while tracing went from about 110 ms, with 380-480 ms stalls at each denoise, to 7 ms median, 21 ms at the 99th percentile and 44 ms at most; 128 samples take about 21 s instead of about 14 s.
- **Test results:** macOS 147 passed (the path tracer tests also with Metal API validation).

## [0.25.0] - 2026-11-19

### Added
- **Material specialization** (`setMaterialSpecialization()`, on by default; Metal and Vulkan). The PBR fragment shader is compiled per set of optional features a material uses (KHR_materials_specular, sheen, clearcoat, transmission with volume and dispersion, iridescence, anisotropy, world mapping, unlit, the debug views), through campello_gpu v0.32.0's shader constants (Metal: function constant 0, `kMaterialFeatures`; Vulkan: specialization constant 0 in `default.frag`, which reads the optional features' fields through `MAT_<field>` macros that give the off value as a constant; a first version copied the uniform block into a local struct, which an Adreno 642L kept in scratch memory and ran 7x slower). Unused features' code is left out. On Vulkan the flat variant (the same fragment shader) is specialized too.
  - A material's features come from its own uniforms, so a feature is only left out when its values already switch it off, and specialized pipelines render exactly as the general one (`SpecializedMaterialsRenderAsTheGeneralShader` compares them per pixel).
  - Plain materials' pipelines are created up front; other combinations the first time they're drawn.
  - Benchmark, frame time, general shader to specialized:
    - Intel UHD 630 (Metal): `fill` 47.0 to 23.3 ms, `draws` 6.67 to 4.47 ms, `shadows` about 9.0 to 6.2 ms.
    - Adreno 642L (Vulkan, Galaxy Tab S7 FE): `fill` 101.6 to 40.1 ms, `draws` 38.1 to 29.7 ms, `shadows` 28.4 to 15.3 ms.
  - The gain is in the image-based lighting, where most optional features' code runs; with IBL off there's little to remove.
  - The benchmark takes `--general` to compare with the general shader, and builds for Android (see `bench/frame_bench.cpp`).
- **View culling in `render(RenderScene)`** (`setViewCulling()`, on by default). Draws whose world-space bounds (`DrawCall::localBounds`, else the mesh's) lie outside the camera's view are skipped, so a scene listed without culling no longer draws everything; `render_system()` already culled, and a caller that does loses only the CPU test. Draws without bounds, instanced or skinned draws without `localBounds`, and wide lines are always drawn, and sun shadows still take every caster. `RenderStats::culledNodeCount` / `visibleNodeCount` count the draws skipped and drawn (before, always 0 and all). `isOutsideFrustum()` (`culling.hpp`) takes a box too. New test `ViewCullingSkipsOnlyDrawsOutOfView`.
  - Benchmark `shadows` (229 of 4,900 draws in view), frame time without culling (`--no-cull`) to with: Intel UHD 630 5.6 to 4.9 ms, Adreno 642L 15.0 to 12.6 ms.
- Benchmark: scenario `fill` (one opaque quad filling a 3840x2160 view: fragment cost), `--no-cull`, `--no-ibl` (pipelines after `setAsset()`, no environment bake), and a check that `fill` drew. Its render targets are kept for the run (on Vulkan a texture view doesn't keep its texture alive).

### Changed
- **campello_gpu v0.33.0**: pipeline-time shader constants (material specialization), texture views that keep their texture alive on Vulkan, and failed Vulkan submissions reported.
- **Test results:** macOS 145 passed (also with Metal API validation); Android: Adreno 642L (Galaxy Tab S7 FE, Android 14) 111 passed, 0 failed.

### Fixed
- **Nothing drew after the environment bake on some Adreno GPUs** (Vulkan). With pipelines created before `setAsset()` (the usual order), `setAsset()` bakes the environment's image-based lighting, and its 67 jobs were queued at once: an Adreno 642L (Galaxy Tab S7 FE) lost the device, so every later submission failed and frames came out empty, and an Adreno 618 wedged its driver. Natively each job is now waited for before the next. New test `FramesDrawAfterTheEnvironmentBake` (fails on the Adreno 642L without the fix).
- **Scene-asset materials in `render(RenderScene)`.** `uploadMaterial()` for a material of the scene's own asset reused the bind groups `setScene()` made but never wrote its uniforms (only the glTF `render()` path did, every frame), so a RenderScene draw before or without a glTF render read an empty slot: black base colour, zero roughness. The slot is written on upload now. New test `SceneAssetMaterialsDrawInRenderScene`. Earlier `draws`/`shadows`/`gbuffer` benchmark numbers were measured with those empty materials.

## [0.24.0] - 2026-11-18

### Added
- **`DrawCall::instanceTransformBuffer`.** A `RenderScene` draw can be instanced: one column-major float4x4 per instance, applied in model space before `worldTransform` (create it with `BufferUsage::vertex`), as glTF's EXT_mesh_gpu_instancing. Before, `DrawCall` had no way to pass instance transforms.
- **`DrawCall::localBounds`**: model-space bounds covering a draw's instances or the poses of its skin, for fitting sun shadows; unset, the mesh's bounds.
- **`RenderStats::shadowCastersDrawn` / `shadowCasterCount`**: the casters drawn into the far shadow map the last time it was redrawn, and how many there were.
- New benchmark scenario `shadows`: a third-person view of one corner of the grid with one cube moving, so the shadow map is redrawn every frame.
- New tests `DrawCallInstancesDrawAtTheirTransforms`, `SunShadowsFollowInstancesAndSkins`, `SunShadowsDrawOnlyCastersTheViewNeeds`.
- **DirectX catches up with Vulkan, except ray tracing.** Everything below was Metal and Vulkan only. Tested on an Intel Iris 550 (Windows 10) with the D3D12 debug layer on: the suite is 107 passed, 34 skipped (ray tracing, path tracing and its optional denoiser, MetalFX, wide lines, quantized meshes, the custom_lit Metal program), 0 failed, and the debug layer reports no errors. Ray tracing stays Metal-only: neither this GPU nor an Xbox Series X in UWP Developer Mode has DXR.
  - **G-buffer:** `gbufferVertex`, `gbufferSkinnedVertex`, `gbufferLeanVertex` and `gbufferPixel` in `shaders/directx/default.hlsl`, ports of the Vulkan stages. They use the PBR pipeline's combined bind group. The previous transforms are a per-instance vertex buffer at locations 20–23 and 28, as on Vulkan.
  - **TAA and the built-in upscaler:** `taaPixel`, a port of `taa.frag`. At half resolution its RMS error against 4× SSAA is 10.66 levels, against 21.54 for bilinear upscaling; native TAA is 10.37 (Vulkan: 10.65, 21.54, 10.37).
  - **Real-time denoiser:** `RealtimeDenoiser` runs on DirectX with `shaders/directx/denoiser.hlsl` (one entry point per pass), embedded in `src/shaders/directx_denoiser.h`. RMS error goes from 0.147 to 0.022 on the first frame and from 0.147 to 0.032 with a moving camera, as on Metal and Vulkan.
  - **Skinning and morph targets:** `vertexSkinned` and `gbufferSkinnedVertex` morph, skin and apply the instance's transform, as `skinning.glsl`. The joint palettes are bind group 2 and the morph targets bind group 3, raw buffers read as `ByteAddressBuffer`, bound by the same `bindSkin()`/`bindMorph()` as Vulkan (renamed from `bindVulkanSkin()`/`bindVulkanMorph()`). The renderer's own palettes and deltas stay CPU-written upload buffers on DirectX. Skinned draws have G-buffer motion.
  - **EXT_mesh_gpu_instancing:** the instance's transform at locations 12–15, the per-draw transforms with stride 0, as on Vulkan.
  - **Sun shadow maps:** a depth-only pipeline with `leanVertex`; `pixelMain` reads the map with a comparison sampler (`shadowLookup()`, 3×3 PCF, near and far cascades). The combined bind group gains the shadow map, its sampler and the near cascade at 39–41.
  - **SSAO:** `ssaoPixel` and `ssaoApplyPixel`, ports of `ssao.frag` and `ssao_apply.frag`.
  - **Clip planes and section caps:** `pixelMain` and `gbufferPixel` read the clip block; `supportsClipPlanes()` is true on DirectX. The camera block (b31) is bound whole now (464 bytes, was 160), which also brings exposure (`setExposure()`) to DirectX, in `pixelMain` and the skybox.
  - **Lines** (`lineVertex`, `supportsLines()`) and **position-only meshes** (`leanVertex`, `VertexLayout::positionOnly()`). `pixelMain` derives the flat face normal from screen-space derivatives, as on Vulkan. World-space box mapping (`GpuMaterial::worldMapping`) works in `pixelMain` too.
- **App shader programs on DirectX:** the element id test program has an HLSL source (`test/shaders/element_id.hlsl`). `campello_renderer_add_shader_program()` looks for dxc in the Windows SDK, where it ships, when it isn't on PATH.

### Changed
- **The far shadow map draws only the casters that can shadow what's in view** (when `RenderScene::shadowCasters` is empty, so every opaque draw casts and receives). The view, cut to the draws' bounds, gives a light-view rectangle; casters whose light-view bounds miss it (plus a 25% margin) aren't drawn. The map keeps the same projection and is redrawn when the casters or the light change or the view leaves the rectangle drawn. The near cascade uses the same caster test. Benchmark `shadows` (Intel UHD 630): 644 of 4,900 casters drawn, CPU time per frame 10.6 to 9.7 ms.
- **Test results:** macOS 141 passed (also with Metal API validation); Android (Adreno 618) 107 passed, 34 skipped (ray tracing), 0 failed.
- **DirectX uses Vulkan's split bind groups.** The PBR program's resources were one combined material+frame bind group per material and frame in flight, all rebuilt every frame (with 64 materials, 192 groups a frame), because campello_gpu's D3D12 `setBindGroup()` took a root parameter index. With that fixed in campello_gpu, DirectX uses the same layouts as Vulkan: group 0 is the material, built once; group 1 the frame (lights, camera, environment, scene colour, shadows), the only group rebuilt per frame or pass; groups 2 and 3 the joint palettes and morph targets. `default.hlsl` declares each group in its own register space. The DirectX-only combined-group code is gone (`ensureDirectXPbrBindGroupLayout()`, `buildDirectXCombinedBindGroup()`, `rebuildDirectXCombinedBindGroups()`, `GpuMaterial::directxBindGroup`).
- **`build_directx_shaders.ps1` is list-driven.** Each entry point is one line: name, profile, entry, C symbol, source and header. It writes `directx_default.h` and `directx_denoiser.h`.
- **campello_gpu v0.32.1** with the D3D12 fixes this needs (see its changelog): `setBindGroup()` by bind group index, sampled depth textures, raw views of storage buffers, storage buffers that can be read and uploaded, deferred release of resources still in use, views that keep their texture alive, shared sampler tables, cube mipmaps, depth-only pipelines and comparison samplers.
- **The DirectX procedural texture bake reads and writes raw buffers** (`ByteAddressBuffer` / `RWByteAddressBuffer`), matching campello_gpu's views; its output was a structured buffer with no matching UAV range, so the pipeline didn't match its root signature.

### Fixed
- **Out-of-bounds reads with `DrawCall::instanceCount` above 1.** Every instance read its matrix from a buffer holding one (the identity), so instances past the first read past its end. A draw now draws no more instances than its `instanceTransformBuffer` holds, one without a buffer; `RenderStats::instanceCount` counts those.
- **Instances of position-only meshes drew on top of each other.** The lean vertex stages (`vertexLean`, `gbufferVertexLean`, `default_lean.vert`, `gbuffer_lean.vert`; DirectX: `leanVertex`, `gbufferLeanVertex`) ignored the instance matrices; they apply them now, as the full stages do (glTF EXT_mesh_gpu_instancing and `DrawCall`).
- **Sun shadows of skinned and instanced casters** (Metal, Vulkan and DirectX: `shadowVertex`, `shadowSkinnedVertex`). The shadow pass drew every caster's first instance in its rest pose, so an animated character's shadow stayed in the bind pose and an instanced mesh cast one shadow. Casters with instances or a skin are drawn with their own depth-only vertex stages (`shadowVertexInstanced` / `shadowVertexSkinned`; `shadow.vert` / `shadow_skinned.vert`), one pipeline per joint/weight format. Skinned casters redraw the shadow map every frame, since their pose changes in their buffers. WebGPU and DirectX still draw them at rest.
- **DirectX: the sky was black in offscreen renders, and anything sampling the environment's lower mips read black.** campello_gpu's D3D12 `generateMipmaps()` skipped cube textures and textures without render target usage, so the environment cube's mips were never filled. Fixed in campello_gpu.
- **DirectX: `render()` of a glTF scene with many materials hung the GPU** (the benchmark's 64-material grid). Each combined bind group copied its samplers into the 2048-slot sampler heap, and the groups are rebuilt every frame. Fixed in campello_gpu by sharing identical sampler tables.
- **DirectX: the benchmark's output went wrong after a few frames.** `target()` keeps only the texture's view, and a D3D12 view didn't keep its texture alive, so its render target descriptor was reused by other textures. Fixed in campello_gpu (views keep their texture).
- **DirectX: thin glass in `render(RenderScene)` showed the wrong background.** Before the transparent pass, the combined bind groups weren't switched to the opaque scene copy, as the glTF `render()` path does. Transmission also skipped the sRGB-to-linear decode of the opaque scene that Metal and Vulkan do, so the background seen through the glass was too bright.
- **DirectX: the opaque passes bound the scene colour texture as a shader resource while it was their render target** (TAA, FXAA, SSAA), which D3D12 rejects. They bind the 1×1 default texture there now. Transmissive draws still get the opaque scene copy.
- **`RealtimeEffects::raytracingSupported()` was true where no ray traced effect can run.** It only checked `Feature::raytracing` and the G-buffer, so with the G-buffer on DirectX, a device with DXR (WARP) turned on effects whose shaders exist only on Metal and every effect timing stayed 0. It now also requires a backend with the effects (Metal).

## [0.23.0] - 2026-11-18

### Added
- **Frame benchmark** (`bench/frame_bench.cpp`, `-DCAMPELLO_RENDERER_BUILD_BENCH=ON`): CPU time per call and frame time for `render(RenderScene)` with 4,900 draws over 64 materials, the same grid as a glTF scene, `renderGBuffer()`, and the G-buffer plus `RealtimeDenoiser`, at 1280x720 offscreen.

### Changed
- **Less CPU per frame.** Measured with the benchmark on an Intel Mac (UHD 630), Release, CPU time per call: `render(RenderScene)` 8.64 to 7.17 ms (now GPU-bound), glTF `render()` 7.82 to 5.11 ms, `renderGBuffer()` 9.01 to 4.27 ms.
  - Bind groups already bound on the pass aren't bound again (`setMeshBindGroup()`). Every draw bound its material and frame groups; on Metal that is a call per texture, sampler and buffer in the group.
  - Draw transforms, the G-buffer's previous transforms and the shadow casters' transforms are staged and uploaded once per frame, not once per draw.
  - `render(RenderScene)` sorts pointers to the draws instead of copying every `DrawCall` (twice for front-to-back order), and keeps its scratch lists across frames.
  - `meshPipeline()` hashes the attributes a stage reads in place (`VertexLayout::hash(mask)`) instead of building a filtered layout per draw, and returns the last answer straight away when a draw asks for the same pipeline as the one before.
  - Material uniforms (and the per-frame globals patched into them) are uploaded only where they changed. Every material was rewritten every frame, in a buffer the frames in flight share, so a write could race the GPU reading an earlier frame.
  - The sun shadow map's cache key hashes 8 bytes per step instead of one.
- **Effects don't wait for the previous frame.** `RealtimeDenoiser`, `DdgiVolume`, `SkinnedMeshes`, `RayTracedShadows`/`RayTracedAmbientOcclusion`, `RestirDirectLighting` and `RestirGlobalIllumination` waited for their previous submission before rewriting their uniforms, so the CPU and GPU took turns whenever one was on. Their uniforms (and parameter buffers) are kept per frame in flight now, and a frame waits only for the one three frames back.
- **campello_gpu v0.31.2.** On Metal, render targets and storage textures are in GPU-only (Private) memory, which Apple GPUs compress losslessly, so every pass that writes or samples one (scene color, G-buffer, history, effect targets) moves less data. Uploads to them go through a blit.
- **Test results:** macOS 138 passed (also with Metal API validation); Android (Adreno 618) 104 passed, 34 skipped (ray tracing), 0 failed.

## [0.22.2] - 2026-11-18

### Fixed
- **No sky on mobile GPUs in the browser.** Environment cube maps from 32-bit float images (the built-in sky among them) were rgba32float, which WebGPU only filters with the optional `float32-filterable` feature; an Adreno 619 (Galaxy Tab S7 FE, Chrome) lacks it, so the skybox and IBL bake bind groups failed validation and no sky drew. `loadEnvironmentMap()` and `convertEquirectangularImageToCubemap()` store such maps as rgba16float now, clamped to the largest finite half (65504) so the sun doesn't turn into infinity.

## [0.22.1] - 2026-11-17

### Fixed
- **IBL bake on Apple GPUs in the browser.** The bake (BRDF LUT, prefiltered specular mips, irradiance; ~370M environment samples with the built-in sky) went out as one command buffer and ran past the command buffer watchdog on an M2 in Chrome (Dawn/Metal): "Caused GPU Hang Error", WebGPU device lost on the first frame. Each face of each mip is now its own command buffer, with one wait after the last.

## [0.22.0] - 2026-11-17

### Added
- **Morph targets on Vulkan.** POSITION and NORMAL targets blend on Vulkan as on Metal (up to 8 targets, with the same weights: node, mesh, animation).
  - They're in the deforming vertex stages (`default_skinned.vert`, `gbuffer_skinned.vert`, shared code in `skinning.glsl`), which now morph and then skin, in glTF's order. `meshPipeline()` picks them for draws with a skin or morph targets.
  - The node's `MorphInfo` and the primitive's deltas are storage buffers at set 3 of the mesh pipelines, bound by `bindVulkanMorph()` (cached per buffers). The renderer's morph and fallback buffers are storage buffers on Vulkan.
  - `MorphTargetMovesEveryVertex` runs on Vulkan now.
- **EXT_mesh_gpu_instancing on Vulkan.** Each instance's matrix is a per-instance vertex attribute (locations 12-15) of the PBR and G-buffer stages. The per-draw transforms (and the G-buffer's previous transforms) have stride 0, so every instance of a draw reads the draw's own; before, instance *n* read the transforms of the *n*th next draw.
- New test `GltfInstancesDrawAtTheirTransforms`: a glTF node with two instances draws a quad at each instance's translation.

### Fixed
- **EXT_mesh_gpu_instancing on Metal.** The vertex stages read the first instance's matrix for every instance, so all instances were drawn on top of each other. They index the instance matrices by `[[instance_id]]` now (`vertexMain`, `gbufferVertex`).

### Changed
- **Android results:** 104 passed, 34 skipped (ray tracing), 0 failed.

## [0.21.0] - 2026-11-17

### Added
- **The games foundation on Vulkan: G-buffer, TAA, built-in upscaling and the real-time denoiser.** Until now these were Metal-only. On Vulkan `supportsGBuffer()` was false and their tests skipped. Tested on an Adreno 618 (Android 13).
  - **G-buffer:** `renderGBuffer()` writes normal + roughness, base colour + metallic, motion and depth in one pass, using `shaders/vulkan/gbuffer.vert`, `gbuffer_lean.vert` and `gbuffer.frag`, ports of the Metal stages.
    - The previous frame's transform and jitter, which Metal reads as a raw buffer, are a per-instance vertex buffer here (locations 20–23 and 28). Mesh pipelines for the G-buffer add it after the per-draw transforms (`MeshPipelineRecipe::gbufferPreviousTransforms`).
    - Skinned draws have skinned motion: the previous joint palette places the previous position (see skinning below).
  - **TAA and the built-in upscaler:** `shaders/vulkan/taa.frag`, a port of `taaFragment`. At half resolution its RMS error against 4× SSAA is 10.65 levels, against 21.54 for bilinear upscaling; native TAA is 10.37.
  - **Real-time denoiser:** `RealtimeDenoiser` runs on Vulkan with `shaders/vulkan/denoise_*.frag` (one per pass, shared code in `denoiser_common.glsl`), embedded in `src/shaders/vulkan_denoiser.h`. RMS error goes from 0.148 to 0.022 on the first frame and from 0.148 to 0.032 with a moving camera, as on Metal.
- **Skinning on Vulkan.** Skinned meshes are posed by their joints on Vulkan, as on Metal: glTF skins and `DrawCall::jointMatrixBuffer`. Before, Vulkan drew every mesh in its bind pose.
  - New vertex stages `default_skinned.vert` and `gbuffer_skinned.vert` (shared code in `skinning.glsl`; the unskinned stages are the same sources without `SKINNED`). They read JOINTS_0 and WEIGHTS_0 and blend up to four joint matrices; weights summing to zero leave a vertex unskinned.
  - The joint palette is a storage buffer at set 2 of the mesh pipelines (binding 0). The G-buffer stage also reads the previous palette (binding 1, `DrawCall::previousJointMatrixBuffer`), so a moving skin has motion vectors.
  - `meshPipeline()` picks the skinning stage only for draws with a skin, so other meshes keep the lighter stage even though the glTF standard layout always has joints. `createDefaultPipelines()` prewarms the unskinned pipelines; a skinned one is made on first use.
  - Bind groups for the palettes are cached per buffer and range (`bindVulkanSkin()`). The joint buffers the renderer creates are storage buffers on Vulkan.
  - New tests: a glTF skin and a `DrawCall` palette each pose a quad where its joint puts it (Metal and Vulkan). `GBufferHoldsSurfacesAndMotion` checks skinned motion on Vulkan too. All three fail with the skinning stage disabled.
  - No morph targets on Vulkan yet.
- **`run_android_tests.sh`** builds the test suite for arm64, pushes it to the connected device with adb and runs it, optionally with a gtest filter. `CAMPELLO_GPU_SOURCE` builds against a local campello_gpu checkout.

### Changed
- **campello_gpu v0.31.1.** v0.31.0 brings multiple render targets on Vulkan (the G-buffer's three targets, TAA's two); v0.31.1 fixes the Vulkan bugs behind the last Android failures (storage buffer bindings, depth kept across passes, compute barriers).
- **`build_vulkan_shaders.sh` is list-driven.** Each shader is one line: source, `.spv`, C symbol and header. It writes `vulkan_default.h` and `vulkan_denoiser.h`, and resolves `#include` in GLSL.
- **Android results:** 102 passed, 35 skipped (ray tracing), 0 failed. Before this release, 88 passed, 40 skipped and 7 failed.

### Fixed
- **GPU procedural bakes on Vulkan** (`bakeProceduralTextureGPU()`). They returned all zeros because campello_gpu left the shader's storage buffers unbound; fixed in campello_gpu v0.31.1. The bake shader's uniform block also padded with `uint[3]`, which std140 lays out with 16-byte strides (80 bytes, against the 32-byte struct bound to it); it pads with scalars now. The 6 bake comparisons pass on Android.
- **Screen-space AO on Vulkan devices without dynamic rendering** (Vulkan 1.1, e.g. Adreno 618). Their render passes discarded the scene's depth, so SSAO sampled nothing and had no effect; fixed in campello_gpu v0.31.1.
- The GPU bake comparison tests print the first mismatching pixel.

## [0.20.0] - 2026-11-16

### Added
- **Skinned meshes in ray tracing.** `SkinnedMeshes` (`rt_effects.hpp`) runs a compute pass that poses each skinned draw's positions and normals with its joint matrices, as the raster vertex stage does, into a posed copy of its mesh.
  - `RayTracingScene` and `PathTracer` trace these copies, so every ray traced effect, the reflections, ReSTIR GI and DDGI probes see characters where they're posed. Before, the bind pose was traced.
  - Joint and weight formats come from the mesh's layout, or from the buffer sizes for the standard layout.
  - The posed meshes' structures are rebuilt every frame; `RayTracingScene::skinnedRebuilds()` counts them.
  - In the test, a plate moved 1 m by its joint shadows the ground where it's posed: visibility 0.00 there, 1.00 under its rest pose, and the same in the path tracer.
- **Alpha-tested geometry in ray tracing** (glTF `MASK`: foliage, fences). These instances aren't opaque, and an `intersection_query` loop counts a candidate hit only where the base colour factor times the texture's alpha reaches the cutoff, as the raster shader discards.
  - `PathTracer` uses its own UVs and textures, including `KHR_texture_transform`.
  - The effects (shadows, AO, ReSTIR DI and GI visibility) use `RayTracingScene`'s per-instance alpha data, triangle UVs gathered on the GPU, and up to 16 textures.
  - `GpuMaterial::alphaCutoff` and `baseColorAlpha` are new.
  - In the test, a plate whose texture is transparent on one half shadows the ground only with its opaque half: visibility 1.00 against 0.00, and the same in the path tracer.
- **Quality presets: `RealtimeEffects`** (new header `realtime_effects.hpp`). It runs the Games track's effect chain for an app: the G-buffer, then the enabled ray traced effects, each denoised and handed to the renderer, then the frame with TAA at the render scale.
  - `EffectQuality` presets:
    - `off`: raster only.
    - `low`: RT sun shadows, half resolution.
    - `medium`: + RT AO.
    - `high`: AO, reflections and ReSTIR DI.
    - `ultra`: + ReSTIR GI.
  - `Settings` changes the effects one by one; `settingsFor()` returns a preset's.
  - Without `Feature::raytracing` the presets fall back to raster only. MetalFX is the upscaler wherever it's supported.
  - `setFrameBudget(ms, minScale)` adds dynamic resolution: the render scale steps by 0.05, at most every 8 frames, to keep the frame's GPU time under the budget.
  - `timings()` gives the last finished frame's GPU milliseconds per effect, without waiting.
  - At `ultra`, at 64 × 64 on the Intel UHD 630, a frame takes 6.9 ms: denoising 2.3, GI 1.4, reflections 1.1, raster 1.5.
- **Per-effect GPU timing**:
  - `gpuTimeMs()` on `RayTracedShadows`, `RayTracedAmbientOcclusion`, `RestirDirectLighting`, `RestirGlobalIllumination`, `RealtimeDenoiser` and `DdgiVolume`. These are command-buffer GPU times read once each fence signals (`GpuTiming`, `gpu_timing.hpp`).
  - `PathTracer::passGpuTimeMs()` for its reflection, GI and probe passes.
  - `RenderStats::gbufferGpuTimeMs`: `renderGBuffer()` no longer overwrites the frame's `gpuFrameTimeMs`.
- **ReSTIR GI: one-bounce indirect light** (Metal, `Feature::raytracing`). `RestirGlobalIllumination` (`rt_effects.hpp`), after Ouyang et al. 2021, traces one cosine-weighted path per pixel from the G-buffer with its own `PathTracer`. The paths see the tracer's materials, sun, sky and emitters.
  - The samples are resampled by the irradiance they bring and reused along the motion vectors (M capped at 30).
  - While a pixel's history is short, neighbours' samples are reused too, with the reconnection's Jacobian and a visibility test.
  - The output is irradiance, the same quantity as `DdgiVolume`'s, for `Renderer::setIndirectDiffuse()`. Denoise it with demodulation off.
  - `Settings` covers temporal reuse and its cap, spatial reuse with its samples, radius and history limit, and bounces.
  - In the DDGI test's scene (a red wall beside white ground under the sun), the relative error against 256 frames of raw paths is 2.31 with one path per pixel and 0.31 after 16 frames. The wall's red on the ground near it is r 0.256 against 0.241. Through the denoiser into the raster frame, that ground goes from red 206 to 232 with green unchanged.
- **`PathTracer::traceGlobalIllumination()`** (mode 4): one cosine-weighted path per G-buffer pixel. `globalIlluminationSamples()` holds the radiance leaving the hit, its distance, and the direction and hit normal (octahedral); `setGlobalIlluminationBounces()` sets the bounces.
- **ReSTIR DI: many dynamic lights** (Metal, `Feature::raytracing`). `RestirDirectLighting` (`rt_effects.hpp`), after Bitterli et al. 2020, lights every pixel from all of `RenderScene::lights` (directional, point, spot; every light casts shadows) with one shadow ray per pixel.
  - Each pixel draws 32 candidates by light power and resamples them by their unshadowed contribution, using the raster BRDF from the G-buffer. It traces the winner's visibility, then reuses the previous frame's reservoir along the motion vectors (M capped at 20 frames of candidates).
  - While a pixel's history is short, it also reuses neighbours' reservoirs (radius: 1% of the width). It then traces one shadow ray for the result.
  - The history is the temporal pass's reservoirs: feeding back the spatial ones compounded their bias in the test.
  - `renderReference()` is the ground truth: every light with its own shadow ray.
  - `Settings` covers candidates, temporal reuse and its cap, and spatial reuse with its samples, radius and history limit.
  - In the test, 64 coloured point lights shine over a ground plane with a plate shadowing it. Relative RMS error against the ground truth is 1.66 with one light sample, 0.37 resampling 32 candidates and 0.10 after 16 frames of reuse (0.23 with a moving camera). The mean is within 0.1% and the plate's shadow is exact (0.2392 vs 0.2393).
- **`Renderer::setDirectLighting(radiance)`** (Metal): a direct light texture (ReSTIR's, denoised) replaces the raster light loop, which handles 4 lights with sun shadows only, for opaque, non-transmissive surfaces. The screen lighting texture now holds three columns: reflections, indirect diffuse and direct light. For app programs: `campello::hasDirectLighting()` / `directRadiance()` in `program_interface.metal.h`. With one sun, the frame matches the light loop within 1 level of 255.
- **Lighting inputs for app shader programs** (Metal): `program_interface.metal.h` now exposes what the renderer's effects give the built-in PBR shader, as structs (`campello::CameraUniforms`, `Light`, `LightsUniform`), binding constants and helpers:
  - `campello::sunVisibility()` for the shadow maps, or the ray traced mask when `setShadowMask()` is set.
  - `hasReflections()`, `reflectedRadiance()` and `reflectionWeight()` for ray traced reflections.
  - `hasIndirectDiffuse()` and `indirectIrradiance()` for indirect diffuse light, such as DDGI.
  - `exposureScale()`, and `CAMPELLO_LIGHTING_PARAMETERS` to declare the inputs in a fragment function.
  - The built-in `default.metal` includes the same header, so app programs see exactly what PBR uses. The effects' internal packing stays behind the helpers.
- **App programs' own G-buffer stages** (Metal): `ShaderProgramDesc::metalGBufferVertexEntryPoint` / `metalGBufferFragmentEntryPoint`. `renderGBuffer()` draws a program's materials with its own stages, so geometry its vertex shader moves, and normals its fragment shader computes, reach TAA, the upscalers, the denoisers and the ray traced effects. The header adds `campello::GBufferTransforms`, `unjitteredClip()`, `GBufferOut` and `gbufferOutput()`.
  - Without these stages, the built-in G-buffer stages draw the program's meshes as before.
  - In the test, a program that moves its geometry 0.4 sideways gets its G-buffer albedo and motion at the moved place (motion −0.0549 against an expected −0.0549). The built-in stages put it at the unmoved place.
- **`docs/using-the-renderer.md`** covers:
  - the ways to use the renderer (glTF viewer, your own `RenderScene`, offline path tracing, building blocks);
  - every feature's switch, default and backends, and what custom programs get from it;
  - a full ray traced effect chain;
  - custom shading, and what doesn't carry over yet.
- **MetalFX upscaling** (Metal, macOS 13 / iOS 16 and later): `Renderer::setUpscaler(Upscaler::metalFX)` makes TAA, and TAA with a render scale, use Apple's MetalFX temporal scaler (campello_gpu's new `TemporalUpscaler`) in place of the built-in resolve.
  - MetalFX takes the jittered frame, the G-buffer's depth and its motion vectors.
  - `supportsUpscaler()` tells which upscalers the device supports. `activeUpscaler()` gives the one the last frame used.
  - The renderer falls back to the built-in upscaler where MetalFX is unsupported, or where the ratio exceeds its maximum (`Device::getTemporalUpscalerMaxScale()`: 2x on some GPUs).
  - In the test, at half scale: 16.0 levels RMS from the 4× SSAA reference, against 21.5 for bilinear, and no trail behind a moving object. MetalFX blends edges a little darker than a box average, so its error is above the built-in upscaler's 10.7, although the edges are in the same place.
- **Temporal upscaling** (Metal): `Renderer::setRenderScale(0.25 .. 1)` draws the scene, the G-buffer and so the ray traced effects and denoisers at that fraction of the output size, and with `setTemporalAntiAliasing()` on the TAA resolve becomes a temporal upscaler: each output pixel gathers the nearest jittered samples weighted by their distance in output pixels into an output-size history (8 / scale^2 jitter positions), so detail approaches native resolution after a few frames; without TAA a bilinear scale-up. `internalWidth()` / `internalHeight()`. In the test, at half scale: 10.7 levels RMS from a 4x SSAA reference against native TAA's 10.4 and bilinear's 21.5; no trail behind a moving object.
- **Probe-based global illumination** (Metal, DDGI after Majercik et al. 2019): `DdgiVolume` (new header `ddgi.hpp`) keeps a grid of light probes (`Grid`, `fitting(min, max, spacing)`), each tracing rays every frame along rotated spherical Fibonacci directions with the path tracer (`PathTracer::traceProbeRays()`, a new kernel mode: full materials and lights), blended over frames (hysteresis) into octahedral irradiance and distance atlases; `resolve(gbuffer, camera)` gives each G-buffer pixel its 8 surrounding probes' irradiance, weighted by position, facing and Chebyshev visibility. `Renderer::setIndirectDiffuse(irradiance)` uses it instead of the irradiance map for opaque, non-transmissive surfaces (reflections and indirect diffuse are copied side by side into the scene colour slot). In the tests: a red wall bleeds onto white ground (r 0.142 vs g 0.011 near it, weaker far away), the raster ground near it turns redder (206 -> 222), and a sealed box on sunlit ground gets no light inside (0.0000 vs 0.0258 outside; 0.0034 leaks in with the visibility term disabled).
- **Ray traced reflections** (Metal, `Feature::raytracing`): `PathTracer::traceReflections(scene, renderer, gbuffer)` starts a path at each G-buffer pixel's surface in a direction of its GGX lobe (its roughness) and shades it as the path tracer does (materials, textures, the sun, the sky, emitters, `setReflectionBounces()` more bounces), writing the arriving radiance to `reflections()` (one sample per pixel, for `RealtimeDenoiser`); a mode of the path tracing kernel (G-buffer bindings 66-68). `Renderer::setReflections(radiance, maxRoughness)` uses it instead of the prefiltered environment in the specular term of opaque, non-transmissive surfaces, fading back towards the roughness cut-off (default 0.6). In the test: a mirror floor before a red wall goes from the sky's near-white (green 237) to the wall's red (green 78); a rough floor is unchanged.
- **Ray traced ambient occlusion** (Metal, `Feature::raytracing`): `RayTracedAmbientOcclusion` traces one ray per pixel from the G-buffer's surface in a cosine-weighted random direction, as far as `setRadius()` (default 1 m); `Renderer::setAmbientOcclusionMask(occlusion)` uses it instead of SSAO's estimate (SSAO's apply pass, whether or not SSAO is on). `RayTracingScene`: the effects' acceleration structures, shareable so shadows and AO build them once per frame (`RayTracedShadows` / `RayTracedAmbientOcclusion` take one; each makes its own without). In the test: ground beside a 1 m wall at 0.58, beyond the radius 1.0; denoised into the raster frame, 240 -> 160 beside the wall, far ground unchanged.
- **Ray traced sun shadows** (Metal, `Feature::raytracing`): `RayTracedShadows` (new header `rt_effects.hpp`) traces one shadow ray per pixel from the G-buffer's surface towards a random point of the sun's disc (`setSunAngularDiameter()`, default 0.53 degrees: soft shadows from a larger sun), new random numbers every frame, against its own acceleration structures (a BLAS per mesh, cached; the instances only when transforms change). `Renderer::setShadowMask(visibility)` shades the sun with a screen-space visibility mask instead of the shadow maps (whose passes are then skipped). `RealtimeDenoiser::Settings::demodulate` (false for visibility signals). The chain `renderGBuffer()` -> `RayTracedShadows::trace()` -> `RealtimeDenoiser::denoise()` -> `setShadowMask()` -> `render()` gives the shadow maps' result (shadowed ground at 14 of 206 vs 0 of 206 after 16 frames in the test); shaders: `shaders/metal/rt_effects.metal`.
- **Real-time denoiser** (Metal): `RealtimeDenoiser` (new header `realtime_denoiser.hpp`) denoises a noisy image (about one sample per pixel of a ray traced effect) guided by `Renderer::gBuffer()`: SVGF-style (Schied et al. 2017) albedo demodulation, temporal accumulation along the motion vectors with depth / normal disocclusion rejection, per-pixel variance from luminance moments (a 7 x 7 spatial estimate while a history is short), five edge-aware a-trous wavelet passes (depth, normal, variance-guided luminance), the first feeding the next frame's history; the background passes through. Fixed cost per frame, render passes only (shaders: `shaders/metal/denoiser.metal`, embedded by `build_metal_shaders.sh` for macOS, iOS and the Simulator). `setSettings()`, `reset()`. In the test (input = clean image x 2u per pixel): 8.6x less error after 16 frames, 2.5x at a hard shadow edge, 4.6x with a moving camera.
- **Temporal anti-aliasing** (Metal): `Renderer::setTemporalAntiAliasing(true)` jitters each frame's projection by a sub-pixel Halton(2, 3) offset (8 positions), runs the G-buffer pass first for motion (same jitter, unjittered motion), and resolves into a history reprojected along the motion vectors, clamped to the frame's 3 x 3 neighbourhood in YCoCg against ghosting, blended at `setTemporalAntiAliasingBlend()` (default 0.1). Replaces FXAA while on; off with SSAA. In the test: hard edges (0 partly covered pixels) become anti-aliased (94) at the same brightness, and a moving object leaves no trail.
- **G-buffer with motion vectors** (Metal): `Renderer::renderGBuffer(scene)` draws the opaque draws into `gBuffer()` at the render size: world normal (normal textures applied) + roughness (rgba16float), base colour + metallic (rgba8unorm), motion (rg16float: previous minus current position in image uv, y down) and depth. Motion covers the camera (the previous `renderGBuffer()` call's; `resetMotionHistory()` for cuts), objects (`DrawCall::previousWorldTransform`) and skinning (`DrawCall::previousJointMatrixBuffer`); the ECS fills both (`Transform::previousWorldMatrix` from `transform_hierarchy_system()`, `MeshRenderer::previousJointMatrixBuffer` from `skinning_system()`'s two alternating buffers). A built-in program behind a reserved handle, so it reuses the mesh pipeline cache and every vertex layout (standard, position-only, quantized). The Games track's first prerequisite: input for ray traced effects from the first hit, temporal reuse, denoising and upscaling. `supportsGBuffer()`.
- **Path tracer: the sky sampled as a light.** At each diffuse or glossy hit, a direction is picked by the environment's brightness (a cumulative distribution over a 6 x 64 x 64 cube of its luminance, built on the GPU when the environment changes) and traced as a shadow ray, combined with the BSDF's own directions by multiple importance sampling (power heuristic). Shadow rays cross glass, tinted as a path crossing it would be on average, so interiors lit through windows benefit too. Much less noise for skies with bright regions (a test sky with a small bright patch: about 9 times less error at the same sample count), the same image on average. `PathTracer::setEnvironmentSampling()` (default on).
- **Low-discrepancy sampling**: Owen-scrambled Sobol random numbers (Burley 2020, hash-based scrambling, 2D pairs padded across each path vertex's fixed dimensions) instead of independent PCG numbers. Each sampling strategy (BSDF lobe and direction, sky texel and point) draws from a single 2D pair, the choice's leftover fraction reused as a coordinate, so Sobol stratifies the whole estimate; about 23% less error at 16 samples in the same test. `PathTracer::setSampler(Sampler::sobol | Sampler::random)`.
- **Smooth shading, normal and metallic-roughness textures in the path tracer**: the meshes' float3 NORMALs are interpolated across each triangle (face normals before: curved surfaces looked faceted), normal textures bend them (a tangent per triangle from TEXCOORD_0, or the world mapping's frame; the normal texture's own KHR_texture_transform and scale), and metallic-roughness textures scale the factors (through the base colour's transform). Rays leave along the geometric normal and directions below it are dropped, so bent normals don't leak light. Textures on TEXCOORD_1 are left out (the tracer reads TEXCOORD_0 only). `GpuMaterial::normalTexture`, `metallicRoughnessTexture`.
- **Path tracer camera effects**: depth of field from a thin lens (`PathTracer::setDepthOfField(apertureRadius, focusDistance)`, the lens point from its own Sobol pair; `apertureRadius(focalLength, fNumber)`), tone mapping options for output() and denoise() (`setToneMapping(pbrNeutral | aces | agx)`: Khronos PBR Neutral as before, ACES after Stephen Hill's fit, AgX; `toneMap()` takes the mode too), and `exposureForCamera(fNumber, shutter, ISO)` for `Renderer::setExposure()` in physically lit scenes.
- **Emissive surfaces as lights** (luminaires): every emissive triangle (up to 2^20; computed in world space on the GPU) goes into a light list sampled by power, a point uniformly on the chosen triangle, a shadow ray through glass, with multiple importance sampling against the BSDF (emission found by bounces is weighted through the instance's offset into the list). A closed room lit by one small ceiling panel: about 25 times less error at 64 samples, the same image on average. `PathTracer::setEmissiveSampling()` (default on).
- **Light portals**: `PathTracer::setPortals()` takes rectangles (windows, glass openings) the sky's light comes through; at each hit, most of the sky's light samples aim at the portals in sight (one picked by its solid angle, a direction uniformly within it: Ureña et al. 2013's spherical rectangle sampling), the rest at the sky by brightness, all combined with the BSDF by multiple importance sampling. Interiors lit through windows converge much faster (a closed room under a uniform sky, lit through one skylight: about 6 times less error at the same sample count, the same image on average). `PathTracer::portalsFromGlass()` makes portals of a scene's flat glass panes (transmissive transparent draws with thin bounds).
- **Firefly control**: `PathTracer::setIndirectClamp(maxRadiance)` clamps light that reached the camera after two or more diffuse or glossy bounces (keeping its hue); direct light is never clamped. Default off.
- **Adaptive sampling**: `PathTracer::setNoiseTarget(relativeError, minSamples)` stops each pixel once the relative standard error of its average (luminance, from a per-pixel second moment; against a floor of 0.05 after exposure) is under the target; stopped pixels are still shown. `Stats::convergedPixels` / `tracedPixels` give a noise target to stop at instead of a sample count (the test scene: 64% of pixels stopped after 32 samples at a 2% target, 99% after 256). The denoiser guides keep a per-pixel sample count.
- **Several samples per `render()`**: `PathTracer::setSamplesPerRender(n)` traces n samples per pixel in one dispatch (the same image as n renders of one), and `setFrameBudget(ms)` picks the count from the measured GPU time per sample and pixel to fill a budget (1 to 64), never past the end of the preview or of the first samples' fewer bounces. `render()` still doesn't wait. `Stats::samplesPerRender`.
- **Reduced resolution preview**: `PathTracer::setPreviewResolution(divisor, samples)` traces an image's first samples at 1 / divisor of the resolution in each direction, shown upscaled (blocks), then starts again at full resolution; `denoise()` during the preview filters the small image and upscales it (bilinear). `previewing()`. Default off.
- **Fewer bounces while converging**: `PathTracer::setProgressiveBounces(bounces, samples)` traces an image's first samples with fewer bounces, for a faster first preview. Default off.
- **Denoising on the GPU**: `PathTracer::denoise()` uses a GPU device of Open Image Denoise where the library has one (the macOS arm64 package ships its Metal device, `libOpenImageDenoise_device_metal`; CUDA, HIP and SYCL builds elsewhere), else the CPU as before. `PathTracer::setDenoiserDevice(automatic | cpu | gpu)`, `denoiserDeviceType()`. Apps bundling the denoiser ship the device module next to the core library (`CAMPELLO_RENDERER_DENOISER_LIBRARY_DIR`).
- **Denoiser quality**: `PathTracer::setDenoiserQuality(fast | balanced | high)` (default high, as before) and `setDenoiserPrefilter()` (denoises the albedo and normal guides with their own filters first, `cleanAux`, so noise in them isn't kept as detail; for stills). On the CPU at 64 x 64, 2 samples: fast 4 ms, high 7 ms, high with prefiltered guides 27 ms; error against the converged image 5.6, 2.9 and 2.0 levels of 255 (14.3 noisy).
- **`PathTracer::stats()`**: GPU time of the last sample and the mean over the image (command buffer GPU time), the last scene rebuild and `denoise()` (wall clock), time from an image's first sample to its first denoised result, and the traced triangle and instance counts. Doesn't wait for the GPU.

### Fixed
- **BLAS geometry is built non-opaque.** The instance now decides opacity. On Metal an opaque geometry overrides a non-opaque instance, which kept the alpha tests from running.

### Changed
- campello_gpu v0.29.0 -> v0.30.1 (`TemporalUpscaler`).
- **TAA's resolve** gathers each output pixel's 3 x 3 nearest samples weighted by their jittered distance (it took the pixel's own sample before), the neighbourhood clamp widened to 1.5 standard deviations: smoother edges (the test triangle's partly covered edge pixels 94 -> 137).
- SSAO's apply pass reads its AO texture at the texture's own size (a mask or the internal size).
- The intermediate scene textures follow the scene target's size (SSAA or a render scale), not only SSAA's.
- `vertexMain`'s body is now the shared `pbrVertex()` helper (the G-buffer's vertex stage reuses it); no change in output.
- **Path tracer emission is one-sided** unless the material is double-sided, as glTF has it (and the raster renderer, which culls back faces): a panel facing down no longer also lights what's above it. Mirroring transforms flip the front, as in glTF.
- **Denoiser guides through mirrors**: smooth surfaces (roughness under 0.15) whose path reflects pass the albedo and normal guides on to what they reflect, the albedo tinted by the reflection colour (as through glass already), so reflections keep their detail when denoised.
- **The path tracer rebuilds only what changed**: moving instances rebuilds the instance list and the scene's acceleration structure only (the meshes' structures, face normals, texture coordinates and materials are kept: 1 ms instead of 24 in the test scene); lights and lighting settings start a new image without a rebuild; exposure no longer starts the image again (it applies to the average when shown). `Stats::fullRebuilds`, `transformRebuilds`.
- A mesh whose position or index buffer was replaced gets a new acceleration structure (the cached one was reused before).
- The BSDF's next-direction choice and Russian roulette use the path's sampler dimensions; images differ from 0.19.0 sample for sample (not on average).

## [0.19.0] - 2026-11-13

### Changed
- campello_gpu v0.29.0 (`Surface`: one presentable swapchain per native window, so apps can drive several windows from one device; Vulkan devices accept recording from several threads). The renderer draws into whatever target it is given, so nothing else changes.

## [0.18.2] - 2026-11-12

### Added
- **Textures in the path tracer**: base colour textures are sampled at each hit (up to 64 distinct 8-bit textures per scene, filtered linearly), through the mesh's TEXCOORD_0 (float, or normalized 16/8-bit) and KHR_texture_transform, or the renderer's world mapping (`setMaterialWorldMapping()`, the raster shader's box mapping), so traced images match the viewport's textures. The denoiser's albedo guide carries them too, so denoising keeps their detail. Meshes without coordinates, and textures beyond 64, shade with the texture's average colour as before. `GpuMaterial::baseColorTexture`.

## [0.18.1] - 2026-11-12

### Changed
- **`PathTracer::render()` doesn't wait for the GPU** any more: the sample is submitted and the next `render()`, `readOutput()`, `readLinear()` and `denoise()` wait for it; `sampleInFlight()` tells an interactive caller to skip a frame instead of waiting (a live view's UI thread spent each sample's GPU time waiting).

### Fixed
- **Path tracer: glass spent the path's bounces**, so interiors seen through windows were nearly black (a view into a room crosses glass twice: in, then out to the sky, half of the default 4 bounces). Crossing or reflecting off thin glass no longer counts as a bounce (up to 8 crossings per path), and the sky seen through glass is at full strength, as seen directly.

## [0.18.0] - 2026-11-12

### Changed
- campello_gpu v0.28.2 (Metal ray tracing scenes keep their per-mesh structures resident: geometry vanished from path traced images under memory pressure).
- Path tracer camera rays start at the eye from the view matrix (not `CameraData::position`) and go through the pixel's point on the near plane; the far plane isn't unprojected any more (32-bit precision with deep projections).

### Added
- **Path tracer presentation images**: `PathTracer::setProjection(Projection::panorama)` traces a 360 x 180 degree equirectangular panorama around the camera (the image centre straight ahead, the camera's up at the top; use a 2:1 size). `denoise()` cleans up the image so far with **Intel Open Image Denoise** 2.5.1 (Apache-2.0; CPU device), guided by the first surfaces' albedo and normals that the path tracer now accumulates, and shows it in `output()` until the next `render()`; `denoiserAvailable()`, `denoised()`. `readOutput()` (rgba8, display-encoded) and `readLinear()` (linear rgb, optionally the albedo and normal guides) read the image back; `toneMap()` is the shader's tone mapping on the CPU.
- **`CAMPELLO_RENDERER_DENOISER`** CMake option (default on for macOS, the path tracer's platform): downloads OIDN's prebuilt release (`dependencies/oidn.cmake`, checksummed; macOS x86_64 and arm64 for now) and links it; `CAMPELLO_RENDERER_DENOISER_LIBRARY_DIR` names its libraries for apps bundling them.
- The denoiser keeps its Open Image Denoise device, filter and buffers between images of the same size (committing a filter loads its weights): a walkthrough's denoised frames take less than half as long.
- `RenderSystemOptions::noCulling`: `build_render_scene()` lists every draw whatever the camera sees (no frustum or size culling, `MeshRenderer::visible` ignored), for path tracing and panoramas.

## [0.17.0] - 2026-11-11

### Added
- **Path tracing** (`PathTracer`, new header `path_tracer.hpp`; Metal, where campello_gpu reports `Feature::raytracing`): progressive, one sample per pixel per `render()`, tone mapped like the raster renderer and restarted when the geometry, materials, lights, settings, camera or size change. One acceleration structure per mesh (cached), a scene structure of the draws (`RenderScene::shadowCasters` when filled, so geometry off screen counts, plus the transparent draws), face normals computed on the GPU from the meshes' own buffers. Diffuse + GGX specular, up to four directional lights sampled directly as soft 0.53-degree discs with shadow rays, the environment as sky light (seen directly at full strength, as the skybox), thin glass (Fresnel reflection or straight through; it doesn't block the sun), emissive and unlit materials, Russian roulette. Base colour textures contribute their average colour (`GpuMaterial::baseColorTextureAverage`); textures, denoising and Vulkan: later. Shaders: `shaders/metal/path_tracer.metal`, embedded by `build_metal_shaders.sh`. Test: a plate over a lit ground plane shadows it; moving the camera restarts the image.
- **`build_render_scene()`** (ECS): the RenderScene `render_system()` draws, without drawing it; `RenderSystemOptions::listAllCasters` lists every opaque mesh as a caster even without a shadow-casting light.
- `Renderer::getEnvironmentMap()`, `getSkyboxEnabled()`, `materialUniforms()` (a material's uploaded constants).

### Fixed
- **Diffuse image-based light was pi times too bright**: the irradiance map was baked as pi x the average radiance and shaders multiply it by the albedo; it's now the average radiance (as glTF-Sample-Viewer's Lambertian map), on every backend. Scenes lit by the environment get darker; raise `setEnvironmentIntensity()` by pi for the previous look.

## [0.16.0] - 2026-11-11

### Added
- **Antialiased wide lines** (`GpuMesh::wideLines`, Metal): `Renderer::expandLines()` stores each segment of a line list once (two float4: start and half width, end: the mesh's POSITION buffer) with one indexed quad per segment; `vertexWideLine` reads its segment by vertex index (no vertex attributes, `[[buffer(26)]]`) and expands it in screen space by a width in output pixels plus a fringe, sideways and past both ends, and `fragmentWideLine` fades the fringe by distance to the centre line. The fringe is 1 output pixel, narrowed by the SSAA scale (the downsample smooths too; `CameraUniforms::lineParams.x`). "Darkest wins" blending (min): overlapping lines don't stack into a heavier line. Segments shorter than 0.75 px on screen aren't drawn and fade from 30% up to 4 px. The material's base colour, clip planes apply, the same depth nudge as hardware lines; drawn after everything opaque. Wide lines keep their width and soft edges at any SSAA scale, unlike hardware lines (one pixel of the supersampled target: faint at SSAA 2x, dotted at 4x). `supportsWideLines()` says whether the backend can draw them (Metal for now; elsewhere use a `lineList`). Found in campello_bim. Tests: the expansion; a 3 px diagonal over a coplanar face drawn along its whole length with soft edges and the same width (within 0.2 px) at SSAA 1, 2 and 4; the same line drawn twice looks like once.
- **Transmission on WebGPU** (thin-walled: the opaque scene behind at the fragment's position, blurred by roughness, Fresnel-weighted) and a **WGSL copy/downsample pipeline** (`shaders/webgpu/copy.wgsl`), which the transmission path needs to copy the opaque scene and which also turns on SSAA on the web.
- **World-space box mapping** (`setMaterialWorldMapping(material, repeatSize)`): a material's textures (base colour, normal, metallic-roughness, occlusion, emissive) projected from the world position for geometry without texture coordinates (CAD, BIM): faces facing up or down take the plan projection, the others a projection along their own horizontal direction (v down), one repeat per `repeatSize` world units, with the matching tangent frame for normal maps; the up axis is the environment's. Material uniform floats 81-82 (the padding after `dispersion`), so no layout change. Metal, Vulkan, WebGPU. Test: a checker on a quad without UVs shows both colours only with world mapping.
- **Mipmaps for glTF images uploaded by `uploadMaterial()`**: 8-bit images get a box-filtered mip chain (linear light for sRGB), so repeating patterns don't shimmer in the distance.
- **Sky and image-based lighting on WebGPU**: the skybox (`shaders/webgpu/sky.wgsl`) and the IBL precompute (`shaders/webgpu/ibl_bake.wgsl`: BRDF LUT, GGX-prefiltered specular cube, irradiance cube) are WGSL ports of the Metal shaders, so the web is lit by the environment like the desktop instead of sampling the raw environment cube. FXAA and SSAA remain desktop-only.
- **Two shadow cascades** (`setShadowCascades()`, on by default): besides the map over all casters, a near map covers the part of the view closest to the camera (from the camera to halfway between the logarithmic and uniform split of the casters' depth range), so shadows near the camera are finer when zoomed in; the shaders use the near map where it covers the point. The light's view stays fixed by the casters' bounds; the near map's rectangle has a 25% margin and is redrawn only when the view leaves it or it's more than 4x larger than needed, and only with the casters that reach it. It's skipped where it wouldn't be much finer (the whole model in view). Frame binding 11 (Vulkan, WebGPU), texture 30 (Metal); camera block `shadowNearViewProj`/`shadowNearParams` (`kCameraUniformSize` 464). No measurable cost while orbiting an office model.
- **Screen-space ambient occlusion** (`setAmbientOcclusion()`, `setAmbientOcclusionRadius()` (world units, 0.5), `setAmbientOcclusionIntensity()` (1.5), `supportsAmbientOcclusion()`; off by default), Metal, Vulkan and WebGPU. Two fullscreen passes after anti-aliasing: the AO from the scene depth at the output size (hemisphere samples around the view-space position, normal from the depth, per-pixel rotated kernel), then a depth-aware 4x4 blur multiplied into the frame by blending, so it works with SSAA, FXAA or neither. The depth texture is now sampleable. Shaders: `ssaoFragment`/`ssaoApplyFragment` (Metal), `ssao.frag`/`ssao_apply.frag` (GLSL), `shaders/webgpu/ssao.wgsl`. Costs about 0.6-3.4 ms on an Intel UHD 630 at 1080x800 depending on coverage.
- **Exposure** (`setExposure(ev)`): colours x 2^EV before tone mapping, meshes and sky (camera block `viewParams.x`, `kCameraUniformSize` 384; skybox uniforms `_pad.x`). 0 by default.
- Test: a floor meeting a wall darkens at the crease with AO and nowhere else, never brightens; -1 EV darkens a lit pixel (Metal, WebGPU, Vulkan).

### Changed
- `bakeIblResources()` records all its draws (about 70 for a 1024 environment) into one command buffer with a uniform slot each and waits once, instead of a submit and wait per draw.
- Metal's shadow pass uses `vertexLean` with light-space transforms like the other backends (`vertexShadow` removed).
- **The shadow map is redrawn only when it changes**: its light, resolution and casters (meshes, position buffers, transforms) are hashed; while only the camera moves the map is reused (an office model: 9 ms of GPU per frame saved). `invalidateShadowMap()` for meshes rewritten in place.
- **Shadow casters don't depend on the camera**: `RenderScene::shadowCasters` (filled by `render_system()` before frustum and size culling, clipped-away geometry excluded) cast, so off-screen geometry shadows what's on screen; without it the opaque draws cast, as before. `MeshRenderer::visible = false` hides a mesh but keeps it casting (for app occlusion culling). Every backend's shadow pass now reads light-space transforms uploaded after the draws' own.
- **Sun shadows** (Metal): the first directional light with `castShadows` (`LightData::castShadows`, from the ECS `DirectionalLight`) renders the opaque triangle draws into a depth map from its direction (orthographic, fitted to their bounds; slope-scaled bias), and `fragmentMain_textured` darkens that light's contribution by 3x3 PCF of the map, so soft-edged shadows with the sky still filling them. `setShadowsEnabled()` (on by default), `setShadowMapResolution()` (2048), `supportsShadows()`. The camera uniforms gain a shadow block (`kCameraUniformSize` 368). Lines, blended and app-layout draws don't cast. Test: a plate over a lit ground plane shadows the ground under it and nothing else.
- **Sun shadows on WebGPU and Vulkan.** The shadow pass draws through the lean vertex stage (`vs_lean`, `default_lean.vert`) with a second set of per-draw transforms whose MVP is the light's view-projection times the model matrix (after the draws' own in the transform buffer), so no new vertex shaders. `fs_main` (WGSL) and `default.frag` (GLSL) do the same 3x3 PCF lookup as Metal; the frame bind group gains the shadow map (binding 9, depth) and a comparison sampler (binding 10). Vulkan needs campello_gpu 0.28.1 (depth-only render passes sized to their depth attachment, sampled depth attachments made shader-readable, copy row pitch, no anisotropy with nearest filters); the pin is bumped. Verified: the shadow test passes in headless Chrome (WebGPU) and on Mesa lavapipe (Vulkan); `run_wgsl_check.sh` layouts updated.
- **Shadow receiver bias.** The lookup offsets the surface along its normal by one shadow-map texel in world units (more at grazing light) and compares one texel of depth closer, which removes self-shadowing ("acne") on large scenes where the rasterizer bias alone isn't enough (a 35 m house: 2 cm texels).

### Fixed
- **Transmission showed the background brighter and washed out**: the opaque scene behind transmissive materials is stored display-encoded (tone mapped, sRGB) and was used as linear light, then encoded again; it's now decoded first (Metal, Vulkan, WebGPU).
- **Transmission with SSAA sampled the wrong place** (the sky fallback instead of the scene): the camera block's `screenSize` was the output size while fragment positions are at the SSAA size; it's now the scene's render size. Test: thin glass over a background at SSAA 2x shows it at its brightness (fails with either bug).
- **Vulkan/WebGPU materials after the material uniform buffer grows** read the old buffer: their bind groups hold the buffer itself (Metal binds the current one per draw). Bind groups holding it are now rebuilt against the new buffer when it grows. Found when a test created materials into an empty renderer (a material drew another's colour).
- **`uploadMaterial()` with glTFs other than the scene's** returned the scene's material at the same index (and shared its uniform slot, textures and samplers): materials, textures and samplers of other assets are now cached by (asset, index) with their own slots. Found in campello_bim, whose per-material glTFs all drew as its first material. Test: two assets with different materials at index 0.
- Directional lights in `render(RenderScene)` lit nothing: the shaders read a directional light's direction from `position`, which the scene path filled with the light's world position (the origin for most lights, giving a zero vector). It now uploads the direction towards the light there, as the glTF path does.

## [0.15.0] - 2026-11-10

### Added
- **`Renderer::setEnvironmentUpAxis(UpAxis::z)`** for Z-up worlds (CAD, BIM, IFC). Equirectangular environments (`loadEquirectangularEnvironmentMap()` and the built-in default) are converted with their zenith towards +Z instead of +Y, so the horizon is level; the built-in map is rebuilt (and IBL rebaked) if it's already in use. Cube faces given directly are unchanged. Default `UpAxis::y`, as before. Found in campello_bim, whose sky was turned 90 degrees. Test: the default camera (looking along -Z) sees the horizon with Y up and only ground with Z up.

## [0.14.0] - 2026-11-10

### Added
- **WebGPU backend for mesh rendering** (programmable pipelines phase 7, first part). Web builds (`__EMSCRIPTEN__`) take the Vulkan bind-group code paths (`CAMPELLO_RENDERER_VK_BINDINGS`) with hand-written WGSL shaders (`shaders/webgpu/default.wgsl`, embedded by `build_webgpu_shaders.sh`): `vs_main`, `vs_lean`, `vs_line`, `fs_main` with the glTF core textures, alpha mask/blend (real blending), unlit, IBL, punctual lights, clip planes and section caps, lean normals and lines. Within WebGPU's 16 textures per stage the KHR extension textures, transmission/volume, sheen, clearcoat, iridescence and anisotropy are left out (material bind groups drop bindings 10-23). WebGPU vertex inputs stop at location 15, so per-draw transforms are a read-only storage buffer (group 2) indexed by `firstInstance`; draws pass it on every backend (0 elsewhere). App programs: `ShaderProgramDesc::wgsl` with `wgslVertexEntryPoint`/`wgslFragmentEntryPoint`; `program_interface.wgsl` documents the contract. Not on the web yet: skybox, FXAA/SSAA, IBL baking. `test/webgpu/run_wgsl_check.sh` validates the WGSL against these layouts and render-checks it in headless Chrome (11 checks; no Emscripten needed). The test suite builds with Emscripten and runs in headless Chrome (`test/webgpu/run_web_tests.sh`): 86 of 88 pass, the other two (KHR_mesh_quantization layouts, morph targets) are Metal-only and skip elsewhere. `campello_renderer_add_shader_program()` takes a `WGSL` source (embedded as text); the element-id test program has one. Uses campello_gpu 0.28.0 (WebGPU buffers of any size), vector_math 0.6.1 and gltf 0.5.2 (Emscripten fixes).
- **Lines** (programmable pipelines phase 6). `GpuMesh::topology` (lineList, lineStrip, ...) is part of the pipeline key; line meshes are drawn by the built-in program with a line vertex stage (`vertexLine` on Metal, `default_line.vert` on Vulkan; not DirectX yet, see `supportsLines()`) that moves them slightly towards the camera in clip space, so edges drawn on their own faces stay visible (depth bias doesn't apply to lines). One pixel wide, no culling, clip planes apply; use an unlit material (`KHR_materials_unlit`) for a flat colour. Uses campello_gpu 0.28.0 (Metal primitive types from the pipeline topology). Test: a line over a coplanar quad is drawn along its whole length (fully hidden without the offset).
- **Slang shader programs** (programmable pipelines phase 5). `campello_renderer_add_shader_program(SLANG file.slang ...)` compiles one Slang source with slangc (`CAMPELLO_SLANGC`, or `$SLANG_DIR/bin`) to Metal, SPIR-V (`-fvk-invert-y`), HLSL (then dxc) and WGSL (embedded as `ShaderProgramDesc::wgsl` for the future WebGPU path); without slangc it falls back to the per-backend sources. New interface header `program_interface.slang` (`CAMPELLO_INPUT`, `CAMPELLO_TRANSFORM_INPUTS`, `CAMPELLO_CLIP`). The element-id test program is now one `.slang` file (Metal and GLSL sources kept as the fallback). Spike findings and the PBR plan in `docs/design/programmable-pipelines.md`.
- **Clip planes and section caps** (programmable pipelines phase 4). `Renderer::setClipPlanes()` takes up to 6 world-space planes; the built-in PBR shader (Metal, Vulkan) discards fragments on their clipped side. While planes are set, built-in materials draw without back-face culling, and with `setSectionCaps()` (on by default) back faces of single-sided materials, i.e. the inside of cut solids, are painted in the cap colour, so cuts look solid. `render_system()` skips draws whose bounds lie entirely on a clipped side (`isClippedAway()` in `culling.hpp`, exact for affine transforms). `supportsClipPlanes()` is false on DirectX until `default.hlsl` reads the clip block. The camera uniform block grows from 160 to 288 bytes (Metal, Vulkan); material uniforms carry the double-sided flag at offset 88. Tests: clipped half discarded, cap colour on a single-sided back face but not a double-sided one, box-vs-plane culling.
- **App shader programs** (programmable pipelines phase 3, new header `program.hpp`). `Renderer::registerProgram(ShaderProgramDesc)` loads an app's compiled shaders (Metal library, SPIR-V, DXIL per backend) and returns a `ProgramHandle`; the built-in PBR shaders are `kPbrProgram`. `GpuMaterial::program` selects a material's program; `renderProgramPass(scene, target, ProgramPassOptions)` draws a whole scene with one program into any target format (no skybox, anti-aliasing or transmission, own depth buffer), e.g. element ids into an `r32uint` texture for GPU picking. `supportsProgram()` tells whether a layout can be drawn by a program. Shader contract in `program.hpp`, with interface headers per language in `inc/campello_renderer/shaders/` (`program_interface.metal.h`, `.glsl`, `.hlsli`; `.wgsl` reserved).
- **`campello_renderer_add_shader_program()`** (CMake, `cmake/CampelloRendererShaders.cmake`): compiles an app program with the available compilers (xcrun metal, glslc or glslangValidator, dxc) and embeds the binaries in a generated header with a ready `ShaderProgramDesc`.
- **Custom vertex semantics** `VertexSemantic::custom0`-`custom3` (locations 8-11) for app data such as element ids; the built-in shaders ignore them. `GpuMesh::vertexBuffers` has one slot per semantic (was 6), and the draw path binds a mesh's own COLOR_0/TEXCOORD_1 when present.
- Test: an element-id program draws two overlapping quads into an `r32uint` target (ids, depth order and clear value read back).
- **Frustum culling in `render_system()` (ECS path).** `GpuMesh` gets optional local-space bounds (`hasBounds`, `boundsMin`/`boundsMax`, `setBoundsFromPositions()`); draws whose bounds lie entirely outside the camera frustum are skipped. Meshes without bounds are never culled, so existing users see no change; skinned meshes are never culled (bind-pose bounds). `uploadMesh()` fills the bounds from a float POSITION accessor's min/max. The test is `isOutsideFrustum()` in the new header `culling.hpp` (no campello_core dependency), run in the mesh's local space so it stays exact under rotation and scale; the near plane is conservative. `render_system()` takes an optional `RenderSystemStats*` reporting drawn and culled draws.
- **Vertex layouts and a mesh pipeline cache** (phase 1 of `docs/design/programmable-pipelines.md`). `VertexLayout` (new header `vertex_layout.hpp`) describes a mesh's attributes and formats; `GpuMesh::layout` points at one (nullptr: the standard glTF/PBR layout, `Renderer::standardVertexLayout()`). Mesh pipelines are no longer 13 hardcoded members: `createDefaultPipelines()` fills a per-backend recipe (vertex stages with the attributes each reads, fragment entry points, blend support), and pipelines are built per (program, layout, culling, blending) on first use and cached; the combinations the built-in paths use are created up front. `Renderer::supportsVertexLayout()` tells whether a layout can be drawn; `meshPipelineCount()` reports the cache size. Output is pixel-identical (campello_bim office model).
- **Position-only meshes (`VertexLayout::positionOnly()`), Metal and Vulkan.** Meshes with only a POSITION stream are drawn with a lean vertex stage: `vertexLean` reads one 12-byte stream instead of eight, and `fragmentMain_textured` derives the flat face normal from screen-space derivatives of the world position (turned towards the camera) when the vertex normal is zero. Vertices can then be shared between faces of different orientation (several times fewer for flat-shaded CAD/BIM geometry). On Vulkan, `default_lean.vert` plays the role of `vertexLean` and `default.frag` derives the normal the same way. `supportsVertexLayout(VertexLayout::positionOnly())` reports whether the backend has them; DirectX doesn't yet.
- `build_vulkan_shaders.sh` falls back to `glslangValidator -V` when `glslc` isn't installed.
- **Front-to-back opaque order (`Renderer::setOpaqueOrder(OpaqueOrder::frontToBack)`).** Opaque draws are kept grouped by pipeline class (single/double-sided) and drawn nearest first (camera to the world-space centre of the mesh bounds), so early depth testing skips hidden fragments on GPUs that shade in submission order. The default stays `byState` (material, then mesh). In campello_bim's office model on an Intel UHD 630, GPU time zoomed in went from 23 to 16 ms (FXAA) and from 72 to 46 ms (SSAA 2x), with an unchanged image.
- **`RenderStats::gpuFrameTimeMs`.** GPU execution time of the most recent completed `render()` (frames in flight: usually two or three frames old), read from the command buffer once its fence has signalled. Reported on Metal; 0 where the backend doesn't report it.
- Design proposal: `docs/design/programmable-pipelines.md` (app-defined vertex layouts and shader programs).
- **Size culling in `render_system()`.** With `RenderSystemOptions{minPixelRadius, viewportHeight}`, draws whose bounding sphere covers fewer pixels than that on screen are skipped (counted in `RenderSystemStats::tooSmall`). The estimate is `projectedRadiusPixels()` in `culling.hpp`; meshes without bounds and spheres reaching the camera are never "small". Found in campello_bim, whose IFC reference models have thousands of batches.

### Fixed
- **Materials uploaded outside the asset lost their data on DirectX.** `uploadMaterial()` for a material not in the asset takes a slot from 10000 up in the shared material uniform buffer; growing the buffer for a later material allocated a new one without copying the old contents, and DirectX rebuilds its material bind groups every pass from the current buffer, so earlier materials rendered with zeroed uniforms (e.g. no emissive colour). The buffer now grows geometrically and is refilled from a CPU copy of the written slots. New test: the first of 41 standalone materials still renders with its own colour.
- **GPU frame time on the web without blocking.** `RenderStats::gpuFrameTimeMs` came from `CommandBuffer::getGPUExecutionTime()`, which on WebGPU yields (ASYNCIFY) until the timestamp readback resolves: inside a browser frame callback that ended the app's frame loop (campello_bim on the web stopped after its first frame). Web builds now ask with `getGPUExecutionTimeAsync()` at submit and report the latest result.
- **Morph targets on Metal.** The vertex shader read the morph-target deltas as `device const float3*` (16-byte stride) while they are uploaded tightly packed (12 bytes per vertex), so only the first vertex of each primitive got its delta (the rest read neighbouring components or past the end). They are now `packed_float3`. New test: a morph target moves every vertex.

### Changed
- **One transform contract on every backend.** On Metal, the per-draw transforms are now also per-instance vertex attributes (locations 16-19 MVP, 24-27 model, in the slot after the attributes), as on Vulkan and DirectX, so one program source can serve every backend (programmable pipelines phase 5, Slang). The built-in Metal shaders keep reading `[[buffer(16)]]`; both are bound. `program_interface.metal.h` gains `CAMPELLO_TRANSFORM_ATTRIBUTES` and `CAMPELLO_MVP()`/`CAMPELLO_MODEL()`; the element-id test program uses them.
- **Compact vertex buffer slots** (programmable pipelines phase 2). A pipeline's attributes take slots 0..n-1 in semantic order, then the per-draw transforms (Vulkan/DirectX); each cached pipeline carries a map from logical slot (`VERTEX_SLOT_*`) to physical slot, and draw code binds through it (buffers the pipeline doesn't read are no longer bound). The standard Vulkan/DirectX pipeline uses 7 vertex buffers instead of 18 (WebGPU allows 8). No shader changes; Metal is unchanged.
- **KHR_mesh_quantization through vertex layouts.** The separate quantized pipeline set (Metal) is replaced by a quantized `VertexLayout` (integer POSITION/NORMAL/TEXCOORD_0/TANGENT, normalized) drawn through the pipeline cache. `VertexAttributeFormat::stride()` rounds up to 4 bytes. New test: quantized positions render like float positions.
- **campello_gpu v0.28.0** (was v0.26.0). On Metal, pipelines now honour `cullMode`/`frontFace` (0.27.1), so the opaque pipelines (`back`, `ccw`) cull back faces on macOS and iOS as they already did on Vulkan and DirectX; double-sided materials keep `none`. 0.28.0 adds the vertex formats, Metal topology and WebGPU buffer fixes this release relies on.
- **vector_math v0.6.1** and **gltf v0.5.2** (were v0.6.0 and v0.5.1): Emscripten builds.

## [0.13.2] - 2026-11-08

### Fixed
- **FXAA and SSAA broke after the first resize.** The cached FXAA, SSAA-downsample and refraction-copy bind groups were never rebuilt when `resize()` or `setSsaaScale()` recreated the textures they reference, so they kept sampling the old ones (FXAA showed nothing, SSAA an old frame). Any app whose first frame renders before its final size hit this; enabling AA only after resizing hid it. They are now dropped whenever their textures are recreated or released.
- **SSAA never actually supersampled.** Scene passes kept the output-size viewport, scissor and depth buffer, so the scene was drawn at 1x into the corner of the scaled target, and the downsample shader divided by the scene texture's size, which happened to copy that corner back 1:1. Scene passes now use the scaled target size (`sceneTargetWidth/Height()`), the depth buffer matches it, the skybox gets it, and the downsample samples with output-normalized UVs from `fxaaVertex` (Metal, Vulkan and DirectX).
- **Metal FXAA produced a flat frame.** It sampled with pixel positions instead of normalized UVs, so every tap clamped to an edge texel. It now uses the vertex UV (the Vulkan and DirectX FXAA were already correct).
- Shader headers regenerated: Metal (xcrun), Vulkan (glslc) and DirectX (dxc).

## [0.13.1] - 2026-11-08

### Fixed
- **`render(const RenderScene&)` drew opaque meshes with the wrong transforms.** Per-draw transforms were uploaded in `scene.opaque` order, but the draws were then sorted by material and mesh, and each sorted draw read the transform at its new position. Any scene where the sort reordered draws placed meshes at other draws' transforms. Draws are now sorted before their transforms are uploaded. Found in campello_bim, whose meshes each have their own world transform.
- **Out-of-bounds read for instanced ECS draws.** `renderPrimitive()` read `GpuMesh::vertexBuffers[VERTEX_SLOT_INSTANCE_MATRIX]` (index 19) from the 6-element per-vertex array when `instanceCount > 1`. It now uses the default instance matrix unless the array actually has that slot.

### Changed
- **campello_gpu v0.26.0.** Brings the Metal readback fix (`Texture::download()` no longer aborts under Metal API Validation) and the emdawnwebgpu WebGPU backend; no API changes for the Metal, Vulkan or DirectX 12 paths.
- **The public `inc/` directory is now a `PUBLIC` include directory of the `campello_renderer` target**, so consumers using `FetchContent`/`add_subdirectory` no longer need to add it by hand.

## [0.13.0] - 2026-11-05

### Added
- **UWP support, including Xbox in Developer Mode.** The library builds with `CMAKE_SYSTEM_NAME=WindowsStore`. The new `examples/uwp` renders DamagedHelmet with an HDR environment on a Windows PC or an Xbox, with gamepad controls, logs in `LocalState`, and a 4K output mode in Game mode on Xbox. On an Xbox Series X it holds 60 fps at true 4K, at about 2 ms of GPU time per frame.
- **`Renderer::setOverlayCallback()`** lets an app draw on top of the finished frame, in its own render pass, just before `render()` submits it. `examples/uwp` uses it to show the output size, frame rate and GPU time.
- **`examples/xbox_probe`**, a standalone UWP app that reports what a console's D3D12 supports. It found that UWP on Xbox Series X only exposes feature level 11_0.

### Fixed
- **DirectX: triangle edges appeared as light and dark lines in lit output on Xbox Series X (AMD).** An Intel GPU didn't show this. The early returns and `discard` in `pixelMain` now come after the material texture samples, which removes the lines. The likely cause is `Sample()` choosing wrong mips at triangle edges, though that wasn't measured directly. The Metal and Vulkan shaders were not changed.
- **DirectX: two more implicit-LOD samples in the transmission path could be affected the same way.** `thicknessTexture` is now sampled with the other material textures, and the `environmentMap` fallback uses an explicit LOD of 0. No transmissive asset was tested on Xbox.
- **DirectX: `directx_default.h` was out of date.** It's now regenerated from the current HLSL, so the 0.11.2 fix that forces alpha to 1.0 for OPAQUE and MASK materials now takes effect on DirectX.

### Changed
- Requires `campello_gpu` 0.25.0 and `campello_image` 0.5.2.

## [0.12.1] - 2026-09-05

### Fixed
- **`buildCameraData()` (`inc/campello_renderer/ecs.hpp`) produced a view matrix with the wrong Z-forward convention, breaking any consumer that rendered from a scene's own `Camera` entity.** `Transform.worldMatrix` uses the standard "-Z is forward" convention (the same one `buildLightData()` already assumes when it negates `worldMatrix`'s Z column to get a light's direction), but `buildCameraData()` computed `view = transform.worldMatrix.inverted()` as a plain matrix inverse, which carries that same -Z-forward convention into view space. `Matrix4::perspective()` (and `Matrix4::rotateX/Y/Z()`) instead assume a view space where **+Z** is in front of the camera — `OrbitCamera::getViewMatrix()` already compensates for this internally (see its own "row2 = -forward" comment), which is why edit-mode/tool-camera rendering was never affected. Any renderer using an actual `Camera` entity (e.g. an editor's Play-mode preview, or any consumer relying on `render_system()`'s built-in "first Camera entity" query) got a view matrix with the opposite convention: points in front of the camera produced a **negative** clip-space `w`, sending them outside the visible NDC range and making the entire scene disappear even though the camera, meshes, and matrices were all otherwise correct. Fixed by negating the view matrix's Z row (`data[8..11]`) in `buildCameraData()`, mirroring `OrbitCamera`'s own compensation. Verified by hand-computing clip-space coordinates for a known camera/mesh pair before and after the fix (`clip.w` flipped from -6.31 to +6.31; `ndc.z` moved from 1.016, just past the far-clip boundary, to 0.984, correctly inside `[0,1]`) and confirmed visually in a real consumer (campello_editor's Play-mode viewport).

## [0.12.0] - 2026-09-05

### Added
- **`render_system()` (`inc/campello_renderer/ecs.hpp`) accepts an optional `const CameraData* cameraOverride`** — when non-null, it's used verbatim as the scene camera and the world's own `Transform`+`Camera` entity query is skipped entirely. Lets a consumer with two distinct camera needs on the same `World` — e.g. an editor viewport driven by a free-floating tool camera while editing versus the scene's own `Camera` entity once actually playing — pick per call, without maintaining a separate render path just for the edit-mode case. Passing `nullptr` (the default) preserves the exact existing entity-driven/built-in-default-fallback behavior.

### Fixed
- **`ecs.hpp` had never actually been compiled** — this header-only "optional ECS bridge" ships in the package but nothing in this repo (or, as far as could be determined, any known consumer) includes or exercises it, so three real bugs sat undetected:
  - `animation_system()`/`material_animation_sync_system()` destructured `world.query<GltfAnimation>()` (a single-component query) into two names (`[entity, anim]`) — a single-component query only yields the component, not the entity, so this failed to compile at all. Neither loop body actually used the entity; fixed by destructuring to `[anim]`.
  - `skinning_system()` declared `using GPU = systems::leal::campello_gpu;` — `using` aliases a *type*, not a namespace; the correct form is `namespace GPU = ...`. Fixed, which also resolved a knock-on "undeclared identifier GPU" a few lines later.
  - `skinning_system()` called `Buffer::getSize()`, which doesn't exist on `campello_gpu::Buffer` — the real accessor is `getLength()`. Fixed.
  - Verified via a standalone `-fsyntax-only` compile against the real `campello_core`/`campello_gpu`/`vector_math`/`gltf` headers (this repo's own build/test suite doesn't touch `ecs.hpp`, so this is the only way it's currently validated) — clean, zero errors, after the fix. No regression test added yet since no example/test in this repo currently exercises the ECS bridge at all; tracked as follow-up work for whoever adopts it.

## [0.11.2] - 2026-09-05

### Fixed
- **`alphaMode: OPAQUE`/`MASK` materials wrote a non-1.0 alpha into the color target** — `fragmentMain_flat`/`fragmentMain_textured` (Metal), the combined PBR fragment shader (Vulkan/GLSL), and the DirectX HLSL equivalent all returned `baseColor.a` verbatim as the framebuffer alpha, regardless of the material's `alphaMode`. Per the glTF spec, `OPAQUE` (the default) and `MASK` must render fully opaque — alpha is either ignored entirely or only relevant for the cutoff test already applied earlier in the shader. A consumer that presents this renderer's output directly to a window's opaque drawable never noticed (the OS compositor discards alpha there), but a consumer that composites the render target as a 2D texture (e.g. an editor viewport backed by an offscreen render target, displayed via an alpha-aware image widget) could see an "opaque" material's real color rendered fully or partially transparent wherever its albedo texture's alpha channel wasn't a clean 1.0 — common for authored PNG/JPEG textures with unused/garbage alpha. Fixed by forcing alpha to 1.0 for `alphaMode < 1.5` (OPAQUE/MASK) in all three shader backends; only `BLEND` (2) still uses the real `baseColor.a`.
  - **Known gap**: the Metal (`src/shaders/metal_default.h`) and Vulkan (`src/shaders/vulkan_default.h`) embedded binaries were regenerated and are current as of this release. The DirectX embedded binary (`src/shaders/directx_default.h`) was **not** regenerated — this release only has the fixed HLSL source (`shaders/directx/default.hlsl`); no Windows/`dxc.exe` toolchain was available to rebuild it. Anyone consuming the DirectX backend should run `build_directx_shaders.ps1` on Windows and commit the regenerated header before this fix takes effect there.

## [0.11.1] - 2026-09-04

### Changed
- Upgraded `campello_gpu` dependency from `v0.23.1` to `v0.24.1`:
  - **MSAA support** added across the Metal, Vulkan, and DirectX backends (`RenderPipelineDescriptor.sampleCount`, `resolveTarget`) — not yet wired into `campello_renderer`'s own pipeline creation, so no behavior change here, but now available for future use.
  - Fixed `ShaderStage` missing bitwise operators (`operator|`, `operator&`, `operator|=`) — `campello_renderer` currently only ever assigns a single `ShaderStage` value per bind group entry (never combines vertex+fragment visibility), so this doesn't change existing behavior, but unblocks doing so later.
  - Various Metal/Vulkan/DirectX correctness fixes not otherwise exercised by this library (MSAA texture storage mode, stencil state wiring, DirectX descriptor-heap leaks/contiguity, swapchain resize) — see [campello_gpu's changelog](https://github.com/rusoleal/campello_gpu/blob/v0.24.1/CHANGELOG.md) for the full list.

## [0.11.0] - 2026-09-01

### Fixed
- **`Renderer::uploadMaterial()` never handled `KHR_texture_basisu`** — the lazy per-material texture path (used by `MeshAssetCache` and any other caller resolving one material at a time, as opposed to the scene-wide upload loop in `setScene()`) always fell through to the plain `campello_image::Image::fromMemory()` branch, which cannot parse KTX2/`.basis` bytes at all; any material referencing a `KHR_texture_basisu` texture silently failed to upload. `ensureTexture()` now takes an `isBasisu` flag and, when set, uses the same `campello_image::TextureData`-based GPU-compressed decode+upload path (`chooseBasisTargetFormat()` / `textureDataFormatToPixelFormat()` / `uploadTextureDataWithMips()`) the scene-wide loader already used, instead of duplicating it. `getTextureAndSampler()`'s image-index resolution now checks `gt.khr_texture_basisu` before falling back to `ext_texture_webp`/`source` — it previously skipped basisu entirely. Not independently visually verified against a real KTX2/basisu-textured glTF asset (none available locally); this is code-parity with the already-proven-correct scene-wide path, confirmed by a clean build, not an isolated regression test.

### Changed
- Upgraded `campello_image` dependency from `v0.5.0` to `v0.5.1` (basis_universal transcoder upgraded `1.16.4` → `v2_50`; see that project's changelog).
- `dependencies/campello_image.cmake` no longer supports a local-checkout override — always fetches from GitHub via `FetchContent`, removing the same footgun already removed from `dependencies/campello_gpu.cmake` in 0.10.0 (a developer's local `campello_image` checkout could silently diverge from the pinned released version).

## [0.10.0] - 2026-08-19

### Added
- **Full glTF PBR pipeline ported to DirectX 12 (Windows)** — metallic-roughness Cook-Torrance GGX BRDF, image-based lighting (GGX-prefiltered specular + Lambertian diffuse irradiance + BRDF LUT), normal mapping, punctual lights (`KHR_lights_punctual`), and `KHR_materials_specular`/`anisotropy`/`iridescence`/`clearcoat`/`sheen`/`transmission`, matching the existing Metal/Vulkan feature set. Windows rendering (previously non-functional — DXIL binaries weren't compiled) now fully works, including resize.
- **DirectX combined per-(material, frame-in-flight) bind group scheme** (`Renderer::ensureDirectXPbrBindGroupLayout()` / `buildDirectXCombinedBindGroup()` / `rebuildDirectXCombinedBindGroups()`) — works around a `campello_gpu` D3D12 root-signature bug where a second `setBindGroup()` call landed on the wrong descriptor table, by combining all per-material and per-frame PBR resources (textures, samplers, lights, camera, environment/IBL, scene-color) into one bind group per (material, frame-in-flight) pair instead of Vulkan/Metal's material+frame split.
- New Windows example app (`examples/windows/`, GLFW-based) — drag-and-drop glTF/HDR loading, orbit camera, animation playback, and the same view-mode/hotkey set as the macOS/Linux examples. Enabled via `-DCAMPELLO_RENDERER_BUILD_WINDOWS_EXAMPLE=ON`.
- `build_directx_shaders.ps1` — compiles `shaders/directx/default.hlsl` to DXIL via `dxc.exe` and regenerates the embedded `src/shaders/directx_default.h` header, mirroring `build_metal_shaders.sh` / `build_vulkan_shaders.sh`.
- `WINDOWS_EXPORT_ALL_SYMBOLS` enabled for the Windows DLL build — without it, MSVC's linker built the `.dll` but never emitted the companion import `.lib`, failing every consumer (tests, example apps) at link time with `LNK1104`.

### Changed
- Upgraded `campello_gpu` dependency from v0.23.0 to v0.23.1 (Windows/DirectX descriptor-heap contiguity, DSV/sampler-heap leaks, swapchain resize, and cube-texture fixes).
- `dependencies/campello_gpu.cmake` no longer supports a local-checkout override — always fetches from GitHub via `FetchContent`, removing a footgun where a developer's local `campello_gpu` checkout could silently diverge from the pinned released version.
- Upgraded `gltf` dependency from v0.5.0 to v0.5.1, and added an explicit `dependencies/vector_math.cmake` (declared before `gltf.cmake`) pinning `vector_math` v0.6.0 directly — both now agree, instead of relying on `gltf`'s older transitive pin (previously v0.3.5).

### Fixed
- **Mirrored default camera** — `render(view)`'s and the ECS `render_system()`'s built-in fallback camera (used whenever no glTF/ECS camera exists) rendered a left-right-mirrored image and incorrectly backface-culled single-sided, CCW-front geometry near the origin (including `OffscreenRenderTest.ECSPathRendersToOffscreen`, which builds its own camera the same way as an external ECS caller would). Root cause: `vector_math::Matrix4::lookAt()` built its right/up axes with a cross-product argument order that's mirrored relative to what `Matrix4::perspective()`'s +Z-forward convention expects — confirmed empirically (geometry translated to world +X rendered on the left half of the frame instead of the right). Same root cause as the long-disabled `DISABLED_MeshRendersNonClearPixels` and sibling Vulkan tests. Fixed upstream in `vector_math` v0.6.0 (see that project's changelog) and picked up here via the `vector_math` upgrade above; real cameras (glTF-embedded, ECS `Transform`-driven, both of which use `Matrix4::inverted()` instead of `lookAt()`) were never affected.
- **DirectX offscreen-render test crashes** — `BasicOffscreenRenderDoesNotCrash` / `MultipleConsecutiveRenders` / `ResizeOffscreenTarget` / `DifferentPixelFormatBGRA8` released their local render-target texture before the GPU finished executing the render into it (`render()`'s `submit()` is async and doesn't block), which D3D12's debug layer treats as fatal resource corruption. Fixed by adding `device->waitForIdle()` before the texture goes out of scope, matching the pattern `readBackPixels()` already used.
- **`Renderer::resize()` flaky DirectX crash on real usage (not just tests)** — reassigning `depthTexture` (and, via `ensureSceneColorTexture()`, `sceneColorTexture`) drops the previous size's texture, destroying it as soon as its refcount hits zero. With no wait beforehand, calling `resize()` shortly after `render()` (e.g. a live window resize, which is exactly what `OffscreenRenderTest.ResizeOffscreenTarget` does) could release a texture the GPU was still executing against — the same class of bug as the offscreen-test crashes above, but reproducible with the *library's own* public API, not just test code. Observed as an intermittent (~2/7 runs) `ClearRenderTargetView`-adjacent SEH corruption when re-running the test suite repeatedly; not caught by a single run, which is why enabling Windows CI tests (below) surfaced it. Fixed by adding `device->waitForIdle()` at the top of `resize()`, mirroring `setAsset()`'s identical guard. Verified with 17 consecutive clean `ctest` runs after the fix (2 failures were reproduced in 7 runs before it).

### CI
- **Re-enabled the Windows test step**, disabled since it was first added, citing the `WINDOWS_EXPORT_ALL_SYMBOLS` DLL-export bug fixed earlier in this same release. Also added a POST_BUILD DLL-copy step for the `campello_renderer_test` target itself (mirroring the one `examples/windows/CMakeLists.txt` already had) — without it, `ctest`/the test binary fails with `STATUS_DLL_NOT_FOUND`, since `cmake --build` only refreshes DLLs inside their own dependency build directories, not next to every consumer executable.

## [0.9.0] - 2026-08-13

### Added
- **Vulkan IBL precompute pipeline** (`Renderer::bakeIblResources()`) — bakes the BRDF LUT, a GGX-prefiltered specular cubemap, and a Lambertian diffuse irradiance cubemap from the current environment map via a new fullscreen-triangle shader (`shaders/vulkan/ibl_bake.frag`), matching the reference glTF-Sample-Renderer's IBL approach that was already implemented for Metal. The PBR fragment shader now samples these baked resources (frame bind group bindings 6/7/8) instead of the raw environment map, falling back to it when a bake hasn't run yet.
- **Vulkan skybox rendering** — new `shaders/vulkan/skybox.frag` plus `pipelineSkybox` wiring; `drawSkybox()` was already shared by both render paths and only needed the Vulkan pipeline to light up.
- `Renderer::getBoundsCenter()` — world-space center of the current scene's geometry bounding box (via new `computeSceneAABB()`), so camera code can orbit the scene's actual visual center instead of assuming the origin. Wired into the Linux and macOS example apps' `Camera::fitBounds()`.
- `Renderer::~Renderer()` — explicit destructor that waits for all in-flight GPU frames to finish before member buffers/textures/bind groups are released.

### Changed
- `skyboxEnabled` now defaults to `true` (previously `false`).
- Default embedded environment map replaced with "Cannon Exterior" by Greg Zaal — the Khronos glTF-Sample-Viewer's own default HDRI — instead of "Kiara 5 Noon", so out-of-the-box IBL brightness is directly comparable to the reference viewer.
- `sceneColorTexture`/`sceneColorView` (FXAA/SSAA) and `opaqueSceneTexture`/`opaqueSceneView` (screen-space refraction) are now per-frame-in-flight arrays instead of single shared instances, fixing a race where one frame's write could stomp another still-in-flight frame's read of the same texture — visible as corrupted/torn output on animated scenes using FXAA/SSAA or `KHR_materials_transmission`.
- Upgraded `campello_gpu` dependency from v0.21.1 to v0.23.0.

### Fixed
- `OffscreenRenderTest.ClearColorIsApplied` now explicitly disables the skybox, since `skyboxEnabled` defaulting to `true` would otherwise paint over the clear color under test.

## [0.8.0] - 2026-08-02

### Added
- **Sparse accessor support** — `resolveAccessorBytes()` resolves any vertex attribute accessor's base bufferView data (or an implicit zero base if absent) with `accessor.sparse` indices/values overlaid on top. Applied to the deinterleaving pass, COLOR_0 conversion, and morph target deltas. Scoped to vertex attributes only — sparse `primitive.indices` is not handled (rare in practice).
- **Morph target support** — POSITION and NORMAL blending (TANGENT not implemented) for up to 8 targets per primitive:
  - `GltfAnimator` gained an `animatedWeights` map + `getAnimatedWeights()`, and `sampleAnimation()` now handles the `"weights"` animation channel path (linear and step interpolation)
  - `Renderer::updateMorphWeights()` resolves weight precedence (`node.weights` > `mesh.weights` > animated override) into a per-node `MorphInfo` uniform each frame
  - New vertex buffer slots 21–23 (`MorphInfo`, position deltas, normal deltas), bound in both the primary glTF-driven `renderPrimitive()` path and the ECS `drawDrawCall()` path (the latter always binds the zero-weight fallback, since standalone `uploadMesh()` meshes don't carry morph data)
  - Metal shader blends targets before skinning, matching glTF's evaluation order
- **COLOR_0 and multi-UV (`texCoord`) support** — `COLOR_0` (VEC3/VEC4, float or normalized ubyte/ushort) is normalized to a canonical float4 buffer and multiplied into `baseColor` per spec; every texture reference now respects its own `texCoord` index (0 or 1) via a per-material bitmask and `selectUV()` in the shader, instead of always sampling `TEXCOORD_0`. New vertex slots 6 (COLOR_0) and 7 (TEXCOORD_1).
- **KHR_materials_iridescence rewritten** to the actual Khronos reference algorithm (multi-bounce Airy summation with CIE XYZ sensitivity curves), replacing a naive 3-wavelength cosine approximation.
- **KHR_materials_emissive_strength** — `emissiveFactor` is now multiplied by the extension's strength before upload (was previously ignored).
- **On-demand rendering** in the macOS example — MTKView's continuous CVDisplayLink drives a per-tick dirty-flag gate (`requestRedraw`/`_needsRedraw`) instead of manual `setNeedsDisplay:` invalidation, avoiding cross-display vsync races while still skipping GPU work for static scenes.
- Embedded default environment map replaced with a genuine 1024×512 Radiance HDR (was an 8-bit tonemapped JPEG that hard-capped highlights at 1.0), with exposure correction (`kBuiltinEnvironmentExposure`) calibrated against the source HDRI's real radiance values; built-in cubemap bake resolution raised from 128px to 512px per face.

### Changed
- Upgraded `gltf` dependency from v0.4.2 to v0.5.0.
- Upgraded `campello_gpu` dependency from v0.21.0 to v0.21.1.
- Fixed stale `campello_gpu.cmake` / `campello_image.cmake` local-path overrides pointing at nonexistent `/Users/rubenleal/Projects/...` directories (silently always falling back to GitHub fetch for local dev builds); now point at the actual local checkouts.

### Fixed
- **Animation delta-time bug (macOS example)** — `drawInMTKView:` was advancing the animation clock by a hardcoded 1/60s every call regardless of actual elapsed time, so any frame slower than 16.7ms (heavy scenes, the per-frame `waitForIdle()`, GPU contention) silently ran animations in slow motion. Now measures real elapsed time via `CACurrentMediaTime()`, captured every display-link tick (so idle/static periods don't accumulate into a catch-up jump when animation resumes) and clamped to 0.25s for genuine stalls.
- **MaterialUniforms iridescence offset bug** — a phantom padding float inserted before `iridescenceFactor` (based on a false assumption that Metal needed alignment padding between two plain floats) shifted every field from `iridescenceFactor` through `dispersion` one float late relative to the compiled Metal struct layout, silently disabling `KHR_materials_iridescence` for every asset using it.
- **Transparent draw-order sort** — used the node's own local translation as a distance proxy, which ties (and falls to undefined `std::sort` order) for coincident "shell" meshes sharing a parent with zero local offset; now uses `nodeWorldBounds[nodeIndex]`'s center.
- **Metallic materials routed to the flat pipeline** — materials with `metallicFactor > 0` but no textures (e.g. `metallicFactor` defaulting to 1.0 when omitted) were rendered with zero reflections; now routed to the textured/IBL pipeline like other reflective materials.
- **UNSIGNED_BYTE index buffers** — the index-format ternary defaulted any non-`UNSIGNED_SHORT` component type to `uint32`, including the spec-legal but uncommon `UNSIGNED_BYTE`, reading 4× too much data per index from a 1-byte-per-index buffer and producing corrupted geometry. Now widened to `uint16` at scene-load time.
- `fragmentMain_flat` was dropping `emissiveFactor` entirely (materials with near-black `baseColor` + `KHR_materials_emissive_strength` and no textures rendered invisible).
- Skybox fragment shader was missing tonemapping, clipping/saturating raw HDR background values.
- `__weak` local variable in `ViewController.mm` (compiles under MRC, not ARC) caused a hard compile error.

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
