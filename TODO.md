# campello_renderer TODO

## Baseline (v0.7.0, historical)

What worked at v0.7.0. `CHANGELOG.md` lists everything that has landed since (the library is now v0.19.0).

### What works
- `setAsset(shared_ptr<GLTF>)` — loads asset, allocates GPU buffer/texture/sampler/bind-group slot arrays
- `setScene(index)` — uploads GPU buffers, decodes + uploads textures via `campello_image`, builds per-texture bind groups from GLTF samplers, computes scene bounding radius, allocates material uniform buffer and per-node transform buffer
- `setCamera(index)` — selects active GLTF camera
- `resize(w, h)` — creates `depth32float` texture and view
- `createDefaultPipelines(colorFormat)` — compiles flat + textured pipeline variants; Metal (macOS) fully implemented; Vulkan (Android) one variant (flat = textured workaround); DirectX (embedded DXIL, `build_directx_shaders.ps1`)
- `render()` / `render(colorView)` — full scene graph traversal, per-node MVP, camera from GLTF or override, depth pass, POSITION + NORMAL + TEXCOORD_0 + TANGENT binding, fallback UV buffer, per-primitive pipeline selection, indexed + non-indexed draw
- `setCameraMatrices(view, proj)` / `clearCameraOverride()` — external camera override
- `getBoundsRadius()` — scene bounds
- `getGpuBuffer/Texture/BindGroup` accessors
- `getVersion()` — works; test asserts exact match

---

## Immediate (pre-commit)

- [x] **Commit pending changes** — `campello_gpu` v0.9.0, `gltf` v0.4.1, `LICENSE` — PBR textures (metallicRoughness, normal, emissive, occlusion) implemented.
- [x] **Triple-buffer frame-in-flight ring** — 3 `FrameResources` with fences; prevents CPU/GPU desync.
- [x] **Build directory split** — separate `build/macos/debug` and `build/macos/release` to prevent mixed objects.
- [x] **Local `campello_gpu` override** — `dependencies/campello_gpu.cmake` points to `/Users/rubenleal/Projects/campello_gpu` for rapid iteration.
- [x] **Metal validation fixes** — removed `instLayout` from vertex descriptor (fixed missing binding 3/4 crash); added `fallbackTangentBuffer`; fixed vertex stage material buffer binding at slot 17.
- [x] **Depth/stencil format mismatch fix** — patched `campello_gpu` Metal backend to skip stencil attachment when depth format is `depth32float` (no stencil component).
- [x] **Redundant buffer binding deduplication** — `setVertexBufferIfChanged()` tracks last-bound `(buffer, offset)` per slot to avoid Metal debug-layer asserts.
- [x] **Frustum culling** — AABB-vs-frustum plane tests skip whole node subtrees outside the camera view. Fixed row-major matrix plane extraction bug.
- [x] **View mode hotkeys** — 17 debug visualization modes: normal, worldNormal, baseColor, metallic, roughness, occlusion, emissive, alpha, uv0, specularFactor, specularColor, sheenColor, sheenRoughness, clearcoat, clearcoatRoughness, clearcoatNormal, transmission.

---

## Bugs / Upstream Blockers

- [ ] **Thickness texture read from R on Metal and Vulkan** — KHR_materials_volume stores thickness in the G channel (WebGPU since v0.30.0 and DirectX read G). Fix both shaders and rebuild their headers (Metal needs a Mac).
- [ ] **Metal: the file's point lights don't light the quad in `FifthLightInSceneOrderStillLights`** — on macOS CI red and green both read 192 (the ambient level), so none of its five point lights (four dim, one red, a unit in front of the quad on its normal's side) reaches it, while DirectX and WebGPU light it red. The test is skipped on Apple until this is understood; no other test covers punctual lights on Metal.
- [ ] **4 punctual lights on Metal and Vulkan** — `LightsUniform` holds `kMaxLights` = 8 (528 bytes, strongest first) and the DirectX and WebGPU shaders read them all; `default.metal` / `default.frag` (and `rt_effects.metal`) still declare and loop over 4. Raise them to 8 and rebuild the headers (Metal needs a Mac).
- [ ] **Vulkan vertex input not applied** — `campello_gpu` Vulkan backend hardcodes `vertexBindingDescriptionCount = 0`, so vertex buffer layouts passed via `createDefaultPipelines()` are ignored. Fix upstream in `campello_gpu`. Tracked in the Vulkan `createDefaultPipelines` path.
- [ ] **No acceleration structure compacted-size query** in `campello_gpu`, so the path tracer can't compact its BLASes (memory).
- [ ] **Metal compute passes don't bind textures** — `campello_gpu`'s Metal `ComputePassEncoder::setBindGroup()` binds buffers and acceleration structures only. The path tracer's `environmentLuminance` kernel reads the environment cube, so it runs as a ray tracing pass (which binds textures) instead. Fix upstream, then move it back to a compute pipeline.

---

## Asset Loading

- [x] **`data:uri` image support** — The gltf library already decodes data:uri images automatically. The renderer now uses `image.data` directly with `campello_image::Image::fromMemory()`.
- [x] **External image URIs** — `.gltf` files referencing external image files are loaded via `Renderer::imageBasePath` / `setAssetBasePath()`. Both `setAsset()` and `uploadMaterial()` resolve external image paths relative to the base path.
- [x] **Non-standard buffer view strides** — interleaved vertex data (`byteStride > 0` in buffer views) is now deinterleaved into contiguous GPU buffers per accessor at load time. The renderer scans all mesh primitives in `setScene()`, detects accessors in buffer views with `byteStride > 0`, and extracts each attribute into its own tightly-packed GPU buffer. `renderPrimitive()` prefers the deinterleaved buffer when binding standard GLTF attributes.

---

## Shaders

### Vulkan (Android)
- [ ] **Flat fragment shader** — compile a dedicated SPIR-V fragment shader for the flat (no-texture) variant. Currently `pipelineFlat = pipelineTextured` as workaround.
- [ ] **Material slot 17 in GLSL** — verify the `baseColorFactor` uniform at `VERTEX_SLOT_MATERIAL` (slot 17) is read correctly by the current SPIR-V; the Vulkan pipeline descriptor doesn't include slot 17 layout.
- [ ] **Add `glslc` compile step to CMake** — automate SPIR-V compilation for Vulkan targets instead of embedding pre-compiled binaries.

### DirectX (Windows)
- [x] **Embedded DXIL** — `build_directx_shaders.ps1` compiles `shaders/directx/*.hlsl` with dxc (Windows SDK) into `src/shaders/directx_default.h` and `directx_denoiser.h`, one line per entry point.
- [x] **Rebuilding without a Windows machine** — `.github/workflows/directx-shaders.yml` runs the script on `windows-latest` and commits the headers to the branch: on a push to a `dxil/...` branch that changes the HLSL, or by hand (Actions > DirectX shaders > Run workflow). Its own commit doesn't start CI (GitHub's token rule): push another commit or re-run CI to test it.
- [x] **Headers rebuilt for the sky blur (v0.39.0) and per-texture `KHR_texture_transform` (v0.41.0)** — by the workflow's first run (#60); their tests run on Windows again. A change to the HLSL needs the same: otherwise Windows keeps running the old DXIL.
- [ ] **Add `dxc` compile step to CMake** — guard with `if(WIN32)`, so the headers can't fall behind the HLSL.

### Apple (macOS / iOS)
- [x] **iOS SDK compile** — `build_metal_shaders.sh` compiles for macOS, iOS simulator, and iOS device SDKs.
- [x] **Add Metal compile step to CMake** — `metal_shaders` custom target + `add_dependencies(campello_renderer metal_shaders)` auto-rebuilds embedded `.metallib` header when `default.metal` changes.

### Longer term
- [ ] **Single shader source** — evaluate HLSL → MSL + SPIR-V via DXC + SPIRV-Cross to eliminate per-platform divergence.

---

## Animation

- [x] **Implement `update(double dt)`** — GLTF animation channels fully implemented with multi-animation support:
  - Multiple animations can play simultaneously on different node sets
  - `playAnimation(index)` / `pauseAnimation(index)` / `stopAnimation(index)` for per-animation playback control
  - `stopAllAnimations()` to stop all animations at once
  - `setAnimationTime(index, t)` / `getAnimationTime(index)` for per-animation seeking
  - `setAnimationLoop(index, bool)` / `isAnimationLooping(index)` for per-animation loop control
  - `isAnimationPlaying(index)` to check individual animation state
  - `getAnimationCount()` / `getAnimationName(i)` / `getAnimationDuration(i)` for introspection
  - Keyframe sampling with LINEAR interpolation (translation, rotation via slerp, scale)
  - STEP interpolation supported
  - Animated TRS merged from all playing animations before `render()` computes world matrices
  - Last-animation-wins when multiple animations target the same node/property
- [x] **Skeletal meshes** — joint matrix palette implemented. `setScene()` reads `inverseBindMatrices` and caches per-skin joint data. Each frame, `computeSkinningTransforms()` computes `jointMatrix = inverse(skinNodeWorld) * jointWorld * inverseBindMatrix` per glTF spec and uploads to a triple-buffered GPU palette. `renderPrimitive()` binds `JOINTS_0` (slot 4), `WEIGHTS_0` (slot 5), and joint matrix buffer (slot 20). Metal shader blends up to 4 joint matrices in vertexMain before instance/node transforms. Fallback zero buffers and identity joint matrix handle non-skinned primitives cleanly. Pipeline vertex descriptors dynamically match detected `JOINTS_0` (`UNSIGNED_BYTE`/`UNSIGNED_SHORT`) and `WEIGHTS_0` (`FLOAT`/`UNSIGNED_BYTE` normalized/`UNSIGNED_SHORT` normalized) component types.

---

## Rendering Performance (large scenes)

### Priority 1 — prerequisites for visibility culling

- [x] **Add per-primitive bounds** — local AABB data is now cached per primitive from `POSITION` accessor min/max when available, with a CPU fallback scan for float position buffers.
- [x] **Cache per-node world bounds** — node-local bounds are derived from mesh primitives in `setScene()`, then world bounds are refreshed during transform updates and merged up the hierarchy each frame.
- [x] **Build a visible draw list before the render pass** — scene traversal now gathers opaque / transparent draw items before the render pass instead of drawing recursively during traversal.

### Priority 2 — primary large-scene optimization

- [x] **Implement camera frustum culling** — the renderer now extracts the six frustum planes from the active view-projection matrix and skips whole node subtrees whose cached world bounds are outside the frustum before issuing draw calls.
- [x] **Reduce transform buffer uploads** — `uploadVisibleNodeTransforms()` now finds the highest visible node index and uploads only `[0, (maxIndex+1) * 128]` bytes instead of the entire buffer.
- [x] **Cull `EXT_mesh_gpu_instancing` instances** — `updateVisibleInstances()` now tests each instance's world-space translation against the frustum planes. Visible instances are compacted into `visibleMatrices` and only the surviving count is uploaded/drawn.

### Priority 3 — draw-call and state-change reduction

- [x] **Add mesh / material batching and state sorting** — opaque draws sorted by `(pipelineVariant << 48) | (materialPtr << 16) | meshPtr`. Both legacy `renderToTarget()` and ECS `render(const RenderScene&)` paths sort before submission.
- [x] **Guard debug logging in hot paths** — `std::cout` diagnostics removed from `computeNodeTransform()` and `renderPrimitive()` hot paths.

### Priority 4 — longer-term scalability

- [x] **Add renderer performance counters** — `RenderStats` exposes `opaqueDrawCount`, `transparentDrawCount`, `totalDrawCount`, `culledNodeCount`, `visibleNodeCount`, `cpuFrameTimeMs`. Populated in both legacy and ECS render paths.
- [ ] **Add distance-based LOD hooks** — define per-mesh LOD selection and optional far-distance impostor / low-poly fallback support so large environments do not submit full-resolution geometry at every distance.

---

## Path Tracing (`PathTracer`)

Two goals with different trade-offs: **real-time** (the live viewport: first clean image as
fast as possible, never blocking the UI) and **stills** (rendered images, panoramas and
walkthrough frames: the best image within minutes). Techniques that help both are listed once.

What's there today (`shaders/metal/path_tracer.metal`): unidirectional path tracing,
Owen-scrambled Sobol random numbers (or PCG), cosine-weighted diffuse + GGX specular, the sun
sampled directly (soft disc, shadow rays), the sky sampled as a light by brightness with MIS
(shadow rays through glass), optional indirect clamp, Russian roulette, 4 bounces (optionally
fewer for the first samples), glass passing through without spending a bounce. Denoising with
OIDN `RT` (CPU device, `hdr`, albedo + normal from the first hit) on request. `stats()` gives
GPU time per sample, rebuild and denoise times.

Baseline (`PathTracerMeasuresItsWork`, 64 x 64, ground plane, Intel UHD 630): about 0.35 ms GPU
per sample, 4–8 ms rebuild, 11 ms denoise (CPU OIDN), 17 ms to the first denoised image. Larger
scenes and sizes still to measure.

### Both — fewer samples for the same noise
- [x] **Low-discrepancy sampling** — Owen-scrambled Sobol (Burley 2020), 2D pairs padded per
      path vertex, one pair per sampling strategy (a choice's remainder reused as a coordinate):
      about 23% less error at 16 spp in the sky test. `setSampler()`
- [ ] **Blue-noise error distribution** — so the remaining error denoises better at low counts.
      Tried and backed out: one image-wide Sobol scramble shifted per pixel by a 64 x 64
      void-and-cluster mask (Georgiev & Fajardo 2016). At 1 spp it lowered the error of 4 x 4
      block means by only ~20% in the sky test; from 4 spp on, pixels sharing the scramble's fine
      digits made per-pixel error up to 4x worse (a pure 1D test: 0.111 vs 0.029) and the error
      no longer blue. A progressive tracer needs a sampler built for both: Heitz et al. 2019
      ("A Low-Discrepancy Sampler that Distributes Monte Carlo Errors as a Blue Noise in Screen
      Space", ranking + scrambling keys optimised per tile) or Ahmed & Wonka 2020 (hierarchical
      pixel ordering; needs a fixed sample count, so stills). For games at ~1 spp with temporal
      accumulation, spatiotemporal blue-noise masks (STBN) are the usual choice instead
- [x] **Sky importance sampling + MIS** — the environment's luminance on a 6 x 64 x 64 cube, a CDF
      inverted by binary search (keeps Sobol stratification), shadow rays through glass (average
      tint), power heuristic against the BSDF. About 9x less error at 256 spp with a bright sky
      patch. `setEnvironmentSampling()`
  - [ ] Follow-ups: MIS for the sun discs too (glossy highlights of the sun are BSDF-unsampled
        today); the table assumes the environment texture's contents don't change under the same
        pointer; a finer table (or a two-level one) for very small HDR suns
- [x] **Light portals** — `setPortals()` (rectangles, up to 32), `portalsFromGlass()` (flat glass
      panes from their bounds); spherical-rectangle sampling by solid angle, 80% of the sky's light
      samples where portals are in sight, MIS with the table and the BSDF. About 6x less error at
      256 spp in a room lit through one skylight
  - [ ] Follow-ups: portal-masked environment sampling (Bitterli et al. 2015: aim through the
        portal at the bright part of the sky, e.g. the sun's region of an HDR map); many portals
        (a light BVH instead of a loop over all, past ~32); glass meshes that aren't single flat
        panes (one mesh per facade, curved glass) as several portals
- [x] **Firefly control** — `setIndirectClamp()`: light after two or more diffuse / glossy
      bounces clamped to a max component, hue kept; off by default (stills)
- [x] **OIDN on the GPU** — `setDenoiserDevice()`: the first GPU physical device OIDN reports
      (Metal in the arm64 macOS package), CPU fallback. Only the fallback is tested so far (the
      dev machine is an Intel Mac, whose OIDN package is CPU-only)
  - [ ] Measure the Metal device on Apple silicon (denoise ms at 1280 x 720 vs CPU)
  - [ ] No readback: share campello_gpu's buffers with OIDN (`oidnNewSharedBufferFromMetal`, and
        the CUDA/HIP equivalents) and denoise on the same queue; needs native buffer handles
        from campello_gpu

### Real-time (viewport) — performance
- [x] **Measure first** — `stats()`: GPU time per sample (command buffer GPU time) and its mean,
      rebuild and denoise time, time to first denoised image; baseline above
  - [ ] Measure larger scenes and viewport sizes (e.g. a BIM model at 1280 x 720) and record them
- [x] **Fewer bounces while converging** — `setProgressiveBounces(bounces, samples)`; the early
      samples stay in the average (their bias shrinks as samples are added)
  - [ ] Russian roulette tuned earlier (starts after 2 bounces today)
- [x] **Reduced resolution first samples** — `setPreviewResolution(divisor, samples)`: blocks in
      `output()`, `denoise()` at the small size then bilinear upscaling; full resolution restarts
      the count
  - [ ] Keep the preview's samples (splat them into the full image as a prior) instead of
        restarting; an edge-aware upscale (guided by a full-resolution albedo / normal pass)
- [x] **Denoise more often, cheaper** — `setDenoiserQuality(fast | balanced | high)`; fast is about
      half the time of high on the CPU
  - [ ] The schedule is the application's: denoise every N samples or by time budget instead of
        only at 8 / 32 / 128 / 512, fast early, high later (the viewer); `stats().denoiseMs` to
        budget it
- [x] **Adaptive sample budget per frame** — `setSamplesPerRender()`, `setFrameBudget(ms)` (from
      the measured GPU time per sample and pixel, 1 to 64 per dispatch); none when slow is the
      caller skipping `render()` while `sampleInFlight()`, as before
- [ ] **Temporal reuse while moving** (longer term) — reproject the previous accumulated image with
      motion vectors so camera moves don't restart from zero; ReSTIR-style reuse of light samples;
      a real-time denoiser (SVGF-like, or OIDN with temporal input) for interactive tracing while
      the camera moves. Also what walkthrough video frames need. See Games: Denoising and
      reconstruction, and Light sampling (ReSTIR)
- [x] **Acceleration structure cost** — transforms only: the instances and the TLAS rebuilt (1 ms vs
      24 in the test), BLASes, normals, UVs and materials kept; lights / settings: no rebuild;
      exposure: no restart; BLASes kept per mesh (rebuilt when its buffers change);
      `preferFastTrace` for all
  - [ ] Compaction: needs a compacted-size query in campello_gpu (it has `allowCompaction` and
        `copyAccelerationStructure`, no size)
  - [ ] TLAS refit: campello_gpu's `updateAccelerationStructure` refits from the original build
        inputs, so it can't take new instance transforms; a TLAS rebuild is cheap anyway. BLAS
        refit is what skinned / morphing meshes need (Games: Dynamic scenes)
- [ ] **Hybrid** — raster primary visibility (G-buffer from the raster renderer), rays only for
      secondary effects (shadows, reflections, indirect): the first hit costs nothing extra. See
      Games: Frame structure (G-buffer and motion vectors, ray queries)

### Stills — quality
- [x] **Adaptive sampling** — `setNoiseTarget(relativeError, minSamples)`: per-pixel second moment,
      converged pixels stop (still shown), `Stats::convergedPixels` as the stopping criterion
  - [ ] Error estimates over small neighbourhoods (or a denoised reference) instead of each
        pixel's own, which stops a few pixels early; a tile-based dispatch so stopped regions
        cost no threads at all
- [x] **OIDN quality settings** — `setDenoiserQuality(high)` (default) and
      `setDenoiserPrefilter()` (`cleanAux`, the guides denoised by their own `RT` filters): 30% less
      error at 2 spp in the sky test, about 4x the time
- [x] **More bounces, no clamping** — `setMaxBounces(8)` or more for interiors; the firefly clamp is
      off by default (glass crossings don't spend bounces)
- [x] **Better guides for the denoiser** — through glass (before) and mirrors (roughness < 0.15:
      the reflected surface, tinted by the Fresnel colour), averaged over samples
- [x] **Richer materials: shading normals and maps** — interpolated vertex normals (face normals
      before), normal textures (per-triangle tangents from TEXCOORD_0, or the world mapping frame),
      metallic-roughness textures; geometric-normal offsets and rejection against light leaks
  - [ ] Rough glass with IOR (refraction through thick glass, GGX transmission) and volume
        absorption (KHR_materials_volume)
  - [x] Emissive surfaces sampled directly as area lights — a light list of world-space
        emissive triangles, a power CDF, uniform points, MIS; one-sided unless double-sided.
        About 25x less error in a room lit by one panel. `setEmissiveSampling()`
  - [ ] Emissive textures (the list uses the factor; a textured emitter's texture average would
        do for sampling); a light BVH for many emitters with spatial importance (the CDF picks
        by power regardless of distance: fine for a few luminaires, poor for a city at night)
  - [ ] TEXCOORD_1 (occlusion and other textures on the second set); the mesh's TANGENT
        attribute where present (MikkTSpace-exact normal maps)
- [ ] **Bidirectional path tracing / MLT** (longer term) — paths from the lights as well as the
      camera: much better for interiors lit through small openings and for caustics through glass.
      Metropolis (MLT/PSSMLT) on top for the hardest light paths. Only for stills: slow per sample,
      doesn't fit progressive viewport use
- [ ] **Path guiding** (alternative to BDPT, longer term) — learn where light comes from during
      the first samples (e.g. Open PGL, by Intel, works with OIDN) and aim later paths there.
      Keeps the unidirectional tracer
- [x] **Camera effects** — `setDepthOfField()` (thin lens), `exposureForCamera()`,
      `setToneMapping(pbrNeutral | aces | agx)` on the GPU and in `toneMap()`
  - [ ] Bloom / glare in post (a GPU pass over the linear image before tone mapping; on the
        denoised image for stills); bokeh shapes (polygonal apertures); an autofocus helper
        (focus distance from a ray through the image centre)
- [ ] **No-GPU-RT fallback for stills** — a CPU path tracer (Embree) running the same shading, for
      machines where campello_gpu has no `Feature::raytracing` (Vulkan without RT, web). Time
      matters less for stills than for the viewport

### Games — real-time, moving camera
campello_renderer must be capable of first-class games: about 16 ms per frame (8 ms at 120 Hz),
a camera that never stops, moving and skinned objects, under 1 sample per pixel. No progressive
convergence: noise is hidden by temporal reuse, a real-time denoiser and upscaling. OIDN, BDPT/MLT,
adaptive-to-a-noise-target and the CPU fallback above are offline-only and don't apply.

Shared with the groups above (needed for games as well): low-discrepancy / blue-noise sampling
(spatiotemporal blue noise here), sky importance sampling + MIS, firefly clamping, measuring
first, few bounces, reduced resolution, per-frame budget, temporal reuse / ReSTIR, acceleration
structure cost, hybrid raster + rays, richer materials, camera effects.

**Frame structure**
- [ ] **Ray-traced effects as separate, budgeted passes** on top of the raster renderer, each
      optional and with a quality level: RT shadows (sun, local lights, soft), RT reflections,
      RT ambient occlusion, RT global illumination, RT translucency/refraction. Turned on per
      effect by the game, by quality preset and by `Feature::raytracing`
  - [x] RT sun shadows (Metal) — `RayTracedShadows` (`rt_effects.hpp`): one ray per pixel from
        the G-buffer to a random point of the sun's disc (`setSunAngularDiameter()`), its own
        BLAS cache / TLAS (instances only when transforms change); `RealtimeDenoiser` with
        `Settings::demodulate = false`; `Renderer::setShadowMask()` replaces the shadow maps (the
        mask is packed into the near cascade's slot: the PBR shader is at iOS's 31 texture limit)
  - [x] RT ambient occlusion (Metal) — `RayTracedAmbientOcclusion`: one cosine-weighted ray per
        pixel within a radius; `Renderer::setAmbientOcclusionMask()` replaces SSAO's estimate (its
        apply pass). Effects share a `RayTracingScene` (built once per frame)
  - [ ] Local light shadows (point / spot), several lights; alpha-tested casters; PathTracer on
        `RayTracingScene` too (it builds its own today)
  - [ ] The denoiser's first frames on a binary signal are soft (a 9 px wide shadow: mask 0.63
        after 1 frame, 0.025 after 16): a smaller variance boost for short histories, or
        temporal-only accumulation where the signal agrees with the history
  - [x] RT reflections (Metal) — `PathTracer::traceReflections()`: paths from the G-buffer's
        surface along its GGX lobe, shaded by the path tracer (materials, textures, sun, sky,
        emitters, `setReflectionBounces()`); `Renderer::setReflections(radiance, maxRoughness)`
        replaces the prefiltered environment in opaque surfaces' specular term (in the scene colour
        slot for the opaque draws: the transparent pass keeps the scene colour there)
  - [ ] Reflections on alpha-blended and transmissive surfaces; clear coat; reflection rays from
        rough surfaces at reduced resolution; specular-aware denoising (reprojection along the
        reflected virtual point, not the surface: reflections lag a moving camera today)
  - [ ] RT GI, RT translucency; AO applied to the indirect light only (SSAO's apply pass darkens
        the whole frame, direct light included)
- [ ] **Full path-traced mode** (longer term, high-end GPUs): the "path tracing" setting of modern
      games — 1–2 samples per pixel, ReSTIR DI/GI, real-time denoising, upscaling
- [ ] **Ray queries from raster and compute shaders** (inline RT), not only the RT pipeline: what
      most RT effects use, and the shape the WebGPU proposal and compute emulation would offer
      (see campello_gpu's TODO: emulated ray tracing on WebGPU)
- [x] **G-buffer and motion vectors** (Metal) — `renderGBuffer()`: normal + roughness, albedo +
      metallic, motion (uv, previous - current), depth; camera, object and skinned motion; filled
      automatically by the ECS
  - [ ] Vulkan / DirectX / WebGPU G-buffer shaders (`supportsGBuffer()` is false there)
  - [ ] Morph target motion (the previous weights; motion ignores morphing today), per-instance
        previous transforms (EXT_mesh_gpu_instancing instances are assumed static)
  - [x] App programs: their own G-buffer stages (`ShaderProgramDesc::metalGBuffer*EntryPoint`) so
        geometry their vertex shader moves reaches TAA, denoisers and ray traced effects; the
        lighting inputs (`program_interface.metal.h`: sun visibility / shadow mask, reflections,
        indirect diffuse, exposure) shared with the built-in PBR shader. See
        `docs/using-the-renderer.md`
  - [ ] App programs on Vulkan / DirectX / WebGPU: the lighting inputs and G-buffer stages in
        the other interface headers (with the effects themselves); ray traced passes shading
        app materials (today they see the GpuMaterial's PBR parameters); a pass hook between
        built-in passes
  - [ ] Write the G-buffer from the main pass (multiple render targets) instead of a separate
        geometry pass, once a deferred or hybrid frame needs it every frame
  - [ ] Projection jitter support (TAA / upscalers): a jittered projection with unjittered motion
  - [ ] An ECS test harness: `build_render_scene()`'s motion pass-through is compile-checked only

**Denoising and reconstruction**
- [x] **Real-time denoiser** (Metal) — `RealtimeDenoiser`: SVGF (albedo demodulation, temporal
      accumulation along the G-buffer's motion with depth / normal disocclusion rejection,
      luminance moments, a spatial variance estimate for short histories, 5 edge-aware a-trous
      passes, the first feeding the history). In the test: 8.6x less error after 16 frames, 2.5x
      at a hard shadow edge, 4.6x with a moving camera
  - [ ] Per signal (diffuse, specular, shadows, AO filtered separately, specular with
        roughness-aware reprojection); A-SVGF temporal gradients (history rejection on lighting
        change, not only disocclusion); bilinear / bicubic history reprojection
  - [ ] Vulkan / DirectX / WebGPU shaders (render passes: no compute texture binding needed);
        orthographic projections (linear depth assumes perspective); GPU timing in a stats()
  - [ ] Optionally NVIDIA NRD where available
- [ ] **Upscaling** — temporal upscaling behind one interface: MetalFX (Apple), FSR (all GPUs),
      DLSS / XeSS where available; a built-in TAA-upscale fallback. Ray tracing at a lower
      internal resolution depends on it
  - [x] Built-in temporal upscaler (Metal) — `Renderer::setRenderScale(0.25 .. 1)` with TAA: the
        scene, G-buffer, ray traced effects and denoisers at the internal size; the TAA resolve
        gathers the jittered samples by their distance in output pixels into an output-size
        history (8 / scale^2 jitter positions). At half scale: 10.7 levels RMS from a 4x SSAA
        reference, native TAA 10.4, bilinear 21.5. Without TAA: bilinear
  - [x] MetalFX temporal scaler — `Renderer::setUpscaler(Upscaler::metalFX)` on campello_gpu's
        `TemporalUpscaler` (v0.30.0): the jittered frame, the G-buffer's depth and motion. Falls
        back to the built-in one where unsupported or past MetalFX's ratio (2x on the Intel UHD 630
        dev machine). Half scale: 16.0 levels RMS (MetalFX blends edges a little darker than a box
        average), bilinear 21.5
  - [ ] FSR / DLSS / XeSS on Vulkan / DirectX (campello_gpu `TemporalUpscaler` backends), as
        further `Upscaler` values
  - [ ] MetalFX: pass the reactive mask (transparent / particle draws) and exposure; its
        `inputContentMinScale` / dynamic resolution
  - [ ] The upscaler works on the tone-mapped image (as TAA): a linear (HDR) path; depth-based
        disocclusion rejection; thin features in ray traced masks lose contrast at low scales
        (the denoisers run at the internal size)
- [ ] **Denoise-aware upscaling** (longer term) — DLSS Ray Reconstruction-style combined
      denoise + upscale where available
- [x] **TAA** (Metal) — `setTemporalAntiAliasing()`: Halton(2, 3) jitter, the G-buffer's motion,
      YCoCg neighbourhood clamping, two alternating history textures
  - [ ] Catmull-Rom history sampling (sharper), depth / disocclusion rejection, sub-pixel
        jitter removal from the current frame, luminance-weighted blending against flicker
  - [ ] With SSAA; motion from the main pass instead of an extra G-buffer pass (MRT)

**Light sampling**
- [x] **ReSTIR DI** (Metal) — `RestirDirectLighting` (`rt_effects.hpp`): 32 candidates by power,
      resampled by their contribution (the raster BRDF), visibility reuse, temporal reuse along the
      motion vectors (M capped at 20 frames), spatial reuse while a history is short (radius 1% of
      the width); one shadow ray per pixel for the result. `Renderer::setDirectLighting()` replaces
      the 4-light raster loop; `renderReference()` is the ground truth. 64 point lights in the
      test: relative error 1.66 with one light sample, 0.37 resampling 32, 0.10 after 16 frames of
      reuse (moving camera 0.23); shadows exact (0.2392 vs 0.2393)
  - [ ] Emissive triangles as lights (the path tracer's light list); an alias table instead of the
        CDF search; area / sphere lights (a point per sample, not just the light index)
  - [ ] Unbiased reuse (visibility in the spatial weights, pairwise MIS) instead of feeding back
        the temporal reservoirs and limiting spatial reuse to short histories; the history's
        surface checks against the G-buffer's own previous frame rather than ReSTIR's copy
  - [ ] Sheen, clearcoat and anisotropy in the target function (they lose their direct light
        while `setDirectLighting()` is on); the transparent pass (still the light loop)
  - [ ] Light culling for thousands of lights (a light BVH or grid, see "Light lists")
- [x] **ReSTIR GI** (Metal) — `RestirGlobalIllumination` (`rt_effects.hpp`): one cosine-weighted
      path per pixel from its own `PathTracer` (`traceGlobalIllumination()`, mode 4: the hit,
      its normal, the radiance leaving it), resampled by the irradiance it brings, temporal reuse
      (M capped at 30), spatial reuse while a history is short (reconnection Jacobian, a
      visibility test for borrowed samples); irradiance for `setIndirectDiffuse()`. The DDGI
      test's scene: relative error 2.31 with one path, 0.31 after 16 frames (reference: 256
      frames of raw paths); the wall's red on the ground r 0.256 vs 0.241
  - [ ] Specular indirect from the same samples (a glossy target function), replacing the
        reflections pass for rough surfaces; a moving-camera test; temporal Jacobian
  - [ ] Combine with DDGI: probes for the samples' further bounces and as a fallback where a
        history is short
- [ ] **Light lists / light BVH** for many-light sampling (point, spot, area, emissive meshes)

**Global illumination without per-pixel path tracing**
- [x] **Probe-based GI (DDGI-style)** (Metal) — `DdgiVolume` (`ddgi.hpp`): a probe grid tracing
      rays with the path tracer (`PathTracer::traceProbeRays()`, rotated spherical Fibonacci),
      octahedral irradiance / distance atlases with hysteresis, a screen-space resolve with
      backface and Chebyshev visibility weights (no leaks through walls: tested by disabling the
      visibility term, which then leaks); `Renderer::setIndirectDiffuse()` replaces the
      irradiance map (reflections and GI share the scene colour slot, side by side)
  - [ ] Probe relocation / classification (probes inside geometry), cascades / scrolling volumes
        for large or open worlds, probe updates spread over frames; infinite bounces from the
        probes at ray hits (the path tracer's bounces today)
- [ ] **Screen-space fallbacks** — SSAO / SSR / SSGI where ray tracing is off or too expensive,
      blended with ray-traced results at screen edges
- [ ] **Software ray tracing for GPUs without RT hardware** — signed distance fields or simplified
      scene proxies traced in compute (as Unreal's Lumen does), for GI and reflections on Intel/older
      GPUs and on WebGPU. Complements the compute BVH emulation in campello_gpu

**Dynamic scenes**
- [x] **TLAS rebuilt when transforms change** (instances only), BLASes built once per mesh and
      reused (`RayTracingScene`, `PathTracer`)
- [x] **Skinned meshes** — `SkinnedMeshes` (`rt_effects.hpp`): a compute pass poses each skinned
      draw's positions and normals (joint matrices, as the raster vertex stage) into a posed copy
      of its mesh; `RayTracingScene` and `PathTracer` trace the posed copies (every ray traced
      effect, reflections, ReSTIR GI, DDGI probes). Joint / weight formats from the layout or
      the buffer sizes. In the test the shadow follows the pose (visibility 0.00 vs 1.00)
  - [ ] Refit instead of rebuilding posed BLASes every frame (campello_gpu's Metal refit passes
        a nil descriptor: needs the geometry descriptor), full rebuild every N frames
  - [ ] Morph targets (RenderScene draws don't carry their weights / deltas yet); posing once
        per frame shared by the renderer's consumers instead of one SkinnedMeshes each
- [ ] **Instancing at scale** — thousands of instances (vegetation, props) sharing BLASes,
      `EXT_mesh_gpu_instancing` into the TLAS
- [ ] **Async compute** — acceleration structure builds and denoising overlapped with raster work
      where the backend has a separate compute queue
- [ ] **LOD for ray tracing** — lower-detail BLASes for distant objects and for secondary rays
      (ties into the LOD item in Rendering Performance)
- [x] **Alpha-tested geometry** (foliage, fences) — glTF MASK materials' instances aren't opaque
      and `intersection_query` loops alpha test candidate hits (base colour factor x texture alpha
      against the cutoff): `PathTracer` (its UVs and textures, KHR_texture_transform) and the
      effects (`RayTracingScene`: per-instance alpha data, gathered triangle UVs, up to 16
      textures). In the test only the opaque half of a cut-out plate shadows. BLAS geometry is
      built non-opaque (Metal: an opaque geometry overrides a non-opaque instance)
  - [ ] Opacity micromaps where the hardware has them; more than 16 alpha textures in the
        effects; the texture transform in the effects' test; alpha-blended (not masked) casters

**Budgets and scaling**
- [x] **Quality presets** — `RealtimeEffects` (`realtime_effects.hpp`): `EffectQuality` off / low /
      medium / high / ultra choosing the ray traced effects (each denoised and handed to the
      renderer) and the render scale; `Settings` changes them one by one; raster only without
      `Feature::raytracing`
  - [ ] A "path traced" preset (ReSTIR DI + GI with more bounces, no raster lighting); DDGI as an
        alternative to ReSTIR GI (needs a grid: a scene-bounds fit); per-effect resolution
        (reflections / GI at half the internal size)
- [x] **Per-effect GPU timing and dynamic resolution** — `gpuTimeMs()` on every effect and the
      denoiser (`GpuTiming`: command buffer GPU time, not waiting), `PathTracer::passGpuTimeMs()`,
      `RenderStats::gbufferGpuTimeMs`; `RealtimeEffects::timings()` and `setFrameBudget(ms,
      minScale)` (render scale steps of 0.05, at most every 8 frames). Ultra at 64 x 64 on the
      UHD 630: 6.9 ms (denoising 2.3, GI 1.4, reflections 1.1)
  - [ ] Timestamp queries inside one command buffer (per pass rather than per submission), on
        every backend; budget per effect (lower rays / candidates before the resolution)
- [ ] **Stable frame pacing** — fixed cost per frame (no work that grows with time spent still,
      unlike the viewer's accumulation); BLAS builds spread over frames when streaming geometry

---

## GLTF Extension Rendering Support

The gltf library (v0.4.1) parses **23 extensions** into structured data. The renderer currently uses a small subset. Below is what remains to implement, grouped by complexity.

### Currently rendered
- [x] Core PBR `baseColorFactor` — uploaded to `materialUniformBuffer` (slot 17) per material
- [x] Core PBR `baseColorTexture` — textured pipeline variant, bound via per-GLTF-texture bind groups
- [x] `EXT_texture_webp` — renderer prefers `ext_texture_webp` image index when building bind groups
- [x] GLTF samplers — `gpuSamplers` built from `asset->samplers` with wrap/filter mode mapping
- [x] Debug rendering mode — `setDebugMode(true)` renders geometry with normal visualization (flat shading, no textures/lighting)

### Core material properties (not yet rendered)

- [x] **`metallicRoughnessTexture`** — Full PBR metallic-roughness workflow implemented with Lambert diffuse + Blinn-Phong specular.
- [x] **`normalTexture`** — Tangent-space normal mapping with TBN matrix construction. Supports `normalScale` from `NormalTextureInfo`.
- [x] **`occlusionTexture`** — parsed as `Material::occlusionTexture` (`OcclusionTextureInfo` with `strength` field). Multiply lighting by R channel of the occlusion texture. Implemented in fragment shader with `occlusionStrength` uniform.
- [x] **`emissiveFactor` + `emissiveTexture`** — parsed as `Material::emissiveFactor` (Vector3) and `emissiveTexture` (TextureInfo). Add emissive contribution after lighting. `KHR_materials_emissive_strength` scalar multiplier is already parsed as `Material::khrMaterialsEmissiveStrength`. Fixed Metal `float3` alignment (16-byte aligned, starts at offset 96).
- [x] **Alpha modes: MASK** — `Material::alphaMode` (OPAQUE/MASK/BLEND) and `alphaCutoff`. MASK: `discard` in fragment shader when alpha < cutoff.
- [x] **Alpha modes: BLEND (macOS/Metal)** — alpha blending enabled in pipeline color state with `srcAlpha * oneMinusSrcAlpha`. Blend pipelines created for all variants (flat/textured × single/double-sided). Depth write disabled for transparent materials. Android/Windows: blend pipeline variants alias to opaque (TODO per-platform). Back-to-front sort: transparent primitives collected during opaque traversal, sorted by squared camera distance (descending), drawn after all opaque geometry.
- [x] **Double-sided materials** — `Material::doubleSided`. Pipeline variants with `CullMode::none` for double-sided materials.

### KHR_materials_unlit

- [x] **`KHR_materials_unlit`** — parsed as `Material::khrMaterialsUnlit` (bool). When true, render using only `baseColorFactor × baseColorTexture` with no lighting calculation. Implemented as shader branch using materialFlags.z.

### KHR_texture_transform

- [x] **`KHR_texture_transform`** — UV transform (T × R × S) as two float4 rows per texture in the material slot: base colour, normal (own rows), and since v0.41.0 metallic-roughness, occlusion and emissive, on every backend. The `texCoord` override is honoured (v0.41.0).
- [ ] **The material extensions' textures** (clearcoat, sheen, specular, transmission, volume, iridescence, anisotropy) still have no transform of their own: none on WebGPU, the base colour texture's elsewhere. Their rows don't fit in the 512-byte slot (`kMaterialUniformStride`); a 1 KB slot would hold them.

### KHR_lights_punctual

- [x] **`KHR_lights_punctual`** — parsed into `GLTF::khrLightsPunctual` (vector of `KHRLightPunctual`). Each light has `type` (directional/point/spot), `color`, `intensity`, `range`, and spot cone angles. Node references via `Node::light` index.
  - Lights collected by traversing scene graph, finding nodes with `node.light >= 0`
  - Light transforms computed from node world matrices (directional: -Z axis, point/spot: position)
  - Up to 4 lights supported, uploaded to vertex buffer slot 15 (`VERTEX_SLOT_LIGHTS`)
  - Shader iterates lights, computes inverse-square attenuation (point/spot), spot cone factor (spot), Reinhard tone mapping at output
  - C++ injects a default directional light (5 lux) when no lights exist in the scene (no hardcoded shader fallback)

### KHR_materials_ior

- [x] **`KHR_materials_ior`** — uploaded to material uniform buffer slot 27 (byte 108). Fragment shader replaces hardcoded `F0 = 0.04` with `((ior-1)/(ior+1))²`. Default IOR 1.5 produces F0=0.04, so no regression on assets without the extension.

### KHR_materials_specular

- [x] **`KHR_materials_specular`** — `specularFactor` (scalar) and `specularColorFactor` (vec3) uploaded to `materialUniformBuffer` (offsets 112, 128). `specularTexture` (A channel) and `specularColorTexture` (RGB, sRGB) bound at slots 11–14. Fragment shader computes `F0_dielectric = min(IOR_f0 × specularColor, 1) × specular` before mixing with metallic F0. Default values (factor=1, color=[1,1,1]) are backward-compatible with KHR_materials_ior behaviour.

### KHR_materials_clearcoat

- [x] **`KHR_materials_clearcoat`** — GGX NDF + Smith-GGX visibility + Schlick Fresnel (fixed F0=0.04, IOR 1.5). Three textures: clearcoatTexture (R, binding 17), clearcoatRoughnessTexture (G, binding 18), clearcoatNormalTexture (binding 19). All reuse baseColorSampler. Clearcoat normal decoded with TBN and `clearcoatNormalScale`. Base layer attenuated by `(1 - ccFactor × F_Schlick(0.04, NdotV))`. Uniforms at material buffer offsets 168–191.

### KHR_materials_sheen

- [x] **`KHR_materials_sheen`** — Charlie NDF + Neubelt visibility term. `sheenColorFactor` × `sheenColorTexture` (RGB sRGB, binding 15) tints the sheen lobe; `sheenRoughnessFactor` × `sheenRoughnessTexture` (R linear, binding 16) controls lobe width. Uniforms packed at material buffer offsets 144–167. Both sheen textures reuse `baseColorSampler` (Metal sampler slot limit of 16 is reached).

### KHR_materials_transmission

- [x] **`KHR_materials_transmission`** (simplified) — parsed as `Material::khrMaterialsTransmission` (`transmissionFactor`, `transmissionTexture`). 
  - Simplified implementation: treats transmission as additional alpha transparency
  - `transmissionFactor` 0.0 = fully opaque, 1.0 = fully transparent
  - Multiplied with base color alpha in both `fragmentMain_flat` and `fragmentMain_textured`
  - No render-to-texture (simpler, good for thin glass/panels)
  - Uniform at material buffer offset 228 (index 57), correctly aligned for Metal
  
- [x] **`KHR_materials_transmission`** (sophisticated / physically based) — screen-space refraction implemented. Opaque scene rendered to offscreen `opaqueSceneTexture` (binding 22) with mipmap generation. Transparent pass samples background with refracted UVs using IOR. `transmissionTexture` (binding 20) scales factor. Beer-Lambert attenuation integrated with `KHR_materials_volume`.

### KHR_materials_volume

- [x] **`KHR_materials_volume`** — `thicknessTexture` sampled at binding 23. Beer-Lambert attenuation `exp(-thickness * attn)` applied in fragment shader. Uniforms: `thicknessFactor`, `hasThicknessTex`, `attenuationDistance`, `attenuationColor`.

### KHR_materials_iridescence

- [x] **`KHR_materials_iridescence`** — `iridescenceTexture` (binding 24) and `iridescenceThicknessTexture` (binding 25). `ThinFilmIridescence()` helper in Metal shader computes optical path difference for RGB wavelengths. Modulates `F0` in direct and IBL specular paths. Uniforms expanded to 78 floats.

### KHR_materials_anisotropy

- [x] **`KHR_materials_anisotropy`** — parsed as `Material::khrMaterialsAnisotropy` (`anisotropyStrength`, `anisotropyRotation`, `anisotropyTexture`). Directional specular highlights (brushed metal). Tangent-space anisotropic GGX BRDF implemented in Metal shader with `aspect = sqrt(1.0 - 0.9 * strength)`. Anisotropy texture (binding 26) samples R=strength and G=rotation (0–1 maps to 0–2π). View mode and macOS example hotkey (`p`) added.

### KHR_materials_dispersion

- [ ] **`KHR_materials_dispersion`** — parsed as `Material::khrMaterialsDispersion` (scalar, default 0.0). Chromatic aberration / prism effect for transmissive materials. Depends on `KHR_materials_transmission`.

### EXT_mesh_gpu_instancing

- [x] **`EXT_mesh_gpu_instancing`** — parsed into `Node::extMeshGpuInstancing` (`attributes` map: `TRANSLATION`, `ROTATION`, `SCALE` → accessor IDs). Enables efficient GPU instancing for large repeated meshes. 
  - Instance data loaded in `setScene()`: reads accessors for translation, rotation, scale
  - Per-instance transform matrices uploaded to GPU buffer (`nodeInstanceData`)
  - Vertex buffer slot 19 added for per-instance matrices (stepMode=instance)
  - Metal shader updated to apply instance transform before node transform
  - `renderPrimitive()` uses `drawIndexed(..., instanceCount)` / `draw(..., instanceCount)`

### KHR_materials_variants

- [x] **`KHR_materials_variants`** — `setMaterialVariant(int32_t)` API implemented. `getMaterialVariantCount()` / `getMaterialVariantName()` accessors added. `renderPrimitive()` overrides `matIdx` from `khrMaterialsVariantsMappings` when a variant is active. Pass -1 to restore asset defaults. `activeVariant` reset on `setAsset()`.

### KHR_texture_basisu (KTX2)

- [x] **`KHR_texture_basisu`** — parsed as `Texture::khr_texture_basisu` (image index). KTX2 / Basis Universal compressed GPU textures transcoded via `campello_image::TextureData`:
  - Format selection: ASTC 4×4 → BC7 → ETC2 → RGBA8 fallback based on `device->getFeatures()`
  - Full mip chain uploaded via `copyBufferToTexture`
  - `imageIndexForTex` resolves `khr_texture_basisu` before `ext_texture_webp` and `source`

### Compression (call before rendering)

- [x] **`KHR_draco_mesh_compression`** — `GLTF::loadGLB()` already calls `decompressDraco()` internally, so the renderer should NOT call it again in `setScene()`. Fixed double-decompression bug.
- [x] **`KHR_draco_mesh_compression`** for .gltf files — `import_gltf()` now uses the callback-based `GLTF::loadGLTF(text, callback)` overload for `.gltf` files. The callback loads external `.bin` buffers (and images) relative to the file's base directory. `decompressDraco()` and `decompressMeshopt()` are called automatically by the gltf library after all external buffers arrive.
- [x] **`EXT_meshopt_compression` / `KHR_meshopt_compression`** — `GLTF::decompressMeshopt()` called in `setScene()`, decoded buffer view data used automatically since it's stored in the standard buffer view structure.

### KHR_animation_pointer

- [x] **`KHR_animation_pointer`** — `GltfAnimator::sampleAnimation()` handles `target.path == "pointer"` channels. Node pointers (`/nodes/{N}/translation|rotation|scale`) route to the existing `AnimatedTRS` system. Material pointers (`/materials/{N}/...`) and light pointers (`/extensions/KHR_lights_punctual/lights/{N}/...`) are stored in `animatedPointers` and resolved by `applyAnimatedPointers()`, which mutates the glTF asset and returns modified material indices. Legacy `Renderer::update()` re-uploads affected material slots automatically. ECS path has `material_animation_sync_system()` for the same purpose.

### KHR_mesh_quantization

- [ ] **`KHR_mesh_quantization`** — pipeline variants and `normalized` flag support are implemented, but the test asset (`BoxQuantized.glb`) revealed geometry/normal issues. Needs revisiting to verify end-to-end correctness with proper Khronos sample assets. Deferred until higher-priority extensions are complete.

---

## Materials (PBR metallic-roughness)

- [x] Base color factor — `materialUniformBuffer` uploads `baseColorFactor` per material
- [x] Base color texture — textured pipeline variant binds `baseColorTexture` via bind groups
- [x] Metallic-roughness texture — G=roughness, B=metallic channels sampled and multiplied by factors
- [x] Normal map — tangent-space normal mapping with TBN matrix, `normalScale` support
- [x] Occlusion texture — R channel sampled, multiplied with ambient and diffuse lighting
- [x] Emissive factor + texture — RGB texture sampled, multiplied by `emissiveFactor`, added after lighting
- [x] Alpha modes: MASK (cutout) — implemented
- [x] Alpha modes: BLEND (macOS/Metal) — pipelines ready; Android/Windows alias to opaque; back-to-front sort not yet implemented
- [x] Double-sided materials (toggle `CullMode::none`)

---

## Dependencies

- [x] **`webp.cmake` removed** — `campello_image` already fetches and links its own `libwebp` internally (v1.6.0); the unused root-level `dependencies/webp.cmake` has been deleted.

---

## CI / Testing

- [ ] **Enable tests on Linux** — commented out; blocked by Vulkan vertex input gap upstream. Re-enable once `campello_gpu` Vulkan backend is fixed.
- [x] **Tests on Windows** — CI runs them on `windows-latest` (DirectX), Debug and Release.
- [ ] **GPU-backed integration tests** — all current tests use `nullptr` device. Add macOS-only test suite using a real Metal device to verify render output (render to texture, compare pixel values).
- [ ] **Test with KhronosGroup sample assets** — DamagedHelmet, Sponza, BoxAnimated.

---

## Examples

### macOS (`examples/macos/`)
- [x] **Lighting controls menu** — `Lighting` menu with toggles for punctual lights (Cmd+Shift+L) and default light (Cmd+Option+L), plus `Background` submenu with Dark/Gray/Light presets (1/2/3). Scene presentation API: `setClearColor()`, `setPunctualLightsEnabled()`, `setDefaultLightEnabled()`.
- [x] **Drag-and-drop `.gltf` / `.glb` file loading**
- [ ] Node hierarchy / scene picker UI
- [x] **Load `.gltf` text format** — macOS example uses `GLTF::loadGLTF()` with a URI resolver callback for external buffers/images. Both `.glb` and `.gltf` are supported.

### Android (`examples/android/`)
- [ ] Verify render with campello_gpu v0.7.0 + gltf v0.3.6 after dependency bumps

---

## Documentation

- [ ] Update README with build instructions for all platforms
- [ ] Document the `createDefaultPipelines()` + `resize()` + `render()` required call sequence
- [ ] Document `VERTEX_SLOT_*` shader contract for consumers writing custom shaders
