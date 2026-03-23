# campello_renderer TODO

## Current State (v0.1.0)

### What works
- `setAsset()` — loads GLTF/GLB via the gltf library, allocates GPU resource slot arrays
- `setScene()` — uploads GPU buffers and textures from the loaded asset; allocates transform buffer
- `setCamera()` / `setScene()` — API exists, camera drives view/projection in `render()`
- `resize(w, h)` — creates depth buffer and view
- `setPipeline(pipeline)` — stores the caller-supplied render pipeline
- `render()` — full scene graph traversal, per-node MVP transforms, depth pass, POSITION + NORMAL + TEXCOORD_0 + TANGENT vertex binding, indexed and non-indexed draw calls
- `getVersion()` — works

### Known bugs
1. **gltf library does not compile** (v0.3.0 tag): blocking all integration testing. Waiting for upstream fix.

---

## Pending tasks

### Core — high priority (bugs / missing uploads)
- [x] Fix image upload: call `texture->upload(0, x*y*4, img)` after `stbi_load_from_memory`
- [x] Fix stbi memory leak: call `stbi_image_free(img)` after upload
- [x] Fix `Image::~Image()`: use `free()` instead of `delete`
- [x] Upload GLTF binary buffers in `setScene()`: iterate `asset->buffers`, call `device->createBuffer(size, usage, data)`
- [x] Add WebP image loading support via `libwebp` (v1.5.0)
- [ ] Handle `data:uri` image paths (currently a stub)
- [ ] Handle external image URIs (`.gltf` + external `.bin` / image files)

### Core — render pipeline
- [x] `resize(w, h)`: creates `depth32float` texture and view
- [x] `render()`: scene graph traversal with cumulative TRS node transforms
- [x] `render()`: per-node MVP matrix (VP × M) computed on CPU, uploaded to `transformBuffer`, bound at `VERTEX_SLOT_MVP` (slot 16) per draw
- [x] `render()`: GLTF camera used for view/projection when available; fallback to lookAt(0,0,5) + 60° perspective
- [x] `render()`: depth buffer attached to render pass (clear to 1.0, discard after)
- [x] `render()`: POSITION (slot 0), NORMAL (slot 1), TEXCOORD_0 (slot 2), TANGENT (slot 3) bound per primitive
- [x] `render()`: indexed draw (`drawIndexed`) and non-indexed draw (`draw`) dispatch
- [ ] Handle non-standard buffer view strides (interleaved vertex data — `byteStride` > 0)
- [ ] `update(double dt)`: advance skeletal animations (samplers, channels, interpolation)
- [ ] Skin / joint matrix palette for skeletal meshes

### Core — default shader infrastructure (3 platforms)

The renderer must ship pre-compiled default shaders for each graphics backend so
consuming apps do not need to write or compile shaders for the common case.
The shader contract (vertex slot numbers) is defined by `Renderer::VERTEX_SLOT_*`.

#### Shader contract — vertex inputs (all platforms)
| Slot | Semantic   | Type    | Notes                            |
|------|------------|---------|----------------------------------|
| 0    | POSITION   | float3  | object-space vertex position     |
| 1    | NORMAL     | float3  | object-space vertex normal       |
| 2    | TEXCOORD_0 | float2  | primary UV                       |
| 3    | TANGENT    | float4  | tangent + bitangent sign (w)     |
| 16   | MVP matrix | float4x4| row-major, per-draw, step=instance|

#### Metal (macOS, iOS)
- [x] Write `default.metal` with vertex + fragment functions implementing a simple
      PBR-lite shading model (base colour × diffuse light, no texture binding yet)
      → `shaders/metal/default.metal`
- [x] Compile offline: `xcrun -sdk macosx metal -c default.metal -o default.air`
      then `xcrun -sdk macosx metallib default.air -o default.metallib`
- [x] Embed `default.metallib` as a C byte array → `src/shaders/metal_default.h`
      (namespace `systems::leal::campello_renderer::shaders`, symbols
       `kDefaultMetalShader` / `kDefaultMetalShaderSize`)
- [ ] Repeat compile step for `iphoneos` and `iphonesimulator` SDKs
- [ ] Add Metal compilation step to `CMakeLists.txt` for Apple targets
      (guard with `if(APPLE)`, use `EXECUTE_PROCESS` or a custom target)

#### Vulkan (Android, Linux, future Windows Vulkan)
- [x] Write `default.vert` + `default.frag` in GLSL (targeting SPIR-V 1.0)
      with the same shading model as the Metal shader
      → `shaders/vulkan/default.vert` / `shaders/vulkan/default.frag`
- [x] Compile offline: `glslc default.vert -o default_vert.spv`
      and `glslc default.frag -o default_frag.spv`
      (uses NDK glslc: `$ANDROID_NDK/shader-tools/darwin-x86_64/glslc`)
- [x] Embed both `.spv` files as C byte arrays → `src/shaders/vulkan_default.h`
      (symbols: `kDefaultVulkanVertShader/Size`, `kDefaultVulkanFragShader/Size`)
- [ ] **Upstream blocker**: campello_gpu Vulkan backend hardcodes
      `vertexBindingDescriptionCount = 0` — vertex attributes are not wired up.
      `createDefaultPipeline(ANDROID)` creates the pipeline but vertices won't be
      fetched until this is fixed in campello_gpu.
- [ ] Add `glslc` compilation step to `CMakeLists.txt` for Vulkan targets
      (guard with `if(ANDROID)`, require `glslc` in `find_program`)

#### DirectX 12 (Windows)
- [x] Write `default.hlsl` with vertex + pixel shader functions matching the
      same shading model and slot layout → `shaders/directx/default.hlsl`
      Uses `TEXCOORD<N>` semantics throughout (campello_gpu DX backend maps
      all attributes to SemanticName="TEXCOORD", SemanticIndex=shaderLocation).
      MVP arrives as 4 × float4 rows (TEXCOORD16–19) from slot 16, per-instance.
- [ ] **Compile on Windows** (dxc not available on macOS without extra install):
      `dxc -T vs_6_0 -E vertexMain default.hlsl -Fo default_vs.dxil`
      `dxc -T ps_6_0 -E pixelMain  default.hlsl -Fo default_ps.dxil`
      Then fill `src/shaders/directx_default.h` with the generated byte arrays
      (currently a placeholder with empty arrays — `createDefaultPipeline` returns
      nullptr until the arrays are populated).
- [ ] Add DXC compilation step to `CMakeLists.txt` for Windows targets
      (guard with `if(WIN32)`, require `dxc` in `find_program`)

#### Shader selection at runtime
- [x] Implement `Renderer::createDefaultPipeline(PixelFormat colorFormat)` factory
      that selects the correct embedded binary at compile time (`#if defined(__APPLE__)`)
      and returns a ready-to-use `RenderPipeline`. Metal (macOS) implemented.
      Vulkan and DirectX stubs return `nullptr` until their shaders are written.
- [x] The factory sets `depthStencil.format = depth32float` to match `resize()`

#### Longer term — single shader source
- [ ] Evaluate writing shaders in HLSL and cross-compiling to MSL and SPIR-V
      via `DXC + SPIRV-Cross` so there is a single source of truth
      (reduces divergence risk between platforms)

### Core — materials (PBR metallic-roughness)
- [ ] Base color factor (uniform, push constant / inline constant)
- [ ] Base color texture (requires `setBindGroup` on `RenderPassEncoder` — see gap below)
- [ ] Metallic-roughness texture
- [ ] Normal map
- [ ] Occlusion texture
- [ ] Emissive factor + texture
- [ ] Alpha modes: OPAQUE, MASK (cutout), BLEND (transparency)
- [ ] Double-sided materials (toggle `CullMode::none`)

### Core — campello_gpu gaps (track upstream)
- [ ] **`setBindGroup` missing** from `RenderPassEncoder` v0.3.4: blocks texture and
      sampler binding in shaders — all material textures depend on this
- [ ] **`setFragmentBuffer` missing** from `RenderPassEncoder`: blocks passing material
      uniforms to the fragment stage
- [ ] **`copyBufferToTexture` not implemented** in `CommandEncoder`: needed for
      staging-buffer texture uploads

### Examples — macOS
- [ ] Full render loop via `Renderer::render()` once default pipeline is available
- [ ] Call `renderer->createDefaultPipeline(swapchainFormat)` + `setPipeline()`
- [ ] Texture rendering once `setBindGroup` is available upstream
- [ ] Load `.gltf` (text) files in addition to `.glb`
- [ ] Drag-and-drop file loading onto the window
- [ ] Node hierarchy / scene graph picker UI

### Examples — Android
- [ ] Replace placeholder rendering with `Renderer::render()` + `createDefaultPipeline()`

### Examples — Windows (future)
- [ ] Add Windows example project (DirectX 12 backend)
- [ ] Validate default HLSL shaders against the Windows renderer example

### Testing
- [ ] Unit tests for asset loading (via `CAMPELLO_RENDERER_BUILD_TEST` CMake option)
- [ ] Camera matrix computation tests (perspective, orthographic, lookAt)
- [ ] Test with KhronosGroup glTF-Sample-Assets: DamagedHelmet, Sponza, BoxAnimated
- [ ] Shader visual regression test: render a known asset, compare output image

### Documentation
- [ ] Update README with macOS, Android (and future Windows) build instructions
- [ ] Document public `Renderer` API
- [ ] Document the `createDefaultPipeline()` / `setPipeline()` lifecycle
- [ ] Document `VERTEX_SLOT_*` shader contract for apps writing custom shaders
- [ ] Document how to compile and embed custom shaders per platform
