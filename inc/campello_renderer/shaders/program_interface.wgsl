// campello_renderer shader program interface, WGSL (WebGPU).
// WGSL has no #include: copy what you need into an app program's WGSL
// (ShaderProgramDesc::wgsl, entry points wgslVertexEntryPoint /
// wgslFragmentEntryPoint). See campello_renderer/program.hpp for the
// contract; shaders/webgpu/default.wgsl follows it.
//
// Vertex attributes: @location(semantic), 0-11 (position 0, normal 1,
// texcoord0 2, tangent 3, color0 6, texcoord1 7, custom0-3 8-11).
//
// Per-draw transforms: WebGPU vertex inputs stop at location 15, so instead
// of the per-instance attributes other backends use (16-19, 24-27) they are
// a storage buffer in group 2, indexed by the instance index (the renderer
// passes each draw's transform index as firstInstance):

struct NodeTransforms {
    mvp: mat4x4f,    // clip from local: clip = mvp * vec4f(position, 1.0)
    model: mat4x4f,  // world from local
}
@group(2) @binding(0) var<storage, read> transforms: array<NodeTransforms>;

// In the vertex entry point:
//   @vertex fn vertexMain(..., @builtin(instance_index) instance: u32) -> ... {
//       let t = transforms[instance];
//       ... t.mvp * vec4f(position, 1.0) ...
//   }
// Groups 0 (material) and 1 (frame: lights, camera, environment) follow
// default.wgsl's declarations. Clip space: Y up, Z in [0, 1] (no flip).
