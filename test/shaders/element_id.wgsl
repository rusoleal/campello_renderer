// Test shader program, WGSL (WebGPU): writes a per-vertex uint32 id
// (VertexSemantic::custom0, location 8) into an r32uint target. Transforms
// come from the group 2 storage buffer (program_interface.wgsl).

struct NodeTransforms {
    mvp: mat4x4f,
    model: mat4x4f,
}
@group(2) @binding(0) var<storage, read> transforms: array<NodeTransforms>;

struct VertexOut {
    @builtin(position) position: vec4f,
    @location(0) @interpolate(flat) id: u32,
}

@vertex
fn vertexMain(@location(0) position: vec3f, @location(8) id: u32,
              @builtin(instance_index) instance: u32) -> VertexOut {
    var o: VertexOut;
    o.position = transforms[instance].mvp * vec4f(position, 1.0);
    o.id = id;
    return o;
}

@fragment
fn fragmentMain(v: VertexOut) -> @location(0) u32 {
    return v.id;
}
