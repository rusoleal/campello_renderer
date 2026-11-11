// Copy / downsample for WebGPU (Metal's downsampleFragment): // a texture onto the target, bilinear (SSAA downsample: the texture is
// ssaaScale x the target; also the copy of the opaque scene before
// transmissive draws). Its own group 0 (texture, sampler), so as Metal's
// downsampleFragment.

@group(0) @binding(0) var copySource: texture_2d<f32>;
@group(0) @binding(1) var copySampler: sampler;

struct CopyOut {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,  // (0, 0) = top-left
}

@vertex
fn vs_copy(@builtin(vertex_index) index: u32) -> CopyOut {
    var pos = vec2f(-1.0, -1.0);
    if (index == 1u) { pos = vec2f(3.0, -1.0); }
    if (index == 2u) { pos = vec2f(-1.0, 3.0); }
    var o: CopyOut;
    o.position = vec4f(pos, 0.0, 1.0);
    o.uv = vec2f((pos.x + 1.0) * 0.5, (1.0 - pos.y) * 0.5);
    return o;
}

@fragment
fn fs_copy(v: CopyOut) -> @location(0) vec4f {
    return textureSampleLevel(copySource, copySampler, v.uv, 0.0);
}
