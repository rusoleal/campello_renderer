// Screen-space ambient occlusion (Renderer::setAmbientOcclusion()), WebGPU.
// Same method as shaders/metal/default.metal (ssaoFragment,
// ssaoApplyFragment): fs_ssao turns the scene depth into AO at the output
// size; fs_ssao_apply blurs it 4x4 (depth-aware) and the blend state
// multiplies it into the frame. vs_fullscreen draws one triangle.

struct AoUniforms {
    proj: mat4x4f,     // view -> clip
    invProj: mat4x4f,  // clip -> view
    params: vec4f,     // x = radius (view units), y = intensity (power), z = bias, w = samples
    size: vec4f,       // depth texture width, height; output width, height
}

@group(0) @binding(0) var depthTex: texture_depth_2d;
@group(0) @binding(1) var aoTex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> u: AoUniforms;

struct FullscreenOut {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,  // (0, 0) = top-left
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) index: u32) -> FullscreenOut {
    var pos = vec2f(-1.0, -1.0);
    if (index == 1u) { pos = vec2f(3.0, -1.0); }
    if (index == 2u) { pos = vec2f(-1.0, 3.0); }
    var o: FullscreenOut;
    o.position = vec4f(pos, 1.0, 1.0);
    o.uv = vec2f((pos.x + 1.0) * 0.5, (1.0 - pos.y) * 0.5);
    return o;
}

fn aoDepthAt(uv: vec2f) -> f32 {
    let texel = vec2i(clamp(uv * u.size.xy, vec2f(0.0), u.size.xy - 1.0));
    return textureLoad(depthTex, texel, 0);
}

fn aoViewPos(uv: vec2f, depth: f32) -> vec3f {
    let p = u.invProj * vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    return p.xyz / p.w;
}

@fragment
fn fs_ssao(v: FullscreenOut) -> @location(0) vec4f {
    let depth = aoDepthAt(v.uv);
    if (depth >= 1.0) {
        return vec4f(1.0);
    }
    let P = aoViewPos(v.uv, depth);
    let px = 1.0 / u.size.zw;
    let R = aoViewPos(v.uv + vec2f(px.x, 0.0), aoDepthAt(v.uv + vec2f(px.x, 0.0)));
    let L = aoViewPos(v.uv - vec2f(px.x, 0.0), aoDepthAt(v.uv - vec2f(px.x, 0.0)));
    let D = aoViewPos(v.uv + vec2f(0.0, px.y), aoDepthAt(v.uv + vec2f(0.0, px.y)));
    let U = aoViewPos(v.uv - vec2f(0.0, px.y), aoDepthAt(v.uv - vec2f(0.0, px.y)));
    let dx = select(P - L, R - P, abs(R.z - P.z) < abs(P.z - L.z));
    let dy = select(P - U, D - P, abs(D.z - P.z) < abs(P.z - U.z));
    var N = normalize(cross(dx, dy));
    if (dot(N, -P) < 0.0) { N = -N; }

    let radius = u.params.x;
    let bias = u.params.z;
    let samples = i32(u.params.w);
    let noise = fract(52.9829189 * fract(dot(v.position.xy, vec2f(0.06711056, 0.00583715))));
    let angle = noise * 6.28318530718;
    let helper = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(N.y) < 0.99);
    let T = normalize(cross(helper, N));
    let B = cross(N, T);
    let Tr = T * cos(angle) + B * sin(angle);
    let Br = cross(N, Tr);

    var occlusion = 0.0;
    for (var i = 0; i < samples; i++) {
        let t = (f32(i) + 0.5) / f32(samples);
        let phi = f32(i) * 2.39996323;
        let cosTheta = sqrt(1.0 - t * 0.9);
        let sinTheta = sqrt(1.0 - cosTheta * cosTheta);
        let scale = mix(0.15, 1.0, t * t);
        let k = vec3f(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta) * scale;
        let S = P + (Tr * k.x + Br * k.y + N * k.z) * radius;
        let c = u.proj * vec4f(S, 1.0);
        let suv = vec2f(c.x / c.w * 0.5 + 0.5, 0.5 - c.y / c.w * 0.5);
        if (any(suv < vec2f(0.0)) || any(suv > vec2f(1.0))) { continue; }
        let sceneDepth = aoDepthAt(suv);
        if (sceneDepth >= 1.0) { continue; }
        let Q = aoViewPos(suv, sceneDepth);
        let range = smoothstep(0.0, 1.0, radius / max(abs(P.z - Q.z), 1e-4));
        occlusion += select(0.0, 1.0, abs(Q.z) <= abs(S.z) - bias) * range;
    }
    let ao = pow(clamp(1.0 - occlusion / f32(max(samples, 1)), 0.0, 1.0), u.params.y);
    return vec4f(ao, ao, ao, 1.0);
}

@fragment
fn fs_ssao_apply(v: FullscreenOut) -> @location(0) vec4f {
    let depth = aoDepthAt(v.uv);
    if (depth >= 1.0) {
        return vec4f(1.0);
    }
    let z = aoViewPos(v.uv, depth).z;
    let centre = vec2i(v.uv * u.size.zw);
    let limit = vec2i(u.size.zw) - 1;
    let px = 1.0 / u.size.zw;
    var sum = 0.0;
    var weight = 0.0;
    for (var y = -2; y < 2; y++) {
        for (var x = -2; x < 2; x++) {
            let at = clamp(centre + vec2i(x, y), vec2i(0), limit);
            let uv = (vec2f(at) + 0.5) * px;
            let d = aoDepthAt(uv);
            let sz = aoViewPos(uv, d).z;
            let w = select(0.0, 1.0, d < 1.0 && abs(sz - z) < 0.05 * abs(z) + u.params.z);
            sum += textureLoad(aoTex, at, 0).r * w;
            weight += w;
        }
    }
    let ao = select(1.0, sum / weight, weight > 0.0);
    return vec4f(ao, ao, ao, 1.0);
}
