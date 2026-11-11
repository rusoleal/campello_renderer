// Image-based lighting precompute for WebGPU (fs_ibl_bake, as Metal's
// iblBakeFragment): BRDF LUT, GGX-prefiltered specular cube and Lambertian
// irradiance cube; Renderer::bakeIblResources() draws it once per (mode,
// face, mip). Face directions match the CPU-side convention of
// Renderer::convertEquirectangularImageToCubemap() (face order +X, -X, +Y,
// -Y, +Z, -Z). vs_sky_fullscreen draws one triangle.

const PI: f32 = 3.14159265358979323846;

@vertex
fn vs_sky_fullscreen(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
    var pos = vec2f(-1.0, -1.0);
    if (index == 1u) { pos = vec2f(3.0, -1.0); }
    if (index == 2u) { pos = vec2f(-1.0, 3.0); }
    return vec4f(pos, 1.0, 1.0);
}

struct IblBakeUniforms {
    mode: i32,         // 0 = BRDF LUT, 1 = GGX prefilter, 2 = irradiance convolution
    faceIndex: i32,    // 0..5, modes 1 and 2
    roughness: f32,    // mode 1
    outputSize: f32,   // texels (square) of the current target
}

@group(0) @binding(0) var<uniform> bake: IblBakeUniforms;
@group(0) @binding(1) var srcEnvMap: texture_cube<f32>;
@group(0) @binding(2) var srcEnvSampler: sampler;

fn iblFaceDirection(face: i32, uu: f32, vv: f32) -> vec3f {
    var d: vec3f;
    if (face == 0) { d = vec3f(1.0, -vv, -uu); }
    else if (face == 1) { d = vec3f(-1.0, -vv, uu); }
    else if (face == 2) { d = vec3f(uu, 1.0, vv); }
    else if (face == 3) { d = vec3f(uu, -1.0, -vv); }
    else if (face == 4) { d = vec3f(uu, -vv, 1.0); }
    else { d = vec3f(-uu, -vv, -1.0); }
    return normalize(d);
}

fn iblRadicalInverseVdC(bitsIn: u32) -> f32 {
    var bits = bitsIn;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10;
}

fn iblHammersley(i: u32, n: u32) -> vec2f {
    return vec2f(f32(i) / f32(n), iblRadicalInverseVdC(i));
}

fn iblImportanceSampleGGX(xi: vec2f, N: vec3f, roughness: f32) -> vec3f {
    let a = roughness * roughness;
    let phi = 2.0 * PI * xi.x;
    let cosTheta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    let sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    let H = vec3f(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
    let up = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(N.z) < 0.999);
    let tangent = normalize(cross(up, N));
    let bitangent = cross(N, tangent);
    return normalize(tangent * H.x + bitangent * H.y + N * H.z);
}

fn iblGeometrySchlickGGX(NdotV: f32, roughness: f32) -> f32 {
    let k = (roughness * roughness) / 2.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

fn iblIntegrateBRDF(NdotV: f32, roughness: f32) -> vec2f {
    let V = vec3f(sqrt(max(1.0 - NdotV * NdotV, 0.0)), 0.0, NdotV);
    let N = vec3f(0.0, 0.0, 1.0);
    var A = 0.0;
    var B = 0.0;
    let count = 256u;
    for (var i = 0u; i < count; i++) {
        let H = iblImportanceSampleGGX(iblHammersley(i, count), N, roughness);
        let L = normalize(2.0 * dot(V, H) * H - V);
        let NdotL = max(L.z, 0.0);
        let NdotH = max(H.z, 0.0);
        let VdotH = max(dot(V, H), 0.0);
        if (NdotL > 0.0) {
            let G = iblGeometrySchlickGGX(NdotV, roughness) * iblGeometrySchlickGGX(NdotL, roughness);
            let gVis = (G * VdotH) / max(NdotH * NdotV, 0.0001);
            let fc = pow(1.0 - VdotH, 5.0);
            A += (1.0 - fc) * gVis;
            B += fc * gVis;
        }
    }
    return vec2f(A, B) / f32(count);
}

fn iblPrefilterEnvironment(N: vec3f, roughness: f32) -> vec3f {
    if (roughness < 0.01) {
        return textureSampleLevel(srcEnvMap, srcEnvSampler, N, 0.0).rgb;
    }
    var color = vec3f(0.0);
    var weight = 0.0;
    let count = 512u;
    for (var i = 0u; i < count; i++) {
        let H = iblImportanceSampleGGX(iblHammersley(i, count), N, roughness);
        let L = normalize(2.0 * dot(N, H) * H - N);
        let NdotL = max(dot(N, L), 0.0);
        if (NdotL > 0.0) {
            color += textureSampleLevel(srcEnvMap, srcEnvSampler, L, 0.0).rgb * NdotL;
            weight += NdotL;
        }
    }
    return color / max(weight, 0.0001);
}

fn iblConvolveIrradiance(N: vec3f) -> vec3f {
    let up = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(N.z) < 0.999);
    let tangent = normalize(cross(up, N));
    let bitangent = cross(N, tangent);
    var irradiance = vec3f(0.0);
    let count = 16384u;
    for (var i = 0u; i < count; i++) {
        let xi = iblHammersley(i, count);
        let phi = 2.0 * PI * xi.x;
        let cosTheta = sqrt(1.0 - xi.y);
        let sinTheta = sqrt(xi.y);
        let dir = sinTheta * cos(phi) * tangent + sinTheta * sin(phi) * bitangent + cosTheta * N;
        irradiance += textureSampleLevel(srcEnvMap, srcEnvSampler, dir, 0.0).rgb;
    }
    return PI * irradiance / f32(count);
}

@fragment
fn fs_ibl_bake(@builtin(position) position: vec4f) -> @location(0) vec4f {
    let uv = position.xy / bake.outputSize;
    if (bake.mode == 0) {
        return vec4f(iblIntegrateBRDF(clamp(uv.x, 0.001, 1.0), clamp(uv.y, 0.001, 1.0)), 0.0, 1.0);
    }
    let ndc = uv * 2.0 - 1.0;
    let dir = iblFaceDirection(bake.faceIndex, ndc.x, ndc.y);
    if (bake.mode == 1) {
        return vec4f(iblPrefilterEnvironment(dir, bake.roughness), 1.0);
    }
    return vec4f(iblConvolveIrradiance(dir), 1.0);
}
