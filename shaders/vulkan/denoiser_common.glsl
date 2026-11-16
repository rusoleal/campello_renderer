// Real-time denoiser (RealtimeDenoiser), the Vulkan port of
// shaders/metal/denoiser.metal (see it for the passes): shared declarations.
// Each pass is its own fragment shader (denoise_*.frag) with fxaa.vert's
// fullscreen triangle; textures are read texel by texel without samplers.

#extension GL_EXT_samplerless_texture_functions : require

struct DenoiseUniforms {
    vec4 params;   // x colour blend floor, y moments blend floor, z history valid, w max history length
    vec4 depth;    // x, y: the projection's depth terms (linear depth = y / (device depth + x)); z: depth tolerance (relative); w: demodulate by albedo
    vec4 filterParams;  // x step size (pixels), y phi colour, z phi normal (power), w phi depth
};

layout(location = 0) in vec2 fragUV;  // (0, 0) = top-left

const float ALBEDO_FLOOR = 0.01;

float luminance(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

ivec2 pixelOf() {
    return ivec2(gl_FragCoord.xy);
}

// Linear depth from device depth 0..1; 0 for the background.
float linearDepth(float d, DenoiseUniforms u) {
    if (d >= 1.0) return 0.0;
    return abs(u.depth.y / (d + u.depth.x));
}

// Edge-stopping weights between the centre (z, n) and a tap.
float geometryWeight(float z, vec3 normal, vec4 tap, float depthScale, float phiNormal) {
    if (tap.w <= 0.0) return 0.0;  // background
    float wz = exp(-abs(z - tap.w) / depthScale);
    float wn = pow(max(dot(normal, tap.xyz), 0.0), phiNormal);
    return wz * wn;
}

// A 5 x 5 a-trous step (B3 spline), edge-stopping on depth, normal and
// luminance (variance-guided).
vec4 atrous(texture2D source, texture2D geometry, DenoiseUniforms u) {
    ivec2 p = pixelOf();
    ivec2 size = textureSize(source, 0);
    vec4 c = texelFetch(source, p, 0);
    vec4 g = texelFetch(geometry, p, 0);
    float depthGradient = max(fwidth(g.w), 1e-4);
    if (g.w <= 0.0) return c;
    float variance = 0.0;
    float gauss[2] = float[2](0.25, 0.125);
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            ivec2 q = clamp(p + ivec2(dx, dy), ivec2(0), size - 1);
            variance += gauss[abs(dx)] * gauss[abs(dy)] * 4.0 * texelFetch(source, q, 0).a;
        }
    }
    float l = luminance(c.rgb);
    float luminanceScale = u.filterParams.y * sqrt(max(variance, 0.0)) + 1e-4;
    float spline[3] = float[3](3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0);
    int stepSize = int(u.filterParams.x);
    float weights = 0.0;
    vec3 color = vec3(0.0);
    float filteredVariance = 0.0;
    for (int ky = -2; ky <= 2; ++ky) {
        for (int kx = -2; kx <= 2; ++kx) {
            ivec2 offset = ivec2(kx, ky) * stepSize;
            ivec2 q = p + offset;
            if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, size))) continue;
            vec4 tc = texelFetch(source, q, 0);
            vec4 tg = texelFetch(geometry, q, 0);
            float h = spline[abs(kx)] * spline[abs(ky)];
            float w = h * geometryWeight(g.w, g.xyz, tg, u.filterParams.w * depthGradient * length(vec2(offset)) + 1e-3, u.filterParams.z);
            w *= exp(-abs(l - luminance(tc.rgb)) / luminanceScale);
            weights += w;
            color += w * tc.rgb;
            filteredVariance += w * w * tc.a;
        }
    }
    weights = max(weights, 1e-6);
    return vec4(color / weights, filteredVariance / (weights * weights));
}
