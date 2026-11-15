// Real-time denoiser (RealtimeDenoiser, realtime_denoiser.hpp): SVGF-style
// spatiotemporal filtering of a noisy image (about 1 sample per pixel of a
// ray traced effect) guided by the renderer's G-buffer (Schied et al. 2017,
// "Spatiotemporal Variance-Guided Filtering").
//
//   denoiseTemporal  the noisy image demodulated by albedo, accumulated with
//                    the history reprojected along the motion vectors where
//                    depth and normal agree (else the history restarts);
//                    luminance moments for a per-pixel variance; this
//                    frame's normal and linear depth for the next frame.
//   denoiseVariance  the variance: temporal, or a 7 x 7 spatial estimate
//                    while a pixel's history is short.
//   denoiseAtrous    a 5 x 5 a-trous wavelet step (B3 spline), edge-stopping
//                    on depth, normal and luminance (variance-guided); run
//                    five times with growing steps. The first step's output
//                    is the next frame's history (denoiseAtrousFirst).
//   denoiseModulate  the albedo multiplied back in; background passed through.
//
// Fullscreen triangles (render passes): campello_gpu's Metal compute passes
// don't bind textures.

#include <metal_stdlib>
using namespace metal;

struct FullscreenOut {
    float4 position [[position]];
    float2 uv;  // (0, 0) = top-left
};

vertex FullscreenOut denoiseVertex(uint vertexID [[vertex_id]]) {
    float2 pos;
    if (vertexID == 0) pos = float2(-1, -1);
    else if (vertexID == 1) pos = float2(3, -1);
    else pos = float2(-1, 3);
    return FullscreenOut{float4(pos, 1.0, 1.0), float2((pos.x + 1.0) * 0.5, (1.0 - pos.y) * 0.5)};
}

struct DenoiseUniforms {
    float4 params;   // x colour blend floor (e.g. 0.2), y moments blend floor, z history valid, w max history length
    float4 depth;    // x, y: the projection's depth terms (linear depth = y / (device depth + x)); z: depth tolerance (relative)
    float4 filter;   // x step size (pixels), y phi colour, z phi normal (power), w phi depth
};

constant float ALBEDO_FLOOR = 0.01;

inline float luminance(float3 c) {
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

inline uint2 pixelOf(float4 position) {
    return uint2(position.xy);
}

// Linear depth (distance along the view axis) from device depth 0..1; 0 for the background.
inline float linearDepth(float d, constant DenoiseUniforms& u) {
    if (d >= 1.0) return 0.0;
    return abs(u.depth.y / (d + u.depth.x));
}

struct TemporalOut {
    float4 color    [[color(0)]];  // rgb illumination accumulated, a history length
    float4 moments  [[color(1)]];  // x mean luminance, y mean squared luminance, z variance, w history length
    float4 geometry [[color(2)]];  // xyz normal, w linear depth (0: background)
};

fragment TemporalOut denoiseTemporal(FullscreenOut in [[stage_in]],
                                     texture2d<float> noisy           [[texture(0)]],
                                     texture2d<float> historyColor    [[texture(1)]],
                                     texture2d<float> historyMoments  [[texture(2)]],
                                     texture2d<float> normalRoughness [[texture(3)]],
                                     depth2d<float>   depthTexture    [[texture(4)]],
                                     texture2d<float> motion          [[texture(5)]],
                                     texture2d<float> previousGeometry [[texture(6)]],
                                     texture2d<float> albedoMetallic  [[texture(7)]],
                                     constant DenoiseUniforms& u      [[buffer(8)]])
{
    const uint2 p = pixelOf(in.position);
    const uint2 size = uint2(noisy.get_width(), noisy.get_height());
    const float4 n = noisy.read(p);
    const float z = linearDepth(depthTexture.read(p), u);
    const float3 normal = normalRoughness.read(p).xyz;
    TemporalOut out;
    if (z <= 0.0) {
        // Background: nothing to filter.
        out.color = float4(n.rgb, 0.0);
        out.moments = float4(0.0);
        out.geometry = float4(0.0);
        return out;
    }
    const float3 albedo = max(albedoMetallic.read(p).rgb, ALBEDO_FLOOR);
    const float3 illumination = n.rgb / albedo;
    const float l = luminance(illumination);

    // The history where this surface was, if it's the same surface there.
    float4 previousColor = float4(0.0);
    float4 previousMoments = float4(0.0);
    bool valid = false;
    if (u.params.z > 0.5) {
        const float2 previousUV = in.uv + motion.read(p).xy;
        if (all(previousUV >= 0.0) && all(previousUV < 1.0)) {
            const uint2 q = min(uint2(previousUV * float2(size)), size - 1);
            const float4 g = previousGeometry.read(q);
            if (g.w > 0.0 && abs(g.w - z) < u.depth.z * z && dot(g.xyz, normal) > 0.9) {
                previousColor = historyColor.read(q);
                previousMoments = historyMoments.read(q);
                valid = true;
            }
        }
    }
    const float history = valid ? min(previousColor.a + 1.0, u.params.w) : 1.0;
    const float colorBlend = valid ? max(1.0 / history, u.params.x) : 1.0;
    const float momentsBlend = valid ? max(1.0 / history, u.params.y) : 1.0;
    const float3 color = mix(previousColor.rgb, illumination, colorBlend);
    const float m1 = mix(previousMoments.x, l, momentsBlend);
    const float m2 = mix(previousMoments.y, l * l, momentsBlend);
    out.color = float4(color, history);
    out.moments = float4(m1, m2, max(m2 - m1 * m1, 0.0), history);
    out.geometry = float4(normal, z);
    return out;
}

// Edge-stopping weights between the centre (z, n) and a tap.
inline float geometryWeight(float z, float3 normal, float4 tap, float depthScale, float phiNormal) {
    if (tap.w <= 0.0) return 0.0;  // background
    const float wz = exp(-abs(z - tap.w) / depthScale);
    const float wn = pow(max(dot(normal, tap.xyz), 0.0), phiNormal);
    return wz * wn;
}

fragment float4 denoiseVariance(FullscreenOut in [[stage_in]],
                                texture2d<float> accumulated [[texture(0)]],
                                texture2d<float> moments     [[texture(1)]],
                                texture2d<float> geometry    [[texture(2)]],
                                constant DenoiseUniforms& u  [[buffer(3)]])
{
    const uint2 p = pixelOf(in.position);
    const int2 size = int2(accumulated.get_width(), accumulated.get_height());
    const float4 c = accumulated.read(p);
    const float4 m = moments.read(p);
    const float4 g = geometry.read(p);
    const float depthGradient = max(fwidth(g.w), 1e-4);
    if (g.w <= 0.0) return float4(c.rgb, 0.0);
    if (m.w >= 4.0) return float4(c.rgb, m.z);
    // A short history: the variance of the luminance over a 7 x 7
    // neighbourhood of the same surface (and its colour, a little smoothed).
    float weights = 0.0, m1 = 0.0, m2 = 0.0;
    float3 color = float3(0.0);
    for (int dy = -3; dy <= 3; ++dy) {
        for (int dx = -3; dx <= 3; ++dx) {
            const int2 q = clamp(int2(p) + int2(dx, dy), int2(0), size - 1);
            const float4 tg = geometry.read(uint2(q));
            const float w = geometryWeight(g.w, g.xyz, tg, u.filter.w * depthGradient * length(float2(dx, dy)) + 1e-3, u.filter.z);
            const float4 tc = accumulated.read(uint2(q));
            const float l = luminance(tc.rgb);
            weights += w;
            m1 += w * l;
            m2 += w * l * l;
            color += w * tc.rgb;
        }
    }
    weights = max(weights, 1e-6);
    m1 /= weights;
    m2 /= weights;
    // (Boosted for very short histories, as SVGF does.)
    return float4(color / weights, max(m2 - m1 * m1, 0.0) * 4.0 / max(m.w, 1.0));
}

inline float4 atrous(FullscreenOut in, texture2d<float> source, texture2d<float> geometry, constant DenoiseUniforms& u) {
    const uint2 p = pixelOf(in.position);
    const int2 size = int2(source.get_width(), source.get_height());
    const float4 c = source.read(p);
    const float4 g = geometry.read(p);
    const float depthGradient = max(fwidth(g.w), 1e-4);
    if (g.w <= 0.0) return c;
    // The variance, a little blurred (3 x 3 Gaussian), for the luminance weight.
    float variance = 0.0;
    const float gauss[2] = {0.25, 0.125};
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            const int2 q = clamp(int2(p) + int2(dx, dy), int2(0), size - 1);
            variance += gauss[abs(dx)] * gauss[abs(dy)] * 4.0 * source.read(uint2(q)).a;  // weights sum to 1
        }
    }
    const float l = luminance(c.rgb);
    const float luminanceScale = u.filter.y * sqrt(max(variance, 0.0)) + 1e-4;
    const float spline[3] = {3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0};
    const int step = int(u.filter.x);
    float weights = 0.0;
    float3 color = float3(0.0);
    float filteredVariance = 0.0;
    for (int ky = -2; ky <= 2; ++ky) {
        for (int kx = -2; kx <= 2; ++kx) {
            const int2 offset = int2(kx, ky) * step;
            const int2 q = int2(p) + offset;
            if (any(q < int2(0)) || any(q >= size)) continue;
            const float4 tc = source.read(uint2(q));
            const float4 tg = geometry.read(uint2(q));
            const float h = spline[abs(kx)] * spline[abs(ky)];
            float w = h * geometryWeight(g.w, g.xyz, tg, u.filter.w * depthGradient * length(float2(offset)) + 1e-3, u.filter.z);
            w *= exp(-abs(l - luminance(tc.rgb)) / luminanceScale);
            weights += w;
            color += w * tc.rgb;
            filteredVariance += w * w * tc.a;
        }
    }
    weights = max(weights, 1e-6);
    return float4(color / weights, filteredVariance / (weights * weights));
}

fragment float4 denoiseAtrous(FullscreenOut in [[stage_in]],
                              texture2d<float> source    [[texture(0)]],
                              texture2d<float> geometry  [[texture(1)]],
                              constant DenoiseUniforms& u [[buffer(2)]])
{
    return atrous(in, source, geometry, u);
}

struct AtrousFirstOut {
    float4 color   [[color(0)]];  // rgb, a variance (the next step's input)
    float4 history [[color(1)]];  // rgb, a history length: the next frame's history
};

fragment AtrousFirstOut denoiseAtrousFirst(FullscreenOut in [[stage_in]],
                                           texture2d<float> source    [[texture(0)]],
                                           texture2d<float> geometry  [[texture(1)]],
                                           texture2d<float> moments   [[texture(2)]],
                                           constant DenoiseUniforms& u [[buffer(3)]])
{
    const float4 filtered = atrous(in, source, geometry, u);
    return AtrousFirstOut{filtered, float4(filtered.rgb, moments.read(pixelOf(in.position)).w)};
}

fragment float4 denoiseModulate(FullscreenOut in [[stage_in]],
                                texture2d<float> filtered       [[texture(0)]],
                                texture2d<float> albedoMetallic [[texture(1)]],
                                texture2d<float> noisy          [[texture(2)]],
                                texture2d<float> geometry       [[texture(3)]])
{
    const uint2 p = pixelOf(in.position);
    if (geometry.read(p).w <= 0.0) return float4(noisy.read(p).rgb, 1.0);
    return float4(filtered.read(p).rgb * max(albedoMetallic.read(p).rgb, ALBEDO_FLOOR), 1.0);
}
