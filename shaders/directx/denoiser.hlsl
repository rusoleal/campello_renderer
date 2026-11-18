// Real-time denoiser (RealtimeDenoiser), the DirectX port of
// shaders/vulkan/denoise_*.frag and denoiser_common.glsl (themselves ports
// of shaders/metal/denoiser.metal; see it for the passes). Each pass is an
// entry point drawn with denoiseVertex's fullscreen triangle; textures are
// read texel by texel without samplers.
//
// Registers follow RealtimeDenoiser's bind group layouts (makeLayout() in
// src/realtime_denoiser.cpp): textures at t0..t(n-1), the uniform buffer at
// b(n). Entry points reuse register numbers: dxc keeps only the resources an
// entry point references (see default.hlsl's header comment).
//
// Compiled by build_directx_shaders.ps1 into src/shaders/directx_denoiser.h.

struct DenoiseUniforms {
    float4 params;        // x colour blend floor, y moments blend floor, z history valid, w max history length
    float4 depth;         // x, y: the projection's depth terms (linear depth = y / (device depth + x)); z: depth tolerance (relative); w: demodulate by albedo
    float4 filterParams;  // x step size (pixels), y phi colour, z phi normal (power), w phi depth
};

struct DenoiseIn {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;  // (0, 0) = top-left
};

DenoiseIn denoiseVertex(uint vertexID : SV_VertexID) {
    DenoiseIn output;
    float2 pos;
    if (vertexID == 0) pos = float2(-1.0, -1.0);
    else if (vertexID == 1) pos = float2(3.0, -1.0);
    else pos = float2(-1.0, 3.0);
    output.position = float4(pos, 1.0, 1.0);
    // D3D clip space has +Y up; texture coordinates start at the top.
    output.uv = float2((pos.x + 1.0) * 0.5, (1.0 - pos.y) * 0.5);
    return output;
}

static const float ALBEDO_FLOOR = 0.01;

float luminance(float3 c) {
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

int2 textureSize(Texture2D<float4> t) {
    uint w, h;
    t.GetDimensions(w, h);
    return int2(w, h);
}

float4 fetch(Texture2D<float4> t, int2 p) {
    return t.Load(int3(p, 0));
}

// Linear depth from device depth 0..1; 0 for the background.
float linearDepth(float d, DenoiseUniforms u) {
    if (d >= 1.0) return 0.0;
    return abs(u.depth.y / (d + u.depth.x));
}

// Edge-stopping weights between the centre (z, n) and a tap.
float geometryWeight(float z, float3 normal, float4 tap, float depthScale, float phiNormal) {
    if (tap.w <= 0.0) return 0.0;  // background
    float wz = exp(-abs(z - tap.w) / depthScale);
    float wn = pow(max(dot(normal, tap.xyz), 0.0), phiNormal);
    return wz * wn;
}

// A 5 x 5 a-trous step (B3 spline), edge-stopping on depth, normal and
// luminance (variance-guided).
float4 atrous(Texture2D<float4> source, Texture2D<float4> geometry, DenoiseUniforms u, int2 p) {
    int2 size = textureSize(source);
    float4 c = fetch(source, p);
    float4 g = fetch(geometry, p);
    float depthGradient = max(fwidth(g.w), 1e-4);
    if (g.w <= 0.0) return c;
    float variance = 0.0;
    const float gauss[2] = {0.25, 0.125};
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int2 q = clamp(p + int2(dx, dy), int2(0, 0), size - 1);
            variance += gauss[abs(dx)] * gauss[abs(dy)] * 4.0 * fetch(source, q).a;
        }
    }
    float l = luminance(c.rgb);
    float luminanceScale = u.filterParams.y * sqrt(max(variance, 0.0)) + 1e-4;
    const float spline[3] = {3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0};
    int stepSize = int(u.filterParams.x);
    float weights = 0.0;
    float3 color = float3(0.0, 0.0, 0.0);
    float filteredVariance = 0.0;
    for (int ky = -2; ky <= 2; ++ky) {
        for (int kx = -2; kx <= 2; ++kx) {
            int2 offset = int2(kx, ky) * stepSize;
            int2 q = p + offset;
            if (any(q < 0) || any(q >= size)) continue;
            float4 tc = fetch(source, q);
            float4 tg = fetch(geometry, q);
            float h = spline[abs(kx)] * spline[abs(ky)];
            float w = h * geometryWeight(g.w, g.xyz, tg, u.filterParams.w * depthGradient * length(float2(offset)) + 1e-3,
                                         u.filterParams.z);
            w *= exp(-abs(l - luminance(tc.rgb)) / luminanceScale);
            weights += w;
            color += w * tc.rgb;
            filteredVariance += w * w * tc.a;
        }
    }
    weights = max(weights, 1e-6);
    return float4(color / weights, filteredVariance / (weights * weights));
}

// ---------------------------------------------------------------------------
// denoiseTemporal: the noisy image demodulated by albedo, accumulated with
// the history reprojected along the motion vectors where depth and normal
// agree; luminance moments; this frame's normal and linear depth.
// ---------------------------------------------------------------------------
Texture2D<float4> temporalNoisy           : register(t0);
Texture2D<float4> temporalHistoryColor    : register(t1);
Texture2D<float4> temporalHistoryMoments  : register(t2);
Texture2D<float4> temporalNormalRoughness : register(t3);
Texture2D<float>  temporalDepth           : register(t4);
Texture2D<float4> temporalMotion          : register(t5);
Texture2D<float4> temporalPreviousGeometry : register(t6);
Texture2D<float4> temporalAlbedoMetallic  : register(t7);
ConstantBuffer<DenoiseUniforms> temporalUniforms : register(b8);

struct TemporalOut {
    float4 color    : SV_Target0;  // rgb illumination accumulated, a history length
    float4 moments  : SV_Target1;  // x mean luminance, y mean squared, z variance, w history length
    float4 geometry : SV_Target2;  // xyz normal, w linear depth (0: background)
};

TemporalOut denoiseTemporal(DenoiseIn input) {
    DenoiseUniforms u = temporalUniforms;
    int2 p = int2(input.position.xy);
    int2 size = textureSize(temporalNoisy);
    float4 n = fetch(temporalNoisy, p);
    float z = linearDepth(temporalDepth.Load(int3(p, 0)), u);
    float3 normal = fetch(temporalNormalRoughness, p).xyz;
    TemporalOut output;
    if (z <= 0.0) {
        output.color = float4(n.rgb, 0.0);
        output.moments = float4(0.0, 0.0, 0.0, 0.0);
        output.geometry = float4(0.0, 0.0, 0.0, 0.0);
        return output;
    }
    float3 albedo = u.depth.w > 0.5 ? max(fetch(temporalAlbedoMetallic, p).rgb, float3(ALBEDO_FLOOR, ALBEDO_FLOOR, ALBEDO_FLOOR))
                                    : float3(1.0, 1.0, 1.0);
    float3 illumination = n.rgb / albedo;
    float l = luminance(illumination);

    float4 previousColor = float4(0.0, 0.0, 0.0, 0.0);
    float4 previousMoments = float4(0.0, 0.0, 0.0, 0.0);
    bool valid = false;
    if (u.params.z > 0.5) {
        float2 previousUV = input.uv + fetch(temporalMotion, p).xy;
        if (all(previousUV >= 0.0) && all(previousUV < 1.0)) {
            int2 q = min(int2(previousUV * float2(size)), size - 1);
            float4 g = fetch(temporalPreviousGeometry, q);
            if (g.w > 0.0 && abs(g.w - z) < u.depth.z * z && dot(g.xyz, normal) > 0.9) {
                previousColor = fetch(temporalHistoryColor, q);
                previousMoments = fetch(temporalHistoryMoments, q);
                valid = true;
            }
        }
    }
    float history = valid ? min(previousColor.a + 1.0, u.params.w) : 1.0;
    float colorBlend = valid ? max(1.0 / history, u.params.x) : 1.0;
    float momentsBlend = valid ? max(1.0 / history, u.params.y) : 1.0;
    float3 color = lerp(previousColor.rgb, illumination, colorBlend);
    float m1 = lerp(previousMoments.x, l, momentsBlend);
    float m2 = lerp(previousMoments.y, l * l, momentsBlend);
    output.color = float4(color, history);
    output.moments = float4(m1, m2, max(m2 - m1 * m1, 0.0), history);
    output.geometry = float4(normal, z);
    return output;
}

// ---------------------------------------------------------------------------
// denoiseVariance: the temporal variance, or a 7 x 7 spatial estimate while
// a pixel's history is short.
// ---------------------------------------------------------------------------
Texture2D<float4> varianceAccumulated : register(t0);
Texture2D<float4> varianceMoments     : register(t1);
Texture2D<float4> varianceGeometry    : register(t2);
ConstantBuffer<DenoiseUniforms> varianceUniforms : register(b3);

float4 denoiseVariance(DenoiseIn input) : SV_Target0 {
    DenoiseUniforms u = varianceUniforms;
    int2 p = int2(input.position.xy);
    int2 size = textureSize(varianceAccumulated);
    float4 c = fetch(varianceAccumulated, p);
    float4 m = fetch(varianceMoments, p);
    float4 g = fetch(varianceGeometry, p);
    float depthGradient = max(fwidth(g.w), 1e-4);
    if (g.w <= 0.0) return float4(c.rgb, 0.0);
    if (m.w >= 4.0) return float4(c.rgb, m.z);
    float weights = 0.0, m1 = 0.0, m2 = 0.0;
    float3 color = float3(0.0, 0.0, 0.0);
    for (int dy = -3; dy <= 3; ++dy) {
        for (int dx = -3; dx <= 3; ++dx) {
            int2 q = clamp(p + int2(dx, dy), int2(0, 0), size - 1);
            float4 tg = fetch(varianceGeometry, q);
            float w = geometryWeight(g.w, g.xyz, tg, u.filterParams.w * depthGradient * length(float2(dx, dy)) + 1e-3,
                                     u.filterParams.z);
            float4 tc = fetch(varianceAccumulated, q);
            float l = luminance(tc.rgb);
            weights += w;
            m1 += w * l;
            m2 += w * l * l;
            color += w * tc.rgb;
        }
    }
    weights = max(weights, 1e-6);
    m1 /= weights;
    m2 /= weights;
    return float4(color / weights, max(m2 - m1 * m1, 0.0) * 4.0 / max(m.w, 1.0));
}

// ---------------------------------------------------------------------------
// denoiseAtrous: one a-trous step.
// ---------------------------------------------------------------------------
Texture2D<float4> atrousSource   : register(t0);
Texture2D<float4> atrousGeometry : register(t1);
ConstantBuffer<DenoiseUniforms> atrousUniforms : register(b2);

float4 denoiseAtrous(DenoiseIn input) : SV_Target0 {
    return atrous(atrousSource, atrousGeometry, atrousUniforms, int2(input.position.xy));
}

// ---------------------------------------------------------------------------
// denoiseAtrousFirst: the first a-trous step, whose output is also the next
// frame's history.
// ---------------------------------------------------------------------------
Texture2D<float4> atrousFirstSource   : register(t0);
Texture2D<float4> atrousFirstGeometry : register(t1);
Texture2D<float4> atrousFirstMoments  : register(t2);
ConstantBuffer<DenoiseUniforms> atrousFirstUniforms : register(b3);

struct AtrousFirstOut {
    float4 color   : SV_Target0;  // rgb, a variance (the next step's input)
    float4 history : SV_Target1;  // rgb, a history length
};

AtrousFirstOut denoiseAtrousFirst(DenoiseIn input) {
    int2 p = int2(input.position.xy);
    float4 filtered = atrous(atrousFirstSource, atrousFirstGeometry, atrousFirstUniforms, p);
    AtrousFirstOut output;
    output.color = filtered;
    output.history = float4(filtered.rgb, fetch(atrousFirstMoments, p).w);
    return output;
}

// ---------------------------------------------------------------------------
// denoiseModulate: the albedo multiplied back in; background passed through.
// ---------------------------------------------------------------------------
Texture2D<float4> modulateFiltered       : register(t0);
Texture2D<float4> modulateAlbedoMetallic : register(t1);
Texture2D<float4> modulateNoisy          : register(t2);
Texture2D<float4> modulateGeometry       : register(t3);
ConstantBuffer<DenoiseUniforms> modulateUniforms : register(b4);

float4 denoiseModulate(DenoiseIn input) : SV_Target0 {
    DenoiseUniforms u = modulateUniforms;
    int2 p = int2(input.position.xy);
    if (fetch(modulateGeometry, p).w <= 0.0) return float4(fetch(modulateNoisy, p).rgb, 1.0);
    float3 albedo = u.depth.w > 0.5 ? max(fetch(modulateAlbedoMetallic, p).rgb, float3(ALBEDO_FLOOR, ALBEDO_FLOOR, ALBEDO_FLOOR))
                                    : float3(1.0, 1.0, 1.0);
    return float4(fetch(modulateFiltered, p).rgb * albedo, 1.0);
}
