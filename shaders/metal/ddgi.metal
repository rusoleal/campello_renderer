// Probe-based global illumination (DdgiVolume, ddgi.hpp), after Majercik et
// al. 2019, "Dynamic Diffuse Global Illumination with Ray-Traced Irradiance
// Fields": a grid of probes, each tracing a few rays a frame (the path
// tracer's probe mode, PathTracer::traceProbeRays()), keeping irradiance
// and hit distance per direction in octahedral atlases.
//
//   ddgiUpdateIrradiance  each atlas texel (a probe's direction): the rays'
//                         radiance weighted by the cosine to it, blended
//                         into the history (hysteresis).
//   ddgiUpdateDistance    the same for the mean and mean squared hit
//                         distance (visibility), a sharper weight.
//   ddgiResolve           per G-buffer pixel: the 8 surrounding probes'
//                         irradiance, weighted by trilinear position, the
//                         surface's facing and Chebyshev visibility.
//
// Atlases: probe i = x + nx (y + ny z) at tile (i mod nx ny, i / nx ny), each
// tile the octahedral map with a 1 texel border (copies of the opposite
// edges, so bilinear sampling doesn't seam).

#include <metal_stdlib>
using namespace metal;

struct FullscreenOut {
    float4 position [[position]];
    float2 uv;  // (0, 0) = top-left
};

vertex FullscreenOut ddgiVertex(uint vertexID [[vertex_id]]) {
    float2 pos;
    if (vertexID == 0) pos = float2(-1, -1);
    else if (vertexID == 1) pos = float2(3, -1);
    else pos = float2(-1, 3);
    return FullscreenOut{float4(pos, 1.0, 1.0), float2((pos.x + 1.0) * 0.5, (1.0 - pos.y) * 0.5)};
}

struct DdgiUniforms {
    float4   origin;    // xyz: probe (0, 0, 0); w: hysteresis (0..1, the history's weight)
    float4   spacing;   // xyz: between probes; w: max ray distance
    uint4    count;     // xyz: probes per axis; w: rays per probe
    float4x4 rotation;  // this frame's ray rotation (the tracer's)
    float4x4 invViewProj;  // resolve: the G-buffer's camera, clip -> world
    float4   eye;       // resolve: xyz the camera; w: history valid (1)
    float4   bias;      // x normal bias, y view bias (fractions of the smallest spacing)
};

constant uint IRRADIANCE_RES = 8;   // interior texels per probe and axis (as ddgi.cpp)
constant uint DISTANCE_RES = 16;

inline float2 signNotZero(float2 v) {
    return float2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0);
}

// Unit direction <-> [-1, 1]^2.
inline float2 octEncode(float3 n) {
    n /= abs(n.x) + abs(n.y) + abs(n.z);
    return n.z >= 0.0 ? n.xy : (1.0 - abs(n.yx)) * signNotZero(n.xy);
}

inline float3 octDecode(float2 p) {
    float3 n = float3(p, 1.0 - abs(p.x) - abs(p.y));
    if (n.z < 0.0) n.xy = (1.0 - abs(n.yx)) * signNotZero(n.xy);
    return normalize(n);
}

// The direction a ray of the frame took (spherical Fibonacci, rotated).
inline float3 rayDirection(uint ray, constant DdgiUniforms& u) {
    const float golden = 1.618033988749895;
    const float n = float(u.count.w);
    const float phi = 2.0 * M_PI_F * fract(float(ray) / golden);
    const float cosTheta = 1.0 - (2.0 * float(ray) + 1.0) / n;
    const float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    const float3 d = float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
    return normalize((u.rotation * float4(d, 0.0)).xyz);
}

// The probe and the interior texel (0 .. res-1) an atlas texel stands for;
// border texels map to the interior texel they copy. False outside any probe.
inline bool atlasTexel(uint2 texel, uint res, constant DdgiUniforms& u, thread uint& probe, thread uint2& interior) {
    const uint tile = res + 2;
    const uint2 t = texel / tile;
    const uint tilesPerRow = u.count.x * u.count.y;
    probe = t.y * tilesPerRow + t.x;
    if (t.x >= tilesPerRow || probe >= u.count.x * u.count.y * u.count.z) return false;
    int2 l = int2(texel % tile);  // 0 .. res+1
    const int r = int(res);
    // Borders: mirrored copies of the opposite edges (octahedral wrap).
    if (l.x == 0 || l.x == r + 1 || l.y == 0 || l.y == r + 1) {
        if ((l.x == 0 || l.x == r + 1) && (l.y == 0 || l.y == r + 1)) {
            l = int2(l.x == 0 ? r : 1, l.y == 0 ? r : 1);  // corners: the opposite corner
        } else if (l.x == 0 || l.x == r + 1) {
            l = int2(l.x == 0 ? 1 : r, r + 1 - l.y);
        } else {
            l = int2(r + 1 - l.x, l.y == 0 ? 1 : r);
        }
    }
    interior = uint2(l - 1);
    return true;
}

inline float3 texelDirection(uint2 interior, uint res) {
    return octDecode((float2(interior) + 0.5) / float(res) * 2.0 - 1.0);
}

fragment float4 ddgiUpdateIrradiance(FullscreenOut in [[stage_in]],
                                     texture2d<float> rays     [[texture(0)]],  // x ray, y probe: rgb radiance, a hit distance
                                     texture2d<float> previous [[texture(1)]],
                                     constant DdgiUniforms& u  [[buffer(2)]])
{
    uint probe;
    uint2 interior;
    if (!atlasTexel(uint2(in.position.xy), IRRADIANCE_RES, u, probe, interior)) return float4(0.0);
    const float3 dir = texelDirection(interior, IRRADIANCE_RES);
    float3 sum = float3(0.0);
    float weights = 0.0;
    for (uint r = 0; r < u.count.w; ++r) {
        const float w = max(0.0, dot(dir, rayDirection(r, u)));
        sum += w * rays.read(uint2(r, probe)).rgb;
        weights += w;
    }
    float3 value = weights > 0.0 ? sum / weights : float3(0.0);
    if (u.eye.w > 0.5) value = mix(value, previous.read(uint2(in.position.xy)).rgb, u.origin.w);
    return float4(value, 1.0);
}

fragment float4 ddgiUpdateDistance(FullscreenOut in [[stage_in]],
                                   texture2d<float> rays     [[texture(0)]],
                                   texture2d<float> previous [[texture(1)]],
                                   constant DdgiUniforms& u  [[buffer(2)]])
{
    uint probe;
    uint2 interior;
    if (!atlasTexel(uint2(in.position.xy), DISTANCE_RES, u, probe, interior)) return float4(0.0);
    const float3 dir = texelDirection(interior, DISTANCE_RES);
    float2 sum = float2(0.0);
    float weights = 0.0;
    for (uint r = 0; r < u.count.w; ++r) {
        const float w = pow(max(0.0, dot(dir, rayDirection(r, u))), 50.0);
        const float d = min(rays.read(uint2(r, probe)).a, u.spacing.w);
        sum += w * float2(d, d * d);
        weights += w;
    }
    float2 value = weights > 0.0 ? sum / weights : float2(u.spacing.w, u.spacing.w * u.spacing.w);
    if (u.eye.w > 0.5) value = mix(value, previous.read(uint2(in.position.xy)).rg, u.origin.w);
    return float4(value, 0.0, 1.0);
}

// A probe's atlas value in direction d (bilinear, inside its tile).
inline float4 atlasSample(texture2d<float> atlas, uint probe, float3 d, uint res, constant DdgiUniforms& u) {
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    const uint tile = res + 2;
    const uint tilesPerRow = u.count.x * u.count.y;
    const float2 corner = float2(float((probe % tilesPerRow) * tile), float((probe / tilesPerRow) * tile));
    const float2 texel = corner + 1.0 + (octEncode(d) * 0.5 + 0.5) * float(res);
    return atlas.sample(s, texel / float2(atlas.get_width(), atlas.get_height()));
}

fragment float4 ddgiResolve(FullscreenOut in [[stage_in]],
                            texture2d<float> irradiance      [[texture(0)]],
                            texture2d<float> distance        [[texture(1)]],
                            texture2d<float> normalRoughness [[texture(2)]],
                            depth2d<float>   depthTexture    [[texture(3)]],
                            constant DdgiUniforms& u         [[buffer(4)]])
{
    const uint2 pixel = uint2(in.position.xy);
    const float depth = depthTexture.read(pixel);
    if (depth >= 1.0) return float4(0.0);
    const float2 size = float2(depthTexture.get_width(), depthTexture.get_height());
    const float2 ndc = float2((float(pixel.x) + 0.5) / size.x * 2.0 - 1.0, 1.0 - (float(pixel.y) + 0.5) / size.y * 2.0);
    const float4 h = u.invViewProj * float4(ndc, depth, 1.0);
    const float3 p = h.xyz / h.w;
    const float3 n = normalize(normalRoughness.read(pixel).xyz);
    const float3 v = normalize(u.eye.xyz - p);
    // Off the surface, towards the viewer: probes behind thin walls don't leak.
    const float minSpacing = min(u.spacing.x, min(u.spacing.y, u.spacing.z));
    const float3 biased = p + (n * u.bias.x + v * u.bias.y) * minSpacing;
    const int3 last = int3(u.count.xyz) - 1;
    const float3 grid = (biased - u.origin.xyz) / u.spacing.xyz;
    const int3 base = clamp(int3(floor(grid)), int3(0), max(last - 1, int3(0)));
    const float3 alpha = clamp(grid - float3(base), 0.0, 1.0);
    float3 sum = float3(0.0);
    float weights = 0.0;
    for (uint i = 0; i < 8; ++i) {
        const int3 offset = int3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        const int3 cell = min(base + offset, last);
        const uint probe = uint(cell.x) + u.count.x * (uint(cell.y) + u.count.y * uint(cell.z));
        const float3 probeP = u.origin.xyz + float3(cell) * u.spacing.xyz;
        const float3 trilinear = mix(1.0 - alpha, alpha, float3(offset));
        // Facing: probes behind the surface count little.
        const float3 toProbe = normalize(probeP - p);
        float w = (dot(toProbe, n) + 1.0) * 0.5;
        w = w * w + 0.2;
        // Visibility: Chebyshev's bound from the probe's distances towards the point.
        const float3 fromProbe = biased - probeP;
        const float d = length(fromProbe);
        const float2 moments = atlasSample(distance, probe, fromProbe / max(d, 1e-6), DISTANCE_RES, u).rg;
        if (d > moments.x) {
            const float variance = abs(moments.y - moments.x * moments.x);
            const float chebyshev = variance / (variance + (d - moments.x) * (d - moments.x));
            w *= max(chebyshev * chebyshev * chebyshev, 0.0);
        }
        w = max(w, 1e-6);
        // Crush tiny weights (as DDGI) so they don't add a floor of light.
        if (w < 0.2) w *= w * w / 0.04;
        w *= trilinear.x * trilinear.y * trilinear.z;
        sum += w * atlasSample(irradiance, probe, n, IRRADIANCE_RES, u).rgb;
        weights += w;
    }
    return float4(weights > 0.0 ? sum / weights : float3(0.0), 1.0);
}
