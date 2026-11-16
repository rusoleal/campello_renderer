// Ray traced effects on top of the raster renderer (Games track), from the
// G-buffer's first hit:
//
//   rayTracedShadows  one shadow ray per pixel towards a random point of the
//                     sun's disc (soft shadows, about 1 sample per pixel,
//                     new random numbers every frame): visibility 1 lit .. 0
//                     shadowed, for RealtimeDenoiser and
//                     Renderer::setShadowMask().
//   rayTracedAmbientOcclusion  one ray per pixel in a cosine-weighted random
//                     direction of the surface's hemisphere, as far as a
//                     radius: 1 open .. 0 occluded, for
//                     Renderer::setAmbientOcclusionMask().
//
// Metal ray tracing (RayTracedShadows, RayTracedAmbientOcclusion: rt_effects.hpp).

#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;

constant uint MASK_OPAQUE = 1u;  // as path_tracer.metal: glass (2) doesn't cast sun shadows
constant float PI = 3.14159265358979323846;

struct EffectUniforms {
    float4x4 invViewProj;   // clip -> world (the G-buffer's camera)
    float4   eye;           // xyz: the camera position
    float4   params;        // shadows: xyz towards the sun (unit), w cos of its angular radius; AO: x radius
    uint4    size;          // width, height, frame index, 0
};

// The G-buffer's surface point at a pixel; false for the background.
inline bool surfacePoint(uint2 gid, depth2d<float> depthTexture, constant EffectUniforms& u, thread float3& p) {
    const float d = depthTexture.read(gid);
    if (d >= 1.0) return false;
    const float2 ndc = float2((float(gid.x) + 0.5) / float(u.size.x) * 2.0 - 1.0,
                              1.0 - (float(gid.y) + 0.5) / float(u.size.y) * 2.0);
    const float4 h = u.invViewProj * float4(ndc, d, 1.0);
    p = h.xyz / h.w;
    return true;
}

inline void basis(float3 n, thread float3& t, thread float3& b) {
    const float3 helper = abs(n.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    t = normalize(cross(helper, n));
    b = cross(n, t);
}

inline uint hashU(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

inline float2 random2(uint2 pixel, uint frame) {
    const uint a = hashU(pixel.x * 1973u + pixel.y * 9277u + frame * 26699u + 0x9e3779b9u);
    const uint b = hashU(a ^ 0x68e31da4u);
    return float2(float(a >> 8), float(b >> 8)) / 16777216.0;
}

kernel void rayTracedShadows(instance_acceleration_structure scene          [[buffer(0)]],
                             constant EffectUniforms&        u              [[buffer(1)]],
                             texture2d<float>                normalRoughness [[texture(0)]],
                             depth2d<float>                  depthTexture   [[texture(1)]],
                             texture2d<float, access::write> visibility     [[texture(2)]],
                             uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= u.size.x || gid.y >= u.size.y) return;
    float3 p;
    if (!surfacePoint(gid, depthTexture, u, p)) {
        visibility.write(float4(1.0), gid);  // background: lit
        return;
    }
    const float3 n = normalRoughness.read(gid).xyz;
    // A point of the sun's disc.
    const float2 r = random2(gid, u.size.z);
    const float3 s = u.params.xyz;
    float3 t, b;
    basis(s, t, b);
    const float cosT = 1.0 - r.x * (1.0 - u.params.w);
    const float sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
    const float phi = 2.0 * PI * r.y;
    const float3 l = normalize(t * (sinT * cos(phi)) + b * (sinT * sin(phi)) + s * cosT);
    if (dot(n, l) <= 0.0) {
        visibility.write(float4(0.0), gid);  // facing away from the sun
        return;
    }
    // Offset along the normal by the depth's precision at that distance.
    const float offset = max(1e-3, 2e-4 * length(p - u.eye.xyz));
    ray shadowRay;
    shadowRay.origin = p + n * offset;
    shadowRay.direction = l;
    shadowRay.min_distance = 1e-4;
    shadowRay.max_distance = 1e30;
    intersector<instancing> isect;
    isect.assume_geometry_type(geometry_type::triangle);
    isect.force_opacity(forced_opacity::opaque);
    isect.accept_any_intersection(true);
    const auto hit = isect.intersect(shadowRay, scene, MASK_OPAQUE);
    visibility.write(float4(hit.type == intersection_type::none ? 1.0 : 0.0), gid);
}

kernel void rayTracedAmbientOcclusion(instance_acceleration_structure scene          [[buffer(0)]],
                                      constant EffectUniforms&        u              [[buffer(1)]],
                                      texture2d<float>                normalRoughness [[texture(0)]],
                                      depth2d<float>                  depthTexture   [[texture(1)]],
                                      texture2d<float, access::write> visibility     [[texture(2)]],
                                      uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= u.size.x || gid.y >= u.size.y) return;
    float3 p;
    if (!surfacePoint(gid, depthTexture, u, p)) {
        visibility.write(float4(1.0), gid);  // background: open
        return;
    }
    const float3 n = normalRoughness.read(gid).xyz;
    // A cosine-weighted direction of the hemisphere (the AO integral's own weight).
    const float2 r = random2(gid, u.size.z);
    float3 t, b;
    basis(n, t, b);
    const float radial = sqrt(r.x);
    const float phi = 2.0 * PI * r.y;
    const float3 l = normalize(t * (radial * cos(phi)) + b * (radial * sin(phi)) + n * sqrt(max(0.0, 1.0 - r.x)));
    const float offset = max(1e-3, 2e-4 * length(p - u.eye.xyz));
    ray aoRay;
    aoRay.origin = p + n * offset;
    aoRay.direction = l;
    aoRay.min_distance = 1e-4;
    aoRay.max_distance = u.params.x;
    intersector<instancing> isect;
    isect.assume_geometry_type(geometry_type::triangle);
    isect.force_opacity(forced_opacity::opaque);
    isect.accept_any_intersection(true);
    const auto hit = isect.intersect(aoRay, scene, MASK_OPAQUE);
    visibility.write(float4(hit.type == intersection_type::none ? 1.0 : 0.0), gid);
}

// ---------------------------------------------------------------------------
// ReSTIR DI (RestirDirectLighting): direct light from many lights with one
// shadow ray per pixel, after Bitterli et al. 2020, "Spatiotemporal
// reservoir resampling for real-time ray tracing with dynamic direct
// lighting". Each pixel keeps a reservoir: one chosen light, the resampling
// weight sum, the number of candidates it stands for (M) and its unbiased
// contribution weight W.
//
//   restirInitial  M candidates drawn by the lights' power, resampled by
//                  their unshadowed contribution at the pixel (RIS); the
//                  chosen one's shadow ray (visibility reuse); combined with
//                  the previous frame's reservoir along the motion vector
//                  (similar depth and normal), its M capped.
//   restirSpatial  a few neighbours' reservoirs (similar surfaces) combined;
//                  the result's shadow ray and contribution: the pixel's
//                  direct light (linear, before exposure). Its reservoir is
//                  the next frame's temporal input.
//
// The BRDF is the raster PBR shader's direct-light term (Lambert diffuse +
// GGX specular, Fresnel-weighted, F0 0.04 for dielectrics) from the
// G-buffer's base colour, metallic and roughness.
// ---------------------------------------------------------------------------

struct RestirLight {
    float4 position;   // xyz position (point, spot) or unit direction towards the light (directional); w type (0, 1, 2)
    float4 color;      // rgb colour x intensity; w range (0: none)
    float4 direction;  // xyz spot direction (unit, the way it shines); w cos(outer cone)
    float4 extra;      // x cos(inner cone); y the CDF at this light (inclusive, normalised); z its sampling pdf
};

struct RestirUniforms {
    float4x4 invViewProj;  // clip -> world (the G-buffer's camera)
    float4   eye;          // xyz the camera
    uint4    size;         // width, height, frame index, light count
    uint4    counts;       // x candidates, y temporal M cap (x candidates), z spatial samples, w history valid
    float4   params;       // x spatial radius (pixels); y temporal on; z spatial on; w spatial below this
                           // fraction of the temporal cap
};

constant uint NO_LIGHT = 0xFFFFFFFFu;

struct Reservoir {
    uint  light;
    float wSum;
    float M;
    float W;
    float pHat;  // the chosen light's target value at this pixel
};

inline Reservoir emptyReservoir() {
    Reservoir r;
    r.light = NO_LIGHT;
    r.wSum = 0.0;
    r.M = 0.0;
    r.W = 0.0;
    r.pHat = 0.0;
    return r;
}

// Streaming RIS: keep `light` with probability weight / (sum of weights).
inline void addSample(thread Reservoir& r, uint light, float weight, float pHat, float u) {
    if (!(weight > 0.0)) return;
    r.wSum += weight;
    if (u * r.wSum < weight) {
        r.light = light;
        r.pHat = pHat;
    }
}

inline void finalizeReservoir(thread Reservoir& r) {
    r.W = (r.light != NO_LIGHT && r.pHat > 0.0 && r.M > 0.0) ? r.wSum / (r.M * r.pHat) : 0.0;
}

inline float4 packReservoir(Reservoir r) {
    return float4(as_type<float>(r.light), r.W, r.M, r.pHat);
}

inline Reservoir unpackReservoir(float4 v) {
    Reservoir r;
    r.light = as_type<uint>(v.x);
    r.W = v.y;
    r.M = v.z;
    r.pHat = v.w;
    r.wSum = 0.0;
    return r;
}

// The G-buffer's surface at a pixel.
struct Surface {
    float3 p;
    float3 n;
    float3 v;  // towards the camera
    float3 albedo;
    float  metallic;
    float  roughness;
    float  viewDepth;
};

inline bool readSurface(uint2 gid, constant RestirUniforms& u, texture2d<float> normalRoughness, depth2d<float> depthTexture,
                        texture2d<float> albedoMetallic, thread Surface& s) {
    const float d = depthTexture.read(gid);
    if (d >= 1.0) return false;
    const float2 ndc = float2((float(gid.x) + 0.5) / float(u.size.x) * 2.0 - 1.0,
                              1.0 - (float(gid.y) + 0.5) / float(u.size.y) * 2.0);
    const float4 h = u.invViewProj * float4(ndc, d, 1.0);
    s.p = h.xyz / h.w;
    const float4 nr = normalRoughness.read(gid);
    s.n = normalize(nr.xyz);
    s.roughness = clamp(nr.w, 0.04, 1.0);
    const float4 am = albedoMetallic.read(gid);
    s.albedo = am.rgb;
    s.metallic = am.a;
    s.viewDepth = length(s.p - u.eye.xyz);
    s.v = (s.p - u.eye.xyz) / max(s.viewDepth, 1e-6) * -1.0;
    return true;
}

inline float restirLuminance(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

// A light's unshadowed contribution at the surface (radiance towards the
// camera), the direction to it and its distance (infinite for directional).
inline float3 lightContribution(RestirLight light, Surface s, thread float3& l, thread float& distance) {
    float attenuation = 1.0;
    if (light.position.w < 0.5) {
        l = normalize(light.position.xyz);
        distance = INFINITY;
    } else {
        const float3 toLight = light.position.xyz - s.p;
        distance = length(toLight);
        l = toLight / max(distance, 1e-6);
        attenuation = 1.0 / max(distance * distance, 0.0001);
        if (light.color.w > 0.0) {
            const float nd = distance / light.color.w;
            const float falloff = clamp(1.0 - nd * nd * nd * nd, 0.0, 1.0);
            attenuation *= falloff * falloff;
        }
        if (light.position.w > 1.5) {
            attenuation *= smoothstep(light.direction.w, light.extra.x, dot(-l, light.direction.xyz));
        }
    }
    const float NdotL = dot(s.n, l);
    if (NdotL <= 0.0 || attenuation <= 0.0) return float3(0.0);
    const float NdotV = max(dot(s.n, s.v), 1e-4);
    const float3 h = normalize(l + s.v);
    const float NdotH = max(dot(s.n, h), 0.0);
    const float VdotH = max(dot(s.v, h), 0.0);
    const float3 F0 = mix(float3(0.04), s.albedo, s.metallic);
    const float3 F = F0 + (1.0 - F0) * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
    const float a = s.roughness * s.roughness;
    const float a2 = a * a;
    const float dd = NdotH * NdotH * (a2 - 1.0) + 1.0;
    const float D = a2 / (PI * dd * dd + 0.0001);
    const float GL = NdotV * sqrt(NdotL * NdotL * (1.0 - a) + a);
    const float GV = NdotL * sqrt(NdotV * NdotV * (1.0 - a) + a);
    const float V = 0.5 / max(GL + GV, 0.0001);
    const float3 Lc = light.color.rgb * attenuation;
    const float3 diffuse = s.albedo / PI * (1.0 - F) * (1.0 - s.metallic);
    return (diffuse + F * D * V) * NdotL * Lc;
}

inline bool visible(instance_acceleration_structure scene, Surface s, float3 l, float distance, constant RestirUniforms& u) {
    const float offset = max(1e-3, 2e-4 * s.viewDepth);
    ray r;
    r.origin = s.p + s.n * offset;
    r.direction = l;
    r.min_distance = 1e-4;
    r.max_distance = isinf(distance) ? 1e30 : max(distance - 2.0 * offset, 1e-4);
    intersector<instancing> isect;
    isect.assume_geometry_type(geometry_type::triangle);
    isect.force_opacity(forced_opacity::opaque);
    isect.accept_any_intersection(true);
    return isect.intersect(r, scene, MASK_OPAQUE).type == intersection_type::none;
}

// A light by the lights' power (binary search in the CDF).
inline uint sampleLight(device const RestirLight* lights, uint count, float u) {
    uint lo = 0, hi = count - 1;
    while (lo < hi) {
        const uint mid = (lo + hi) / 2;
        if (lights[mid].extra.y < u) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

inline float targetValue(device const RestirLight* lights, uint light, Surface s) {
    float3 l;
    float distance;
    return restirLuminance(lightContribution(lights[light], s, l, distance));
}

inline float random1(uint2 pixel, uint frame, uint index) {
    return float(hashU(pixel.x * 1973u + pixel.y * 9277u + frame * 26699u + index * 0x632be5abu + 0x51ed270bu) >> 8) /
           16777216.0;
}

inline bool similar(float3 n0, float d0, float3 n1, float d1) {
    return dot(n0, n1) > 0.9 && abs(d0 - d1) < 0.1 * max(d0, 1e-3);
}

// Bindings shared by both passes.
#define RESTIR_PARAMETERS                                                         \
    instance_acceleration_structure scene            [[buffer(0)]],               \
    constant RestirUniforms&        u                [[buffer(1)]],               \
    device const RestirLight*       lights           [[buffer(2)]],               \
    texture2d<float>                normalRoughness  [[texture(0)]],              \
    depth2d<float>                  depthTexture     [[texture(1)]],              \
    texture2d<float>                albedoMetallic   [[texture(2)]],              \
    texture2d<float>                motion           [[texture(3)]],              \
    texture2d<float>                reservoirIn      [[texture(4)]],              \
    texture2d<float>                surfaceIn        [[texture(5)]],              \
    texture2d<float, access::write> reservoirOut     [[texture(6)]],              \
    texture2d<float, access::write> surfaceOut       [[texture(7)]],              \
    texture2d<float, access::write> radianceOut      [[texture(8)]],              \
    uint2 gid [[thread_position_in_grid]]

kernel void restirInitial(RESTIR_PARAMETERS)
{
    if (gid.x >= u.size.x || gid.y >= u.size.y) return;
    Surface s;
    if (!readSurface(gid, u, normalRoughness, depthTexture, albedoMetallic, s) || u.size.w == 0) {
        reservoirOut.write(packReservoir(emptyReservoir()), gid);
        surfaceOut.write(float4(0.0, 0.0, 0.0, -1.0), gid);
        return;
    }
    // Candidates by power, resampled by their contribution here.
    Reservoir r = emptyReservoir();
    const uint candidates = max(u.counts.x, 1u);
    for (uint i = 0; i < candidates; ++i) {
        const uint light = sampleLight(lights, u.size.w, random1(gid, u.size.z, 2u * i));
        const float pdf = lights[light].extra.z;
        const float pHat = targetValue(lights, light, s);
        addSample(r, light, pdf > 0.0 ? pHat / pdf : 0.0, pHat, random1(gid, u.size.z, 2u * i + 1u));
    }
    r.M = float(candidates);
    finalizeReservoir(r);
    // Visibility reuse: an occluded choice contributes nothing (and isn't
    // passed on to neighbours or the next frame).
    if (r.light != NO_LIGHT) {
        float3 l;
        float distance;
        lightContribution(lights[r.light], s, l, distance);
        if (!visible(scene, s, l, distance, u)) r.W = 0.0;
    }
    // Temporal: the previous frame's reservoir where this surface was.
    if (u.params.y > 0.5 && u.counts.w != 0) {
        const float2 uv = (float2(gid) + 0.5) / float2(u.size.xy) + motion.read(gid).xy;
        const int2 q = int2(floor(uv * float2(u.size.xy)));
        if (all(q >= 0) && all(q < int2(u.size.xy))) {
            const float4 previousSurface = surfaceIn.read(uint2(q));
            if (previousSurface.w > 0.0 && similar(s.n, s.viewDepth, previousSurface.xyz, previousSurface.w)) {
                Reservoir prev = unpackReservoir(reservoirIn.read(uint2(q)));
                if (prev.light < u.size.w) {
                    prev.M = min(prev.M, float(u.counts.y * candidates));
                    Reservoir c = emptyReservoir();
                    addSample(c, r.light, r.pHat * r.W * r.M, r.pHat, 0.0);
                    const float prevPHat = targetValue(lights, prev.light, s);
                    addSample(c, prev.light, prevPHat * prev.W * prev.M, prevPHat, random1(gid, u.size.z, 1001u));
                    c.M = r.M + prev.M;
                    finalizeReservoir(c);
                    r = c;
                }
            }
        }
    }
    reservoirOut.write(packReservoir(r), gid);
    surfaceOut.write(float4(s.n, s.viewDepth), gid);
}

kernel void restirSpatial(RESTIR_PARAMETERS)
{
    if (gid.x >= u.size.x || gid.y >= u.size.y) return;
    Surface s;
    if (!readSurface(gid, u, normalRoughness, depthTexture, albedoMetallic, s) || u.size.w == 0) {
        reservoirOut.write(packReservoir(emptyReservoir()), gid);
        radianceOut.write(float4(0.0, 0.0, 0.0, 1.0), gid);
        return;
    }
    const Reservoir own = unpackReservoir(reservoirIn.read(gid));
    Reservoir r = emptyReservoir();
    if (own.light < u.size.w) addSample(r, own.light, own.pHat * own.W * own.M, own.pHat, 0.0);
    r.M = own.M;
    // Neighbours fill in short histories (the first frames, disocclusions,
    // fast motion); a long one is better on its own: their samples carry no
    // visibility at this pixel, a bias that grows with the radius and buys
    // nothing once the history is long (in the test, temporal reuse alone
    // beat temporal + spatial after 16 frames).
    const float longHistory = u.params.w * float(u.counts.y * max(u.counts.x, 1u));
    if (u.params.z > 0.5 && own.M < longHistory) {
        for (uint k = 0; k < u.counts.z; ++k) {
            const float angle = 2.0 * PI * random1(gid, u.size.z, 3000u + 2u * k);
            const float radius = u.params.x * sqrt(random1(gid, u.size.z, 3001u + 2u * k));
            const int2 q = int2(gid) + int2(round(float2(cos(angle), sin(angle)) * radius));
            if (any(q < 0) || any(q >= int2(u.size.xy)) || all(q == int2(gid))) continue;
            const float4 other = surfaceIn.read(uint2(q));
            if (other.w <= 0.0 || !similar(s.n, s.viewDepth, other.xyz, other.w)) continue;
            const Reservoir n = unpackReservoir(reservoirIn.read(uint2(q)));
            if (n.light >= u.size.w) {
                r.M += n.M;
                continue;
            }
            const float pHat = targetValue(lights, n.light, s);
            addSample(r, n.light, pHat * n.W * n.M, pHat, random1(gid, u.size.z, 4000u + k));
            r.M += n.M;
        }
    }
    finalizeReservoir(r);
    // The chosen light's shadow ray and contribution.
    float3 radiance = float3(0.0);
    if (r.light != NO_LIGHT && r.W > 0.0) {
        float3 l;
        float distance;
        const float3 contribution = lightContribution(lights[r.light], s, l, distance);
        if (visible(scene, s, l, distance, u)) {
            radiance = contribution * r.W;
        } else {
            r.W = 0.0;
        }
    }
    reservoirOut.write(packReservoir(r), gid);
    radianceOut.write(float4(radiance, 1.0), gid);
}

// Ground truth for RestirDirectLighting::renderReference(): every light's
// contribution with its own shadow ray.
kernel void restirReference(RESTIR_PARAMETERS)
{
    if (gid.x >= u.size.x || gid.y >= u.size.y) return;
    Surface s;
    if (!readSurface(gid, u, normalRoughness, depthTexture, albedoMetallic, s)) {
        radianceOut.write(float4(0.0, 0.0, 0.0, 1.0), gid);
        return;
    }
    float3 radiance = float3(0.0);
    for (uint i = 0; i < u.size.w; ++i) {
        float3 l;
        float distance;
        const float3 contribution = lightContribution(lights[i], s, l, distance);
        if (any(contribution > 0.0) && visible(scene, s, l, distance, u)) radiance += contribution;
    }
    radianceOut.write(float4(radiance, 1.0), gid);
}

// ---------------------------------------------------------------------------
// ReSTIR GI (RestirGlobalIllumination): one-bounce indirect light, after
// Ouyang et al. 2021, "ReSTIR GI: Path Resampling for Real-Time Path
// Tracing". A sample is a path's second vertex: its point, normal and the
// radiance leaving it towards the pixel's surface (PathTracer::
// traceGlobalIllumination(), one cosine-weighted ray per pixel). Reservoirs
// keep one sample, resampled by the irradiance it brings (luminance x
// cosine), reused over frames and, while a history is short, from
// neighbours (with the reconnection's Jacobian and a visibility test).
// Output: the irradiance (what a white diffuse surface reflects: the
// irradiance map's and DdgiVolume's quantity), for Renderer::
// setIndirectDiffuse().
// ---------------------------------------------------------------------------

struct GiUniforms {
    float4x4 invViewProj;  // clip -> world (the G-buffer's camera)
    float4   eye;          // xyz the camera
    uint4    size;         // width, height, frame index, history valid
    uint4    counts;       // x temporal M cap, y spatial samples
    float4   params;       // x spatial radius (pixels), y temporal on, z spatial on, w spatial below this M
};

inline float3 giOctDecode(float2 p) {
    float3 n = float3(p, 1.0 - abs(p.x) - abs(p.y));
    if (n.z < 0.0) n.xy = (1.0 - abs(n.yx)) * float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    return normalize(n);
}

struct GiSample {
    float3 x;   // the second vertex
    float3 n;   // its normal
    float3 L;   // radiance leaving it towards the first
};

struct GiReservoir {
    GiSample s;
    float wSum;
    float M;
    float W;
    float pHat;
    bool  valid;
};

inline GiReservoir emptyGiReservoir() {
    GiReservoir r;
    r.s.x = float3(0.0);
    r.s.n = float3(0.0);
    r.s.L = float3(0.0);
    r.wSum = 0.0;
    r.M = 0.0;
    r.W = 0.0;
    r.pHat = 0.0;
    r.valid = false;
    return r;
}

// The sample's target at a surface (point p, normal n): the irradiance it
// brings, as luminance.
inline float giTarget(float3 p, float3 n, GiSample s) {
    const float3 d = normalize(s.x - p);
    return restirLuminance(s.L) * max(dot(n, d), 0.0);
}

inline void giAdd(thread GiReservoir& r, GiSample s, float weight, float pHat, float u) {
    if (!(weight > 0.0)) return;
    r.wSum += weight;
    if (u * r.wSum < weight) {
        r.s = s;
        r.pHat = pHat;
        r.valid = true;
    }
}

inline void giFinalize(thread GiReservoir& r) {
    r.W = (r.valid && r.pHat > 0.0 && r.M > 0.0) ? r.wSum / (r.M * r.pHat) : 0.0;
}

// The G-buffer's surface point and normal at a pixel; false: background.
inline bool giSurface(uint2 gid, constant GiUniforms& u, texture2d<float> normalRoughness, depth2d<float> depthTexture,
                      thread float3& p, thread float3& n) {
    const float d = depthTexture.read(gid);
    if (d >= 1.0) return false;
    const float2 ndc = float2((float(gid.x) + 0.5) / float(u.size.x) * 2.0 - 1.0,
                              1.0 - (float(gid.y) + 0.5) / float(u.size.y) * 2.0);
    const float4 h = u.invViewProj * float4(ndc, d, 1.0);
    p = h.xyz / h.w;
    n = normalize(normalRoughness.read(gid).xyz);
    return true;
}

#define RESTIR_GI_PARAMETERS                                                      \
    instance_acceleration_structure scene            [[buffer(0)]],               \
    constant GiUniforms&            u                [[buffer(1)]],               \
    texture2d<float>                normalRoughness  [[texture(0)]],              \
    depth2d<float>                  depthTexture     [[texture(1)]],              \
    texture2d<float>                motion           [[texture(2)]],              \
    texture2d<float>                samples          [[texture(3)]],              \
    texture2d<float>                pointIn          [[texture(4)]],              \
    texture2d<float>                radianceIn       [[texture(5)]],              \
    texture2d<float>                normalIn         [[texture(6)]],              \
    texture2d<float>                surfaceIn        [[texture(7)]],              \
    texture2d<float, access::write> pointOut         [[texture(8)]],              \
    texture2d<float, access::write> radianceOut      [[texture(9)]],              \
    texture2d<float, access::write> normalOut        [[texture(10)]],             \
    texture2d<float, access::write> surfaceOut       [[texture(11)]],             \
    texture2d<float, access::write> irradianceOut    [[texture(12)]],             \
    uint2 gid [[thread_position_in_grid]]

// Reservoirs: point (xyz, W), radiance (rgb, M), normal (xyz, pHat).
inline GiReservoir readGiReservoir(uint2 q, texture2d<float> pointIn, texture2d<float> radianceIn, texture2d<float> normalIn) {
    const float4 a = pointIn.read(q), b = radianceIn.read(q), c = normalIn.read(q);
    GiReservoir r = emptyGiReservoir();
    r.s.x = a.xyz;
    r.W = a.w;
    r.s.L = b.rgb;
    r.M = b.a;
    r.s.n = c.xyz;
    r.pHat = c.w;
    r.valid = r.M > 0.0 && r.W > 0.0;
    return r;
}

inline void writeGiReservoir(uint2 q, GiReservoir r, texture2d<float, access::write> pointOut,
                             texture2d<float, access::write> radianceOut, texture2d<float, access::write> normalOut) {
    pointOut.write(float4(r.s.x, r.valid ? r.W : 0.0), q);
    radianceOut.write(float4(r.s.L, r.M), q);
    normalOut.write(float4(r.s.n, r.pHat), q);
}

kernel void restirGiInitial(RESTIR_GI_PARAMETERS)
{
    if (gid.x >= u.size.x || gid.y >= u.size.y) return;
    float3 p, n;
    if (!giSurface(gid, u, normalRoughness, depthTexture, p, n)) {
        writeGiReservoir(gid, emptyGiReservoir(), pointOut, radianceOut, normalOut);
        surfaceOut.write(float4(0.0, 0.0, 0.0, -1.0), gid);
        return;
    }
    // This frame's sample (cosine-weighted: pdf cos / pi).
    const float4 a = samples.read(gid);
    const float4 b = samples.read(gid + uint2(u.size.x, 0));
    const float3 direction = giOctDecode(b.xy);
    GiSample s;
    s.x = p + direction * a.w;
    s.n = giOctDecode(b.zw);
    s.L = a.rgb;
    GiReservoir r = emptyGiReservoir();
    const float pHat = giTarget(p, n, s);
    const float pdf = max(dot(n, direction), 0.0) / PI;
    giAdd(r, s, pdf > 0.0 ? pHat / pdf : 0.0, pHat, 0.0);
    r.M = 1.0;
    giFinalize(r);
    const float viewDepth = length(p - u.eye.xyz);
    // Temporal: the previous frame's reservoir where this surface was.
    if (u.params.y > 0.5 && u.size.w != 0) {
        const float2 uv = (float2(gid) + 0.5) / float2(u.size.xy) + motion.read(gid).xy;
        const int2 q = int2(floor(uv * float2(u.size.xy)));
        if (all(q >= 0) && all(q < int2(u.size.xy))) {
            const float4 previousSurface = surfaceIn.read(uint2(q));
            if (previousSurface.w > 0.0 && similar(n, viewDepth, previousSurface.xyz, previousSurface.w)) {
                GiReservoir prev = readGiReservoir(uint2(q), pointIn, radianceIn, normalIn);
                if (prev.M > 0.0) {
                    prev.M = min(prev.M, float(u.counts.x));
                    GiReservoir c = emptyGiReservoir();
                    if (r.valid) giAdd(c, r.s, r.pHat * r.W * r.M, r.pHat, 0.0);
                    if (prev.valid) {
                        const float prevPHat = giTarget(p, n, prev.s);
                        giAdd(c, prev.s, prevPHat * prev.W * prev.M, prevPHat, random1(gid, u.size.z, 5001u));
                    }
                    c.M = r.M + prev.M;
                    giFinalize(c);
                    r = c;
                }
            }
        }
    }
    writeGiReservoir(gid, r, pointOut, radianceOut, normalOut);
    surfaceOut.write(float4(n, viewDepth), gid);
}

// The Jacobian of moving a sample's first vertex from `from` to `to`
// (solid angle at the first vertex): the second vertex's cosine and
// distance from each.
inline float giJacobian(GiSample s, float3 from, float3 to) {
    const float3 dTo = to - s.x, dFrom = from - s.x;
    const float lenTo = length(dTo), lenFrom = length(dFrom);
    const float cosTo = abs(dot(s.n, dTo / max(lenTo, 1e-6)));
    const float cosFrom = abs(dot(s.n, dFrom / max(lenFrom, 1e-6)));
    if (cosFrom <= 1e-4) return 0.0;
    return (cosTo / cosFrom) * (lenFrom * lenFrom) / max(lenTo * lenTo, 1e-8);
}

kernel void restirGiSpatial(RESTIR_GI_PARAMETERS)
{
    if (gid.x >= u.size.x || gid.y >= u.size.y) return;
    float3 p, n;
    if (!giSurface(gid, u, normalRoughness, depthTexture, p, n)) {
        writeGiReservoir(gid, emptyGiReservoir(), pointOut, radianceOut, normalOut);
        irradianceOut.write(float4(0.0, 0.0, 0.0, 1.0), gid);
        return;
    }
    const float viewDepth = length(p - u.eye.xyz);
    const GiReservoir own = readGiReservoir(gid, pointIn, radianceIn, normalIn);
    GiReservoir r = emptyGiReservoir();
    if (own.valid) giAdd(r, own.s, own.pHat * own.W * own.M, own.pHat, 0.0);
    r.M = own.M;
    bool borrowed = false;
    // Neighbours while the history is short (as ReSTIR DI).
    if (u.params.z > 0.5 && own.M < u.params.w) {
        for (uint k = 0; k < u.counts.y; ++k) {
            const float angle = 2.0 * PI * random1(gid, u.size.z, 6000u + 2u * k);
            const float radius = u.params.x * sqrt(random1(gid, u.size.z, 6001u + 2u * k));
            const int2 q = int2(gid) + int2(round(float2(cos(angle), sin(angle)) * radius));
            if (any(q < 0) || any(q >= int2(u.size.xy)) || all(q == int2(gid))) continue;
            const float4 other = surfaceIn.read(uint2(q));
            if (other.w <= 0.0 || !similar(n, viewDepth, other.xyz, other.w)) continue;
            float3 pq, nq;
            if (!giSurface(uint2(q), u, normalRoughness, depthTexture, pq, nq)) continue;
            const GiReservoir nr = readGiReservoir(uint2(q), pointIn, radianceIn, normalIn);
            if (!nr.valid) {
                r.M += nr.M;
                continue;
            }
            // The sample seen from here: its density changes by the Jacobian.
            const float jacobian = giJacobian(nr.s, pq, p);
            if (!(jacobian > 0.1 && jacobian < 10.0)) continue;
            const float pHat = giTarget(p, n, nr.s);
            const float before = r.wSum;
            giAdd(r, nr.s, pHat / jacobian * nr.W * nr.M, pHat, random1(gid, u.size.z, 7000u + k));
            if (r.wSum != before && r.s.x.x == nr.s.x.x && r.s.x.y == nr.s.x.y && r.s.x.z == nr.s.x.z) borrowed = true;
            r.M += nr.M;
        }
    }
    giFinalize(r);
    // A neighbour's sample may be hidden from here: check, else keep our own.
    if (borrowed && r.valid) {
        const float3 toSample = r.s.x - p;
        const float distance = length(toSample);
        const float offset = max(1e-3, 2e-4 * viewDepth);
        ray v;
        v.origin = p + n * offset;
        v.direction = toSample / max(distance, 1e-6);
        v.min_distance = 1e-4;
        v.max_distance = max(distance - 2.0 * offset, 1e-4);
        intersector<instancing> isect;
        isect.assume_geometry_type(geometry_type::triangle);
        isect.force_opacity(forced_opacity::opaque);
        isect.accept_any_intersection(true);
        if (isect.intersect(v, scene, MASK_OPAQUE).type != intersection_type::none) {
            r = own;
            r.M = max(own.M, 1.0);
        }
    }
    float3 irradiance = float3(0.0);
    if (r.valid && r.W > 0.0) {
        const float3 d = normalize(r.s.x - p);
        irradiance = r.s.L * max(dot(n, d), 0.0) / PI * r.W;
    }
    writeGiReservoir(gid, r, pointOut, radianceOut, normalOut);
    irradianceOut.write(float4(irradiance, 1.0), gid);
}
