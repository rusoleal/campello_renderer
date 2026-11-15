// Ray traced effects on top of the raster renderer (Games track), from the
// G-buffer's first hit:
//
//   rayTracedShadows  one shadow ray per pixel towards a random point of the
//                     sun's disc (soft shadows, about 1 sample per pixel,
//                     new random numbers every frame): visibility 1 lit .. 0
//                     shadowed, for RealtimeDenoiser and
//                     Renderer::setShadowMask().
//
// Metal ray tracing (RayTracedShadows, rt_effects.hpp).

#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;

constant uint MASK_OPAQUE = 1u;  // as path_tracer.metal: glass (2) doesn't cast sun shadows
constant float PI = 3.14159265358979323846;

struct ShadowUniforms {
    float4x4 invViewProj;   // clip -> world (the G-buffer's camera)
    float4   eye;           // xyz: the camera position
    float4   sun;           // xyz towards the sun (unit), w cos of its angular radius
    uint4    size;          // width, height, frame index, 0
};

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
                             constant ShadowUniforms&        u              [[buffer(1)]],
                             texture2d<float>                normalRoughness [[texture(0)]],
                             depth2d<float>                  depthTexture   [[texture(1)]],
                             texture2d<float, access::write> visibility     [[texture(2)]],
                             uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= u.size.x || gid.y >= u.size.y) return;
    const float d = depthTexture.read(gid);
    if (d >= 1.0) {
        visibility.write(float4(1.0), gid);  // background: lit
        return;
    }
    // The surface point from depth.
    const float2 ndc = float2((float(gid.x) + 0.5) / float(u.size.x) * 2.0 - 1.0,
                              1.0 - (float(gid.y) + 0.5) / float(u.size.y) * 2.0);
    const float4 h = u.invViewProj * float4(ndc, d, 1.0);
    const float3 p = h.xyz / h.w;
    const float3 n = normalRoughness.read(gid).xyz;
    // A point of the sun's disc.
    const float2 r = random2(gid, u.size.z);
    const float3 s = u.sun.xyz;
    const float3 helper = abs(s.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    const float3 t = normalize(cross(helper, s));
    const float3 b = cross(s, t);
    const float cosT = 1.0 - r.x * (1.0 - u.sun.w);
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
