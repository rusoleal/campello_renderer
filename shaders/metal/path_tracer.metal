// Path tracer (PathTracer, path_tracer.hpp): progressive, one sample per
// pixel per dispatch, accumulated in a float buffer and shown tone mapped.
//
//   faceNormals   one geometric normal per triangle of a mesh (object
//                 space), from its own vertex and index buffers, into the
//                 scene's normal buffer at the mesh's offset; and its three
//                 texture coordinates (TEXCOORD_0) into the uv buffer.
//   pathTrace     camera paths through the scene's instance acceleration
//                 structure: diffuse + GGX specular, the sun sampled
//                 directly (a soft 0.27 degree disc, shadow rays), the sky
//                 from the environment cube, thin glass (Fresnel reflection
//                 or straight through), Russian roulette. Also the first
//                 hit's albedo and normal (averaged like the colour), the
//                 denoiser's guides. Perspective cameras, or a 360 x 180
//                 degree equirectangular panorama around the camera.
//
// Lighting units match the raster renderer (default.metal): directional
// lights give irradiance colour x intensity, the sky lights with radiance
// environment x intensity and is seen (camera rays) at full strength, as the
// skybox draws it, so a path-traced image and the viewport agree.

#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;

// ---------------------------------------------------------------------------
// Face normals
// ---------------------------------------------------------------------------
struct NormalParams {
    uint triangleCount;
    uint outOffset;     // first triangle's index in the normal buffer
    uint indexFormat;   // 0 none (triangle soup), 1 uint16, 2 uint32
    uint vertexStride;  // bytes between positions
    uint uvFormat;      // 0 none, 1 float2, 2 unorm16x2, 3 unorm8x2
    uint uvStride;      // bytes between texture coordinates
    uint pad0;
    uint pad1;
};

inline float2 readUV(device const uchar* uvs, uint v, constant NormalParams& p) {
    device const uchar* at = uvs + v * p.uvStride;
    if (p.uvFormat == 1u) return float2(*(device const packed_float2*)at);
    if (p.uvFormat == 2u) return float2(((device const ushort*)at)[0], ((device const ushort*)at)[1]) / 65535.0;
    return float2(float(at[0]), float(at[1])) / 255.0;
}

kernel void faceNormals(device const uchar*   positions [[buffer(0)]],
                        device const uchar*   indices   [[buffer(1)]],
                        device float4*        normals   [[buffer(2)]],
                        constant NormalParams& p        [[buffer(3)]],
                        device const uchar*   texcoords [[buffer(4)]],
                        device float4*        uvs       [[buffer(5)]],
                        uint id [[thread_position_in_grid]])
{
    if (id >= p.triangleCount) return;
    uint v[3];
    for (uint k = 0; k < 3; ++k) {
        const uint i = id * 3 + k;
        if (p.indexFormat == 1) v[k] = ((device const ushort*)indices)[i];
        else if (p.indexFormat == 2) v[k] = ((device const uint*)indices)[i];
        else v[k] = i;
    }
    const float3 a = *(device const packed_float3*)(positions + v[0] * p.vertexStride);
    const float3 b = *(device const packed_float3*)(positions + v[1] * p.vertexStride);
    const float3 c = *(device const packed_float3*)(positions + v[2] * p.vertexStride);
    const float3 n = cross(b - a, c - a);
    const float len = length(n);
    normals[p.outOffset + id] = float4(len > 1e-20 ? n / len : float3(0.0, 0.0, 1.0), 0.0);
    // Texture coordinates of the corners (w of the second: 1 when present).
    if (p.uvFormat != 0u) {
        const float2 t0 = readUV(texcoords, v[0], p);
        const float2 t1 = readUV(texcoords, v[1], p);
        const float2 t2 = readUV(texcoords, v[2], p);
        uvs[(p.outOffset + id) * 2] = float4(t0, t1);
        uvs[(p.outOffset + id) * 2 + 1] = float4(t2, 1.0, 0.0);
    } else {
        uvs[(p.outOffset + id) * 2] = float4(0.0);
        uvs[(p.outOffset + id) * 2 + 1] = float4(0.0);
    }
}

// ---------------------------------------------------------------------------
// Path tracing
// ---------------------------------------------------------------------------
struct TraceUniforms {
    float4x4 invViewProj;   // clip -> world
    float4   cameraPos;     // xyz: the eye (from the view matrix)
    float4   sunDirection[4];  // directional lights: xyz towards the light (unit), w = cos of its angular radius
    float4   sunRadiance[4];   // rgb irradiance (colour x intensity); w = 1 for a light in use
    float4   params;        // x environment intensity, y exposure (EV), z max bounces, w sample index
    uint4    size;          // width, height, sky visible (1) or clear colour, projection (0 perspective, 1 panorama)
    float4   clearColor;    // rgb when the sky isn't visible
    float4   cameraForward; // panorama: the image centre's direction (unit)
    float4   cameraRight;   // panorama: right (unit, perpendicular to forward)
    float4   cameraUp;      // panorama: up, the poles
};

struct InstanceInfo {
    float4 normalMatrix0;  // rows of the world normal matrix (inverse transpose of the model's 3x3)
    float4 normalMatrix1;
    float4 normalMatrix2;
    uint   normalOffset;   // first triangle in the normal buffer
    uint   material;
    uint   pad0;
    uint   pad1;
};

struct MaterialInfo {
    float4 baseColor;  // linear rgb, a = opacity
    float4 emissive;   // rgb, w unused
    float  roughness;
    float  metallic;
    float  transmission;
    float  ior;
    float  unlit;
    float  worldMapping;    // world units per repeat of a box-mapped texture (0: the mesh's TEXCOORD_0)
    float  worldMappingUp;  // 1: z up, 0: y up
    int    texture;         // base colour texture in `textures`, -1 none
    float4 uvRow0;          // KHR_texture_transform row 0 (a, b, tx, has)
    float4 uvRow1;          // row 1 (c, d, ty, 0)
    float4 average;         // rgb: the texture's average colour, for meshes without coordinates
};

constant uint MAX_TEXTURES = 64;  // base colour textures bound (path_tracer.cpp)

// As worldMappingFrame() in default.metal (the raster renderer's box
// mapping), so traced and drawn textures line up.
inline float2 worldMappingUV(float3 P, float3 N, float size, bool zUp) {
    const float3 U = zUp ? float3(0.0, 0.0, 1.0) : float3(0.0, 1.0, 0.0);
    const float nu = dot(N, U);
    float3 T, B;
    if (abs(nu) > 0.7071) {
        T = float3(1.0, 0.0, 0.0);
        B = zUp ? float3(0.0, -1.0, 0.0) : float3(0.0, 0.0, 1.0);
    } else {
        const float3 h = normalize(N - U * nu);
        T = cross(U, h);
        B = -U;
    }
    return float2(dot(P, T), dot(P, B)) / size;
}

constant float PI = 3.14159265358979323846;

// PCG random numbers, one stream per pixel and sample.
inline uint pcg(thread uint& state) {
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
inline float rand01(thread uint& state) {
    return float(pcg(state)) * (1.0 / 4294967296.0);
}

inline void basis(float3 n, thread float3& t, thread float3& b) {
    const float3 helper = abs(n.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    t = normalize(cross(helper, n));
    b = cross(n, t);
}

inline float3 cosineSample(float3 n, float u1, float u2) {
    float3 t, b;
    basis(n, t, b);
    const float r = sqrt(u1);
    const float phi = 2.0 * PI * u2;
    return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(0.0, 1.0 - u1)));
}

// GGX (Trowbridge-Reitz) half vector sample around n.
inline float3 ggxHalfVector(float3 n, float alpha, float u1, float u2) {
    float3 t, b;
    basis(n, t, b);
    const float phi = 2.0 * PI * u1;
    const float cosTheta = sqrt((1.0 - u2) / (1.0 + (alpha * alpha - 1.0) * u2));
    const float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    return normalize(t * (sinTheta * cos(phi)) + b * (sinTheta * sin(phi)) + n * cosTheta);
}

inline float ggxD(float NdotH, float alpha) {
    const float a2 = alpha * alpha;
    const float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

inline float smithG1(float NdotX, float alpha) {
    const float a2 = alpha * alpha;
    return 2.0 * NdotX / (NdotX + sqrt(a2 + (1.0 - a2) * NdotX * NdotX));
}

inline float3 fresnelSchlick(float3 f0, float cosTheta) {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Diffuse + GGX specular for light direction l (all unit, n towards v).
inline float3 brdf(float3 n, float3 v, float3 l, float3 albedo, float metallic, float alpha, float3 f0) {
    const float NdotL = dot(n, l);
    const float NdotV = max(dot(n, v), 1e-4);
    if (NdotL <= 0.0) return float3(0.0);
    const float3 h = normalize(v + l);
    const float NdotH = max(dot(n, h), 0.0);
    const float3 F = fresnelSchlick(f0, max(dot(v, h), 0.0));
    const float3 specular = F * ggxD(NdotH, alpha) * smithG1(NdotV, alpha) * smithG1(NdotL, alpha) / (4.0 * NdotV * NdotL);
    const float3 diffuse = (1.0 - F) * (1.0 - metallic) * albedo / PI;
    return diffuse + specular;
}

inline float3 toneMapKhronosPbrNeutral(float3 color) {
    const float startCompression = 0.8 - 0.04;
    const float desaturation = 0.15;
    const float x = min(color.r, min(color.g, color.b));
    const float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
    color -= offset;
    const float peak = max(color.r, max(color.g, color.b));
    if (peak < startCompression) return color;
    const float d = 1.0 - startCompression;
    const float newPeak = 1.0 - d * d / (peak + d - startCompression);
    color *= newPeak / peak;
    const float g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
    return mix(color, float3(newPeak), g);
}

// Instance masks: opaque geometry casts sun shadows, glass doesn't.
constant uint MASK_OPAQUE = 1u;
constant uint MASK_GLASS  = 2u;

kernel void pathTrace(instance_acceleration_structure scene      [[buffer(0)]],
                      device float4*                  accum      [[buffer(1)]],
                      constant TraceUniforms&         u          [[buffer(2)]],
                      device const InstanceInfo*      instances  [[buffer(3)]],
                      device const MaterialInfo*      materials  [[buffer(4)]],
                      device const float4*            normals    [[buffer(5)]],
                      texture2d<float, access::write> output     [[texture(0)]],
                      device float4*                  aux        [[buffer(6)]],
                      device const float4*            uvs        [[buffer(7)]],
                      texturecube<float>              environment [[texture(1)]],
                      array<texture2d<float>, MAX_TEXTURES> textures [[texture(2)]],
                      uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= u.size.x || gid.y >= u.size.y) return;
    constexpr sampler envSampler(filter::linear, mip_filter::linear);
    constexpr sampler texSampler(filter::linear, mip_filter::linear, address::repeat);
    const uint sampleIndex = uint(u.params.w);
    uint rng = (gid.y * u.size.x + gid.x) * 9781u + sampleIndex * 6271u + 1u;
    pcg(rng);

    // Camera ray through a jittered point of the pixel (antialiasing).
    const float2 jitter = float2(rand01(rng), rand01(rng));
    const float2 ndc = float2((float(gid.x) + jitter.x) / float(u.size.x) * 2.0 - 1.0,
                              1.0 - (float(gid.y) + jitter.y) / float(u.size.y) * 2.0);
    float3 origin;
    float3 direction;
    if (u.size.w == 1u) {
        // Equirectangular: longitude across (the centre straight ahead,
        // right towards the right), latitude down from the up pole.
        const float lon = (ndc.x * 0.5) * 2.0 * PI;
        const float lat = ndc.y * 0.5 * PI;
        origin = u.cameraPos.xyz;
        direction = normalize(u.cameraForward.xyz * (cos(lat) * cos(lon)) + u.cameraRight.xyz * (cos(lat) * sin(lon)) +
                              u.cameraUp.xyz * sin(lat));
    } else {
        // From the eye through the pixel's point on the near plane: the far
        // plane isn't unprojected (in 32-bit floats it loses its precision
        // when far / near is large, e.g. 5000 m / 5 cm).
        const float4 near = u.invViewProj * float4(ndc, 0.0, 1.0);
        origin = near.xyz / near.w;
        direction = normalize(origin - u.cameraPos.xyz);
    }
    if (any(isnan(direction))) direction = float3(0.0, 0.0, -1.0);
    // Denoiser guides: the first surface seen (through glass).
    float3 firstAlbedo = float3(0.0);
    float3 firstNormal = float3(0.0);
    bool guided = false;

    intersector<triangle_data, instancing> isect;
    isect.assume_geometry_type(geometry_type::triangle);
    isect.force_opacity(forced_opacity::opaque);
    intersector<triangle_data, instancing> shadowIsect;
    shadowIsect.assume_geometry_type(geometry_type::triangle);
    shadowIsect.force_opacity(forced_opacity::opaque);
    shadowIsect.accept_any_intersection(true);

    float3 radiance = float3(0.0);
    float3 throughput = float3(1.0);
    const uint maxBounces = uint(u.params.z);
    // Panes of glass don't spend bounces (a view into a room crosses one
    // twice: in, and out to the sky), up to this many crossings per path.
    uint glassLeft = 8u;
    const float envIntensity = u.params.x;
    for (uint bounce = 0; bounce <= maxBounces; ++bounce) {
        ray r;
        r.origin = origin;
        r.direction = direction;
        r.min_distance = 1e-4;
        r.max_distance = 1e30;
        const auto hit = isect.intersect(r, scene, MASK_OPAQUE | MASK_GLASS);
        if (hit.type == intersection_type::none) {
            if (bounce == 0 && u.size.z == 0u) {
                radiance += throughput * u.clearColor.rgb;
            } else {
                // Seen directly: the sky as the skybox draws it; as light:
                // scaled by the environment intensity.
                const float scale = bounce == 0 ? 1.0 : envIntensity;
                const float3 sky = environment.sample(envSampler, direction, level(0.0)).rgb * scale;
                radiance += throughput * sky;
                if (!guided) firstAlbedo = clamp(sky, 0.0, 1.0);
            }
            if (!guided && bounce == 0 && u.size.z == 0u) firstAlbedo = clamp(u.clearColor.rgb, 0.0, 1.0);
            guided = true;
            break;
        }
        const InstanceInfo inst = instances[hit.instance_id];
        const MaterialInfo mat = materials[inst.material];
        const float3 objectNormal = normals[inst.normalOffset + hit.primitive_id].xyz;
        float3 n = normalize(float3(dot(inst.normalMatrix0.xyz, objectNormal), dot(inst.normalMatrix1.xyz, objectNormal),
                                    dot(inst.normalMatrix2.xyz, objectNormal)));
        if (dot(n, direction) > 0.0) n = -n;  // two-sided: towards the incoming ray
        const float3 p = origin + direction * hit.distance;
        const float3 v = -direction;
        float3 albedo = mat.baseColor.rgb;
        if (mat.texture >= 0 && uint(mat.texture) < MAX_TEXTURES) {
            float2 uv;
            bool mapped = true;
            if (mat.worldMapping > 0.0) {
                uv = worldMappingUV(p, n, mat.worldMapping, mat.worldMappingUp > 0.5);
            } else {
                const uint tri = inst.normalOffset + hit.primitive_id;
                const float4 a = uvs[tri * 2];
                const float4 b = uvs[tri * 2 + 1];
                mapped = b.z > 0.5;
                const float2 bary = hit.triangle_barycentric_coord;
                uv = a.xy * (1.0 - bary.x - bary.y) + a.zw * bary.x + b.xy * bary.y;
                if (mat.uvRow0.w > 0.5) {
                    const float3 uv3 = float3(uv, 1.0);
                    uv = float2(dot(mat.uvRow0.xyz, uv3), dot(mat.uvRow1.xyz, uv3));
                }
            }
            // Meshes without texture coordinates take its average colour.
            albedo = mapped ? mat.baseColor.rgb * textures[mat.texture].sample(texSampler, uv, level(0.0)).rgb
                            : mat.baseColor.rgb * mat.average.rgb;
        }
        if (mat.unlit > 0.5) {
            radiance += throughput * albedo;
            if (!guided) {
                firstAlbedo = clamp(albedo, 0.0, 1.0);
                firstNormal = n;
                guided = true;
            }
            break;
        }
        radiance += throughput * mat.emissive.rgb;

        // Thin glass: Fresnel reflection, else straight through, tinted.
        if (mat.transmission > 0.0) {
            if (glassLeft > 0u) {
                --glassLeft;
                --bounce;  // unsigned: wraps and the loop's ++ brings it back
            }
            const float f0 = pow((mat.ior - 1.0) / (mat.ior + 1.0), 2.0);
            const float F = f0 + (1.0 - f0) * pow(clamp(1.0 - dot(n, v), 0.0, 1.0), 5.0);
            const float transmit = mat.transmission * (1.0 - F);
            if (rand01(rng) < transmit) {
                throughput *= albedo;
                origin = p + direction * 1e-3;
            } else {
                direction = reflect(direction, n);
                origin = p + n * 1e-3;
            }
            continue;
        }

        if (!guided) {
            firstAlbedo = clamp(albedo, 0.0, 1.0);
            firstNormal = n;
            guided = true;
        }
        const float alpha = max(mat.roughness * mat.roughness, 0.002);
        const float3 f0 = mix(float3(0.04), albedo, mat.metallic);

        // Directional lights, sampled directly: a point of each one's disc,
        // a shadow ray.
        for (uint light = 0; light < 4; ++light) {
            if (u.sunRadiance[light].w < 0.5) continue;
            float3 st, sb;
            basis(u.sunDirection[light].xyz, st, sb);
            const float cosMax = u.sunDirection[light].w;
            const float cosT = 1.0 - rand01(rng) * (1.0 - cosMax);
            const float sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
            const float phi = 2.0 * PI * rand01(rng);
            const float3 l = normalize(st * (sinT * cos(phi)) + sb * (sinT * sin(phi)) + u.sunDirection[light].xyz * cosT);
            const float NdotL = dot(n, l);
            if (NdotL > 0.0) {
                ray s;
                s.origin = p + n * 1e-3;
                s.direction = l;
                s.min_distance = 1e-4;
                s.max_distance = 1e30;
                const auto blocked = shadowIsect.intersect(s, scene, MASK_OPAQUE);
                if (blocked.type == intersection_type::none) {
                    radiance += throughput * brdf(n, v, l, albedo, mat.metallic, alpha, f0) * u.sunRadiance[light].rgb * NdotL;
                }
            }
        }

        // Next direction: specular or diffuse, by the Fresnel-weighted chance.
        const float3 Fv = fresnelSchlick(f0, max(dot(n, v), 0.0));
        const float specularWeight = clamp((Fv.r + Fv.g + Fv.b) / 3.0 + mat.metallic * 0.5, 0.05, 0.95);
        if (rand01(rng) < specularWeight) {
            const float3 h = ggxHalfVector(n, alpha, rand01(rng), rand01(rng));
            const float3 l = reflect(-v, h);
            const float NdotL = dot(n, l);
            if (NdotL <= 0.0) break;
            const float NdotV = max(dot(n, v), 1e-4);
            const float VdotH = max(dot(v, h), 1e-4);
            const float NdotH = max(dot(n, h), 1e-4);
            // Weight of a GGX-sampled direction: F G VdotH / (NdotV NdotH).
            const float3 F = fresnelSchlick(f0, VdotH);
            const float G = smithG1(NdotV, alpha) * smithG1(NdotL, alpha);
            throughput *= F * G * VdotH / (NdotV * NdotH) / specularWeight;
            direction = l;
        } else {
            direction = cosineSample(n, rand01(rng), rand01(rng));
            throughput *= (1.0 - fresnelSchlick(f0, max(dot(n, v), 0.0))) * (1.0 - mat.metallic) * albedo / (1.0 - specularWeight);
        }
        origin = p + n * 1e-3;

        // Russian roulette after two bounces.
        if (bounce >= 2) {
            const float survive = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.05, 0.95);
            if (rand01(rng) > survive) break;
            throughput /= survive;
        }
    }
    if (any(isnan(radiance)) || any(isinf(radiance))) radiance = float3(0.0);

    const uint pixel = gid.y * u.size.x + gid.x;
    const float4 sum = (sampleIndex == 0 ? float4(0.0) : accum[pixel]) + float4(radiance, 1.0);
    accum[pixel] = sum;
    const uint pixels = u.size.x * u.size.y;
    aux[pixel] = (sampleIndex == 0 ? float4(0.0) : aux[pixel]) + float4(firstAlbedo, 0.0);
    aux[pixels + pixel] = (sampleIndex == 0 ? float4(0.0) : aux[pixels + pixel]) + float4(firstNormal, 0.0);
    float3 color = sum.rgb / max(sum.w, 1.0) * exp2(u.params.y);
    color = pow(max(toneMapKhronosPbrNeutral(color), 0.0), float3(1.0 / 2.2));
    output.write(float4(color, 1.0), gid);
}
