// Path tracer (PathTracer, path_tracer.hpp): progressive, one sample per
// pixel per dispatch, accumulated in a float buffer and shown tone mapped.
//
//   faceNormals   one geometric normal per triangle of a mesh (object
//                 space), from its own vertex and index buffers, into the
//                 scene's normal buffer at the mesh's offset; and its three
//                 texture coordinates (TEXCOORD_0) into the uv buffer.
//   environmentLuminance  the environment's brightness per texel of a
//                 low resolution cube: the weights of its sampling table.
//   pathTrace     camera paths through the scene's instance acceleration
//                 structure: diffuse + GGX specular, the sun sampled
//                 directly (a soft 0.27 degree disc, shadow rays), the sky
//                 from the environment cube, sampled by its brightness as
//                 a light and combined with the BSDF's directions by
//                 multiple importance sampling, thin glass (Fresnel
//                 reflection or straight through), Russian roulette,
//                 optional clamping of indirect light (fireflies), Owen-
//                 scrambled Sobol random numbers. Also the first
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
    uint normalFormat;  // 0 none (face normals), 1 float3
    uint normalStride;  // bytes between normals
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
                        device const uchar*   vertexNormalsIn [[buffer(6)]],
                        device float4*        vertexNormals   [[buffer(7)]],
                        device float4*        tangents        [[buffer(8)]],
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
    const float3 face = len > 1e-20 ? n / len : float3(0.0, 0.0, 1.0);
    normals[p.outOffset + id] = float4(face, 0.0);
    // The corners' normals (smooth shading), the face's without them.
    for (uint k = 0; k < 3; ++k) {
        float3 vn = face;
        if (p.normalFormat == 1u) {
            const float3 read = *(device const packed_float3*)(vertexNormalsIn + v[k] * p.normalStride);
            const float l = length(read);
            if (l > 1e-12) vn = read / l;
        }
        vertexNormals[(p.outOffset + id) * 3 + k] = float4(vn, 0.0);
    }
    // A tangent along the texture's u (normal maps), w its handedness.
    float4 tangent = float4(0.0, 0.0, 0.0, 1.0);
    if (p.uvFormat != 0u) {
        const float2 d1 = readUV(texcoords, v[1], p) - readUV(texcoords, v[0], p);
        const float2 d2 = readUV(texcoords, v[2], p) - readUV(texcoords, v[0], p);
        const float r = d1.x * d2.y - d2.x * d1.y;
        if (abs(r) > 1e-20) {
            const float3 e1 = b - a, e2 = c - a;
            const float3 T = (e1 * d2.y - e2 * d1.y) / r;
            const float3 B = (e2 * d1.x - e1 * d2.x) / r;
            if (length(T) > 1e-20) tangent = float4(normalize(T), dot(cross(face, T), B) < 0.0 ? -1.0 : 1.0);
        }
    }
    tangents[p.outOffset + id] = tangent;
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
    float4   params;        // x environment intensity, y exposure (EV), z max bounces, w first sample number
    uint4    size;          // width, height, sky visible (1) or clear colour, mode (0 perspective, 1 panorama, 2 reflections from the G-buffer)
    float4   clearColor;    // rgb when the sky isn't visible
    float4   cameraForward; // panorama: the image centre's direction (unit)
    float4   cameraRight;   // panorama: right (unit, perpendicular to forward)
    float4   cameraUp;      // panorama: up, the poles
    float4   tuning;        // x indirect clamp (0 off), y sampler (0 random, 1 Sobol), z noise target (0 off), w its minimum samples
    uint4    envSampling;   // x environment distribution resolution per cube face (0: none), y light portals, z resolution divisor (preview: 1 / z of the output's), w samples per pixel in this dispatch
    uint4    lights;        // x emissive triangles sampled as lights, y first row of this dispatch (a band of a sample)
    float4   camera;        // x lens aperture radius (0: a pinhole), y focus distance, z tone mapping (0 PBR Neutral, 1 ACES, 2 AgX)
    // Probe rays (mode 3, PathTracer::traceProbeRays()): a grid of probes,
    // each tracing rays along rotated spherical Fibonacci directions.
    float4   probeOrigin;   // xyz: probe (0, 0, 0)
    float4   probeSpacing;  // xyz: between probes
    uint4    probeCount;    // xyz: probes per axis, w: rays per probe
    float4x4 probeRotation; // this frame's rotation of the directions
};

struct InstanceInfo {
    float4 normalMatrix0;  // rows of the world normal matrix (inverse transpose of the model's 3x3)
    float4 normalMatrix1;
    float4 normalMatrix2;
    uint   normalOffset;   // first triangle in the normal buffer
    uint   material;
    uint   lightOffset;    // its triangles' first entry in the emissive light list + 1 (0: none)
    uint   pad1;
    float4 model0;         // rows of the model's 3x3 (tangents)
    float4 model1;
    float4 model2;
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
    int    normalTexture;   // in `textures`, -1 none
    int    mrTexture;       // metallic-roughness (g roughness, b metallic), -1 none
    float  normalScale;
    float  doubleSided;     // 1: emits from both sides (else the front only, as glTF)
    float4 normalUvRow0;    // the normal texture's own KHR_texture_transform (a, b, tx, has)
    float4 normalUvRow1;
    float  alphaCutoff;     // glTF MASK: hits count where alpha >= it (0: opaque)
    float  pad0, pad1, pad2;
};

constant uint MAX_TEXTURES = 64;  // base colour textures bound (path_tracer.cpp)

// As worldMappingFrame() in default.metal (the raster renderer's box
// mapping), so traced and drawn textures line up.
inline void worldMappingFrame(float3 N, bool zUp, thread float3& T, thread float3& B) {
    const float3 U = zUp ? float3(0.0, 0.0, 1.0) : float3(0.0, 1.0, 0.0);
    const float nu = dot(N, U);
    if (abs(nu) > 0.7071) {
        T = float3(1.0, 0.0, 0.0);
        B = zUp ? float3(0.0, -1.0, 0.0) : float3(0.0, 0.0, 1.0);
    } else {
        const float3 h = normalize(N - U * nu);
        T = cross(U, h);
        B = -U;
    }
}

inline float2 worldMappingUV(float3 P, float3 N, float size, bool zUp) {
    float3 T, B;
    worldMappingFrame(N, zUp, T, B);
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
    return min(float(pcg(state)) * (1.0 / 4294967296.0), 0x1.fffffep-1);
}

// Random numbers of a path: Owen-scrambled Sobol points (Burley 2020,
// "Practical Hash-based Owen Scrambling"), as 2D pairs padded across the
// path's dimensions: every pair has its own scramble and index shuffle,
// per pixel. Or plain PCG (mode 0).
inline uint hashU(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}
inline uint sobolSecond(uint i) {  // the second Sobol dimension (the first is reverse_bits)
    uint r = 0u;
    for (uint v = 1u << 31; i != 0u; i >>= 1, v ^= v >> 1) {
        if (i & 1u) r ^= v;
    }
    return r;
}
inline uint laineKarras(uint x, uint seed) {
    x += seed;
    x ^= x * 0x6c50b47cu;
    x ^= x * 0xb82f1e52u;
    x ^= x * 0xc7afe638u;
    x ^= x * 0x8d22f6e6u;
    return x;
}
inline uint owenScramble(uint x, uint seed) {
    return reverse_bits(laineKarras(reverse_bits(x), seed));
}

struct PathSampler {
    uint pixelSeed;
    uint index;  // the sample number
    uint mode;   // 0 PCG, 1 Sobol scrambled per pixel
    uint rng;    // PCG state (mode 0)
};

// Pair `dim` of the path's random numbers, in [0, 1).
inline float2 sample2D(thread PathSampler& s, uint dim) {
    if (s.mode == 0u) {
        const float a = rand01(s.rng);
        return float2(a, rand01(s.rng));
    }
    const uint seed = hashU(s.pixelSeed ^ hashU(dim * 0x9e3779b9u + 0x632be5abu));
    const uint i = owenScramble(s.index, seed);
    const uint x = owenScramble(reverse_bits(i), hashU(seed ^ 0xa511e9b3u));
    const uint y = owenScramble(sobolSecond(i), hashU(seed ^ 0x63d83595u));
    return min(float2(float(x), float(y)) * (1.0 / 4294967296.0), float2(0x1.fffffep-1));
}

// Pairs of a path: the camera jitter, then these per path vertex.
constant uint DIM_CAMERA = 0u;
// Each sampling strategy takes one pair (a choice's leftover fraction
// reused as a coordinate): its estimate is then a function of a single 2D
// point, which Sobol stratifies as a whole (2D nets).
constant uint DIMS_PER_VERTEX = 8u;
constant uint DIM_BSDF = 0u;      // x: lobe (or glass) choice, its remainder and y: the direction
constant uint DIM_ROULETTE = 1u;  // x: Russian roulette
constant uint DIM_ENV = 2u;       // the sky's light sample: x strategy, portal or texel, their remainder and y the point
constant uint DIM_LIGHTS = 3u;    // 3..6: the directional lights
constant uint DIM_EMISSIVE = 7u;  // an emissive triangle: x which (its remainder: a point coordinate), y the other
constant uint DIM_LENS = 1u;  // a point on the lens (depth of field)
inline uint vertexDim(uint pathVertex, uint offset) {
    return 2u + pathVertex * DIMS_PER_VERTEX + offset;
}

// ---------------------------------------------------------------------------
// Environment importance sampling
// ---------------------------------------------------------------------------
// The environment cube parameterised as six faces of R x R texels over
// uv in [-1, 1]^2 (our own layout: only needs to be a bijection, it doesn't
// have to match the hardware's face order). Texels are picked by inverting
// their cumulative distribution, built from environmentLuminance()
// (path_tracer.cpp): monotonic, so stratified random numbers (Sobol) stay
// stratified in the choice of texel, which an alias table would scramble.
struct EnvTexel {
    float cdf;  // probability of this texel and the ones before it
    float pdf;  // this texel's probability
};

inline float3 cubeDirection(uint face, float2 uv) {
    switch (face) {
        case 0u: return float3(1.0, -uv.y, -uv.x);
        case 1u: return float3(-1.0, -uv.y, uv.x);
        case 2u: return float3(uv.x, 1.0, uv.y);
        case 3u: return float3(uv.x, -1.0, -uv.y);
        case 4u: return float3(uv.x, -uv.y, 1.0);
        default: return float3(-uv.x, -uv.y, -1.0);
    }
}

inline uint cubeFace(float3 d, thread float2& uv) {
    const float3 a = abs(d);
    if (a.x >= a.y && a.x >= a.z) {
        uv = float2(d.x > 0.0 ? -d.z : d.z, -d.y) / a.x;
        return d.x > 0.0 ? 0u : 1u;
    }
    if (a.y >= a.z) {
        uv = float2(d.x, d.y > 0.0 ? d.z : -d.z) / a.y;
        return d.y > 0.0 ? 2u : 3u;
    }
    uv = float2(d.z > 0.0 ? d.x : -d.x, -d.y) / a.z;
    return d.z > 0.0 ? 4u : 5u;
}

// Solid angle density of the distribution in direction d.
inline float environmentPdf(device const EnvTexel* table, uint R, float3 d) {
    float2 uv;
    const uint face = cubeFace(d, uv);
    const uint2 t = min(uint2((uv * 0.5 + 0.5) * float(R)), uint2(R - 1u));
    const float p = table[(face * R + t.y) * R + t.x].pdf;
    return p * float(R * R) * 0.25 * pow(1.0 + dot(uv, uv), 1.5);
}

inline float3 sampleEnvironment(device const EnvTexel* table, uint R, float2 sample, thread float& pdf) {
    // The first texel whose cumulative probability exceeds u; where in
    // that texel's step u fell, the point's first coordinate.
    const float u = sample.x;
    uint lo = 0u;
    uint hi = 6u * R * R - 1u;
    while (lo < hi) {
        const uint mid = (lo + hi) / 2u;
        if (table[mid].cdf > u) hi = mid;
        else lo = mid + 1u;
    }
    const uint t = lo;
    const float before = t > 0u ? table[t - 1u].cdf : 0.0;
    const float2 inTexel = float2(clamp((u - before) / max(table[t].pdf, 1e-30), 0.0, 0x1.fffffep-1), sample.y);
    const uint face = t / (R * R);
    const uint rest = t - face * R * R;
    const float2 uv = (float2(float(rest % R), float(rest / R)) + inTexel) / float(R) * 2.0 - 1.0;
    pdf = table[t].pdf * float(R * R) * 0.25 * pow(1.0 + dot(uv, uv), 1.5);
    return normalize(cubeDirection(face, uv));
}

// ---------------------------------------------------------------------------
// Light portals: rectangles (windows) the sky's light comes through
// ---------------------------------------------------------------------------
// A portal is picked by its solid angle from the shading point, then a
// direction uniformly within it (Urena, Fajardo & King 2013, "An
// Area-Preserving Parametrization for Spherical Rectangles"); two-sided.
// Mixed with the environment table (PORTAL_SHARE of the light samples),
// so sky that isn't behind a portal is still sampled.
struct Portal {
    float4 corner;  // xyz: a corner (world)
    float4 axisX;   // xyz: unit edge direction, w: its length
    float4 axisY;   // xyz: unit edge direction (perpendicular to axisX), w: its length
};

constant uint MAX_PORTALS = 32u;          // path_tracer.cpp
constant float PORTAL_SHARE = 0.8;        // of the light samples, where both kinds exist
constant float PORTAL_MIN_SOLID_ANGLE = 1e-4;  // smaller (far) portals: left to the table (float precision)

struct SphericalRect {
    float3 o, x, y, z;
    float z0, x0, y0, x1, y1, b0, b1, k, S;
};

inline SphericalRect sphericalRect(Portal portal, float3 o) {
    SphericalRect r;
    r.o = o;
    r.x = portal.axisX.xyz;
    r.y = portal.axisY.xyz;
    r.z = cross(r.x, r.y);
    const float3 d = portal.corner.xyz - o;
    r.z0 = dot(d, r.z);
    if (r.z0 > 0.0) {
        r.z = -r.z;
        r.z0 = -r.z0;
    }
    r.x0 = dot(d, r.x);
    r.y0 = dot(d, r.y);
    r.x1 = r.x0 + portal.axisX.w;
    r.y1 = r.y0 + portal.axisY.w;
    r.S = 0.0;
    r.b0 = r.b1 = r.k = 0.0;
    if (r.z0 > -1e-6) return r;  // in the portal's plane: no solid angle
    const float3 v00 = float3(r.x0, r.y0, r.z0);
    const float3 v01 = float3(r.x0, r.y1, r.z0);
    const float3 v10 = float3(r.x1, r.y0, r.z0);
    const float3 v11 = float3(r.x1, r.y1, r.z0);
    const float3 n0 = normalize(cross(v00, v10));
    const float3 n1 = normalize(cross(v10, v11));
    const float3 n2 = normalize(cross(v11, v01));
    const float3 n3 = normalize(cross(v01, v00));
    const float g0 = acos(clamp(-dot(n0, n1), -1.0, 1.0));
    const float g1 = acos(clamp(-dot(n1, n2), -1.0, 1.0));
    const float g2 = acos(clamp(-dot(n2, n3), -1.0, 1.0));
    const float g3 = acos(clamp(-dot(n3, n0), -1.0, 1.0));
    r.b0 = n0.z;
    r.b1 = n2.z;
    r.k = 2.0 * PI - g2 - g3;
    r.S = max(g0 + g1 - r.k, 0.0);
    return r;
}

// A direction towards a uniformly distributed point (in solid angle) of the rectangle.
inline float3 sampleSphericalRect(thread const SphericalRect& r, float2 u) {
    const float au = u.x * r.S + r.k;
    const float sau = sin(au);
    const float fu = (cos(au) * r.b0 - r.b1) / (abs(sau) > 1e-12 ? sau : 1e-12);
    const float cu = clamp((fu > 0.0 ? 1.0 : -1.0) / sqrt(fu * fu + r.b0 * r.b0), -1.0, 1.0);
    const float xu = clamp(-(cu * r.z0) / max(sqrt(1.0 - cu * cu), 1e-12), r.x0, r.x1);
    const float d2 = xu * xu + r.z0 * r.z0;
    const float h0 = r.y0 / sqrt(d2 + r.y0 * r.y0);
    const float h1 = r.y1 / sqrt(d2 + r.y1 * r.y1);
    const float hv = h0 + u.y * (h1 - h0);
    const float hv2 = hv * hv;
    const float yv = hv2 < 1.0 - 1e-6 ? hv * sqrt(d2) / sqrt(1.0 - hv2) : r.y1;
    return normalize(xu * r.x + yv * r.y + r.z0 * r.z);
}

inline bool portalHit(Portal portal, float3 o, float3 l) {
    const float3 nz = cross(portal.axisX.xyz, portal.axisY.xyz);
    const float denom = dot(l, nz);
    if (abs(denom) < 1e-8) return false;
    const float t = dot(portal.corner.xyz - o, nz) / denom;
    if (t <= 0.0) return false;
    const float3 q = o + l * t - portal.corner.xyz;
    const float a = dot(q, portal.axisX.xyz);
    const float b = dot(q, portal.axisY.xyz);
    return a >= 0.0 && a <= portal.axisX.w && b >= 0.0 && b <= portal.axisY.w;
}

// The sky's light samples: the environment table and the portals.
struct SkyLights {
    device const EnvTexel* table;
    uint envR;  // 0: no table
    device const Portal* portals;
    uint portalCount;
};

inline float portalSolidAngle(thread const SkyLights& sky, uint i, float3 o) {
    const float S = sphericalRect(sky.portals[i], o).S;
    return S >= PORTAL_MIN_SOLID_ANGLE ? S : 0.0;
}

// The share of portal samples at o, given the portals' total solid angle there.
inline float portalShare(thread const SkyLights& sky, float total) {
    if (total <= 0.0) return 0.0;
    return sky.envR > 0u ? PORTAL_SHARE : 1.0;
}

// Solid angle density of the sky's light samples at o in direction l.
inline float skyLightPdf(thread const SkyLights& sky, float3 o, float3 l) {
    float total = 0.0;
    float hit = 0.0;
    for (uint i = 0; i < sky.portalCount; ++i) {
        const float S = portalSolidAngle(sky, i, o);
        total += S;
        if (S > 0.0 && portalHit(sky.portals[i], o, l)) hit += 1.0;
    }
    const float share = portalShare(sky, total);
    float pdf = share > 0.0 ? share * hit / total : 0.0;
    if (sky.envR > 0u && share < 1.0) pdf += (1.0 - share) * environmentPdf(sky.table, sky.envR, l);
    return pdf;
}

// A light sample of the sky from o (one 2D pair: the strategy, the portal
// and the direction's first coordinate share u.x), its density as
// skyLightPdf() gives it.
inline float3 sampleSkyLight(thread const SkyLights& sky, float3 o, float2 u, thread float& pdf) {
    float total = 0.0;
    for (uint i = 0; i < sky.portalCount; ++i) total += portalSolidAngle(sky, i, o);
    const float share = portalShare(sky, total);
    if (share <= 0.0 && sky.envR == 0u) {
        pdf = 0.0;  // no portal in sight, no table: nothing to sample here
        return float3(0.0, 0.0, 1.0);
    }
    float3 l;
    if (u.x < share) {
        // A portal by its solid angle, the remainder for the direction.
        float x = u.x / share * total;
        uint chosen = 0u;
        float S = 0.0;
        for (uint i = 0; i < sky.portalCount; ++i) {
            S = portalSolidAngle(sky, i, o);
            chosen = i;
            if (S > 0.0 && x < S) break;
            x -= S;
        }
        const SphericalRect r = sphericalRect(sky.portals[chosen], o);
        l = sampleSphericalRect(r, float2(clamp(x / max(S, 1e-30), 0.0, 0x1.fffffep-1), u.y));
    } else {
        float envPdf;
        l = sampleEnvironment(sky.table, sky.envR, float2(min((u.x - share) / (1.0 - share), 0x1.fffffep-1), u.y), envPdf);
    }
    pdf = skyLightPdf(sky, o, l);
    return l;
}

struct EnvParams {
    uint resolution;
    uint pad0;
    uint pad1;
    uint pad2;
};

// Each texel's luminance x solid angle (up to a constant): the weights of
// the sampling table. 4 x 4 samples per texel, so small bright spots (the sun
// in an HDR sky) aren't missed between texel centres. One thread per
// texel, a 6R x R grid (the faces side by side). Dispatched as a ray
// tracing pass (path_tracer.cpp): campello_gpu's Metal compute passes
// don't bind textures.
kernel void environmentLuminance(texturecube<float> environment [[texture(0)]],
                                 device float*      weights     [[buffer(0)]],
                                 constant EnvParams& p          [[buffer(1)]],
                                 uint2 gid [[thread_position_in_grid]])
{
    const uint R = p.resolution;
    if (gid.x >= 6u * R || gid.y >= R) return;
    constexpr sampler s(filter::linear, mip_filter::linear);
    const uint face = gid.x / R;
    const uint id = (face * R + gid.y) * R + (gid.x - face * R);
    const float2 texel = float2(float(gid.x - face * R), float(gid.y));
    const float lod = max(log2(float(environment.get_width()) / float(R * 4u)), 0.0);
    float sum = 0.0;
    for (uint j = 0; j < 16u; ++j) {
        const float2 uv = (texel + (float2(float(j % 4u), float(j / 4u)) + 0.5) * 0.25) / float(R) * 2.0 - 1.0;
        const float3 c = max(environment.sample(s, normalize(cubeDirection(face, uv)), level(lod)).rgb, 0.0);
        sum += dot(c, float3(0.2126, 0.7152, 0.0722)) / pow(1.0 + dot(uv, uv), 1.5);
    }
    weights[id] = sum / 16.0;
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

// ACES filmic (Stephen Hill's fit of the RRT + ODT, sRGB primaries).
inline float3 toneMapAces(float3 color) {
    const float3 i0 = float3(0.59719, 0.35458, 0.04823), i1 = float3(0.07600, 0.90834, 0.01566),
                 i2 = float3(0.02840, 0.13383, 0.83777);
    float3 v = float3(dot(i0, color), dot(i1, color), dot(i2, color));
    v = (v * (v + 0.0245786) - 0.000090537) / (v * (0.983729 * v + 0.4329510) + 0.238081);
    const float3 o0 = float3(1.60475, -0.53108, -0.07367), o1 = float3(-0.10208, 1.10813, -0.00605),
                 o2 = float3(-0.00327, -0.07276, 1.07602);
    return clamp(float3(dot(o0, v), dot(o1, v), dot(o2, v)), 0.0, 1.0);
}

// AgX (Troy Sobotka's, the polynomial fit of its default contrast):
// linear in, linear out (display-encoded inside, decoded with 2.2).
inline float3 toneMapAgx(float3 color) {
    const float3 i0 = float3(0.842479062253094, 0.0784335999999992, 0.0792237451477643),
                 i1 = float3(0.0423282422610123, 0.878468636469772, 0.0791661274605434),
                 i2 = float3(0.0423756549057051, 0.0784336, 0.879142973793104);
    float3 v = float3(dot(i0, color), dot(i1, color), dot(i2, color));
    const float minEv = -12.47393, maxEv = 4.026069;
    v = (clamp(log2(max(v, 1e-10)), minEv, maxEv) - minEv) / (maxEv - minEv);
    const float3 v2 = v * v, v4 = v2 * v2;
    v = 15.5 * v4 * v2 - 40.14 * v4 * v + 31.96 * v4 - 6.868 * v2 * v + 0.4298 * v2 + 0.1191 * v - 0.00232;
    const float3 o0 = float3(1.19687900512017, -0.0980208811401368, -0.0990297440797205),
                 o1 = float3(-0.0528968517574562, 1.15190312990417, -0.0989611768448433),
                 o2 = float3(-0.0529716355144438, -0.0980434501171241, 1.15107367264116);
    v = float3(dot(o0, v), dot(o1, v), dot(o2, v));
    return pow(clamp(v, 0.0, 1.0), float3(2.2));
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

// The BRDF as the path's direction sampling estimates it (the diffuse
// lobe weighted by 1 - F(n.v), not 1 - F(v.h) as brdf()): what the
// environment's light samples must agree with for multiple importance
// sampling to add up.
inline float3 brdfSampled(float3 n, float3 v, float3 l, float3 albedo, float metallic, float alpha, float3 f0) {
    const float NdotL = dot(n, l);
    const float NdotV = max(dot(n, v), 1e-4);
    if (NdotL <= 0.0) return float3(0.0);
    const float3 h = normalize(v + l);
    const float NdotH = max(dot(n, h), 0.0);
    const float3 F = fresnelSchlick(f0, max(dot(v, h), 0.0));
    const float3 specular = F * ggxD(NdotH, alpha) * smithG1(NdotV, alpha) * smithG1(NdotL, alpha) / (4.0 * NdotV * NdotL);
    const float3 diffuse = (1.0 - fresnelSchlick(f0, NdotV)) * (1.0 - metallic) * albedo / PI;
    return diffuse + specular;
}

// Solid angle density of the path's next direction l: GGX half vectors
// with probability specularWeight, else cosine-weighted.
inline float bsdfPdf(float3 n, float3 v, float3 l, float alpha, float specularWeight) {
    const float NdotL = dot(n, l);
    if (NdotL <= 0.0) return 0.0;
    const float3 h = normalize(v + l);
    const float NdotH = max(dot(n, h), 1e-4);
    const float VdotH = max(dot(v, h), 1e-4);
    return specularWeight * ggxD(NdotH, alpha) * NdotH / (4.0 * VdotH) + (1.0 - specularWeight) * NdotL / PI;
}

inline float powerHeuristic(float a, float b) {
    const float a2 = a * a;
    const float b2 = b * b;
    return a2 + b2 > 0.0 ? a2 / (a2 + b2) : 0.0;
}

// Surfaces this smooth pass the denoiser guides on to what they reflect.
constant float MIRROR_ROUGHNESS = 0.15;

// Instance masks: opaque geometry casts sun shadows, glass doesn't.
constant uint MASK_OPAQUE = 1u;
constant uint MASK_GLASS  = 2u;

inline float maxComponent(float3 c) {
    return max(c.r, max(c.g, c.b));
}

// ---------------------------------------------------------------------------
// Emissive triangles as lights
// ---------------------------------------------------------------------------
struct EmissiveTri {
    float4 v0;        // xyz: a corner (world)
    float4 e1;        // xyz: to the second corner
    float4 e2;        // xyz: to the third corner
    float4 radiance;  // rgb: emitted radiance, w: area
};

struct EmissiveParams {
    uint   triangleCount;
    uint   outOffset;
    uint   indexFormat;   // 0 none, 1 uint16, 2 uint32
    uint   vertexStride;
    float4 model0;        // rows of the model matrix (with translation)
    float4 model1;
    float4 model2;
    float4 radiance;      // rgb; w: 1 when double-sided
};

// An emissive instance's triangles in world space, for the light list:
// e1 x e2 along the front face's normal (its winding flipped under a
// mirroring transform, as glTF has it); e1.w 1 when it emits from both sides.
kernel void emissiveTriangles(device const uchar*     positions [[buffer(0)]],
                              device const uchar*     indices   [[buffer(1)]],
                              device EmissiveTri*     out       [[buffer(2)]],
                              constant EmissiveParams& p        [[buffer(3)]],
                              uint id [[thread_position_in_grid]])
{
    if (id >= p.triangleCount) return;
    float3 w[3];
    for (uint k = 0; k < 3; ++k) {
        const uint i = id * 3 + k;
        uint vi = i;
        if (p.indexFormat == 1u) vi = ((device const ushort*)indices)[i];
        else if (p.indexFormat == 2u) vi = ((device const uint*)indices)[i];
        const float4 o = float4(*(device const packed_float3*)(positions + vi * p.vertexStride), 1.0);
        w[k] = float3(dot(p.model0, o), dot(p.model1, o), dot(p.model2, o));
    }
    float3 e1 = w[1] - w[0], e2 = w[2] - w[0];
    if (determinant(float3x3(p.model0.xyz, p.model1.xyz, p.model2.xyz)) < 0.0) {
        const float3 t = e1;  // mirrored: the front is the other way round
        e1 = e2;
        e2 = t;
    }
    out[p.outOffset + id] = EmissiveTri{float4(w[0], 0.0), float4(e1, p.radiance.w), float4(e2, 0.0),
                                        float4(p.radiance.rgb, 0.5 * length(cross(e1, e2)))};
}

// The first entry of a cumulative table (EnvTexel: cdf, pdf) whose
// cumulative probability exceeds u, and where in its step u fell.
inline uint invertCdf(device const EnvTexel* table, uint count, float u, thread float& remainder) {
    uint lo = 0u;
    uint hi = count - 1u;
    while (lo < hi) {
        const uint mid = (lo + hi) / 2u;
        if (table[mid].cdf > u) hi = mid;
        else lo = mid + 1u;
    }
    const float before = lo > 0u ? table[lo - 1u].cdf : 0.0;
    remainder = clamp((u - before) / max(table[lo].pdf, 1e-30), 0.0, 0x1.fffffep-1);
    return lo;
}

// Alpha-masked materials (glTF MASK; their instances aren't opaque): a
// candidate hit counts only where the base colour's alpha reaches the
// cutoff, as the raster shader discards.
struct AlphaScene {
    device const InstanceInfo* instances;
    device const MaterialInfo* materials;
    device const float4*       uvs;
};

inline bool alphaPasses(AlphaScene a, const array<texture2d<float>, MAX_TEXTURES> textures, uint instance, uint primitive,
                        float2 bary) {
    const InstanceInfo inst = a.instances[instance];
    const MaterialInfo mat = a.materials[inst.material];
    if (mat.alphaCutoff <= 0.0) return true;
    float alpha = mat.baseColor.a;
    if (mat.texture >= 0 && uint(mat.texture) < MAX_TEXTURES && mat.worldMapping <= 0.0) {
        const uint tri = inst.normalOffset + primitive;
        const float4 ta = a.uvs[tri * 2];
        const float4 tb = a.uvs[tri * 2 + 1];
        if (tb.z > 0.5) {
            const float3 corner = float3(1.0 - bary.x - bary.y, bary.x, bary.y);
            float2 uv = ta.xy * corner.x + ta.zw * corner.y + tb.xy * corner.z;
            if (mat.uvRow0.w > 0.5) uv = float2(dot(mat.uvRow0.xyz, float3(uv, 1.0)), dot(mat.uvRow1.xyz, float3(uv, 1.0)));
            constexpr sampler alphaSampler(filter::linear, mip_filter::linear, address::repeat);
            alpha *= textures[mat.texture].sample(alphaSampler, uv, level(0.0)).a;
        }
    }
    return alpha >= mat.alphaCutoff;
}

// The intersector's result, from an intersection query that alpha tests.
struct TraceHit {
    intersection_type type;
    float  distance;
    uint   instance_id;
    uint   primitive_id;
    float2 triangle_barycentric_coord;
};

inline TraceHit traceRay(ray r, instance_acceleration_structure scene, uint mask, bool anyHit, AlphaScene a,
                         const array<texture2d<float>, MAX_TEXTURES> textures) {
    intersection_params params;
    params.assume_geometry_type(geometry_type::triangle);
    params.accept_any_intersection(anyHit);
    intersection_query<triangle_data, instancing> q(r, scene, mask, params);
    while (q.next()) {
        if (q.get_candidate_intersection_type() == intersection_type::triangle &&
            alphaPasses(a, textures, q.get_candidate_instance_id(), q.get_candidate_primitive_id(),
                        q.get_candidate_triangle_barycentric_coord())) {
            q.commit_triangle_intersection();
        }
    }
    TraceHit h;
    h.type = q.get_committed_intersection_type();
    h.distance = 0.0;
    h.instance_id = 0;
    h.primitive_id = 0;
    h.triangle_barycentric_coord = float2(0.0);
    if (h.type == intersection_type::triangle) {
        h.distance = q.get_committed_distance();
        h.instance_id = q.get_committed_instance_id();
        h.primitive_id = q.get_committed_primitive_id();
        h.triangle_barycentric_coord = q.get_committed_triangle_barycentric_coord();
    }
    return h;
}

// What reaches `origin` from `maxDistance` along l: nothing past opaque
// geometry, tinted through up to 8 panes of glass as a path crossing them
// would be on average (Fresnel and transmission).
inline float3 shadowTransmittance(instance_acceleration_structure scene, device const InstanceInfo* instances,
                                  device const MaterialInfo* materials, device const float4* normals, float3 origin,
                                  float3 l, float maxDistance, AlphaScene alpha,
                                  const array<texture2d<float>, MAX_TEXTURES> textures) {
    ray s;
    s.origin = origin;
    s.direction = l;
    s.min_distance = 1e-4;
    s.max_distance = maxDistance;
    if (traceRay(s, scene, MASK_OPAQUE, true, alpha, textures).type != intersection_type::none) return float3(0.0);
    float3 transmittance = float3(1.0);
    for (uint pane = 0; pane < 8u; ++pane) {
        const TraceHit g = traceRay(s, scene, MASK_GLASS, false, alpha, textures);
        if (g.type == intersection_type::none) return transmittance;
        const InstanceInfo gi = instances[g.instance_id];
        const MaterialInfo gm = materials[gi.material];
        const float3 gon = normals[gi.normalOffset + g.primitive_id].xyz;
        const float3 gn = normalize(float3(dot(gi.normalMatrix0.xyz, gon), dot(gi.normalMatrix1.xyz, gon),
                                           dot(gi.normalMatrix2.xyz, gon)));
        const float gf0 = pow((gm.ior - 1.0) / (gm.ior + 1.0), 2.0);
        const float gF = gf0 + (1.0 - gf0) * pow(clamp(1.0 - abs(dot(gn, l)), 0.0, 1.0), 5.0);
        const float3 tint = gm.texture >= 0 ? gm.baseColor.rgb * gm.average.rgb : gm.baseColor.rgb;
        transmittance *= tint * (gm.transmission * (1.0 - gF));
        s.origin = s.origin + l * (g.distance + 1e-3);
        s.max_distance -= g.distance + 1e-3;
        if (s.max_distance <= s.min_distance) return transmittance;
    }
    return float3(0.0);
}

// Adds light that reached the camera after `depth` diffuse / glossy
// scatterings; from two on (indirect light) clamped to `clampMax` (max
// component; 0: no clamp), keeping its hue.
inline void addLight(thread float3& radiance, float3 c, uint depth, float clampMax) {
    if (clampMax > 0.0 && depth >= 2u) {
        const float m = maxComponent(c);
        if (m > clampMax) c *= clampMax / m;
    }
    radiance += c;
}

// Unit vector <-> [-1, 1]^2 (octahedral), as giOctDecode() in rt_effects.metal.
inline float2 giOctEncode(float3 n) {
    n /= abs(n.x) + abs(n.y) + abs(n.z);
    const float2 s = float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    return n.z >= 0.0 ? n.xy : (1.0 - abs(n.yx)) * s;
}

kernel void pathTrace(instance_acceleration_structure scene      [[buffer(0)]],
                      device float4*                  accum      [[buffer(1)]],
                      constant TraceUniforms&         u          [[buffer(2)]],
                      device const InstanceInfo*      instances  [[buffer(3)]],
                      device const MaterialInfo*      materials  [[buffer(4)]],
                      device const float4*            normals    [[buffer(5)]],
                      texture2d<float, access::write> output     [[texture(0)]],
                      device float4*                  aux        [[buffer(6)]],
                      device const float4*            uvs        [[buffer(7)]],
                      device const EnvTexel*          envTable   [[buffer(8)]],
                      device const Portal*            portals    [[buffer(9)]],
                      device float*                   moments    [[buffer(10)]],
                      device atomic_uint*             converged  [[buffer(11)]],
                      device const float4*            vertexNormals [[buffer(12)]],
                      device const float4*            tangents   [[buffer(13)]],
                      device const EmissiveTri*       emissives  [[buffer(14)]],
                      device const EnvTexel*          lightTable [[buffer(15)]],
                      texturecube<float>              environment [[texture(1)]],
                      array<texture2d<float>, MAX_TEXTURES> textures [[texture(2)]],
                      texture2d<float>                gbufferNormal [[texture(66)]],
                      depth2d<float>                  gbufferDepth  [[texture(67)]],
                      texture2d<float, access::write> reflectionOut [[texture(68)]],
                      uint2 gridPosition [[thread_position_in_grid]])
{
    // A band of rows (PathTracer::setFrameBudget()): the grid covers the
    // band, which starts at row u.lights.y (0 for a whole image).
    const uint2 gid = uint2(gridPosition.x, gridPosition.y + u.lights.y);
    if (gid.x >= u.size.x || gid.y >= u.size.y) return;
    // Reflections (PathTracer::traceReflections(), mode 2): paths start at
    // the G-buffer's surface, the first direction from its GGX lobe; the
    // radiance arriving along it goes to reflectionOut (no accumulation).
    const bool reflections = u.size.w == 2u;
    // Probe rays (mode 3): x a ray, y a probe; the radiance and the first
    // hit's distance go to reflectionOut.
    const bool probes = u.size.w == 3u;
    // Indirect light samples (PathTracer::traceGlobalIllumination(), mode
    // 4): as reflections, the direction cosine-weighted; the radiance and
    // the hit's distance go to reflectionOut's left half, the hit's normal
    // to its right half.
    const bool gi = u.size.w == 4u;
    const AlphaScene alphaScene{instances, materials, uvs};
    const bool fromGBuffer = reflections || gi;
    float3 firstHitNormal = float3(0.0);
    float3 firstDirection = float3(0.0);
    float3 surfaceP = float3(0.0), surfaceN = float3(0.0), surfaceV = float3(0.0);
    float surfaceAlpha = 0.0;
    if (fromGBuffer) {
        const float d = gbufferDepth.read(gid);
        if (d >= 1.0) {
            reflectionOut.write(float4(0.0), gid);  // background: no surface
            if (gi) reflectionOut.write(float4(0.0), gid + uint2(u.size.x, 0));
            return;
        }
        const float2 ndc0 = float2((float(gid.x) + 0.5) / float(u.size.x) * 2.0 - 1.0,
                                   1.0 - (float(gid.y) + 0.5) / float(u.size.y) * 2.0);
        const float4 h = u.invViewProj * float4(ndc0, d, 1.0);
        surfaceP = h.xyz / h.w;
        const float4 nr = gbufferNormal.read(gid);
        surfaceN = nr.xyz;
        surfaceAlpha = max(nr.w * nr.w, 0.002);
        surfaceV = normalize(u.cameraPos.xyz - surfaceP);
    }
    constexpr sampler envSampler(filter::linear, mip_filter::linear);
    constexpr sampler texSampler(filter::linear, mip_filter::linear, address::repeat);
    // Several samples per dispatch (setSamplesPerRender(), setFrameBudget()):
    // sample numbers firstSample .. firstSample + samplesNow - 1.
    const uint firstSample = uint(u.params.w);
    const uint pixel = gid.y * u.size.x + gid.x;
    // (The other modes don't accumulate: their grids aren't the image's.)
    const bool accumulates = !fromGBuffer && !probes;
    const float4 before = firstSample == 0 || !accumulates ? float4(0.0) : accum[pixel];
    const float momentBefore = firstSample == 0 || !accumulates ? 0.0 : moments[pixel];
    // Adaptive sampling (setNoiseTarget()): a pixel whose average's
    // relative standard error (luminance, from its samples' second moment)
    // is under the target takes no more samples; it's still shown.
    uint samplesNow = max(u.envSampling.w, 1u);
    if (u.tuning.z > 0.0 && before.w >= max(u.tuning.w, 2.0)) {
        const float n = before.w;
        const float mean = dot(before.rgb, float3(0.2126, 0.7152, 0.0722)) / n;
        const float variance = max(momentBefore / n - mean * mean, 0.0) * n / (n - 1.0);
        // Relative to what's shown: dark pixels against a floor of 0.05 after exposure.
        const float shown = max(mean, 0.05 / exp2(u.params.y));
        if (sqrt(variance / n) / shown < u.tuning.z) {
            samplesNow = 0u;
            atomic_fetch_add_explicit(converged, 1u, memory_order_relaxed);
        }
    }
    float3 radianceSum = float3(0.0);
    float firstHitDistance = 65000.0;  // probes: the first segment's (a miss: far; fits a half float)
    float momentSum = 0.0;
    float3 albedoSum = float3(0.0);
    float3 normalSum = float3(0.0);
    for (uint sampleStep = 0; sampleStep < samplesNow; ++sampleStep) {
        const uint sampleIndex = firstSample + sampleStep;
        PathSampler rs;
        rs.pixelSeed = hashU(gid.y * u.size.x + gid.x + 0x5bd1e995u);
        rs.index = sampleIndex;
        rs.mode = u.tuning.y > 0.5 ? 1u : 0u;
        rs.rng = (gid.y * u.size.x + gid.x) * 9781u + sampleIndex * 6271u + 1u;
        pcg(rs.rng);

        // Camera ray through a jittered point of the pixel (antialiasing).
        const float2 jitter = sample2D(rs, DIM_CAMERA);
        const float2 ndc = float2((float(gid.x) + jitter.x) / float(u.size.x) * 2.0 - 1.0,
                                  1.0 - (float(gid.y) + jitter.y) / float(u.size.y) * 2.0);
        float3 origin;
        float3 direction;
        if (gi) {
            // Cosine-weighted about the surface's normal.
            const float2 l = sample2D(rs, DIM_LENS);
            float3 t, b;
            const float3 helper = abs(surfaceN.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
            t = normalize(cross(helper, surfaceN));
            b = cross(surfaceN, t);
            const float radial = sqrt(l.x);
            const float phi = 2.0 * PI * l.y;
            direction = normalize(t * (radial * cos(phi)) + b * (radial * sin(phi)) + surfaceN * sqrt(max(0.0, 1.0 - l.x)));
            firstDirection = direction;
            origin = surfaceP + surfaceN * max(1e-3, 2e-4 * length(surfaceP - u.cameraPos.xyz));
        } else if (reflections) {
            // A GGX half vector (the surface's roughness) mirrors the view
            // direction; directions under the surface take the mirror one.
            const float2 l = sample2D(rs, DIM_LENS);
            const float3 h = ggxHalfVector(surfaceN, surfaceAlpha, l.x, l.y);
            direction = reflect(-surfaceV, h);
            if (dot(direction, surfaceN) <= 0.0) direction = reflect(-surfaceV, surfaceN);
            origin = surfaceP + surfaceN * max(1e-3, 2e-4 * length(surfaceP - u.cameraPos.xyz));
        } else if (probes) {
            const uint3 n = u.probeCount.xyz;
            const uint index = gid.y;
            const uint3 cell = uint3(index % n.x, (index / n.x) % n.y, index / (n.x * n.y));
            origin = u.probeOrigin.xyz + float3(cell) * u.probeSpacing.xyz;
            // As rayDirection() in ddgi.metal.
            const float golden = 1.618033988749895;
            const float rays = float(u.probeCount.w);
            const float phi = 2.0 * PI * fract(float(gid.x) / golden);
            const float cosTheta = 1.0 - (2.0 * float(gid.x) + 1.0) / rays;
            const float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
            direction = normalize((u.probeRotation * float4(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta, 0.0)).xyz);
        } else if (u.size.w == 1u) {
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
            if (u.camera.x > 0.0) {
                // Depth of field, a thin lens: from a point of the lens
                // (concentric disc) through where this ray meets the focus
                // plane; starting as far along as the near plane was.
                const float2 l = sample2D(rs, DIM_LENS) * 2.0 - 1.0;
                float r = 0.0, phi = 0.0;
                if (any(l != 0.0)) {
                    if (abs(l.x) > abs(l.y)) {
                        r = l.x;
                        phi = PI / 4.0 * (l.y / l.x);
                    } else {
                        r = l.y;
                        phi = PI / 2.0 - PI / 4.0 * (l.x / l.y);
                    }
                }
                const float3 forward = u.cameraForward.xyz;
                const float along = max(dot(direction, forward), 1e-6);
                const float3 focus = u.cameraPos.xyz + direction * (u.camera.y / along);
                const float3 lens = u.cameraPos.xyz + (u.cameraRight.xyz * cos(phi) + u.cameraUp.xyz * sin(phi)) * (r * u.camera.x);
                const float nearAlong = dot(origin - u.cameraPos.xyz, forward);
                direction = normalize(focus - lens);
                origin = lens + direction * (nearAlong / max(dot(direction, forward), 1e-6));
            }
        }
        if (any(isnan(direction))) direction = float3(0.0, 0.0, -1.0);
        // Denoiser guides: the first surface seen, through glass and
        // mirrors (smooth surfaces whose path reflects pass the guide on,
        // tinted by their reflection colour: Open Image Denoise's advice
        // for specular surfaces).
        float3 firstAlbedo = float3(0.0);
        float3 firstNormal = float3(0.0);
        bool guided = false;
        float3 guideTint = float3(1.0);
        float3 mirrorAlbedo = float3(0.0);  // the first mirror's, should the path end there
        float3 mirrorNormal = float3(0.0);
        bool mirrored = false;

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
        // Firefly control: light that reached the camera after two or more
        // diffuse / glossy scatterings is clamped to this (max component).
        const float indirectClamp = u.tuning.x;
        uint scatters = 0u;  // diffuse / glossy vertices so far
        // Panes of glass don't spend bounces (a view into a room crosses one
        // twice: in, and out to the sky), up to this many crossings per path.
        uint glassLeft = 8u;
        const float envIntensity = u.params.x;
        const uint envR = u.envSampling.x;
        SkyLights skyLights;
        skyLights.table = envTable;
        skyLights.envR = envR;
        skyLights.portals = portals;
        skyLights.portalCount = min(u.envSampling.y, MAX_PORTALS);
        const bool envSampled = (envR > 0u || skyLights.portalCount > 0u) && envIntensity > 0.0;
    const uint emissiveCount = u.lights.x;
        // Multiple importance sampling of the sky: the density with which the
        // last diffuse / glossy vertex chose the current direction (0: the sky
        // found here wasn't also sampled as a light, full weight).
        float misBsdfPdf = 0.0;
        float3 misOrigin = float3(0.0);  // that vertex (the sky's light density depends on where from)
        uint pathVertex = 0u;
        // (Reflections and probes: indirect light, as from a bounce.)
        for (uint bounce = fromGBuffer || probes ? 1u : 0u; bounce <= maxBounces; ++bounce, ++pathVertex) {
            ray r;
            r.origin = origin;
            r.direction = direction;
            r.min_distance = 1e-4;
            r.max_distance = 1e30;
            const TraceHit hit = traceRay(r, scene, MASK_OPAQUE | MASK_GLASS, false, alphaScene, textures);
            if (pathVertex == 0u && hit.type != intersection_type::none) firstHitDistance = min(hit.distance, 65000.0);
            if (hit.type == intersection_type::none) {
                if (bounce == 0 && u.size.z == 0u) {
                    radiance += throughput * u.clearColor.rgb;
                } else {
                    // Seen directly: the sky as the skybox draws it; as light:
                    // scaled by the environment intensity.
                    const float scale = bounce == 0 ? 1.0 : envIntensity;
                    const float3 sky = environment.sample(envSampler, direction, level(0.0)).rgb * scale;
                    float weight = 1.0;
                    if (envSampled && misBsdfPdf > 0.0) {
                        weight = powerHeuristic(misBsdfPdf, skyLightPdf(skyLights, misOrigin, direction));
                    }
                    addLight(radiance, throughput * sky * weight, scatters, indirectClamp);
                    if (!guided) firstAlbedo = clamp(sky * guideTint, 0.0, 1.0);
                }
                if (!guided && bounce == 0 && u.size.z == 0u) firstAlbedo = clamp(u.clearColor.rgb, 0.0, 1.0);
                guided = true;
                break;
            }
            const InstanceInfo inst = instances[hit.instance_id];
            const MaterialInfo mat = materials[inst.material];
            const float3 objectNormal = normals[inst.normalOffset + hit.primitive_id].xyz;
            const float3 ng0 = normalize(float3(dot(inst.normalMatrix0.xyz, objectNormal), dot(inst.normalMatrix1.xyz, objectNormal),
                                                dot(inst.normalMatrix2.xyz, objectNormal)));
            const bool backFace = dot(ng0, direction) > 0.0;
            const float3 ng = backFace ? -ng0 : ng0;  // geometric, two-sided: towards the incoming ray
            if (pathVertex == 0u) firstHitNormal = ng;
            const float3 p = origin + direction * hit.distance;
            const float3 v = -direction;
            const uint tri = inst.normalOffset + hit.primitive_id;
            const float2 bary = hit.triangle_barycentric_coord;
            const float3 corner = float3(1.0 - bary.x - bary.y, bary.x, bary.y);

            // Texture coordinates: the mesh's TEXCOORD_0, or the world mapping.
            float2 uv = float2(0.0);
            bool mapped = mat.worldMapping > 0.0;
            if (mapped) {
                uv = worldMappingUV(p, ng0, mat.worldMapping, mat.worldMappingUp > 0.5);
            } else {
                const float4 ta = uvs[tri * 2];
                const float4 tb = uvs[tri * 2 + 1];
                mapped = tb.z > 0.5;
                uv = ta.xy * corner.x + ta.zw * corner.y + tb.xy * corner.z;
            }
            const bool transformed = mat.worldMapping <= 0.0;  // KHR_texture_transform: the mesh's coordinates only
            float2 baseUV = uv;
            if (transformed && mat.uvRow0.w > 0.5) {
                baseUV = float2(dot(mat.uvRow0.xyz, float3(uv, 1.0)), dot(mat.uvRow1.xyz, float3(uv, 1.0)));
            }
            float3 albedo = mat.baseColor.rgb;
            if (mat.texture >= 0 && uint(mat.texture) < MAX_TEXTURES) {
                // Meshes without texture coordinates take its average colour.
                albedo = mapped ? mat.baseColor.rgb * textures[mat.texture].sample(texSampler, baseUV, level(0.0)).rgb
                                : mat.baseColor.rgb * mat.average.rgb;
            }
            float roughness = mat.roughness;
            float metallic = mat.metallic;
            if (mapped && mat.mrTexture >= 0 && uint(mat.mrTexture) < MAX_TEXTURES) {
                const float3 mr = textures[mat.mrTexture].sample(texSampler, baseUV, level(0.0)).rgb;
                roughness *= mr.g;
                metallic *= mr.b;
            }

            // Shading normal: the corners' normals interpolated (smooth
            // shading), bent by the normal texture; on the incoming ray's
            // side, and the geometric normal where it would face away.
            const float3 objectShading = vertexNormals[tri * 3].xyz * corner.x + vertexNormals[tri * 3 + 1].xyz * corner.y +
                                         vertexNormals[tri * 3 + 2].xyz * corner.z;
            float3 n0 = float3(dot(inst.normalMatrix0.xyz, objectShading), dot(inst.normalMatrix1.xyz, objectShading),
                               dot(inst.normalMatrix2.xyz, objectShading));
            n0 = length(n0) > 1e-12 ? normalize(n0) : ng0;
            if (mapped && mat.normalTexture >= 0 && uint(mat.normalTexture) < MAX_TEXTURES) {
                float2 normalUV = uv;
                if (transformed && mat.normalUvRow0.w > 0.5) {
                    normalUV = float2(dot(mat.normalUvRow0.xyz, float3(uv, 1.0)), dot(mat.normalUvRow1.xyz, float3(uv, 1.0)));
                }
                float3 tn = textures[mat.normalTexture].sample(texSampler, normalUV, level(0.0)).xyz * 2.0 - 1.0;
                tn.xy *= mat.normalScale;
                float3 T, B;
                if (mat.worldMapping > 0.0) {
                    worldMappingFrame(ng0, mat.worldMappingUp > 0.5, T, B);
                    T = normalize(T - n0 * dot(n0, T));
                    B = normalize(B - n0 * dot(n0, B) - T * dot(T, B));
                } else {
                    const float4 t = tangents[tri];
                    T = float3(dot(inst.model0.xyz, t.xyz), dot(inst.model1.xyz, t.xyz), dot(inst.model2.xyz, t.xyz));
                    T = T - n0 * dot(n0, T);
                    T = length(T) > 1e-12 ? normalize(T) : float3(0.0);
                    B = cross(n0, T) * t.w;
                }
                if (any(T != 0.0)) {
                    const float3 bent = T * tn.x + B * tn.y + n0 * tn.z;
                    if (length(bent) > 1e-12) n0 = normalize(bent);
                }
            }
            float3 n = backFace ? -n0 : n0;
            if (dot(n, ng) <= 0.0 || dot(n, v) <= 1e-4) n = ng;
            if (mat.unlit > 0.5) {
                addLight(radiance, throughput * albedo, scatters, indirectClamp);
                if (!guided) {
                    firstAlbedo = clamp(albedo * guideTint, 0.0, 1.0);
                    firstNormal = n;
                    guided = true;
                }
                break;
            }
            {
                // Emission: weighted against the light samples where this
                // triangle is a light and the direction came from a
                // diffuse / glossy vertex that also sampled the lights.
                float weight = 1.0;
                // Emitted from the front only, unless double-sided (glTF).
                const bool emits = !backFace || mat.doubleSided > 0.5;
                if (emits && emissiveCount > 0u && inst.lightOffset > 0u && misBsdfPdf > 0.0 && any(mat.emissive.rgb > 0.0)) {
                    const uint index = inst.lightOffset - 1u + hit.primitive_id;
                    const float area = emissives[index].radiance.w;
                    const float cosL = abs(dot(ng0, direction));
                    const float3 toHit = p - misOrigin;
                    if (area > 0.0 && cosL > 1e-6) {
                        weight = powerHeuristic(misBsdfPdf, lightTable[index].pdf / area * dot(toHit, toHit) / cosL);
                    }
                }
                if (emits) addLight(radiance, throughput * mat.emissive.rgb * weight, scatters, indirectClamp);
            }

            // Thin glass: Fresnel reflection, else straight through, tinted.
            if (mat.transmission > 0.0) {
                if (glassLeft > 0u) {
                    --glassLeft;
                    --bounce;  // unsigned: wraps and the loop's ++ brings it back
                }
                const float f0 = pow((mat.ior - 1.0) / (mat.ior + 1.0), 2.0);
                const float F = f0 + (1.0 - f0) * pow(clamp(1.0 - dot(n, v), 0.0, 1.0), 5.0);
                const float transmit = mat.transmission * (1.0 - F);
                if (sample2D(rs, vertexDim(pathVertex, DIM_BSDF)).x < transmit) {
                    // Straight through: the sky beyond was also sampled as a
                    // light (through glass) at the last vertex, keep its weight.
                    throughput *= albedo;
                    origin = p + direction * 1e-3;
                } else {
                    direction = reflect(direction, n);
                    origin = p + ng * 1e-3;
                    misBsdfPdf = 0.0;  // light samples don't follow reflections
                }
                continue;
            }

            const bool mirror = roughness < MIRROR_ROUGHNESS;
            if (!guided && !mirror) {
                firstAlbedo = clamp(albedo * guideTint, 0.0, 1.0);
                firstNormal = n;
                guided = true;
            }
            const float alpha = max(roughness * roughness, 0.002);
            const float3 f0 = mix(float3(0.04), albedo, metallic);
            const float3 Fv = fresnelSchlick(f0, max(dot(n, v), 0.0));
            const float specularWeight = clamp((Fv.r + Fv.g + Fv.b) / 3.0 + metallic * 0.5, 0.05, 0.95);

            // Directional lights, sampled directly: a point of each one's disc,
            // a shadow ray.
            for (uint light = 0; light < 4; ++light) {
                if (u.sunRadiance[light].w < 0.5) continue;
                float3 st, sb;
                basis(u.sunDirection[light].xyz, st, sb);
                const float2 ls = sample2D(rs, vertexDim(pathVertex, DIM_LIGHTS + light));
                const float cosMax = u.sunDirection[light].w;
                const float cosT = 1.0 - ls.x * (1.0 - cosMax);
                const float sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
                const float phi = 2.0 * PI * ls.y;
                const float3 l = normalize(st * (sinT * cos(phi)) + sb * (sinT * sin(phi)) + u.sunDirection[light].xyz * cosT);
                const float NdotL = dot(n, l);
                if (NdotL > 0.0 && dot(ng, l) > 0.0) {
                    ray s;
                    s.origin = p + ng * 1e-3;
                    s.direction = l;
                    s.min_distance = 1e-4;
                    s.max_distance = 1e30;
                    const TraceHit blocked = traceRay(s, scene, MASK_OPAQUE, true, alphaScene, textures);
                    if (blocked.type == intersection_type::none) {
                        addLight(radiance, throughput * brdf(n, v, l, albedo, metallic, alpha, f0) * u.sunRadiance[light].rgb * NdotL,
                                 scatters + 1u, indirectClamp);
                    }
                }
            }

            // The sky as a light: a direction by the environment's brightness
            // or through a portal, a shadow ray (through glass, tinted as a
            // path crossing it would be on average), weighted against finding
            // it by the BSDF. At the last vertex the BSDF's direction isn't
            // traced: full weight.
            const float3 shadingOrigin = p + ng * 1e-3;
            if (envSampled) {
                float envPdf = 0.0;
                const float3 l = sampleSkyLight(skyLights, shadingOrigin, sample2D(rs, vertexDim(pathVertex, DIM_ENV)), envPdf);
                const float NdotL = dot(n, l);
                if (NdotL > 0.0 && dot(ng, l) > 0.0 && envPdf > 0.0) {
                    const float3 transmittance = shadowTransmittance(scene, instances, materials, normals, shadingOrigin, l, 1e30, alphaScene, textures);
                    if (any(transmittance > 0.0)) {
                        const float weight =
                            bounce == maxBounces ? 1.0 : powerHeuristic(envPdf, bsdfPdf(n, v, l, alpha, specularWeight));
                        const float3 sky = environment.sample(envSampler, l, level(0.0)).rgb * envIntensity;
                        addLight(radiance, throughput * brdfSampled(n, v, l, albedo, metallic, alpha, f0) * sky * transmittance *
                                     (NdotL * weight / envPdf),
                                 scatters + 1u, indirectClamp);
                    }
                }
            }
            // Emissive triangles as lights: one by power, a point on it, a
            // shadow ray (through glass), weighted against the BSDF.
            if (emissiveCount > 0u) {
                const float2 es = sample2D(rs, vertexDim(pathVertex, DIM_EMISSIVE));
                float remainder;
                const uint index = invertCdf(lightTable, emissiveCount, es.x, remainder);
                const EmissiveTri e = emissives[index];
                const float su = sqrt(remainder);
                const float3 q = e.v0.xyz + e.e1.xyz * (su * (1.0 - es.y)) + e.e2.xyz * (su * es.y);
                const float3 toLight = q - shadingOrigin;
                const float dist2 = dot(toLight, toLight);
                const float dist = sqrt(dist2);
                const float3 l = toLight / max(dist, 1e-12);
                const float3 cr = cross(e.e1.xyz, e.e2.xyz);
                // Towards the shading point: the front (-l), or either side when double-sided.
                const float facing = length(cr) > 0.0 ? -dot(normalize(cr), l) : 0.0;
                const float cosL = e.e1.w > 0.5 ? abs(facing) : facing;
                const float NdotL = dot(n, l);
                if (NdotL > 0.0 && dot(ng, l) > 0.0 && cosL > 1e-6 && e.radiance.w > 0.0 && dist > 1e-4) {
                    const float lightPdf = lightTable[index].pdf / e.radiance.w * dist2 / cosL;
                    const float3 transmittance =
                        shadowTransmittance(scene, instances, materials, normals, shadingOrigin, l, dist * (1.0 - 1e-4) - 1e-4, alphaScene,
                                            textures);
                    if (any(transmittance > 0.0) && lightPdf > 0.0) {
                        const float weight =
                            bounce == maxBounces ? 1.0 : powerHeuristic(lightPdf, bsdfPdf(n, v, l, alpha, specularWeight));
                        addLight(radiance,
                                 throughput * brdfSampled(n, v, l, albedo, metallic, alpha, f0) * e.radiance.rgb * transmittance *
                                     (NdotL * weight / lightPdf),
                                 scatters + 1u, indirectClamp);
                    }
                }
            }
            ++scatters;

            // Next direction: specular or diffuse, by the Fresnel-weighted chance.
            const float2 choice = sample2D(rs, vertexDim(pathVertex, DIM_BSDF));
            const bool specular = choice.x < specularWeight;
            const float2 dirSample = float2(
                min(specular ? choice.x / specularWeight : (choice.x - specularWeight) / (1.0 - specularWeight), 0x1.fffffep-1),
                choice.y);
            if (!guided && mirror) {
                // A mirror: the guide is what it reflects (diffuse paths off
                // it: the mirror itself).
                if (!mirrored) {
                    mirrorAlbedo = clamp(albedo * guideTint, 0.0, 1.0);
                    mirrorNormal = n;
                    mirrored = true;
                }
                if (specular) {
                    guideTint *= Fv;
                } else {
                    firstAlbedo = clamp(albedo * guideTint, 0.0, 1.0);
                    firstNormal = n;
                    guided = true;
                }
            }
            if (specular) {
                const float3 h = ggxHalfVector(n, alpha, dirSample.x, dirSample.y);
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
                direction = cosineSample(n, dirSample.x, dirSample.y);
                throughput *= (1.0 - Fv) * (1.0 - metallic) * albedo / (1.0 - specularWeight);
            }
            if (dot(direction, ng) <= 0.0) break;  // into the surface (a bent shading normal): no light
            misBsdfPdf = bsdfPdf(n, v, direction, alpha, specularWeight);
            origin = shadingOrigin;
            misOrigin = shadingOrigin;

            // Russian roulette after two bounces.
            if (bounce >= 2) {
                const float survive = clamp(maxComponent(throughput), 0.05, 0.95);
                if (sample2D(rs, vertexDim(pathVertex, DIM_ROULETTE)).x > survive) break;
                throughput /= survive;
            }
        }
        if (any(isnan(radiance)) || any(isinf(radiance))) radiance = float3(0.0);
        if (!guided && mirrored) {  // the path ended at (or after) a mirror, before another surface
            firstAlbedo = mirrorAlbedo;
            firstNormal = mirrorNormal;
        }
        radianceSum += radiance;
        const float luminance = dot(radiance, float3(0.2126, 0.7152, 0.0722));
        momentSum += luminance * luminance;
        albedoSum += firstAlbedo;
        normalSum += firstNormal;
    }

    if (gi) {
        reflectionOut.write(float4(radianceSum, firstHitDistance), gid);
        // Unit vectors, octahedral: the direction, the hit's normal (the sky: facing back).
        const float3 hitNormal = any(firstHitNormal != 0.0) ? firstHitNormal : -firstDirection;
        reflectionOut.write(float4(giOctEncode(firstDirection), giOctEncode(hitNormal)), gid + uint2(u.size.x, 0));
        return;
    }
    if (reflections || probes) {
        reflectionOut.write(float4(radianceSum, probes ? firstHitDistance : 1.0), gid);
        return;
    }
    const float4 sum = before + float4(radianceSum, float(samplesNow));
    accum[pixel] = sum;
    moments[pixel] = momentBefore + momentSum;
    // The guides with their own sample count (w), as pixels may stop early.
    const uint pixels = u.size.x * u.size.y;
    aux[pixel] = (firstSample == 0 ? float4(0.0) : aux[pixel]) + float4(albedoSum, float(samplesNow));
    aux[pixels + pixel] = (firstSample == 0 ? float4(0.0) : aux[pixels + pixel]) + float4(normalSum, float(samplesNow));
    float3 color = sum.rgb / max(sum.w, 1.0) * exp2(u.params.y);
    const uint toneMapping = uint(u.camera.z + 0.5);
    color = toneMapping == 1u ? toneMapAces(color) : (toneMapping == 2u ? toneMapAgx(color) : toneMapKhronosPbrNeutral(color));
    color = pow(max(color, 0.0), float3(1.0 / 2.2));
    // Reduced resolution (a preview): this pixel covers scale x scale of the output.
    const uint scale = max(u.envSampling.z, 1u);
    const uint2 outSize = uint2(output.get_width(), output.get_height());
    for (uint dy = 0; dy < scale; ++dy) {
        for (uint dx = 0; dx < scale; ++dx) {
            const uint2 at = gid * scale + uint2(dx, dy);
            if (at.x < outSize.x && at.y < outSize.y) output.write(float4(color, 1.0), at);
        }
    }
}
