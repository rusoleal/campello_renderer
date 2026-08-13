#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// campello_renderer default Metal shader with PBR metallic-roughness
//
// Vertex slot contract (must match Renderer::VERTEX_SLOT_* constants):
//   slot  0  POSITION     float3  — object-space vertex position
//   slot  1  NORMAL       float3  — object-space vertex normal
//   slot  2  TEXCOORD_0   float2  — primary UV
//   slot  3  TANGENT      float4  — tangent + bitangent sign (w)
//   slot 16  Matrices     float4x4[2] — MVP (clip) and Model (world)
//   slot 17  MaterialUniforms — per-primitive material constants
//   slot 18  float3*      — camera world position
//   slot 19  InstanceMatrix — float4x4 per-instance transform (EXT_mesh_gpu_instancing)
//
// Bind group 0 (fragment stage):
//   [[texture(0)]]  — baseColorTexture (RGBA8)
//   [[sampler(1)]]  — baseColorSampler
//   [[texture(2)]]  — metallicRoughnessTexture (G=roughness, B=metallic)
//   [[sampler(3)]]  — metallicRoughnessSampler
//   [[texture(4)]]  — normalTexture (RGB=tangent-space normal)
//   [[sampler(5)]]  — normalSampler
//   [[texture(6)]]  — emissiveTexture (RGB=emissive color)
//   [[sampler(7)]]  — emissiveSampler
//   [[texture(8)]]  — occlusionTexture (R=occlusion factor)
//   [[sampler(9)]]  — occlusionSampler
//   [[texture(11)]] — specularTexture (A=specularFactor)
//   [[sampler(12)]] — specularSampler
//   [[texture(13)]] — specularColorTexture (RGB sRGB)
//   [[sampler(14)]] — specularColorSampler
//   [[texture(15)]] — sheenColorTexture (RGB sRGB, reuses baseColorSampler)
//   [[texture(16)]] — sheenRoughnessTexture (R linear, reuses baseColorSampler)
//   [[texture(17)]] — clearcoatTexture (R linear, reuses baseColorSampler)
//   [[texture(18)]] — clearcoatRoughnessTexture (G linear, reuses baseColorSampler)
//   [[texture(19)]] — clearcoatNormalTexture (RGB linear, reuses baseColorSampler)
//   [[texture(20)]] — transmissionTexture (R linear, scales transmissionFactor)
//                     reuses baseColorSampler (slot limit reached)
//   [[texture(22)]] — sceneColorTexture (screen-space refraction source)
//   [[sampler(23)]]  — sceneColorSampler (clamp-to-edge)
//
// Pipeline variants:
//   fragmentMain_flat     — Lambert + baseColorFactor, no texture sampling
//   fragmentMain_textured — PBR metallic-roughness with normal mapping
//   fragmentMain_debug    — Normal visualization for debugging
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

// Per-material constants uploaded to buffer slot 17.
// Layout (368 bytes, within the 512-byte per-slot stride). Offsets below are
// the actual Metal-compiled byte offsets (verified against MSL's alignment
// rules: float4 members require 16-byte alignment and pull in implicit
// padding before them; plain float members after another plain float need
// NO padding). Keep this comment in sync with the struct below by hand —
// a previous drift here (a phantom pad float assumed before iridescenceFactor
// that Metal never actually inserts) shifted every field from
// iridescenceFactor through dispersion one float late relative to what the
// CPU-side uniform builder (buildSlotRaw in campello_renderer.cpp) wrote,
// silently zeroing KHR_materials_iridescence for every asset that used it.
//   [0..15]   baseColorFactor         — RGBA multiplier for base color
//   [16..31]  uvTransformRow0         — row 0 of KHR_texture_transform [a, b, tx, hasTransform]
//   [32..47]  uvTransformRow1         — row 1 of KHR_texture_transform [c, d, ty, 0]
//   [48..51]  metallicFactor          — scalar metallic multiplier (default 1.0)
//   [52..55]  roughnessFactor         — scalar roughness multiplier (default 1.0)
//   [56..59]  normalScale             — scalar normal map intensity (default 1.0)
//   [60..63]  alphaMode               — 0=opaque, 1=mask, 2=blend
//   [64..67]  alphaCutoff             — alpha test threshold for mask mode
//   [68..71]  unlit                   — 0=lit, 1=unlit
//   [72..75]  hasNormalTexture        — 0=no normal map, 1=has normal map
//   [76..79]  hasEmissiveTexture      — 0=no emissive map, 1=has emissive map
//   [80..83]  hasOcclusionTexture     — 0=no occlusion map, 1=has occlusion map
//   [84..87]  occlusionStrength       — scalar occlusion strength (default 1.0)
//   [88..91]  _padding                — explicit pad; Metal inserts 4 more implicit bytes before float4
//   [96..111] emissiveFactor          — RGB emissive factor (float4, 16-byte aligned in Metal → offset 96)
//   [112..115] ior                    — KHR_materials_ior index of refraction (default 1.5)
//   [116..119] specularFactor         — KHR_materials_specular scalar weight (default 1.0)
//   [120..123] hasSpecularTexture     — 0=no specular texture, 1=has specular texture (A channel)
//   [124..127] hasSpecularColorTexture — 0=no specular color texture, 1=has specular color texture (RGB)
//   [128..131] _pad2                  — explicit pad; Metal inserts 12 more implicit bytes before float4
//   [144..159] specularColorFactor    — KHR_materials_specular F0 color tint (float4, default [1,1,1])
//   [160..163] _pad3                  — explicit pad; Metal inserts 12 more implicit bytes before float4
//   [176..191] sheenColorFactor       — KHR_materials_sheen color (float4, default [0,0,0])
//   [192..195] sheenRoughnessFactor   — KHR_materials_sheen roughness (default 0.0)
//   [196..199] hasSheenColorTexture   — 0=no sheen color texture, 1=has sheen color texture (RGB sRGB)
//   [200..203] hasSheenRoughnessTexture — 0=no sheen roughness texture, 1=has sheen roughness texture (R)
//   [204..207] clearcoatFactor         — KHR_materials_clearcoat layer intensity (default 0.0)
//   [208..211] clearcoatRoughnessFactor — KHR_materials_clearcoat roughness (default 0.0)
//   [212..215] hasClearcoatTexture     — 0=no, 1=has clearcoat intensity texture (R channel)
//   [216..219] hasClearcoatRoughnessTexture — 0=no, 1=has clearcoat roughness texture (G channel)
//   [220..223] hasClearcoatNormalTexture — 0=no, 1=has clearcoat normal texture
//   [224..227] clearcoatNormalScale    — clearcoat normal map intensity (default 1.0)
//   [228..231] transmissionFactor      — KHR_materials_transmission scalar (default 0.0 = opaque)
//   [232..235] hasTransmissionTexture  — 0=no, 1=has transmission texture (R channel)
//   [236..239] thicknessFactor         — KHR_materials_volume thickness (default 0.0)
//   [240..243] attenuationDistance     — KHR_materials_volume mean free path (default +inf)
//   [244..247] hasThicknessTexture     — 0=no, 1=has thickness texture
//   [248..251] _padVol                 — explicit pad; Metal inserts 4 more implicit bytes before float4
//   [256..271] attenuationColor        — KHR_materials_volume absorption tint (float4, default [1,1,1])
//   [272..275] viewMode                — renderer inspection mode enum
//   [276..279] environmentIntensity    — IBL/environment multiplier
//   [280..283] iblEnabled              — 0=no IBL, 1=IBL active
//   [284..287] iridescenceFactor       — KHR_materials_iridescence scalar (no padding before this: two
//                                        plain floats back to back need none)
//   [288..291] iridescenceIor          — thin-film IOR
//   [292..295] iridescenceThicknessMin — thin-film thickness minimum (nm)
//   [296..299] iridescenceThicknessMax — thin-film thickness maximum (nm)
//   [300..303] hasIridescenceTexture   — 0=no, 1=has iridescence texture
//   [304..307] hasIridescenceThicknessTexture — 0=no, 1=has thickness texture
//   [308..311] anisotropyStrength      — KHR_materials_anisotropy strength
//   [312..315] anisotropicRotation     — rotation angle (radians)
//   [316..319] hasAnisotropicTexture   — 0=no, 1=has anisotropy texture
//   [320..323] dispersion              — KHR_materials_dispersion scalar (default 0.0)
//   [336..351] normalUvTransformRow0   — independent KHR_texture_transform for normalTexture:
//                                        row 0 [a, b, tx, hasTransform] (Metal inserts 12 implicit
//                                        bytes at [324..335] before this float4)
//   [352..367] normalUvTransformRow1   — row 1 [c, d, ty, unused]
struct MaterialUniforms {
    float4 baseColorFactor;
    float4 uvTransformRow0;
    float4 uvTransformRow1;
    float  metallicFactor;
    float  roughnessFactor;
    float  normalScale;
    float  alphaMode;
    float  alphaCutoff;
    float  unlit;
    float  hasNormalTexture;
    float  hasEmissiveTexture;
    float  hasOcclusionTexture;
    float  occlusionStrength;
    float  _padding;
    float4 emissiveFactor;        // xyz = emissive RGB, w unused
    float  ior;
    float  specularFactor;
    float  hasSpecularTexture;
    float  hasSpecularColorTexture;
    float  _pad2;
    float4 specularColorFactor;   // xyz = specular color, w unused
    float  _pad3;
    float4 sheenColorFactor;      // xyz = sheen color, w unused
    float  sheenRoughnessFactor;
    float  hasSheenColorTexture;
    float  hasSheenRoughnessTexture;
    float  clearcoatFactor;
    float  clearcoatRoughnessFactor;
    float  hasClearcoatTexture;
    float  hasClearcoatRoughnessTexture;
    float  hasClearcoatNormalTexture;
    float  clearcoatNormalScale;
    float  transmissionFactor;
    float  hasTransmissionTexture;
    float  thicknessFactor;
    float  attenuationDistance;
    float  hasThicknessTexture;
    float  _padVol;
    float4 attenuationColor;      // xyz = attenuation color, w unused
    float  viewMode;
    float  environmentIntensity;
    float  iblEnabled;
    float  iridescenceFactor;
    float  iridescenceIor;
    float  iridescenceThicknessMin;
    float  iridescenceThicknessMax;
    float  hasIridescenceTexture;
    float  hasIridescenceThicknessTexture;
    float  anisotropyStrength;
    float  anisotropyRotation;
    float  hasAnisotropicTexture;
    float  dispersion;            // KHR_materials_dispersion (default 0.0)
    float4 normalUvTransformRow0; // independent KHR_texture_transform for normalTexture
    float4 normalUvTransformRow1;
    // KHR spec: each texture reference has its own texCoord index (which
    // TEXCOORD_n set it samples), independent of every other texture on the
    // same material — e.g. occlusionTexture very commonly uses TEXCOORD_1
    // (a separate baked-AO UV set) while baseColorTexture uses TEXCOORD_0.
    // One bit per texture slot: set when that texture uses TEXCOORD_1
    // instead of TEXCOORD_0. See TexCoord1Bit_* constants below.
    float texCoord1Mask;
};

// Bit assignment for MaterialUniforms.texCoord1Mask — must match the order
// textures are declared in fragmentMain_textured's argument list.
constant uint kUV1BaseColor        = 1u << 0;
constant uint kUV1MetallicRoughness = 1u << 1;
constant uint kUV1Normal           = 1u << 2;
constant uint kUV1Emissive         = 1u << 3;
constant uint kUV1Occlusion        = 1u << 4;
constant uint kUV1Specular         = 1u << 5;
constant uint kUV1SpecularColor    = 1u << 6;
constant uint kUV1SheenColor       = 1u << 7;
constant uint kUV1SheenRoughness   = 1u << 8;
constant uint kUV1Clearcoat        = 1u << 9;
constant uint kUV1ClearcoatRoughness = 1u << 10;
constant uint kUV1ClearcoatNormal  = 1u << 11;
constant uint kUV1Transmission     = 1u << 12;
constant uint kUV1Thickness        = 1u << 13;
constant uint kUV1Iridescence      = 1u << 14;
constant uint kUV1IridescenceThickness = 1u << 15;
constant uint kUV1Anisotropic      = 1u << 16;

inline float2 selectUV(float2 uv0, float2 uv1, float packedMask, uint bit) {
    return (uint(packedMask) & bit) != 0 ? uv1 : uv0;
}

// Khronos PBR Neutral tone mapping — the glTF-Sample-Renderer's actual current
// default (see GltfState.ToneMaps.KHR_PBR_NEUTRAL), not the plain Reinhard this
// replaces. Ported verbatim from the reference's tonemapping.glsl. Shared by
// fragmentMain_flat, fragmentMain_textured, and skyboxFragment so all three
// output paths agree on the same final color transform.
inline float3 toneMapKhronosPbrNeutral(float3 color) {
    const float startCompression = 0.8f - 0.04f;
    const float desaturation = 0.15f;

    float x = min(color.r, min(color.g, color.b));
    float offset = x < 0.08f ? x - 6.25f * x * x : 0.04f;
    color -= offset;

    float peak = max(color.r, max(color.g, color.b));
    if (peak < startCompression) return color;

    float d = 1.0f - startCompression;
    float newPeak = 1.0f - d * d / (peak + d - startCompression);
    color *= newPeak / peak;

    float g = 1.0f - 1.0f / (desaturation * (peak - newPeak) + 1.0f);
    return mix(color, float3(newPeak), g);
}

// The swapchain target is plain (non-sRGB-tagged) unorm (bgra8unorm — see
// examples/macos/ViewController.mm's colorPixelFormat), so, as in the
// reference's own toneMap() wrapper, linear-to-sRGB encoding must happen here
// rather than relying on the hardware to do it on write.
inline float3 linearToSRGBFast(float3 color) {
    return pow(color, float3(1.0f / 2.2f));
}

struct CameraUniforms {
    float4   cameraPos;
    float4x4 viewMatrix;
    float4x4 projMatrix;
    float2   screenSize;
    float2   _pad;
};

constant float VIEW_MODE_NORMAL       = 0.0;
constant float VIEW_MODE_WORLD_NORMAL = 1.0;
constant float VIEW_MODE_BASE_COLOR   = 2.0;
constant float VIEW_MODE_METALLIC     = 3.0;
constant float VIEW_MODE_ROUGHNESS    = 4.0;
constant float VIEW_MODE_OCCLUSION    = 5.0;
constant float VIEW_MODE_EMISSIVE     = 6.0;
constant float VIEW_MODE_ALPHA        = 7.0;
constant float VIEW_MODE_UV0          = 8.0;
constant float VIEW_MODE_SPECULAR_FACTOR     = 9.0;
constant float VIEW_MODE_SPECULAR_COLOR      = 10.0;
constant float VIEW_MODE_SHEEN_COLOR         = 11.0;
constant float VIEW_MODE_SHEEN_ROUGHNESS     = 12.0;
constant float VIEW_MODE_CLEARCOAT           = 13.0;
constant float VIEW_MODE_CLEARCOAT_ROUGHNESS = 14.0;
constant float VIEW_MODE_CLEARCOAT_NORMAL    = 15.0;
constant float VIEW_MODE_TRANSMISSION        = 16.0;
constant float VIEW_MODE_ENVIRONMENT         = 17.0;
constant float VIEW_MODE_IRIDESCENCE         = 18.0;
constant float VIEW_MODE_ANISOTROPY          = 19.0;
constant float VIEW_MODE_DISPERSION          = 20.0;

struct VertexIn {
    float3 position  [[attribute(0)]];
    float3 normal    [[attribute(1)]];
    float2 texcoord0 [[attribute(2)]];
    float4 tangent   [[attribute(3)]];
    uint4  joints    [[attribute(4)]];
    float4 weights   [[attribute(5)]];
    float4 color0    [[attribute(6)]]; // COLOR_0, normalized to float4 on upload; (1,1,1,1) fallback
    float2 texcoord1 [[attribute(7)]]; // TEXCOORD_1; (0,0) fallback
};

// VertexOut carries only geometric interpolants — 7 user attribute slots,
// well within Metal's limit of 32. All material constants are read in the
// fragment shader directly from the constant MaterialUniforms buffer (slot 17).
struct VertexOut {
    float4 clipPosition [[position]];
    float3 worldPos;
    float3 worldNormal;
    float3 worldTangent;
    float3 worldBitangent;
    float2 texcoord0;
    float2 normalUV;
    float2 texcoord1;
    float4 color0;
};

// ---------------------------------------------------------------------------
// Vertex shader (shared by all fragment variants)
//
// Buffer 16 contains per-node transforms as an array of NodeTransforms structs.
// Each struct contains MVP (clip space) and Model (world space) matrices.
// The buffer offset is set per-draw to select the correct node's transforms.
// ---------------------------------------------------------------------------
struct NodeTransforms {
    float4x4 mvp;
    float4x4 model;
};

// Morph targets: packed as [targetCount, hasNormal, vertexCount, weights[8]]
// to match the 11-float buffer the renderer uploads per node (updateMorphWeights).
// Delta buffers are laid out target-major: target t's deltas for vertex v are
// at index t * vertexCount + v, selected by [[vertex_id]] below.
struct MorphInfo {
    float targetCount;
    float hasNormal;
    float vertexCount;
    float weights[8];
};

vertex VertexOut vertexMain(
    VertexIn                  in   [[stage_in]],
    device const NodeTransforms *nodeTransforms  [[buffer(16)]],
    constant MaterialUniforms &mat [[buffer(17)]],
    device const float4x4    *instanceMatrices   [[buffer(19)]],
    device const float4x4    *jointMatrices      [[buffer(20)]],
    constant MorphInfo       &morph              [[buffer(21)]],
    device const float3      *morphPositions     [[buffer(22)]],
    device const float3      *morphNormals       [[buffer(23)]],
    uint                       vertexID          [[vertex_id]])
{
    // Apply KHR_texture_transform when hasUVTransform flag (row0.w) is set.
    // NOTE: transformedUV becomes VertexOut.texcoord0, the shared fallback UV
    // every other texture's selectUV() call reads when its own texCoord1 bit
    // is unset — it must stay TEXCOORD_0-based (transformed or not) so that
    // fallback stays correct regardless of what baseColorTexture itself does.
    // baseColorTexture's own possible texCoord=1 is applied separately, in
    // the fragment shader, without touching this shared value.
    float2 transformedUV;
    if (mat.uvTransformRow0.w > 0.5) {
        float3 uv3   = float3(in.texcoord0, 1.0);
        transformedUV = float2(dot(mat.uvTransformRow0.xyz, uv3),
                               dot(mat.uvTransformRow1.xyz, uv3));
    } else {
        transformedUV = in.texcoord0;
    }

    // normalTexture commonly carries its own KHR_texture_transform, distinct
    // from baseColorTexture's (e.g. a tiled micro-detail normal map on a
    // material with only a flat baseColorFactor) — must not reuse transformedUV.
    float2 normalUV;
    if (mat.normalUvTransformRow0.w > 0.5) {
        // KHR_texture_transform is assumed to be authored against TEXCOORD_0
        // even if normalTexture separately specifies texCoord=1 — combining
        // both is a rare enough combination that it's not handled here.
        float3 uv3n = float3(in.texcoord0, 1.0);
        normalUV = float2(dot(mat.normalUvTransformRow0.xyz, uv3n),
                          dot(mat.normalUvTransformRow1.xyz, uv3n));
    } else {
        normalUV = selectUV(in.texcoord0, in.texcoord1, mat.texCoord1Mask, kUV1Normal);
    }

    // Morph targets applied before skinning, per glTF 2.0's evaluation order.
    float3 morphedPosition = in.position;
    float3 morphedNormal = in.normal;
    uint morphTargetCount = uint(morph.targetCount);
    if (morphTargetCount > 0) {
        uint morphVertexCount = uint(morph.vertexCount);
        bool morphHasNormal = morph.hasNormal > 0.5;
        for (uint i = 0; i < morphTargetCount; i++) {
            float w = morph.weights[i];
            if (w != 0.0) {
                morphedPosition += w * morphPositions[i * morphVertexCount + vertexID];
                if (morphHasNormal) {
                    morphedNormal += w * morphNormals[i * morphVertexCount + vertexID];
                }
            }
        }
    }

    // Skeletal mesh skinning: blend up to 4 joint matrices.
    float4 skinnedPos = float4(morphedPosition, 1.0);
    float3 skinnedNormal = morphedNormal;
    float3 skinnedTangent = in.tangent.xyz;
    float weightSum = in.weights.x + in.weights.y + in.weights.z + in.weights.w;
    if (weightSum > 0.001 && jointMatrices != nullptr) {
        skinnedPos = float4(0.0);
        skinnedNormal = float3(0.0);
        skinnedTangent = float3(0.0);
        for (int i = 0; i < 4; i++) {
            float w = in.weights[i];
            if (w > 0.0) {
                uint j = in.joints[i];
                float4x4 jm = jointMatrices[j];
                skinnedPos    += w * (jm * float4(morphedPosition, 1.0));
                skinnedNormal += w * (jm * float4(morphedNormal, 0.0)).xyz;
                skinnedTangent += w * (jm * float4(in.tangent.xyz, 0.0)).xyz;
            }
        }
    }

    float4x4 mvp   = nodeTransforms[0].mvp;
    float4x4 model = nodeTransforms[0].model;

    // EXT_mesh_gpu_instancing: apply per-instance transform if available.
    // instanceMatrices is a per-instance buffer (stepMode=instance).
    float4x4 instM = instanceMatrices[0];
    float4 localPos = instM * skinnedPos;
    float3 localNormal = (instM * float4(skinnedNormal, 0.0)).xyz;
    float3 localTangent = (instM * float4(skinnedTangent, 0.0)).xyz;

    float4 worldPos4 = model * localPos;

    float3 N = normalize((model * float4(localNormal, 0.0)).xyz);

    float3 T;
    if (length(in.tangent.xyz) < 0.001) {
        if (abs(N.y) < 0.999) {
            T = normalize(cross(float3(0,1,0), N));
        } else {
            T = normalize(cross(float3(1,0,0), N));
        }
    } else {
        T = normalize((model * float4(localTangent, 0.0)).xyz);
        T = normalize(T - dot(T, N) * N);
    }

    float3 B = cross(N, T);

    VertexOut out;
    out.clipPosition   = mvp * localPos;
    out.worldPos       = worldPos4.xyz;
    out.worldNormal    = N;
    out.worldTangent   = T;
    out.worldBitangent = B;
    out.texcoord0      = transformedUV;
    out.normalUV       = normalUV;
    out.texcoord1      = in.texcoord1;
    out.color0         = in.color0;

    return out;
}

// ---------------------------------------------------------------------------
// Fragment variant: flat — Lambert shading with baseColorFactor only.
// ---------------------------------------------------------------------------
fragment float4 fragmentMain_flat(
    VertexOut                 in  [[stage_in]],
    constant MaterialUniforms &mat [[buffer(17)]])
{
    float4 baseColor = mat.baseColorFactor * in.color0;

    // KHR_materials_transmission (simplified): additional transparency
    if (mat.transmissionFactor > 0.0) {
        baseColor.a *= (1.0 - mat.transmissionFactor);
    }

    if (mat.viewMode > VIEW_MODE_NORMAL + 0.5) {
        if (abs(mat.viewMode - VIEW_MODE_WORLD_NORMAL) < 0.5) {
            float3 N = normalize(in.worldNormal);
            return float4(N * 0.5 + 0.5, 1.0);
        }
        if (abs(mat.viewMode - VIEW_MODE_BASE_COLOR) < 0.5) {
            return float4(baseColor.rgb, 1.0);
        }
        if (abs(mat.viewMode - VIEW_MODE_METALLIC) < 0.5) {
            return float4(mat.metallicFactor, mat.metallicFactor, mat.metallicFactor, 1.0);
        }
        if (abs(mat.viewMode - VIEW_MODE_ROUGHNESS) < 0.5) {
            return float4(mat.roughnessFactor, mat.roughnessFactor, mat.roughnessFactor, 1.0);
        }
        if (abs(mat.viewMode - VIEW_MODE_OCCLUSION) < 0.5) {
            return float4(1.0, 1.0, 1.0, 1.0);
        }
        if (abs(mat.viewMode - VIEW_MODE_EMISSIVE) < 0.5) {
            return float4(mat.emissiveFactor.xyz, 1.0);
        }
        if (abs(mat.viewMode - VIEW_MODE_ALPHA) < 0.5) {
            return float4(baseColor.a, baseColor.a, baseColor.a, 1.0);
        }
        if (abs(mat.viewMode - VIEW_MODE_UV0) < 0.5) {
            return float4(fract(in.texcoord0), 0.0, 1.0);
        }
        if (abs(mat.viewMode - VIEW_MODE_SPECULAR_FACTOR) < 0.5) {
            return float4(1.0, 1.0, 1.0, 1.0);
        }
        if (abs(mat.viewMode - VIEW_MODE_SPECULAR_COLOR) < 0.5) {
            return float4(1.0, 1.0, 1.0, 1.0);
        }
        if (abs(mat.viewMode - VIEW_MODE_SHEEN_COLOR) < 0.5) {
            return float4(0.0, 0.0, 0.0, 1.0);
        }
        if (abs(mat.viewMode - VIEW_MODE_SHEEN_ROUGHNESS) < 0.5) {
            return float4(0.0, 0.0, 0.0, 1.0);
        }
        if (abs(mat.viewMode - VIEW_MODE_CLEARCOAT) < 0.5) {
            return float4(0.0, 0.0, 0.0, 1.0);
        }
        if (abs(mat.viewMode - VIEW_MODE_CLEARCOAT_ROUGHNESS) < 0.5) {
            return float4(0.0, 0.0, 0.0, 1.0);
        }
        if (abs(mat.viewMode - VIEW_MODE_CLEARCOAT_NORMAL) < 0.5) {
            float3 N = normalize(in.worldNormal);
            return float4(N * 0.5 + 0.5, 1.0);
        }
        if (abs(mat.viewMode - VIEW_MODE_TRANSMISSION) < 0.5) {
            return float4(baseColor.a, baseColor.a, baseColor.a, 1.0);
        }
        if (abs(mat.viewMode - VIEW_MODE_ENVIRONMENT) < 0.5) {
            return float4(0.0, 0.0, 0.0, 1.0);
        }
        if (abs(mat.viewMode - VIEW_MODE_IRIDESCENCE) < 0.5) {
            return float4(0.0, 0.0, 0.0, 1.0);
        }
        if (abs(mat.viewMode - VIEW_MODE_ANISOTROPY) < 0.5) {
            return float4(mat.anisotropyStrength, mat.anisotropyStrength, mat.anisotropyStrength, 1.0);
        }
        if (abs(mat.viewMode - VIEW_MODE_DISPERSION) < 0.5) {
            return float4(mat.dispersion, mat.dispersion, mat.dispersion, 1.0);
        }
        return float4(0.0, 0.0, 0.0, 1.0);
    }

    if (mat.unlit > 0.5) {
        return baseColor;
    }

    float3 lightDir = normalize(float3(0.5, 1.0, 0.5));
    float3 N        = normalize(in.worldNormal);

    float NdotL = max(dot(N, lightDir), 0.0);
    float ambient = 0.25;
    float3 diffuse = baseColor.rgb * NdotL * 0.8;
    float3 ambientColor = baseColor.rgb * ambient;

    float3 finalColor = ambientColor + diffuse + mat.emissiveFactor.xyz;
    finalColor = toneMapKhronosPbrNeutral(finalColor);
    finalColor = linearToSRGBFast(finalColor);

    return float4(finalColor, baseColor.a);
}

// ---------------------------------------------------------------------------
// Fragment variant: debug — normal visualization.
// ---------------------------------------------------------------------------
fragment float4 fragmentMain_debug(
    VertexOut                 in  [[stage_in]],
    constant MaterialUniforms &mat [[buffer(17)]])
{
    float3 N = normalize(in.worldNormal);
    float3 color = N * 0.5 + 0.5;

    if (mat.unlit > 0.5) {
        color = mix(color, float3(1.0, 0.8, 0.4), 0.3);
    }

    return float4(color, 1.0);
}

// ---------------------------------------------------------------------------
// Fragment variant: textured — PBR metallic-roughness with normal mapping.
// ---------------------------------------------------------------------------
struct Light {
    float4 position;   // xyz = position/dir, w = type (0=dir, 1=point, 2=spot)
    float4 color;      // xyz = rgb, w = intensity
    float4 direction;  // xyz = spot dir, w = range
    float4 spotAngles; // x = innerCone, y = outerCone, zw = padding
};

struct LightsUniform {
    uint32_t count;
    uint32_t _pad0;
    uint32_t _pad1;
    uint32_t _pad2;
    Light lights[4];
};

// ---------------------------------------------------------------------------
// GGX BRDF helpers — KHR_materials_clearcoat
// ---------------------------------------------------------------------------
float D_GGX(float roughness, float NdotH) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float d  = NdotH * NdotH * (a2 - 1.0f) + 1.0f;
    return a2 / (M_PI_F * d * d + 0.0001f);
}

float V_SmithGGX(float NdotL, float NdotV, float roughness) {
    float a   = roughness * roughness;
    float GL  = NdotV * sqrt(NdotL * NdotL * (1.0f - a) + a);
    float GV  = NdotL * sqrt(NdotV * NdotV * (1.0f - a) + a);
    return 0.5f / max(GL + GV, 0.0001f);
}

float F_Schlick_scalar(float F0, float cosTheta) {
    float x = 1.0f - cosTheta;
    return F0 + (1.0f - F0) * (x * x * x * x * x);
}

// ---------------------------------------------------------------------------
// Thin-film iridescence — KHR_materials_iridescence
// Port of the Khronos glTF-Sample-Viewer reference implementation (Belcour &
// Barla, "A Practical Extension to Microfacet Theory for the Modeling of
// Varying Iridescence"): multi-bounce Airy summation (m=0,1,2) over the
// film's two interfaces with CIE XYZ sensitivity curves, rather than a naive
// per-RGB-wavelength cosine approximation. The naive model gets the
// interference *frequency* right but not the actual hue produced at a given
// viewing angle (it was showing purple head-on / yellow at grazing angles
// where the Khronos viewer — and this asset was authored against exactly
// that viewer — shows blue head-on / purple at grazing).
// ---------------------------------------------------------------------------
constant float3x3 kXyzToRec709 = float3x3(
    float3( 3.2404542, -0.9692660,  0.0556434),
    float3(-1.5371385,  1.8760108, -0.2040259),
    float3(-0.4985314,  0.0415560,  1.0572252)
);

float3 Fresnel0ToIor(float3 fresnel0) {
    float3 sqrtF0 = sqrt(clamp(fresnel0, 0.0, 0.9999));
    return (float3(1.0) + sqrtF0) / (float3(1.0) - sqrtF0);
}

// Unpolarized Fresnel reflectance at a dielectric/dielectric interface.
float FresnelDielectric(float cosTheta1, float n1, float n2) {
    float sinTheta2Sq = (n1 * n1) / (n2 * n2) * (1.0 - cosTheta1 * cosTheta1);
    if (sinTheta2Sq > 1.0) return 1.0; // total internal reflection
    float cosTheta2 = sqrt(1.0 - sinTheta2Sq);
    float r_s = (n1 * cosTheta1 - n2 * cosTheta2) / (n1 * cosTheta1 + n2 * cosTheta2);
    float r_p = (n2 * cosTheta1 - n1 * cosTheta2) / (n2 * cosTheta1 + n1 * cosTheta2);
    return 0.5 * (r_s * r_s + r_p * r_p);
}

float3 FresnelDielectric3(float cosTheta1, float n1, float3 n2) {
    float3 sinTheta2Sq = (float3(n1 * n1) / (n2 * n2)) * (1.0 - cosTheta1 * cosTheta1);
    float3 cosTheta2 = sqrt(clamp(float3(1.0) - sinTheta2Sq, float3(0.0), float3(1.0)));
    float3 r_s = (n1 * cosTheta1 - n2 * cosTheta2) / (n1 * cosTheta1 + n2 * cosTheta2);
    float3 r_p = (n2 * cosTheta1 - n1 * cosTheta2) / (n2 * cosTheta1 + n1 * cosTheta2);
    return 0.5 * (r_s * r_s + r_p * r_p);
}

// Evaluate CIE XYZ sensitivity curves (as a sum of Gaussians in Fourier
// space) for a given optical path difference (nanometers) and phase shift,
// then convert to linear sRGB.
float3 EvalSensitivity(float opd, float3 shift) {
    float phase = 2.0 * M_PI_F * opd * 1.0e-9;
    float3 val = float3(5.4856e-13, 4.4201e-13, 5.2481e-13);
    float3 pos = float3(1.6810e+06, 1.7953e+06, 2.2084e+06);
    float3 var = float3(4.3278e+09, 9.3046e+09, 6.6121e+09);

    float3 xyz = val * sqrt(2.0 * M_PI_F * var) * cos(pos * phase + shift) * exp(-phase * phase * var);
    xyz.x += 9.7470e-14 * sqrt(2.0 * M_PI_F * 4.5282e+09) * cos(2.2399e+06 * phase + shift.x) * exp(-4.5282e+09 * phase * phase);
    xyz /= 1.0685e-7;

    return kXyzToRec709 * xyz;
}

// Fresnel reflectance of a thin dielectric film (thickness in nm, IOR eta2)
// sitting on top of a base material with Fresnel-0 reflectance baseF0,
// viewed from a medium of IOR outsideIOR (1.0 = air) at cosTheta1 = NdotV.
float3 EvalIridescence(float outsideIOR, float eta2, float cosTheta1, float thinFilmThickness, float3 baseF0) {
    // Force iridescenceIor -> outsideIOR as thinFilmThickness -> 0, so a
    // vanishing film smoothly falls back to the uncoated base F0.
    float iridescenceIor = mix(outsideIOR, eta2, smoothstep(0.0, 0.03, thinFilmThickness));
    float sinTheta2Sq = (outsideIOR * outsideIOR) / (iridescenceIor * iridescenceIor)
                        * (1.0 - cosTheta1 * cosTheta1);
    float cosTheta2Sq = 1.0 - sinTheta2Sq;
    if (cosTheta2Sq < 0.0) {
        return float3(1.0); // total internal reflection at the first interface
    }
    float cosTheta2 = sqrt(cosTheta2Sq);

    // First interface: outside medium -> film.
    float R12   = FresnelDielectric(cosTheta1, outsideIOR, iridescenceIor);
    float T121  = 1.0 - R12;
    float phi12 = (iridescenceIor < outsideIOR) ? M_PI_F : 0.0;
    float phi21 = M_PI_F - phi12;

    // Second interface: film -> base material.
    float3 baseIOR = Fresnel0ToIor(baseF0);
    float3 R23      = FresnelDielectric3(cosTheta2, iridescenceIor, baseIOR);
    float3 phi23    = float3(baseIOR.x < iridescenceIor ? M_PI_F : 0.0,
                             baseIOR.y < iridescenceIor ? M_PI_F : 0.0,
                             baseIOR.z < iridescenceIor ? M_PI_F : 0.0);

    // Phase shift and optical path difference for this viewing angle.
    float  opd = 2.0 * iridescenceIor * thinFilmThickness * cosTheta2;
    float3 phi = float3(phi21) + phi23;

    // Multi-bounce (Airy summation) compound terms.
    float3 R123 = clamp(float3(R12) * R23, float3(1e-5), float3(0.9999));
    float3 r123 = sqrt(R123);
    float3 Rs   = (T121 * T121) * R23 / (float3(1.0) - R123);

    float3 C0 = float3(R12) + Rs;
    float3 I  = C0;

    float3 Cm = Rs - float3(T121);
    for (int m = 1; m <= 2; ++m) {
        Cm *= r123;
        float3 Sm = 2.0 * EvalSensitivity(float(m) * opd, float(m) * phi);
        I += Cm * Sm;
    }

    return max(I, float3(0.0));
}

// ---------------------------------------------------------------------------
// Anisotropic GGX NDF — KHR_materials_anisotropy
// ---------------------------------------------------------------------------
float D_GGX_Anisotropic(float NdotH, float HdotT, float HdotB, float ax, float ay) {
    float X = HdotT / ax;
    float Y = HdotB / ay;
    float tmp = X * X + Y * Y + NdotH * NdotH;
    return 1.0 / (M_PI_F * ax * ay * tmp * tmp);
}

// ---------------------------------------------------------------------------
// Charlie sheen NDF and Neubelt visibility — KHR_materials_sheen
// ---------------------------------------------------------------------------
float D_Charlie(float roughness, float NdotH) {
    float invAlpha = 1.0 / max(roughness * roughness, 0.0001f);
    float cos2h    = NdotH * NdotH;
    float sin2h    = max(1.0f - cos2h, 0.0078125f);
    return (2.0f + invAlpha) * pow(sin2h, invAlpha * 0.5f) / (2.0f * M_PI_F);
}

float V_Neubelt(float NdotV, float NdotL) {
    return clamp(1.0f / (4.0f * (NdotL + NdotV - NdotL * NdotV)), 0.0f, 1.0f);
}

// Roughness-aware IBL Fresnel with single-scattering (FssEss) + multi-scattering
// (FmsEms) energy compensation — glTF-Sample-Renderer's getIBLGGXFresnel
// (ibl.glsl), from Fdez-Aguera's "A Multiple-Scattering Microfacet Model for
// Real-Time Image-based Lighting" (https://bruop.github.io/ibl/#single_scattering_results).
// Replaces the plain pow(1-NdotV,5) Schlick this used to use for IBL specular,
// which had no roughness dependence at all (rough dielectrics kept a bright
// grazing rim they shouldn't, and rough metals lost energy with no
// compensation).
float3 iblGGXFresnel(float NdotV, float roughness, float3 F0,
                      texture2d<float> brdfLut, sampler brdfLutSamp) {
    float2 fAB = brdfLut.sample(brdfLutSamp, clamp(float2(NdotV, roughness), 0.0f, 1.0f)).rg;
    float3 Fr = max(float3(1.0f - roughness), F0) - F0;
    float3 kS = F0 + Fr * pow(clamp(1.0f - NdotV, 0.0f, 1.0f), 5.0f);
    float3 FssEss = kS * fAB.x + fAB.y;

    float Ems = 1.0f - (fAB.x + fAB.y);
    float3 Favg = F0 + (float3(1.0f) - F0) / 21.0f;
    float3 FmsEms = Ems * FssEss * Favg / (float3(1.0f) - Favg * Ems);

    return FssEss + FmsEms;
}

fragment float4 fragmentMain_textured(
    VertexOut        in                        [[stage_in]],
    constant MaterialUniforms &mat             [[buffer(17)]],
    constant CameraUniforms  &camera           [[buffer(18)]],
    texture2d<float> baseColorTexture          [[texture(0)]],
    sampler          baseColorSampler          [[sampler(1)]],
    texture2d<float> metallicRoughnessTexture  [[texture(2)]],
    sampler          metallicRoughnessSampler  [[sampler(3)]],
    texture2d<float> normalTexture             [[texture(4)]],
    sampler          normalSampler             [[sampler(5)]],
    texture2d<float> emissiveTexture           [[texture(6)]],
    sampler          emissiveSampler           [[sampler(7)]],
    texture2d<float> occlusionTexture          [[texture(8)]],
    sampler          occlusionSampler          [[sampler(9)]],
    constant LightsUniform &lights             [[buffer(10)]],
    texture2d<float> specularTexture           [[texture(11)]],
    sampler          specularSampler           [[sampler(12)]],
    texture2d<float> specularColorTexture      [[texture(13)]],
    sampler          specularColorSampler      [[sampler(14)]],
    texture2d<float> sheenColorTexture         [[texture(15)]],
    texture2d<float> sheenRoughnessTexture     [[texture(16)]],
    texture2d<float> clearcoatTexture          [[texture(17)]],
    texture2d<float> clearcoatRoughnessTexture [[texture(18)]],
    texture2d<float> clearcoatNormalTexture    [[texture(19)]],
    texture2d<float> transmissionTexture       [[texture(20)]],
    texturecube<float> environmentMap          [[texture(21)]],
    texture2d<float> sceneColorTexture         [[texture(22)]],
    texture2d<float> thicknessTexture          [[texture(23)]],
    texture2d<float> iridescenceTexture        [[texture(24)]],
    texture2d<float> iridescenceThicknessTexture [[texture(25)]],
    texture2d<float> anisotropicTexture          [[texture(26)]],
    texturecube<float> irradianceEnvironmentMap  [[texture(27)]],
    texture2d<float> brdfLutTexture              [[texture(28)]])
{
    // uv is the shared TEXCOORD_0-based fallback every other texture below
    // reads unless it opts into texCoord=1 via its own selectUV() call.
    float2 uv = in.texcoord0;

    // baseColorTexture's own texCoord=1 is only honored when it has no
    // KHR_texture_transform of its own — when it does, uv (== in.texcoord0)
    // already carries that transform and combining both isn't handled (see
    // transformedUV's comment in vertexMain).
    float2 baseColorUV = (mat.uvTransformRow0.w > 0.5)
        ? uv : selectUV(uv, in.texcoord1, mat.texCoord1Mask, kUV1BaseColor);

    // Sample base color texture. COLOR_0 (glTF spec): finalBaseColor =
    // baseColorFactor * baseColorTexture * COLOR_0. Fallback COLOR_0 is
    // (1,1,1,1), so this is a no-op for primitives without vertex colors.
    float4 baseColor = baseColorTexture.sample(baseColorSampler, baseColorUV) * mat.baseColorFactor * in.color0;

    // KHR_materials_transmission: sample texture R channel to scale transmissionFactor
    float transmission = mat.transmissionFactor;
    if (mat.hasTransmissionTexture > 0.5) {
        float2 transUV = selectUV(in.texcoord0, in.texcoord1, mat.texCoord1Mask, kUV1Transmission);
        float transmissionTex = transmissionTexture.sample(baseColorSampler, transUV).r;
        transmission *= transmissionTex;
    }

    if (mat.viewMode > VIEW_MODE_NORMAL + 0.5) {
        if (abs(mat.viewMode - VIEW_MODE_WORLD_NORMAL) < 0.5) {
            float3 N = normalize(in.worldNormal);
            return float4(N * 0.5 + 0.5, 1.0);
        }
        if (abs(mat.viewMode - VIEW_MODE_BASE_COLOR) < 0.5) {
            return float4(baseColor.rgb, 1.0);
        }
    }

    // Alpha mask.
    if (mat.alphaMode > 0.5 && mat.alphaMode < 1.5 && baseColor.a < mat.alphaCutoff) {
        discard_fragment();
    }

    // Unlit: return base color without lighting.
    if (mat.unlit > 0.5) {
        return baseColor;
    }

    // Metallic-roughness.
    float2 mrUV = selectUV(in.texcoord0, in.texcoord1, mat.texCoord1Mask, kUV1MetallicRoughness);
    float4 mrSample = metallicRoughnessTexture.sample(metallicRoughnessSampler, mrUV);
    float metallic  = mrSample.b * mat.metallicFactor;
    float roughness = clamp(mrSample.g * mat.roughnessFactor, 0.04, 1.0);

    // Tangent frame (needed for normal mapping and anisotropy).
    float3 T = normalize(in.worldTangent);
    float3 B = normalize(in.worldBitangent);
    float3 N;
    if (mat.hasNormalTexture > 0.5) {
        float3 ns = normalTexture.sample(normalSampler, in.normalUV).rgb * 2.0 - 1.0;
        ns.xy *= mat.normalScale;
        float3 Nbase = normalize(in.worldNormal);
        N = normalize(float3x3(T, B, Nbase) * normalize(ns));
    } else {
        N = normalize(in.worldNormal);
    }

    // Occlusion. Very commonly authored on its own baked TEXCOORD_1 UV set,
    // distinct from the main material's TEXCOORD_0.
    float occlusion = 1.0;
    if (mat.hasOcclusionTexture > 0.5) {
        float2 occUV = selectUV(in.texcoord0, in.texcoord1, mat.texCoord1Mask, kUV1Occlusion);
        float occ = occlusionTexture.sample(occlusionSampler, occUV).r;
        occlusion = mix(1.0, occ, mat.occlusionStrength);
    }

    // Emissive.
    float3 emissive = mat.emissiveFactor.xyz;
    if (mat.hasEmissiveTexture > 0.5) {
        float2 emisUV = selectUV(in.texcoord0, in.texcoord1, mat.texCoord1Mask, kUV1Emissive);
        emissive *= emissiveTexture.sample(emissiveSampler, emisUV).rgb;
    }

    float specularFactor = mat.specularFactor;
    if (mat.hasSpecularTexture > 0.5) {
        float2 specUV = selectUV(in.texcoord0, in.texcoord1, mat.texCoord1Mask, kUV1Specular);
        specularFactor *= specularTexture.sample(specularSampler, specUV).a;
    }

    float3 specularColor = mat.specularColorFactor.xyz;
    if (mat.hasSpecularColorTexture > 0.5) {
        float2 specColUV = selectUV(in.texcoord0, in.texcoord1, mat.texCoord1Mask, kUV1SpecularColor);
        specularColor *= specularColorTexture.sample(specularColorSampler, specColUV).rgb;
    }

    // Anisotropy.
    float anisoStrength = mat.anisotropyStrength;
    float anisoRotation = mat.anisotropyRotation;
    if (mat.hasAnisotropicTexture > 0.5) {
        float2 anisoUV = selectUV(in.texcoord0, in.texcoord1, mat.texCoord1Mask, kUV1Anisotropic);
        float2 anisoTex = anisotropicTexture.sample(baseColorSampler, anisoUV).rg;
        anisoStrength *= anisoTex.r;
        anisoRotation += anisoTex.g * 2.0 * M_PI_F;
    }

    // KHR_materials_iridescence: sample factor and thickness.
    float iridescenceFactor = mat.iridescenceFactor;
    if (mat.hasIridescenceTexture > 0.5) {
        float2 iridUV = selectUV(in.texcoord0, in.texcoord1, mat.texCoord1Mask, kUV1Iridescence);
        iridescenceFactor *= iridescenceTexture.sample(baseColorSampler, iridUV).r;
    }
    // Per KHR_materials_iridescence, iridescenceThicknessMinimum only matters
    // as the low end of the texture-driven range; with no thickness texture
    // the material is at its nominal iridescenceThicknessMaximum, not the
    // (min+max) average — averaging shifted the interference phase (and
    // therefore the rendered hue at any given angle) away from what the
    // Khronos reference implementation produces for texture-less materials.
    float iridescenceThickness = mat.iridescenceThicknessMax;
    if (mat.hasIridescenceThicknessTexture > 0.5) {
        float2 iridThickUV = selectUV(in.texcoord0, in.texcoord1, mat.texCoord1Mask, kUV1IridescenceThickness);
        iridescenceThickness = mix(mat.iridescenceThicknessMin, mat.iridescenceThicknessMax,
                                   iridescenceThicknessTexture.sample(baseColorSampler, iridThickUV).g);
    }

    if (mat.viewMode > VIEW_MODE_NORMAL + 0.5) {
        if (abs(mat.viewMode - VIEW_MODE_METALLIC) < 0.5) {
            return float4(metallic, metallic, metallic, 1.0);
        }
        if (abs(mat.viewMode - VIEW_MODE_ROUGHNESS) < 0.5) {
            return float4(roughness, roughness, roughness, 1.0);
        }
        if (abs(mat.viewMode - VIEW_MODE_OCCLUSION) < 0.5) {
            return float4(occlusion, occlusion, occlusion, 1.0);
        }
        if (abs(mat.viewMode - VIEW_MODE_EMISSIVE) < 0.5) {
            return float4(emissive, 1.0);
        }
        if (abs(mat.viewMode - VIEW_MODE_ALPHA) < 0.5) {
            return float4(baseColor.a, baseColor.a, baseColor.a, 1.0);
        }
        if (abs(mat.viewMode - VIEW_MODE_UV0) < 0.5) {
            return float4(fract(uv), 0.0, 1.0);
        }
        if (abs(mat.viewMode - VIEW_MODE_SPECULAR_FACTOR) < 0.5) {
            return float4(specularFactor, specularFactor, specularFactor, 1.0);
        }
        if (abs(mat.viewMode - VIEW_MODE_SPECULAR_COLOR) < 0.5) {
            return float4(specularColor, 1.0);
        }
    }

    // KHR_materials_sheen: sample sheen color and roughness.
    // Reuse baseColorSampler — Metal allows only 16 sampler slots (0–15).
    float3 sheenColor = mat.sheenColorFactor.xyz;
    if (mat.hasSheenColorTexture > 0.5) {
        float2 sheenColUV = selectUV(in.texcoord0, in.texcoord1, mat.texCoord1Mask, kUV1SheenColor);
        sheenColor *= sheenColorTexture.sample(baseColorSampler, sheenColUV).rgb;
    }
    float sheenRoughness = clamp(mat.sheenRoughnessFactor, 0.07f, 1.0f);
    if (mat.hasSheenRoughnessTexture > 0.5) {
        float2 sheenRoughUV = selectUV(in.texcoord0, in.texcoord1, mat.texCoord1Mask, kUV1SheenRoughness);
        sheenRoughness = clamp(sheenRoughness * sheenRoughnessTexture.sample(baseColorSampler, sheenRoughUV).r, 0.07f, 1.0f);
    }

    // KHR_materials_clearcoat.
    float ccFactor = mat.clearcoatFactor;
    if (mat.hasClearcoatTexture > 0.5) {
        float2 ccUV = selectUV(in.texcoord0, in.texcoord1, mat.texCoord1Mask, kUV1Clearcoat);
        ccFactor *= clearcoatTexture.sample(baseColorSampler, ccUV).r;
    }
    ccFactor = clamp(ccFactor, 0.0f, 1.0f);

    float ccRoughness = clamp(mat.clearcoatRoughnessFactor, 0.001f, 1.0f);
    if (mat.hasClearcoatRoughnessTexture > 0.5) {
        float2 ccRoughUV = selectUV(in.texcoord0, in.texcoord1, mat.texCoord1Mask, kUV1ClearcoatRoughness);
        ccRoughness = clamp(ccRoughness * clearcoatRoughnessTexture.sample(baseColorSampler, ccRoughUV).g, 0.001f, 1.0f);
    }

    // Per spec: "If clearcoatNormalTexture is not given, no normal mapping is
    // applied to the clear coat layer, even if normal mapping is applied to
    // the base material" — so the fallback must be the smooth geometric
    // normal, not the base layer's (possibly normal-mapped) N. Reusing N here
    // made a mirror-smooth clearcoat (clearcoatRoughnessFactor default 0)
    // reflect the environment through the SAME high-frequency tiled normal
    // perturbation as the base paint, breaking it into a sparkly/speckled
    // mess instead of one smooth, coherent reflection.
    float3 ccN = normalize(in.worldNormal);
    if (mat.hasClearcoatNormalTexture > 0.5) {
        float2 ccNormUV = selectUV(in.texcoord0, in.texcoord1, mat.texCoord1Mask, kUV1ClearcoatNormal);
        float3 ccNS = clearcoatNormalTexture.sample(baseColorSampler, ccNormUV).rgb * 2.0f - 1.0f;
        ccNS.xy *= mat.clearcoatNormalScale;
        float3 T2    = normalize(in.worldTangent);
        float3 B2    = normalize(in.worldBitangent);
        float3 Nbase = normalize(in.worldNormal);
        ccN = normalize(float3x3(T2, B2, Nbase) * normalize(ccNS));
    }

    // View direction.
    float3 camPos  = camera.cameraPos.xyz;
    float3 viewDir = normalize(camPos - in.worldPos);

    // Double-sided rendering: flip normals for back-facing fragments so
    // lighting and refraction treat them as front-facing.
    if (dot(N, viewDir) < 0.0) N = -N;
    if (dot(ccN, viewDir) < 0.0) ccN = -ccN;

    float  NdotV   = max(dot(N, viewDir), 0.0001f);
    float  ccNdotV = max(dot(ccN, viewDir), 0.0001f);

    // IBL (image-based lighting) from environment cubemap.
    float3 iblDiffuse  = float3(0.0);
    float3 iblSpecular = float3(0.0);
    float3 iblClearcoat = float3(0.0);
    if (mat.iblEnabled > 0.5) {
        // Diffuse: Lambertian-convolved irradiance cubemap (bakeIblResources()
        // mode 2) — not a raw environmentMap sample, so no hand-tuned
        // brightness fudge factor is needed the way the old raw-sample
        // approximation required.
        float3 envDiffuse = irradianceEnvironmentMap.sample(baseColorSampler, N).rgb;
        iblDiffuse = baseColor.rgb * (1.0 - metallic) * envDiffuse * mat.environmentIntensity;

        // Roughness-based LOD so rough surfaces reflect a blurred (lower mip)
        // environment instead of always sampling the sharpest level.
        // environmentMap here is the GGX-prefiltered specular cubemap
        // (bakeIblResources() mode 1) — each mip actually holds that
        // roughness's GGX lobe instead of a box-filtered downsample.
        float envMaxLod = max(float(environmentMap.get_num_mip_levels()) - 1.0, 0.0);

        // Specular: sample using reflection direction, blurred by roughness.
        float3 R = reflect(-viewDir, N);
        float3 envSpecular = environmentMap.sample(baseColorSampler, R, level(roughness * envMaxLod)).rgb;
        float f0_scalar = (mat.ior - 1.0) / (mat.ior + 1.0);
        f0_scalar *= f0_scalar;
        float3 F0 = mix(float3(f0_scalar) * mat.specularColorFactor.xyz, baseColor.rgb, metallic);
        // Apply thin-film iridescence to IBL Fresnel.
        if (iridescenceFactor > 0.0) {
            float3 iridF0 = EvalIridescence(1.0, mat.iridescenceIor, NdotV, iridescenceThickness, F0);
            F0 = mix(F0, iridF0, iridescenceFactor);
        }
        float3 F = iblGGXFresnel(NdotV, roughness, F0, brdfLutTexture, baseColorSampler);
        iblSpecular = envSpecular * F * mat.environmentIntensity;

        // IBL clearcoat: sample environment with clearcoat normal and Fresnel,
        // blurred by the clearcoat layer's own roughness.
        if (ccFactor > 0.0) {
            float3 ccR = reflect(-viewDir, ccN);
            float3 envCC = environmentMap.sample(baseColorSampler, ccR, level(ccRoughness * envMaxLod)).rgb;
            float ccFresnel = F_Schlick_scalar(0.04f, ccNdotV);
            iblClearcoat = envCC * ccFresnel * ccFactor * mat.environmentIntensity;
        }
    }

    // KHR_materials_transmission — environment-based refraction.
    // Sample the environment cubemap in the refracted direction and apply
    // KHR_materials_volume attenuation (Beer-Lambert law).
    float3 transmitted = float3(0.0);
    float transmittance = 0.0;
    if (transmission > 0.0 && metallic < 0.99) {
        float eta = 1.0 / mat.ior;
        float3 T = refract(-viewDir, N, eta);
        if (all(T == 0.0)) T = -viewDir; // total internal reflection fallback

        // Sample thickness texture if present (modulates thicknessFactor).
        float thickness = mat.thicknessFactor;
        if (mat.hasThicknessTexture > 0.5) {
            float2 thickUV = selectUV(in.texcoord0, in.texcoord1, mat.texCoord1Mask, kUV1Thickness);
            thickness *= thicknessTexture.sample(baseColorSampler, thickUV).r;
        }

        // Determine the UV at which to sample the opaque scene.
        // For thin materials (no KHR_materials_volume) the spec says to ignore
        // macroscopic refraction and sample straight through at the fragment's
        // screen position. For volume materials we use the refracted ray.
        // KHR_materials_dispersion: when thickness > 0 and dispersion > 0,
        // compute a different refracted UV per RGB channel.
        float2 sampleUV[3];
        bool uvValid[3];
        sampleUV[0] = sampleUV[1] = sampleUV[2] = float2(0.0);
        uvValid[0] = uvValid[1] = uvValid[2] = true;
        if (thickness > 0.0) {
            // Volume material: project refracted ray through thickness.
            float3 viewPos = (camera.viewMatrix * float4(in.worldPos, 1.0)).xyz;
            float3 viewNormal = normalize((camera.viewMatrix * float4(N, 0.0)).xyz);
            float3 viewIncident = normalize(viewPos);
            if (mat.dispersion > 0.0) {
                float dispersionScale = mat.dispersion * 0.02;
                float3 iorRGB = float3(mat.ior + dispersionScale, mat.ior, max(mat.ior - dispersionScale, 1.001));
                for (int ch = 0; ch < 3; ch++) {
                    float etaCh = 1.0 / iorRGB[ch];
                    float3 viewRefractCh = refract(viewIncident, viewNormal, etaCh);
                    if (all(viewRefractCh == 0.0)) viewRefractCh = viewIncident;
                    float3 backPosCh = viewPos + viewRefractCh * thickness;
                    float4 backClipCh = camera.projMatrix * float4(backPosCh, 1.0);
                    sampleUV[ch] = backClipCh.xy / backClipCh.w * 0.5 + 0.5;
                    sampleUV[ch].y = 1.0 - sampleUV[ch].y;
                    uvValid[ch] = all(sampleUV[ch] >= 0.0) && all(sampleUV[ch] <= 1.0);
                }
            } else {
                float3 viewRefract = refract(viewIncident, viewNormal, eta);
                if (all(viewRefract == 0.0)) viewRefract = viewIncident;
                float3 backPos = viewPos + viewRefract * thickness;
                float4 backClip = camera.projMatrix * float4(backPos, 1.0);
                sampleUV[0] = sampleUV[1] = sampleUV[2] = backClip.xy / backClip.w * 0.5 + 0.5;
                sampleUV[0].y = 1.0 - sampleUV[0].y;
                uvValid[0] = uvValid[1] = uvValid[2] = all(sampleUV[0] >= 0.0) && all(sampleUV[0] <= 1.0);
            }
        } else {
            // Thin material: sample at the fragment's own screen position.
            // in.clipPosition is in viewport pixel coords (origin top-left).
            sampleUV[0] = sampleUV[1] = sampleUV[2] = in.clipPosition.xy / camera.screenSize;
        }

        // mip_filter::linear is required for the explicit level(lod) below to have
        // any effect — with the MSL default (mip_filter::none), level() is ignored
        // and the base mip is always sampled, silently defeating the roughness-based
        // transmission blur (the visual mechanism for a frosted-glass look).
        // mip_filter::linear is required for the explicit level(lod) below to have
        // any effect — with the MSL default (mip_filter::none), level() is ignored
        // and the base mip is always sampled, silently defeating the roughness-based
        // transmission blur (the visual mechanism for a frosted-glass look).
        constexpr sampler scSampler(coord::normalized, filter::linear, mip_filter::linear, address::clamp_to_edge);
        // Official glTF-Sample-Viewer LOD formula:
        // lod = log2(textureWidth) * perceptualRoughness * clamp(ior * 2.0 - 2.0, 0.0, 1.0)
        float iorScale = clamp(mat.ior * 2.0 - 2.0, 0.0, 1.0);
        float lod = log2(camera.screenSize.x) * roughness * iorScale;

        // Sample background per-channel (dispersion splits R/G/B refracted UVs).
        for (int ch = 0; ch < 3; ch++) {
            if (uvValid[ch]) {
                transmitted[ch] = sceneColorTexture.sample(scSampler, sampleUV[ch], level(lod))[ch];
            } else {
                transmitted[ch] = environmentMap.sample(baseColorSampler, T)[ch];
            }
        }

        // KHR_materials_volume attenuation.
        if (thickness > 0.0 && mat.attenuationDistance > 0.0 && !isinf(mat.attenuationDistance)) {
            float3 attn = (float3(1.0) - mat.attenuationColor.xyz) / mat.attenuationDistance;
            transmitted *= exp(-thickness * attn);
        }

        transmitted *= baseColor.rgb;

        float f0 = (mat.ior - 1.0) / (mat.ior + 1.0);
        f0 *= f0;
        float fresnel = f0 + (1.0 - f0) * pow(1.0 - NdotV, 5.0);
        transmittance = transmission * (1.0 - fresnel) * (1.0 - metallic);
    }

    if (mat.viewMode > VIEW_MODE_NORMAL + 0.5) {
        if (abs(mat.viewMode - VIEW_MODE_SHEEN_COLOR) < 0.5) {
            return float4(sheenColor, 1.0);
        }
        if (abs(mat.viewMode - VIEW_MODE_SHEEN_ROUGHNESS) < 0.5) {
            return float4(sheenRoughness, sheenRoughness, sheenRoughness, 1.0);
        }
        if (abs(mat.viewMode - VIEW_MODE_CLEARCOAT) < 0.5) {
            return float4(ccFactor, ccFactor, ccFactor, 1.0);
        }
        if (abs(mat.viewMode - VIEW_MODE_CLEARCOAT_ROUGHNESS) < 0.5) {
            return float4(ccRoughness, ccRoughness, ccRoughness, 1.0);
        }
        if (abs(mat.viewMode - VIEW_MODE_CLEARCOAT_NORMAL) < 0.5) {
            return float4(ccN * 0.5 + 0.5, 1.0);
        }
        if (abs(mat.viewMode - VIEW_MODE_TRANSMISSION) < 0.5) {
            return float4(transmission, transmission, transmission, 1.0);
        }
        if (abs(mat.viewMode - VIEW_MODE_ENVIRONMENT) < 0.5) {
            return float4(iblDiffuse + iblSpecular, 1.0);
        }
        if (abs(mat.viewMode - VIEW_MODE_IRIDESCENCE) < 0.5) {
            float dbgF0Scalar = (mat.ior - 1.0) / (mat.ior + 1.0);
            dbgF0Scalar *= dbgF0Scalar;
            float3 dbgBaseF0 = mix(float3(dbgF0Scalar) * mat.specularColorFactor.xyz, baseColor.rgb, metallic);
            float3 iridF0 = EvalIridescence(1.0, mat.iridescenceIor, NdotV, iridescenceThickness, dbgBaseF0);
            return float4(iridF0 * iridescenceFactor, 1.0);
        }
        if (abs(mat.viewMode - VIEW_MODE_ANISOTROPY) < 0.5) {
            return float4(anisoStrength, anisoStrength, anisoStrength, 1.0);
        }
        if (abs(mat.viewMode - VIEW_MODE_DISPERSION) < 0.5) {
            return float4(mat.dispersion, mat.dispersion, mat.dispersion, 1.0);
        }
        return float4(0.0, 0.0, 0.0, 1.0);
    }

    // Accumulate per-light contributions.
    float3 totalDiffuse   = float3(0.0);
    float3 totalSpecular  = float3(0.0);
    float3 totalSheen     = float3(0.0);
    float3 totalClearcoat = float3(0.0);

    uint32_t lightCount = lights.count;

    for (uint32_t i = 0; i < lightCount && i < 4; i++) {
        Light light = lights.lights[i];

        float3 lightDir;
        float attenuation = 1.0;
        float spotFactor  = 1.0;
        float typeVal     = light.position.w;

        if (typeVal < 0.5) {
            lightDir = normalize(light.position.xyz);
        } else {
            float3 toLight = light.position.xyz - in.worldPos;
            float dist = length(toLight);
            lightDir = normalize(toLight);

            float distSq = max(dist * dist, 0.0001);
            attenuation = 1.0 / distSq;

            float rangeVal = light.direction.w;
            if (rangeVal > 0.0) {
                float nd  = dist / rangeVal;
                float nd4 = nd * nd * nd * nd;
                float falloff = clamp(1.0 - nd4, 0.0, 1.0);
                attenuation *= falloff * falloff;
            }

            if (typeVal > 1.5) {
                float cosAngle = dot(-lightDir, normalize(light.direction.xyz));
                float innerCos = cos(light.spotAngles.x);
                float outerCos = cos(light.spotAngles.y);
                spotFactor = smoothstep(outerCos, innerCos, cosAngle);
            }
        }

        float  NdotL      = max(dot(N, lightDir), 0.0);
        float3 lightColor = light.color.xyz * light.color.w * attenuation * spotFactor;

        float3 halfDir = normalize(lightDir + viewDir);
        float  NdotH   = max(dot(N, halfDir), 0.0);
        float  VdotH   = max(dot(viewDir, halfDir), 0.0f);

        // KHR_materials_specular F0.
        float f0_scalar = (mat.ior - 1.0) / (mat.ior + 1.0);
        f0_scalar *= f0_scalar;

        float spec = mat.specularFactor;
        if (mat.hasSpecularTexture > 0.5) {
            float2 specUV2 = selectUV(in.texcoord0, in.texcoord1, mat.texCoord1Mask, kUV1Specular);
            spec *= specularTexture.sample(specularSampler, specUV2).a;
        }

        float3 specColor = mat.specularColorFactor.xyz;
        if (mat.hasSpecularColorTexture > 0.5) {
            float2 specColUV2 = selectUV(in.texcoord0, in.texcoord1, mat.texCoord1Mask, kUV1SpecularColor);
            specColor *= specularColorTexture.sample(specularColorSampler, specColUV2).rgb;
        }

        float3 F0_dielectric = min(float3(f0_scalar) * specColor, float3(1.0)) * spec;
        float3 F0 = mix(F0_dielectric, baseColor.rgb, metallic);
        // Apply thin-film iridescence to direct-light specular F0.
        if (iridescenceFactor > 0.0) {
            float3 iridF0 = EvalIridescence(1.0, mat.iridescenceIor, NdotV, iridescenceThickness, F0);
            F0 = mix(F0, iridF0, iridescenceFactor);
        }
        // Schlick Fresnel at this light's half-vector angle (F90 = 1, per the
        // reference's F_Schlick default) — used both to weight specular's
        // grazing-angle brightening and, via energy conservation, to reduce the
        // diffuse response by the same amount (glTF-Sample-Renderer pbr.frag:
        // l_dielectric_brdf = mix(l_diffuse, l_specular, dielectric_fresnel)).
        float3 fresnel = F0 + (float3(1.0) - F0) * pow(clamp(1.0 - VdotH, 0.0f, 1.0f), 5.0f);

        // GGX microfacet BRDF (isotropic or anisotropic).
        float D, V;
        if (anisoStrength > 0.001) {
            float aspect = sqrt(1.0 - 0.9 * anisoStrength);
            float alphaX = max(0.001, roughness * roughness / aspect);
            float alphaY = max(0.001, roughness * roughness * aspect);
            float cosR = cos(anisoRotation);
            float sinR = sin(anisoRotation);
            float3 anisoT = cosR * T + sinR * B;
            float3 anisoB = -sinR * T + cosR * B;
            float HdotT = dot(halfDir, anisoT);
            float HdotB = dot(halfDir, anisoB);
            D = D_GGX_Anisotropic(NdotH, HdotT, HdotB, alphaX, alphaY);
        } else {
            D = D_GGX(roughness, NdotH);
        }
        V = V_SmithGGX(NdotL, NdotV, roughness);

        // BRDF_lambertian(baseColor) = baseColor / PI (glTF-Sample-Renderer brdf.glsl).
        float3 lDiffuse  = (baseColor.rgb / M_PI_F) * NdotL * lightColor;
        float3 lSpecular = fresnel * D * V * NdotL * lightColor;

        // Metals have no diffuse response at all; dielectrics split energy
        // between diffuse and specular by the Fresnel reflectance instead of
        // adding both unconditionally (which double-counts energy at grazing
        // angles).
        totalDiffuse  += lDiffuse * (float3(1.0) - fresnel) * (1.0 - metallic);
        totalSpecular += lSpecular;

        // KHR_materials_sheen.
        float sheenD = D_Charlie(sheenRoughness, NdotH);
        float sheenV = V_Neubelt(NdotV, max(NdotL, 0.0001f));
        totalSheen += sheenColor * sheenD * sheenV * NdotL * lightColor;

        // KHR_materials_clearcoat.
        float ccNdotH = max(dot(ccN, halfDir), 0.0f);
        float ccNdotL = max(dot(ccN, lightDir), 0.0f);
        float cc_D    = D_GGX(ccRoughness, ccNdotH);
        float cc_V    = V_SmithGGX(ccNdotL, ccNdotV, ccRoughness);
        float cc_F    = F_Schlick_scalar(0.04f, VdotH);
        totalClearcoat += float3(cc_D * cc_V * cc_F) * ccFactor * ccNdotL * lightColor;
    }

    // Occlusion (baked AO) applies only to indirect/ambient light, never to
    // direct (punctual) lights — those already have their own visibility via
    // NdotL. totalDiffuse/totalSpecular/totalSheen/totalClearcoat come from
    // the direct-light loop above and must stay untouched by it; ambientColor
    // (our no-IBL placeholder) and the real IBL diffuse/specular terms are
    // the indirect ones and are what occlusion is meant to darken.
    //
    // The flat ambient-color hack is only a stand-in for real indirect
    // lighting when IBL is off; with IBL on, iblDiffuse already supplies the
    // ambient term and this would otherwise double-count it.
    float3 ambientColor = (mat.iblEnabled > 0.5) ? float3(0.0) : baseColor.rgb * 0.25 * occlusion;
    float3 diffuse      = totalDiffuse;
    iblDiffuse  *= occlusion;
    iblSpecular *= occlusion;

    // Scale diffuse/ambient terms by (1 - transmittance); specular/clearcoat remain.
    float diffuseScale = 1.0 - transmittance;
    ambientColor *= diffuseScale;
    diffuse      *= diffuseScale;
    iblDiffuse   *= diffuseScale;

    // KHR_materials_clearcoat: coated_material = mix(material, clearcoat_brdf, clearcoat*fresnel).
    // The whole underlying material response — including emission and any
    // transmitted light — sits below the coat and must be attenuated by it,
    // not just the diffuse/specular lobes.
    float ccAmbientAtten = 1.0f - ccFactor * F_Schlick_scalar(0.04f, ccNdotV);
    float3 finalColor = (ambientColor + diffuse + totalSpecular + totalSheen + iblDiffuse + iblSpecular
                         + emissive + transmitted * transmittance) * ccAmbientAtten
                        + totalClearcoat + iblClearcoat;

    finalColor = toneMapKhronosPbrNeutral(finalColor);
    finalColor = linearToSRGBFast(finalColor);

    return float4(finalColor, baseColor.a);
}

// ---------------------------------------------------------------------------
// Skybox shader — fullscreen triangle that samples an environment cubemap.
// ---------------------------------------------------------------------------
struct SkyboxOut {
    float4 position [[position]];
};

vertex SkyboxOut skyboxVertex(uint vertexID [[vertex_id]]) {
    float2 pos;
    if (vertexID == 0) pos = float2(-1, -1);
    else if (vertexID == 1) pos = float2(3, -1);
    else pos = float2(-1, 3);
    SkyboxOut out;
    out.position = float4(pos, 1.0, 1.0);
    return out;
}

struct SkyboxUniforms {
    float4x4 invVP;
    float2   screenSize;
    float2   _pad;
    float4   cameraPos;  // w ignored — float3 would pad to 16 bytes anyway
};

fragment float4 skyboxFragment(SkyboxOut in [[stage_in]],
                               constant SkyboxUniforms &u [[buffer(2)]],
                               texturecube<float> envMap [[texture(0)]],
                               sampler envSampler [[sampler(1)]])
{
    float2 ndc = float2(
        (in.position.x / u.screenSize.x) * 2.0 - 1.0,
        (1.0 - in.position.y / u.screenSize.y) * 2.0 - 1.0
    );
    float4 worldFar = u.invVP * float4(ndc, 1.0, 1.0);
    float3 worldDir = normalize(worldFar.xyz / worldFar.w - u.cameraPos.xyz);
    float3 color = envMap.sample(envSampler, worldDir).rgb;

    // Match the tonemapping + sRGB encoding every material's fragment shader
    // applies (see fragmentMain_textured/_flat) before writing to the 8-bit
    // swapchain. The environment is genuine HDR (sky/sun routinely exceed 1.0,
    // more so now that the built-in default is baked with a 6x exposure
    // correction — see createBuiltinDefaultEnvironmentMap), so returning it
    // raw here hard-clips to solid white/saturated colors at the output
    // format instead of compressing smoothly like every reflection of the
    // same environment does.
    color = toneMapKhronosPbrNeutral(color);
    color = linearToSRGBFast(color);

    return float4(color, 1.0);
}

// ---------------------------------------------------------------------------
// IBL precompute shader — bakes the three environment-independent/derived
// resources the glTF-Sample-Renderer reference uses for physically correct
// image-based lighting, none of which this renderer previously computed.
// Direct MSL port of shaders/vulkan/ibl_bake.frag — see that file's header
// comment for the full mode/algorithm description. Reuses skyboxVertex's
// fullscreen triangle (SkyboxOut) for the vertex stage — see
// Renderer::bakeIblResources() for the per-(mode,face,mip) draw orchestration.
// ---------------------------------------------------------------------------
struct IblBakeUniforms {
    int32_t mode;       // 0 = BRDF LUT, 1 = GGX prefilter, 2 = irradiance convolution
    int32_t faceIndex;  // 0..5, used for mode 1/2
    float   roughness;  // used for mode 1
    float   outputSize; // resolution (texels, square) of the current render target
};

// Direction for cubemap face `face` at signed uv in [-1,1] — must match the
// CPU-side convention in Renderer::convertEquirectangularImageToCubemap()
// exactly (right-handed, +Z forward, face order +X,-X,+Y,-Y,+Z,-Z), so baked
// faces stay aligned with faces uploaded via the equirect/6-file loaders.
inline float3 iblFaceDirection(int face, float uu, float vv) {
    float3 d;
    if (face == 0)      d = float3( 1.0, -vv, -uu);
    else if (face == 1) d = float3(-1.0, -vv,  uu);
    else if (face == 2) d = float3( uu,  1.0,  vv);
    else if (face == 3) d = float3( uu, -1.0, -vv);
    else if (face == 4) d = float3( uu, -vv,  1.0);
    else                 d = float3(-uu, -vv, -1.0);
    return normalize(d);
}

inline float iblRadicalInverseVdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

inline float2 iblHammersley(uint i, uint n) {
    return float2(float(i) / float(n), iblRadicalInverseVdC(i));
}

inline float3 iblImportanceSampleGGX(float2 xi, float3 N, float roughness) {
    float a = roughness * roughness;

    float phi = 2.0 * M_PI_F * xi.x;
    float cosTheta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    float3 H;
    H.x = cos(phi) * sinTheta;
    H.y = sin(phi) * sinTheta;
    H.z = cosTheta;

    float3 up = (abs(N.z) < 0.999) ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 tangent = normalize(cross(up, N));
    float3 bitangent = cross(N, tangent);

    return normalize(tangent * H.x + bitangent * H.y + N * H.z);
}

inline float iblGeometrySchlickGGX(float NdotV, float roughness) {
    float a = roughness;
    float k = (a * a) / 2.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

inline float iblGeometrySmith(float NdotV, float NdotL, float roughness) {
    return iblGeometrySchlickGGX(NdotV, roughness) * iblGeometrySchlickGGX(NdotL, roughness);
}

// Karis split-sum BRDF LUT integration — the standard closed-form approximation
// (UE4 2013 course notes) also used, in spirit, by the reference's own offline
// ibl_sampler.js LUT bake (sampled here at (NdotV, roughness) instead of stored
// as a shipped asset, since this renderer has no offline asset pipeline).
inline float2 iblIntegrateBRDF(float NdotV, float roughness) {
    float3 V = float3(sqrt(max(1.0 - NdotV * NdotV, 0.0)), 0.0, NdotV);
    float3 N = float3(0.0, 0.0, 1.0);

    float A = 0.0;
    float B = 0.0;
    const uint SAMPLE_COUNT = 256u;
    for (uint i = 0u; i < SAMPLE_COUNT; ++i) {
        float2 xi = iblHammersley(i, SAMPLE_COUNT);
        float3 H = iblImportanceSampleGGX(xi, N, roughness);
        float3 L = normalize(2.0 * dot(V, H) * H - V);

        float NdotL = max(L.z, 0.0);
        float NdotH = max(H.z, 0.0);
        float VdotH = max(dot(V, H), 0.0);

        if (NdotL > 0.0) {
            float G = iblGeometrySmith(NdotV, NdotL, roughness);
            float G_Vis = (G * VdotH) / max(NdotH * NdotV, 0.0001);
            float Fc = pow(1.0 - VdotH, 5.0);

            A += (1.0 - Fc) * G_Vis;
            B += Fc * G_Vis;
        }
    }
    return float2(A, B) / float(SAMPLE_COUNT);
}

// GGX-importance-sampled prefiltered specular radiance for direction N at the
// given roughness — Karis's split-sum "pre-filtered environment map" half.
inline float3 iblPrefilterEnvironment(float3 N, float roughness, texturecube<float> srcEnvMap, sampler srcEnvSampler) {
    // At roughness ~0 the GGX lobe collapses to a delta function (H = N for
    // every sample regardless of xi — see iblImportanceSampleGGX's cosTheta
    // formula), so importance sampling has already converged to the exact
    // answer: a single direct sample. Skipping the loop here removes the
    // single most expensive mip (mip 0, the source's full resolution) from
    // the bake entirely, which is what makes raising SAMPLE_COUNT below
    // affordable for the mips that actually need it.
    if (roughness < 0.01) {
        return srcEnvMap.sample(srcEnvSampler, N, level(0.0)).rgb;
    }

    float3 V = N;
    float3 prefilteredColor = float3(0.0);
    float totalWeight = 0.0;
    // 64 samples produced visible Monte-Carlo fireflies/noise wherever a
    // material's local roughness landed on a mid mip level reflecting a
    // bright part of the environment (the built-in default env is baked with
    // a 6x exposure boost, so its highlights are well above 1.0) — this is a
    // one-time bake, not a per-frame cost, so a much higher sample count is
    // affordable, especially with the roughness~0 fast path above removing
    // the largest/most expensive mip from needing it at all.
    const uint SAMPLE_COUNT = 512u;
    for (uint i = 0u; i < SAMPLE_COUNT; ++i) {
        float2 xi = iblHammersley(i, SAMPLE_COUNT);
        float3 H = iblImportanceSampleGGX(xi, N, roughness);
        float3 L = normalize(2.0 * dot(V, H) * H - V);
        float NdotL = max(dot(N, L), 0.0);
        if (NdotL > 0.0) {
            prefilteredColor += srcEnvMap.sample(srcEnvSampler, L, level(0.0)).rgb * NdotL;
            totalWeight += NdotL;
        }
    }
    return prefilteredColor / max(totalWeight, 0.0001);
}

// Cosine-weighted hemisphere convolution of the source environment around
// normal N — the Lambertian diffuse irradiance integral the reference bakes
// offline into u_LambertianEnvSampler (ibl_sampler.js), computed here on GPU.
//
// Cosine-weighted importance sampling via Hammersley (same technique as
// iblPrefilterEnvironment/iblIntegrateBRDF above) rather than a structured
// (phi,theta) grid walked with two nested float-indexed loops — the Vulkan
// port of the original nested-float-loop version produced visibly speckled/
// incoherent output on Intel Mesa ANV despite being mathematically standard;
// porting the same fix here for consistency even though it's unverified on
// Metal. A single uint-indexed loop matching this file's other two integrals
// sidesteps whatever miscompiled. With cosine-weighted sampling the pdf
// (cosTheta/PI) exactly cancels the integrand's own cosTheta factor, leaving
// a plain PI * average(L).
inline float3 iblConvolveIrradiance(float3 N, texturecube<float> srcEnvMap, sampler srcEnvSampler) {
    float3 up = (abs(N.z) < 0.999) ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 tangent = normalize(cross(up, N));
    float3 bitangent = cross(N, tangent);

    float3 irradiance = float3(0.0);
    const uint SAMPLE_COUNT = 16384u;
    for (uint i = 0u; i < SAMPLE_COUNT; ++i) {
        float2 xi = iblHammersley(i, SAMPLE_COUNT);
        float phi = 2.0 * M_PI_F * xi.x;
        float cosTheta = sqrt(1.0 - xi.y);
        float sinTheta = sqrt(xi.y);
        float3 localDir = float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
        float3 sampleDir = localDir.x * tangent + localDir.y * bitangent + localDir.z * N;
        irradiance += srcEnvMap.sample(srcEnvSampler, sampleDir, level(0.0)).rgb;
    }
    return M_PI_F * irradiance / float(SAMPLE_COUNT);
}

fragment float4 iblBakeFragment(
    SkyboxOut                  in           [[stage_in]],
    constant IblBakeUniforms  &u            [[buffer(0)]],
    texturecube<float>         srcEnvMap    [[texture(0)]],
    sampler                    srcEnvSampler [[sampler(1)]])
{
    if (u.mode == 0) {
        float2 uv = in.position.xy / u.outputSize;
        float2 ab = iblIntegrateBRDF(clamp(uv.x, 0.001, 1.0), clamp(uv.y, 0.001, 1.0));
        return float4(ab, 0.0, 1.0);
    }

    float2 ndc = (in.position.xy / u.outputSize) * 2.0 - 1.0;
    float3 dir = iblFaceDirection(u.faceIndex, ndc.x, ndc.y);

    if (u.mode == 1) {
        return float4(iblPrefilterEnvironment(dir, u.roughness, srcEnvMap, srcEnvSampler), 1.0);
    } else {
        return float4(iblConvolveIrradiance(dir, srcEnvMap, srcEnvSampler), 1.0);
    }
}

// ---------------------------------------------------------------------------
// FXAA shader — fullscreen post-process anti-aliasing.
// Based on FXAA 3.11 by Timothy Lottes (simplified).
// ---------------------------------------------------------------------------
struct FxaaOut {
    float4 position [[position]];
};

vertex FxaaOut fxaaVertex(uint vertexID [[vertex_id]]) {
    float2 pos;
    if (vertexID == 0) pos = float2(-1, -1);
    else if (vertexID == 1) pos = float2(3, -1);
    else pos = float2(-1, 3);
    return FxaaOut{float4(pos, 1.0, 1.0)};
}

struct FxaaUniforms {
    float2 rcpFrame;
    float2 _pad;
};

static float fxaaLuma(float3 rgb) {
    return dot(rgb, float3(0.299, 0.587, 0.114));
}

fragment float4 fxaaFragment(FxaaOut in [[stage_in]],
                             constant FxaaUniforms &u [[buffer(2)]],
                             texture2d<float> sceneTex [[texture(0)]],
                             sampler sceneSampler [[sampler(1)]])
{
    float2 pos = in.position.xy;
    float2 rcpFrame = u.rcpFrame;

    // Sample center and 4 direct neighbours.
    float3 rgbNW = sceneTex.sample(sceneSampler, pos, int2(-1, -1)).rgb;
    float3 rgbNE = sceneTex.sample(sceneSampler, pos, int2( 1, -1)).rgb;
    float3 rgbSW = sceneTex.sample(sceneSampler, pos, int2(-1,  1)).rgb;
    float3 rgbSE = sceneTex.sample(sceneSampler, pos, int2( 1,  1)).rgb;
    float3 rgbM  = sceneTex.sample(sceneSampler, pos, int2( 0,  0)).rgb;

    float lumaNW = fxaaLuma(rgbNW);
    float lumaNE = fxaaLuma(rgbNE);
    float lumaSW = fxaaLuma(rgbSW);
    float lumaSE = fxaaLuma(rgbSE);
    float lumaM  = fxaaLuma(rgbM);

    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

    // Early exit if contrast is too low.
    if (lumaMax - lumaMin < max(0.0833, lumaMax * 0.166)) {
        return float4(rgbM, 1.0);
    }

    // Compute edge direction.
    float2 dir;
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));

    float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * 0.25 * 0.125, 1.0/128.0);
    float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    dir = clamp(dir * rcpDirMin, float2(-8.0), float2(8.0)) * rcpFrame;

    // Two taps along the edge.
    float3 rgbA = 0.5 * (
        sceneTex.sample(sceneSampler, pos + dir * (1.0/3.0 - 0.5)).rgb +
        sceneTex.sample(sceneSampler, pos + dir * (2.0/3.0 - 0.5)).rgb);

    // Four taps along the edge.
    float3 rgbB = rgbA * 0.5 + 0.25 * (
        sceneTex.sample(sceneSampler, pos + dir * -0.5).rgb +
        sceneTex.sample(sceneSampler, pos + dir *  0.5).rgb);

    float lumaB = fxaaLuma(rgbB);

    // Choose the result that stays within the local luma range.
    if (lumaB < lumaMin || lumaB > lumaMax) {
        return float4(rgbA, 1.0);
    }
    return float4(rgbB, 1.0);
}

// ---------------------------------------------------------------------------
// Downsample shader — bilinear downsample from scaled scene texture to screen.
// Reuses fxaaVertex for the fullscreen triangle.
// ---------------------------------------------------------------------------
fragment float4 downsampleFragment(FxaaOut in [[stage_in]],
                                   texture2d<float> sceneTex [[texture(0)]],
                                   sampler sceneSampler [[sampler(1)]])
{
    float2 uv = in.position.xy / float2(sceneTex.get_width(), sceneTex.get_height());
    return sceneTex.sample(sceneSampler, uv);
}

// ---------------------------------------------------------------------------
// Procedural texture bake compute shader — uber-shader VM interpreter.
//
// Each thread evaluates the full procedural graph for one pixel.
// The graph is flattened to an array of ProceduralNode structs (bytecode).
// A fixed-size register file of float4 values holds intermediate results.
//
// Bindings:
//   [[buffer(0)]]  — ProceduralBakeUniforms (constant)
//   [[buffer(1)]]  — ProceduralNode[] node bytecode (read-only storage)
//   [[buffer(2)]]  — float4[] output pixels (read-write storage)
// ---------------------------------------------------------------------------

struct ProceduralNode {
    float4   value;     // inline constant / parameters
    uint32_t op;        // ProceduralOp opcode
    uint32_t outReg;    // output register index
    uint32_t inA;       // input register A  (0xFFFFFFFF = none)
    uint32_t inB;       // input register B
    uint32_t inC;       // input register C
    uint32_t inD;       // input register D
    uint32_t flags;     // type / swizzle / presence mask
    uint32_t pad;
};

struct ProceduralBakeUniforms {
    uint32_t width;
    uint32_t height;
    uint32_t nodeCount;
    uint32_t outputReg;
    uint32_t outputComponents; // 1=float, 2=vec2, 3=vec3/color3, 4=color4
    uint32_t pad[3];
};

// Opcodes — must match C++ ProceduralOp enum exactly.
constant uint32_t OP_CONSTANT      = 0;
constant uint32_t OP_TEXCOORD      = 1;
constant uint32_t OP_ADD           = 2;
constant uint32_t OP_SUBTRACT      = 3;
constant uint32_t OP_MULTIPLY      = 4;
constant uint32_t OP_DIVIDE        = 5;
constant uint32_t OP_FLOOR         = 6;
constant uint32_t OP_MODULO        = 7;
constant uint32_t OP_ABS           = 8;
constant uint32_t OP_CLAMP         = 9;
constant uint32_t OP_POWER         = 10;
constant uint32_t OP_SQRT          = 11;
constant uint32_t OP_MAX           = 12;
constant uint32_t OP_MIN           = 13;
constant uint32_t OP_SIN           = 14;
constant uint32_t OP_COS           = 15;
constant uint32_t OP_DOTPRODUCT    = 16;
constant uint32_t OP_CROSSPRODUCT  = 17;
constant uint32_t OP_LENGTH        = 18;
constant uint32_t OP_DISTANCE      = 19;
constant uint32_t OP_NORMALIZE     = 20;
constant uint32_t OP_MIX           = 21;
constant uint32_t OP_NOISE2D       = 22;
constant uint32_t OP_CHECKERBOARD  = 23;
constant uint32_t OP_PLACE2D       = 24;
constant uint32_t OP_SWIZZLE       = 25;
constant uint32_t OP_COMBINE       = 26;
constant uint32_t OP_EXTRACT       = 27;
constant uint32_t OP_IFGREATER     = 28;
constant uint32_t OP_IFEQUAL       = 29;

// Presence flags occupy bits 8-11 so they don't collide with swizzle
// (bits 0-7), extract (bits 0-1), or combine count (bits 0-3).
constant uint32_t FLAG_HAS_A = 0x0100;
constant uint32_t FLAG_HAS_B = 0x0200;
constant uint32_t FLAG_HAS_C = 0x0400;
constant uint32_t FLAG_HAS_D = 0x0800;

// Noise permutation table (identical to CPU baker).
constant int kPerm[512] = {
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
    8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
    35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,
    134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
    55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,
    18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,
    250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,
    189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,
    172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,
    228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,
    107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
    138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
    8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
    35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,
    134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
    55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,
    18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,
    250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,
    189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,
    172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,
    228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,
    107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
    138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
};

static inline float fade(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

static inline float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

static inline float grad(int hash, float x, float y) {
    int h = hash & 7;
    float u = h < 4 ? x : -x;
    float v = h < 2 || h == 5 || h == 6 ? y : -y;
    return u + v;
}

static float perlin2D(float x, float y) {
    int X = int(floor(x)) & 255;
    int Y = int(floor(y)) & 255;
    x -= floor(x);
    y -= floor(y);
    float u = fade(x);
    float v = fade(y);
    int A = kPerm[X] + Y;
    int B = kPerm[X + 1] + Y;
    return lerp(v, lerp(u, grad(kPerm[A], x, y), grad(kPerm[B], x - 1.0f, y)),
                   lerp(u, grad(kPerm[A + 1], x, y - 1.0f), grad(kPerm[B + 1], x - 1.0f, y - 1.0f)));
}

static float noise2D(float x, float y) {
    return (perlin2D(x, y) + 1.0f) * 0.5f;
}

static float fbm2D(float x, float y, int octaves, float lacunarity, float gain) {
    float total = 0.0f;
    float amplitude = 1.0f;
    float frequency = 1.0f;
    float maxValue = 0.0f;
    for (int i = 0; i < octaves; i++) {
        total += noise2D(x * frequency, y * frequency) * amplitude;
        maxValue += amplitude;
        amplitude *= gain;
        frequency *= lacunarity;
    }
    return total / maxValue;
}

kernel void proceduralBakeKernel(
    uint3                           tpitg      [[thread_position_in_threadgroup]],
    uint3                           tpg        [[threadgroup_position_in_grid]],
    uint3                           tptg       [[threads_per_threadgroup]],
    constant ProceduralBakeUniforms &uniforms  [[buffer(0)]],
    device const ProceduralNode    *nodes      [[buffer(1)]],
    device float4                  *outPixels  [[buffer(2)]])
{
    // campello_gpu dispatches 1-D threadgroups of size (threadExecutionWidth,1,1).
    // We reconstruct a flat thread index and divide by threads-per-workgroup to
    // get the pixel index, so that exactly one work-group maps to one pixel.
    uint flatThread = tpg.y * (uniforms.width * tptg.x) + tpg.x * tptg.x + tpitg.x;
    uint pixelIndex = flatThread / tptg.x;
    if (pixelIndex >= uniforms.width * uniforms.height) return;

    uint x = pixelIndex % uniforms.width;
    uint y = pixelIndex / uniforms.width;

    const uint32_t REG_COUNT = 64;
    float4 regs[REG_COUNT];
    for (uint32_t i = 0; i < REG_COUNT; i++) regs[i] = float4(0.0f);

    float u = (float(x) + 0.5f) / float(uniforms.width);
    float v = (float(y) + 0.5f) / float(uniforms.height);

    for (uint32_t n = 0; n < uniforms.nodeCount; n++) {
        ProceduralNode node = nodes[n];
        float4 a = (node.flags & FLAG_HAS_A) ? regs[node.inA] : float4(0.0f);
        float4 b = (node.flags & FLAG_HAS_B) ? regs[node.inB] : float4(0.0f);
        float4 c = (node.flags & FLAG_HAS_C) ? regs[node.inC] : float4(0.0f);
        float4 d = (node.flags & FLAG_HAS_D) ? regs[node.inD] : float4(0.0f);

        float4 r = float4(0.0f);

        switch (node.op) {
            case OP_CONSTANT:
                r = node.value;
                break;
            case OP_TEXCOORD:
                r = float4(u, v, 0.0f, 0.0f);
                break;
            case OP_ADD:
                r = a + b;
                break;
            case OP_SUBTRACT:
                r = a - b;
                break;
            case OP_MULTIPLY:
                r = a * b;
                break;
            case OP_DIVIDE:
                r = float4(
                    b.x != 0.0f ? a.x / b.x : 0.0f,
                    b.y != 0.0f ? a.y / b.y : 0.0f,
                    b.z != 0.0f ? a.z / b.z : 0.0f,
                    b.w != 0.0f ? a.w / b.w : 0.0f);
                break;
            case OP_FLOOR:
                r = floor(a);
                break;
            case OP_MODULO: {
                float4 bv = b;
                bv.x = bv.x == 0.0f ? 1e-6f : bv.x;
                bv.y = bv.y == 0.0f ? 1e-6f : bv.y;
                bv.z = bv.z == 0.0f ? 1e-6f : bv.z;
                bv.w = bv.w == 0.0f ? 1e-6f : bv.w;
                float4 m = fmod(a, bv);
                m.x = m.x < 0.0f ? m.x + abs(bv.x) : m.x;
                m.y = m.y < 0.0f ? m.y + abs(bv.y) : m.y;
                m.z = m.z < 0.0f ? m.z + abs(bv.z) : m.z;
                m.w = m.w < 0.0f ? m.w + abs(bv.w) : m.w;
                r = m;
                break;
            }
            case OP_ABS:
                r = abs(a);
                break;
            case OP_CLAMP:
                r = clamp(a, b, c);
                break;
            case OP_POWER:
                r = pow(a, b);
                break;
            case OP_SQRT:
                r = sqrt(abs(a));
                break;
            case OP_MAX:
                r = max(a, b);
                break;
            case OP_MIN:
                r = min(a, b);
                break;
            case OP_SIN:
                r = sin(a);
                break;
            case OP_COS:
                r = cos(a);
                break;
            case OP_DOTPRODUCT:
                r.x = dot(a.xyz, b.xyz);
                break;
            case OP_CROSSPRODUCT:
                r.xyz = cross(a.xyz, b.xyz);
                break;
            case OP_LENGTH:
                r.x = length(a.xyz);
                break;
            case OP_DISTANCE:
                r.x = distance(a.xyz, b.xyz);
                break;
            case OP_NORMALIZE:
                r.xyz = normalize(a.xyz);
                break;
            case OP_MIX:
                r = mix(a, b, c.x);
                break;
            case OP_NOISE2D: {
                float u_in = (node.flags & FLAG_HAS_A) ? a.x : u;
                float v_in = (node.flags & FLAG_HAS_A) ? a.y : v;
                int octaves   = int(node.value.x);
                float lacun   = node.value.y;
                float gain    = node.value.z;
                float scale   = node.value.w;
                if (octaves < 1) octaves = 1;
                if (octaves > 8) octaves = 8;
                float val = fbm2D(u_in * scale, v_in * scale, octaves, lacun, gain);
                r = float4(val, val, val, 1.0f);
                break;
            }
            case OP_CHECKERBOARD: {
                float u_in = (node.flags & FLAG_HAS_A) ? a.x : u;
                float v_in = (node.flags & FLAG_HAS_A) ? a.y : v;
                float tx = node.value.x;
                float ty = node.value.y;
                bool check = (int(floor(u_in * tx)) + int(floor(v_in * ty))) % 2 == 0;
                r = check ? b : c;
                break;
            }
            case OP_PLACE2D: {
                float u_in = (node.flags & FLAG_HAS_A) ? a.x : u;
                float v_in = (node.flags & FLAG_HAS_A) ? a.y : v;
                float2 offset  = node.value.xy;
                float2 scale   = node.value.zw;
                float rot = node.flags & FLAG_HAS_B ? b.x : 0.0f;
                float2 pivot = float2(0.5f, 0.5f);
                float2 uv = float2(u_in, v_in);
                float cosR = cos(rot);
                float sinR = sin(rot);
                float2 rv = uv - pivot;
                float2 rotUV = float2(rv.x * cosR - rv.y * sinR, rv.x * sinR + rv.y * cosR);
                rotUV += pivot;
                float2 outUV = (rotUV - pivot) * scale + pivot + offset;
                r = float4(outUV.x, outUV.y, 0.0f, 0.0f);
                break;
            }
            case OP_SWIZZLE: {
                uint32_t sx = (node.flags >> 0) & 3;
                uint32_t sy = (node.flags >> 2) & 3;
                uint32_t sz = (node.flags >> 4) & 3;
                uint32_t sw = (node.flags >> 6) & 3;
                float4 v = a;
                r.x = (sx == 0) ? v.x : (sx == 1) ? v.y : (sx == 2) ? v.z : v.w;
                r.y = (sy == 0) ? v.x : (sy == 1) ? v.y : (sy == 2) ? v.z : v.w;
                r.z = (sz == 0) ? v.x : (sz == 1) ? v.y : (sz == 2) ? v.z : v.w;
                r.w = (sw == 0) ? v.x : (sw == 1) ? v.y : (sw == 2) ? v.z : v.w;
                break;
            }
            case OP_COMBINE: {
                uint32_t count = node.flags & 0xFF;
                r.x = a.x;
                r.y = count > 1 ? b.x : 0.0f;
                r.z = count > 2 ? c.x : 0.0f;
                r.w = count > 3 ? d.x : 1.0f;
                break;
            }
            case OP_EXTRACT: {
                uint32_t comp = node.flags & 3;
                r = float4(a[comp], a[comp], a[comp], a[comp]);
                break;
            }
            case OP_IFGREATER:
                r = a.x > b.x ? c : d;
                break;
            case OP_IFEQUAL:
                r = abs(a.x - b.x) < 1e-5f ? c : d;
                break;
            default:
                break;
        }

        regs[node.outReg] = r;
    }

    float4 final = regs[uniforms.outputReg];
    uint32_t comp = uniforms.outputComponents;
    if (comp == 1) {
        // Match CPU baker: float outputs only populate the R channel.
        final = float4(final.x, 0.0f, 0.0f, 1.0f);
    } else if (comp == 2) {
        final = float4(final.x, final.y, 0.0f, 1.0f);
    } else if (comp == 3) {
        final = float4(final.x, final.y, final.z, 1.0f);
    } else if (comp == 4) {
        final = float4(final.x, final.y, final.z, final.w);
    }

    uint32_t idx = y * uniforms.width + x;
    outPixels[idx] = saturate(final);
}
