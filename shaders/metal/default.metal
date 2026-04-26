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
// Layout (256-byte stride for Metal alignment):
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
//   [88..91]  _padding                — explicit pad; Metal inserts 4 more implicit bytes before float3
//   [96..107] emissiveFactor          — RGB emissive factor (float3, 16-byte aligned in Metal → offset 96)
//   [108..111] ior                    — KHR_materials_ior index of refraction (default 1.5)
//   [112..115] specularFactor         — KHR_materials_specular scalar weight (default 1.0)
//   [116..119] hasSpecularTexture     — 0=no specular texture, 1=has specular texture (A channel)
//   [120..123] hasSpecularColorTexture — 0=no specular color texture, 1=has specular color texture (RGB)
//   [124..127] _pad2                  — explicit pad before float3
//   [128..139] specularColorFactor    — KHR_materials_specular F0 color tint (float3, default [1,1,1])
//   [140..143] _pad3                  — explicit pad before float3
//   [144..155] sheenColorFactor       — KHR_materials_sheen color (float3, default [0,0,0])
//   [156..159] sheenRoughnessFactor   — KHR_materials_sheen roughness (default 0.0)
//   [160..163] hasSheenColorTexture   — 0=no sheen color texture, 1=has sheen color texture (RGB sRGB)
//   [164..167] hasSheenRoughnessTexture — 0=no sheen roughness texture, 1=has sheen roughness texture (R)
//   [168..171] clearcoatFactor         — KHR_materials_clearcoat layer intensity (default 0.0)
//   [172..175] clearcoatRoughnessFactor — KHR_materials_clearcoat roughness (default 0.0)
//   [176..179] hasClearcoatTexture     — 0=no, 1=has clearcoat intensity texture (R channel)
//   [180..183] hasClearcoatRoughnessTexture — 0=no, 1=has clearcoat roughness texture (G channel)
//   [184..187] hasClearcoatNormalTexture — 0=no, 1=has clearcoat normal texture
//   [188..191] clearcoatNormalScale    — clearcoat normal map intensity (default 1.0)
//   [192..195] transmissionFactor      — KHR_materials_transmission scalar (default 0.0 = opaque)
//   [196..199] hasTransmissionTexture  — 0=no, 1=has transmission texture (R channel)
//   [200..203] thicknessFactor         — KHR_materials_volume thickness (default 0.0)
//   [204..207] attenuationDistance     — KHR_materials_volume mean free path (default +inf)
//   [208..223] attenuationColor        — KHR_materials_volume absorption tint (float3, default [1,1,1])
//   [224..227] viewMode                — renderer inspection mode enum
//   [228..231] environmentIntensity    — IBL/environment multiplier
//   [232..235] iblEnabled              — 0=no IBL, 1=IBL active
//   [236..239] iridescenceFactor       — KHR_materials_iridescence scalar
//   [240..243] iridescenceIor          — thin-film IOR
//   [244..247] iridescenceThicknessMin — thin-film thickness minimum (nm)
//   [248..251] iridescenceThicknessMax — thin-film thickness maximum (nm)
//   [252..255] hasIridescenceTexture   — 0=no, 1=has iridescence texture
//   [256..259] hasIridescenceThicknessTexture — 0=no, 1=has thickness texture
//   [260..263] anisotropyStrength      — KHR_materials_anisotropy strength
//   [264..267] anisotropicRotation     — rotation angle (radians)
//   [268..271] hasAnisotropicTexture   — 0=no, 1=has anisotropy texture
//   [272..275] dispersion              — KHR_materials_dispersion scalar (default 0.0)
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
};

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
};

// VertexOut carries only geometric interpolants — 5 user attribute slots,
// well within Metal's limit of 32. All material constants are read in the
// fragment shader directly from the constant MaterialUniforms buffer (slot 17).
struct VertexOut {
    float4 clipPosition [[position]];
    float3 worldPos;
    float3 worldNormal;
    float3 worldTangent;
    float3 worldBitangent;
    float2 texcoord0;
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

vertex VertexOut vertexMain(
    VertexIn                  in   [[stage_in]],
    device const NodeTransforms *nodeTransforms  [[buffer(16)]],
    constant MaterialUniforms &mat [[buffer(17)]],
    device const float4x4    *instanceMatrices   [[buffer(19)]],
    device const float4x4    *jointMatrices      [[buffer(20)]])
{
    // Apply KHR_texture_transform when hasUVTransform flag (row0.w) is set.
    float2 transformedUV;
    if (mat.uvTransformRow0.w > 0.5) {
        float3 uv3   = float3(in.texcoord0, 1.0);
        transformedUV = float2(dot(mat.uvTransformRow0.xyz, uv3),
                               dot(mat.uvTransformRow1.xyz, uv3));
    } else {
        transformedUV = in.texcoord0;
    }

    // Skeletal mesh skinning: blend up to 4 joint matrices.
    float4 skinnedPos = float4(in.position, 1.0);
    float3 skinnedNormal = in.normal;
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
                skinnedPos    += w * (jm * float4(in.position, 1.0));
                skinnedNormal += w * (jm * float4(in.normal, 0.0)).xyz;
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

    return out;
}

// ---------------------------------------------------------------------------
// Fragment variant: flat — Lambert shading with baseColorFactor only.
// ---------------------------------------------------------------------------
fragment float4 fragmentMain_flat(
    VertexOut                 in  [[stage_in]],
    constant MaterialUniforms &mat [[buffer(17)]])
{
    float4 baseColor = mat.baseColorFactor;

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

    return float4(ambientColor + diffuse, baseColor.a);
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
// Simplified analytical model: interference of reflected light from a thin
// film. Returns an RGB Fresnel-like factor that modulates the base specular F0.
// ---------------------------------------------------------------------------
float3 ThinFilmIridescence(float cosTheta, float thickness, float ior) {
    // Wavelengths for RGB in nanometers
    const float3 wavelengths = float3(650.0, 530.0, 470.0);

    // Refracted angle via Snell's law
    float sinTheta2 = 1.0 - cosTheta * cosTheta;
    float cosTheta2 = sqrt(max(1.0 - sinTheta2 / (ior * ior), 0.0));

    // Optical path difference
    float opticalPath = 2.0 * ior * thickness * cosTheta2;

    // Phase for each wavelength
    float3 phase = 6.28318530718 * opticalPath / wavelengths;

    // Fresnel reflection at air-film interface
    float R = pow((1.0 - ior) / (1.0 + ior), 2.0);

    // First-order interference
    float3 interference = 0.5 + 0.5 * cos(phase);

    // Combine with base Fresnel
    return saturate(interference * (1.0 - R) + R);
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
    texture2d<float> anisotropicTexture          [[texture(26)]])
{
    float2 uv = in.texcoord0;

    // Sample base color texture.
    float4 baseColor = baseColorTexture.sample(baseColorSampler, uv) * mat.baseColorFactor;

    // KHR_materials_transmission: sample texture R channel to scale transmissionFactor
    float transmission = mat.transmissionFactor;
    if (mat.hasTransmissionTexture > 0.5) {
        float transmissionTex = transmissionTexture.sample(baseColorSampler, uv).r;
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
    float4 mrSample = metallicRoughnessTexture.sample(metallicRoughnessSampler, uv);
    float metallic  = mrSample.b * mat.metallicFactor;
    float roughness = clamp(mrSample.g * mat.roughnessFactor, 0.04, 1.0);

    // Tangent frame (needed for normal mapping and anisotropy).
    float3 T = normalize(in.worldTangent);
    float3 B = normalize(in.worldBitangent);
    float3 N;
    if (mat.hasNormalTexture > 0.5) {
        float3 ns = normalTexture.sample(normalSampler, uv).rgb * 2.0 - 1.0;
        ns.xy *= mat.normalScale;
        float3 Nbase = normalize(in.worldNormal);
        N = normalize(float3x3(T, B, Nbase) * normalize(ns));
    } else {
        N = normalize(in.worldNormal);
    }

    // Occlusion.
    float occlusion = 1.0;
    if (mat.hasOcclusionTexture > 0.5) {
        float occ = occlusionTexture.sample(occlusionSampler, uv).r;
        occlusion = mix(1.0, occ, mat.occlusionStrength);
    }

    // Emissive.
    float3 emissive = mat.emissiveFactor.xyz;
    if (mat.hasEmissiveTexture > 0.5) {
        emissive *= emissiveTexture.sample(emissiveSampler, uv).rgb;
    }

    float specularFactor = mat.specularFactor;
    if (mat.hasSpecularTexture > 0.5) {
        specularFactor *= specularTexture.sample(specularSampler, uv).a;
    }

    float3 specularColor = mat.specularColorFactor.xyz;
    if (mat.hasSpecularColorTexture > 0.5) {
        specularColor *= specularColorTexture.sample(specularColorSampler, uv).rgb;
    }

    // Anisotropy.
    float anisoStrength = mat.anisotropyStrength;
    float anisoRotation = mat.anisotropyRotation;
    if (mat.hasAnisotropicTexture > 0.5) {
        float2 anisoTex = anisotropicTexture.sample(baseColorSampler, uv).rg;
        anisoStrength *= anisoTex.r;
        anisoRotation += anisoTex.g * 2.0 * M_PI_F;
    }

    // KHR_materials_iridescence: sample factor and thickness.
    float iridescenceFactor = mat.iridescenceFactor;
    if (mat.hasIridescenceTexture > 0.5) {
        iridescenceFactor *= iridescenceTexture.sample(baseColorSampler, uv).r;
    }
    float iridescenceThickness = mat.iridescenceThicknessMin;
    if (mat.hasIridescenceThicknessTexture > 0.5) {
        iridescenceThickness += (mat.iridescenceThicknessMax - mat.iridescenceThicknessMin)
                              * iridescenceThicknessTexture.sample(baseColorSampler, uv).g;
    } else {
        iridescenceThickness = (mat.iridescenceThicknessMin + mat.iridescenceThicknessMax) * 0.5;
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
        sheenColor *= sheenColorTexture.sample(baseColorSampler, uv).rgb;
    }
    float sheenRoughness = clamp(mat.sheenRoughnessFactor, 0.07f, 1.0f);
    if (mat.hasSheenRoughnessTexture > 0.5) {
        sheenRoughness = clamp(sheenRoughness * sheenRoughnessTexture.sample(baseColorSampler, uv).r, 0.07f, 1.0f);
    }

    // KHR_materials_clearcoat.
    float ccFactor = mat.clearcoatFactor;
    if (mat.hasClearcoatTexture > 0.5)
        ccFactor *= clearcoatTexture.sample(baseColorSampler, uv).r;
    ccFactor = clamp(ccFactor, 0.0f, 1.0f);

    float ccRoughness = clamp(mat.clearcoatRoughnessFactor, 0.001f, 1.0f);
    if (mat.hasClearcoatRoughnessTexture > 0.5) {
        ccRoughness = clamp(ccRoughness * clearcoatRoughnessTexture.sample(baseColorSampler, uv).g, 0.001f, 1.0f);
    }

    float3 ccN = N;
    if (mat.hasClearcoatNormalTexture > 0.5) {
        float3 ccNS = clearcoatNormalTexture.sample(baseColorSampler, uv).rgb * 2.0f - 1.0f;
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
        // Diffuse: sample irradiance using normal direction.
        float3 envDiffuse = environmentMap.sample(baseColorSampler, N).rgb;
        iblDiffuse = baseColor.rgb * (1.0 - metallic) * envDiffuse * mat.environmentIntensity * 0.3;

        // Specular: sample using reflection direction.
        float3 R = reflect(-viewDir, N);
        float3 envSpecular = environmentMap.sample(baseColorSampler, R).rgb;
        // Simple Fresnel approximation for IBL specular.
        float f0_scalar = (mat.ior - 1.0) / (mat.ior + 1.0);
        f0_scalar *= f0_scalar;
        float3 F0 = mix(float3(f0_scalar) * mat.specularColorFactor.xyz, baseColor.rgb, metallic);
        // Apply thin-film iridescence to IBL Fresnel.
        if (iridescenceFactor > 0.0) {
            float3 iridF0 = ThinFilmIridescence(NdotV, iridescenceThickness, mat.iridescenceIor);
            F0 = mix(F0, iridF0, iridescenceFactor);
        }
        float fresnel = pow(1.0 - NdotV, 5.0);
        float3 F = F0 + (float3(1.0) - F0) * fresnel;
        iblSpecular = envSpecular * F * mat.environmentIntensity;

        // IBL clearcoat: sample environment with clearcoat normal and Fresnel.
        if (ccFactor > 0.0) {
            float3 ccR = reflect(-viewDir, ccN);
            float3 envCC = environmentMap.sample(baseColorSampler, ccR).rgb;
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
            thickness *= thicknessTexture.sample(baseColorSampler, uv).r;
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

        constexpr sampler scSampler(coord::normalized, filter::linear, address::clamp_to_edge);
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
            float3 iridF0 = ThinFilmIridescence(NdotV, iridescenceThickness, mat.iridescenceIor);
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
        totalDiffuse += baseColor.rgb * NdotL * (1.0 - metallic) * lightColor;

        float3 halfDir = normalize(lightDir + viewDir);
        float  NdotH   = max(dot(N, halfDir), 0.0);

        // KHR_materials_specular F0.
        float f0_scalar = (mat.ior - 1.0) / (mat.ior + 1.0);
        f0_scalar *= f0_scalar;

        float spec = mat.specularFactor;
        if (mat.hasSpecularTexture > 0.5)
            spec *= specularTexture.sample(specularSampler, uv).a;

        float3 specColor = mat.specularColorFactor.xyz;
        if (mat.hasSpecularColorTexture > 0.5)
            specColor *= specularColorTexture.sample(specularColorSampler, uv).rgb;

        float3 F0_dielectric = min(float3(f0_scalar) * specColor, float3(1.0)) * spec;
        float3 F0 = mix(F0_dielectric, baseColor.rgb, metallic);
        // Apply thin-film iridescence to direct-light specular F0.
        if (iridescenceFactor > 0.0) {
            float3 iridF0 = ThinFilmIridescence(NdotV, iridescenceThickness, mat.iridescenceIor);
            F0 = mix(F0, iridF0, iridescenceFactor);
        }

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
        totalSpecular += F0 * D * V * NdotL * lightColor;

        // KHR_materials_sheen.
        float sheenD = D_Charlie(sheenRoughness, NdotH);
        float sheenV = V_Neubelt(NdotV, max(NdotL, 0.0001f));
        totalSheen += sheenColor * sheenD * sheenV * NdotL * lightColor;

        // KHR_materials_clearcoat.
        float VdotH   = max(dot(viewDir, halfDir), 0.0f);
        float ccNdotH = max(dot(ccN, halfDir), 0.0f);
        float ccNdotL = max(dot(ccN, lightDir), 0.0f);
        float cc_D    = D_GGX(ccRoughness, ccNdotH);
        float cc_V    = V_SmithGGX(ccNdotL, ccNdotV, ccRoughness);
        float cc_F    = F_Schlick_scalar(0.04f, VdotH);
        totalClearcoat += float3(cc_D * cc_V * cc_F) * ccFactor * ccNdotL * lightColor;
    }

    float3 ambientColor = baseColor.rgb * 0.25 * occlusion;
    float3 diffuse      = totalDiffuse * occlusion;

    // Scale diffuse/ambient terms by (1 - transmittance); specular/clearcoat remain.
    float diffuseScale = 1.0 - transmittance;
    ambientColor *= diffuseScale;
    diffuse      *= diffuseScale;
    iblDiffuse   *= diffuseScale;

    float ccAmbientAtten = 1.0f - ccFactor * F_Schlick_scalar(0.04f, ccNdotV);
    float3 finalColor = (ambientColor + diffuse + totalSpecular + totalSheen + iblDiffuse + iblSpecular) * ccAmbientAtten
                        + totalClearcoat + iblClearcoat + emissive
                        + transmitted * transmittance;

    // Reinhard tone mapping.
    finalColor = finalColor / (float3(1.0) + finalColor);

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
    return envMap.sample(envSampler, worldDir);
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
