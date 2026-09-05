#version 450

// campello_renderer default Vulkan fragment shader — full glTF PBR, transliterated
// directly from shaders/metal/default.metal's fragmentMain_textured (the single
// source of truth for this math). Implements: metallic-roughness Cook-Torrance
// GGX BRDF, image-based lighting (diffuse irradiance + roughness-blurred specular
// reflection from an environment cubemap), normal mapping, and KHR_materials_
// specular/anisotropy/iridescence/clearcoat/sheen/transmission.
//
// MaterialUniforms/CameraUniforms/LightsUniform are read as std140 UBOs here
// (set 0 binding 24, set 1 bindings 1/0) instead of Metal's `constant`/argument-
// buffer reads — binding numbers are fresh/Vulkan-only (see campello_renderer.cpp's
// ensureVulkanPbrBindGroupLayouts()) because the shared bindGroupLayout Metal also
// uses has real 17/18 binding-number collisions with two of the clearcoat texture
// bindings that are legal on Metal's independent texture/buffer argument-index
// spaces but not on Vulkan's single-descriptor-type-per-binding model.
//
// KHR_texture_transform is applied here in the fragment stage (not the vertex
// stage, unlike the vertex-attribute-based placeholder this replaces) — affine 2D
// transforms commute with barycentric interpolation, so this is exact, not an
// approximation, and keeps MaterialUniforms fragment-stage-only.

// --- Material (set 0, binding 24) ---------------------------------------------
// Byte-exact mirror of MaterialUniforms in default.metal (and the authoritative
// C++ writer, Renderer::buildSlotRaw() in campello_renderer.cpp) — every scalar
// gap below is a deliberate, explicitly-sized padding run reproducing Metal's own
// (non-minimal) manual alignment, not GLSL's default std140 packing. Do not
// remove the _padN fields or reorder members — offsets must match byte-for-byte
// since both platforms read the same uploaded buffer.
layout(std140, set = 0, binding = 24) uniform MaterialUniforms {
    vec4  baseColorFactor;           // 0
    vec4  uvTransformRow0;           // 16   [a, b, tx, hasTransform]
    vec4  uvTransformRow1;           // 32   [c, d, ty, 0]
    float metallicFactor;            // 48
    float roughnessFactor;           // 52
    float normalScale;               // 56
    float alphaMode;                 // 60   0=opaque, 1=mask, 2=blend
    float alphaCutoff;               // 64
    float unlit;                     // 68
    float hasNormalTexture;          // 72
    float hasEmissiveTexture;        // 76
    float hasOcclusionTexture;       // 80
    float occlusionStrength;         // 84
    float _pad0;                     // 88
    float _pad1;                     // 92
    vec4  emissiveFactor;            // 96
    float ior;                       // 112
    float specularFactor;            // 116
    float hasSpecularTexture;        // 120
    float hasSpecularColorTexture;   // 124
    float _pad2;                     // 128
    float _pad3;                     // 132
    float _pad4;                     // 136
    float _pad5;                     // 140
    vec4  specularColorFactor;       // 144
    float _pad6;                     // 160
    float _pad7;                     // 164
    float _pad8;                     // 168
    float _pad9;                     // 172
    vec4  sheenColorFactor;          // 176
    float sheenRoughnessFactor;      // 192
    float hasSheenColorTexture;      // 196
    float hasSheenRoughnessTexture;  // 200
    float clearcoatFactor;           // 204
    float clearcoatRoughnessFactor;  // 208
    float hasClearcoatTexture;       // 212
    float hasClearcoatRoughnessTexture; // 216
    float hasClearcoatNormalTexture; // 220
    float clearcoatNormalScale;      // 224
    float transmissionFactor;        // 228
    float hasTransmissionTexture;    // 232
    float thicknessFactor;           // 236
    float attenuationDistance;       // 240
    float hasThicknessTexture;       // 244
    float _pad10;                    // 248
    float _pad11;                    // 252
    vec4  attenuationColor;          // 256
    float viewMode;                  // 272
    float environmentIntensity;      // 276
    float iblEnabled;                // 280
    float iridescenceFactor;         // 284
    float iridescenceIor;            // 288
    float iridescenceThicknessMin;   // 292
    float iridescenceThicknessMax;   // 296
    float hasIridescenceTexture;     // 300
    float hasIridescenceThicknessTexture; // 304
    float anisotropyStrength;        // 308
    float anisotropyRotation;        // 312
    float hasAnisotropicTexture;     // 316
    float dispersion;                // 320
    float _pad12;                    // 324
    float _pad13;                    // 328
    float _pad14;                    // 332
    vec4  normalUvTransformRow0;     // 336  independent KHR_texture_transform for normalTexture
    vec4  normalUvTransformRow1;     // 352
    float texCoord1Mask;             // 368
} mat;

// --- Per-frame (set 1) ----------------------------------------------------------
struct Light {
    vec4 position;   // xyz = position/dir, w = type (0=dir, 1=point, 2=spot)
    vec4 color;      // xyz = rgb, w = intensity
    vec4 direction;  // xyz = spot dir, w = range
    vec4 spotAngles; // x = innerCone, y = outerCone, zw = padding
};
layout(std140, set = 1, binding = 0) uniform LightsUniform {
    uint  lightCount;
    uint  _lpad0;
    uint  _lpad1;
    uint  _lpad2;
    Light lights[4];
} lightsBlock;

layout(std140, set = 1, binding = 1) uniform CameraUniforms {
    vec4 cameraPos;    // xyz used, w unused
    mat4 viewMatrix;
    mat4 projMatrix;
    vec2 screenSize;
    vec2 _cpad;
} camera;

layout(set = 1, binding = 2) uniform textureCube environmentMap;
layout(set = 1, binding = 3) uniform sampler     environmentSampler;
layout(set = 1, binding = 4) uniform texture2D   sceneColorTexture;
layout(set = 1, binding = 5) uniform sampler     sceneColorSampler;
// IBL precompute resources — see Renderer::bakeIblResources() /
// shaders/vulkan/ibl_bake.frag. environmentMap above (binding 2) is now the
// GGX-prefiltered specular cubemap (bakeIblResources() rebinds it there);
// irradianceMap is the separate Lambertian-convolved diffuse cubemap.
layout(set = 1, binding = 6) uniform textureCube irradianceMap;
layout(set = 1, binding = 7) uniform texture2D   brdfLutTexture;
layout(set = 1, binding = 8) uniform sampler     brdfLutSampler;

// --- Per-material textures/samplers (set 0) -------------------------------------
layout(set = 0, binding = 0)  uniform texture2D baseColorTexture;
layout(set = 0, binding = 1)  uniform sampler   baseColorSampler;
layout(set = 0, binding = 2)  uniform texture2D metallicRoughnessTexture;
layout(set = 0, binding = 3)  uniform sampler   metallicRoughnessSampler;
layout(set = 0, binding = 4)  uniform texture2D normalTexture;
layout(set = 0, binding = 5)  uniform sampler   normalSampler;
layout(set = 0, binding = 6)  uniform texture2D emissiveTexture;
layout(set = 0, binding = 7)  uniform sampler   emissiveSampler;
layout(set = 0, binding = 8)  uniform texture2D occlusionTexture;
layout(set = 0, binding = 9)  uniform sampler   occlusionSampler;
layout(set = 0, binding = 10) uniform texture2D specularTexture;
layout(set = 0, binding = 11) uniform sampler   specularSampler;
layout(set = 0, binding = 12) uniform texture2D specularColorTexture;
layout(set = 0, binding = 13) uniform sampler   specularColorSampler;
layout(set = 0, binding = 14) uniform texture2D sheenColorTexture;          // reuses baseColorSampler
layout(set = 0, binding = 15) uniform texture2D sheenRoughnessTexture;      // reuses baseColorSampler
layout(set = 0, binding = 16) uniform texture2D clearcoatTexture;           // reuses baseColorSampler
layout(set = 0, binding = 17) uniform texture2D clearcoatRoughnessTexture;  // reuses baseColorSampler
layout(set = 0, binding = 18) uniform texture2D clearcoatNormalTexture;     // reuses baseColorSampler
layout(set = 0, binding = 19) uniform texture2D transmissionTexture;        // reuses baseColorSampler
layout(set = 0, binding = 20) uniform texture2D thicknessTexture;           // reuses baseColorSampler
layout(set = 0, binding = 21) uniform texture2D iridescenceTexture;         // reuses baseColorSampler
layout(set = 0, binding = 22) uniform texture2D iridescenceThicknessTexture;// reuses baseColorSampler
layout(set = 0, binding = 23) uniform texture2D anisotropicTexture;         // reuses baseColorSampler

// --- Vertex stage inputs ---------------------------------------------------------
layout(location = 0) in vec3 worldPos;
layout(location = 1) in vec3 worldNormalIn;
layout(location = 2) in vec3 worldTangentIn;
layout(location = 3) in vec3 worldBitangentIn;
layout(location = 4) in vec2 fragTexcoord0;
layout(location = 5) in vec2 fragTexcoord1;
layout(location = 6) in vec4 fragColor0;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265359;

// --- KHR_texture_transform / TEXCOORD_1 selection -------------------------------
const uint kUV1BaseColor            = 1u << 0;
const uint kUV1MetallicRoughness    = 1u << 1;
const uint kUV1Normal               = 1u << 2;
const uint kUV1Emissive             = 1u << 3;
const uint kUV1Occlusion            = 1u << 4;
const uint kUV1Specular             = 1u << 5;
const uint kUV1SpecularColor        = 1u << 6;
const uint kUV1SheenColor           = 1u << 7;
const uint kUV1SheenRoughness       = 1u << 8;
const uint kUV1Clearcoat            = 1u << 9;
const uint kUV1ClearcoatRoughness   = 1u << 10;
const uint kUV1ClearcoatNormal      = 1u << 11;
const uint kUV1Transmission         = 1u << 12;
const uint kUV1Thickness            = 1u << 13;
const uint kUV1Iridescence          = 1u << 14;
const uint kUV1IridescenceThickness = 1u << 15;
const uint kUV1Anisotropic          = 1u << 16;

vec2 selectUV(vec2 uv0, vec2 uv1, float packedMask, uint bit) {
    return (uint(packedMask) & bit) != 0u ? uv1 : uv0;
}

vec2 applyUvTransform(vec2 uv, vec4 row0, vec4 row1) {
    vec3 uv3 = vec3(uv, 1.0);
    return vec2(dot(row0.xyz, uv3), dot(row1.xyz, uv3));
}

// --- BRDF helper functions (verbatim transliteration of default.metal) ---------
float D_GGX(float roughness, float NdotH) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float d  = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d + 0.0001);
}

float V_SmithGGX(float NdotL, float NdotV, float roughness) {
    float a  = roughness * roughness;
    float GL = NdotV * sqrt(NdotL * NdotL * (1.0 - a) + a);
    float GV = NdotL * sqrt(NdotV * NdotV * (1.0 - a) + a);
    return 0.5 / max(GL + GV, 0.0001);
}

float F_Schlick_scalar(float F0, float cosTheta) {
    float x = 1.0 - cosTheta;
    return F0 + (1.0 - F0) * (x * x * x * x * x);
}

const mat3 kXyzToRec709 = mat3(
    3.2404542, -0.9692660,  0.0556434,
   -1.5371385,  1.8760108, -0.2040259,
   -0.4985314,  0.0415560,  1.0572252
);

vec3 Fresnel0ToIor(vec3 fresnel0) {
    vec3 sqrtF0 = sqrt(clamp(fresnel0, 0.0, 0.9999));
    return (vec3(1.0) + sqrtF0) / (vec3(1.0) - sqrtF0);
}

float FresnelDielectric(float cosTheta1, float n1, float n2) {
    float sinTheta2Sq = (n1 * n1) / (n2 * n2) * (1.0 - cosTheta1 * cosTheta1);
    if (sinTheta2Sq > 1.0) return 1.0;
    float cosTheta2 = sqrt(1.0 - sinTheta2Sq);
    float r_s = (n1 * cosTheta1 - n2 * cosTheta2) / (n1 * cosTheta1 + n2 * cosTheta2);
    float r_p = (n2 * cosTheta1 - n1 * cosTheta2) / (n2 * cosTheta1 + n1 * cosTheta2);
    return 0.5 * (r_s * r_s + r_p * r_p);
}

vec3 FresnelDielectric3(float cosTheta1, float n1, vec3 n2) {
    vec3 sinTheta2Sq = (vec3(n1 * n1) / (n2 * n2)) * (1.0 - cosTheta1 * cosTheta1);
    vec3 cosTheta2 = sqrt(clamp(vec3(1.0) - sinTheta2Sq, vec3(0.0), vec3(1.0)));
    vec3 r_s = (n1 * cosTheta1 - n2 * cosTheta2) / (n1 * cosTheta1 + n2 * cosTheta2);
    vec3 r_p = (n2 * cosTheta1 - n1 * cosTheta2) / (n2 * cosTheta1 + n1 * cosTheta2);
    return 0.5 * (r_s * r_s + r_p * r_p);
}

vec3 EvalSensitivity(float opd, vec3 shift) {
    float phase = 2.0 * PI * opd * 1.0e-9;
    vec3 val = vec3(5.4856e-13, 4.4201e-13, 5.2481e-13);
    vec3 pos = vec3(1.6810e+06, 1.7953e+06, 2.2084e+06);
    vec3 var = vec3(4.3278e+09, 9.3046e+09, 6.6121e+09);

    vec3 xyz = val * sqrt(2.0 * PI * var) * cos(pos * phase + shift) * exp(-phase * phase * var);
    xyz.x += 9.7470e-14 * sqrt(2.0 * PI * 4.5282e+09) * cos(2.2399e+06 * phase + shift.x) * exp(-4.5282e+09 * phase * phase);
    xyz /= 1.0685e-7;

    return kXyzToRec709 * xyz;
}

vec3 EvalIridescence(float outsideIOR, float eta2, float cosTheta1, float thinFilmThickness, vec3 baseF0) {
    float iridescenceIor = mix(outsideIOR, eta2, smoothstep(0.0, 0.03, thinFilmThickness));
    float sinTheta2Sq = (outsideIOR * outsideIOR) / (iridescenceIor * iridescenceIor)
                        * (1.0 - cosTheta1 * cosTheta1);
    float cosTheta2Sq = 1.0 - sinTheta2Sq;
    if (cosTheta2Sq < 0.0) {
        return vec3(1.0);
    }
    float cosTheta2 = sqrt(cosTheta2Sq);

    float R12   = FresnelDielectric(cosTheta1, outsideIOR, iridescenceIor);
    float T121  = 1.0 - R12;
    float phi12 = (iridescenceIor < outsideIOR) ? PI : 0.0;
    float phi21 = PI - phi12;

    vec3 baseIOR = Fresnel0ToIor(baseF0);
    vec3 R23   = FresnelDielectric3(cosTheta2, iridescenceIor, baseIOR);
    vec3 phi23 = vec3(baseIOR.x < iridescenceIor ? PI : 0.0,
                       baseIOR.y < iridescenceIor ? PI : 0.0,
                       baseIOR.z < iridescenceIor ? PI : 0.0);

    float opd = 2.0 * iridescenceIor * thinFilmThickness * cosTheta2;
    vec3  phi = vec3(phi21) + phi23;

    vec3 R123 = clamp(vec3(R12) * R23, vec3(1e-5), vec3(0.9999));
    vec3 r123 = sqrt(R123);
    vec3 Rs   = (T121 * T121) * R23 / (vec3(1.0) - R123);

    vec3 C0 = vec3(R12) + Rs;
    vec3 I  = C0;

    vec3 Cm = Rs - vec3(T121);
    for (int m = 1; m <= 2; ++m) {
        Cm *= r123;
        vec3 Sm = 2.0 * EvalSensitivity(float(m) * opd, float(m) * phi);
        I += Cm * Sm;
    }

    return max(I, vec3(0.0));
}

float D_GGX_Anisotropic(float NdotH, float HdotT, float HdotB, float ax, float ay) {
    float X = HdotT / ax;
    float Y = HdotB / ay;
    float tmp = X * X + Y * Y + NdotH * NdotH;
    return 1.0 / (PI * ax * ay * tmp * tmp);
}

float D_Charlie(float roughness, float NdotH) {
    float invAlpha = 1.0 / max(roughness * roughness, 0.0001);
    float cos2h    = NdotH * NdotH;
    float sin2h    = max(1.0 - cos2h, 0.0078125);
    return (2.0 + invAlpha) * pow(sin2h, invAlpha * 0.5) / (2.0 * PI);
}

float V_Neubelt(float NdotV, float NdotL) {
    return clamp(1.0 / (4.0 * (NdotL + NdotV - NdotL * NdotV)), 0.0, 1.0);
}

// Roughness-aware IBL Fresnel with single-scattering (FssEss) + multi-scattering
// (FmsEms) energy compensation — glTF-Sample-Renderer's getIBLGGXFresnel
// (ibl.glsl), from Fdez-Aguera's "A Multiple-Scattering Microfacet Model for
// Real-Time Image-based Lighting" (https://bruop.github.io/ibl/#single_scattering_results).
// Replaces the plain pow(1-NdotV,5) Schlick this used to use for IBL specular,
// which had no roughness dependence at all (rough dielectrics kept a bright
// grazing rim they shouldn't, and rough metals lost energy with no
// compensation).
vec3 iblGGXFresnel(float NdotV, float roughness, vec3 F0) {
    vec2 fAB = texture(sampler2D(brdfLutTexture, brdfLutSampler), clamp(vec2(NdotV, roughness), 0.0, 1.0)).rg;
    vec3 Fr = max(vec3(1.0 - roughness), F0) - F0;
    vec3 kS = F0 + Fr * pow(clamp(1.0 - NdotV, 0.0, 1.0), 5.0);
    vec3 FssEss = kS * fAB.x + fAB.y;

    float Ems = 1.0 - (fAB.x + fAB.y);
    vec3 Favg = F0 + (vec3(1.0) - F0) / 21.0;
    vec3 FmsEms = Ems * FssEss * Favg / (vec3(1.0) - Favg * Ems);

    return FssEss + FmsEms;
}

// Khronos PBR Neutral tone mapping — the glTF-Sample-Renderer's actual current
// default (see GltfState.ToneMaps.KHR_PBR_NEUTRAL), not the plain Reinhard this
// replaces. Ported verbatim from the reference's tonemapping.glsl.
vec3 toneMapKhronosPbrNeutral(vec3 color) {
    const float startCompression = 0.8 - 0.04;
    const float desaturation = 0.15;

    float x = min(color.r, min(color.g, color.b));
    float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
    color -= offset;

    float peak = max(color.r, max(color.g, color.b));
    if (peak < startCompression) return color;

    float d = 1.0 - startCompression;
    float newPeak = 1.0 - d * d / (peak + d - startCompression);
    color *= newPeak / peak;

    float g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
    return mix(color, vec3(newPeak), g);
}

// The swapchain target is plain (non-sRGB-tagged) unorm — see
// Device::getSwapchainPixelFormat() picking VK_FORMAT_B8G8R8A8_UNORM — so, as in
// the reference's own toneMap() wrapper, linear-to-sRGB encoding must happen here
// rather than relying on the hardware to do it on write.
vec3 linearToSRGBFast(vec3 color) {
    return pow(color, vec3(1.0 / 2.2));
}

void main() {
    vec2 uv0 = fragTexcoord0;
    if (mat.uvTransformRow0.w > 0.5) {
        uv0 = applyUvTransform(fragTexcoord0, mat.uvTransformRow0, mat.uvTransformRow1);
    }
    vec2 baseColorUV = (mat.uvTransformRow0.w > 0.5)
        ? uv0 : selectUV(uv0, fragTexcoord1, mat.texCoord1Mask, kUV1BaseColor);

    vec4 baseColor = texture(sampler2D(baseColorTexture, baseColorSampler), baseColorUV)
                    * mat.baseColorFactor * fragColor0;

    float transmission = mat.transmissionFactor;
    if (mat.hasTransmissionTexture > 0.5) {
        vec2 transUV = selectUV(uv0, fragTexcoord1, mat.texCoord1Mask, kUV1Transmission);
        transmission *= texture(sampler2D(transmissionTexture, baseColorSampler), transUV).r;
    }

    // Alpha mask.
    if (mat.alphaMode > 0.5 && mat.alphaMode < 1.5 && baseColor.a < mat.alphaCutoff) {
        discard;
    }

    // Unlit: return base color without lighting.
    if (mat.unlit > 0.5) {
        outColor = baseColor;
        return;
    }

    // Metallic-roughness.
    vec2 mrUV = selectUV(uv0, fragTexcoord1, mat.texCoord1Mask, kUV1MetallicRoughness);
    vec4 mrSample = texture(sampler2D(metallicRoughnessTexture, metallicRoughnessSampler), mrUV);
    float metallic  = mrSample.b * mat.metallicFactor;
    float roughness = clamp(mrSample.g * mat.roughnessFactor, 0.04, 1.0);

    // Tangent frame (needed for normal mapping and anisotropy).
    vec3 T = normalize(worldTangentIn);
    vec3 B = normalize(worldBitangentIn);
    vec3 N;
    if (mat.hasNormalTexture > 0.5) {
        vec2 normalUV = (mat.normalUvTransformRow0.w > 0.5)
            ? applyUvTransform(fragTexcoord0, mat.normalUvTransformRow0, mat.normalUvTransformRow1)
            : selectUV(fragTexcoord0, fragTexcoord1, mat.texCoord1Mask, kUV1Normal);
        vec3 ns = texture(sampler2D(normalTexture, normalSampler), normalUV).rgb * 2.0 - 1.0;
        ns.xy *= mat.normalScale;
        vec3 Nbase = normalize(worldNormalIn);
        N = normalize(mat3(T, B, Nbase) * normalize(ns));
    } else {
        N = normalize(worldNormalIn);
    }

    // Occlusion.
    float occlusion = 1.0;
    if (mat.hasOcclusionTexture > 0.5) {
        vec2 occUV = selectUV(uv0, fragTexcoord1, mat.texCoord1Mask, kUV1Occlusion);
        float occ = texture(sampler2D(occlusionTexture, occlusionSampler), occUV).r;
        occlusion = mix(1.0, occ, mat.occlusionStrength);
    }

    // Emissive.
    vec3 emissive = mat.emissiveFactor.xyz;
    if (mat.hasEmissiveTexture > 0.5) {
        vec2 emisUV = selectUV(uv0, fragTexcoord1, mat.texCoord1Mask, kUV1Emissive);
        emissive *= texture(sampler2D(emissiveTexture, emissiveSampler), emisUV).rgb;
    }

    float specularFactor = mat.specularFactor;
    if (mat.hasSpecularTexture > 0.5) {
        vec2 specUV = selectUV(uv0, fragTexcoord1, mat.texCoord1Mask, kUV1Specular);
        specularFactor *= texture(sampler2D(specularTexture, specularSampler), specUV).a;
    }

    vec3 specularColorFactorSampled = mat.specularColorFactor.xyz;
    if (mat.hasSpecularColorTexture > 0.5) {
        vec2 specColUV = selectUV(uv0, fragTexcoord1, mat.texCoord1Mask, kUV1SpecularColor);
        specularColorFactorSampled *= texture(sampler2D(specularColorTexture, specularColorSampler), specColUV).rgb;
    }

    // Anisotropy.
    float anisoStrength = mat.anisotropyStrength;
    float anisoRotation = mat.anisotropyRotation;
    if (mat.hasAnisotropicTexture > 0.5) {
        vec2 anisoUV = selectUV(uv0, fragTexcoord1, mat.texCoord1Mask, kUV1Anisotropic);
        vec2 anisoTex = texture(sampler2D(anisotropicTexture, baseColorSampler), anisoUV).rg;
        anisoStrength *= anisoTex.r;
        anisoRotation += anisoTex.g * 2.0 * PI;
    }

    // KHR_materials_iridescence.
    float iridescenceFactor = mat.iridescenceFactor;
    if (mat.hasIridescenceTexture > 0.5) {
        vec2 iridUV = selectUV(uv0, fragTexcoord1, mat.texCoord1Mask, kUV1Iridescence);
        iridescenceFactor *= texture(sampler2D(iridescenceTexture, baseColorSampler), iridUV).r;
    }
    float iridescenceThickness = mat.iridescenceThicknessMax;
    if (mat.hasIridescenceThicknessTexture > 0.5) {
        vec2 iridThickUV = selectUV(uv0, fragTexcoord1, mat.texCoord1Mask, kUV1IridescenceThickness);
        iridescenceThickness = mix(mat.iridescenceThicknessMin, mat.iridescenceThicknessMax,
            texture(sampler2D(iridescenceThicknessTexture, baseColorSampler), iridThickUV).g);
    }

    // KHR_materials_sheen.
    vec3 sheenColor = mat.sheenColorFactor.xyz;
    if (mat.hasSheenColorTexture > 0.5) {
        vec2 sheenColUV = selectUV(uv0, fragTexcoord1, mat.texCoord1Mask, kUV1SheenColor);
        sheenColor *= texture(sampler2D(sheenColorTexture, baseColorSampler), sheenColUV).rgb;
    }
    float sheenRoughness = clamp(mat.sheenRoughnessFactor, 0.07, 1.0);
    if (mat.hasSheenRoughnessTexture > 0.5) {
        vec2 sheenRoughUV = selectUV(uv0, fragTexcoord1, mat.texCoord1Mask, kUV1SheenRoughness);
        sheenRoughness = clamp(sheenRoughness * texture(sampler2D(sheenRoughnessTexture, baseColorSampler), sheenRoughUV).r, 0.07, 1.0);
    }

    // KHR_materials_clearcoat.
    float ccFactor = mat.clearcoatFactor;
    if (mat.hasClearcoatTexture > 0.5) {
        vec2 ccUV = selectUV(uv0, fragTexcoord1, mat.texCoord1Mask, kUV1Clearcoat);
        ccFactor *= texture(sampler2D(clearcoatTexture, baseColorSampler), ccUV).r;
    }
    ccFactor = clamp(ccFactor, 0.0, 1.0);

    float ccRoughness = clamp(mat.clearcoatRoughnessFactor, 0.001, 1.0);
    if (mat.hasClearcoatRoughnessTexture > 0.5) {
        vec2 ccRoughUV = selectUV(uv0, fragTexcoord1, mat.texCoord1Mask, kUV1ClearcoatRoughness);
        ccRoughness = clamp(ccRoughness * texture(sampler2D(clearcoatRoughnessTexture, baseColorSampler), ccRoughUV).g, 0.001, 1.0);
    }

    vec3 ccN = normalize(worldNormalIn);
    if (mat.hasClearcoatNormalTexture > 0.5) {
        vec2 ccNormUV = selectUV(uv0, fragTexcoord1, mat.texCoord1Mask, kUV1ClearcoatNormal);
        vec3 ccNS = texture(sampler2D(clearcoatNormalTexture, baseColorSampler), ccNormUV).rgb * 2.0 - 1.0;
        ccNS.xy *= mat.clearcoatNormalScale;
        vec3 T2    = normalize(worldTangentIn);
        vec3 B2    = normalize(worldBitangentIn);
        vec3 Nbase = normalize(worldNormalIn);
        ccN = normalize(mat3(T2, B2, Nbase) * normalize(ccNS));
    }

    // View direction & double-sided normal flip.
    vec3 camPos  = camera.cameraPos.xyz;
    vec3 viewDir = normalize(camPos - worldPos);

    if (dot(N, viewDir) < 0.0) N = -N;
    if (dot(ccN, viewDir) < 0.0) ccN = -ccN;

    float NdotV   = max(dot(N, viewDir), 0.0001);
    float ccNdotV = max(dot(ccN, viewDir), 0.0001);

    // --- IBL (image-based lighting) from environment cubemap ---
    vec3 iblDiffuse   = vec3(0.0);
    vec3 iblSpecular  = vec3(0.0);
    vec3 iblClearcoat = vec3(0.0);
    if (mat.iblEnabled > 0.5) {
        // Lambertian-convolved diffuse irradiance (bakeIblResources() mode 2) —
        // not a raw environmentMap sample, so no hand-tuned brightness fudge
        // factor is needed the way the old raw-sample approximation required.
        vec3 envDiffuse = texture(samplerCube(irradianceMap, environmentSampler), N).rgb;
        iblDiffuse = baseColor.rgb * (1.0 - metallic) * envDiffuse * mat.environmentIntensity;

        // environmentMap here is the GGX-prefiltered specular cubemap
        // (bakeIblResources() mode 1) — same roughness*envMaxLod mip selection
        // as before, but each mip now actually holds that roughness's GGX
        // lobe instead of a box-filtered downsample.
        float envMaxLod = max(float(textureQueryLevels(samplerCube(environmentMap, environmentSampler))) - 1.0, 0.0);

        vec3 R = reflect(-viewDir, N);
        vec3 envSpecular = textureLod(samplerCube(environmentMap, environmentSampler), R, roughness * envMaxLod).rgb;
        float f0_scalar = (mat.ior - 1.0) / (mat.ior + 1.0);
        f0_scalar *= f0_scalar;
        vec3 F0 = mix(vec3(f0_scalar) * mat.specularColorFactor.xyz, baseColor.rgb, metallic);
        if (iridescenceFactor > 0.0) {
            vec3 iridF0 = EvalIridescence(1.0, mat.iridescenceIor, NdotV, iridescenceThickness, F0);
            F0 = mix(F0, iridF0, iridescenceFactor);
        }
        vec3 F = iblGGXFresnel(NdotV, roughness, F0);
        iblSpecular = envSpecular * F * mat.environmentIntensity;

        if (ccFactor > 0.0) {
            vec3 ccR = reflect(-viewDir, ccN);
            vec3 envCC = textureLod(samplerCube(environmentMap, environmentSampler), ccR, ccRoughness * envMaxLod).rgb;
            float ccFresnel = F_Schlick_scalar(0.04, ccNdotV);
            iblClearcoat = envCC * ccFresnel * ccFactor * mat.environmentIntensity;
        }
    }

    // --- KHR_materials_transmission: screen-space refraction ---
    vec3  transmitted   = vec3(0.0);
    float transmittance = 0.0;
    if (transmission > 0.0 && metallic < 0.99) {
        float eta = 1.0 / mat.ior;
        vec3 Trefr = refract(-viewDir, N, eta);
        if (all(equal(Trefr, vec3(0.0)))) Trefr = -viewDir;

        float thickness = mat.thicknessFactor;
        if (mat.hasThicknessTexture > 0.5) {
            vec2 thickUV = selectUV(uv0, fragTexcoord1, mat.texCoord1Mask, kUV1Thickness);
            thickness *= texture(sampler2D(thicknessTexture, baseColorSampler), thickUV).r;
        }

        vec2 sampleUV[3];
        bool uvValid[3];
        sampleUV[0] = sampleUV[1] = sampleUV[2] = vec2(0.0);
        uvValid[0] = uvValid[1] = uvValid[2] = true;
        if (thickness > 0.0) {
            vec3 viewPos = (camera.viewMatrix * vec4(worldPos, 1.0)).xyz;
            vec3 viewNormal = normalize((camera.viewMatrix * vec4(N, 0.0)).xyz);
            vec3 viewIncident = normalize(viewPos);
            if (mat.dispersion > 0.0) {
                float dispersionScale = mat.dispersion * 0.02;
                vec3 iorRGB = vec3(mat.ior + dispersionScale, mat.ior, max(mat.ior - dispersionScale, 1.001));
                for (int ch = 0; ch < 3; ch++) {
                    float etaCh = 1.0 / iorRGB[ch];
                    vec3 viewRefractCh = refract(viewIncident, viewNormal, etaCh);
                    if (all(equal(viewRefractCh, vec3(0.0)))) viewRefractCh = viewIncident;
                    vec3 backPosCh = viewPos + viewRefractCh * thickness;
                    vec4 backClipCh = camera.projMatrix * vec4(backPosCh, 1.0);
                    sampleUV[ch] = backClipCh.xy / backClipCh.w * 0.5 + 0.5;
                    sampleUV[ch].y = 1.0 - sampleUV[ch].y;
                    uvValid[ch] = all(greaterThanEqual(sampleUV[ch], vec2(0.0))) && all(lessThanEqual(sampleUV[ch], vec2(1.0)));
                }
            } else {
                vec3 viewRefract = refract(viewIncident, viewNormal, eta);
                if (all(equal(viewRefract, vec3(0.0)))) viewRefract = viewIncident;
                vec3 backPos = viewPos + viewRefract * thickness;
                vec4 backClip = camera.projMatrix * vec4(backPos, 1.0);
                sampleUV[0] = sampleUV[1] = sampleUV[2] = backClip.xy / backClip.w * 0.5 + 0.5;
                sampleUV[0].y = 1.0 - sampleUV[0].y;
                uvValid[0] = uvValid[1] = uvValid[2] = all(greaterThanEqual(sampleUV[0], vec2(0.0))) && all(lessThanEqual(sampleUV[0], vec2(1.0)));
            }
        } else {
            sampleUV[0] = sampleUV[1] = sampleUV[2] = gl_FragCoord.xy / camera.screenSize;
        }

        // Official glTF-Sample-Viewer LOD formula.
        float iorScale = clamp(mat.ior * 2.0 - 2.0, 0.0, 1.0);
        float lod = log2(camera.screenSize.x) * roughness * iorScale;

        for (int ch = 0; ch < 3; ch++) {
            if (uvValid[ch]) {
                transmitted[ch] = textureLod(sampler2D(sceneColorTexture, sceneColorSampler), sampleUV[ch], lod)[ch];
            } else {
                transmitted[ch] = texture(samplerCube(environmentMap, environmentSampler), Trefr)[ch];
            }
        }

        if (thickness > 0.0 && mat.attenuationDistance > 0.0 && !isinf(mat.attenuationDistance)) {
            vec3 attn = (vec3(1.0) - mat.attenuationColor.xyz) / mat.attenuationDistance;
            transmitted *= exp(-thickness * attn);
        }

        transmitted *= baseColor.rgb;

        float f0 = (mat.ior - 1.0) / (mat.ior + 1.0);
        f0 *= f0;
        float fresnelT = f0 + (1.0 - f0) * pow(1.0 - NdotV, 5.0);
        transmittance = transmission * (1.0 - fresnelT) * (1.0 - metallic);
    }

    // --- Punctual light loop (up to 4 lights) ---
    vec3 totalDiffuse   = vec3(0.0);
    vec3 totalSpecular  = vec3(0.0);
    vec3 totalSheen     = vec3(0.0);
    vec3 totalClearcoat = vec3(0.0);

    uint lightCount = lightsBlock.lightCount;
    for (uint i = 0u; i < lightCount && i < 4u; i++) {
        Light light = lightsBlock.lights[i];

        vec3  lightDir;
        float attenuation = 1.0;
        float spotFactor   = 1.0;
        float typeVal       = light.position.w;

        if (typeVal < 0.5) {
            lightDir = normalize(light.position.xyz);
        } else {
            vec3 toLight = light.position.xyz - worldPos;
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

        float NdotL      = max(dot(N, lightDir), 0.0);
        vec3  lightColor = light.color.xyz * light.color.w * attenuation * spotFactor;

        vec3  halfDir = normalize(lightDir + viewDir);
        float NdotH   = max(dot(N, halfDir), 0.0);
        float VdotH   = max(dot(viewDir, halfDir), 0.0);

        float f0_scalarL = (mat.ior - 1.0) / (mat.ior + 1.0);
        f0_scalarL *= f0_scalarL;

        float spec = specularFactor;
        vec3 specColor = specularColorFactorSampled;

        vec3 F0_dielectric = min(vec3(f0_scalarL) * specColor, vec3(1.0)) * spec;
        vec3 F0 = mix(F0_dielectric, baseColor.rgb, metallic);
        if (iridescenceFactor > 0.0) {
            vec3 iridF0 = EvalIridescence(1.0, mat.iridescenceIor, NdotV, iridescenceThickness, F0);
            F0 = mix(F0, iridF0, iridescenceFactor);
        }
        // Schlick Fresnel at this light's half-vector angle (F90 = 1, per the
        // reference's F_Schlick default) — used both to weight specular's
        // grazing-angle brightening and, via energy conservation, to reduce the
        // diffuse response by the same amount (glTF-Sample-Renderer pbr.frag:
        // l_dielectric_brdf = mix(l_diffuse, l_specular, dielectric_fresnel)).
        vec3 fresnel = F0 + (vec3(1.0) - F0) * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);

        float D, V;
        if (anisoStrength > 0.001) {
            float aspect = sqrt(1.0 - 0.9 * anisoStrength);
            float alphaX = max(0.001, roughness * roughness / aspect);
            float alphaY = max(0.001, roughness * roughness * aspect);
            float cosR = cos(anisoRotation);
            float sinR = sin(anisoRotation);
            vec3 anisoT = cosR * T + sinR * B;
            vec3 anisoB = -sinR * T + cosR * B;
            float HdotT = dot(halfDir, anisoT);
            float HdotB = dot(halfDir, anisoB);
            D = D_GGX_Anisotropic(NdotH, HdotT, HdotB, alphaX, alphaY);
        } else {
            D = D_GGX(roughness, NdotH);
        }
        V = V_SmithGGX(NdotL, NdotV, roughness);

        // BRDF_lambertian(baseColor) = baseColor / PI (glTF-Sample-Renderer brdf.glsl).
        vec3 lDiffuse  = (baseColor.rgb / PI) * NdotL * lightColor;
        vec3 lSpecular = fresnel * D * V * NdotL * lightColor;

        // Metals have no diffuse response at all; dielectrics split energy
        // between diffuse and specular by the Fresnel reflectance instead of
        // adding both unconditionally (which double-counts energy at grazing
        // angles).
        totalDiffuse  += lDiffuse * (vec3(1.0) - fresnel) * (1.0 - metallic);
        totalSpecular += lSpecular;

        float sheenD = D_Charlie(sheenRoughness, NdotH);
        float sheenV = V_Neubelt(NdotV, max(NdotL, 0.0001));
        totalSheen += sheenColor * sheenD * sheenV * NdotL * lightColor;

        float ccNdotH = max(dot(ccN, halfDir), 0.0);
        float ccNdotL = max(dot(ccN, lightDir), 0.0);
        float cc_D    = D_GGX(ccRoughness, ccNdotH);
        float cc_V    = V_SmithGGX(ccNdotL, ccNdotV, ccRoughness);
        float cc_F    = F_Schlick_scalar(0.04, VdotH);
        totalClearcoat += vec3(cc_D * cc_V * cc_F) * ccFactor * ccNdotL * lightColor;
    }

    // --- Final composition ---
    // The flat ambient-color hack is only a stand-in for real indirect
    // lighting when IBL is off; with IBL on, iblDiffuse already supplies the
    // ambient term and this would otherwise double-count it.
    vec3 ambientColor = (mat.iblEnabled > 0.5) ? vec3(0.0) : baseColor.rgb * 0.25 * occlusion;
    vec3 diffuse      = totalDiffuse;
    iblDiffuse  *= occlusion;
    iblSpecular *= occlusion;

    float diffuseScale = 1.0 - transmittance;
    ambientColor *= diffuseScale;
    diffuse      *= diffuseScale;
    iblDiffuse   *= diffuseScale;

    float ccAmbientAtten = 1.0 - ccFactor * F_Schlick_scalar(0.04, ccNdotV);
    vec3 finalColor = (ambientColor + diffuse + totalSpecular + totalSheen + iblDiffuse + iblSpecular
                       + emissive + transmitted * transmittance) * ccAmbientAtten
                      + totalClearcoat + iblClearcoat;

    finalColor = toneMapKhronosPbrNeutral(finalColor);
    finalColor = linearToSRGBFast(finalColor);

    // glTF spec: alphaMode OPAQUE (0) and MASK (1) must render fully opaque
    // -- alpha is ignored (OPAQUE) or only used for the cutoff test already
    // applied above (MASK); only BLEND (2) actually blends by baseColor.a.
    float outAlpha = (mat.alphaMode < 1.5) ? 1.0 : baseColor.a;
    outColor = vec4(finalColor, outAlpha);
}
