#version 450

// G-buffer fragment stage (Renderer::renderGBuffer()), the Vulkan port of
// default.metal's gbufferFragment: world normal + roughness, base colour +
// metallic and motion, with default.frag's flat normals, world mapping,
// clip planes, alpha mask, metallic-roughness and normal mapping.
// The bindings are default.frag's (the PBR pipeline layout).

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
    float doubleSided;               // 88: 1 = double-sided material (no section cap)
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
    float worldMapping;              // 324  world-space box mapping: world units per repeat (0: off)
    float worldMappingUp;            // 328  1: z up, 0: y up
    float _pad14;                    // 332
    vec4  normalUvTransformRow0;     // 336  independent KHR_texture_transform for normalTexture
    vec4  normalUvTransformRow1;     // 352
    float texCoord1Mask;             // 368
    // KHR_texture_transform of these textures, each its own.
    vec4  metallicRoughnessUvTransformRow0; // 384
    vec4  metallicRoughnessUvTransformRow1; // 400
    vec4  occlusionUvTransformRow0;  // 416
    vec4  occlusionUvTransformRow1;  // 432
    vec4  emissiveUvTransformRow0;   // 448
    vec4  emissiveUvTransformRow1;   // 464
} mat;

layout(std140, set = 1, binding = 1) uniform CameraUniforms {
    vec4 cameraPos;    // xyz used, w unused
    mat4 viewMatrix;
    mat4 projMatrix;
    vec2 screenSize;
    vec2 _cpad;
    // Clip block (Renderer::uploadClipUniforms()).
    uvec4 clipInfo;       // x = plane count, y = section caps on
    vec4  capColor;
    vec4  clipPlanes[6];  // xyz = unit normal, w = d: kept where dot(n, p) + d >= 0
    // Sun shadows (Renderer::render()): shadow map clip from world.
    mat4  shadowViewProj;
    vec4  shadowParams;   // x = on, y = texel size (uv)
    vec4  viewParams;     // x = exposure in EV (colour x 2^EV before tone mapping)
    mat4  shadowNearViewProj;  // the near cascade (sun shadows)
    vec4  shadowNearParams;    // x = near cascade on
} camera;

layout(set = 0, binding = 0) uniform texture2D baseColorTexture;
layout(set = 0, binding = 1) uniform sampler   baseColorSampler;
layout(set = 0, binding = 2) uniform texture2D metallicRoughnessTexture;
layout(set = 0, binding = 3) uniform sampler   metallicRoughnessSampler;
layout(set = 0, binding = 4) uniform texture2D normalTexture;
layout(set = 0, binding = 5) uniform sampler   normalSampler;

layout(location = 0) in vec3 worldPos;
layout(location = 1) in vec3 worldNormalRaw;
layout(location = 2) in vec3 worldTangentRaw;
layout(location = 3) in vec3 worldBitangentRaw;
layout(location = 4) in vec2 fragTexcoord0;
layout(location = 5) in vec2 fragTexcoord1;
layout(location = 6) in vec4 fragColor0;
layout(location = 7) in vec4 currentClip;
layout(location = 8) in vec4 previousClip;

layout(location = 0) out vec4 outNormalRoughness;  // xyz world normal (towards the viewer), w perceptual roughness
layout(location = 1) out vec4 outAlbedoMetallic;   // rgb base colour (linear), a metallic
layout(location = 2) out vec2 outMotion;           // previous minus current image uv, y down

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

// A texture's coordinates: its UV set (by its bit in the mask), then its own
// KHR_texture_transform when it has one (row0.w).
vec2 textureUV(vec2 uv0, vec2 uv1, float packedMask, uint bit, vec4 row0, vec4 row1) {
    vec2 uv = selectUV(uv0, uv1, packedMask, bit);
    return row0.w > 0.5 ? applyUvTransform(uv, row0, row1) : uv;
}

void worldMappingFrame(vec3 P, vec3 N, float size, bool zUp, out vec2 uv, out vec3 T, out vec3 B) {
    vec3 U = zUp ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
    float nu = dot(N, U);
    if (abs(nu) > 0.7071) {
        T = vec3(1.0, 0.0, 0.0);
        B = zUp ? vec3(0.0, -1.0, 0.0) : vec3(0.0, 0.0, 1.0);
    } else {
        vec3 h = normalize(N - U * nu);
        T = cross(U, h);
        B = -U;
    }
    uv = vec2(dot(P, T), dot(P, B)) / size;
}

void main() {
    vec2 texcoord0 = fragTexcoord0;
    vec3 faceNormal = cross(dFdx(worldPos), dFdy(worldPos));
    vec3 worldNormal = worldNormalRaw;
    vec3 worldTangent = worldTangentRaw;
    vec3 worldBitangent = worldBitangentRaw;
    if (dot(worldNormalRaw, worldNormalRaw) < 1e-8) {
        vec3 N = length(faceNormal) > 1e-20 ? normalize(faceNormal) : vec3(0.0, 0.0, 1.0);
        if (dot(N, camera.cameraPos.xyz - worldPos) < 0.0) N = -N;
        vec3 T = abs(N.y) < 0.999 ? normalize(cross(vec3(0.0, 1.0, 0.0), N)) : normalize(cross(vec3(1.0, 0.0, 0.0), N));
        worldNormal = N;
        worldTangent = T;
        worldBitangent = cross(N, T);
    }
    if (mat.worldMapping > 0.0) {
        vec3 T, B;
        worldMappingFrame(worldPos, normalize(worldNormal), mat.worldMapping, mat.worldMappingUp > 0.5, texcoord0, T, B);
        worldTangent = T;
        worldBitangent = B;
    }
    for (uint i = 0u; i < camera.clipInfo.x; ++i) {
        if (dot(camera.clipPlanes[i].xyz, worldPos) + camera.clipPlanes[i].w < 0.0) discard;
    }

    vec2 uv0 = texcoord0;
    if (mat.uvTransformRow0.w > 0.5) uv0 = applyUvTransform(texcoord0, mat.uvTransformRow0, mat.uvTransformRow1);
    vec2 baseColorUV = (mat.uvTransformRow0.w > 0.5) ? uv0 : selectUV(uv0, fragTexcoord1, mat.texCoord1Mask, kUV1BaseColor);
    vec4 baseColor = texture(sampler2D(baseColorTexture, baseColorSampler), baseColorUV) * mat.baseColorFactor * fragColor0;
    if (mat.alphaMode > 0.5 && mat.alphaMode < 1.5 && baseColor.a < mat.alphaCutoff) discard;

    float metallic = 0.0;
    float roughness = 1.0;
    vec3 N = normalize(worldNormal);
    if (mat.unlit < 0.5) {
        vec2 mrUV = textureUV(texcoord0, fragTexcoord1, mat.texCoord1Mask, kUV1MetallicRoughness,
                              mat.metallicRoughnessUvTransformRow0, mat.metallicRoughnessUvTransformRow1);
        vec4 mr = texture(sampler2D(metallicRoughnessTexture, metallicRoughnessSampler), mrUV);
        metallic = mr.b * mat.metallicFactor;
        roughness = clamp(mr.g * mat.roughnessFactor, 0.04, 1.0);
        if (mat.hasNormalTexture > 0.5) {
            vec2 normalUV = (mat.normalUvTransformRow0.w > 0.5)
                ? applyUvTransform(texcoord0, mat.normalUvTransformRow0, mat.normalUvTransformRow1)
                : selectUV(texcoord0, fragTexcoord1, mat.texCoord1Mask, kUV1Normal);
            vec3 ns = texture(sampler2D(normalTexture, normalSampler), normalUV).rgb * 2.0 - 1.0;
            ns.xy *= mat.normalScale;
            N = normalize(mat3(normalize(worldTangent), normalize(worldBitangent), N) * normalize(ns));
        }
    }
    // Towards the viewer (back faces of two-sided surfaces).
    if (dot(N, camera.cameraPos.xyz - worldPos) < 0.0 && !gl_FrontFacing) N = -N;

    outNormalRoughness = vec4(N, roughness);
    outAlbedoMetallic = vec4(clamp(baseColor.rgb, 0.0, 1.0), metallic);
    // The clips are before the Y flip (y up), as on Metal.
    vec2 current = currentClip.xy / currentClip.w;
    vec2 previous = previousClip.xy / previousClip.w;
    outMotion = (previous - current) * vec2(0.5, -0.5);
}
