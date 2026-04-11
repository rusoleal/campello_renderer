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
// Layout (112 bytes data, 256-byte stride for Metal alignment):
//   [0..15]   baseColorFactor   — RGBA multiplier for base color
//   [16..31]  uvTransformRow0   — row 0 of KHR_texture_transform [a, b, tx, hasTransform]
//   [32..47]  uvTransformRow1   — row 1 of KHR_texture_transform [c, d, ty, 0]
//   [48..51]  metallicFactor    — scalar metallic multiplier (default 1.0)
//   [52..55]  roughnessFactor   — scalar roughness multiplier (default 1.0)
//   [56..59]  normalScale       — scalar normal map intensity (default 1.0)
//   [60..63]  alphaMode         — 0=opaque, 1=mask, 2=blend
//   [64..67]  alphaCutoff       — alpha test threshold for mask mode
//   [68..71]  unlit             — 0=lit, 1=unlit
//   [72..75]  hasNormalTexture  — 0=no normal map, 1=has normal map
//   [76..79]  hasEmissiveTexture — 0=no emissive map, 1=has emissive map
//   [80..83]  hasOcclusionTexture — 0=no occlusion map, 1=has occlusion map
//   [84..87]  occlusionStrength — scalar occlusion strength (default 1.0)
//   [88..91]  _padding          — padding to align emissiveFactor to 16 bytes
//   [92..103] emissiveFactor    — RGB emissive factor (default 0,0,0, aligned to 16 bytes)
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
    float3 emissiveFactor;
};

struct VertexIn {
    float3 position  [[attribute(0)]];
    float3 normal    [[attribute(1)]];
    float2 texcoord0 [[attribute(2)]];
    float4 tangent   [[attribute(3)]];
};

struct VertexOut {
    float4 clipPosition [[position]];
    float3 worldPos;
    float3 worldNormal;
    float3 worldTangent;
    float3 worldBitangent;
    float2 texcoord0;
    float4 baseColor;
    float  metallic;
    float  roughness;
    float  normalScale;
    float  alphaMode;
    float  alphaCutoff;
    float  unlit;
    float  hasNormalTexture;
    float  hasEmissiveTexture;
    float  hasOcclusionTexture;
    float  occlusionStrength;
    float3 emissiveFactor;
    float3 cameraPosWorld;  // Camera position in world space for lighting
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
    device const float3      *cameraPos [[buffer(18)]])
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

    // Read matrices from structured buffer
    // All meshes in this draw use node 0 (DamagedHelmet has 1 node)
    float4x4 mvp   = nodeTransforms[0].mvp;
    float4x4 model = nodeTransforms[0].model;

    // Transform position to world space for lighting.
    float4 worldPos4 = model * float4(in.position, 1.0);
    
    // Transform normal/tangent/bitangent to world space
    float3 N = normalize((model * float4(in.normal, 0.0)).xyz);
    
    // For Tangent: if input tangent is zero (no tangent attribute), 
    // construct a reasonable tangent from normal
    float3 T;
    if (length(in.tangent.xyz) < 0.001) {
        // Construct tangent perpendicular to normal
        // Choose axis least aligned with normal
        if (abs(N.y) < 0.999) {
            T = normalize(cross(float3(0,1,0), N));
        } else {
            T = normalize(cross(float3(1,0,0), N));
        }
    } else {
        T = normalize((model * float4(in.tangent.xyz, 0.0)).xyz);
        // Re-orthogonalize
        T = normalize(T - dot(T, N) * N);
    }
    
    float3 B = cross(N, T); // Bitangent from N and T (no tangent.w sign for now)

    // Read camera position from buffer. This is needed for specular lighting
    // and must be updated each frame when the camera moves.
    float3 camPos = *cameraPos;

    VertexOut out;
    out.clipPosition    = mvp * float4(in.position, 1.0);
    out.worldPos        = worldPos4.xyz;
    out.worldNormal     = N;
    out.worldTangent    = T;
    out.worldBitangent  = B;
    out.texcoord0       = transformedUV;
    out.baseColor       = mat.baseColorFactor;
    out.metallic        = mat.metallicFactor;
    out.roughness       = mat.roughnessFactor;
    out.normalScale     = mat.normalScale;
    out.alphaMode       = mat.alphaMode;
    out.alphaCutoff     = mat.alphaCutoff;
    out.unlit           = mat.unlit;
    out.hasNormalTexture = mat.hasNormalTexture;
    out.hasEmissiveTexture = mat.hasEmissiveTexture;
    out.hasOcclusionTexture = mat.hasOcclusionTexture;
    out.occlusionStrength = mat.occlusionStrength;
    out.emissiveFactor  = mat.emissiveFactor;
    out.cameraPosWorld  = camPos;
    
    return out;
}

// ---------------------------------------------------------------------------
// Fragment variant: flat — Lambert shading with baseColorFactor only.
// Used for primitives that have no textures.
// ---------------------------------------------------------------------------
fragment float4 fragmentMain_flat(
    VertexOut in [[stage_in]],
    device const float3 *cameraPos [[buffer(18)]])
{
    if (in.unlit > 0.5) {
        return in.baseColor;
    }

    // Hardcoded light direction (from front-top-right).
    float3 lightDir = normalize(float3(0.5, 1.0, 0.5));
    float3 N        = normalize(in.worldNormal);

    float NdotL = max(dot(N, lightDir), 0.0);
    float ambient = 0.25;
    float3 diffuse = in.baseColor.rgb * NdotL * 0.8;
    float3 ambientColor = in.baseColor.rgb * ambient;

    return float4(ambientColor + diffuse, in.baseColor.a);
}

// ---------------------------------------------------------------------------
// Fragment variant: debug — normal visualization.
// Renders world-space normals as RGB colors.
// This helps verify that normals are correctly transformed.
// ---------------------------------------------------------------------------
fragment float4 fragmentMain_debug(
    VertexOut in [[stage_in]],
    device const float3 *cameraPos [[buffer(18)]])
{
    // Visualize world-space normal.
    // X (right) = Red
    // Y (up)    = Green  
    // Z (front) = Blue
    float3 N = normalize(in.worldNormal);
    float3 color = N * 0.5 + 0.5;
    
    // Add a hint for unlit materials.
    if (in.unlit > 0.5) {
        color = mix(color, float3(1.0, 0.8, 0.4), 0.3);
    }
    
    return float4(color, 1.0);
}

// ---------------------------------------------------------------------------
// Fragment variant: textured — PBR metallic-roughness with normal mapping.
// ---------------------------------------------------------------------------
fragment float4 fragmentMain_textured(
    VertexOut        in                [[stage_in]],
    texture2d<float> baseColorTexture  [[texture(0)]],
    sampler          baseColorSampler  [[sampler(1)]],
    texture2d<float> metallicRoughnessTexture [[texture(2)]],
    sampler          metallicRoughnessSampler [[sampler(3)]],
    texture2d<float> normalTexture     [[texture(4)]],
    sampler          normalSampler     [[sampler(5)]],
    texture2d<float> emissiveTexture   [[texture(6)]],
    sampler          emissiveSampler   [[sampler(7)]],
    texture2d<float> occlusionTexture  [[texture(8)]],
    sampler          occlusionSampler  [[sampler(9)]])
{
    // Sample base color texture.
    float4 baseColor = baseColorTexture.sample(baseColorSampler, in.texcoord0) * in.baseColor;
    
    // Alpha mask: discard fragment when alphaMode == 1 (mask) and alpha < alphaCutoff.
    if (in.alphaMode > 0.5 && in.alphaMode < 1.5 && baseColor.a < in.alphaCutoff) {
        discard_fragment();
    }
    
    // Unlit: return base color without lighting.
    if (in.unlit > 0.5) {
        return baseColor;
    }
    
    // Sample metallic-roughness texture.
    // GLTF: G channel = roughness, B channel = metallic.
    float4 mrSample = metallicRoughnessTexture.sample(metallicRoughnessSampler, in.texcoord0);
    float metallic = mrSample.b * in.metallic;
    float roughness = mrSample.g * in.roughness;
    
    // Clamp roughness for numerical stability.
    roughness = clamp(roughness, 0.04, 1.0);
    
    // Normal mapping: sample and decode normal texture.
    float3 N;
    if (in.hasNormalTexture > 0.5) {
        float3 normalSample = normalTexture.sample(normalSampler, in.texcoord0).rgb;
        // Decode from [0,1] to [-1,1].
        normalSample = normalSample * 2.0 - 1.0;
        // Apply normal scale.
        normalSample.xy *= in.normalScale;
        // Re-normalize.
        normalSample = normalize(normalSample);
        
        // Construct TBN matrix and transform to world space.
        float3 T = normalize(in.worldTangent);
        float3 B = normalize(in.worldBitangent);
        float3 Nbase = normalize(in.worldNormal);
        
        // TBN matrix columns.
        float3x3 TBN = float3x3(T, B, Nbase);
        N = TBN * normalSample;
    } else {
        N = normalize(in.worldNormal);
    }
    
    // Sample occlusion texture (R channel)
    float occlusion = 1.0;
    if (in.hasOcclusionTexture > 0.5) {
        occlusion = occlusionTexture.sample(occlusionSampler, in.texcoord0).r;
        occlusion = mix(1.0, occlusion, in.occlusionStrength);
    }

    // Sample emissive texture
    float3 emissive = in.emissiveFactor;
    if (in.hasEmissiveTexture > 0.5) {
        float3 emissiveSample = emissiveTexture.sample(emissiveSampler, in.texcoord0).rgb;
        emissive *= emissiveSample;
    }
    
    // Hardcoded light direction (from front-top-right, looking toward origin).
    float3 lightDir = normalize(float3(0.5, 1.0, 0.5));
    
    // View direction from fragment to camera.
    float3 viewDir = normalize(in.cameraPosWorld - in.worldPos);
    float3 halfDir = normalize(lightDir + viewDir);
    
    // Diffuse (Lambert).
    float NdotL = max(dot(N, lightDir), 0.0);
    
    // Final PBR shading with occlusion
    float3 ambientColor = baseColor.rgb * 0.25 * occlusion;
    float3 diffuse = baseColor.rgb * NdotL * (1.0 - metallic) * 0.8 * occlusion;
    
    // Specular
    float NdotH = max(dot(N, halfDir), 0.0);
    float specPower = max(1.0 / (roughness * roughness) - 1.0, 4.0);
    float specular = pow(NdotH, specPower) * metallic;
    float3 F0 = mix(float3(0.04), baseColor.rgb, metallic);
    float3 specColor = F0 * specular;
    
    return float4(ambientColor + diffuse + specColor + emissive, baseColor.a);
}
