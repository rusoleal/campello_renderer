// campello_renderer default DirectX 12 shader (HLSL shader model 6.0)
//
// Full glTF PBR pipeline — transliterated directly from shaders/vulkan/default.vert
// and shaders/vulkan/default.frag (itself transliterated from shaders/metal/default.metal,
// the single source of truth for this math). Implements: metallic-roughness Cook-Torrance
// GGX BRDF, image-based lighting (diffuse irradiance + roughness-blurred specular
// reflection from a prefiltered environment cubemap), normal mapping, punctual lights
// (KHR_lights_punctual), and KHR_materials_specular/anisotropy/iridescence/clearcoat/
// sheen/transmission. See that file's own comments for the derivation of anything that
// isn't a mechanical GLSL->HLSL syntax change; this file intentionally mirrors its
// section order and math line-for-line so the two stay easy to diff against each other.
//
// --- Vertex slot contract (must match Renderer::VERTEX_SLOT_* constants) -------------
//   slot  0  POSITION      float3  — object-space vertex position
//   slot  1  NORMAL        float3  — object-space vertex normal
//   slot  2  TEXCOORD_0    float2  — primary UV
//   slot  3  TANGENT       float4  — tangent (bitangent sign in .w is unused, matching
//                                    Metal/Vulkan's plain cross(N,T) — see vertexMain())
//   slot  6  COLOR_0       float4  — vertex color, (1,1,1,1) fallback
//   slot  7  TEXCOORD_1    float2  — secondary UV, (0,0) fallback
//   slot 16  NodeTransforms.mvp    float4x4, per-instance, 4 columns at locations 16-19
//   slot 16  NodeTransforms.model  float4x4, same buffer at byte offset 64, locations 24-27
// campello_gpu's DirectX backend maps every vertex attribute to semantic TEXCOORD with
// SemanticIndex = shaderLocation, so every struct below uses TEXCOORD semantics
// throughout (not the standard POSITION/NORMAL/TEXCOORD/TANGENT names).
//
// --- Matrix convention -----------------------------------------------------------------
// Renderer::computeNodeTransform()/uploadOneTransform() upload MVP/MODEL as 4 separate
// float4 *columns* (the same column-major buffer layout every backend, Metal included,
// shares — see default.vert's own comment) split one-per-vertex-attribute-location. Rather
// than reconstruct a float4x4 from those 4 column vectors (ambiguous: HLSL's float4x4(v0,v1,v2,v3)
// constructor fills ROWS, not columns, so a naive reconstruction would silently transpose
// the matrix), M*v is computed directly as the linear combination v.x*col0 + v.y*col1 +
// v.z*col2 + v.w*col3 — mathematically identical to GLSL's `mat4 * vec4`, and correct by
// inspection with no row/column-major ambiguity to get wrong.
//
// CameraUniforms.viewMatrix/projMatrix below, by contrast, arrive as a real cbuffer (not
// per-vertex columns), so they use HLSL's default (column_major) cbuffer packing — which
// reads the same column-major buffer bytes GLSL's std140 mat4 does — combined with
// mul(M, v), which for a correctly-loaded column-major M reproduces GLSL's `M * v` exactly.
//
// --- NDC/rasterizer convention -----------------------------------------------------------
// Unlike shaders/vulkan/default.vert, vertexMain() below does NOT negate clip-space Y.
// D3D12's rasterizer, like Metal's, maps clip-space +Y to the top of the viewport natively;
// only Vulkan's rasterizer inverts that, which is why only the Vulkan vertex shader needs a
// compensating flip (see its own comment for the full explanation). Screen-space (SV_Position)
// reconstruction in the skybox/transmission code below, by contrast, uses the *same*
// `1.0 - y/screenSize.y` flip Vulkan and Metal both use — that one is unrelated to the
// rasterizer convention (it converts y-down pixel coordinates, identical on all three APIs,
// into the "logical" Y-up NDC the CPU-side matrices are expressed in) — see
// shaders/vulkan/skybox.frag's header comment.
//
// --- Resource binding scheme --------------------------------------------------------------
// The PBR pipeline's bind groups are Vulkan's (ensureVulkanPbrBindGroupLayouts() in
// campello_renderer.cpp): group 0 = material (built once per material), group 1 = frame
// (lights, camera, environment, scene colour, shadows), group 2 = joint palettes, group 3 =
// morph targets. campello_gpu's D3D12 backend puts group N in register space N, so each
// group's registers below are its Vulkan binding numbers in spaceN (t/s/b by resource type:
// D3D12 gives textures, samplers and buffers separate register namespaces).
//
// Compile on Windows with DXC — see build_directx_shaders.ps1 for the full per-entry-point
// invocation list and src/shaders/directx_default.h regeneration.

// ---------------------------------------------------------------------------
// Shared constants / helpers
// ---------------------------------------------------------------------------
static const float PI = 3.14159265359;

// Khronos PBR Neutral tone mapping — the glTF-Sample-Renderer's actual current default (see
// GltfState.ToneMaps.KHR_PBR_NEUTRAL). Ported verbatim from the reference's tonemapping.glsl,
// shared by both pixelMain() and skyboxPixel() below (unlike the Vulkan/Metal sources, which
// each keep their own copy since GLSL has no #include across those separate files in this
// build — this one file can just share the definition).
float3 toneMapKhronosPbrNeutral(float3 color) {
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
    return lerp(color, float3(newPeak, newPeak, newPeak), g);
}

// The swapchain target is a plain (non-sRGB-tagged) unorm format, so linear-to-sRGB encoding
// must happen here rather than relying on the hardware to do it on write.
float3 linearToSRGBFast(float3 color) {
    return pow(color, float3(1.0 / 2.2, 1.0 / 2.2, 1.0 / 2.2));
}

// ---------------------------------------------------------------------------
// Main PBR pipeline — resources (material and frame bind groups; see header comment above)
// ---------------------------------------------------------------------------

// --- Material (buffer 24) -------------------------------------------------------------
// Byte-exact mirror of MaterialUniforms in default.metal/default.frag (and the
// authoritative C++ writer, Renderer::buildSlotRaw() in campello_renderer.cpp) — every
// scalar gap below is a deliberate, explicitly-sized padding run reproducing Metal's own
// manual alignment. Do not remove the _padN fields or reorder members: HLSL's default
// cbuffer packing (pack into 4-float registers, never split a float4 across one) follows
// the same algorithm as GLSL's std140, so matching field order/type here reproduces the
// exact same byte offsets as shaders/vulkan/default.frag's copy — both platforms read the
// same uploaded buffer.
cbuffer MaterialUniforms : register(b24) {
    float4 baseColorFactor;           // 0
    float4 uvTransformRow0;           // 16   [a, b, tx, hasTransform]
    float4 uvTransformRow1;           // 32   [c, d, ty, 0]
    float metallicFactor;             // 48
    float roughnessFactor;            // 52
    float normalScale;                // 56
    float alphaMode;                  // 60   0=opaque, 1=mask, 2=blend
    float alphaCutoff;                // 64
    float unlit;                      // 68
    float hasNormalTexture;           // 72
    float hasEmissiveTexture;         // 76
    float hasOcclusionTexture;        // 80
    float occlusionStrength;          // 84
    float doubleSided;                // 88: 1 = double-sided material (no section cap)
    float _pad1;                      // 92
    float4 emissiveFactor;            // 96
    float ior;                        // 112
    float specularFactor;             // 116
    float hasSpecularTexture;         // 120
    float hasSpecularColorTexture;    // 124
    float _pad2;                      // 128
    float _pad3;                      // 132
    float _pad4;                      // 136
    float _pad5;                      // 140
    float4 specularColorFactor;       // 144
    float _pad6;                      // 160
    float _pad7;                      // 164
    float _pad8;                      // 168
    float _pad9;                      // 172
    float4 sheenColorFactor;          // 176
    float sheenRoughnessFactor;       // 192
    float hasSheenColorTexture;       // 196
    float hasSheenRoughnessTexture;   // 200
    float clearcoatFactor;            // 204
    float clearcoatRoughnessFactor;   // 208
    float hasClearcoatTexture;        // 212
    float hasClearcoatRoughnessTexture; // 216
    float hasClearcoatNormalTexture;  // 220
    float clearcoatNormalScale;       // 224
    float transmissionFactor;         // 228
    float hasTransmissionTexture;     // 232
    float thicknessFactor;            // 236
    float attenuationDistance;        // 240
    float hasThicknessTexture;        // 244
    float _pad10;                     // 248
    float _pad11;                     // 252
    float4 attenuationColor;          // 256
    float viewMode;                   // 272
    float environmentIntensity;       // 276
    float iblEnabled;                 // 280
    float iridescenceFactor;          // 284
    float iridescenceIor;             // 288
    float iridescenceThicknessMin;    // 292
    float iridescenceThicknessMax;    // 296
    float hasIridescenceTexture;      // 300
    float hasIridescenceThicknessTexture; // 304
    float anisotropyStrength;         // 308
    float anisotropyRotation;         // 312
    float hasAnisotropicTexture;      // 316
    float dispersion;                 // 320
    float worldMapping;               // 324  world-space box mapping: world units per repeat (0: off; G-buffer only here)
    float worldMappingUp;             // 328  1: z up, 0: y up
    float _pad14;                     // 332
    float4 normalUvTransformRow0;     // 336  independent KHR_texture_transform for normalTexture
    float4 normalUvTransformRow1;     // 352
    float texCoord1Mask;              // 368
};

// --- Per-frame (bind group 1, space1: buffers 0/1) ---------------------------------------
struct Light {
    float4 position;   // xyz = position/dir, w = type (0=dir, 1=point, 2=spot)
    float4 color;      // xyz = rgb, w = intensity
    float4 direction;  // xyz = spot dir, w = range
    float4 spotAngles; // x = innerCone, y = outerCone, z = casts the shadow map
};
cbuffer LightsUniform : register(b0, space1) {
    uint  lightCount;
    uint  _lpad0;
    uint  _lpad1;
    uint  _lpad2;
    Light lights[4];
};

// Byte-exact mirror of default.frag's CameraUniforms (kCameraUniformSize, 464
// bytes); HLSL cbuffer packing matches std140 for these members.
cbuffer CameraUniforms : register(b1, space1) {
    float4   cameraPos;    // xyz used, w unused
    float4x4 viewMatrix;
    float4x4 projMatrix;
    float2   screenSize;
    float2   _cpad;
    // Clip block (Renderer::uploadClipUniforms()).
    uint4    clipInfo;       // x = plane count, y = section caps on
    float4   capColor;
    float4   clipPlanes[6];  // xyz = unit normal, w = d: kept where dot(n, p) + d >= 0
    // Sun shadows (Renderer::render()): shadow map clip from world.
    float4x4 shadowViewProj;
    float4   shadowParams;   // x = on, y = texel size (uv)
    float4   viewParams;     // x = exposure in EV (colour x 2^EV before tone mapping)
    float4x4 shadowNearViewProj;  // the near cascade (sun shadows)
    float4   shadowNearParams;    // x = near cascade on
};

// --- Per-material textures/samplers -----------------------------------------------------
Texture2D<float4> baseColorTexture          : register(t0);
SamplerState      baseColorSampler          : register(s1);
Texture2D<float4> metallicRoughnessTexture  : register(t2);
SamplerState      metallicRoughnessSampler  : register(s3);
Texture2D<float4> normalTexture             : register(t4);
SamplerState      normalSampler             : register(s5);
Texture2D<float4> emissiveTexture           : register(t6);
SamplerState      emissiveSampler           : register(s7);
Texture2D<float4> occlusionTexture          : register(t8);
SamplerState      occlusionSampler          : register(s9);
Texture2D<float4> specularTexture           : register(t10);
SamplerState      specularSampler           : register(s11);
Texture2D<float4> specularColorTexture      : register(t12);
SamplerState      specularColorSampler      : register(s13);
Texture2D<float4> sheenColorTexture         : register(t14); // reuses baseColorSampler
Texture2D<float4> sheenRoughnessTexture     : register(t15); // reuses baseColorSampler
Texture2D<float4> clearcoatTexture          : register(t16); // reuses baseColorSampler
Texture2D<float4> clearcoatRoughnessTexture : register(t17); // reuses baseColorSampler
Texture2D<float4> clearcoatNormalTexture    : register(t18); // reuses baseColorSampler
Texture2D<float4> transmissionTexture       : register(t19); // reuses baseColorSampler
Texture2D<float4> thicknessTexture          : register(t20); // reuses baseColorSampler
Texture2D<float4> iridescenceTexture        : register(t21); // reuses baseColorSampler
Texture2D<float4> iridescenceThicknessTexture : register(t22); // reuses baseColorSampler
Texture2D<float4> anisotropicTexture        : register(t23); // reuses baseColorSampler

// --- Per-frame textures/samplers (bind group 1, space1: environment, IBL, scene colour, shadows) ---
TextureCube<float4> environmentMap  : register(t2, space1) ; // GGX-prefiltered specular cubemap (bakeIblResources())
SamplerState         environmentSampler : register(s3, space1);
Texture2D<float4>    sceneColorTexture  : register(t4, space1) ; // opaque scene color, for KHR_materials_transmission
SamplerState          sceneColorSampler : register(s5, space1);
TextureCube<float4>  irradianceMap  : register(t6, space1); // Lambertian-convolved diffuse cubemap (bakeIblResources())
Texture2D<float4>    brdfLutTexture : register(t7, space1); // Karis split-sum LUT indexed by (NdotV, roughness)
SamplerState          brdfLutSampler : register(s8, space1);
Texture2D<float>       shadowMap      : register(t9, space1) ; // sun shadows (depth)
SamplerComparisonState shadowSampler  : register(s10, space1); // compare less-or-equal
Texture2D<float>       shadowMapNear  : register(t11, space1); // sun shadows, near cascade

// One shadow map's verdict on P: 1 lit .. 0 shadowed, 3x3 percentage-closer
// filtering; `inside` false where the map doesn't cover P. Receiver bias as
// default.frag / default.metal's shadowLookup(). M arrives column-major
// (cbuffer default), so M[c][r] in GLSL is M[r][c] here.
float shadowLookup(Texture2D<float> map, float4x4 M, float texel, float3 P, float3 N, float3 L, out bool inside) {
    float worldTexel = 2.0 * texel / max(length(float3(M[0][0], M[0][1], M[0][2])), 1e-6);
    float depthPerWorld = length(float3(M[2][0], M[2][1], M[2][2]));
    float NdotL = clamp(dot(N, L), 0.0, 1.0);
    float slope = sqrt(1.0 - NdotL * NdotL) / max(NdotL, 0.2);
    float4 clip = mul(M, float4(P + N * worldTexel * (1.0 + slope), 1.0));
    float3 ndc = clip.xyz / clip.w;
    float2 uv = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    inside = all(uv >= 2.0 * texel) && all(uv <= 1.0 - 2.0 * texel) && ndc.z <= 1.0;
    if (!inside) return 1.0;
    float depth = ndc.z - depthPerWorld * worldTexel;
    float lit = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            lit += map.SampleCmpLevelZero(shadowSampler, uv + float2(x, y) * texel, depth);
        }
    }
    return lit / 9.0;
}

// How lit P is by the shadow-casting light: the near cascade where it covers
// P, else the far one (the whole scene); outside both, lit.
float sunShadow(float3 P, float3 N, float3 L) {
    if (shadowParams.x < 0.5) return 1.0;
    bool inside = false;
    if (shadowNearParams.x > 0.5) {
        float lit = shadowLookup(shadowMapNear, shadowNearViewProj, shadowParams.y, P, N, L, inside);
        if (inside) return lit;
    }
    return shadowLookup(shadowMap, shadowViewProj, shadowParams.y, P, N, L, inside);
}

// ---------------------------------------------------------------------------
// Vertex stage
// ---------------------------------------------------------------------------
struct VertexIn {
    float3 position : TEXCOORD0;   // slot 0
    float3 normal   : TEXCOORD1;   // slot 1
    float2 uv0      : TEXCOORD2;   // slot 2
    float4 tangent  : TEXCOORD3;   // slot 3
    float4 color0   : TEXCOORD6;   // slot 6
    float2 uv1      : TEXCOORD7;   // slot 7

    // NodeTransforms.mvp, per-instance, one column per attribute (slot 16, stride 128).
    float4 mvpCol0   : TEXCOORD16;
    float4 mvpCol1   : TEXCOORD17;
    float4 mvpCol2   : TEXCOORD18;
    float4 mvpCol3   : TEXCOORD19;
    // NodeTransforms.model, same buffer at byte offset 64 (locations 24-27).
    float4 modelCol0 : TEXCOORD24;
    float4 modelCol1 : TEXCOORD25;
    float4 modelCol2 : TEXCOORD26;
    float4 modelCol3 : TEXCOORD27;
    // EXT_mesh_gpu_instancing: this instance's transform, per instance
    // (locations 12-15; the identity for draws without instances).
    float4 instanceCol0 : TEXCOORD12;
    float4 instanceCol1 : TEXCOORD13;
    float4 instanceCol2 : TEXCOORD14;
    float4 instanceCol3 : TEXCOORD15;
};

// ---------------------------------------------------------------------------
// Deformation (shaders/vulkan/skinning.glsl), in glTF's order: morph targets,
// then skeletal skinning, then the instance's transform. Only the skinned
// entry points (vertexSkinned, gbufferSkinnedVertex) read the palette and the
// deltas; the renderer picks them for draws with a skin or morph targets.
//
// Bind group 2 of the pipeline layout (register space 2, campello_gpu numbers
// spaces by the layout's position): the draw's joint palette (column-major
// float4x4 per joint) and the previous frame's (G-buffer motion). Group 3
// (space 3): the node's MorphInfo (target count, whether targets have
// normals, vertex count, up to 8 weights) and the primitive's POSITION and
// NORMAL deltas, target-major, 3 floats per vertex (target t, vertex v at
// t * vertexCount + v). Raw buffers: campello_gpu views read-only storage
// buffers as ByteAddressBuffers.
// ---------------------------------------------------------------------------
ByteAddressBuffer jointMatrices         : register(t0, space2);
ByteAddressBuffer previousJointMatrices : register(t1, space2);
ByteAddressBuffer morphInfo             : register(t0, space3);
ByteAddressBuffer morphPositions        : register(t1, space3);
ByteAddressBuffer morphNormals          : register(t2, space3);

// A float4x4 as its four columns (M*v as a column combination; see the
// header comment on HLSL's row-filling constructor).
struct Columns {
    float4 c0, c1, c2, c3;
};

Columns columns(float4 c0, float4 c1, float4 c2, float4 c3) {
    Columns m;
    m.c0 = c0; m.c1 = c1; m.c2 = c2; m.c3 = c3;
    return m;
}

float4 transform(Columns m, float4 v) {
    return m.c0 * v.x + m.c1 * v.y + m.c2 * v.z + m.c3 * v.w;
}

Columns compose(Columns a, Columns b) {  // a * b
    return columns(transform(a, b.c0), transform(a, b.c1), transform(a, b.c2), transform(a, b.c3));
}

Columns identityColumns() {
    return columns(float4(1, 0, 0, 0), float4(0, 1, 0, 0), float4(0, 0, 1, 0), float4(0, 0, 0, 1));
}

Columns loadJoint(ByteAddressBuffer palette, uint joint) {
    uint base = joint * 64;
    return columns(asfloat(palette.Load4(base)), asfloat(palette.Load4(base + 16)),
                   asfloat(palette.Load4(base + 32)), asfloat(palette.Load4(base + 48)));
}

// Up to four joints blended by weight; weights summing to zero leave the
// vertex unskinned (as Metal).
Columns blendJoints(ByteAddressBuffer palette, uint4 j, float4 w) {
    if (w.x + w.y + w.z + w.w <= 0.001) return identityColumns();
    Columns m = columns(float4(0, 0, 0, 0), float4(0, 0, 0, 0), float4(0, 0, 0, 0), float4(0, 0, 0, 0));
    [unroll] for (int i = 0; i < 4; ++i) {
        if (w[i] > 0.0) {
            Columns joint = loadJoint(palette, j[i]);
            m.c0 += w[i] * joint.c0;
            m.c1 += w[i] * joint.c1;
            m.c2 += w[i] * joint.c2;
            m.c3 += w[i] * joint.c3;
        }
    }
    return m;
}

void applyMorph(inout float3 p, inout float3 n, uint vertexId) {
    uint targets = min(uint(asfloat(morphInfo.Load(0))), 8u);
    bool hasNormal = asfloat(morphInfo.Load(4)) > 0.5;
    uint vertexCount = uint(asfloat(morphInfo.Load(8)));
    for (uint t = 0u; t < targets; ++t) {
        float w = asfloat(morphInfo.Load(12 + t * 4));
        if (w == 0.0) continue;
        uint offset = (t * vertexCount + vertexId) * 12;
        p += w * asfloat(morphPositions.Load3(offset));
        if (hasNormal) n += w * asfloat(morphNormals.Load3(offset));
    }
}

struct PixelIn {
    float4 clipPosition  : SV_Position;
    float3 worldPos       : TEXCOORD0;
    float3 worldNormal    : TEXCOORD1;
    float3 worldTangent   : TEXCOORD2;
    float3 worldBitangent : TEXCOORD3;
    float2 uv0             : TEXCOORD4;
    float2 uv1             : TEXCOORD5;
    float4 color0           : TEXCOORD6;
};

// The PBR vertex stage. `skin` is the instance's transform times the joints'
// blend (the identity for both when unused); position and normal are morphed.
PixelIn pbrVertex(VertexIn input, Columns skin, float3 position, float3 normal) {
    PixelIn output;
    Columns mvp = columns(input.mvpCol0, input.mvpCol1, input.mvpCol2, input.mvpCol3);
    Columns model = columns(input.modelCol0, input.modelCol1, input.modelCol2, input.modelCol3);

    float4 localPos = transform(skin, float4(position, 1.0));
    output.clipPosition = transform(mvp, localPos);
    output.worldPos = transform(model, localPos).xyz;

    // Normal/tangent use w=0 (direction, not position), exactly like GLSL's
    // `(model * vec4(v, 0.0)).xyz`.
    float3 N = normalize(transform(model, transform(skin, float4(normal, 0.0))).xyz);

    // Tangent frame: synthesize an arbitrary tangent orthogonal to N when the mesh has no
    // real TANGENT attribute (near-zero length), otherwise transform the vertex tangent by
    // the model matrix and re-orthogonalize against N (Gram-Schmidt) — mirrors Metal/
    // Vulkan's vertexMain exactly, including not using tangent.w to flip the bitangent sign.
    float3 T;
    if (length(input.tangent.xyz) < 0.001) {
        if (abs(N.y) < 0.999) {
            T = normalize(cross(float3(0.0, 1.0, 0.0), N));
        } else {
            T = normalize(cross(float3(1.0, 0.0, 0.0), N));
        }
    } else {
        T = normalize(transform(model, transform(skin, float4(input.tangent.xyz, 0.0))).xyz);
        T = normalize(T - dot(T, N) * N);
    }
    float3 B = cross(N, T);

    output.worldNormal    = N;
    output.worldTangent   = T;
    output.worldBitangent = B;
    output.uv0    = input.uv0;
    output.uv1    = input.uv1;
    output.color0 = input.color0;

    return output;
}

Columns instanceColumns(VertexIn input) {
    return columns(input.instanceCol0, input.instanceCol1, input.instanceCol2, input.instanceCol3);
}

PixelIn vertexMain(VertexIn input) {
    return pbrVertex(input, instanceColumns(input), input.position, input.normal);
}

// Skinned and morphed draws: JOINTS_0 / WEIGHTS_0 at locations 4 and 5.
PixelIn vertexSkinned(VertexIn input, uint4 joints : TEXCOORD4, float4 weights : TEXCOORD5,
                      uint vertexId : SV_VertexID) {
    float3 position = input.position;
    float3 normal = input.normal;
    applyMorph(position, normal, vertexId);
    Columns skin = compose(instanceColumns(input), blendJoints(jointMatrices, joints, weights));
    return pbrVertex(input, skin, position, normal);
}

// ---------------------------------------------------------------------------
// Fragment (pixel) stage — BRDF helper functions (verbatim transliteration of
// shaders/vulkan/default.frag, itself a transliteration of default.metal)
// ---------------------------------------------------------------------------
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

static const float3x3 kXyzToRec709 = float3x3(
    3.2404542, -1.5371385, -0.4985314,
   -0.9692660,  1.8760108,  0.0415560,
    0.0556434, -0.2040259,  1.0572252
);

float3 Fresnel0ToIor(float3 fresnel0) {
    float3 sqrtF0 = sqrt(clamp(fresnel0, 0.0, 0.9999));
    return (float3(1.0, 1.0, 1.0) + sqrtF0) / (float3(1.0, 1.0, 1.0) - sqrtF0);
}

float FresnelDielectric(float cosTheta1, float n1, float n2) {
    float sinTheta2Sq = (n1 * n1) / (n2 * n2) * (1.0 - cosTheta1 * cosTheta1);
    if (sinTheta2Sq > 1.0) return 1.0;
    float cosTheta2 = sqrt(1.0 - sinTheta2Sq);
    float r_s = (n1 * cosTheta1 - n2 * cosTheta2) / (n1 * cosTheta1 + n2 * cosTheta2);
    float r_p = (n2 * cosTheta1 - n1 * cosTheta2) / (n2 * cosTheta1 + n1 * cosTheta2);
    return 0.5 * (r_s * r_s + r_p * r_p);
}

float3 FresnelDielectric3(float cosTheta1, float n1, float3 n2) {
    float3 sinTheta2Sq = (float3(n1 * n1, n1 * n1, n1 * n1) / (n2 * n2)) * (1.0 - cosTheta1 * cosTheta1);
    float3 cosTheta2 = sqrt(clamp(float3(1.0, 1.0, 1.0) - sinTheta2Sq, float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0)));
    float3 r_s = (n1 * cosTheta1 - n2 * cosTheta2) / (n1 * cosTheta1 + n2 * cosTheta2);
    float3 r_p = (n2 * cosTheta1 - n1 * cosTheta2) / (n2 * cosTheta1 + n1 * cosTheta2);
    return 0.5 * (r_s * r_s + r_p * r_p);
}

float3 EvalSensitivity(float opd, float3 shift) {
    float phase = 2.0 * PI * opd * 1.0e-9;
    float3 val = float3(5.4856e-13, 4.4201e-13, 5.2481e-13);
    float3 pos = float3(1.6810e+06, 1.7953e+06, 2.2084e+06);
    float3 var = float3(4.3278e+09, 9.3046e+09, 6.6121e+09);

    float3 xyz = val * sqrt(2.0 * PI * var) * cos(pos * phase + shift) * exp(-phase * phase * var);
    xyz.x += 9.7470e-14 * sqrt(2.0 * PI * 4.5282e+09) * cos(2.2399e+06 * phase + shift.x) * exp(-4.5282e+09 * phase * phase);
    xyz /= 1.0685e-7;

    return mul(kXyzToRec709, xyz);
}

float3 EvalIridescence(float outsideIOR, float eta2, float cosTheta1, float thinFilmThickness, float3 baseF0) {
    float iridescenceIor = lerp(outsideIOR, eta2, smoothstep(0.0, 0.03, thinFilmThickness));
    float sinTheta2Sq = (outsideIOR * outsideIOR) / (iridescenceIor * iridescenceIor)
                        * (1.0 - cosTheta1 * cosTheta1);
    float cosTheta2Sq = 1.0 - sinTheta2Sq;
    if (cosTheta2Sq < 0.0) {
        return float3(1.0, 1.0, 1.0);
    }
    float cosTheta2 = sqrt(cosTheta2Sq);

    float R12   = FresnelDielectric(cosTheta1, outsideIOR, iridescenceIor);
    float T121  = 1.0 - R12;
    float phi12 = (iridescenceIor < outsideIOR) ? PI : 0.0;
    float phi21 = PI - phi12;

    float3 baseIOR = Fresnel0ToIor(baseF0);
    float3 R23   = FresnelDielectric3(cosTheta2, iridescenceIor, baseIOR);
    float3 phi23 = float3(baseIOR.x < iridescenceIor ? PI : 0.0,
                           baseIOR.y < iridescenceIor ? PI : 0.0,
                           baseIOR.z < iridescenceIor ? PI : 0.0);

    float opd = 2.0 * iridescenceIor * thinFilmThickness * cosTheta2;
    float3  phi = float3(phi21, phi21, phi21) + phi23;

    float3 R123 = clamp(float3(R12, R12, R12) * R23, float3(1e-5, 1e-5, 1e-5), float3(0.9999, 0.9999, 0.9999));
    float3 r123 = sqrt(R123);
    float3 Rs   = (T121 * T121) * R23 / (float3(1.0, 1.0, 1.0) - R123);

    float3 C0 = float3(R12, R12, R12) + Rs;
    float3 I  = C0;

    float3 Cm = Rs - float3(T121, T121, T121);
    for (int m = 1; m <= 2; ++m) {
        Cm *= r123;
        float3 Sm = 2.0 * EvalSensitivity(float(m) * opd, float(m) * phi);
        I += Cm * Sm;
    }

    return max(I, float3(0.0, 0.0, 0.0));
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

// Roughness-aware IBL Fresnel with single-scattering (FssEss) + multi-scattering (FmsEms)
// energy compensation — glTF-Sample-Renderer's getIBLGGXFresnel (ibl.glsl), from
// Fdez-Aguera's "A Multiple-Scattering Microfacet Model for Real-Time Image-based Lighting".
float3 iblGGXFresnel(float NdotV, float roughness, float3 F0) {
    float2 fAB = brdfLutTexture.Sample(brdfLutSampler, clamp(float2(NdotV, roughness), 0.0, 1.0)).rg;
    float3 Fr = max(float3(1.0 - roughness, 1.0 - roughness, 1.0 - roughness), F0) - F0;
    float3 kS = F0 + Fr * pow(clamp(1.0 - NdotV, 0.0, 1.0), 5.0);
    float3 FssEss = kS * fAB.x + fAB.y;

    float Ems = 1.0 - (fAB.x + fAB.y);
    float3 Favg = F0 + (float3(1.0, 1.0, 1.0) - F0) / 21.0;
    float3 FmsEms = Ems * FssEss * Favg / (float3(1.0, 1.0, 1.0) - Favg * Ems);

    return FssEss + FmsEms;
}

// --- KHR_texture_transform / TEXCOORD_1 selection -------------------------------
static const uint kUV1BaseColor            = 1u << 0;
static const uint kUV1MetallicRoughness    = 1u << 1;
static const uint kUV1Normal               = 1u << 2;
static const uint kUV1Emissive             = 1u << 3;
static const uint kUV1Occlusion            = 1u << 4;
static const uint kUV1Specular             = 1u << 5;
static const uint kUV1SpecularColor        = 1u << 6;
static const uint kUV1SheenColor           = 1u << 7;
static const uint kUV1SheenRoughness       = 1u << 8;
static const uint kUV1Clearcoat            = 1u << 9;
static const uint kUV1ClearcoatRoughness   = 1u << 10;
static const uint kUV1ClearcoatNormal      = 1u << 11;
static const uint kUV1Transmission         = 1u << 12;
static const uint kUV1Thickness            = 1u << 13;
static const uint kUV1Iridescence          = 1u << 14;
static const uint kUV1IridescenceThickness = 1u << 15;
static const uint kUV1Anisotropic          = 1u << 16;

float2 selectUV(float2 uv0, float2 uv1, float packedMask, uint bit) {
    return (uint(packedMask) & bit) != 0u ? uv1 : uv0;
}

float2 applyUvTransform(float2 uv, float4 row0, float4 row1) {
    float3 uv3 = float3(uv, 1.0);
    return float2(dot(row0.xyz, uv3), dot(row1.xyz, uv3));
}

// ---------------------------------------------------------------------------
// Fragment (pixel) stage — main
// ---------------------------------------------------------------------------
// World-space box mapping (MaterialUniforms::worldMapping, default.frag):
// texture coordinates from the world position on the plane facing the
// normal's dominant direction, with that plane's tangent frame.
void worldMappingFrame(float3 P, float3 N, float size, bool zUp, out float2 uv, out float3 T, out float3 B) {
    float3 U = zUp ? float3(0.0, 0.0, 1.0) : float3(0.0, 1.0, 0.0);
    float nu = dot(N, U);
    if (abs(nu) > 0.7071) {
        T = float3(1.0, 0.0, 0.0);
        B = zUp ? float3(0.0, -1.0, 0.0) : float3(0.0, 0.0, 1.0);
    } else {
        float3 h = normalize(N - U * nu);
        T = cross(U, h);
        B = -U;
    }
    uv = float2(dot(P, T), dot(P, B)) / size;
}

float4 pixelMain(PixelIn input, bool isFrontFace : SV_IsFrontFace) : SV_Target {
    // Position-only meshes (leanVertex) have a zero normal: the flat face
    // normal from screen-space derivatives of the world position, turned
    // towards the camera (default.frag). Derivatives in uniform control flow.
    float3 faceNormal = cross(ddx(input.worldPos), ddy(input.worldPos));
    if (dot(input.worldNormal, input.worldNormal) < 1e-8) {
        float3 Nf = length(faceNormal) > 1e-20 ? normalize(faceNormal) : float3(0.0, 0.0, 1.0);
        if (dot(Nf, cameraPos.xyz - input.worldPos) < 0.0) Nf = -Nf;
        float3 Tf = abs(Nf.y) < 0.999 ? normalize(cross(float3(0.0, 1.0, 0.0), Nf)) : normalize(cross(float3(1.0, 0.0, 0.0), Nf));
        input.worldNormal    = Nf;
        input.worldTangent   = Tf;
        input.worldBitangent = cross(Nf, Tf);
    }
    if (worldMapping > 0.0) {
        float3 Tw, Bw;
        worldMappingFrame(input.worldPos, normalize(input.worldNormal), worldMapping, worldMappingUp > 0.5, input.uv0, Tw, Bw);
        input.worldTangent   = Tw;
        input.worldBitangent = Bw;
    }
    float2 uv0 = input.uv0;
    if (uvTransformRow0.w > 0.5) {
        uv0 = applyUvTransform(input.uv0, uvTransformRow0, uvTransformRow1);
    }
    float2 baseColorUV = (uvTransformRow0.w > 0.5)
        ? uv0 : selectUV(uv0, input.uv1, texCoord1Mask, kUV1BaseColor);

    float4 baseColor = baseColorTexture.Sample(baseColorSampler, baseColorUV)
                      * baseColorFactor * input.color0;

    float transmission = transmissionFactor;
    if (hasTransmissionTexture > 0.5) {
        float2 transUV = selectUV(uv0, input.uv1, texCoord1Mask, kUV1Transmission);
        transmission *= transmissionTexture.Sample(baseColorSampler, transUV).r;
    }

    // Metallic-roughness.
    float2 mrUV = selectUV(uv0, input.uv1, texCoord1Mask, kUV1MetallicRoughness);
    float4 mrSample = metallicRoughnessTexture.Sample(metallicRoughnessSampler, mrUV);
    float metallic  = mrSample.b * metallicFactor;
    float roughness = clamp(mrSample.g * roughnessFactor, 0.04, 1.0);

    // Tangent frame (needed for normal mapping and anisotropy).
    float3 T = normalize(input.worldTangent);
    float3 B = normalize(input.worldBitangent);
    float3 N;
    if (hasNormalTexture > 0.5) {
        float2 normalUV = (normalUvTransformRow0.w > 0.5)
            ? applyUvTransform(input.uv0, normalUvTransformRow0, normalUvTransformRow1)
            : selectUV(input.uv0, input.uv1, texCoord1Mask, kUV1Normal);
        float3 ns = normalTexture.Sample(normalSampler, normalUV).rgb * 2.0 - 1.0;
        ns.xy *= normalScale;
        float3 Nbase = normalize(input.worldNormal);
        // TBN * ns via explicit column linear-combination (T,B,Nbase as columns) — see
        // header comment for why this avoids HLSL's row-vs-column matrix-constructor trap.
        float3 nsN = normalize(ns);
        N = normalize(nsN.x * T + nsN.y * B + nsN.z * Nbase);
    } else {
        N = normalize(input.worldNormal);
    }

    // Occlusion.
    float occlusion = 1.0;
    if (hasOcclusionTexture > 0.5) {
        float2 occUV = selectUV(uv0, input.uv1, texCoord1Mask, kUV1Occlusion);
        float occ = occlusionTexture.Sample(occlusionSampler, occUV).r;
        occlusion = lerp(1.0, occ, occlusionStrength);
    }

    // Emissive.
    float3 emissive = emissiveFactor.xyz;
    if (hasEmissiveTexture > 0.5) {
        float2 emisUV = selectUV(uv0, input.uv1, texCoord1Mask, kUV1Emissive);
        emissive *= emissiveTexture.Sample(emissiveSampler, emisUV).rgb;
    }

    float specularFactorSampled = specularFactor;
    if (hasSpecularTexture > 0.5) {
        float2 specUV = selectUV(uv0, input.uv1, texCoord1Mask, kUV1Specular);
        specularFactorSampled *= specularTexture.Sample(specularSampler, specUV).a;
    }

    float3 specularColorFactorSampled = specularColorFactor.xyz;
    if (hasSpecularColorTexture > 0.5) {
        float2 specColUV = selectUV(uv0, input.uv1, texCoord1Mask, kUV1SpecularColor);
        specularColorFactorSampled *= specularColorTexture.Sample(specularColorSampler, specColUV).rgb;
    }

    // Anisotropy.
    float anisoStrength = anisotropyStrength;
    float anisoRotation = anisotropyRotation;
    if (hasAnisotropicTexture > 0.5) {
        float2 anisoUV = selectUV(uv0, input.uv1, texCoord1Mask, kUV1Anisotropic);
        float2 anisoTex = anisotropicTexture.Sample(baseColorSampler, anisoUV).rg;
        anisoStrength *= anisoTex.r;
        anisoRotation += anisoTex.g * 2.0 * PI;
    }

    // KHR_materials_iridescence.
    float iridescenceFactorSampled = iridescenceFactor;
    if (hasIridescenceTexture > 0.5) {
        float2 iridUV = selectUV(uv0, input.uv1, texCoord1Mask, kUV1Iridescence);
        iridescenceFactorSampled *= iridescenceTexture.Sample(baseColorSampler, iridUV).r;
    }
    float iridescenceThickness = iridescenceThicknessMax;
    if (hasIridescenceThicknessTexture > 0.5) {
        float2 iridThickUV = selectUV(uv0, input.uv1, texCoord1Mask, kUV1IridescenceThickness);
        iridescenceThickness = lerp(iridescenceThicknessMin, iridescenceThicknessMax,
            iridescenceThicknessTexture.Sample(baseColorSampler, iridThickUV).g);
    }

    // KHR_materials_sheen.
    float3 sheenColor = sheenColorFactor.xyz;
    if (hasSheenColorTexture > 0.5) {
        float2 sheenColUV = selectUV(uv0, input.uv1, texCoord1Mask, kUV1SheenColor);
        sheenColor *= sheenColorTexture.Sample(baseColorSampler, sheenColUV).rgb;
    }
    float sheenRoughness = clamp(sheenRoughnessFactor, 0.07, 1.0);
    if (hasSheenRoughnessTexture > 0.5) {
        float2 sheenRoughUV = selectUV(uv0, input.uv1, texCoord1Mask, kUV1SheenRoughness);
        sheenRoughness = clamp(sheenRoughness * sheenRoughnessTexture.Sample(baseColorSampler, sheenRoughUV).r, 0.07, 1.0);
    }

    // KHR_materials_clearcoat.
    float ccFactor = clearcoatFactor;
    if (hasClearcoatTexture > 0.5) {
        float2 ccUV = selectUV(uv0, input.uv1, texCoord1Mask, kUV1Clearcoat);
        ccFactor *= clearcoatTexture.Sample(baseColorSampler, ccUV).r;
    }
    ccFactor = clamp(ccFactor, 0.0, 1.0);

    float ccRoughness = clamp(clearcoatRoughnessFactor, 0.001, 1.0);
    if (hasClearcoatRoughnessTexture > 0.5) {
        float2 ccRoughUV = selectUV(uv0, input.uv1, texCoord1Mask, kUV1ClearcoatRoughness);
        ccRoughness = clamp(ccRoughness * clearcoatRoughnessTexture.Sample(baseColorSampler, ccRoughUV).g, 0.001, 1.0);
    }

    float3 ccN = normalize(input.worldNormal);
    if (hasClearcoatNormalTexture > 0.5) {
        float2 ccNormUV = selectUV(uv0, input.uv1, texCoord1Mask, kUV1ClearcoatNormal);
        float3 ccNS = clearcoatNormalTexture.Sample(baseColorSampler, ccNormUV).rgb * 2.0 - 1.0;
        ccNS.xy *= clearcoatNormalScale;
        float3 T2    = normalize(input.worldTangent);
        float3 B2    = normalize(input.worldBitangent);
        float3 Nbase = normalize(input.worldNormal);
        float3 ccNsN = normalize(ccNS);
        ccN = normalize(ccNsN.x * T2 + ccNsN.y * B2 + ccNsN.z * Nbase);
    }

    // KHR_materials_volume.
    float thickness = thicknessFactor;
    if (hasThicknessTexture > 0.5) {
        float2 thickUV = selectUV(uv0, input.uv1, texCoord1Mask, kUV1Thickness);
        thickness *= thicknessTexture.Sample(baseColorSampler, thickUV).r;
    }

    // Every early return and the discard stay below the material texture samples above.
    // With them above the samples, Xbox Series X (AMD) drew the mesh's triangle edges as light
    // and dark lines in the lit output; an Intel GPU didn't. The likely cause is Sample()
    // choosing wrong mips at triangle edges, since it derives the mip from neighbouring pixels.
    // Debug view modes — ported from shaders/metal/default.metal's identical
    // three-checkpoint structure (each checkpoint fires only once the data
    // it needs has actually been computed, avoiding paying for IBL/lighting
    // just to show a debug view). This entire feature was previously absent
    // from the DirectX shader (confirmed: viewMode was declared in the
    // cbuffer but never read anywhere), so every ViewMode hotkey (0-9,
    // q/w/e/r/t/y/u/i/o/p/z) silently did nothing on Windows — pressing e.g.
    // '4' (roughness) just kept showing normal shading.

    // Clip planes (Renderer::setClipPlanes()): discard the clipped side. With
    // section caps on, back faces of single-sided surfaces (the inside of a
    // cut solid, visible through the cut) take the cap colour (default.frag).
    for (uint ci = 0u; ci < clipInfo.x; ++ci) {
        if (dot(clipPlanes[ci].xyz, input.worldPos) + clipPlanes[ci].w < 0.0) discard;
    }
    if (clipInfo.y != 0u && !isFrontFace && doubleSided < 0.5) {
        return float4(capColor.rgb, 1.0);
    }

    if (viewMode > 0.5) {
        if (abs(viewMode - 1.0) < 0.5) { // VIEW_MODE_WORLD_NORMAL
            float3 Ndbg = normalize(input.worldNormal);
            return float4(Ndbg * 0.5 + 0.5, 1.0);
        }
        if (abs(viewMode - 2.0) < 0.5) { // VIEW_MODE_BASE_COLOR
            return float4(baseColor.rgb, 1.0);
        }
    }

    // Alpha mask.
    if (alphaMode > 0.5 && alphaMode < 1.5 && baseColor.a < alphaCutoff) {
        discard;
    }

    // Unlit: return base color without lighting.
    if (unlit > 0.5) {
        return baseColor;
    }

    // Debug view modes, checkpoint 2 — see the first checkpoint's comment above.
    if (viewMode > 0.5) {
        if (abs(viewMode - 3.0) < 0.5) { // VIEW_MODE_METALLIC
            return float4(metallic, metallic, metallic, 1.0);
        }
        if (abs(viewMode - 4.0) < 0.5) { // VIEW_MODE_ROUGHNESS
            return float4(roughness, roughness, roughness, 1.0);
        }
        if (abs(viewMode - 5.0) < 0.5) { // VIEW_MODE_OCCLUSION
            return float4(occlusion, occlusion, occlusion, 1.0);
        }
        if (abs(viewMode - 6.0) < 0.5) { // VIEW_MODE_EMISSIVE
            return float4(emissive, 1.0);
        }
        if (abs(viewMode - 7.0) < 0.5) { // VIEW_MODE_ALPHA
            return float4(baseColor.a, baseColor.a, baseColor.a, 1.0);
        }
        if (abs(viewMode - 8.0) < 0.5) { // VIEW_MODE_UV0
            return float4(frac(uv0), 0.0, 1.0);
        }
        if (abs(viewMode - 9.0) < 0.5) { // VIEW_MODE_SPECULAR_FACTOR
            return float4(specularFactorSampled, specularFactorSampled, specularFactorSampled, 1.0);
        }
        if (abs(viewMode - 10.0) < 0.5) { // VIEW_MODE_SPECULAR_COLOR
            return float4(specularColorFactorSampled, 1.0);
        }
    }

    // View direction & double-sided normal flip.
    float3 camPosXYZ = cameraPos.xyz;
    float3 viewDir = normalize(camPosXYZ - input.worldPos);

    if (dot(N, viewDir) < 0.0) N = -N;
    if (dot(ccN, viewDir) < 0.0) ccN = -ccN;

    float NdotV   = max(dot(N, viewDir), 0.0001);
    float ccNdotV = max(dot(ccN, viewDir), 0.0001);

    // --- IBL (image-based lighting) from environment cubemap ---
    float3 iblDiffuse   = float3(0.0, 0.0, 0.0);
    float3 iblSpecular  = float3(0.0, 0.0, 0.0);
    float3 iblClearcoat = float3(0.0, 0.0, 0.0);
    if (iblEnabled > 0.5) {
        // Lambertian-convolved diffuse irradiance (bakeIblResources() mode 2) — not a raw
        // environmentMap sample, so no hand-tuned brightness fudge factor is needed.
        float3 envDiffuse = irradianceMap.Sample(environmentSampler, N).rgb;
        iblDiffuse = baseColor.rgb * (1.0 - metallic) * envDiffuse * environmentIntensity;

        // environmentMap here is the GGX-prefiltered specular cubemap (bakeIblResources()
        // mode 1) — roughness*envMaxLod selects the mip matching that roughness's GGX lobe.
        uint envW, envH, envLevels;
        environmentMap.GetDimensions(0, envW, envH, envLevels);
        float envMaxLod = max(float(envLevels) - 1.0, 0.0);

        float3 R = reflect(-viewDir, N);
        float3 envSpecular = environmentMap.SampleLevel(environmentSampler, R, roughness * envMaxLod).rgb;
        float f0_scalar = (ior - 1.0) / (ior + 1.0);
        f0_scalar *= f0_scalar;
        float3 F0 = lerp(float3(f0_scalar, f0_scalar, f0_scalar) * specularColorFactor.xyz, baseColor.rgb, metallic);
        if (iridescenceFactorSampled > 0.0) {
            float3 iridF0 = EvalIridescence(1.0, iridescenceIor, NdotV, iridescenceThickness, F0);
            F0 = lerp(F0, iridF0, iridescenceFactorSampled);
        }
        float3 F = iblGGXFresnel(NdotV, roughness, F0);
        iblSpecular = envSpecular * F * environmentIntensity;

        if (ccFactor > 0.0) {
            float3 ccR = reflect(-viewDir, ccN);
            float3 envCC = environmentMap.SampleLevel(environmentSampler, ccR, ccRoughness * envMaxLod).rgb;
            float ccFresnel = F_Schlick_scalar(0.04, ccNdotV);
            iblClearcoat = envCC * ccFresnel * ccFactor * environmentIntensity;
        }
    }

    // --- KHR_materials_transmission: screen-space refraction ---
    float3 transmitted   = float3(0.0, 0.0, 0.0);
    float  transmittance = 0.0;
    if (transmission > 0.0 && metallic < 0.99) {
        float eta = 1.0 / ior;
        float3 Trefr = refract(-viewDir, N, eta);
        if (all(Trefr == float3(0.0, 0.0, 0.0))) Trefr = -viewDir;

        float2 sampleUV[3];
        bool uvValid[3];
        sampleUV[0] = sampleUV[1] = sampleUV[2] = float2(0.0, 0.0);
        uvValid[0] = uvValid[1] = uvValid[2] = true;
        if (thickness > 0.0) {
            float3 viewPos = mul(viewMatrix, float4(input.worldPos, 1.0)).xyz;
            float3 viewNormal = normalize(mul(viewMatrix, float4(N, 0.0)).xyz);
            float3 viewIncident = normalize(viewPos);
            if (dispersion > 0.0) {
                float dispersionScale = dispersion * 0.02;
                float3 iorRGB = float3(ior + dispersionScale, ior, max(ior - dispersionScale, 1.001));
                for (int ch = 0; ch < 3; ch++) {
                    float etaCh = 1.0 / iorRGB[ch];
                    float3 viewRefractCh = refract(viewIncident, viewNormal, etaCh);
                    if (all(viewRefractCh == float3(0.0, 0.0, 0.0))) viewRefractCh = viewIncident;
                    float3 backPosCh = viewPos + viewRefractCh * thickness;
                    float4 backClipCh = mul(projMatrix, float4(backPosCh, 1.0));
                    sampleUV[ch] = backClipCh.xy / backClipCh.w * 0.5 + 0.5;
                    sampleUV[ch].y = 1.0 - sampleUV[ch].y;
                    uvValid[ch] = all(sampleUV[ch] >= float2(0.0, 0.0)) && all(sampleUV[ch] <= float2(1.0, 1.0));
                }
            } else {
                float3 viewRefract = refract(viewIncident, viewNormal, eta);
                if (all(viewRefract == float3(0.0, 0.0, 0.0))) viewRefract = viewIncident;
                float3 backPos = viewPos + viewRefract * thickness;
                float4 backClip = mul(projMatrix, float4(backPos, 1.0));
                sampleUV[0] = sampleUV[1] = sampleUV[2] = backClip.xy / backClip.w * 0.5 + 0.5;
                sampleUV[0].y = 1.0 - sampleUV[0].y;
                uvValid[0] = uvValid[1] = uvValid[2] = all(sampleUV[0] >= float2(0.0, 0.0)) && all(sampleUV[0] <= float2(1.0, 1.0));
            }
        } else {
            sampleUV[0] = sampleUV[1] = sampleUV[2] = input.clipPosition.xy / screenSize;
        }

        // Official glTF-Sample-Viewer LOD formula.
        float iorScale = clamp(ior * 2.0 - 2.0, 0.0, 1.0);
        float lod = log2(screenSize.x) * roughness * iorScale;

        for (int ch2 = 0; ch2 < 3; ch2++) {
            if (uvValid[ch2]) {
                // Stored display-encoded (tone mapped, sRGB): back to linear.
                transmitted[ch2] = pow(max(sceneColorTexture.SampleLevel(sceneColorSampler, sampleUV[ch2], lod)[ch2], 0.0), 2.2);
            } else {
                // Explicit LOD: this branch varies per pixel, so Sample() can't pick a mip here.
                transmitted[ch2] = environmentMap.SampleLevel(environmentSampler, Trefr, 0.0)[ch2];
            }
        }

        if (thickness > 0.0 && attenuationDistance > 0.0 && !isinf(attenuationDistance)) {
            float3 attn = (float3(1.0, 1.0, 1.0) - attenuationColor.xyz) / attenuationDistance;
            transmitted *= exp(-thickness * attn);
        }

        transmitted *= baseColor.rgb;

        float f0 = (ior - 1.0) / (ior + 1.0);
        f0 *= f0;
        float fresnelT = f0 + (1.0 - f0) * pow(1.0 - NdotV, 5.0);
        transmittance = transmission * (1.0 - fresnelT) * (1.0 - metallic);
    }

    // Debug view modes, checkpoint 3 (final) — see the first checkpoint's
    // comment above. Covers every remaining mode; falls through to black for
    // any unrecognized value (matches Metal's fallback).
    if (viewMode > 0.5) {
        if (abs(viewMode - 11.0) < 0.5) { // VIEW_MODE_SHEEN_COLOR
            return float4(sheenColor, 1.0);
        }
        if (abs(viewMode - 12.0) < 0.5) { // VIEW_MODE_SHEEN_ROUGHNESS
            return float4(sheenRoughness, sheenRoughness, sheenRoughness, 1.0);
        }
        if (abs(viewMode - 13.0) < 0.5) { // VIEW_MODE_CLEARCOAT
            return float4(ccFactor, ccFactor, ccFactor, 1.0);
        }
        if (abs(viewMode - 14.0) < 0.5) { // VIEW_MODE_CLEARCOAT_ROUGHNESS
            return float4(ccRoughness, ccRoughness, ccRoughness, 1.0);
        }
        if (abs(viewMode - 15.0) < 0.5) { // VIEW_MODE_CLEARCOAT_NORMAL
            return float4(ccN * 0.5 + 0.5, 1.0);
        }
        if (abs(viewMode - 16.0) < 0.5) { // VIEW_MODE_TRANSMISSION
            return float4(transmission, transmission, transmission, 1.0);
        }
        if (abs(viewMode - 17.0) < 0.5) { // VIEW_MODE_ENVIRONMENT
            return float4(iblDiffuse + iblSpecular, 1.0);
        }
        if (abs(viewMode - 18.0) < 0.5) { // VIEW_MODE_IRIDESCENCE
            float dbgF0Scalar = (ior - 1.0) / (ior + 1.0);
            dbgF0Scalar *= dbgF0Scalar;
            float3 dbgBaseF0 = lerp(float3(dbgF0Scalar, dbgF0Scalar, dbgF0Scalar) * specularColorFactor.xyz, baseColor.rgb, metallic);
            float3 iridF0 = EvalIridescence(1.0, iridescenceIor, NdotV, iridescenceThickness, dbgBaseF0);
            return float4(iridF0 * iridescenceFactorSampled, 1.0);
        }
        if (abs(viewMode - 19.0) < 0.5) { // VIEW_MODE_ANISOTROPY
            return float4(anisoStrength, anisoStrength, anisoStrength, 1.0);
        }
        if (abs(viewMode - 20.0) < 0.5) { // VIEW_MODE_DISPERSION
            return float4(dispersion, dispersion, dispersion, 1.0);
        }
        return float4(0.0, 0.0, 0.0, 1.0);
    }

    // --- Punctual light loop (up to 4 lights) ---
    float3 totalDiffuse   = float3(0.0, 0.0, 0.0);
    float3 totalSpecular  = float3(0.0, 0.0, 0.0);
    float3 totalSheen     = float3(0.0, 0.0, 0.0);
    float3 totalClearcoat = float3(0.0, 0.0, 0.0);

    uint activeLightCount = lightCount;
    for (uint i = 0; i < activeLightCount && i < 4u; i++) {
        Light light = lights[i];

        float3 lightDir;
        float attenuation = 1.0;
        float spotFactor   = 1.0;
        float typeVal       = light.position.w;

        if (typeVal < 0.5) {
            lightDir = normalize(light.position.xyz);
        } else {
            float3 toLight = light.position.xyz - input.worldPos;
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
        // The sun (spotAngles.z): darkened where the shadow map says so.
        float shadow     = light.spotAngles.z > 0.5 ? sunShadow(input.worldPos, N, lightDir) : 1.0;
        float3  lightColor = light.color.xyz * light.color.w * attenuation * spotFactor * shadow;

        float3  halfDir = normalize(lightDir + viewDir);
        float NdotH   = max(dot(N, halfDir), 0.0);
        float VdotH   = max(dot(viewDir, halfDir), 0.0);

        float f0_scalarL = (ior - 1.0) / (ior + 1.0);
        f0_scalarL *= f0_scalarL;

        float spec = specularFactorSampled;
        float3 specColor = specularColorFactorSampled;

        float3 F0_dielectric = min(float3(f0_scalarL, f0_scalarL, f0_scalarL) * specColor, float3(1.0, 1.0, 1.0)) * spec;
        float3 F0 = lerp(F0_dielectric, baseColor.rgb, metallic);
        if (iridescenceFactorSampled > 0.0) {
            float3 iridF0 = EvalIridescence(1.0, iridescenceIor, NdotV, iridescenceThickness, F0);
            F0 = lerp(F0, iridF0, iridescenceFactorSampled);
        }
        // Schlick Fresnel at this light's half-vector angle (F90 = 1) — used both to weight
        // specular's grazing-angle brightening and, via energy conservation, to reduce the
        // diffuse response by the same amount.
        float3 fresnel = F0 + (float3(1.0, 1.0, 1.0) - F0) * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);

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

        // BRDF_lambertian(baseColor) = baseColor / PI.
        float3 lDiffuse  = (baseColor.rgb / PI) * NdotL * lightColor;
        float3 lSpecular = fresnel * D * V * NdotL * lightColor;

        // Metals have no diffuse response at all; dielectrics split energy between diffuse
        // and specular by the Fresnel reflectance instead of adding both unconditionally.
        totalDiffuse  += lDiffuse * (float3(1.0, 1.0, 1.0) - fresnel) * (1.0 - metallic);
        totalSpecular += lSpecular;

        float sheenD = D_Charlie(sheenRoughness, NdotH);
        float sheenV = V_Neubelt(NdotV, max(NdotL, 0.0001));
        totalSheen += sheenColor * sheenD * sheenV * NdotL * lightColor;

        float ccNdotH = max(dot(ccN, halfDir), 0.0);
        float ccNdotL = max(dot(ccN, lightDir), 0.0);
        float cc_D    = D_GGX(ccRoughness, ccNdotH);
        float cc_V    = V_SmithGGX(ccNdotL, ccNdotV, ccRoughness);
        float cc_F    = F_Schlick_scalar(0.04, VdotH);
        totalClearcoat += float3(cc_D * cc_V * cc_F, cc_D * cc_V * cc_F, cc_D * cc_V * cc_F) * ccFactor * ccNdotL * lightColor;
    }

    // --- Final composition ---
    // The flat ambient-color hack is only a stand-in for real indirect lighting when IBL is
    // off; with IBL on, iblDiffuse already supplies the ambient term and this would
    // otherwise double-count it.
    float3 ambientColor = (iblEnabled > 0.5) ? float3(0.0, 0.0, 0.0) : baseColor.rgb * 0.25 * occlusion;
    float3 diffuse      = totalDiffuse;
    iblDiffuse  *= occlusion;
    iblSpecular *= occlusion;

    float diffuseScale = 1.0 - transmittance;
    ambientColor *= diffuseScale;
    diffuse      *= diffuseScale;
    iblDiffuse   *= diffuseScale;

    float ccAmbientAtten = 1.0 - ccFactor * F_Schlick_scalar(0.04, ccNdotV);
    float3 finalColor = (ambientColor + diffuse + totalSpecular + totalSheen + iblDiffuse + iblSpecular
                       + emissive + transmitted * transmittance) * ccAmbientAtten
                      + totalClearcoat + iblClearcoat;

    finalColor *= exp2(viewParams.x);  // exposure (Renderer::setExposure())
    finalColor = toneMapKhronosPbrNeutral(finalColor);
    finalColor = linearToSRGBFast(finalColor);

    // glTF spec: alphaMode OPAQUE (0) and MASK (1) must render fully opaque
    // -- alpha is ignored (OPAQUE) or only used for the cutoff test already
    // applied above (MASK); only BLEND (2) actually blends by baseColor.a.
    float outAlpha = (alphaMode < 1.5) ? 1.0 : baseColor.a;
    return float4(finalColor, outAlpha);
}

// ---------------------------------------------------------------------------
// FXAA post-process shader (HLSL shader model 6.0)
// Based on FXAA 3.11 by Timothy Lottes (simplified).
// ---------------------------------------------------------------------------

struct FxaaOut {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;  // normalized target coordinates, (0, 0) = top-left
};

FxaaOut fxaaVertex(uint vertexID : SV_VertexID)
{
    FxaaOut output;
    float2 pos;
    if (vertexID == 0) pos = float2(-1.0, -1.0);
    else if (vertexID == 1) pos = float2(3.0, -1.0);
    else pos = float2(-1.0, 3.0);
    output.position = float4(pos, 1.0, 1.0);
    // D3D clip space has +Y up; texture coordinates start at the top.
    output.uv = float2((pos.x + 1.0) * 0.5, (1.0 - pos.y) * 0.5);
    return output;
}

cbuffer FxaaUniforms : register(b0)
{
    float2 rcpFrame;
    float2 _pad;
};

Texture2D<float4> sceneTexture : register(t0);
SamplerState      sceneSampler : register(s0);

static float fxaaLuma(float3 rgb)
{
    return dot(rgb, float3(0.299, 0.587, 0.114));
}

float4 fxaaPixel(FxaaOut input) : SV_Target
{
    float2 pos = input.position.xy;
    float2 uv = pos * rcpFrame;

    float3 rgbNW = sceneTexture.Sample(sceneSampler, uv + float2(-rcpFrame.x, -rcpFrame.y)).rgb;
    float3 rgbNE = sceneTexture.Sample(sceneSampler, uv + float2( rcpFrame.x, -rcpFrame.y)).rgb;
    float3 rgbSW = sceneTexture.Sample(sceneSampler, uv + float2(-rcpFrame.x,  rcpFrame.y)).rgb;
    float3 rgbSE = sceneTexture.Sample(sceneSampler, uv + float2( rcpFrame.x,  rcpFrame.y)).rgb;
    float3 rgbM  = sceneTexture.Sample(sceneSampler, uv).rgb;

    float lumaNW = fxaaLuma(rgbNW);
    float lumaNE = fxaaLuma(rgbNE);
    float lumaSW = fxaaLuma(rgbSW);
    float lumaSE = fxaaLuma(rgbSE);
    float lumaM  = fxaaLuma(rgbM);

    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

    if (lumaMax - lumaMin < max(0.0833, lumaMax * 0.166))
        return float4(rgbM, 1.0);

    float2 dir;
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));

    float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * 0.25 * 0.125, 1.0/128.0);
    float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    dir = clamp(dir * rcpDirMin, float2(-8.0, -8.0), float2(8.0, 8.0)) * rcpFrame;

    float3 rgbA = 0.5 * (
        sceneTexture.Sample(sceneSampler, uv + dir * (1.0/3.0 - 0.5)).rgb +
        sceneTexture.Sample(sceneSampler, uv + dir * (2.0/3.0 - 0.5)).rgb);

    float3 rgbB = rgbA * 0.5 + 0.25 * (
        sceneTexture.Sample(sceneSampler, uv + dir * -0.5).rgb +
        sceneTexture.Sample(sceneSampler, uv + dir *  0.5).rgb);

    float lumaB = fxaaLuma(rgbB);
    if (lumaB < lumaMin || lumaB > lumaMax)
        return float4(rgbA, 1.0);
    return float4(rgbB, 1.0);
}

// ---------------------------------------------------------------------------
// Downsample post-process shader (HLSL shader model 6.0)
// Reuses fxaaVertex for the fullscreen triangle.
// ---------------------------------------------------------------------------

Texture2D<float4> downsampleSceneTexture : register(t0);
SamplerState      downsampleSceneSampler : register(s0);

float4 downsamplePixel(FxaaOut input) : SV_Target
{
    // The scene texture is ssaaScale x the output, so use the output-normalized
    // UV from the vertex shader, not SV_Position / scene texture size.
    return downsampleSceneTexture.Sample(downsampleSceneSampler, input.uv);
}

// ---------------------------------------------------------------------------
// Skybox shader — fullscreen triangle (reuses fxaaVertex) that samples the
// prefiltered environment cubemap. Direct HLSL port of shaders/vulkan/skybox.frag
// (itself a direct port of shaders/metal/default.metal's skyboxFragment).
//
// NDC reconstruction note: unlike Vulkan's, this D3D12 pipeline's vertexMain-equivalent
// (fxaaVertex, reused here) never negates clip-space Y (see this file's header comment),
// so u.invVP stays defined in the same "logical" Y-up NDC convention Metal's does.
// Reconstructing world direction from SV_Position must undo that same logical convention,
// which is why this still uses the (1.0 - fragCoord.y/screenSize.y) formula both Metal and
// Vulkan use — see shaders/vulkan/skybox.frag's header comment for the full explanation of
// why that's unrelated to the vertex-stage Y-flip question.
// ---------------------------------------------------------------------------

TextureCube<float4> skyboxEnvMap     : register(t0);
SamplerState         skyboxEnvSampler : register(s1);

cbuffer SkyboxUniforms : register(b2) {
    float4x4 invVP;
    float2   skyboxScreenSize;
    float2   _skyboxPad;      // x = exposure in EV (as CameraUniforms.viewParams.x)
    float4   skyboxCameraPos; // xyz used, w unused
};

float4 skyboxPixel(FxaaOut input) : SV_Target {
    float2 ndc = float2(
        (input.position.x / skyboxScreenSize.x) * 2.0 - 1.0,
        (1.0 - input.position.y / skyboxScreenSize.y) * 2.0 - 1.0
    );
    float4 worldFar = mul(invVP, float4(ndc, 1.0, 1.0));
    float3 worldDir = normalize(worldFar.xyz / worldFar.w - skyboxCameraPos.xyz);
    float3 color = skyboxEnvMap.Sample(skyboxEnvSampler, worldDir).rgb;

    // Match the tonemapping + sRGB encoding pixelMain() applies before writing to the
    // 8-bit swapchain — the environment is genuine HDR, so returning it raw here hard-clips
    // to solid white/saturated colors instead of compressing smoothly.
    color *= exp2(_skyboxPad.x);  // exposure, as pixelMain
    color = toneMapKhronosPbrNeutral(color);
    color = linearToSRGBFast(color);

    return float4(color, 1.0);
}

// ---------------------------------------------------------------------------
// IBL precompute shader — fullscreen triangle (reuses fxaaVertex) that bakes the three
// environment-independent/environment-derived resources the glTF-Sample-Renderer reference
// uses for physically correct image-based lighting. Direct HLSL port of
// shaders/vulkan/ibl_bake.frag (see that file's header comment for what each mode computes:
// mode 0 = BRDF LUT, mode 1 = GGX prefilter, mode 2 = irradiance convolution).
// ---------------------------------------------------------------------------

cbuffer IblBakeUniforms : register(b0) {
    int   iblMode;       // 0 = BRDF LUT, 1 = GGX prefilter, 2 = irradiance convolution
    int   iblFaceIndex;  // 0..5, used for mode 1/2
    float iblRoughness;  // used for mode 1
    float iblOutputSize; // resolution (texels, square) of the current render target
};

TextureCube<float4> iblSrcEnvMap     : register(t0);
SamplerState         iblSrcEnvSampler : register(s1);

// Direction for cubemap face `face` at signed uv in [-1,1] — must match the CPU-side
// convention in Renderer::convertEquirectangularImageToCubemap() exactly (right-handed,
// +Z forward, face order +X,-X,+Y,-Y,+Z,-Z), so baked faces stay aligned with faces
// uploaded via the equirect/6-file loaders.
float3 iblFaceDirection(int face, float uu, float vv) {
    float3 d;
    if (face == 0)      d = float3( 1.0, -vv, -uu);
    else if (face == 1) d = float3(-1.0, -vv,  uu);
    else if (face == 2) d = float3( uu,  1.0,  vv);
    else if (face == 3) d = float3( uu, -1.0, -vv);
    else if (face == 4) d = float3( uu, -vv,  1.0);
    else                 d = float3(-uu, -vv, -1.0);
    return normalize(d);
}

// Van der Corput radical inverse (base 2) + Hammersley low-discrepancy sequence — standard
// quasi-Monte-Carlo sampling for GGX importance sampling (Karis 2013).
float iblRadicalInverseVdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

float2 iblHammersley(uint i, uint n) {
    return float2(float(i) / float(n), iblRadicalInverseVdC(i));
}

float3 iblImportanceSampleGGX(float2 xi, float3 N, float roughness) {
    float a = roughness * roughness;

    float phi = 2.0 * PI * xi.x;
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

float iblGeometrySchlickGGX(float NdotV, float roughness) {
    float a = roughness;
    float k = (a * a) / 2.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float iblGeometrySmith(float NdotV, float NdotL, float roughness) {
    return iblGeometrySchlickGGX(NdotV, roughness) * iblGeometrySchlickGGX(NdotL, roughness);
}

// Karis split-sum BRDF LUT integration.
float2 iblIntegrateBRDF(float NdotV, float roughness) {
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

// GGX-importance-sampled prefiltered specular radiance for direction N at the given
// roughness — Karis's split-sum "pre-filtered environment map" half.
float3 iblPrefilterEnvironment(float3 N, float roughness) {
    // At roughness ~0 the GGX lobe collapses to a delta function, so importance sampling
    // has already converged to the exact answer: a single direct sample.
    if (roughness < 0.01) {
        return iblSrcEnvMap.SampleLevel(iblSrcEnvSampler, N, 0.0).rgb;
    }

    float3 V = N;
    float3 prefilteredColor = float3(0.0, 0.0, 0.0);
    float totalWeight = 0.0;
    // This is a one-time bake, not a per-frame cost, so a high sample count is affordable —
    // see shaders/vulkan/ibl_bake.frag's header comment for why 512.
    const uint SAMPLE_COUNT = 512u;
    for (uint i = 0u; i < SAMPLE_COUNT; ++i) {
        float2 xi = iblHammersley(i, SAMPLE_COUNT);
        float3 H = iblImportanceSampleGGX(xi, N, roughness);
        float3 L = normalize(2.0 * dot(V, H) * H - V);
        float NdotL = max(dot(N, L), 0.0);
        if (NdotL > 0.0) {
            // Explicit mip 0 — see shaders/vulkan/ibl_bake.frag's header comment for why
            // implicit derivative-based LOD selection is wrong for Monte Carlo directions.
            prefilteredColor += iblSrcEnvMap.SampleLevel(iblSrcEnvSampler, L, 0.0).rgb * NdotL;
            totalWeight += NdotL;
        }
    }
    return prefilteredColor / max(totalWeight, 0.0001);
}

// Cosine-weighted hemisphere convolution of the source environment around normal N — the
// Lambertian diffuse irradiance integral the reference bakes offline; computed here on GPU.
float3 iblConvolveIrradiance(float3 N) {
    float3 up = (abs(N.z) < 0.999) ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 tangent = normalize(cross(up, N));
    float3 bitangent = cross(N, tangent);

    float3 irradiance = float3(0.0, 0.0, 0.0);
    const uint SAMPLE_COUNT = 16384u;
    for (uint i = 0u; i < SAMPLE_COUNT; ++i) {
        float2 xi = iblHammersley(i, SAMPLE_COUNT);
        float phi = 2.0 * PI * xi.x;
        float cosTheta = sqrt(1.0 - xi.y);
        float sinTheta = sqrt(xi.y);
        float3 localDir = float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
        float3 sampleDir = localDir.x * tangent + localDir.y * bitangent + localDir.z * N;
        irradiance += iblSrcEnvMap.SampleLevel(iblSrcEnvSampler, sampleDir, 0.0).rgb;
    }
    return irradiance / float(SAMPLE_COUNT); // the average radiance (as glTF-Sample-Viewer): diffuse = albedo x it; with pi it was pi times too bright
}

float4 iblBakePixel(FxaaOut input) : SV_Target {
    if (iblMode == 0) {
        float2 uv = input.position.xy / iblOutputSize;
        return float4(iblIntegrateBRDF(clamp(uv.x, 0.001, 1.0), clamp(uv.y, 0.001, 1.0)), 0.0, 1.0);
    }

    // Modes 1/2 render into one face of a cubemap: convert this fragment's pixel coordinate
    // to a signed [-1,1] uv, then to a world direction via the same per-face basis the
    // CPU-side equirect loader uses.
    float2 ndc = (input.position.xy / iblOutputSize) * 2.0 - 1.0;
    float3 dir = iblFaceDirection(iblFaceIndex, ndc.x, ndc.y);

    if (iblMode == 1) {
        return float4(iblPrefilterEnvironment(dir, iblRoughness), 1.0);
    } else {
        return float4(iblConvolveIrradiance(dir), 1.0);
    }
}

// ---------------------------------------------------------------------------
// Procedural texture bake compute shader — uber-shader VM interpreter.
//
// Each thread evaluates the full procedural graph for one pixel.
// The graph is flattened to an array of ProceduralNode structs (bytecode).
// A fixed-size register file of float4 values holds intermediate results.
//
// Bindings:
//   register(b0)  — ProceduralBakeUniforms (constant buffer)
//   register(t1)  — ProceduralNode[] node bytecode (SRV)
//   register(u2)  — float4[] output pixels (UAV)
//
// Compile on Windows with DXC:
//   dxc -T cs_6_0 -E proceduralBakeKernel default.hlsl -Fo procedural_bake.dxil
// ---------------------------------------------------------------------------

struct ProceduralNode {
    float4   value;     // inline constant / parameters
    uint     op;        // ProceduralOp opcode
    uint     outReg;    // output register index
    uint     inA;       // input register A  (0xFFFFFFFF = none)
    uint     inB;       // input register B
    uint     inC;       // input register C
    uint     inD;       // input register D
    uint     flags;     // type / swizzle / presence mask
    uint     pad;
};

struct ProceduralBakeUniforms {
    uint   width;
    uint   height;
    uint   nodeCount;
    uint   outputReg;
    uint   outputComponents; // 1=float, 2=vec2, 3=vec3/color3, 4=color4
    uint   pad[3];
};

// Opcodes — must match C++ ProceduralOp enum exactly.
static const uint OP_CONSTANT      = 0;
static const uint OP_TEXCOORD      = 1;
static const uint OP_ADD           = 2;
static const uint OP_SUBTRACT      = 3;
static const uint OP_MULTIPLY      = 4;
static const uint OP_DIVIDE        = 5;
static const uint OP_FLOOR         = 6;
static const uint OP_MODULO        = 7;
static const uint OP_ABS           = 8;
static const uint OP_CLAMP         = 9;
static const uint OP_POWER         = 10;
static const uint OP_SQRT          = 11;
static const uint OP_MAX           = 12;
static const uint OP_MIN           = 13;
static const uint OP_SIN           = 14;
static const uint OP_COS           = 15;
static const uint OP_DOTPRODUCT    = 16;
static const uint OP_CROSSPRODUCT  = 17;
static const uint OP_LENGTH        = 18;
static const uint OP_DISTANCE      = 19;
static const uint OP_NORMALIZE     = 20;
static const uint OP_MIX           = 21;
static const uint OP_NOISE2D       = 22;
static const uint OP_CHECKERBOARD  = 23;
static const uint OP_PLACE2D       = 24;
static const uint OP_SWIZZLE       = 25;
static const uint OP_COMBINE       = 26;
static const uint OP_EXTRACT       = 27;
static const uint OP_IFGREATER     = 28;
static const uint OP_IFEQUAL       = 29;

// Presence flags occupy bits 8-11 so they don't collide with swizzle
// (bits 0-7), extract (bits 0-1), or combine count (bits 0-3).
static const uint FLAG_HAS_A = 0x0100;
static const uint FLAG_HAS_B = 0x0200;
static const uint FLAG_HAS_C = 0x0400;
static const uint FLAG_HAS_D = 0x0800;

// Noise permutation table (identical to CPU baker).
static const int kPerm[512] = {
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

static float fade(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

static float lerpVal(float a, float b, float t) {
    return a + t * (b - a);
}

static float grad(int hash, float x, float y) {
    int h = hash & 7;
    float u = h < 4 ? x : -x;
    float v = h < 2 || h == 5 || h == 6 ? y : -y;
    return u + v;
}

static float perlin2D(float x, float y) {
    int X = (int)floor(x) & 255;
    int Y = (int)floor(y) & 255;
    x -= floor(x);
    y -= floor(y);
    float u = fade(x);
    float v = fade(y);
    int A = kPerm[X] + Y;
    int B = kPerm[X + 1] + Y;
    return lerpVal(v, lerpVal(u, grad(kPerm[A], x, y), grad(kPerm[B], x - 1.0f, y)),
                       lerpVal(u, grad(kPerm[A + 1], x, y - 1.0f), grad(kPerm[B + 1], x - 1.0f, y - 1.0f)));
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

// Raw buffers: campello_gpu views storage buffers as (RW)ByteAddressBuffers
// (a bind group layout carries no structure stride). ProceduralNode is 48
// bytes: value, then 8 uints.
ByteAddressBuffer   nodes     : register(t1);
RWByteAddressBuffer outPixels : register(u2);

ProceduralNode loadNode(uint n) {
    uint base = n * 48;
    ProceduralNode node;
    node.value  = asfloat(nodes.Load4(base));
    uint4 a = nodes.Load4(base + 16);
    uint4 b = nodes.Load4(base + 32);
    node.op = a.x; node.outReg = a.y; node.inA = a.z; node.inB = a.w;
    node.inC = b.x; node.inD = b.y; node.flags = b.z; node.pad = b.w;
    return node;
}
ConstantBuffer<ProceduralBakeUniforms> uniforms : register(b0);

[numthreads(1, 1, 1)]
void proceduralBakeKernel(uint3 groupID : SV_GroupID)
{
    uint x = groupID.x;
    uint y = groupID.y;
    if (x >= uniforms.width || y >= uniforms.height) return;

    const uint REG_COUNT = 64;
    float4 regs[REG_COUNT];
    for (uint i = 0; i < REG_COUNT; i++) regs[i] = float4(0.0f, 0.0f, 0.0f, 0.0f);

    float u = ((float)x + 0.5f) / (float)uniforms.width;
    float v = ((float)y + 0.5f) / (float)uniforms.height;

    for (uint n = 0; n < uniforms.nodeCount; n++) {
        ProceduralNode node = loadNode(n);
        float4 a = (node.flags & FLAG_HAS_A) ? regs[node.inA] : float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 b = (node.flags & FLAG_HAS_B) ? regs[node.inB] : float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 c = (node.flags & FLAG_HAS_C) ? regs[node.inC] : float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 d = (node.flags & FLAG_HAS_D) ? regs[node.inD] : float4(0.0f, 0.0f, 0.0f, 0.0f);

        float4 r = float4(0.0f, 0.0f, 0.0f, 0.0f);

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
                r = lerp(a, b, c.x);
                break;
            case OP_NOISE2D: {
                float u_in = (node.flags & FLAG_HAS_A) ? a.x : u;
                float v_in = (node.flags & FLAG_HAS_A) ? a.y : v;
                int octaves = (int)node.value.x;
                float lacun = node.value.y;
                float gain = node.value.z;
                float scale = node.value.w;
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
                bool check = ((int)floor(u_in * tx) + (int)floor(v_in * ty)) % 2 == 0;
                r = check ? b : c;
                break;
            }
            case OP_PLACE2D: {
                float u_in = (node.flags & FLAG_HAS_A) ? a.x : u;
                float v_in = (node.flags & FLAG_HAS_A) ? a.y : v;
                float2 offset = node.value.xy;
                float2 scale = node.value.zw;
                float rot = (node.flags & FLAG_HAS_B) ? b.x : 0.0f;
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
                uint sx = (node.flags >> 0) & 3;
                uint sy = (node.flags >> 2) & 3;
                uint sz = (node.flags >> 4) & 3;
                uint sw = (node.flags >> 6) & 3;
                float4 v = a;
                r.x = (sx == 0) ? v.x : (sx == 1) ? v.y : (sx == 2) ? v.z : v.w;
                r.y = (sy == 0) ? v.x : (sy == 1) ? v.y : (sy == 2) ? v.z : v.w;
                r.z = (sz == 0) ? v.x : (sz == 1) ? v.y : (sz == 2) ? v.z : v.w;
                r.w = (sw == 0) ? v.x : (sw == 1) ? v.y : (sw == 2) ? v.z : v.w;
                break;
            }
            case OP_COMBINE: {
                uint count = node.flags & 0xFF;
                r.x = a.x;
                r.y = count > 1 ? b.x : 0.0f;
                r.z = count > 2 ? c.x : 0.0f;
                r.w = count > 3 ? d.x : 1.0f;
                break;
            }
            case OP_EXTRACT: {
                uint comp = node.flags & 3;
                float val = a[comp];
                r = float4(val, val, val, val);
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
    uint comp = uniforms.outputComponents;
    if (comp == 1) {
        final = float4(final.x, 0.0f, 0.0f, 1.0f);
    } else if (comp == 2) {
        final = float4(final.x, final.y, 0.0f, 1.0f);
    } else if (comp == 3) {
        final = float4(final.x, final.y, final.z, 1.0f);
    } else if (comp == 4) {
        final = float4(final.x, final.y, final.z, final.w);
    }

    uint idx = y * uniforms.width + x;
    outPixels.Store4(idx * 16, asuint(saturate(final)));
}

// ---------------------------------------------------------------------------
// G-buffer stages (Renderer::renderGBuffer()), ports of shaders/vulkan/
// gbuffer_vertex.glsl and gbuffer.frag (themselves ports of default.metal's
// gbufferVertex/gbufferFragment): world normal + roughness, base colour +
// metallic and motion. The pipeline layout and bind group are the PBR
// pipeline's (the combined bind group), so the registers are pixelMain's.
//
// The previous frame's transform per draw is a per-instance vertex buffer at
// the draw's transform offset (Renderer's gbufferPreviousTransforms, the
// 128-byte NodeTransforms stride): previous clip from local at locations
// 20-23, this frame's projection jitter (clip x / w) at 28. Skinned draws
// (gbufferSkinnedVertex) place the previous position with the previous palette.
// ---------------------------------------------------------------------------
struct GBufferVertexIn {
    float3 position : TEXCOORD0;
    float3 normal   : TEXCOORD1;
    float2 uv0      : TEXCOORD2;
    float4 tangent  : TEXCOORD3;
    float4 color0   : TEXCOORD6;
    float2 uv1      : TEXCOORD7;
    float4 mvpCol0   : TEXCOORD16;
    float4 mvpCol1   : TEXCOORD17;
    float4 mvpCol2   : TEXCOORD18;
    float4 mvpCol3   : TEXCOORD19;
    float4 previousMvpCol0 : TEXCOORD20;
    float4 previousMvpCol1 : TEXCOORD21;
    float4 previousMvpCol2 : TEXCOORD22;
    float4 previousMvpCol3 : TEXCOORD23;
    float4 modelCol0 : TEXCOORD24;
    float4 modelCol1 : TEXCOORD25;
    float4 modelCol2 : TEXCOORD26;
    float4 modelCol3 : TEXCOORD27;
    float4 jitter    : TEXCOORD28;
    float4 instanceCol0 : TEXCOORD12;  // EXT_mesh_gpu_instancing (vertexMain)
    float4 instanceCol1 : TEXCOORD13;
    float4 instanceCol2 : TEXCOORD14;
    float4 instanceCol3 : TEXCOORD15;
};

struct GBufferPixelIn {
    float4 clipPosition   : SV_Position;
    float3 worldPos       : TEXCOORD0;
    float3 worldNormal    : TEXCOORD1;
    float3 worldTangent   : TEXCOORD2;
    float3 worldBitangent : TEXCOORD3;
    float2 uv0            : TEXCOORD4;
    float2 uv1            : TEXCOORD5;
    float4 color0         : TEXCOORD6;
    float4 currentClip    : TEXCOORD7;  // unjittered
    float4 previousClip   : TEXCOORD8;
};

struct GBufferOut {
    float4 normalRoughness : SV_Target0;  // xyz world normal (towards the viewer), w perceptual roughness
    float4 albedoMetallic  : SV_Target1;  // rgb base colour (linear), a metallic
    float2 motion          : SV_Target2;  // previous minus current image uv, y down
};

// `skin` / `previousSkin`: this frame's and the previous frame's instance x
// joints transform (the previous palette places the previous position, so a
// moving skin has motion); position and normal are morphed (this frame's
// weights for both, as on Metal).
GBufferPixelIn gbufferVertexCommon(GBufferVertexIn input, Columns skin, Columns previousSkin, float3 position,
                                   float3 normal) {
    GBufferPixelIn output;
    Columns mvp = columns(input.mvpCol0, input.mvpCol1, input.mvpCol2, input.mvpCol3);
    Columns previousMvp = columns(input.previousMvpCol0, input.previousMvpCol1, input.previousMvpCol2, input.previousMvpCol3);
    Columns model = columns(input.modelCol0, input.modelCol1, input.modelCol2, input.modelCol3);
    float4 localPos = transform(skin, float4(position, 1.0));
    float4 clip = transform(mvp, localPos);
    // Motion is the surface's, not the jitter's.
    output.currentClip  = clip - float4(input.jitter.xy * clip.w, 0.0, 0.0);
    output.previousClip = transform(previousMvp, transform(previousSkin, float4(position, 1.0)));
    output.clipPosition = clip;  // D3D clip +Y is up, as the CPU matrices (see vertexMain)

    output.worldPos = transform(model, localPos).xyz;
    float3 N = normalize(transform(model, transform(skin, float4(normal, 0.0))).xyz);
    float3 T;
    if (length(input.tangent.xyz) < 0.001) {
        T = abs(N.y) < 0.999 ? normalize(cross(float3(0.0, 1.0, 0.0), N)) : normalize(cross(float3(1.0, 0.0, 0.0), N));
    } else {
        T = normalize(transform(model, transform(skin, float4(input.tangent.xyz, 0.0))).xyz);
        T = normalize(T - dot(T, N) * N);
    }
    output.worldNormal    = N;
    output.worldTangent   = T;
    output.worldBitangent = cross(N, T);
    output.uv0    = input.uv0;
    output.uv1    = input.uv1;
    output.color0 = input.color0;
    return output;
}

Columns gbufferInstanceColumns(GBufferVertexIn input) {
    return columns(input.instanceCol0, input.instanceCol1, input.instanceCol2, input.instanceCol3);
}

GBufferPixelIn gbufferVertex(GBufferVertexIn input) {
    Columns instance = gbufferInstanceColumns(input);
    return gbufferVertexCommon(input, instance, instance, input.position, input.normal);
}

// Skinned and morphed draws (vertexSkinned): the previous palette places
// where the vertex was; it's this frame's when the draw has none.
GBufferPixelIn gbufferSkinnedVertex(GBufferVertexIn input, uint4 joints : TEXCOORD4, float4 weights : TEXCOORD5,
                                    uint vertexId : SV_VertexID) {
    float3 position = input.position;
    float3 normal = input.normal;
    applyMorph(position, normal, vertexId);
    Columns instance = gbufferInstanceColumns(input);
    Columns skin = compose(instance, blendJoints(jointMatrices, joints, weights));
    Columns previousSkin = compose(instance, blendJoints(previousJointMatrices, joints, weights));
    return gbufferVertexCommon(input, skin, previousSkin, position, normal);
}

// Position-only meshes (VertexLayout::positionOnly()): one 12-byte stream.
// A zero normal, which gbufferPixel/pixelMain replace with the flat face
// normal; same outputs as the full stages (default_lean.vert / gbuffer_lean.vert).
struct LeanVertexIn {
    float3 position  : TEXCOORD0;
    float4 mvpCol0   : TEXCOORD16;
    float4 mvpCol1   : TEXCOORD17;
    float4 mvpCol2   : TEXCOORD18;
    float4 mvpCol3   : TEXCOORD19;
    float4 modelCol0 : TEXCOORD24;
    float4 modelCol1 : TEXCOORD25;
    float4 modelCol2 : TEXCOORD26;
    float4 modelCol3 : TEXCOORD27;
};

PixelIn leanVertex(LeanVertexIn input) {
    PixelIn output;
    float3 p = input.position;
    output.clipPosition   = input.mvpCol0 * p.x + input.mvpCol1 * p.y + input.mvpCol2 * p.z + input.mvpCol3;
    output.worldPos       = (input.modelCol0 * p.x + input.modelCol1 * p.y + input.modelCol2 * p.z + input.modelCol3).xyz;
    output.worldNormal    = float3(0.0, 0.0, 0.0);
    output.worldTangent   = float3(0.0, 0.0, 0.0);
    output.worldBitangent = float3(0.0, 0.0, 0.0);
    output.uv0    = float2(0.0, 0.0);
    output.uv1    = float2(0.0, 0.0);
    output.color0 = float4(1.0, 1.0, 1.0, 1.0);
    return output;
}

// Line meshes (GpuMesh::topology lineList/lineStrip): leanVertex plus a small
// clip-space depth offset towards the camera, so edges drawn on their own
// faces win the depth test (depth bias doesn't apply to lines). As Metal's
// vertexLine and default_line.vert.
PixelIn lineVertex(LeanVertexIn input) {
    PixelIn output = leanVertex(input);
    output.clipPosition.z -= 4.0e-6 * output.clipPosition.w;  // kLineDepthOffset
    return output;
}

struct GBufferLeanVertexIn {
    float3 position  : TEXCOORD0;
    float4 mvpCol0   : TEXCOORD16;
    float4 mvpCol1   : TEXCOORD17;
    float4 mvpCol2   : TEXCOORD18;
    float4 mvpCol3   : TEXCOORD19;
    float4 previousMvpCol0 : TEXCOORD20;
    float4 previousMvpCol1 : TEXCOORD21;
    float4 previousMvpCol2 : TEXCOORD22;
    float4 previousMvpCol3 : TEXCOORD23;
    float4 modelCol0 : TEXCOORD24;
    float4 modelCol1 : TEXCOORD25;
    float4 modelCol2 : TEXCOORD26;
    float4 modelCol3 : TEXCOORD27;
    float4 jitter    : TEXCOORD28;
};

GBufferPixelIn gbufferLeanVertex(GBufferLeanVertexIn input) {
    GBufferPixelIn output;
    float3 p = input.position;
    float4 clip = input.mvpCol0 * p.x + input.mvpCol1 * p.y + input.mvpCol2 * p.z + input.mvpCol3;
    output.currentClip  = clip - float4(input.jitter.xy * clip.w, 0.0, 0.0);
    output.previousClip = input.previousMvpCol0 * p.x + input.previousMvpCol1 * p.y +
                          input.previousMvpCol2 * p.z + input.previousMvpCol3;
    output.clipPosition   = clip;
    output.worldPos       = (input.modelCol0 * p.x + input.modelCol1 * p.y + input.modelCol2 * p.z + input.modelCol3).xyz;
    output.worldNormal    = float3(0.0, 0.0, 0.0);
    output.worldTangent   = float3(0.0, 0.0, 0.0);
    output.worldBitangent = float3(0.0, 0.0, 0.0);
    output.uv0    = float2(0.0, 0.0);
    output.uv1    = float2(0.0, 0.0);
    output.color0 = float4(1.0, 1.0, 1.0, 1.0);
    return output;
}

GBufferOut gbufferPixel(GBufferPixelIn input, bool isFrontFace : SV_IsFrontFace) {
    float2 texcoord0 = input.uv0;
    float3 faceNormal = cross(ddx(input.worldPos), ddy(input.worldPos));
    float3 worldNormal = input.worldNormal;
    float3 worldTangent = input.worldTangent;
    float3 worldBitangent = input.worldBitangent;
    if (dot(input.worldNormal, input.worldNormal) < 1e-8) {
        float3 Nf = length(faceNormal) > 1e-20 ? normalize(faceNormal) : float3(0.0, 0.0, 1.0);
        if (dot(Nf, cameraPos.xyz - input.worldPos) < 0.0) Nf = -Nf;
        float3 Tf = abs(Nf.y) < 0.999 ? normalize(cross(float3(0.0, 1.0, 0.0), Nf)) : normalize(cross(float3(1.0, 0.0, 0.0), Nf));
        worldNormal = Nf;
        worldTangent = Tf;
        worldBitangent = cross(Nf, Tf);
    }
    if (worldMapping > 0.0) {
        float3 Tw, Bw;
        worldMappingFrame(input.worldPos, normalize(worldNormal), worldMapping, worldMappingUp > 0.5, texcoord0, Tw, Bw);
        worldTangent = Tw;
        worldBitangent = Bw;
    }

    // Every material sample before the discard (see pixelMain: implicit-LOD
    // samples after a discard picked wrong mips at triangle edges on AMD).
    float2 uv0 = texcoord0;
    if (uvTransformRow0.w > 0.5) uv0 = applyUvTransform(texcoord0, uvTransformRow0, uvTransformRow1);
    float2 baseColorUV = (uvTransformRow0.w > 0.5) ? uv0 : selectUV(uv0, input.uv1, texCoord1Mask, kUV1BaseColor);
    float4 baseColor = baseColorTexture.Sample(baseColorSampler, baseColorUV) * baseColorFactor * input.color0;
    float2 mrUV = selectUV(uv0, input.uv1, texCoord1Mask, kUV1MetallicRoughness);
    float4 mr = metallicRoughnessTexture.Sample(metallicRoughnessSampler, mrUV);
    float2 normalUV = (normalUvTransformRow0.w > 0.5)
        ? applyUvTransform(texcoord0, normalUvTransformRow0, normalUvTransformRow1)
        : selectUV(texcoord0, input.uv1, texCoord1Mask, kUV1Normal);
    float3 ns = normalTexture.Sample(normalSampler, normalUV).rgb * 2.0 - 1.0;

    if (alphaMode > 0.5 && alphaMode < 1.5 && baseColor.a < alphaCutoff) discard;
    for (uint ci = 0u; ci < clipInfo.x; ++ci) {
        if (dot(clipPlanes[ci].xyz, input.worldPos) + clipPlanes[ci].w < 0.0) discard;
    }

    float metallic = 0.0;
    float roughness = 1.0;
    float3 N = normalize(worldNormal);
    if (unlit < 0.5) {
        metallic = mr.b * metallicFactor;
        roughness = clamp(mr.g * roughnessFactor, 0.04, 1.0);
        if (hasNormalTexture > 0.5) {
            ns.xy *= normalScale;
            float3 nsN = normalize(ns);
            // TBN * ns as a column combination (see the header comment).
            N = normalize(nsN.x * normalize(worldTangent) + nsN.y * normalize(worldBitangent) + nsN.z * N);
        }
    }
    // Towards the viewer (back faces of two-sided surfaces).
    if (dot(N, cameraPos.xyz - input.worldPos) < 0.0 && !isFrontFace) N = -N;

    GBufferOut output;
    output.normalRoughness = float4(N, roughness);
    output.albedoMetallic = float4(clamp(baseColor.rgb, 0.0, 1.0), metallic);
    // D3D clip +Y is up (as the Metal clips, before Vulkan's flip); image y is down.
    float2 current = input.currentClip.xy / input.currentClip.w;
    float2 previous = input.previousClip.xy / input.previousClip.w;
    output.motion = (previous - current) * float2(0.5, -0.5);
    return output;
}

// ---------------------------------------------------------------------------
// TAA resolve and the built-in temporal upscaler, a port of shaders/vulkan/
// taa.frag (default.metal's taaFragment; see it for the method), drawn with
// fxaaVertex's fullscreen triangle. Writes the output and the next history.
// ---------------------------------------------------------------------------
Texture2D<float4> taaCurrent : register(t0);
Texture2D<float4> taaHistory : register(t1);
Texture2D<float4> taaMotion  : register(t2);
SamplerState      taaSampler : register(s3);
cbuffer TaaUniforms : register(b4) {
    float2 taaInputSize;     // the frame's (internal) size in pixels
    float2 taaJitter;        // how far this frame's jitter moved the geometry, in input pixels (x right, y down)
    float  taaBlend;         // the frame's weight where a sample is at the pixel
    float  taaHistoryValid;  // 0: no history (first frame, cut, resize)
    float2 taaOutputSize;    // the output's size in pixels
};

struct TaaOut {
    float4 color   : SV_Target0;
    float4 history : SV_Target1;
};

float3 rgbToYCoCg(float3 c) {
    return float3(dot(c, float3(0.25, 0.5, 0.25)), dot(c, float3(0.5, 0.0, -0.5)), dot(c, float3(-0.25, 0.5, -0.25)));
}
float3 yCoCgToRgb(float3 c) {
    return float3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

TaaOut taaPixel(FxaaOut input) {
    float2 fragUV = input.uv;  // (0, 0) = top-left
    float2 p = fragUV * taaInputSize;
    int2 centre = int2(floor(p));
    int2 limit = int2(taaInputSize) - 1;
    float3 sum = float3(0.0, 0.0, 0.0), m1 = float3(0.0, 0.0, 0.0), m2 = float3(0.0, 0.0, 0.0);
    float weights = 0.0, nearest = 0.0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int2 q = clamp(centre + int2(dx, dy), int2(0, 0), limit);
            float3 c = taaCurrent.Load(int3(q, 0)).rgb;
            // Where its sample was in the unjittered image, in output pixels.
            float2 d = (float2(q) + 0.5 - taaJitter - p) * (taaOutputSize / taaInputSize);
            float w = exp(-2.29 * dot(d, d));
            sum += w * c;
            weights += w;
            nearest = max(nearest, w);
            float3 t = rgbToYCoCg(c);
            m1 += t;
            m2 += t * t;
        }
    }
    float3 c = weights > 1e-6 ? sum / weights : taaCurrent.Load(int3(clamp(centre, int2(0, 0), limit), 0)).rgb;
    float3 mean = m1 / 9.0;
    float3 sigma = sqrt(max(m2 / 9.0 - mean * mean, 0.0));
    uint mw, mh;
    taaMotion.GetDimensions(mw, mh);
    int2 motionSize = int2(mw, mh);
    int2 motionPixel = min(int2(fragUV * float2(motionSize)), motionSize - 1);
    float2 previousUV = fragUV + taaMotion.Load(int3(motionPixel, 0)).xy;
    float3 result = c;
    if (taaHistoryValid > 0.5 && all(previousUV >= 0.0) && all(previousUV <= 1.0)) {
        float3 h = clamp(rgbToYCoCg(taaHistory.SampleLevel(taaSampler, previousUV, 0.0).rgb),
                         mean - 1.5 * sigma, mean + 1.5 * sigma);
        // Less of this frame where its nearest sample is far from the pixel.
        result = lerp(yCoCgToRgb(h), c, clamp(taaBlend * nearest * 2.0, 0.0, 1.0));
    }
    TaaOut output;
    output.color = float4(result, 1.0);
    output.history = float4(result, 1.0);
    return output;
}

// ---------------------------------------------------------------------------
// Screen-space ambient occlusion (setAmbientOcclusion()), ports of
// shaders/vulkan/ssao.frag and ssao_apply.frag (Metal's ssaoFragment and
// ssaoApplyFragment; see default.metal for the method), drawn with
// fxaaVertex's fullscreen triangle. Pass 1: scene depth -> AO at the output
// size. Pass 2: the AO blurred 4x4 (depth-aware), multiplied into the frame
// by the blend state. Registers follow aoBindGroupLayout's bindings.
// ---------------------------------------------------------------------------
Texture2D<float>  aoDepthTexture : register(t0);
Texture2D<float4> aoTexture      : register(t1);
cbuffer AoUniforms : register(b2) {
    float4x4 aoProj;     // view -> clip
    float4x4 aoInvProj;  // clip -> view
    float4   aoParams;   // x = radius (view units), y = intensity (power), z = bias, w = samples
    float4   aoSize;     // depth texture width, height; output width, height
};

float aoDepthAt(float2 uv) {
    int2 texel = int2(clamp(uv * aoSize.xy, float2(0.0, 0.0), aoSize.xy - 1.0));
    return aoDepthTexture.Load(int3(texel, 0));
}

float3 aoViewPos(float2 uv, float depth) {
    float4 p = mul(aoInvProj, float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0));
    return p.xyz / p.w;
}

float4 ssaoPixel(FxaaOut input) : SV_Target {
    float2 fragUV = input.uv;
    float depth = aoDepthAt(fragUV);
    if (depth >= 1.0) return float4(1.0, 1.0, 1.0, 1.0);
    float3 P = aoViewPos(fragUV, depth);
    float2 px = 1.0 / aoSize.zw;
    float3 R = aoViewPos(fragUV + float2(px.x, 0.0), aoDepthAt(fragUV + float2(px.x, 0.0)));
    float3 L = aoViewPos(fragUV - float2(px.x, 0.0), aoDepthAt(fragUV - float2(px.x, 0.0)));
    float3 D = aoViewPos(fragUV + float2(0.0, px.y), aoDepthAt(fragUV + float2(0.0, px.y)));
    float3 U = aoViewPos(fragUV - float2(0.0, px.y), aoDepthAt(fragUV - float2(0.0, px.y)));
    float3 dx = abs(R.z - P.z) < abs(P.z - L.z) ? R - P : P - L;
    float3 dy = abs(D.z - P.z) < abs(P.z - U.z) ? D - P : P - U;
    float3 N = normalize(cross(dx, dy));
    if (dot(N, -P) < 0.0) N = -N;

    float radius = aoParams.x;
    float bias = aoParams.z;
    int samples = int(aoParams.w);
    float noise = frac(52.9829189 * frac(dot(input.position.xy, float2(0.06711056, 0.00583715))));
    float angle = noise * 6.28318530718;
    float3 helper = abs(N.y) < 0.99 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    float3 T = normalize(cross(helper, N));
    float3 B = cross(N, T);
    float3 Tr = T * cos(angle) + B * sin(angle);
    float3 Br = cross(N, Tr);

    float occlusion = 0.0;
    for (int i = 0; i < samples; ++i) {
        float t = (float(i) + 0.5) / float(samples);
        float phi = float(i) * 2.39996323;
        float cosTheta = sqrt(1.0 - t * 0.9);
        float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
        float scale = lerp(0.15, 1.0, t * t);
        float3 k = float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta) * scale;
        float3 S = P + (Tr * k.x + Br * k.y + N * k.z) * radius;
        float4 c = mul(aoProj, float4(S, 1.0));
        float2 suv = float2(c.x / c.w * 0.5 + 0.5, 0.5 - c.y / c.w * 0.5);
        if (any(suv < 0.0) || any(suv > 1.0)) continue;
        float sceneDepth = aoDepthAt(suv);
        if (sceneDepth >= 1.0) continue;
        float3 Q = aoViewPos(suv, sceneDepth);
        float range = smoothstep(0.0, 1.0, radius / max(abs(P.z - Q.z), 1e-4));
        occlusion += (abs(Q.z) <= abs(S.z) - bias ? 1.0 : 0.0) * range;
    }
    float ao = pow(clamp(1.0 - occlusion / float(max(samples, 1)), 0.0, 1.0), aoParams.y);
    return float4(ao, ao, ao, 1.0);
}

float4 ssaoApplyPixel(FxaaOut input) : SV_Target {
    float2 fragUV = input.uv;
    float depth = aoDepthAt(fragUV);
    if (depth >= 1.0) return float4(1.0, 1.0, 1.0, 1.0);
    float z = aoViewPos(fragUV, depth).z;
    int2 centre = int2(fragUV * aoSize.zw);
    int2 limit = int2(aoSize.zw) - 1;
    float2 px = 1.0 / aoSize.zw;
    float sum = 0.0;
    float weight = 0.0;
    for (int y = -2; y < 2; ++y) {
        for (int x = -2; x < 2; ++x) {
            int2 at = clamp(centre + int2(x, y), int2(0, 0), limit);
            float2 uv = (float2(at) + 0.5) * px;
            float d = aoDepthAt(uv);
            float sz = aoViewPos(uv, d).z;
            float w = (d < 1.0 && abs(sz - z) < 0.05 * abs(z) + aoParams.z) ? 1.0 : 0.0;
            sum += aoTexture.Load(int3(at, 0)).r * w;
            weight += w;
        }
    }
    float ao = weight > 0.0 ? sum / weight : 1.0;
    return float4(ao, ao, ao, 1.0);
}
