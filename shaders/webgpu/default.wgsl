// campello_renderer built-in PBR mesh shaders, WGSL (WebGPU).
//
// Hand-written port of the mesh shaders in shaders/metal/default.metal,
// limited to WebGPU's default 16 sampled textures per shader stage (15 are
// used): glTF core textures (base colour, metallic-roughness, normal,
// emissive, occlusion), image-based lighting, and KHR_materials_transmission,
// _volume (screen-space refraction, attenuation, dispersion) and
// _iridescence with their textures. Not on WebGPU yet: the specular, sheen,
// clearcoat and anisotropy extensions and the flat/debug variants
// (programmable pipelines phase 7).
//
// Entry points: vs_main (glTF layout), vs_skinned (glTF layout with
// JOINTS_0/WEIGHTS_0: skinning and morph targets, group 3), vs_instanced
// (glTF layout with per-instance matrices, its transform in group 2),
// vs_lean_instanced (the same for position-only meshes), vs_lean
// (positions only), vs_line (line meshes), fs_main. Bindings follow the Vulkan layout
// (ensureVulkanPbrBindGroupLayouts(), trimmed on the web): group 0 =
// material, group 1 = frame, group 2 = per-draw transforms. WebGPU allows
// vertex input locations 0-15 only, so instead of the per-instance
// attributes other backends use (locations 16-19, 24-27), the transforms
// are a storage buffer indexed by instance_index: each draw passes its
// transform's index (byte offset / 128) as firstInstance. WebGPU's clip
// space has Y up and Z in [0, 1], like Metal: no flip.
//
// WGSL requires implicit-derivative sampling (textureSample) in uniform
// control flow: branches before it only test uniforms (material, camera),
// and the section cap (front_facing, per fragment) is applied at the end.

struct MaterialUniforms {
    baseColorFactor: vec4f,
    uvTransformRow0: vec4f,
    uvTransformRow1: vec4f,
    metallicFactor: f32,
    roughnessFactor: f32,
    normalScale: f32,
    alphaMode: f32,
    alphaCutoff: f32,
    unlit: f32,
    hasNormalTexture: f32,
    hasEmissiveTexture: f32,
    hasOcclusionTexture: f32,
    occlusionStrength: f32,
    doubleSided: f32,
    emissiveFactor: vec4f,
    ior: f32,
    specularFactor: f32,
    hasSpecularTexture: f32,
    hasSpecularColorTexture: f32,
    pad2: f32,
    specularColorFactor: vec4f,
    pad3: f32,
    sheenColorFactor: vec4f,
    sheenRoughnessFactor: f32,
    hasSheenColorTexture: f32,
    hasSheenRoughnessTexture: f32,
    clearcoatFactor: f32,
    clearcoatRoughnessFactor: f32,
    hasClearcoatTexture: f32,
    hasClearcoatRoughnessTexture: f32,
    hasClearcoatNormalTexture: f32,
    clearcoatNormalScale: f32,
    transmissionFactor: f32,
    hasTransmissionTexture: f32,
    thicknessFactor: f32,
    attenuationDistance: f32,
    hasThicknessTexture: f32,
    padVol: f32,
    attenuationColor: vec4f,
    viewMode: f32,
    environmentIntensity: f32,
    iblEnabled: f32,
    iridescenceFactor: f32,
    iridescenceIor: f32,
    iridescenceThicknessMin: f32,
    iridescenceThicknessMax: f32,
    hasIridescenceTexture: f32,
    hasIridescenceThicknessTexture: f32,
    anisotropyStrength: f32,
    anisotropyRotation: f32,
    hasAnisotropicTexture: f32,
    dispersion: f32,
    worldMapping: f32,     // world-space box mapping: world units per repeat (0: off)
    worldMappingUp: f32,   // 1: z up, 0: y up
    normalUvTransformRow0: vec4f,
    normalUvTransformRow1: vec4f,
    texCoord1Mask: f32,
    // KHR_texture_transform of these textures, each its own (offset 384).
    metallicRoughnessUvTransformRow0: vec4f,
    metallicRoughnessUvTransformRow1: vec4f,
    occlusionUvTransformRow0: vec4f,
    occlusionUvTransformRow1: vec4f,
    emissiveUvTransformRow0: vec4f,
    emissiveUvTransformRow1: vec4f,
}

struct CameraUniforms {
    cameraPos: vec4f,
    viewMatrix: mat4x4f,
    projMatrix: mat4x4f,
    screenSize: vec2f,
    pad0: vec2f,
    clipInfo: vec4u,           // x = plane count, y = section caps on
    capColor: vec4f,
    clipPlanes: array<vec4f, 6>, // xyz = unit normal, w = d: kept where dot(n, p) + d >= 0
    shadowViewProj: mat4x4f,     // sun shadows: shadow map clip from world
    shadowParams: vec4f,         // x = on, y = texel size (uv)
    viewParams: vec4f,           // x = exposure in EV (colour x 2^EV before tone mapping)
    shadowNearViewProj: mat4x4f, // sun shadows, near cascade
    shadowNearParams: vec4f,     // x = near cascade on
}

struct Light {
    position: vec4f,   // xyz = position/dir, w = type (0 = dir, 1 = point, 2 = spot)
    color: vec4f,      // xyz = rgb, w = intensity
    direction: vec4f,  // xyz = spot dir, w = range
    spotAngles: vec4f, // x = inner cone, y = outer cone, z = casts the shadow map
}

struct LightsUniform {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    lights: array<Light, 8>,
}

@group(0) @binding(0)  var baseColorTexture: texture_2d<f32>;
@group(0) @binding(1)  var baseColorSampler: sampler;
@group(0) @binding(2)  var metallicRoughnessTexture: texture_2d<f32>;
@group(0) @binding(3)  var metallicRoughnessSampler: sampler;
@group(0) @binding(4)  var normalTexture: texture_2d<f32>;
@group(0) @binding(5)  var normalSampler: sampler;
@group(0) @binding(6)  var emissiveTexture: texture_2d<f32>;
@group(0) @binding(7)  var emissiveSampler: sampler;
@group(0) @binding(8)  var occlusionTexture: texture_2d<f32>;
@group(0) @binding(9)  var occlusionSampler: sampler;
// KHR extension textures, sampled with baseColorSampler (as on Vulkan).
@group(0) @binding(19) var transmissionTexture: texture_2d<f32>;          // R = transmission
@group(0) @binding(20) var thicknessTexture: texture_2d<f32>;             // G = thickness
@group(0) @binding(21) var iridescenceTexture: texture_2d<f32>;           // R = iridescence
@group(0) @binding(22) var iridescenceThicknessTexture: texture_2d<f32>;  // G = film thickness
@group(0) @binding(24) var<uniform> mat: MaterialUniforms;

@group(1) @binding(0)  var<uniform> lights: LightsUniform;
@group(1) @binding(1)  var<uniform> camera: CameraUniforms;
@group(1) @binding(2)  var environmentMap: texture_cube<f32>;   // GGX-prefiltered specular
@group(1) @binding(3)  var environmentSampler: sampler;
@group(1) @binding(4)  var sceneColorTexture: texture_2d<f32>;  // the opaque scene (transmission)
@group(1) @binding(5)  var sceneColorSampler: sampler;
@group(1) @binding(6)  var irradianceMap: texture_cube<f32>;
@group(1) @binding(7)  var brdfLutTexture: texture_2d<f32>;
@group(1) @binding(8)  var brdfLutSampler: sampler;
@group(1) @binding(9)  var shadowMap: texture_depth_2d;          // sun shadows
@group(1) @binding(10) var shadowSampler: sampler_comparison;
@group(1) @binding(11) var shadowMapNear: texture_depth_2d;      // sun shadows, near cascade

struct NodeTransforms {
    mvp: mat4x4f,    // clip from local
    model: mat4x4f,  // world from local
}
@group(2) @binding(0) var<storage, read> transforms: array<NodeTransforms>;

const PI: f32 = 3.14159265358979323846;
const kUV1BaseColor: u32 = 1u;
const kUV1MetallicRoughness: u32 = 2u;
const kUV1Normal: u32 = 4u;
const kUV1Emissive: u32 = 8u;
const kUV1Occlusion: u32 = 16u;
const kUV1Transmission: u32 = 1u << 12u;
const kUV1Thickness: u32 = 1u << 13u;
const kUV1Iridescence: u32 = 1u << 14u;
const kUV1IridescenceThickness: u32 = 1u << 15u;
// Line meshes: clip-space depth offset towards the camera (Metal vertexLine).
const kLineDepthOffset: f32 = 4.0e-6;

const VIEW_MODE_WORLD_NORMAL: f32 = 1.0;
const VIEW_MODE_BASE_COLOR: f32 = 2.0;
const VIEW_MODE_METALLIC: f32 = 3.0;
const VIEW_MODE_ROUGHNESS: f32 = 4.0;
const VIEW_MODE_OCCLUSION: f32 = 5.0;
const VIEW_MODE_EMISSIVE: f32 = 6.0;
const VIEW_MODE_ALPHA: f32 = 7.0;
const VIEW_MODE_UV0: f32 = 8.0;

// ---------------------------------------------------------------------------
// Vertex stage
// ---------------------------------------------------------------------------
struct VertexIn {
    @location(0)  position: vec3f,
    @location(1)  normal: vec3f,
    @location(2)  texcoord0: vec2f,
    @location(3)  tangent: vec4f,
    @location(6)  color0: vec4f,
    @location(7)  texcoord1: vec2f,
}

struct VertexOut {
    @builtin(position) clipPosition: vec4f,
    @location(0) worldPos: vec3f,
    @location(1) worldNormal: vec3f,
    @location(2) worldTangent: vec3f,
    @location(3) worldBitangent: vec3f,
    @location(4) texcoord0: vec2f,
    @location(5) texcoord1: vec2f,
    @location(6) color0: vec4f,
}

// The PBR vertex stage shared by vs_main, vs_skinned and vs_instanced:
// `position`, `normal` and `tangent` are already in the mesh's local space
// (morphed and skinned, or placed by the instance).
fn pbrVertex(t: NodeTransforms, position: vec3f, normal: vec3f, tangent: vec4f, texcoord0: vec2f, texcoord1: vec2f,
             color0: vec4f) -> VertexOut {
    let localPos = vec4f(position, 1.0);
    var o: VertexOut;
    o.clipPosition = t.mvp * localPos;
    o.worldPos = (t.model * localPos).xyz;
    let N = normalize((t.model * vec4f(normal, 0.0)).xyz);
    var T: vec3f;
    if (length(tangent.xyz) < 0.001) {
        if (abs(N.y) < 0.999) {
            T = normalize(cross(vec3f(0.0, 1.0, 0.0), N));
        } else {
            T = normalize(cross(vec3f(1.0, 0.0, 0.0), N));
        }
    } else {
        T = normalize((t.model * vec4f(tangent.xyz, 0.0)).xyz);
        T = normalize(T - dot(T, N) * N);
    }
    o.worldNormal = N;
    o.worldTangent = T;
    o.worldBitangent = cross(N, T);
    o.texcoord0 = texcoord0;
    o.texcoord1 = texcoord1;
    o.color0 = color0;
    return o;
}

@vertex
fn vs_main(v: VertexIn, @builtin(instance_index) instance: u32) -> VertexOut {
    return pbrVertex(transforms[instance], v.position, v.normal, v.tangent, v.texcoord0, v.texcoord1, v.color0);
}

// ---------------------------------------------------------------------------
// Instanced draws: vs_instanced (EXT_mesh_gpu_instancing)
// ---------------------------------------------------------------------------
// Other draws find their transform through firstInstance (transforms[
// instance_index]), which an instanced draw needs for the instance index.
// So its pipeline layout's group 2 holds the draw's own transform as a
// uniform (binding 1; the renderer copies it to a 256-byte slot), the draw
// starts at instance 0, and the instance matrices are a per-instance
// vertex buffer (locations 8-11), applied in model space as on Vulkan.
@group(2) @binding(1) var<uniform> drawTransform: NodeTransforms;

struct InstancedVertexIn {
    @location(0)  position: vec3f,
    @location(1)  normal: vec3f,
    @location(2)  texcoord0: vec2f,
    @location(3)  tangent: vec4f,
    @location(6)  color0: vec4f,
    @location(7)  texcoord1: vec2f,
    @location(8)  instance0: vec4f,   // the instance's float4x4, column by column
    @location(9)  instance1: vec4f,
    @location(10) instance2: vec4f,
    @location(11) instance3: vec4f,
}

@vertex
fn vs_instanced(v: InstancedVertexIn) -> VertexOut {
    let m = mat4x4f(v.instance0, v.instance1, v.instance2, v.instance3);
    let position = (m * vec4f(v.position, 1.0)).xyz;
    let normal = (m * vec4f(v.normal, 0.0)).xyz;
    let tangent = vec4f((m * vec4f(v.tangent.xyz, 0.0)).xyz, v.tangent.w);
    return pbrVertex(drawTransform, position, normal, tangent, v.texcoord0, v.texcoord1, v.color0);
}

// ---------------------------------------------------------------------------
// Skinned and morphed draws: vs_skinned (shaders/vulkan/skinning.glsl's port)
// ---------------------------------------------------------------------------
// The renderer picks vs_skinned only for draws with a skin or morph
// targets, with a pipeline layout that has group 3 (WebGPU's default limit
// is four groups, and group 2 holds the transforms, so the joint palette
// and the morph targets share one group, unlike Vulkan's sets 2 and 3):
//   binding 0: the draw's joint palette, a mat4 per joint (column-major);
//   binding 1: the node's MorphInfo (target count, whether targets have
//              normals, vertex count, up to 8 weights);
//   bindings 2 and 3: the primitive's POSITION and NORMAL deltas,
//              target-major and tightly packed (3 floats per vertex; target
//              t's delta for vertex v at t * vertexCount + v).
// Draws without a skin get a one-joint identity palette, without morph
// targets a MorphInfo with zero targets.
struct MorphInfo {
    targetCount: f32,
    hasNormal: f32,
    vertexCount: f32,
    weights: array<f32, 8>,
}
@group(3) @binding(0) var<storage, read> jointMatrices: array<mat4x4f>;
@group(3) @binding(1) var<storage, read> morphInfo: MorphInfo;
@group(3) @binding(2) var<storage, read> morphPositions: array<f32>;
@group(3) @binding(3) var<storage, read> morphNormals: array<f32>;

struct SkinnedVertexIn {
    @location(0)  position: vec3f,
    @location(1)  normal: vec3f,
    @location(2)  texcoord0: vec2f,
    @location(3)  tangent: vec4f,
    @location(4)  joints: vec4u,    // JOINTS_0 (uint8x4 or uint16x4)
    @location(5)  weights: vec4f,   // WEIGHTS_0 (float32x4, unorm8x4 or unorm16x4)
    @location(6)  color0: vec4f,
    @location(7)  texcoord1: vec2f,
}

// Up to four joints blended by weight; weights summing to zero leave the
// vertex unskinned (as Metal and Vulkan do).
fn skinMatrix(j: vec4u, w: vec4f) -> mat4x4f {
    if (w.x + w.y + w.z + w.w <= 0.001) {
        return mat4x4f(vec4f(1.0, 0.0, 0.0, 0.0), vec4f(0.0, 1.0, 0.0, 0.0),
                       vec4f(0.0, 0.0, 1.0, 0.0), vec4f(0.0, 0.0, 0.0, 1.0));
    }
    let count = arrayLength(&jointMatrices);
    var m = mat4x4f(vec4f(0.0), vec4f(0.0), vec4f(0.0), vec4f(0.0));
    for (var i = 0u; i < 4u; i++) {
        if (w[i] > 0.0 && j[i] < count) {
            m += w[i] * jointMatrices[j[i]];
        }
    }
    return m;
}

@vertex
fn vs_skinned(v: SkinnedVertexIn, @builtin(vertex_index) vertex: u32,
              @builtin(instance_index) instance: u32) -> VertexOut {
    // glTF's order: morph targets, then the joints.
    var position = v.position;
    var normal = v.normal;
    let targets = min(u32(morphInfo.targetCount), 8u);
    let vertexCount = u32(morphInfo.vertexCount);
    let hasNormal = morphInfo.hasNormal > 0.5;
    for (var t = 0u; t < targets; t++) {
        let w = morphInfo.weights[t];
        if (w == 0.0) { continue; }
        let i = (t * vertexCount + vertex) * 3u;
        position += w * vec3f(morphPositions[i], morphPositions[i + 1u], morphPositions[i + 2u]);
        if (hasNormal) {
            normal += w * vec3f(morphNormals[i], morphNormals[i + 1u], morphNormals[i + 2u]);
        }
    }
    let skin = skinMatrix(v.joints, v.weights);
    let skinnedPos = (skin * vec4f(position, 1.0)).xyz;
    let skinnedNormal = (skin * vec4f(normal, 0.0)).xyz;
    let skinnedTangent = vec4f((skin * vec4f(v.tangent.xyz, 0.0)).xyz, v.tangent.w);
    return pbrVertex(transforms[instance], skinnedPos, skinnedNormal, skinnedTangent, v.texcoord0, v.texcoord1, v.color0);
}

fn positionOnly(t: NodeTransforms, position: vec3f) -> VertexOut {
    let localPos = vec4f(position, 1.0);
    var o: VertexOut;
    o.clipPosition = t.mvp * localPos;
    o.worldPos = (t.model * localPos).xyz;
    o.worldNormal = vec3f(0.0);   // fs_main derives the flat face normal
    o.worldTangent = vec3f(0.0);
    o.worldBitangent = vec3f(0.0);
    o.texcoord0 = vec2f(0.0);
    o.texcoord1 = vec2f(0.0);
    o.color0 = vec4f(1.0);
    return o;
}

// Position-only meshes (VertexLayout::positionOnly()).
@vertex
fn vs_lean(@location(0) position: vec3f, @builtin(instance_index) instance: u32) -> VertexOut {
    return positionOnly(transforms[instance], position);
}

// Position-only meshes drawn with instances (vs_instanced's lean variant:
// the draw's transform in group 2, the instance's matrix at locations 8-11).
@vertex
fn vs_lean_instanced(@location(0) position: vec3f, @location(8) instance0: vec4f, @location(9) instance1: vec4f,
                     @location(10) instance2: vec4f, @location(11) instance3: vec4f) -> VertexOut {
    let m = mat4x4f(instance0, instance1, instance2, instance3);
    return positionOnly(drawTransform, (m * vec4f(position, 1.0)).xyz);
}

// Line meshes: nudged towards the camera so edges win over their faces.
@vertex
fn vs_line(@location(0) position: vec3f, @builtin(instance_index) instance: u32) -> VertexOut {
    var o = positionOnly(transforms[instance], position);
    o.clipPosition.z -= kLineDepthOffset * o.clipPosition.w;
    return o;
}

// ---------------------------------------------------------------------------
// Fragment stage
// ---------------------------------------------------------------------------

// One shadow map's verdict on worldPos: 1 lit .. 0 shadowed, 3x3
// percentage-closer filtering; .y = 1 where the map covers the point.
// Receiver bias as shaders/metal/default.metal's shadowLookup().
// textureSampleCompareLevel: no derivatives, so fine in the light loop.
fn shadowLookup(map: texture_depth_2d, M: mat4x4f, texel: f32, worldPos: vec3f, N: vec3f, L: vec3f) -> vec2f {
    let worldTexel = 2.0 * texel / max(length(vec3f(M[0][0], M[1][0], M[2][0])), 1e-6);
    let depthPerWorld = length(vec3f(M[0][2], M[1][2], M[2][2]));
    let NdotL = clamp(dot(N, L), 0.0, 1.0);
    let slope = sqrt(1.0 - NdotL * NdotL) / max(NdotL, 0.2);
    let clip = M * vec4f(worldPos + N * worldTexel * (1.0 + slope), 1.0);
    let ndc = clip.xyz / clip.w;
    let uv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    let inside = all(uv >= vec2f(2.0 * texel)) && all(uv <= vec2f(1.0 - 2.0 * texel)) && ndc.z <= 1.0;
    if (!inside) {
        return vec2f(1.0, 0.0);
    }
    let depth = ndc.z - depthPerWorld * worldTexel;
    var lit = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            lit += textureSampleCompareLevel(map, shadowSampler, uv + vec2f(f32(x), f32(y)) * texel, depth);
        }
    }
    return vec2f(lit / 9.0, 1.0);
}

// How lit worldPos is by the shadow-casting light: the near cascade where it
// covers the point, else the far one (the whole scene); outside both, lit.
fn sunShadow(worldPos: vec3f, N: vec3f, L: vec3f) -> f32 {
    if (camera.shadowParams.x < 0.5) {
        return 1.0;
    }
    if (camera.shadowNearParams.x > 0.5) {
        let near = shadowLookup(shadowMapNear, camera.shadowNearViewProj, camera.shadowParams.y, worldPos, N, L);
        if (near.y > 0.5) {
            return near.x;
        }
    }
    return shadowLookup(shadowMap, camera.shadowViewProj, camera.shadowParams.y, worldPos, N, L).x;
}

fn selectUV(uv0: vec2f, uv1: vec2f, packedMask: f32, bit: u32) -> vec2f {
    return select(uv0, uv1, (u32(packedMask) & bit) != 0u);
}

fn transformUV(uv: vec2f, row0: vec4f, row1: vec4f) -> vec2f {
    let uv3 = vec3f(uv, 1.0);
    return vec2f(dot(row0.xyz, uv3), dot(row1.xyz, uv3));
}

// A texture's coordinates: its UV set (TEXCOORD_0 or _1, by its bit in the
// mask), then its own KHR_texture_transform when it has one (row0.w).
fn textureUV(uv0: vec2f, uv1: vec2f, packedMask: f32, bit: u32, row0: vec4f, row1: vec4f) -> vec2f {
    let uv = selectUV(uv0, uv1, packedMask, bit);
    return select(uv, transformUV(uv, row0, row1), row0.w > 0.5);
}

// Khronos PBR Neutral tone mapping (as the Metal shader).
fn toneMapKhronosPbrNeutral(colorIn: vec3f) -> vec3f {
    let startCompression = 0.8 - 0.04;
    let desaturation = 0.15;
    var color = colorIn;
    let x = min(color.r, min(color.g, color.b));
    let offset = select(0.04, x - 6.25 * x * x, x < 0.08);
    color -= offset;
    let peak = max(color.r, max(color.g, color.b));
    if (peak < startCompression) {
        return color;
    }
    let d = 1.0 - startCompression;
    let newPeak = 1.0 - d * d / (peak + d - startCompression);
    color *= newPeak / peak;
    let g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
    return mix(color, vec3f(newPeak), g);
}

fn linearToSRGBFast(color: vec3f) -> vec3f {
    return pow(max(color, vec3f(0.0)), vec3f(1.0 / 2.2));
}

fn D_GGX(roughness: f32, NdotH: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d + 0.0001);
}

fn V_SmithGGX(NdotL: f32, NdotV: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let GL = NdotV * sqrt(NdotL * NdotL * (1.0 - a) + a);
    let GV = NdotL * sqrt(NdotV * NdotV * (1.0 - a) + a);
    return 0.5 / max(GL + GV, 0.0001);
}

// Roughness-aware IBL Fresnel with multi-scattering compensation (as Metal).
fn iblGGXFresnel(NdotV: f32, roughness: f32, F0: vec3f) -> vec3f {
    let fAB = textureSampleLevel(brdfLutTexture, brdfLutSampler, clamp(vec2f(NdotV, roughness), vec2f(0.0), vec2f(1.0)), 0.0).rg;
    let Fr = max(vec3f(1.0 - roughness), F0) - F0;
    let kS = F0 + Fr * pow(clamp(1.0 - NdotV, 0.0, 1.0), 5.0);
    let FssEss = kS * fAB.x + fAB.y;
    let Ems = 1.0 - (fAB.x + fAB.y);
    let Favg = F0 + (vec3f(1.0) - F0) / 21.0;
    let FmsEms = Ems * FssEss * Favg / (vec3f(1.0) - Favg * Ems);
    return FssEss + FmsEms;
}

// World-space box mapping (Renderer::setMaterialWorldMapping()), for
// geometry without texture coordinates (CAD, BIM): faces facing up or down
// take the plan projection (u east, v south), the others a projection along
// their own horizontal direction (u to the right seen from outside, v down),
// one repeat per `size` world units, so textures run continuously across
// elements. T and B are the directions u and v grow in (normal mapping).
// KHR_materials_iridescence (thin-film interference), as in
// shaders/vulkan/default.frag and the glTF-Sample-Renderer.
const kXyzToRec709 = mat3x3f(
    vec3f(3.2404542, -0.9692660, 0.0556434),
    vec3f(-1.5371385, 1.8760108, -0.2040259),
    vec3f(-0.4985314, 0.0415560, 1.0572252));

fn fresnel0ToIor(fresnel0: vec3f) -> vec3f {
    let sqrtF0 = sqrt(clamp(fresnel0, vec3f(0.0), vec3f(0.9999)));
    return (vec3f(1.0) + sqrtF0) / (vec3f(1.0) - sqrtF0);
}

fn fresnelDielectric(cosTheta1: f32, n1: f32, n2: f32) -> f32 {
    let sinTheta2Sq = (n1 * n1) / (n2 * n2) * (1.0 - cosTheta1 * cosTheta1);
    if (sinTheta2Sq > 1.0) {
        return 1.0;
    }
    let cosTheta2 = sqrt(1.0 - sinTheta2Sq);
    let rs = (n1 * cosTheta1 - n2 * cosTheta2) / (n1 * cosTheta1 + n2 * cosTheta2);
    let rp = (n2 * cosTheta1 - n1 * cosTheta2) / (n2 * cosTheta1 + n1 * cosTheta2);
    return 0.5 * (rs * rs + rp * rp);
}

fn fresnelDielectric3(cosTheta1: f32, n1: f32, n2: vec3f) -> vec3f {
    let sinTheta2Sq = (vec3f(n1 * n1) / (n2 * n2)) * (1.0 - cosTheta1 * cosTheta1);
    let cosTheta2 = sqrt(clamp(vec3f(1.0) - sinTheta2Sq, vec3f(0.0), vec3f(1.0)));
    let rs = (n1 * cosTheta1 - n2 * cosTheta2) / (n1 * cosTheta1 + n2 * cosTheta2);
    let rp = (n2 * cosTheta1 - n1 * cosTheta2) / (n2 * cosTheta1 + n1 * cosTheta2);
    return 0.5 * (rs * rs + rp * rp);
}

fn evalSensitivity(opd: f32, shift: vec3f) -> vec3f {
    let phase = 2.0 * PI * opd * 1.0e-9;
    let val = vec3f(5.4856e-13, 4.4201e-13, 5.2481e-13);
    let pos = vec3f(1.6810e+06, 1.7953e+06, 2.2084e+06);
    let vr = vec3f(4.3278e+09, 9.3046e+09, 6.6121e+09);
    var xyz = val * sqrt(2.0 * PI * vr) * cos(pos * phase + shift) * exp(-phase * phase * vr);
    xyz.x += 9.7470e-14 * sqrt(2.0 * PI * 4.5282e+09) * cos(2.2399e+06 * phase + shift.x)
           * exp(-4.5282e+09 * phase * phase);
    xyz /= 1.0685e-7;
    return kXyzToRec709 * xyz;
}

fn evalIridescence(outsideIor: f32, eta2: f32, cosTheta1: f32, thinFilmThickness: f32, baseF0: vec3f) -> vec3f {
    let iridescenceIor = mix(outsideIor, eta2, smoothstep(0.0, 0.03, thinFilmThickness));
    let sinTheta2Sq = (outsideIor * outsideIor) / (iridescenceIor * iridescenceIor)
                    * (1.0 - cosTheta1 * cosTheta1);
    let cosTheta2Sq = 1.0 - sinTheta2Sq;
    if (cosTheta2Sq < 0.0) {
        return vec3f(1.0);
    }
    let cosTheta2 = sqrt(cosTheta2Sq);

    let R12 = fresnelDielectric(cosTheta1, outsideIor, iridescenceIor);
    let T121 = 1.0 - R12;
    let phi12 = select(0.0, PI, iridescenceIor < outsideIor);
    let phi21 = PI - phi12;

    let baseIor = fresnel0ToIor(baseF0);
    let R23 = fresnelDielectric3(cosTheta2, iridescenceIor, baseIor);
    let phi23 = select(vec3f(0.0), vec3f(PI), baseIor < vec3f(iridescenceIor));

    let opd = 2.0 * iridescenceIor * thinFilmThickness * cosTheta2;
    let phi = vec3f(phi21) + phi23;

    let R123 = clamp(vec3f(R12) * R23, vec3f(1e-5), vec3f(0.9999));
    let r123 = sqrt(R123);
    let Rs = (T121 * T121) * R23 / (vec3f(1.0) - R123);

    var I = vec3f(R12) + Rs;
    var Cm = Rs - vec3f(T121);
    for (var m = 1; m <= 2; m++) {
        Cm *= r123;
        I += Cm * 2.0 * evalSensitivity(f32(m) * opd, f32(m) * phi);
    }
    return max(I, vec3f(0.0));
}

fn worldMappingFrame(P: vec3f, N: vec3f, size: f32, zUp: bool) -> array<vec3f, 3> {
    let U = select(vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0), zUp);
    let nu = dot(N, U);
    var T: vec3f;
    var B: vec3f;
    if (abs(nu) > 0.7071) {
        T = vec3f(1.0, 0.0, 0.0);
        B = select(vec3f(0.0, 0.0, 1.0), vec3f(0.0, -1.0, 0.0), zUp);
    } else {
        let h = normalize(N - U * nu);
        T = cross(U, h);
        B = -U;
    }
    return array<vec3f, 3>(vec3f(dot(P, T) / size, dot(P, B) / size, 0.0), T, B);
}

@fragment
fn fs_main(vIn: VertexOut, @builtin(front_facing) frontFacing: bool) -> @location(0) vec4f {
    var v = vIn;  // world mapping replaces its texture coordinates
    // Position-only meshes carry no normal: flat face normal from the
    // derivatives of the world position (taken unconditionally), towards
    // the camera.
    let faceNormal = cross(dpdx(v.worldPos), dpdy(v.worldPos));
    var worldNormal = v.worldNormal;
    var worldTangent = v.worldTangent;
    var worldBitangent = v.worldBitangent;
    if (dot(v.worldNormal, v.worldNormal) < 1e-8) {
        var N = vec3f(0.0, 0.0, 1.0);
        if (length(faceNormal) > 1e-20) {
            N = normalize(faceNormal);
        }
        if (dot(N, camera.cameraPos.xyz - v.worldPos) < 0.0) {
            N = -N;
        }
        var T = normalize(cross(vec3f(1.0, 0.0, 0.0), N));
        if (abs(N.y) < 0.999) {
            T = normalize(cross(vec3f(0.0, 1.0, 0.0), N));
        }
        worldNormal = N;
        worldTangent = T;
        worldBitangent = cross(N, T);
    }
    if (mat.worldMapping > 0.0) {
        let frame = worldMappingFrame(v.worldPos, normalize(worldNormal), mat.worldMapping, mat.worldMappingUp > 0.5);
        v.texcoord0 = frame[0].xy;
        worldTangent = frame[1];
        worldBitangent = frame[2];
    }

    // Clip planes: discard the clipped side. The section cap (back faces of
    // single-sided surfaces) is applied at the end (uniformity, see above).
    for (var i = 0u; i < min(camera.clipInfo.x, 6u); i++) {
        let plane = camera.clipPlanes[i];
        if (dot(plane.xyz, v.worldPos) + plane.w < 0.0) {
            discard;
        }
    }
    let cap = camera.clipInfo.y != 0u && !frontFacing && mat.doubleSided < 0.5;
    let capColor = vec4f(camera.capColor.rgb, 1.0);

    // UVs: each texture's own set and KHR_texture_transform (textureUV).
    let baseColorUV = textureUV(v.texcoord0, v.texcoord1, mat.texCoord1Mask, kUV1BaseColor,
                                mat.uvTransformRow0, mat.uvTransformRow1);
    let uv = baseColorUV;  // the UV debug view
    let baseColor = textureSample(baseColorTexture, baseColorSampler, baseColorUV) * mat.baseColorFactor * v.color0;

    if (mat.viewMode > 0.5) {
        if (abs(mat.viewMode - VIEW_MODE_WORLD_NORMAL) < 0.5) {
            return vec4f(normalize(worldNormal) * 0.5 + 0.5, 1.0);
        }
        if (abs(mat.viewMode - VIEW_MODE_BASE_COLOR) < 0.5) {
            return vec4f(baseColor.rgb, 1.0);
        }
    }

    // Alpha mask.
    if (mat.alphaMode > 0.5 && mat.alphaMode < 1.5 && baseColor.a < mat.alphaCutoff) {
        discard;
    }

    // Unlit (KHR_materials_unlit; also line meshes).
    if (mat.unlit > 0.5) {
        return select(baseColor, capColor, cap);
    }

    // Metallic-roughness.
    let mrSample = textureSample(metallicRoughnessTexture, metallicRoughnessSampler,
                                 textureUV(v.texcoord0, v.texcoord1, mat.texCoord1Mask, kUV1MetallicRoughness,
                                           mat.metallicRoughnessUvTransformRow0, mat.metallicRoughnessUvTransformRow1));
    let metallic = mrSample.b * mat.metallicFactor;
    let roughness = clamp(mrSample.g * mat.roughnessFactor, 0.04, 1.0);

    // Normal (normal map with its own KHR_texture_transform).
    let T = normalize(worldTangent);
    let B = normalize(worldBitangent);
    var N = normalize(worldNormal);
    if (mat.hasNormalTexture > 0.5) {
        let normalUV = textureUV(v.texcoord0, v.texcoord1, mat.texCoord1Mask, kUV1Normal,
                                 mat.normalUvTransformRow0, mat.normalUvTransformRow1);
        var ns = textureSample(normalTexture, normalSampler, normalUV).rgb * 2.0 - 1.0;
        ns = vec3f(ns.xy * mat.normalScale, ns.z);
        let n = normalize(ns);
        N = normalize(T * n.x + B * n.y + normalize(worldNormal) * n.z);
    }

    var occlusion = 1.0;
    if (mat.hasOcclusionTexture > 0.5) {
        let occ = textureSample(occlusionTexture, occlusionSampler,
                                textureUV(v.texcoord0, v.texcoord1, mat.texCoord1Mask, kUV1Occlusion,
                                          mat.occlusionUvTransformRow0, mat.occlusionUvTransformRow1)).r;
        occlusion = mix(1.0, occ, mat.occlusionStrength);
    }

    var emissive = mat.emissiveFactor.xyz;
    if (mat.hasEmissiveTexture > 0.5) {
        emissive *= textureSample(emissiveTexture, emissiveSampler,
                                  textureUV(v.texcoord0, v.texcoord1, mat.texCoord1Mask, kUV1Emissive,
                                            mat.emissiveUvTransformRow0, mat.emissiveUvTransformRow1)).rgb;
    }

    // KHR_materials_transmission / _volume / _iridescence textures (sampled
    // here, in uniform control flow; used below).
    var transmission = mat.transmissionFactor;
    if (mat.hasTransmissionTexture > 0.5) {
        transmission *= textureSample(transmissionTexture, baseColorSampler,
                                      selectUV(v.texcoord0, v.texcoord1, mat.texCoord1Mask, kUV1Transmission)).r;
    }
    var thickness = mat.thicknessFactor;
    if (mat.hasThicknessTexture > 0.5) {
        thickness *= textureSample(thicknessTexture, baseColorSampler,
                                   selectUV(v.texcoord0, v.texcoord1, mat.texCoord1Mask, kUV1Thickness)).g;
    }
    var iridescenceFactor = mat.iridescenceFactor;
    if (mat.hasIridescenceTexture > 0.5) {
        iridescenceFactor *= textureSample(iridescenceTexture, baseColorSampler,
                                           selectUV(v.texcoord0, v.texcoord1, mat.texCoord1Mask, kUV1Iridescence)).r;
    }
    var iridescenceThickness = mat.iridescenceThicknessMax;
    if (mat.hasIridescenceThicknessTexture > 0.5) {
        iridescenceThickness = mix(mat.iridescenceThicknessMin, mat.iridescenceThicknessMax,
            textureSample(iridescenceThicknessTexture, baseColorSampler,
                          selectUV(v.texcoord0, v.texcoord1, mat.texCoord1Mask, kUV1IridescenceThickness)).g);
    }

    if (mat.viewMode > 0.5) {
        if (abs(mat.viewMode - VIEW_MODE_METALLIC) < 0.5) {
            return vec4f(vec3f(metallic), 1.0);
        }
        if (abs(mat.viewMode - VIEW_MODE_ROUGHNESS) < 0.5) {
            return vec4f(vec3f(roughness), 1.0);
        }
        if (abs(mat.viewMode - VIEW_MODE_OCCLUSION) < 0.5) {
            return vec4f(vec3f(occlusion), 1.0);
        }
        if (abs(mat.viewMode - VIEW_MODE_EMISSIVE) < 0.5) {
            return vec4f(emissive, 1.0);
        }
        if (abs(mat.viewMode - VIEW_MODE_ALPHA) < 0.5) {
            return vec4f(vec3f(baseColor.a), 1.0);
        }
        if (abs(mat.viewMode - VIEW_MODE_UV0) < 0.5) {
            return vec4f(fract(uv), 0.0, 1.0);
        }
        return vec4f(0.0, 0.0, 0.0, 1.0);  // views of features not on WebGPU yet
    }

    let viewDir = normalize(camera.cameraPos.xyz - v.worldPos);
    if (dot(N, viewDir) < 0.0) {
        N = -N;  // double-sided: light the back face as a front face
    }
    let NdotV = max(dot(N, viewDir), 0.0001);

    // Image-based lighting.
    var iblDiffuse = vec3f(0.0);
    var iblSpecular = vec3f(0.0);
    if (mat.iblEnabled > 0.5) {
        let envDiffuse = textureSample(irradianceMap, environmentSampler, N).rgb;
        iblDiffuse = baseColor.rgb * (1.0 - metallic) * envDiffuse * mat.environmentIntensity;
        let envMaxLod = max(f32(textureNumLevels(environmentMap)) - 1.0, 0.0);
        let R = reflect(-viewDir, N);
        let envSpecular = textureSampleLevel(environmentMap, environmentSampler, R, roughness * envMaxLod).rgb;
        var f0 = (mat.ior - 1.0) / (mat.ior + 1.0);
        f0 *= f0;
        var F0 = mix(vec3f(f0) * mat.specularColorFactor.xyz, baseColor.rgb, metallic);
        if (iridescenceFactor > 0.0) {
            F0 = mix(F0, evalIridescence(1.0, mat.iridescenceIor, NdotV, iridescenceThickness, F0), iridescenceFactor);
        }
        iblSpecular = envSpecular * iblGGXFresnel(NdotV, roughness, F0) * mat.environmentIntensity;
    }

    // Punctual lights. F0 doesn't depend on the light (iridescence uses
    // N.V, as on Vulkan), so it's computed once.
    var lf0 = (mat.ior - 1.0) / (mat.ior + 1.0);
    lf0 *= lf0;
    var lightF0 = mix(min(vec3f(lf0) * mat.specularColorFactor.xyz, vec3f(1.0)) * mat.specularFactor,
                      baseColor.rgb, metallic);
    if (iridescenceFactor > 0.0) {
        lightF0 = mix(lightF0, evalIridescence(1.0, mat.iridescenceIor, NdotV, iridescenceThickness, lightF0),
                      iridescenceFactor);
    }
    var totalDiffuse = vec3f(0.0);
    var totalSpecular = vec3f(0.0);
    for (var i = 0u; i < min(lights.count, 8u); i++) {
        let light = lights.lights[i];
        var lightDir: vec3f;
        var attenuation = 1.0;
        var spotFactor = 1.0;
        let typeVal = light.position.w;
        if (typeVal < 0.5) {
            lightDir = normalize(light.position.xyz);
        } else {
            let toLight = light.position.xyz - v.worldPos;
            let dist = length(toLight);
            lightDir = normalize(toLight);
            attenuation = 1.0 / max(dist * dist, 0.0001);
            let rangeVal = light.direction.w;
            if (rangeVal > 0.0) {
                let nd = dist / rangeVal;
                let falloff = clamp(1.0 - nd * nd * nd * nd, 0.0, 1.0);
                attenuation *= falloff * falloff;
            }
            if (typeVal > 1.5) {
                let cosAngle = dot(-lightDir, normalize(light.direction.xyz));
                spotFactor = smoothstep(cos(light.spotAngles.y), cos(light.spotAngles.x), cosAngle);
            }
        }
        let NdotL = max(dot(N, lightDir), 0.0);
        // The sun (spotAngles.z): darkened where the shadow map says so.
        var shadow = 1.0;
        if (light.spotAngles.z > 0.5) {
            shadow = sunShadow(v.worldPos, N, lightDir);
        }
        let lightColor = light.color.xyz * light.color.w * attenuation * spotFactor * shadow;
        let halfDir = normalize(lightDir + viewDir);
        let NdotH = max(dot(N, halfDir), 0.0);
        let VdotH = max(dot(viewDir, halfDir), 0.0);

        let fresnel = lightF0 + (vec3f(1.0) - lightF0) * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
        let D = D_GGX(roughness, NdotH);
        let Vis = V_SmithGGX(NdotL, NdotV, roughness);

        let lDiffuse = (baseColor.rgb / PI) * NdotL * lightColor;
        let lSpecular = fresnel * D * Vis * NdotL * lightColor;
        totalDiffuse += lDiffuse * (vec3f(1.0) - fresnel) * (1.0 - metallic);
        totalSpecular += lSpecular;
    }

    // Occlusion darkens indirect light only; the flat ambient term stands in
    // for IBL when it is off.
    var ambientColor = baseColor.rgb * 0.25 * occlusion;
    if (mat.iblEnabled > 0.5) {
        ambientColor = vec3f(0.0);
    }
    // KHR_materials_transmission: the opaque scene behind, blurred by
    // roughness and decoded back to linear light (it's stored
    // display-encoded). With KHR_materials_volume (thickness > 0) it's
    // sampled where the refracted ray leaves the volume, per channel with
    // dispersion, and attenuated (Beer's law); thin-walled, straight behind.
    // Off screen, the environment along the refracted ray. It replaces
    // diffuse light by its share. As shaders/vulkan/default.frag.
    var transmitted = vec3f(0.0);
    var transmittance = 0.0;
    if (transmission > 0.0 && metallic < 0.99) {
        let eta = 1.0 / mat.ior;
        var Trefr = refract(-viewDir, N, eta);
        if (all(Trefr == vec3f(0.0))) {
            Trefr = -viewDir;
        }
        var sampleUV = array<vec2f, 3>(vec2f(0.0), vec2f(0.0), vec2f(0.0));
        var uvValid = array<bool, 3>(true, true, true);
        if (thickness > 0.0) {
            let viewPos = (camera.viewMatrix * vec4f(v.worldPos, 1.0)).xyz;
            let viewNormal = normalize((camera.viewMatrix * vec4f(N, 0.0)).xyz);
            let viewIncident = normalize(viewPos);
            let spread = mat.dispersion * 0.02;
            let iorRGB = vec3f(mat.ior + spread, mat.ior, max(mat.ior - spread, 1.001));
            for (var ch = 0; ch < 3; ch++) {
                var r = refract(viewIncident, viewNormal, 1.0 / iorRGB[ch]);
                if (all(r == vec3f(0.0))) {
                    r = viewIncident;
                }
                let backClip = camera.projMatrix * vec4f(viewPos + r * thickness, 1.0);
                var uvc = backClip.xy / backClip.w * 0.5 + 0.5;
                uvc.y = 1.0 - uvc.y;
                sampleUV[ch] = uvc;
                uvValid[ch] = all(uvc >= vec2f(0.0)) && all(uvc <= vec2f(1.0));
            }
        } else {
            let suv = v.clipPosition.xy / camera.screenSize;
            sampleUV = array<vec2f, 3>(suv, suv, suv);
        }
        // The glTF-Sample-Viewer's LOD.
        let lod = log2(camera.screenSize.x) * roughness * clamp(mat.ior * 2.0 - 2.0, 0.0, 1.0);
        for (var ch = 0; ch < 3; ch++) {
            if (uvValid[ch]) {
                let behind = textureSampleLevel(sceneColorTexture, sceneColorSampler, sampleUV[ch], lod)[ch];
                transmitted[ch] = pow(max(behind, 0.0), 2.2);
            } else {
                transmitted[ch] = textureSampleLevel(environmentMap, environmentSampler, Trefr, 0.0)[ch];
            }
        }
        if (thickness > 0.0 && mat.attenuationDistance > 0.0 && mat.attenuationDistance < 3.0e38) {
            let attn = (vec3f(1.0) - mat.attenuationColor.xyz) / mat.attenuationDistance;
            transmitted *= exp(-thickness * attn);
        }
        transmitted *= baseColor.rgb;
        var t0 = (mat.ior - 1.0) / (mat.ior + 1.0);
        t0 *= t0;
        let fresnel = t0 + (1.0 - t0) * pow(1.0 - NdotV, 5.0);
        transmittance = transmission * (1.0 - fresnel) * (1.0 - metallic);
    }
    var finalColor = (ambientColor + totalDiffuse + iblDiffuse * occlusion) * (1.0 - transmittance)
                   + totalSpecular + iblSpecular * occlusion + emissive + transmitted * transmittance;
    finalColor *= exp2(camera.viewParams.x);  // exposure
    finalColor = linearToSRGBFast(toneMapKhronosPbrNeutral(finalColor));

    // OPAQUE and MASK render fully opaque; only BLEND uses baseColor.a.
    let outAlpha = select(baseColor.a, 1.0, mat.alphaMode < 1.5);
    return select(vec4f(finalColor, outAlpha), capColor, cap);
}
