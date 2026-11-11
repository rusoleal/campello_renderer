// campello_renderer built-in PBR mesh shaders, WGSL (WebGPU).
//
// Hand-written port of the mesh shaders in shaders/metal/default.metal,
// limited to WebGPU's default 16 sampled textures per shader stage: glTF
// core textures (base colour, metallic-roughness, normal, emissive,
// occlusion) and image-based lighting. Not on WebGPU yet: the KHR extension
// textures, transmission/volume, sheen, clearcoat, iridescence, anisotropy,
// dispersion and the flat/debug variants (programmable pipelines phase 7).
//
// Entry points: vs_main (glTF layout), vs_lean (positions only), vs_line
// (line meshes), fs_main. Bindings follow the Vulkan layout
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
    normalUvTransformRow0: vec4f,
    normalUvTransformRow1: vec4f,
    texCoord1Mask: f32,
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
    lights: array<Light, 4>,
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
@group(0) @binding(24) var<uniform> mat: MaterialUniforms;

@group(1) @binding(0)  var<uniform> lights: LightsUniform;
@group(1) @binding(1)  var<uniform> camera: CameraUniforms;
@group(1) @binding(2)  var environmentMap: texture_cube<f32>;   // GGX-prefiltered specular
@group(1) @binding(3)  var environmentSampler: sampler;
@group(1) @binding(6)  var irradianceMap: texture_cube<f32>;
@group(1) @binding(7)  var brdfLutTexture: texture_2d<f32>;
@group(1) @binding(8)  var brdfLutSampler: sampler;
@group(1) @binding(9)  var shadowMap: texture_depth_2d;          // sun shadows
@group(1) @binding(10) var shadowSampler: sampler_comparison;

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

@vertex
fn vs_main(v: VertexIn, @builtin(instance_index) instance: u32) -> VertexOut {
    let t = transforms[instance];
    let localPos = vec4f(v.position, 1.0);
    var o: VertexOut;
    o.clipPosition = t.mvp * localPos;
    o.worldPos = (t.model * localPos).xyz;
    let N = normalize((t.model * vec4f(v.normal, 0.0)).xyz);
    var T: vec3f;
    if (length(v.tangent.xyz) < 0.001) {
        if (abs(N.y) < 0.999) {
            T = normalize(cross(vec3f(0.0, 1.0, 0.0), N));
        } else {
            T = normalize(cross(vec3f(1.0, 0.0, 0.0), N));
        }
    } else {
        T = normalize((t.model * vec4f(v.tangent.xyz, 0.0)).xyz);
        T = normalize(T - dot(T, N) * N);
    }
    o.worldNormal = N;
    o.worldTangent = T;
    o.worldBitangent = cross(N, T);
    o.texcoord0 = v.texcoord0;
    o.texcoord1 = v.texcoord1;
    o.color0 = v.color0;
    return o;
}

fn positionOnly(position: vec3f, instance: u32) -> VertexOut {
    let t = transforms[instance];
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
    return positionOnly(position, instance);
}

// Line meshes: nudged towards the camera so edges win over their faces.
@vertex
fn vs_line(@location(0) position: vec3f, @builtin(instance_index) instance: u32) -> VertexOut {
    var o = positionOnly(position, instance);
    o.clipPosition.z -= kLineDepthOffset * o.clipPosition.w;
    return o;
}

// ---------------------------------------------------------------------------
// Fragment stage
// ---------------------------------------------------------------------------

// How lit `worldPos` is by the shadow-casting light: 1 lit .. 0 shadowed.
// 3x3 percentage-closer filtering (hardware compare, bilinear); outside the
// map: lit. Receiver bias against self-shadowing, sized to one shadow-map
// texel in world units (the orthographic scales are the matrix rows'
// lengths): along the normal, more at grazing light, plus a texel of depth.
// textureSampleCompareLevel: no derivatives, so fine in the light loop.
fn sunShadow(worldPos: vec3f, N: vec3f, L: vec3f) -> f32 {
    if (camera.shadowParams.x < 0.5) {
        return 1.0;
    }
    let M = camera.shadowViewProj;
    let texel = camera.shadowParams.y;
    let worldTexel = 2.0 * texel / max(length(vec3f(M[0][0], M[1][0], M[2][0])), 1e-6);
    let depthPerWorld = length(vec3f(M[0][2], M[1][2], M[2][2]));
    let NdotL = clamp(dot(N, L), 0.0, 1.0);
    let slope = sqrt(1.0 - NdotL * NdotL) / max(NdotL, 0.2);
    let clip = M * vec4f(worldPos + N * worldTexel * (1.0 + slope), 1.0);
    let ndc = clip.xyz / clip.w;
    let uv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0)) || ndc.z > 1.0) {
        return 1.0;
    }
    let depth = ndc.z - depthPerWorld * worldTexel;
    var lit = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            lit += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2f(f32(x), f32(y)) * texel, depth);
        }
    }
    return lit / 9.0;
}
fn selectUV(uv0: vec2f, uv1: vec2f, packedMask: f32, bit: u32) -> vec2f {
    return select(uv0, uv1, (u32(packedMask) & bit) != 0u);
}

fn transformUV(uv: vec2f, row0: vec4f, row1: vec4f) -> vec2f {
    let uv3 = vec3f(uv, 1.0);
    return vec2f(dot(row0.xyz, uv3), dot(row1.xyz, uv3));
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

@fragment
fn fs_main(v: VertexOut, @builtin(front_facing) frontFacing: bool) -> @location(0) vec4f {
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

    // UVs: KHR_texture_transform on TEXCOORD_0, per-texture TEXCOORD_1 bits.
    let hasTransform = mat.uvTransformRow0.w > 0.5;
    var uv = v.texcoord0;
    if (hasTransform) {
        uv = transformUV(v.texcoord0, mat.uvTransformRow0, mat.uvTransformRow1);
    }
    let baseColorUV = select(selectUV(uv, v.texcoord1, mat.texCoord1Mask, kUV1BaseColor), uv, hasTransform);
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
                                 selectUV(v.texcoord0, v.texcoord1, mat.texCoord1Mask, kUV1MetallicRoughness));
    let metallic = mrSample.b * mat.metallicFactor;
    let roughness = clamp(mrSample.g * mat.roughnessFactor, 0.04, 1.0);

    // Normal (normal map with its own KHR_texture_transform).
    let T = normalize(worldTangent);
    let B = normalize(worldBitangent);
    var N = normalize(worldNormal);
    if (mat.hasNormalTexture > 0.5) {
        var normalUV = selectUV(v.texcoord0, v.texcoord1, mat.texCoord1Mask, kUV1Normal);
        if (mat.normalUvTransformRow0.w > 0.5) {
            normalUV = transformUV(v.texcoord0, mat.normalUvTransformRow0, mat.normalUvTransformRow1);
        }
        var ns = textureSample(normalTexture, normalSampler, normalUV).rgb * 2.0 - 1.0;
        ns = vec3f(ns.xy * mat.normalScale, ns.z);
        let n = normalize(ns);
        N = normalize(T * n.x + B * n.y + normalize(worldNormal) * n.z);
    }

    var occlusion = 1.0;
    if (mat.hasOcclusionTexture > 0.5) {
        let occ = textureSample(occlusionTexture, occlusionSampler,
                                selectUV(v.texcoord0, v.texcoord1, mat.texCoord1Mask, kUV1Occlusion)).r;
        occlusion = mix(1.0, occ, mat.occlusionStrength);
    }

    var emissive = mat.emissiveFactor.xyz;
    if (mat.hasEmissiveTexture > 0.5) {
        emissive *= textureSample(emissiveTexture, emissiveSampler,
                                  selectUV(v.texcoord0, v.texcoord1, mat.texCoord1Mask, kUV1Emissive)).rgb;
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
        let F0 = mix(vec3f(f0) * mat.specularColorFactor.xyz, baseColor.rgb, metallic);
        iblSpecular = envSpecular * iblGGXFresnel(NdotV, roughness, F0) * mat.environmentIntensity;
    }

    // Punctual lights.
    var totalDiffuse = vec3f(0.0);
    var totalSpecular = vec3f(0.0);
    for (var i = 0u; i < min(lights.count, 4u); i++) {
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

        var f0 = (mat.ior - 1.0) / (mat.ior + 1.0);
        f0 *= f0;
        let F0dielectric = min(vec3f(f0) * mat.specularColorFactor.xyz, vec3f(1.0)) * mat.specularFactor;
        let F0 = mix(F0dielectric, baseColor.rgb, metallic);
        let fresnel = F0 + (vec3f(1.0) - F0) * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
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
    var finalColor = ambientColor + totalDiffuse + totalSpecular
                   + iblDiffuse * occlusion + iblSpecular * occlusion + emissive;
    finalColor *= exp2(camera.viewParams.x);  // exposure
    finalColor = linearToSRGBFast(toneMapKhronosPbrNeutral(finalColor));

    // OPAQUE and MASK render fully opaque; only BLEND uses baseColor.a.
    let outAlpha = select(baseColor.a, 1.0, mat.alphaMode < 1.5);
    return select(vec4f(finalColor, outAlpha), capColor, cap);
}
