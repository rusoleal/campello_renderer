// campello_renderer shader program interface, Metal Shading Language.
// #include "campello_renderer/shaders/program_interface.metal.h" from an app
// program's .metal file (compile with -I <campello_renderer>/inc); see
// campello_renderer/program.hpp for the contract. Mirrors what the built-in
// shaders/metal/default.metal uses.
#pragma once
#include <metal_stdlib>

// Vertex attribute indices ([[attribute(n)]]) = VertexSemantic.
#define CAMPELLO_ATTRIBUTE_POSITION   0
#define CAMPELLO_ATTRIBUTE_NORMAL     1
#define CAMPELLO_ATTRIBUTE_TEXCOORD0  2
#define CAMPELLO_ATTRIBUTE_TANGENT    3
#define CAMPELLO_ATTRIBUTE_JOINTS     4
#define CAMPELLO_ATTRIBUTE_WEIGHTS    5
#define CAMPELLO_ATTRIBUTE_COLOR0     6
#define CAMPELLO_ATTRIBUTE_TEXCOORD1  7
#define CAMPELLO_ATTRIBUTE_CUSTOM0    8
#define CAMPELLO_ATTRIBUTE_CUSTOM1    9
#define CAMPELLO_ATTRIBUTE_CUSTOM2    10
#define CAMPELLO_ATTRIBUTE_CUSTOM3    11

// Per-draw transforms, two ways:
//  - per-instance vertex attributes (preferred: the same contract as Vulkan,
//    DirectX and WebGPU): put CAMPELLO_TRANSFORM_ATTRIBUTES in the
//    [[stage_in]] struct and use CAMPELLO_MVP(in) / CAMPELLO_MODEL(in);
//  - element 0 of [[buffer(CAMPELLO_TRANSFORMS_BUFFER)]] (what the built-in
//    shaders read).
#define CAMPELLO_TRANSFORMS_BUFFER 16
#define CAMPELLO_TRANSFORM_ATTRIBUTES                      \
    float4 campelloMvp0   [[attribute(16)]];               \
    float4 campelloMvp1   [[attribute(17)]];               \
    float4 campelloMvp2   [[attribute(18)]];               \
    float4 campelloMvp3   [[attribute(19)]];               \
    float4 campelloModel0 [[attribute(24)]];               \
    float4 campelloModel1 [[attribute(25)]];               \
    float4 campelloModel2 [[attribute(26)]];               \
    float4 campelloModel3 [[attribute(27)]];
#define CAMPELLO_MVP(in)   metal::float4x4((in).campelloMvp0, (in).campelloMvp1, (in).campelloMvp2, (in).campelloMvp3)
#define CAMPELLO_MODEL(in) metal::float4x4((in).campelloModel0, (in).campelloModel1, (in).campelloModel2, (in).campelloModel3)

namespace campello {

struct NodeTransforms {
    metal::float4x4 mvp;    // clip from local: clip = mvp * float4(position, 1)
    metal::float4x4 model;  // world from local
};

} // namespace campello

// ===========================================================================
// Lighting inputs (fragment stage): what the renderer's effects give the
// built-in PBR shader, for app programs to use the same way. The renderer
// binds these for every draw at the bindings below (the built-in pipeline
// layout); declare the ones you use in the fragment function's parameters,
// e.g. with CAMPELLO_LIGHTING_PARAMETERS, then call the helpers. The
// helpers hide how the renderer packs its effects (which may change between
// versions; recompile programs against the header of the version you link).
//
//  - Sun shadows: campello::sunVisibility(): the shadow maps, or the ray
//    traced visibility mask (Renderer::setShadowMask()) when set.
//  - Reflections: campello::hasReflections() / reflectedRadiance() /
//    reflectionWeight() (Renderer::setReflections()); else your own
//    environment lookup (environment / irradiance / BRDF LUT textures).
//  - Indirect diffuse: campello::hasIndirectDiffuse() / indirectIrradiance()
//    (Renderer::setIndirectDiffuse(), e.g. DDGI).
//  - Direct light: campello::hasDirectLighting() / directRadiance()
//    (Renderer::setDirectLighting(), e.g. ReSTIR DI): in place of your
//    light loop.
//  - Ambient occlusion (SSAO or Renderer::setAmbientOcclusionMask()) is
//    applied to the whole frame after shading: app programs get it with no
//    code.
//  - Exposure: campello::exposureScale(); multiply the colour by it before
//    tone mapping, as the built-in shader does.
//
// Screen lighting (reflections, indirect diffuse, direct light) exists for opaque draws
// only: the transparent pass binds the scene colour (for refraction) in the
// same slot. Pass opaqueDraw = false from programs drawn with alpha blending.
// ===========================================================================

#define CAMPELLO_LIGHTS_BUFFER               10
#define CAMPELLO_MATERIAL_BUFFER             17
#define CAMPELLO_CAMERA_BUFFER               18
#define CAMPELLO_PREVIOUS_TRANSFORMS_BUFFER  27  // G-buffer vertex stage (below)
#define CAMPELLO_PREVIOUS_JOINTS_BUFFER      28  // G-buffer vertex stage: the previous joint palette
#define CAMPELLO_ENVIRONMENT_TEXTURE         21  // texturecube: GGX-prefiltered specular, one roughness per mip
#define CAMPELLO_SCREEN_LIGHTING_TEXTURE     22  // texture2d: reflections / indirect diffuse (opaque pass)
#define CAMPELLO_IRRADIANCE_TEXTURE          27  // texturecube: diffuse irradiance
#define CAMPELLO_BRDF_LUT_TEXTURE            28  // texture2d: split-sum GGX LUT (NdotV, roughness)
#define CAMPELLO_SHADOW_MAP_TEXTURE          29  // depth2d: far cascade
#define CAMPELLO_SHADOW_MAP_NEAR_TEXTURE     30  // depth2d: near cascade (or the visibility mask)

namespace campello {

// The frame's camera block ([[buffer(CAMPELLO_CAMERA_BUFFER)]]).
struct CameraUniforms {
    metal::float4   cameraPos;   // xyz: the camera in world space
    metal::float4x4 viewMatrix;
    metal::float4x4 projMatrix;
    metal::float2   screenSize;
    metal::float2   lineParams;  // x: the SSAA scale the scene is drawn at (wide lines); y: unused
    // Clip block (Renderer::setClipPlanes()).
    metal::uint4    clipInfo;    // x = plane count, y = section caps on
    metal::float4   capColor;
    metal::float4   clipPlanes[6];  // xyz = unit normal, w = d: kept where dot(n, p) + d >= 0
    // Shadow and effect block: read through the helpers below.
    metal::float4x4 shadowViewProj;  // world -> the shadow map's clip space
    metal::float4   shadowParams;    // x = on, y = one texel in uv, z = indirect diffuse (screen lighting, 2nd third),
                                     // w = direct light (3rd third)
    metal::float4   viewParams;      // x = exposure in EV; y = reflections (screen lighting, 1st third),
                                     // z = their roughness cut-off, w = their pixels per target pixel
    metal::float4x4 shadowNearViewProj;  // world -> the near cascade's clip space
    metal::float4   shadowNearParams;    // x = near cascade on; y = the near map is a visibility mask, zw = its
                                         // pixels per target pixel
};

// One light ([[buffer(CAMPELLO_LIGHTS_BUFFER)]], LightsUniform::lights).
struct Light {
    metal::float4 position;    // xyz = position (point, spot) or the unit direction towards the light (directional);
                               // w = type (0 directional, 1 point, 2 spot)
    metal::float4 color;       // xyz = linear rgb, w = intensity
    metal::float4 direction;   // xyz = spot direction, w = range
    metal::float4 spotAngles;  // x = inner cone, y = outer cone, z = casts the sun shadow (1), w = unused
};

struct LightsUniform {
    uint32_t count;
    uint32_t _pad0;
    uint32_t _pad1;
    uint32_t _pad2;
    Light lights[4];
};

// The light that casts the sun shadow (campello::sunVisibility()).
inline bool castsSunShadow(Light light) { return light.spotAngles.z > 0.5; }

// Colour x this before tone mapping (Renderer::setExposure()).
inline float exposureScale(constant CameraUniforms &camera) { return metal::exp2(camera.viewParams.x); }

// A shadow map's 3 x 3 PCF at worldPos (normal and slope offsets in shadow
// texels); `inside`: the point is within the map.
inline float shadowLookup(metal::depth2d<float> shadowMap, metal::float4x4 M, float texel, metal::float3 worldPos,
                          metal::float3 N, metal::float3 L, thread bool &inside) {
    constexpr metal::sampler shadowSampler(metal::coord::normalized, metal::filter::linear,
                                           metal::address::clamp_to_edge, metal::compare_func::less_equal);
    const float worldTexel = 2.0 * texel / metal::max(metal::length(metal::float3(M[0][0], M[1][0], M[2][0])), 1e-6);
    const float depthPerWorld = metal::length(metal::float3(M[0][2], M[1][2], M[2][2]));
    const float NdotL = metal::clamp(metal::dot(N, L), 0.0, 1.0);
    const float slope = metal::sqrt(1.0 - NdotL * NdotL) / metal::max(NdotL, 0.2);
    const metal::float4 clip = M * metal::float4(worldPos + N * worldTexel * (1.0 + slope), 1.0);
    metal::float3 ndc = clip.xyz / clip.w;
    ndc.z -= depthPerWorld * worldTexel;
    const metal::float2 uv = metal::float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    inside = metal::all(uv >= metal::float2(2.0 * texel)) && metal::all(uv <= metal::float2(1.0 - 2.0 * texel)) &&
             ndc.z <= 1.0;
    if (!inside) return 1.0;
    float lit = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            lit += shadowMap.sample_compare(shadowSampler, uv + metal::float2(x, y) * texel, ndc.z);
        }
    }
    return lit / 9.0;
}

// How lit `worldPos` is by the sun (the castsSunShadow() light): 1 lit ..
// 0 shadowed. The ray traced visibility mask at this pixel when set
// (Renderer::setShadowMask()), else the two shadow map cascades (the near
// one where it covers the point); 1 with shadows off. `N`: the surface
// normal, `L`: towards the sun, `fragPosition`: the fragment's [[position]].
inline float sunVisibility(constant CameraUniforms &camera, metal::depth2d<float> shadowMap,
                           metal::depth2d<float> shadowMapNear, metal::float3 worldPos, metal::float3 N,
                           metal::float3 L, metal::float4 fragPosition) {
    if (camera.shadowNearParams.y > 0.5) {
        const metal::uint2 size = metal::uint2(shadowMapNear.get_width(), shadowMapNear.get_height());
        const metal::uint2 pixel = metal::min(metal::uint2(fragPosition.xy * camera.shadowNearParams.zw), size - 1);
        return metal::saturate(shadowMapNear.read(pixel));
    }
    if (camera.shadowParams.x < 0.5) return 1.0;
    bool inside = false;
    if (camera.shadowNearParams.x > 0.5) {
        const float lit = shadowLookup(shadowMapNear, camera.shadowNearViewProj, camera.shadowParams.y, worldPos, N, L, inside);
        if (inside) return lit;
    }
    return shadowLookup(shadowMap, camera.shadowViewProj, camera.shadowParams.y, worldPos, N, L, inside);
}

// The screen lighting texel for this fragment in its first third
// (reflections; indirect diffuse and direct light follow, each as wide).
inline metal::uint2 screenLightingSize(metal::texture2d<float> screenLighting) {
    return metal::uint2(screenLighting.get_width() / 3, screenLighting.get_height());
}

inline metal::uint2 screenLightingPixel(constant CameraUniforms &camera, metal::texture2d<float> screenLighting,
                                        metal::float4 fragPosition) {
    const metal::uint2 size = screenLightingSize(screenLighting);
    return metal::min(metal::uint2(fragPosition.xy * camera.viewParams.w), size - 1);
}

// Ray traced reflections (Renderer::setReflections()) for this draw.
inline bool hasReflections(constant CameraUniforms &camera, bool opaqueDraw) {
    return opaqueDraw && camera.viewParams.y > 0.5;
}

// The radiance arriving along this pixel's reflection lobe (linear, before
// exposure), when hasReflections().
inline metal::float3 reflectedRadiance(constant CameraUniforms &camera, metal::texture2d<float> screenLighting,
                                       metal::float4 fragPosition) {
    return screenLighting.read(screenLightingPixel(camera, screenLighting, fragPosition)).rgb;
}

// How much of the specular environment term the traced reflection replaces
// at this roughness: 1 up to 0.75 x the cut-off, fading to 0 at it (rough
// surfaces keep the prefiltered environment).
inline float reflectionWeight(constant CameraUniforms &camera, float roughness) {
    const float cutoff = camera.viewParams.z;
    return 1.0 - metal::smoothstep(0.75 * cutoff, cutoff, roughness);
}

// Indirect diffuse light (Renderer::setIndirectDiffuse()) for this draw.
inline bool hasIndirectDiffuse(constant CameraUniforms &camera, bool opaqueDraw) {
    return opaqueDraw && camera.shadowParams.z > 0.5;
}

// The irradiance at this pixel's surface (what a white diffuse surface
// reflects, linear, before exposure), when hasIndirectDiffuse(); in place
// of the irradiance cube map.
inline metal::float3 indirectIrradiance(constant CameraUniforms &camera, metal::texture2d<float> screenLighting,
                                        metal::float4 fragPosition) {
    const metal::uint2 size = screenLightingSize(screenLighting);
    return screenLighting.read(screenLightingPixel(camera, screenLighting, fragPosition) + metal::uint2(size.x, 0)).rgb;
}

// Direct light (Renderer::setDirectLighting(), e.g. ReSTIR DI) for this
// draw: when on, it replaces the light loop (campelloLights) and the sun's
// visibility; it already holds the shadows of every light.
inline bool hasDirectLighting(constant CameraUniforms &camera, bool opaqueDraw) {
    return opaqueDraw && camera.shadowParams.w > 0.5;
}

// The direct light reflected towards the camera at this pixel (diffuse and
// specular of the G-buffer's material, linear, before exposure), when
// hasDirectLighting().
inline metal::float3 directRadiance(constant CameraUniforms &camera, metal::texture2d<float> screenLighting,
                                    metal::float4 fragPosition) {
    const metal::uint2 size = screenLightingSize(screenLighting);
    return screenLighting.read(screenLightingPixel(camera, screenLighting, fragPosition) + metal::uint2(2 * size.x, 0)).rgb;
}

} // namespace campello

// The lighting inputs as fragment parameters (add after your own; names
// campelloCamera, campelloLights, campelloScreenLighting, campelloShadowMap,
// campelloShadowMapNear).
#define CAMPELLO_LIGHTING_PARAMETERS                                                                   \
    constant campello::CameraUniforms &campelloCamera        [[buffer(CAMPELLO_CAMERA_BUFFER)]],       \
    constant campello::LightsUniform  &campelloLights        [[buffer(CAMPELLO_LIGHTS_BUFFER)]],       \
    metal::texture2d<float>            campelloScreenLighting [[texture(CAMPELLO_SCREEN_LIGHTING_TEXTURE)]], \
    metal::depth2d<float>              campelloShadowMap      [[texture(CAMPELLO_SHADOW_MAP_TEXTURE)]],     \
    metal::depth2d<float>              campelloShadowMapNear  [[texture(CAMPELLO_SHADOW_MAP_NEAR_TEXTURE)]]

// ===========================================================================
// G-buffer (Renderer::renderGBuffer()): a program's own G-buffer stages
// (ShaderProgramDesc::metalGBufferVertexEntryPoint / FragmentEntryPoint, in
// the same library), used for its materials' draws instead of the built-in
// ones. Needed when the vertex shader moves geometry (wind, displacement,
// its own skinning) or the fragment shader computes its own normals: the G-
// buffer is what TAA, the upscalers, the denoisers and every ray traced
// effect see. Without them the built-in G-buffer stages draw the program's
// meshes as plain PBR geometry.
//
// Vertex stage: the same inputs as the program's vertex stage, plus
// [[buffer(CAMPELLO_PREVIOUS_TRANSFORMS_BUFFER)]] (campello::GBufferTransforms,
// element 0) for where the vertex was a frame ago. Output the current clip
// position (jittered, as usual) and, for the fragment stage, both
// campello::unjitteredClip() of it and the previous clip position.
// Fragment stage: return campello::gbufferOutput().
// EXT_mesh_gpu_instancing instances are assumed static (one previous
// transform per draw), as in the built-in stages.
// ===========================================================================

namespace campello {

// The previous frame's clip transform for the draw (128-byte stride, as
// NodeTransforms).
struct GBufferTransforms {
    metal::float4x4 previousMvp;  // previous clip from local
    metal::float4   jitter;       // xy: this frame's projection jitter (clip x / w), for unjitteredClip()
    metal::float4   pad[3];
};

// The jitter (TAA, upscaling) removed from a clip position: motion is the
// surface's, not the jitter's.
inline metal::float4 unjitteredClip(metal::float4 clip, device const GBufferTransforms &previous) {
    return clip - metal::float4(previous.jitter.xy * clip.w, 0.0, 0.0);
}

struct GBufferOut {
    metal::float4 normalRoughness [[color(0)]];  // xyz world normal (unit, towards the viewer), w perceptual roughness
    metal::float4 albedoMetallic  [[color(1)]];  // rgb base colour (linear, 0..1), a metallic
    metal::float2 motion          [[color(2)]];  // where this point was a frame ago, minus where it is: image uv, y down
};

// One G-buffer fragment. `currentClip`: unjitteredClip() of this frame's
// clip position; `previousClip`: last frame's (previousMvp x the previous
// local position), both interpolated from the vertex stage.
inline GBufferOut gbufferOutput(metal::float3 normal, float roughness, metal::float3 albedo, float metallic,
                                metal::float4 currentClip, metal::float4 previousClip) {
    GBufferOut out;
    out.normalRoughness = metal::float4(metal::normalize(normal), metal::clamp(roughness, 0.0, 1.0));
    out.albedoMetallic = metal::float4(metal::clamp(albedo, 0.0, 1.0), metal::clamp(metallic, 0.0, 1.0));
    const metal::float2 current = currentClip.xy / currentClip.w;
    const metal::float2 previous = previousClip.xy / previousClip.w;
    out.motion = (previous - current) * metal::float2(0.5, -0.5);
    return out;
}

} // namespace campello
