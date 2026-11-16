// Test shader program: an app program using the lighting inputs and its own
// G-buffer stages (program_interface.metal.h).
//  - vertexMain / fragmentMain: a white Lambert surface (normal +y) lit by
//    the sun with campello::sunVisibility(), plus campello::
//    indirectIrradiance() when set; exposure applied, no tone mapping.
//  - Both vertex stages move the geometry +0.4 in local x (a stand-in for
//    wind or displacement), and gbufferVertexMain / gbufferFragmentMain
//    write that moved geometry to the G-buffer: green albedo, normal +z.
#include "campello_renderer/shaders/program_interface.metal.h"
using namespace metal;

struct VertexIn {
    float3 position [[attribute(CAMPELLO_ATTRIBUTE_POSITION)]];
    CAMPELLO_TRANSFORM_ATTRIBUTES
};

constant float3 kDisplacement = float3(0.4, 0.0, 0.0);

struct VertexOut {
    float4 position [[position]];
    float3 worldPos;
};

vertex VertexOut vertexMain(VertexIn in [[stage_in]]) {
    const float3 p = in.position + kDisplacement;
    VertexOut out;
    out.position = CAMPELLO_MVP(in) * float4(p, 1.0);
    out.worldPos = (CAMPELLO_MODEL(in) * float4(p, 1.0)).xyz;
    return out;
}

fragment float4 fragmentMain(VertexOut in [[stage_in]], CAMPELLO_LIGHTING_PARAMETERS) {
    const float3 N = float3(0.0, 1.0, 0.0);
    float3 color = float3(0.0);
    for (uint i = 0; i < campelloLights.count && i < 4; ++i) {
        const campello::Light light = campelloLights.lights[i];
        if (light.position.w != 0.0) continue;  // directional lights only
        const float3 L = normalize(light.position.xyz);
        float visibility = 1.0;
        if (campello::castsSunShadow(light)) {
            visibility = campello::sunVisibility(campelloCamera, campelloShadowMap, campelloShadowMapNear, in.worldPos,
                                                 N, L, in.position);
        }
        color += light.color.rgb * light.color.w * max(dot(N, L), 0.0) * visibility / M_PI_F;
    }
    if (campello::hasIndirectDiffuse(campelloCamera, /*opaqueDraw=*/true)) {
        color += campello::indirectIrradiance(campelloCamera, campelloScreenLighting, in.position);
    }
    return float4(saturate(color * campello::exposureScale(campelloCamera)), 1.0);
}

struct GBufferVertexOut {
    float4 position [[position]];
    float4 currentClip;
    float4 previousClip;
};

vertex GBufferVertexOut gbufferVertexMain(
    VertexIn in [[stage_in]],
    device const campello::GBufferTransforms *previous [[buffer(CAMPELLO_PREVIOUS_TRANSFORMS_BUFFER)]]) {
    const float4 p = float4(in.position + kDisplacement, 1.0);
    GBufferVertexOut out;
    out.position = CAMPELLO_MVP(in) * p;
    out.currentClip = campello::unjitteredClip(out.position, previous[0]);
    out.previousClip = previous[0].previousMvp * p;
    return out;
}

fragment campello::GBufferOut gbufferFragmentMain(GBufferVertexOut in [[stage_in]]) {
    return campello::gbufferOutput(float3(0.0, 0.0, 1.0), 0.5, float3(0.0, 1.0, 0.0), 0.0, in.currentClip,
                                   in.previousClip);
}
