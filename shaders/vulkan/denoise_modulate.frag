#version 450
#extension GL_GOOGLE_include_directive : require
#include "denoiser_common.glsl"

// denoiseModulate: the albedo multiplied back in; background passed through.

layout(set = 0, binding = 0) uniform texture2D filtered;
layout(set = 0, binding = 1) uniform texture2D albedoMetallic;
layout(set = 0, binding = 2) uniform texture2D noisy;
layout(set = 0, binding = 3) uniform texture2D geometry;
layout(std140, set = 0, binding = 4) uniform Uniforms { DenoiseUniforms u; };

layout(location = 0) out vec4 outColor;

void main() {
    ivec2 p = pixelOf();
    if (texelFetch(geometry, p, 0).w <= 0.0) { outColor = vec4(texelFetch(noisy, p, 0).rgb, 1.0); return; }
    vec3 albedo = u.depth.w > 0.5 ? max(texelFetch(albedoMetallic, p, 0).rgb, vec3(ALBEDO_FLOOR)) : vec3(1.0);
    outColor = vec4(texelFetch(filtered, p, 0).rgb * albedo, 1.0);
}
