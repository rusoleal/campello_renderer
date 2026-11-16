#version 450
#extension GL_GOOGLE_include_directive : require
#include "denoiser_common.glsl"

// denoiseAtrousFirst: the first a-trous step, whose output is also the next
// frame's history.

layout(set = 0, binding = 0) uniform texture2D source;
layout(set = 0, binding = 1) uniform texture2D geometry;
layout(set = 0, binding = 2) uniform texture2D moments;
layout(std140, set = 0, binding = 3) uniform Uniforms { DenoiseUniforms u; };

layout(location = 0) out vec4 outColor;    // rgb, a variance (the next step's input)
layout(location = 1) out vec4 outHistory;  // rgb, a history length

void main() {
    vec4 filtered = atrous(source, geometry, u);
    outColor = filtered;
    outHistory = vec4(filtered.rgb, texelFetch(moments, pixelOf(), 0).w);
}
