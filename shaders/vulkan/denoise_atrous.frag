#version 450
#extension GL_GOOGLE_include_directive : require
#include "denoiser_common.glsl"

// denoiseAtrous: one a-trous step.

layout(set = 0, binding = 0) uniform texture2D source;
layout(set = 0, binding = 1) uniform texture2D geometry;
layout(std140, set = 0, binding = 2) uniform Uniforms { DenoiseUniforms u; };

layout(location = 0) out vec4 outColor;

void main() {
    outColor = atrous(source, geometry, u);
}
