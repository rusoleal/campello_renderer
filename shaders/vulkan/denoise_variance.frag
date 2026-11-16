#version 450
#extension GL_GOOGLE_include_directive : require
#include "denoiser_common.glsl"

// denoiseVariance: the temporal variance, or a 7 x 7 spatial estimate while
// a pixel's history is short.

layout(set = 0, binding = 0) uniform texture2D accumulated;
layout(set = 0, binding = 1) uniform texture2D moments;
layout(set = 0, binding = 2) uniform texture2D geometry;
layout(std140, set = 0, binding = 3) uniform Uniforms { DenoiseUniforms u; };

layout(location = 0) out vec4 outColor;

void main() {
    ivec2 p = pixelOf();
    ivec2 size = textureSize(accumulated, 0);
    vec4 c = texelFetch(accumulated, p, 0);
    vec4 m = texelFetch(moments, p, 0);
    vec4 g = texelFetch(geometry, p, 0);
    float depthGradient = max(fwidth(g.w), 1e-4);
    if (g.w <= 0.0) { outColor = vec4(c.rgb, 0.0); return; }
    if (m.w >= 4.0) { outColor = vec4(c.rgb, m.z); return; }
    float weights = 0.0, m1 = 0.0, m2 = 0.0;
    vec3 color = vec3(0.0);
    for (int dy = -3; dy <= 3; ++dy) {
        for (int dx = -3; dx <= 3; ++dx) {
            ivec2 q = clamp(p + ivec2(dx, dy), ivec2(0), size - 1);
            vec4 tg = texelFetch(geometry, q, 0);
            float w = geometryWeight(g.w, g.xyz, tg, u.filterParams.w * depthGradient * length(vec2(dx, dy)) + 1e-3, u.filterParams.z);
            vec4 tc = texelFetch(accumulated, q, 0);
            float l = luminance(tc.rgb);
            weights += w;
            m1 += w * l;
            m2 += w * l * l;
            color += w * tc.rgb;
        }
    }
    weights = max(weights, 1e-6);
    m1 /= weights;
    m2 /= weights;
    outColor = vec4(color / weights, max(m2 - m1 * m1, 0.0) * 4.0 / max(m.w, 1.0));
}
