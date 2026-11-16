#version 450
#extension GL_GOOGLE_include_directive : require
#include "denoiser_common.glsl"

// denoiseTemporal: the noisy image demodulated by albedo, accumulated with
// the history reprojected along the motion vectors where depth and normal
// agree; luminance moments; this frame's normal and linear depth.

layout(set = 0, binding = 0) uniform texture2D noisy;
layout(set = 0, binding = 1) uniform texture2D historyColor;
layout(set = 0, binding = 2) uniform texture2D historyMoments;
layout(set = 0, binding = 3) uniform texture2D normalRoughness;
layout(set = 0, binding = 4) uniform texture2D depthTexture;
layout(set = 0, binding = 5) uniform texture2D motion;
layout(set = 0, binding = 6) uniform texture2D previousGeometry;
layout(set = 0, binding = 7) uniform texture2D albedoMetallic;
layout(std140, set = 0, binding = 8) uniform Uniforms { DenoiseUniforms u; };

layout(location = 0) out vec4 outColor;     // rgb illumination accumulated, a history length
layout(location = 1) out vec4 outMoments;   // x mean luminance, y mean squared, z variance, w history length
layout(location = 2) out vec4 outGeometry;  // xyz normal, w linear depth (0: background)

void main() {
    ivec2 p = pixelOf();
    ivec2 size = textureSize(noisy, 0);
    vec4 n = texelFetch(noisy, p, 0);
    float z = linearDepth(texelFetch(depthTexture, p, 0).r, u);
    vec3 normal = texelFetch(normalRoughness, p, 0).xyz;
    if (z <= 0.0) {
        outColor = vec4(n.rgb, 0.0);
        outMoments = vec4(0.0);
        outGeometry = vec4(0.0);
        return;
    }
    vec3 albedo = u.depth.w > 0.5 ? max(texelFetch(albedoMetallic, p, 0).rgb, vec3(ALBEDO_FLOOR)) : vec3(1.0);
    vec3 illumination = n.rgb / albedo;
    float l = luminance(illumination);

    vec4 previousColor = vec4(0.0);
    vec4 previousMoments = vec4(0.0);
    bool valid = false;
    if (u.params.z > 0.5) {
        vec2 previousUV = fragUV + texelFetch(motion, p, 0).xy;
        if (all(greaterThanEqual(previousUV, vec2(0.0))) && all(lessThan(previousUV, vec2(1.0)))) {
            ivec2 q = min(ivec2(previousUV * vec2(size)), size - 1);
            vec4 g = texelFetch(previousGeometry, q, 0);
            if (g.w > 0.0 && abs(g.w - z) < u.depth.z * z && dot(g.xyz, normal) > 0.9) {
                previousColor = texelFetch(historyColor, q, 0);
                previousMoments = texelFetch(historyMoments, q, 0);
                valid = true;
            }
        }
    }
    float history = valid ? min(previousColor.a + 1.0, u.params.w) : 1.0;
    float colorBlend = valid ? max(1.0 / history, u.params.x) : 1.0;
    float momentsBlend = valid ? max(1.0 / history, u.params.y) : 1.0;
    vec3 color = mix(previousColor.rgb, illumination, colorBlend);
    float m1 = mix(previousMoments.x, l, momentsBlend);
    float m2 = mix(previousMoments.y, l * l, momentsBlend);
    outColor = vec4(color, history);
    outMoments = vec4(m1, m2, max(m2 - m1 * m1, 0.0), history);
    outGeometry = vec4(normal, z);
}
