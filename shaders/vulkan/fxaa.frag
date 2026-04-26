#version 450

// FXAA post-process fragment shader (Vulkan)
// Based on FXAA 3.11 by Timothy Lottes (simplified).

layout(set = 0, binding = 0) uniform texture2D sceneTexture;
layout(set = 0, binding = 1) uniform sampler   sceneSampler;

layout(set = 0, binding = 2) uniform FxaaUniforms {
    vec2 rcpFrame;
    vec2 _pad;
} u;

layout(location = 0) in vec2 fragUV;
layout(location = 0) out vec4 outColor;

float fxaaLuma(vec3 rgb) {
    return dot(rgb, vec3(0.299, 0.587, 0.114));
}

void main() {
    vec2 pos = gl_FragCoord.xy;
    vec2 rcpFrame = u.rcpFrame;

    vec3 rgbNW = texture(sampler2D(sceneTexture, sceneSampler), (pos + vec2(-1.0, -1.0)) * rcpFrame).rgb;
    vec3 rgbNE = texture(sampler2D(sceneTexture, sceneSampler), (pos + vec2( 1.0, -1.0)) * rcpFrame).rgb;
    vec3 rgbSW = texture(sampler2D(sceneTexture, sceneSampler), (pos + vec2(-1.0,  1.0)) * rcpFrame).rgb;
    vec3 rgbSE = texture(sampler2D(sceneTexture, sceneSampler), (pos + vec2( 1.0,  1.0)) * rcpFrame).rgb;
    vec3 rgbM  = texture(sampler2D(sceneTexture, sceneSampler), pos * rcpFrame).rgb;

    float lumaNW = fxaaLuma(rgbNW);
    float lumaNE = fxaaLuma(rgbNE);
    float lumaSW = fxaaLuma(rgbSW);
    float lumaSE = fxaaLuma(rgbSE);
    float lumaM  = fxaaLuma(rgbM);

    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

    if (lumaMax - lumaMin < max(0.0833, lumaMax * 0.166)) {
        outColor = vec4(rgbM, 1.0);
        return;
    }

    vec2 dir;
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));

    float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * 0.25 * 0.125, 1.0/128.0);
    float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    dir = clamp(dir * rcpDirMin, vec2(-8.0), vec2(8.0)) * rcpFrame;

    vec3 rgbA = 0.5 * (
        texture(sampler2D(sceneTexture, sceneSampler), (pos + dir * (1.0/3.0 - 0.5)) * rcpFrame).rgb +
        texture(sampler2D(sceneTexture, sceneSampler), (pos + dir * (2.0/3.0 - 0.5)) * rcpFrame).rgb);

    vec3 rgbB = rgbA * 0.5 + 0.25 * (
        texture(sampler2D(sceneTexture, sceneSampler), (pos + dir * -0.5) * rcpFrame).rgb +
        texture(sampler2D(sceneTexture, sceneSampler), (pos + dir *  0.5) * rcpFrame).rgb);

    float lumaB = fxaaLuma(rgbB);
    if (lumaB < lumaMin || lumaB > lumaMax) {
        outColor = vec4(rgbA, 1.0);
    } else {
        outColor = vec4(rgbB, 1.0);
    }
}
