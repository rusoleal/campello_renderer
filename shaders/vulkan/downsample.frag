#version 450

// Downsample fragment shader (Vulkan) — bilinear downsample from scaled texture.

layout(set = 0, binding = 0) uniform texture2D sceneTexture;
layout(set = 0, binding = 1) uniform sampler   sceneSampler;

layout(location = 0) out vec4 outColor;

void main() {
    vec2 uv = gl_FragCoord.xy / vec2(textureSize(sampler2D(sceneTexture, sceneSampler), 0));
    outColor = texture(sampler2D(sceneTexture, sceneSampler), uv);
}
