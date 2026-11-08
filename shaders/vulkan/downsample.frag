#version 450

// Downsample fragment shader (Vulkan) — bilinear downsample from scaled texture.

layout(set = 0, binding = 0) uniform texture2D sceneTexture;
layout(set = 0, binding = 1) uniform sampler   sceneSampler;

layout(location = 0) in vec2 fragUV;
layout(location = 0) out vec4 outColor;

void main() {
    // The scene texture is ssaaScale x the output, so use the output-normalized
    // UV from the vertex shader, not gl_FragCoord / scene texture size.
    outColor = texture(sampler2D(sceneTexture, sceneSampler), fragUV);
}
