#version 450

// campello_renderer skybox shader (Vulkan) — fullscreen triangle that samples
// an environment cubemap. Direct GLSL port of shaders/metal/default.metal's
// skyboxFragment.
//
// NDC reconstruction note: shaders/vulkan/default.vert negates gl_Position.y
// after the shared (backend-agnostic) mvp transform to match Vulkan's
// rasterizer NDC convention, so the CPU-side view/projection matrices — and
// therefore u.invVP here — stay defined in the same "logical" (Metal/OpenGL-
// style Y-up NDC) convention on both backends. Reconstructing world direction
// from gl_FragCoord must undo that same logical convention, not Vulkan's raw
// rasterizer NDC, which is why this uses the identical
// (1.0 - fragCoord.y/screenSize.y) formula the Metal version uses rather than
// Vulkan's native (fragCoord.y/screenSize.y) mapping.

layout(set = 0, binding = 0) uniform textureCube envMap;
layout(set = 0, binding = 1) uniform sampler     envSampler;

layout(std140, set = 0, binding = 2) uniform SkyboxUniforms {
    mat4 invVP;
    vec2 screenSize;
    vec2 _pad;
    vec4 cameraPos; // xyz used, w unused
} u;

layout(location = 0) out vec4 outColor;

// Khronos PBR Neutral tone mapping — verbatim copy of the same function in
// default.frag (kept local since GLSL has no #include across these files in
// this build; see default.frag's copy for the full derivation comment).
vec3 toneMapKhronosPbrNeutral(vec3 color) {
    const float startCompression = 0.8 - 0.04;
    const float desaturation = 0.15;

    float x = min(color.r, min(color.g, color.b));
    float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
    color -= offset;

    float peak = max(color.r, max(color.g, color.b));
    if (peak < startCompression) return color;

    float d = 1.0 - startCompression;
    float newPeak = 1.0 - d * d / (peak + d - startCompression);
    color *= newPeak / peak;

    float g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
    return mix(color, vec3(newPeak), g);
}

vec3 linearToSRGBFast(vec3 color) {
    return pow(color, vec3(1.0 / 2.2));
}

void main() {
    vec2 ndc = vec2(
        (gl_FragCoord.x / u.screenSize.x) * 2.0 - 1.0,
        (1.0 - gl_FragCoord.y / u.screenSize.y) * 2.0 - 1.0
    );
    vec4 worldFar = u.invVP * vec4(ndc, 1.0, 1.0);
    vec3 worldDir = normalize(worldFar.xyz / worldFar.w - u.cameraPos.xyz);
    vec3 color = texture(samplerCube(envMap, envSampler), worldDir).rgb;

    // Match the tonemapping + sRGB encoding every material's fragment shader
    // applies (default.frag) before writing to the 8-bit swapchain. The
    // environment is genuine HDR (sky/sun routinely exceed 1.0, more so now
    // that the built-in default is baked with a 6x exposure correction — see
    // Renderer::createBuiltinDefaultEnvironmentMap), so returning it raw here
    // hard-clips to solid white/saturated colors instead of compressing
    // smoothly like every reflection of the same environment does.
    color = toneMapKhronosPbrNeutral(color);
    color = linearToSRGBFast(color);

    outColor = vec4(color, 1.0);
}
