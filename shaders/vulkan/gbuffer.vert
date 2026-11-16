#version 450

// G-buffer vertex stage (Renderer::renderGBuffer()), the Vulkan port of
// default.metal's gbufferVertex: default.vert's outputs plus this frame's
// unjittered clip position and the previous frame's, for motion vectors.
//
// The previous frame's transform per draw is a per-instance vertex buffer at
// the draw's transform offset (Renderer's gbufferPreviousTransforms, the
// same 128-byte stride as NodeTransforms): previous clip from local at
// locations 20-23, this frame's projection jitter (clip x / w) at 28.
// No skinning or morphing on Vulkan yet (as default.vert).

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texcoord0;
layout(location = 3) in vec4 tangent;
layout(location = 6) in vec4 color0;
layout(location = 7) in vec2 texcoord1;
layout(location = 16) in mat4 mvp;
layout(location = 20) in mat4 previousMvp;
layout(location = 24) in mat4 model;
layout(location = 28) in vec4 jitter;

layout(location = 0) out vec3 worldPos;
layout(location = 1) out vec3 worldNormal;
layout(location = 2) out vec3 worldTangent;
layout(location = 3) out vec3 worldBitangent;
layout(location = 4) out vec2 fragTexcoord0;
layout(location = 5) out vec2 fragTexcoord1;
layout(location = 6) out vec4 fragColor0;
layout(location = 7) out vec4 currentClip;   // unjittered, before the Y flip
layout(location = 8) out vec4 previousClip;

void main() {
    vec4 clip = mvp * vec4(position, 1.0);
    // Motion is the surface's, not the jitter's.
    currentClip  = clip - vec4(jitter.xy * clip.w, 0.0, 0.0);
    previousClip = previousMvp * vec4(position, 1.0);
    gl_Position   = clip;
    gl_Position.y = -gl_Position.y;  // Vulkan clip Y points down (default.vert)

    worldPos = (model * vec4(position, 1.0)).xyz;
    vec3 N = normalize((model * vec4(normal, 0.0)).xyz);
    vec3 T;
    if (length(tangent.xyz) < 0.001) {
        T = abs(N.y) < 0.999 ? normalize(cross(vec3(0.0, 1.0, 0.0), N)) : normalize(cross(vec3(1.0, 0.0, 0.0), N));
    } else {
        T = normalize((model * vec4(tangent.xyz, 0.0)).xyz);
        T = normalize(T - dot(T, N) * N);
    }
    worldNormal    = N;
    worldTangent   = T;
    worldBitangent = cross(N, T);
    fragTexcoord0  = texcoord0;
    fragTexcoord1  = texcoord1;
    fragColor0     = color0;
}
