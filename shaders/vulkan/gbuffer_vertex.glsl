// G-buffer vertex stage (Renderer::renderGBuffer()), the Vulkan port of
// default.metal's gbufferVertex: default.vert's outputs plus this frame's
// unjittered clip position and the previous frame's, for motion vectors.
//
// The previous frame's transform per draw is a per-instance vertex buffer at
// the draw's transform offset (Renderer's gbufferPreviousTransforms, the
// same 128-byte stride as NodeTransforms): previous clip from local at
// locations 20-23, this frame's projection jitter (clip x / w) at 28.
// Skinned draws (gbuffer_skinned.vert, SKINNED): the previous frame's joint
// palette at set 2 binding 1 places the previous position, so a moving skin
// has motion; it's the current palette when the draw has no previous one.
// Morph targets apply to both positions (this frame's weights, as on Metal).
// EXT_mesh_gpu_instancing: the instance's transform at locations 12-15.

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texcoord0;
layout(location = 3) in vec4 tangent;
layout(location = 6) in vec4 color0;
layout(location = 7) in vec2 texcoord1;
layout(location = 16) in mat4 mvp;
layout(location = 20) in mat4 previousMvp;
layout(location = 24) in mat4 model;
layout(location = 12) in mat4 instanceMatrix;
layout(location = 28) in vec4 jitter;

#include "skinning.glsl"
#ifdef SKINNED
layout(std430, set = 2, binding = 1) readonly buffer PreviousJointMatrices {
    mat4 previousJointMatrices[];
};
#endif

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
    vec3 morphedPosition = position;
    vec3 morphedNormal = normal;
    applyMorph(morphedPosition, morphedNormal);
    mat4 skin = instanceMatrix * skinMatrix();
    vec4 localPos = skin * vec4(morphedPosition, 1.0);
#ifdef SKINNED
    // Where the vertex was: the previous palette, same weights.
    mat4 previousSkin = mat4(1.0);
    if (weights.x + weights.y + weights.z + weights.w > 0.001) {
        previousSkin = mat4(0.0);
        for (int i = 0; i < 4; ++i) {
            if (weights[i] > 0.0) previousSkin += weights[i] * previousJointMatrices[joints[i]];
        }
    }
    vec4 previousLocalPos = instanceMatrix * (previousSkin * vec4(morphedPosition, 1.0));
#else
    vec4 previousLocalPos = localPos;
#endif
    vec4 clip = mvp * localPos;
    // Motion is the surface's, not the jitter's.
    currentClip  = clip - vec4(jitter.xy * clip.w, 0.0, 0.0);
    previousClip = previousMvp * previousLocalPos;
    gl_Position   = clip;
    gl_Position.y = -gl_Position.y;  // Vulkan clip Y points down (default.vert)

    worldPos = (model * localPos).xyz;
    vec3 N = normalize((model * (skin * vec4(morphedNormal, 0.0))).xyz);
    vec3 T;
    if (length(tangent.xyz) < 0.001) {
        T = abs(N.y) < 0.999 ? normalize(cross(vec3(0.0, 1.0, 0.0), N)) : normalize(cross(vec3(1.0, 0.0, 0.0), N));
    } else {
        T = normalize((model * (skin * vec4(tangent.xyz, 0.0))).xyz);
        T = normalize(T - dot(T, N) * N);
    }
    worldNormal    = N;
    worldTangent   = T;
    worldBitangent = cross(N, T);
    fragTexcoord0  = texcoord0;
    fragTexcoord1  = texcoord1;
    fragColor0     = color0;
}
