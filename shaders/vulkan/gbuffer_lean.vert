#version 450

// G-buffer vertex stage for position-only meshes (gbufferVertexLean):
// default_lean.vert's outputs (zero normal: gbuffer.frag derives the flat
// face normal) plus the clips for motion vectors (gbuffer.vert).

layout(location = 0) in vec3 position;
layout(location = 12) in mat4 instanceMatrix;  // per instance (identity without instances)
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
layout(location = 7) out vec4 currentClip;
layout(location = 8) out vec4 previousClip;

void main() {
    vec4 localPos = instanceMatrix * vec4(position, 1.0);
    vec4 clip = mvp * localPos;
    currentClip  = clip - vec4(jitter.xy * clip.w, 0.0, 0.0);
    previousClip = previousMvp * localPos;
    gl_Position   = clip;
    gl_Position.y = -gl_Position.y;
    worldPos       = (model * localPos).xyz;
    worldNormal    = vec3(0.0);
    worldTangent   = vec3(0.0);
    worldBitangent = vec3(0.0);
    fragTexcoord0  = vec2(0.0);
    fragTexcoord1  = vec2(0.0);
    fragColor0     = vec4(1.0);
}
