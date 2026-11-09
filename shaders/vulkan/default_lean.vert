#version 450

// Lean vertex shader: position-only meshes (GpuMesh::positionOnly). Outputs
// a zero normal; default.frag replaces it with the flat face normal derived
// from screen-space derivatives of worldPos. Same varyings as default.vert;
// per-draw transforms arrive per instance at locations 16 (MVP) and 24 (model).

layout(location = 0) in vec3 position;
layout(location = 16) in mat4 mvp;
layout(location = 24) in mat4 model;

layout(location = 0) out vec3 worldPos;
layout(location = 1) out vec3 worldNormal;
layout(location = 2) out vec3 worldTangent;
layout(location = 3) out vec3 worldBitangent;
layout(location = 4) out vec2 fragTexcoord0;
layout(location = 5) out vec2 fragTexcoord1;
layout(location = 6) out vec4 fragColor0;

void main() {
    gl_Position   = mvp * vec4(position, 1.0);
    gl_Position.y = -gl_Position.y;  // as default.vert
    worldPos       = (model * vec4(position, 1.0)).xyz;
    worldNormal    = vec3(0.0);
    worldTangent   = vec3(0.0);
    worldBitangent = vec3(0.0);
    fragTexcoord0  = vec2(0.0);
    fragTexcoord1  = vec2(0.0);
    fragColor0     = vec4(1.0);
}
