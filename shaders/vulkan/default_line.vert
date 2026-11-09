#version 450

// Line vertex shader: line meshes (GpuMesh::topology lineList/lineStrip),
// position only (as default_lean.vert), plus a small clip-space depth offset
// towards the camera so edges drawn on their own faces win the depth test
// (depth bias doesn't apply to lines). Same as Metal's vertexLine.

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
    gl_Position.z -= 4.0e-6 * gl_Position.w;  // kLineDepthOffset (Metal vertexLine)
    gl_Position.y = -gl_Position.y;  // as default.vert
    worldPos       = (model * vec4(position, 1.0)).xyz;
    worldNormal    = vec3(0.0);
    worldTangent   = vec3(0.0);
    worldBitangent = vec3(0.0);
    fragTexcoord0  = vec2(0.0);
    fragTexcoord1  = vec2(0.0);
    fragColor0     = vec4(1.0);
}
