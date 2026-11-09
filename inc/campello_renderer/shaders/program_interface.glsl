// campello_renderer shader program interface, GLSL (Vulkan SPIR-V).
// In an app program's shaders:
//   #extension GL_GOOGLE_include_directive : require
//   #define CAMPELLO_VERTEX_SHADER   // vertex shader only: declares the transforms
//   #include "campello_renderer/shaders/program_interface.glsl"
// (compile with -I <campello_renderer>/inc); see campello_renderer/program.hpp
// for the contract. Mirrors what the built-in shaders/vulkan/default.vert uses.

// Vertex attribute locations = VertexSemantic.
#define CAMPELLO_LOCATION_POSITION   0
#define CAMPELLO_LOCATION_NORMAL     1
#define CAMPELLO_LOCATION_TEXCOORD0  2
#define CAMPELLO_LOCATION_TANGENT    3
#define CAMPELLO_LOCATION_COLOR0     6
#define CAMPELLO_LOCATION_TEXCOORD1  7
#define CAMPELLO_LOCATION_CUSTOM0    8
#define CAMPELLO_LOCATION_CUSTOM1    9
#define CAMPELLO_LOCATION_CUSTOM2    10
#define CAMPELLO_LOCATION_CUSTOM3    11

#ifdef CAMPELLO_VERTEX_SHADER
// Per-draw transforms, per-instance attributes (column-major mat4s).
layout(location = 16) in mat4 campello_mvp;    // clip from local
layout(location = 24) in mat4 campello_model;  // world from local

// Clip position for gl_Position: Vulkan's rasterizer maps clip +Y down, the
// renderer's matrices assume +Y up (as Metal/DirectX), so Y is negated.
vec4 campello_clip(vec3 position) {
    vec4 clip = campello_mvp * vec4(position, 1.0);
    clip.y = -clip.y;
    return clip;
}
#endif
