#version 450
#extension GL_GOOGLE_include_directive : require
#define CAMPELLO_VERTEX_SHADER
#include "campello_renderer/shaders/program_interface.glsl"

// Test shader program: per-vertex uint32 id (custom0) to an r32uint target.
layout(location = CAMPELLO_LOCATION_POSITION) in vec3 position;
layout(location = CAMPELLO_LOCATION_CUSTOM0) in uint elementId;
layout(location = 0) flat out uint vId;

void main() {
    gl_Position = campello_clip(position);
    vId = elementId;
}
