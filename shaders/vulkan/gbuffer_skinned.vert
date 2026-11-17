#version 450
#extension GL_GOOGLE_include_directive : require

// The G-buffer vertex stage (gbuffer_vertex.glsl) for skinned draws.
#define SKINNED
#include "gbuffer_vertex.glsl"
