#version 450
#extension GL_GOOGLE_include_directive : require

// The PBR vertex stage (default_vertex.glsl) for skinned and morphed draws.
#define SKINNED
#include "default_vertex.glsl"
