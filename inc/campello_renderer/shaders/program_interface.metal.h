// campello_renderer shader program interface, Metal Shading Language.
// #include "campello_renderer/shaders/program_interface.metal.h" from an app
// program's .metal file (compile with -I <campello_renderer>/inc); see
// campello_renderer/program.hpp for the contract. Mirrors what the built-in
// shaders/metal/default.metal uses.
#pragma once
#include <metal_stdlib>

// Vertex attribute indices ([[attribute(n)]]) = VertexSemantic.
#define CAMPELLO_ATTRIBUTE_POSITION   0
#define CAMPELLO_ATTRIBUTE_NORMAL     1
#define CAMPELLO_ATTRIBUTE_TEXCOORD0  2
#define CAMPELLO_ATTRIBUTE_TANGENT    3
#define CAMPELLO_ATTRIBUTE_JOINTS     4
#define CAMPELLO_ATTRIBUTE_WEIGHTS    5
#define CAMPELLO_ATTRIBUTE_COLOR0     6
#define CAMPELLO_ATTRIBUTE_TEXCOORD1  7
#define CAMPELLO_ATTRIBUTE_CUSTOM0    8
#define CAMPELLO_ATTRIBUTE_CUSTOM1    9
#define CAMPELLO_ATTRIBUTE_CUSTOM2    10
#define CAMPELLO_ATTRIBUTE_CUSTOM3    11

// Per-draw transforms: read element 0 of [[buffer(CAMPELLO_TRANSFORMS_BUFFER)]].
#define CAMPELLO_TRANSFORMS_BUFFER 16

namespace campello {

struct NodeTransforms {
    metal::float4x4 mvp;    // clip from local: clip = mvp * float4(position, 1)
    metal::float4x4 model;  // world from local
};

} // namespace campello
