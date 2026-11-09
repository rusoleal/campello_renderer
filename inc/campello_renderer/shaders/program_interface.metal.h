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

// Per-draw transforms, two ways:
//  - per-instance vertex attributes (preferred: the same contract as Vulkan,
//    DirectX and WebGPU): put CAMPELLO_TRANSFORM_ATTRIBUTES in the
//    [[stage_in]] struct and use CAMPELLO_MVP(in) / CAMPELLO_MODEL(in);
//  - element 0 of [[buffer(CAMPELLO_TRANSFORMS_BUFFER)]] (what the built-in
//    shaders read).
#define CAMPELLO_TRANSFORMS_BUFFER 16
#define CAMPELLO_TRANSFORM_ATTRIBUTES                      \
    float4 campelloMvp0   [[attribute(16)]];               \
    float4 campelloMvp1   [[attribute(17)]];               \
    float4 campelloMvp2   [[attribute(18)]];               \
    float4 campelloMvp3   [[attribute(19)]];               \
    float4 campelloModel0 [[attribute(24)]];               \
    float4 campelloModel1 [[attribute(25)]];               \
    float4 campelloModel2 [[attribute(26)]];               \
    float4 campelloModel3 [[attribute(27)]];
#define CAMPELLO_MVP(in)   metal::float4x4((in).campelloMvp0, (in).campelloMvp1, (in).campelloMvp2, (in).campelloMvp3)
#define CAMPELLO_MODEL(in) metal::float4x4((in).campelloModel0, (in).campelloModel1, (in).campelloModel2, (in).campelloModel3)

namespace campello {

struct NodeTransforms {
    metal::float4x4 mvp;    // clip from local: clip = mvp * float4(position, 1)
    metal::float4x4 model;  // world from local
};

} // namespace campello
