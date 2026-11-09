// campello_renderer shader program interface, Slang: one source for Metal,
// Vulkan (SPIR-V), DirectX (via HLSL) and WebGPU (WGSL).
// #include "campello_renderer/shaders/program_interface.slang" (compile with
// -I <campello_renderer>/inc); campello_renderer_add_shader_program(SLANG ...)
// does it all. See campello_renderer/program.hpp for the contract.
//
// Vertex inputs need BOTH a [[vk::location(n)]] and a TEXCOORD<n> semantic
// for every target to keep the location (Slang 2026.20: Metal and SPIR-V
// follow vk::location, HLSL and WGSL the semantic): use CAMPELLO_INPUT.
// Vulkan's clip-space Y flip is the -fvk-invert-y compiler option, not code.

#pragma once

// `n` must be a literal number (0 position, 1 normal, 2 texcoord0,
// 3 tangent, 6 color0, 7 texcoord1, 8-11 custom0-3): it is pasted into
// TEXCOORD<n>, and a macro argument isn't expanded before pasting.
#define CAMPELLO_INPUT(type, name, n) [[vk::location(n)]] type name : TEXCOORD##n

// Vertex attribute locations = VertexSemantic (for [[vk::location]] and code;
// CAMPELLO_INPUT needs the literal numbers).
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

// Per-draw transforms (per-instance attributes): put this in the vertex
// input struct, then CAMPELLO_CLIP(input, position) / CAMPELLO_WORLD(input, position).
#define CAMPELLO_TRANSFORM_INPUTS                 \
    CAMPELLO_INPUT(float4, campelloMvp0, 16);     \
    CAMPELLO_INPUT(float4, campelloMvp1, 17);     \
    CAMPELLO_INPUT(float4, campelloMvp2, 18);     \
    CAMPELLO_INPUT(float4, campelloMvp3, 19);     \
    CAMPELLO_INPUT(float4, campelloModel0, 24);   \
    CAMPELLO_INPUT(float4, campelloModel1, 25);   \
    CAMPELLO_INPUT(float4, campelloModel2, 26);   \
    CAMPELLO_INPUT(float4, campelloModel3, 27);

// Matrix columns combined explicitly (identical on every target).
#define CAMPELLO_CLIP(input, p) \
    ((input).campelloMvp0 * (p).x + (input).campelloMvp1 * (p).y + (input).campelloMvp2 * (p).z + (input).campelloMvp3)
#define CAMPELLO_WORLD(input, p) \
    ((input).campelloModel0 * (p).x + (input).campelloModel1 * (p).y + (input).campelloModel2 * (p).z + (input).campelloModel3).xyz
