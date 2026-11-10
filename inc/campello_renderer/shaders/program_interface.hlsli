// campello_renderer shader program interface, HLSL (DirectX DXIL).
// #include "campello_renderer/shaders/program_interface.hlsli" (compile with
// -I <campello_renderer>/inc); see campello_renderer/program.hpp for the
// contract. Mirrors what the built-in shaders/directx/default.hlsl uses:
// every vertex input is TEXCOORD<location>.

// Per-draw transforms: add these to the vertex input struct.
#define CAMPELLO_TRANSFORM_INPUTS              \
    float4 campelloMvp0   : TEXCOORD16;        \
    float4 campelloMvp1   : TEXCOORD17;        \
    float4 campelloMvp2   : TEXCOORD18;        \
    float4 campelloMvp3   : TEXCOORD19;        \
    float4 campelloModel0 : TEXCOORD24;        \
    float4 campelloModel1 : TEXCOORD25;        \
    float4 campelloModel2 : TEXCOORD26;        \
    float4 campelloModel3 : TEXCOORD27;

// Clip position (SV_Position) of a local position: the matrix columns
// combined explicitly, as default.hlsl does (no Y flip on DirectX).
#define CAMPELLO_CLIP(input, p) \
    ((input).campelloMvp0 * (p).x + (input).campelloMvp1 * (p).y + (input).campelloMvp2 * (p).z + (input).campelloMvp3)
#define CAMPELLO_WORLD(input, p) \
    ((input).campelloModel0 * (p).x + (input).campelloModel1 * (p).y + (input).campelloModel2 * (p).z + (input).campelloModel3).xyz
