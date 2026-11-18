// Test shader program, HLSL (DirectX): per-vertex uint32 id (custom0) to an
// r32uint target, as element_id.vert/.frag on Vulkan. Used when slangc isn't
// found (element_id.slang covers every backend then).
#include "campello_renderer/shaders/program_interface.hlsli"

struct VertexIn {
    float3 position : TEXCOORD0;  // VertexSemantic::position
    uint   id       : TEXCOORD8;  // VertexSemantic::custom0
    CAMPELLO_TRANSFORM_INPUTS
};

struct VertexOut {
    float4 position : SV_Position;
    nointerpolation uint id : ID;
};

VertexOut vertexMain(VertexIn input) {
    VertexOut output;
    output.position = CAMPELLO_CLIP(input, input.position);
    output.id = input.id;
    return output;
}

uint pixelMain(VertexOut input) : SV_Target {
    return input.id;
}
