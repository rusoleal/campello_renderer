// Test shader program, Slang: writes a per-vertex uint32 id
// (VertexSemantic::custom0) into an r32uint target. One source for every
// backend (programmable pipelines phase 5).
#include "campello_renderer/shaders/program_interface.slang"

struct VertexIn {
    CAMPELLO_INPUT(float3, position, 0);  // VertexSemantic::position
    CAMPELLO_INPUT(uint, id, 8);          // VertexSemantic::custom0
    CAMPELLO_TRANSFORM_INPUTS
};

struct VertexOut {
    float4 position : SV_Position;
    nointerpolation uint id : ID;
};

[shader("vertex")]
VertexOut vertexMain(VertexIn input) {
    VertexOut output;
    output.position = CAMPELLO_CLIP(input, input.position);
    output.id = input.id;
    return output;
}

[shader("fragment")]
uint fragmentMain(VertexOut input) : SV_Target {
    return input.id;
}
