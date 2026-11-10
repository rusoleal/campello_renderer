// Test shader program: writes a per-vertex uint32 id (VertexSemantic::custom0)
// into an r32uint target (Renderer::renderProgramPass()).
#include "campello_renderer/shaders/program_interface.metal.h"
using namespace metal;

// Transforms as per-instance attributes (the same contract as Vulkan and
// DirectX, so one source, e.g. Slang, serves every backend).
struct VertexIn {
    float3 position [[attribute(CAMPELLO_ATTRIBUTE_POSITION)]];
    uint   id       [[attribute(CAMPELLO_ATTRIBUTE_CUSTOM0)]];
    CAMPELLO_TRANSFORM_ATTRIBUTES
};

struct VertexOut {
    float4 position [[position]];
    uint   id [[flat]];
};

vertex VertexOut vertexMain(VertexIn in [[stage_in]]) {
    VertexOut out;
    out.position = CAMPELLO_MVP(in) * float4(in.position, 1.0);
    out.id = in.id;
    return out;
}

fragment uint fragmentMain(VertexOut in [[stage_in]]) {
    return in.id;
}
