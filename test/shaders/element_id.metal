// Test shader program: writes a per-vertex uint32 id (VertexSemantic::custom0)
// into an r32uint target (Renderer::renderProgramPass()).
#include "campello_renderer/shaders/program_interface.metal.h"
using namespace metal;

struct VertexIn {
    float3 position [[attribute(CAMPELLO_ATTRIBUTE_POSITION)]];
    uint   id       [[attribute(CAMPELLO_ATTRIBUTE_CUSTOM0)]];
};

struct VertexOut {
    float4 position [[position]];
    uint   id [[flat]];
};

vertex VertexOut vertexMain(VertexIn in [[stage_in]],
                            device const campello::NodeTransforms *transforms [[buffer(CAMPELLO_TRANSFORMS_BUFFER)]]) {
    VertexOut out;
    out.position = transforms[0].mvp * float4(in.position, 1.0);
    out.id = in.id;
    return out;
}

fragment uint fragmentMain(VertexOut in [[stage_in]]) {
    return in.id;
}
