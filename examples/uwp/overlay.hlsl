// Stats overlay: flat-coloured quads, positioned on the CPU in clip space.
//
// After changing this file, regenerate overlay_shaders.h from this directory:
//   dxc -T vs_6_0 -E vertexMain overlay.hlsl -Vn kOverlayVertexShader -Fh overlay_vs.tmp
//   dxc -T ps_6_0 -E pixelMain overlay.hlsl -Vn kOverlayPixelShader -Fh overlay_ps.tmp
// then concatenate both into overlay_shaders.h.

struct VertexIn {
    float2 position : TEXCOORD0;
    float3 color    : TEXCOORD1;
};

struct PixelIn {
    float4 position : SV_Position;
    float3 color    : TEXCOORD0;
};

PixelIn vertexMain(VertexIn input) {
    PixelIn output;
    output.position = float4(input.position, 0.0, 1.0);
    output.color = input.color;
    return output;
}

float4 pixelMain(PixelIn input) : SV_Target {
    return float4(input.color, 1.0);
}
