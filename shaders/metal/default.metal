#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// campello_renderer default Metal shader
//
// Vertex slot contract (must match Renderer::VERTEX_SLOT_* constants):
//   slot  0  POSITION     float3  — object-space vertex position
//   slot  1  NORMAL       float3  — object-space vertex normal
//   slot  2  TEXCOORD_0   float2  — primary UV
//   slot  3  TANGENT      float4  — tangent + bitangent sign (w)
//   slot 16  MVP          float4x4 — row-major MVP matrix (one per draw call)
//   slot 17  MaterialUniforms — per-primitive material constants
//
// Pipeline variants:
//   fragmentMain_flat     — Phong + baseColorFactor, no texture sampling
//   fragmentMain_textured — Phong + baseColorTexture × baseColorFactor
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

struct MaterialUniforms {
    float4 baseColorFactor;
};

struct VertexIn {
    float3 position  [[attribute(0)]];
    float3 normal    [[attribute(1)]];
    float2 texcoord0 [[attribute(2)]];
    float4 tangent   [[attribute(3)]];
};

struct VertexOut {
    float4 clipPosition [[position]];
    float3 objectNormal;
    float2 texcoord0;
    float4 baseColor [[flat]];  // baseColorFactor passed flat to fragment stage
};

// ---------------------------------------------------------------------------
// Vertex shader (shared by all fragment variants)
//
// The MVP stored in campello_renderer uses row-major layout (Matrix4<double>
// serialised as float[16] with data[row*4+col]). Metal's float4x4 is
// column-major, so the stored bytes represent the matrix transpose.
// A row-vector multiply  (v * M)  is mathematically equivalent to
// (M^T * v^T)^T, which correctly recovers the original transform.
// ---------------------------------------------------------------------------
vertex VertexOut vertexMain(
    VertexIn                  in   [[stage_in]],
    device const float4x4    *mvp  [[buffer(16)]],
    constant MaterialUniforms &mat [[buffer(17)]])
{
    VertexOut out;
    out.clipPosition = float4(in.position, 1.0) * mvp[0];
    out.objectNormal = in.normal;
    out.texcoord0    = in.texcoord0;
    out.baseColor    = mat.baseColorFactor;
    return out;
}

// ---------------------------------------------------------------------------
// Fragment variant: flat — Phong shading with baseColorFactor only.
// Used for primitives that have no base color texture or no TEXCOORD_0.
// ---------------------------------------------------------------------------
fragment float4 fragmentMain_flat(VertexOut in [[stage_in]])
{
    float3 lightDir = normalize(float3(1.0, 2.0, 1.0));
    float3 N        = normalize(in.objectNormal);

    float diffuse = saturate(dot(N, lightDir));
    float ambient = 0.15;
    float light   = ambient + diffuse * (1.0 - ambient);

    return float4(in.baseColor.rgb * light, in.baseColor.a);
}

// ---------------------------------------------------------------------------
// Fragment variant: textured — samples baseColorTexture, multiplies by
// baseColorFactor, then applies Phong lighting.
//
// Bind group 0:
//   [[texture(0)]] — base color texture (RGBA8)
//   [[sampler(1)]] — sampler for the base color texture
// ---------------------------------------------------------------------------
fragment float4 fragmentMain_textured(
    VertexOut        in                [[stage_in]],
    texture2d<float> baseColorTexture  [[texture(0)]],
    sampler          baseColorSampler  [[sampler(1)]])
{
    float4 texColor = baseColorTexture.sample(baseColorSampler, in.texcoord0);
    float4 color    = texColor * in.baseColor;

    float3 lightDir = normalize(float3(1.0, 2.0, 1.0));
    float3 N        = normalize(in.objectNormal);

    float diffuse = saturate(dot(N, lightDir));
    float ambient = 0.15;
    float light   = ambient + diffuse * (1.0 - ambient);

    return float4(color.rgb * light, color.a);
}
