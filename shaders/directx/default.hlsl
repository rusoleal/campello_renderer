// campello_renderer default DirectX 12 shader (HLSL shader model 6.0)
//
// Vertex slot contract (must match Renderer::VERTEX_SLOT_* constants):
//   slot  0  POSITION   float3  — object-space vertex position
//   slot  1  NORMAL     float3  — object-space vertex normal
//   slot  2  TEXCOORD_0 float2  — primary UV
//   slot  3  TANGENT    float4  — tangent + bitangent sign (w)
//   slot 16  MVP        float4x4 — row-major MVP, per-instance (split into 4 rows)
//
// campello_gpu DirectX backend maps every attribute to semantic TEXCOORD with
// SemanticIndex = shaderLocation, so the HLSL struct uses TEXCOORD semantics
// throughout (not the standard POSITION / NORMAL / TEXCOORD / TANGENT names).
//
// The MVP data is row-major (Matrix4<double> serialised as float[16] with
// data[row*4+col]). HLSL float4x4 is also row-major by default, so
// float4x4(mvpRow0..3) reconstructs the intended matrix without any transpose.
// mul(mvp, float4(pos, 1.0)) then correctly applies M * pos as a column-vector
// transform.
//
// Compile on Windows with DXC:
//   dxc -T vs_6_0 -E vertexMain  default.hlsl -Fo default_vs.dxil
//   dxc -T ps_6_0 -E pixelMain   default.hlsl -Fo default_ps.dxil
//
// On macOS (after: brew install directx-shader-compiler):
//   dxc -T vs_6_0 -E vertexMain  default.hlsl -Fo default_vs.dxil -spirv  (SPIR-V target, not usable for D3D12)
// Note: cross-compilation to true DXIL on macOS is not supported; use Windows.

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

struct VertexIn {
    float3 position : TEXCOORD0;   // slot 0,  shaderLocation = 0
    float3 normal   : TEXCOORD1;   // slot 1,  shaderLocation = 1
    float2 uv       : TEXCOORD2;   // slot 2,  shaderLocation = 2
    float4 tangent  : TEXCOORD3;   // slot 3,  shaderLocation = 3

    // MVP matrix arrives as 4 separate float4 rows from buffer slot 16
    // (per-instance, stride = 64 bytes). Each row is its own attribute.
    float4 mvpRow0 : TEXCOORD16;   // slot 16, shaderLocation = 16
    float4 mvpRow1 : TEXCOORD17;   // slot 16, shaderLocation = 17
    float4 mvpRow2 : TEXCOORD18;   // slot 16, shaderLocation = 18
    float4 mvpRow3 : TEXCOORD19;   // slot 16, shaderLocation = 19
};

struct PixelIn {
    float4 clipPosition : SV_Position;
    float3 objectNormal : TEXCOORD0;
    float2 uv           : TEXCOORD1;
};

// Bind group 0 — base color texture and sampler.
// campello_gpu DirectX backend maps binding=0 texture → t0, binding=1 sampler → s0.
Texture2D<float4> baseColorTexture : register(t0);
SamplerState      baseColorSampler : register(s0);

// ---------------------------------------------------------------------------
// Vertex shader
// ---------------------------------------------------------------------------
PixelIn vertexMain(VertexIn input)
{
    // Reconstruct the row-major MVP from the four per-row inputs.
    // HLSL float4x4(r0,r1,r2,r3) fills rows, matching the row-major layout
    // stored by campello_renderer. mul(m, col_vec) is then a correct M*v.
    float4x4 mvp = float4x4(input.mvpRow0, input.mvpRow1,
                             input.mvpRow2, input.mvpRow3);

    PixelIn output;
    output.clipPosition = mul(mvp, float4(input.position, 1.0));

    // Object-space normal forwarded to the pixel stage.
    // Limitation: correct only for uniformly scaled models. A proper normal
    // transform requires a separate normal matrix (transpose-inverse of M),
    // which will be added once campello_gpu supports constant buffers.
    output.objectNormal = input.normal;
    output.uv           = input.uv;

    return output;
}

// ---------------------------------------------------------------------------
// Pixel shader — samples base color texture, applies Lambertian diffuse
// + ambient from a fixed directional light.
// ---------------------------------------------------------------------------
float4 pixelMain(PixelIn input) : SV_Target
{
    // Sample base color texture.
    float4 texColor = baseColorTexture.Sample(baseColorSampler, input.uv);

    // Directional light pointing from upper-right-front (object space).
    float3 lightDir = normalize(float3(1.0, 2.0, 1.0));
    float3 N        = normalize(input.objectNormal);

    float  diffuse  = saturate(dot(N, lightDir));
    float  ambient  = 0.15;
    float  light    = ambient + diffuse * (1.0 - ambient);

    return float4(texColor.rgb * light, texColor.a);
}
