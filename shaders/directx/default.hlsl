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

    // Material uniforms arrive from buffer slot 17 (per-instance, stride = 256 bytes).
    // Locations 20-23 follow the MVP (16-19) to avoid collision.
    float4 baseColorFactor  : TEXCOORD20;  // slot 17, shaderLocation = 20
    float4 uvTransformRow0  : TEXCOORD21;  // KHR_texture_transform row 0 [a, b, tx, hasTransform]
    float4 uvTransformRow1  : TEXCOORD22;  // KHR_texture_transform row 1 [c, d, ty, 0]
    float4 materialFlags    : TEXCOORD23;  // [alphaMode, alphaCutoff, unlit, 0]
};

struct PixelIn {
    float4 clipPosition : SV_Position;
    float3 objectNormal : TEXCOORD0;
    float2 uv           : TEXCOORD1;
    float4 baseColor    : TEXCOORD2;
    float3 materialParams : TEXCOORD3;  // [alphaMode, alphaCutoff, unlit]
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

    // Apply KHR_texture_transform only when the flag (uvTransformRow0.w > 0.5) is set.
    float2 transformedUV;
    if (input.uvTransformRow0.w > 0.5) {
        float3 uv3   = float3(input.uv, 1.0);
        transformedUV = float2(dot(input.uvTransformRow0.xyz, uv3),
                               dot(input.uvTransformRow1.xyz, uv3));
    } else {
        transformedUV = input.uv;
    }

    PixelIn output;
    output.clipPosition = mul(mvp, float4(input.position, 1.0));

    // Object-space normal forwarded to the pixel stage.
    // Limitation: correct only for uniformly scaled models. A proper normal
    // transform requires a separate normal matrix (transpose-inverse of M),
    // which will be added once campello_gpu supports constant buffers.
    output.objectNormal   = input.normal;
    output.uv             = transformedUV;
    output.baseColor      = input.baseColorFactor;
    output.materialParams = input.materialFlags.xyz; // [alphaMode, alphaCutoff, unlit]

    return output;
}

// ---------------------------------------------------------------------------
// Pixel shader — samples base color texture, applies Lambertian diffuse
// + ambient from a fixed directional light.
// ---------------------------------------------------------------------------
float4 pixelMain(PixelIn input) : SV_Target
{
    // Sample base color texture and multiply by baseColorFactor.
    float4 texColor = baseColorTexture.Sample(baseColorSampler, input.uv);
    
    // Material params: [alphaMode, alphaCutoff, unlit]
    float alphaMode   = input.materialParams.x;
    float alphaCutoff = input.materialParams.y;
    float unlit       = input.materialParams.z;
    
    // Alpha mask: discard fragment when alphaMode == 1 (mask) and alpha < alphaCutoff
    if (alphaMode > 0.5 && alphaMode < 1.5 && texColor.a < alphaCutoff) {
        discard;
    }
    
    float4 color = texColor * input.baseColor;

    // KHR_materials_unlit: skip lighting when unlit flag is set
    if (unlit > 0.5) {
        return color;
    }

    // Directional light pointing from upper-right-front (object space).
    float3 lightDir = normalize(float3(1.0, 2.0, 1.0));
    float3 N        = normalize(input.objectNormal);

    float  diffuse  = saturate(dot(N, lightDir));
    float  ambient  = 0.15;
    float  light    = ambient + diffuse * (1.0 - ambient);

    return float4(color.rgb * light, color.a);
}

// ---------------------------------------------------------------------------
// FXAA post-process shader (HLSL shader model 6.0)
// Based on FXAA 3.11 by Timothy Lottes (simplified).
// ---------------------------------------------------------------------------

struct FxaaOut {
    float4 position : SV_Position;
};

FxaaOut fxaaVertex(uint vertexID : SV_VertexID)
{
    FxaaOut output;
    float2 pos;
    if (vertexID == 0) pos = float2(-1.0, -1.0);
    else if (vertexID == 1) pos = float2(3.0, -1.0);
    else pos = float2(-1.0, 3.0);
    output.position = float4(pos, 1.0, 1.0);
    return output;
}

cbuffer FxaaUniforms : register(b0)
{
    float2 rcpFrame;
    float2 _pad;
};

Texture2D<float4> sceneTexture : register(t0);
SamplerState      sceneSampler : register(s0);

static float fxaaLuma(float3 rgb)
{
    return dot(rgb, float3(0.299, 0.587, 0.114));
}

float4 fxaaPixel(FxaaOut input) : SV_Target
{
    float2 pos = input.position.xy;
    float2 uv = pos * rcpFrame;

    float3 rgbNW = sceneTexture.Sample(sceneSampler, uv + float2(-rcpFrame.x, -rcpFrame.y)).rgb;
    float3 rgbNE = sceneTexture.Sample(sceneSampler, uv + float2( rcpFrame.x, -rcpFrame.y)).rgb;
    float3 rgbSW = sceneTexture.Sample(sceneSampler, uv + float2(-rcpFrame.x,  rcpFrame.y)).rgb;
    float3 rgbSE = sceneTexture.Sample(sceneSampler, uv + float2( rcpFrame.x,  rcpFrame.y)).rgb;
    float3 rgbM  = sceneTexture.Sample(sceneSampler, uv).rgb;

    float lumaNW = fxaaLuma(rgbNW);
    float lumaNE = fxaaLuma(rgbNE);
    float lumaSW = fxaaLuma(rgbSW);
    float lumaSE = fxaaLuma(rgbSE);
    float lumaM  = fxaaLuma(rgbM);

    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

    if (lumaMax - lumaMin < max(0.0833, lumaMax * 0.166))
        return float4(rgbM, 1.0);

    float2 dir;
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));

    float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * 0.25 * 0.125, 1.0/128.0);
    float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    dir = clamp(dir * rcpDirMin, float2(-8.0, -8.0), float2(8.0, 8.0)) * rcpFrame;

    float3 rgbA = 0.5 * (
        sceneTexture.Sample(sceneSampler, uv + dir * (1.0/3.0 - 0.5)).rgb +
        sceneTexture.Sample(sceneSampler, uv + dir * (2.0/3.0 - 0.5)).rgb);

    float3 rgbB = rgbA * 0.5 + 0.25 * (
        sceneTexture.Sample(sceneSampler, uv + dir * -0.5).rgb +
        sceneTexture.Sample(sceneSampler, uv + dir *  0.5).rgb);

    float lumaB = fxaaLuma(rgbB);
    if (lumaB < lumaMin || lumaB > lumaMax)
        return float4(rgbA, 1.0);
    return float4(rgbB, 1.0);
}

// ---------------------------------------------------------------------------
// Downsample post-process shader (HLSL shader model 6.0)
// Reuses fxaaVertex for the fullscreen triangle.
// ---------------------------------------------------------------------------

Texture2D<float4> downsampleSceneTexture : register(t0);
SamplerState      downsampleSceneSampler : register(s0);

float4 downsamplePixel(FxaaOut input) : SV_Target
{
    float width, height;
    downsampleSceneTexture.GetDimensions(width, height);
    float2 uv = input.position.xy / float2(width, height);
    return downsampleSceneTexture.Sample(downsampleSceneSampler, uv);
}
