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

// ---------------------------------------------------------------------------
// Procedural texture bake compute shader — uber-shader VM interpreter.
//
// Each thread evaluates the full procedural graph for one pixel.
// The graph is flattened to an array of ProceduralNode structs (bytecode).
// A fixed-size register file of float4 values holds intermediate results.
//
// Bindings:
//   register(b0)  — ProceduralBakeUniforms (constant buffer)
//   register(t1)  — ProceduralNode[] node bytecode (SRV)
//   register(u2)  — float4[] output pixels (UAV)
//
// Compile on Windows with DXC:
//   dxc -T cs_6_0 -E proceduralBakeKernel default.hlsl -Fo procedural_bake.dxil
// ---------------------------------------------------------------------------

struct ProceduralNode {
    float4   value;     // inline constant / parameters
    uint     op;        // ProceduralOp opcode
    uint     outReg;    // output register index
    uint     inA;       // input register A  (0xFFFFFFFF = none)
    uint     inB;       // input register B
    uint     inC;       // input register C
    uint     inD;       // input register D
    uint     flags;     // type / swizzle / presence mask
    uint     pad;
};

struct ProceduralBakeUniforms {
    uint   width;
    uint   height;
    uint   nodeCount;
    uint   outputReg;
    uint   outputComponents; // 1=float, 2=vec2, 3=vec3/color3, 4=color4
    uint   pad[3];
};

// Opcodes — must match C++ ProceduralOp enum exactly.
static const uint OP_CONSTANT      = 0;
static const uint OP_TEXCOORD      = 1;
static const uint OP_ADD           = 2;
static const uint OP_SUBTRACT      = 3;
static const uint OP_MULTIPLY      = 4;
static const uint OP_DIVIDE        = 5;
static const uint OP_FLOOR         = 6;
static const uint OP_MODULO        = 7;
static const uint OP_ABS           = 8;
static const uint OP_CLAMP         = 9;
static const uint OP_POWER         = 10;
static const uint OP_SQRT          = 11;
static const uint OP_MAX           = 12;
static const uint OP_MIN           = 13;
static const uint OP_SIN           = 14;
static const uint OP_COS           = 15;
static const uint OP_DOTPRODUCT    = 16;
static const uint OP_CROSSPRODUCT  = 17;
static const uint OP_LENGTH        = 18;
static const uint OP_DISTANCE      = 19;
static const uint OP_NORMALIZE     = 20;
static const uint OP_MIX           = 21;
static const uint OP_NOISE2D       = 22;
static const uint OP_CHECKERBOARD  = 23;
static const uint OP_PLACE2D       = 24;
static const uint OP_SWIZZLE       = 25;
static const uint OP_COMBINE       = 26;
static const uint OP_EXTRACT       = 27;
static const uint OP_IFGREATER     = 28;
static const uint OP_IFEQUAL       = 29;

// Presence flags occupy bits 8-11 so they don't collide with swizzle
// (bits 0-7), extract (bits 0-1), or combine count (bits 0-3).
static const uint FLAG_HAS_A = 0x0100;
static const uint FLAG_HAS_B = 0x0200;
static const uint FLAG_HAS_C = 0x0400;
static const uint FLAG_HAS_D = 0x0800;

// Noise permutation table (identical to CPU baker).
static const int kPerm[512] = {
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
    8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
    35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,
    134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
    55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,
    18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,
    250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,
    189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,
    172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,
    228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,
    107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
    138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
    8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
    35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,
    134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
    55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,
    18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,
    250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,
    189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,
    172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,
    228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,
    107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
    138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
};

static float fade(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

static float lerpVal(float a, float b, float t) {
    return a + t * (b - a);
}

static float grad(int hash, float x, float y) {
    int h = hash & 7;
    float u = h < 4 ? x : -x;
    float v = h < 2 || h == 5 || h == 6 ? y : -y;
    return u + v;
}

static float perlin2D(float x, float y) {
    int X = (int)floor(x) & 255;
    int Y = (int)floor(y) & 255;
    x -= floor(x);
    y -= floor(y);
    float u = fade(x);
    float v = fade(y);
    int A = kPerm[X] + Y;
    int B = kPerm[X + 1] + Y;
    return lerpVal(v, lerpVal(u, grad(kPerm[A], x, y), grad(kPerm[B], x - 1.0f, y)),
                       lerpVal(u, grad(kPerm[A + 1], x, y - 1.0f), grad(kPerm[B + 1], x - 1.0f, y - 1.0f)));
}

static float noise2D(float x, float y) {
    return (perlin2D(x, y) + 1.0f) * 0.5f;
}

static float fbm2D(float x, float y, int octaves, float lacunarity, float gain) {
    float total = 0.0f;
    float amplitude = 1.0f;
    float frequency = 1.0f;
    float maxValue = 0.0f;
    for (int i = 0; i < octaves; i++) {
        total += noise2D(x * frequency, y * frequency) * amplitude;
        maxValue += amplitude;
        amplitude *= gain;
        frequency *= lacunarity;
    }
    return total / maxValue;
}

ConstantBuffer<ProceduralBakeUniforms> uniforms : register(b0);
StructuredBuffer<ProceduralNode> nodes : register(t1);
RWStructuredBuffer<float4> outPixels : register(u2);

[numthreads(1, 1, 1)]
void proceduralBakeKernel(uint3 groupID : SV_GroupID)
{
    uint x = groupID.x;
    uint y = groupID.y;
    if (x >= uniforms.width || y >= uniforms.height) return;

    const uint REG_COUNT = 64;
    float4 regs[REG_COUNT];
    for (uint i = 0; i < REG_COUNT; i++) regs[i] = float4(0.0f, 0.0f, 0.0f, 0.0f);

    float u = ((float)x + 0.5f) / (float)uniforms.width;
    float v = ((float)y + 0.5f) / (float)uniforms.height;

    for (uint n = 0; n < uniforms.nodeCount; n++) {
        ProceduralNode node = nodes[n];
        float4 a = (node.flags & FLAG_HAS_A) ? regs[node.inA] : float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 b = (node.flags & FLAG_HAS_B) ? regs[node.inB] : float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 c = (node.flags & FLAG_HAS_C) ? regs[node.inC] : float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 d = (node.flags & FLAG_HAS_D) ? regs[node.inD] : float4(0.0f, 0.0f, 0.0f, 0.0f);

        float4 r = float4(0.0f, 0.0f, 0.0f, 0.0f);

        switch (node.op) {
            case OP_CONSTANT:
                r = node.value;
                break;
            case OP_TEXCOORD:
                r = float4(u, v, 0.0f, 0.0f);
                break;
            case OP_ADD:
                r = a + b;
                break;
            case OP_SUBTRACT:
                r = a - b;
                break;
            case OP_MULTIPLY:
                r = a * b;
                break;
            case OP_DIVIDE:
                r = float4(
                    b.x != 0.0f ? a.x / b.x : 0.0f,
                    b.y != 0.0f ? a.y / b.y : 0.0f,
                    b.z != 0.0f ? a.z / b.z : 0.0f,
                    b.w != 0.0f ? a.w / b.w : 0.0f);
                break;
            case OP_FLOOR:
                r = floor(a);
                break;
            case OP_MODULO: {
                float4 bv = b;
                bv.x = bv.x == 0.0f ? 1e-6f : bv.x;
                bv.y = bv.y == 0.0f ? 1e-6f : bv.y;
                bv.z = bv.z == 0.0f ? 1e-6f : bv.z;
                bv.w = bv.w == 0.0f ? 1e-6f : bv.w;
                float4 m = fmod(a, bv);
                m.x = m.x < 0.0f ? m.x + abs(bv.x) : m.x;
                m.y = m.y < 0.0f ? m.y + abs(bv.y) : m.y;
                m.z = m.z < 0.0f ? m.z + abs(bv.z) : m.z;
                m.w = m.w < 0.0f ? m.w + abs(bv.w) : m.w;
                r = m;
                break;
            }
            case OP_ABS:
                r = abs(a);
                break;
            case OP_CLAMP:
                r = clamp(a, b, c);
                break;
            case OP_POWER:
                r = pow(a, b);
                break;
            case OP_SQRT:
                r = sqrt(abs(a));
                break;
            case OP_MAX:
                r = max(a, b);
                break;
            case OP_MIN:
                r = min(a, b);
                break;
            case OP_SIN:
                r = sin(a);
                break;
            case OP_COS:
                r = cos(a);
                break;
            case OP_DOTPRODUCT:
                r.x = dot(a.xyz, b.xyz);
                break;
            case OP_CROSSPRODUCT:
                r.xyz = cross(a.xyz, b.xyz);
                break;
            case OP_LENGTH:
                r.x = length(a.xyz);
                break;
            case OP_DISTANCE:
                r.x = distance(a.xyz, b.xyz);
                break;
            case OP_NORMALIZE:
                r.xyz = normalize(a.xyz);
                break;
            case OP_MIX:
                r = lerp(a, b, c.x);
                break;
            case OP_NOISE2D: {
                float u_in = (node.flags & FLAG_HAS_A) ? a.x : u;
                float v_in = (node.flags & FLAG_HAS_A) ? a.y : v;
                int octaves = (int)node.value.x;
                float lacun = node.value.y;
                float gain = node.value.z;
                float scale = node.value.w;
                if (octaves < 1) octaves = 1;
                if (octaves > 8) octaves = 8;
                float val = fbm2D(u_in * scale, v_in * scale, octaves, lacun, gain);
                r = float4(val, val, val, 1.0f);
                break;
            }
            case OP_CHECKERBOARD: {
                float u_in = (node.flags & FLAG_HAS_A) ? a.x : u;
                float v_in = (node.flags & FLAG_HAS_A) ? a.y : v;
                float tx = node.value.x;
                float ty = node.value.y;
                bool check = ((int)floor(u_in * tx) + (int)floor(v_in * ty)) % 2 == 0;
                r = check ? b : c;
                break;
            }
            case OP_PLACE2D: {
                float u_in = (node.flags & FLAG_HAS_A) ? a.x : u;
                float v_in = (node.flags & FLAG_HAS_A) ? a.y : v;
                float2 offset = node.value.xy;
                float2 scale = node.value.zw;
                float rot = (node.flags & FLAG_HAS_B) ? b.x : 0.0f;
                float2 pivot = float2(0.5f, 0.5f);
                float2 uv = float2(u_in, v_in);
                float cosR = cos(rot);
                float sinR = sin(rot);
                float2 rv = uv - pivot;
                float2 rotUV = float2(rv.x * cosR - rv.y * sinR, rv.x * sinR + rv.y * cosR);
                rotUV += pivot;
                float2 outUV = (rotUV - pivot) * scale + pivot + offset;
                r = float4(outUV.x, outUV.y, 0.0f, 0.0f);
                break;
            }
            case OP_SWIZZLE: {
                uint sx = (node.flags >> 0) & 3;
                uint sy = (node.flags >> 2) & 3;
                uint sz = (node.flags >> 4) & 3;
                uint sw = (node.flags >> 6) & 3;
                float4 v = a;
                r.x = (sx == 0) ? v.x : (sx == 1) ? v.y : (sx == 2) ? v.z : v.w;
                r.y = (sy == 0) ? v.x : (sy == 1) ? v.y : (sy == 2) ? v.z : v.w;
                r.z = (sz == 0) ? v.x : (sz == 1) ? v.y : (sz == 2) ? v.z : v.w;
                r.w = (sw == 0) ? v.x : (sw == 1) ? v.y : (sw == 2) ? v.z : v.w;
                break;
            }
            case OP_COMBINE: {
                uint count = node.flags & 0xFF;
                r.x = a.x;
                r.y = count > 1 ? b.x : 0.0f;
                r.z = count > 2 ? c.x : 0.0f;
                r.w = count > 3 ? d.x : 1.0f;
                break;
            }
            case OP_EXTRACT: {
                uint comp = node.flags & 3;
                float val = a[comp];
                r = float4(val, val, val, val);
                break;
            }
            case OP_IFGREATER:
                r = a.x > b.x ? c : d;
                break;
            case OP_IFEQUAL:
                r = abs(a.x - b.x) < 1e-5f ? c : d;
                break;
            default:
                break;
        }

        regs[node.outReg] = r;
    }

    float4 final = regs[uniforms.outputReg];
    uint comp = uniforms.outputComponents;
    if (comp == 1) {
        final = float4(final.x, 0.0f, 0.0f, 1.0f);
    } else if (comp == 2) {
        final = float4(final.x, final.y, 0.0f, 1.0f);
    } else if (comp == 3) {
        final = float4(final.x, final.y, final.z, 1.0f);
    } else if (comp == 4) {
        final = float4(final.x, final.y, final.z, final.w);
    }

    uint idx = y * uniforms.width + x;
    outPixels[idx] = saturate(final);
}
