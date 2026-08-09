#version 450

// campello_renderer default Vulkan vertex shader — full PBR pipeline.
//
// Vertex slot contract (must match Renderer::VERTEX_SLOT_* constants):
//   location  0  POSITION      vec3   — object-space vertex position
//   location  1  NORMAL        vec3   — object-space vertex normal
//   location  2  TEXCOORD_0    vec2   — primary UV
//   location  3  TANGENT       vec4   — tangent (bitangent sign in .w is unused,
//                                       matching Metal's plain cross(N,T) — see main())
//   location  6  COLOR_0       vec4   — vertex color, (1,1,1,1) fallback
//   location  7  TEXCOORD_1    vec2   — secondary UV, (0,0) fallback
//   location 16  MVP           mat4   — per-node clip-space transform (binding 16,
//                                       locations 16-19)
//   location 24  MODEL         mat4   — per-node world-space transform, same binding
//                                       16 buffer at byte offset 64 (locations 24-27)
//
// Only geometric interpolants are output here — material data (MaterialUniforms)
// is read directly by the fragment shader from a UBO (set 0, binding 24), matching
// Metal's architecture (constant MaterialUniforms &mat [[buffer(17)]]) instead of
// being smuggled through vertex attributes as an earlier placeholder version did.
//
// KHR_texture_transform UV math is deliberately NOT done here (unlike that earlier
// placeholder) — it's computed in the fragment shader instead. Affine 2D transforms
// commute with barycentric interpolation (weights always sum to 1), so applying the
// transform before or after interpolation is bit-for-bit equivalent; doing it in the
// fragment stage means MaterialUniforms only needs fragment-stage visibility.
//
// Skinning (JOINTS_0/WEIGHTS_0) and morph targets are not implemented here — a
// pre-existing gap versus Metal's vertexMain, not a new regression (the placeholder
// this replaces didn't support them either).

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texcoord0;
layout(location = 3) in vec4 tangent;
layout(location = 6) in vec4 color0;
layout(location = 7) in vec2 texcoord1;

// A mat4 in a vertex shader input occupies 4 consecutive locations.
layout(location = 16) in mat4 mvp;
layout(location = 24) in mat4 model;

layout(location = 0) out vec3 worldPos;
layout(location = 1) out vec3 worldNormal;
layout(location = 2) out vec3 worldTangent;
layout(location = 3) out vec3 worldBitangent;
layout(location = 4) out vec2 fragTexcoord0;
layout(location = 5) out vec2 fragTexcoord1;
layout(location = 6) out vec4 fragColor0;

void main() {
    // Renderer::computeNodeTransform()/uploadOneTransform() transpose the source
    // row-major (vector_math) matrices to column-major before upload — the same
    // layout for every backend, Metal included — so the buffer already holds M in
    // the column-major layout GLSL's mat4 expects. A standard column-vector
    // multiply is therefore correct.
    gl_Position   = mvp * vec4(position, 1.0);
    // Vulkan's clip-space Y points down; unlike Metal (whose native Y-down NDC
    // is absorbed transparently by drawable presentation), campello_gpu's
    // Vulkan backend does no compensating viewport flip anywhere (no negative-
    // height VkViewport trick), so it has to happen here or the whole frame
    // renders upside down.
    gl_Position.y = -gl_Position.y;

    vec4 worldPos4 = model * vec4(position, 1.0);
    worldPos = worldPos4.xyz;

    vec3 N = normalize((model * vec4(normal, 0.0)).xyz);

    // Tangent frame: synthesize an arbitrary tangent orthogonal to N when the
    // mesh has no real TANGENT attribute (near-zero length), otherwise
    // transform the vertex tangent by the model matrix and re-orthogonalize
    // against N (Gram-Schmidt) — mirrors Metal's vertexMain exactly, including
    // not using tangent.w to flip the bitangent sign.
    vec3 T;
    if (length(tangent.xyz) < 0.001) {
        if (abs(N.y) < 0.999) {
            T = normalize(cross(vec3(0.0, 1.0, 0.0), N));
        } else {
            T = normalize(cross(vec3(1.0, 0.0, 0.0), N));
        }
    } else {
        T = normalize((model * vec4(tangent.xyz, 0.0)).xyz);
        T = normalize(T - dot(T, N) * N);
    }
    vec3 B = cross(N, T);

    worldNormal    = N;
    worldTangent   = T;
    worldBitangent = B;
    fragTexcoord0  = texcoord0;
    fragTexcoord1  = texcoord1;
    fragColor0     = color0;
}
