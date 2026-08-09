#version 450

// campello_renderer default Vulkan vertex shader
//
// Vertex slot contract (must match Renderer::VERTEX_SLOT_* constants):
//   location  0  POSITION      vec3   — object-space vertex position
//   location  1  NORMAL        vec3   — object-space vertex normal
//   location  2  TEXCOORD_0    vec2   — primary UV
//   location  3  TANGENT       vec4   — tangent + bitangent sign (w)
//   location 16  MVP           mat4   — row-major MVP, per-instance (binding 16)
//                                       occupies locations 16, 17, 18, 19
//   location 20  baseColorFactor vec4 — per-instance material (binding 17)
//   location 21  uvTransformRow0 vec4 — KHR_texture_transform row 0 [a, b, tx, 0]
//   location 22  uvTransformRow1 vec4 — KHR_texture_transform row 1 [c, d, ty, 0]
//

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texcoord0;
layout(location = 3) in vec4 tangent;

// A mat4 in a vertex shader input occupies 4 consecutive locations (16–19).
layout(location = 16) in mat4 mvp;

// Per-instance material uniforms from buffer slot 17 (stride = 256 bytes).
// Locations 20-23 follow the MVP (16-19) to avoid collision.
layout(location = 20) in vec4 baseColorFactor;
layout(location = 21) in vec4 uvTransformRow0; // [a, b, tx, hasTransform]
layout(location = 22) in vec4 uvTransformRow1; // [c, d, ty, 0]
layout(location = 23) in vec4 materialFlags;   // [alphaMode, alphaCutoff, unlit, 0]

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec2 fragTexcoord;
layout(location = 2) out vec4 fragBaseColor;
layout(location = 3) out vec3 fragMaterialParams;

void main() {
    // Renderer::computeNodeTransform() transposes the source row-major
    // (vector_math) matrix to column-major before upload — the same layout
    // for every backend, Metal included (see its "to column-major (Metal)
    // format" comment there) — so the buffer already holds M in the
    // column-major layout GLSL's mat4 expects. A standard column-vector
    // multiply is therefore correct; a compensating pos*mvp row-vector
    // multiply here would silently apply M^T instead of M.
    gl_Position   = mvp * vec4(position, 1.0);
    // Vulkan's clip-space Y points down; unlike Metal (whose native Y-down NDC
    // is absorbed transparently by drawable presentation), campello_gpu's
    // Vulkan backend does no compensating viewport flip anywhere (no negative-
    // height VkViewport trick), so it has to happen here or the whole frame
    // renders upside down.
    gl_Position.y = -gl_Position.y;

    // Apply KHR_texture_transform only when the flag (uvTransformRow0.w > 0.5) is set.
    if (uvTransformRow0.w > 0.5) {
        vec3 uv3 = vec3(texcoord0, 1.0);
        fragTexcoord = vec2(dot(uvTransformRow0.xyz, uv3),
                            dot(uvTransformRow1.xyz, uv3));
    } else {
        fragTexcoord = texcoord0;
    }

    fragNormal       = normal;
    fragBaseColor    = baseColorFactor;
    fragMaterialParams = materialFlags.xyz; // [alphaMode, alphaCutoff, unlit]
}
