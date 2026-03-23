#version 450

// campello_renderer default Vulkan vertex shader
//
// Vertex slot contract (must match Renderer::VERTEX_SLOT_* constants):
//   location  0  POSITION   vec3  — object-space vertex position
//   location  1  NORMAL     vec3  — object-space vertex normal
//   location  2  TEXCOORD_0 vec2  — primary UV
//   location  3  TANGENT    vec4  — tangent + bitangent sign (w)
//   location 16  MVP        mat4  — row-major MVP, per-instance (binding 16)
//                                   occupies locations 16, 17, 18, 19
//
// Note: campello_gpu Vulkan backend does not yet implement vertex input
// descriptors (VkVertexInputBindingDescription / VkVertexInputAttributeDescription
// are hardcoded to 0). This shader is correct GLSL and will function once that
// upstream gap is resolved.

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texcoord0;
layout(location = 3) in vec4 tangent;

// A mat4 in a vertex shader input occupies 4 consecutive locations (16–19).
layout(location = 16) in mat4 mvp;

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec2 fragTexcoord;

void main() {
    // The MVP is stored row-major in campello_renderer (Matrix4<double> serialised
    // as float[16] with data[row*4+col]). Vulkan/GLSL mat4 is column-major, so
    // the loaded layout is the transpose of the intended matrix.
    // A row-vector multiply (pos * mvp) recovers the correct transform:
    //   vec4 * mat4  →  result[j] = dot(pos, mvp column j)
    //   With mvp = M^T  →  result = M * pos_col  (correct clip-space position).
    gl_Position  = vec4(position, 1.0) * mvp;
    gl_Position.y = -gl_Position.y; // Vulkan NDC has Y pointing down; flip to match GLTF/OpenGL convention
    fragNormal   = normal;
    fragTexcoord = texcoord0;
}
