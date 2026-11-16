// Skeletal skinning for the vertex stages (default.metal's pbrVertex): with
// SKINNED defined, JOINTS_0 / WEIGHTS_0 (locations 4 and 5) blend up to four
// matrices of the draw's joint palette, a storage buffer at set 2 binding 0
// (column-major mat4 per joint, as Metal's [[buffer(20)]]). Weights summing to
// zero leave the vertex unskinned, like Metal. Without SKINNED, skinMatrix()
// is the identity, so the unskinned stages read no joints at all.

#ifdef SKINNED
layout(location = 4) in uvec4 joints;
layout(location = 5) in vec4 weights;

layout(std430, set = 2, binding = 0) readonly buffer JointMatrices {
    mat4 jointMatrices[];
};

mat4 blendJoints(uvec4 j, vec4 w) {
    if (w.x + w.y + w.z + w.w <= 0.001) return mat4(1.0);
    mat4 m = mat4(0.0);
    for (int i = 0; i < 4; ++i) {
        if (w[i] > 0.0) m += w[i] * jointMatrices[j[i]];
    }
    return m;
}

mat4 skinMatrix() { return blendJoints(joints, weights); }
#else
mat4 skinMatrix() { return mat4(1.0); }
#endif
