// Mesh deformation for the vertex stages (default.metal's pbrVertex), in
// glTF's order: morph targets, then skeletal skinning. Only the SKINNED
// stages (default_skinned.vert, gbuffer_skinned.vert) deform; the renderer
// picks them for draws with a skin or morph targets. Without SKINNED,
// applyMorph() does nothing and skinMatrix() is the identity, so the plain
// stages read no joints, palette or deltas at all.
//
// Skinning: JOINTS_0 / WEIGHTS_0 (locations 4 and 5) blend up to four
// matrices of the draw's joint palette, a storage buffer at set 2 binding 0
// (column-major mat4 per joint, as Metal's [[buffer(20)]]). Weights summing
// to zero leave the vertex unskinned, like Metal.
//
// Morph targets: set 3. Binding 0 is the node's MorphInfo (target count,
// whether targets have normals, vertex count, up to 8 weights; Metal's
// [[buffer(21)]]), bindings 1 and 2 the primitive's POSITION and NORMAL
// deltas, target-major and tightly packed (3 floats per vertex; target t's
// delta for vertex v at t * vertexCount + v). A target count of 0 (the
// renderer's default MorphInfo) leaves the vertex as it is.

#ifdef SKINNED
layout(location = 4) in uvec4 joints;
layout(location = 5) in vec4 weights;

layout(std430, set = 2, binding = 0) readonly buffer JointMatrices {
    mat4 jointMatrices[];
};

layout(std430, set = 3, binding = 0) readonly buffer MorphInfo {
    float morphTargetCount;
    float morphHasNormal;
    float morphVertexCount;
    float morphWeights[8];
};
layout(std430, set = 3, binding = 1) readonly buffer MorphPositions {
    float morphPositions[];
};
layout(std430, set = 3, binding = 2) readonly buffer MorphNormals {
    float morphNormals[];
};

void applyMorph(inout vec3 p, inout vec3 n) {
    uint targets = min(uint(morphTargetCount), 8u);
    uint vertexCount = uint(morphVertexCount);
    bool hasNormal = morphHasNormal > 0.5;
    uint v = uint(gl_VertexIndex);
    for (uint t = 0u; t < targets; ++t) {
        float w = morphWeights[t];
        if (w == 0.0) continue;
        uint i = (t * vertexCount + v) * 3u;
        p += w * vec3(morphPositions[i], morphPositions[i + 1u], morphPositions[i + 2u]);
        if (hasNormal) n += w * vec3(morphNormals[i], morphNormals[i + 1u], morphNormals[i + 2u]);
    }
}

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
void applyMorph(inout vec3 p, inout vec3 n) {}
mat4 skinMatrix() { return mat4(1.0); }
#endif
