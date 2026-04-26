#version 450

// FXAA fullscreen triangle vertex shader (Vulkan)
// No vertex inputs — position generated from gl_VertexIndex.

void main() {
    vec2 pos;
    if (gl_VertexIndex == 0) pos = vec2(-1.0, -1.0);
    else if (gl_VertexIndex == 1) pos = vec2(3.0, -1.0);
    else pos = vec2(-1.0, 3.0);
    gl_Position = vec4(pos, 1.0, 1.0);
}
