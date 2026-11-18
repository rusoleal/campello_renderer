// Sun shadow casters that deform (encodeShadowPass()): depth only. The
// shadow pass binds the light's clip from local as the per-draw mvp
// (locations 16-19); each instance's matrix (DrawCall::instanceTransformBuffer)
// is a per-instance attribute (locations 12-15). shadow_skinned.vert adds
// the joint palette (set 2, skinning.glsl). Plain casters use default_lean.vert.
#include "skinning.glsl"

layout(location = 0) in vec3 position;
layout(location = 12) in mat4 instanceMatrix;
layout(location = 16) in mat4 mvp;

void main() {
    gl_Position   = mvp * (instanceMatrix * (skinMatrix() * vec4(position, 1.0)));
    gl_Position.y = -gl_Position.y;  // as default.vert
}
