#version 450

// Screen-space ambient occlusion, pass 2 (Metal ssaoApplyFragment): the AO
// blurred 4x4 (depth-aware), multiplied into the frame by the blend state.

layout(set = 0, binding = 0) uniform texture2D depthTex;
layout(set = 0, binding = 1) uniform texture2D aoTex;
layout(std140, set = 0, binding = 2) uniform AoUniforms {
    mat4 proj;     // view -> clip
    mat4 invProj;  // clip -> view
    vec4 params;   // x = radius (view units), y = intensity (power), z = bias, w = samples
    vec4 size;     // depth texture width, height; output width, height
} u;
layout(set = 0, binding = 3) uniform sampler pointSampler;

layout(location = 0) in vec2 fragUV;
layout(location = 0) out vec4 outColor;

float aoDepthAt(vec2 uv) {
    ivec2 texel = ivec2(clamp(uv * u.size.xy, vec2(0.0), u.size.xy - 1.0));
    return texelFetch(sampler2D(depthTex, pointSampler), texel, 0).r;
}

vec3 aoViewPos(vec2 uv, float depth) {
    vec4 p = u.invProj * vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    return p.xyz / p.w;
}

void main() {
    float depth = aoDepthAt(fragUV);
    if (depth >= 1.0) {
        outColor = vec4(1.0);
        return;
    }
    float z = aoViewPos(fragUV, depth).z;
    ivec2 centre = ivec2(fragUV * u.size.zw);
    ivec2 limit = ivec2(u.size.zw) - 1;
    vec2 px = 1.0 / u.size.zw;
    float sum = 0.0;
    float weight = 0.0;
    for (int y = -2; y < 2; ++y) {
        for (int x = -2; x < 2; ++x) {
            ivec2 at = clamp(centre + ivec2(x, y), ivec2(0), limit);
            vec2 uv = (vec2(at) + 0.5) * px;
            float d = aoDepthAt(uv);
            float sz = aoViewPos(uv, d).z;
            float w = (d < 1.0 && abs(sz - z) < 0.05 * abs(z) + u.params.z) ? 1.0 : 0.0;
            sum += texelFetch(sampler2D(aoTex, pointSampler), at, 0).r * w;
            weight += w;
        }
    }
    float ao = weight > 0.0 ? sum / weight : 1.0;
    outColor = vec4(ao, ao, ao, 1.0);
}
