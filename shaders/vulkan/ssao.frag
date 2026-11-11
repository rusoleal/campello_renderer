#version 450

// Screen-space ambient occlusion, pass 1 (Metal ssaoFragment): scene depth ->
// AO at the output size. See shaders/metal/default.metal for the method.

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
    vec3 P = aoViewPos(fragUV, depth);
    vec2 px = 1.0 / u.size.zw;
    vec3 R = aoViewPos(fragUV + vec2(px.x, 0.0), aoDepthAt(fragUV + vec2(px.x, 0.0)));
    vec3 L = aoViewPos(fragUV - vec2(px.x, 0.0), aoDepthAt(fragUV - vec2(px.x, 0.0)));
    vec3 D = aoViewPos(fragUV + vec2(0.0, px.y), aoDepthAt(fragUV + vec2(0.0, px.y)));
    vec3 U = aoViewPos(fragUV - vec2(0.0, px.y), aoDepthAt(fragUV - vec2(0.0, px.y)));
    vec3 dx = abs(R.z - P.z) < abs(P.z - L.z) ? R - P : P - L;
    vec3 dy = abs(D.z - P.z) < abs(P.z - U.z) ? D - P : P - U;
    vec3 N = normalize(cross(dx, dy));
    if (dot(N, -P) < 0.0) N = -N;

    float radius = u.params.x;
    float bias = u.params.z;
    int samples = int(u.params.w);
    float noise = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
    float angle = noise * 6.28318530718;
    vec3 helper = abs(N.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(helper, N));
    vec3 B = cross(N, T);
    vec3 Tr = T * cos(angle) + B * sin(angle);
    vec3 Br = cross(N, Tr);

    float occlusion = 0.0;
    for (int i = 0; i < samples; ++i) {
        float t = (float(i) + 0.5) / float(samples);
        float phi = float(i) * 2.39996323;
        float cosTheta = sqrt(1.0 - t * 0.9);
        float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
        float scale = mix(0.15, 1.0, t * t);
        vec3 k = vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta) * scale;
        vec3 S = P + (Tr * k.x + Br * k.y + N * k.z) * radius;
        vec4 c = u.proj * vec4(S, 1.0);
        vec2 suv = vec2(c.x / c.w * 0.5 + 0.5, 0.5 - c.y / c.w * 0.5);
        if (any(lessThan(suv, vec2(0.0))) || any(greaterThan(suv, vec2(1.0)))) continue;
        float sceneDepth = aoDepthAt(suv);
        if (sceneDepth >= 1.0) continue;
        vec3 Q = aoViewPos(suv, sceneDepth);
        float range = smoothstep(0.0, 1.0, radius / max(abs(P.z - Q.z), 1e-4));
        occlusion += (abs(Q.z) <= abs(S.z) - bias ? 1.0 : 0.0) * range;
    }
    float ao = pow(clamp(1.0 - occlusion / float(max(samples, 1)), 0.0, 1.0), u.params.y);
    outColor = vec4(ao, ao, ao, 1.0);
}
