#version 450

// TAA resolve and the built-in temporal upscaler (Renderer, setTemporalAntiAliasing()
// / setRenderScale()), the Vulkan port of default.metal's taaFragment: the
// frame (input size) gathered around the output pixel, each sample weighted
// by the distance from its jittered sample to the output pixel's centre; the
// history (output size) reprojected along the G-buffer's motion and clamped
// to the neighbourhood's spread in YCoCg (variance clipping, against
// ghosting); blended, the frame weighing less where no sample fell near the
// pixel. Writes the output and the next history.

layout(set = 0, binding = 0) uniform texture2D current;
layout(set = 0, binding = 1) uniform texture2D history;
layout(set = 0, binding = 2) uniform texture2D motion;
layout(set = 0, binding = 3) uniform sampler   linearSampler;
layout(std140, set = 0, binding = 4) uniform TaaUniforms {
    vec2  inputSize;     // the frame's (internal) size in pixels
    vec2  jitter;        // how far this frame's jitter moved the geometry, in input pixels (x right, y down)
    float blend;         // the frame's weight where a sample is at the pixel (e.g. 0.1)
    float historyValid;  // 0: no history (first frame, cut, resize)
    vec2  outputSize;    // the output's size in pixels
} u;

layout(location = 0) in vec2 fragUV;  // (0, 0) = top-left (fxaa.vert)

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outHistory;

vec3 rgbToYCoCg(vec3 c) {
    return vec3(dot(c, vec3(0.25, 0.5, 0.25)), dot(c, vec3(0.5, 0.0, -0.5)), dot(c, vec3(-0.25, 0.5, -0.25)));
}
vec3 yCoCgToRgb(vec3 c) {
    return vec3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

void main() {
    // The output pixel's centre in input pixels, and the input pixel it's in.
    vec2 p = fragUV * u.inputSize;
    ivec2 centre = ivec2(floor(p));
    ivec2 limit = ivec2(u.inputSize) - 1;
    vec3 sum = vec3(0.0), m1 = vec3(0.0), m2 = vec3(0.0);
    float weights = 0.0, nearest = 0.0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            ivec2 q = clamp(centre + ivec2(dx, dy), ivec2(0), limit);
            vec3 c = texelFetch(sampler2D(current, linearSampler), q, 0).rgb;
            // Where its sample was in the unjittered image (the pixel saw
            // what was at -jitter), in output pixels.
            vec2 d = (vec2(q) + 0.5 - u.jitter - p) * (u.outputSize / u.inputSize);
            float w = exp(-2.29 * dot(d, d));
            sum += w * c;
            weights += w;
            nearest = max(nearest, w);
            vec3 t = rgbToYCoCg(c);
            m1 += t;
            m2 += t * t;
        }
    }
    vec3 c = weights > 1e-6 ? sum / weights : texelFetch(sampler2D(current, linearSampler), clamp(centre, ivec2(0), limit), 0).rgb;
    vec3 mean = m1 / 9.0;
    vec3 sigma = sqrt(max(m2 / 9.0 - mean * mean, 0.0));
    ivec2 motionSize = textureSize(sampler2D(motion, linearSampler), 0);
    ivec2 motionPixel = min(ivec2(fragUV * vec2(motionSize)), motionSize - 1);
    vec2 previousUV = fragUV + texelFetch(sampler2D(motion, linearSampler), motionPixel, 0).xy;
    vec3 result = c;
    if (u.historyValid > 0.5 && all(greaterThanEqual(previousUV, vec2(0.0))) && all(lessThanEqual(previousUV, vec2(1.0)))) {
        vec3 h = clamp(rgbToYCoCg(texture(sampler2D(history, linearSampler), previousUV).rgb), mean - 1.5 * sigma, mean + 1.5 * sigma);
        // Less of this frame where its nearest sample is far from the pixel.
        result = mix(yCoCgToRgb(h), c, clamp(u.blend * nearest * 2.0, 0.0, 1.0));
    }
    outColor = vec4(result, 1.0);
    outHistory = vec4(result, 1.0);
}
