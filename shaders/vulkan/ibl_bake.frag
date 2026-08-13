#version 450

// campello_renderer IBL precompute shader (Vulkan) — bakes the three
// environment-independent/environment-derived resources the glTF-Sample-Renderer
// reference uses for physically correct image-based lighting, none of which
// this renderer previously computed:
//
//   mode 0: BRDF LUT       — Karis split-sum (scale, bias) integral, indexed by
//                             (NdotV, roughness). Environment-independent; baked once.
//   mode 1: GGX prefilter  — one face of one mip level of the specular environment,
//                             importance-sampled from the source (mip-0, unfiltered)
//                             cubemap at this mip's roughness = mip / (mipCount-1).
//   mode 2: Irradiance     — one face of the small diffuse irradiance cubemap,
//                             cosine-weighted convolution of the source cubemap.
//
// Rendered via a fullscreen triangle (reuses fxaaVertex's gl_VertexIndex-based
// vertex shader — no vertex inputs needed) into a single-face/single-mip 2D
// view of the destination texture, one draw call per (mode, face, mip).

layout(std140, set = 0, binding = 0) uniform IblBakeUniforms {
    int mode;          // 0 = BRDF LUT, 1 = GGX prefilter, 2 = irradiance convolution
    int faceIndex;     // 0..5, used for mode 1/2
    float roughness;   // used for mode 1
    float outputSize;  // resolution (texels, square) of the current render target
} u;

layout(set = 0, binding = 1) uniform textureCube srcEnvMap;
layout(set = 0, binding = 2) uniform sampler     srcEnvSampler;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265359;

// Direction for cubemap face `face` at signed uv in [-1,1] — must match the
// CPU-side convention in Renderer::convertEquirectangularImageToCubemap()
// exactly (right-handed, +Z forward, face order +X,-X,+Y,-Y,+Z,-Z), so baked
// faces stay aligned with faces uploaded via the equirect/6-file loaders.
vec3 faceDirection(int face, float uu, float vv) {
    vec3 d;
    if (face == 0)      d = vec3( 1.0, -vv, -uu);
    else if (face == 1) d = vec3(-1.0, -vv,  uu);
    else if (face == 2) d = vec3( uu,  1.0,  vv);
    else if (face == 3) d = vec3( uu, -1.0, -vv);
    else if (face == 4) d = vec3( uu, -vv,  1.0);
    else                 d = vec3(-uu, -vv, -1.0);
    return normalize(d);
}

// Van der Corput radical inverse (base 2) + Hammersley low-discrepancy sequence —
// standard quasi-Monte-Carlo sampling for GGX importance sampling (Karis 2013,
// "Real Shading in Unreal Engine 4").
float radicalInverseVdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

vec2 hammersley(uint i, uint n) {
    return vec2(float(i) / float(n), radicalInverseVdC(i));
}

vec3 importanceSampleGGX(vec2 xi, vec3 N, float roughness) {
    float a = roughness * roughness;

    float phi = 2.0 * PI * xi.x;
    float cosTheta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    vec3 H;
    H.x = cos(phi) * sinTheta;
    H.y = sin(phi) * sinTheta;
    H.z = cosTheta;

    vec3 up = (abs(N.z) < 0.999) ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up, N));
    vec3 bitangent = cross(N, tangent);

    return normalize(tangent * H.x + bitangent * H.y + N * H.z);
}

float geometrySchlickGGX_IBL(float NdotV, float roughness) {
    float a = roughness;
    float k = (a * a) / 2.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float geometrySmith_IBL(float NdotV, float NdotL, float roughness) {
    return geometrySchlickGGX_IBL(NdotV, roughness) * geometrySchlickGGX_IBL(NdotL, roughness);
}

// Karis split-sum BRDF LUT integration — the standard closed-form approximation
// (UE4 2013 course notes) also used, in spirit, by the reference's own offline
// ibl_sampler.js LUT bake (sampled here at (NdotV, roughness) instead of stored
// as a shipped asset, since this renderer has no offline asset pipeline).
vec2 integrateBRDF(float NdotV, float roughness) {
    vec3 V = vec3(sqrt(max(1.0 - NdotV * NdotV, 0.0)), 0.0, NdotV);
    vec3 N = vec3(0.0, 0.0, 1.0);

    float A = 0.0;
    float B = 0.0;
    const uint SAMPLE_COUNT = 256u;
    for (uint i = 0u; i < SAMPLE_COUNT; ++i) {
        vec2 xi = hammersley(i, SAMPLE_COUNT);
        vec3 H = importanceSampleGGX(xi, N, roughness);
        vec3 L = normalize(2.0 * dot(V, H) * H - V);

        float NdotL = max(L.z, 0.0);
        float NdotH = max(H.z, 0.0);
        float VdotH = max(dot(V, H), 0.0);

        if (NdotL > 0.0) {
            float G = geometrySmith_IBL(NdotV, NdotL, roughness);
            float G_Vis = (G * VdotH) / max(NdotH * NdotV, 0.0001);
            float Fc = pow(1.0 - VdotH, 5.0);

            A += (1.0 - Fc) * G_Vis;
            B += Fc * G_Vis;
        }
    }
    return vec2(A, B) / float(SAMPLE_COUNT);
}

// GGX-importance-sampled prefiltered specular radiance for direction N at the
// given roughness — Karis's split-sum "pre-filtered environment map" half.
vec3 prefilterEnvironment(vec3 N, float roughness) {
    // At roughness ~0 the GGX lobe collapses to a delta function (H = N for
    // every sample regardless of xi — see importanceSampleGGX's cosTheta
    // formula), so importance sampling has already converged to the exact
    // answer: a single direct sample. Skipping the loop here removes the
    // single most expensive mip (mip 0, the source's full resolution) from
    // the bake entirely, which is what makes raising SAMPLE_COUNT below
    // affordable for the mips that actually need it.
    if (roughness < 0.01) {
        return textureLod(samplerCube(srcEnvMap, srcEnvSampler), N, 0.0).rgb;
    }

    vec3 V = N;
    vec3 prefilteredColor = vec3(0.0);
    float totalWeight = 0.0;
    // 64 samples produced visible Monte-Carlo fireflies/noise wherever a
    // material's local roughness landed on a mid mip level reflecting a
    // bright part of the environment (the built-in default env is baked with
    // a 6x exposure boost, so its highlights are well above 1.0) — this is a
    // one-time bake, not a per-frame cost, so a much higher sample count is
    // affordable, especially with the roughness~0 fast path above removing
    // the largest/most expensive mip from needing it at all.
    const uint SAMPLE_COUNT = 512u;
    for (uint i = 0u; i < SAMPLE_COUNT; ++i) {
        vec2 xi = hammersley(i, SAMPLE_COUNT);
        vec3 H = importanceSampleGGX(xi, N, roughness);
        vec3 L = normalize(2.0 * dot(V, H) * H - V);
        float NdotL = max(dot(N, L), 0.0);
        if (NdotL > 0.0) {
            // Explicit mip 0 (not implicit texture()/derivative-based LOD):
            // L is a per-fragment, importance-sampled Monte Carlo direction
            // with no smooth screen-space continuity across the fullscreen
            // triangle, so automatic derivative-based mip selection has no
            // valid basis to work from and effectively picks near-arbitrary
            // mip levels — this was the actual source of the "reflective
            // noise"/wrong-region-color bug (the source env's higher mips
            // are also box-filtered downsamples, so accidentally sampling
            // one blends in nearby-in-cube-space but wrong-in-direction
            // content).
            prefilteredColor += textureLod(samplerCube(srcEnvMap, srcEnvSampler), L, 0.0).rgb * NdotL;
            totalWeight += NdotL;
        }
    }
    return prefilteredColor / max(totalWeight, 0.0001);
}

// Cosine-weighted hemisphere convolution of the source environment around
// normal N — the Lambertian diffuse irradiance integral the reference bakes
// offline into u_LambertianEnvSampler (ibl_sampler.js), computed here on GPU.
//
// Cosine-weighted importance sampling via Hammersley (same technique as
// prefilterEnvironment/integrateBRDF above) rather than a structured
// (phi,theta) grid walked with two nested float-indexed loops: the nested
// float-loop version produced visibly speckled/incoherent output on this
// hardware (Intel Mesa ANV) despite being mathematically standard — some
// combination of loop trip count and texture sampling inside doubly-nested
// non-uint-indexed control flow miscompiled. A single uint-indexed loop
// matching this file's other two integrals sidesteps whatever that was.
// With cosine-weighted sampling the pdf (cosTheta/PI) exactly cancels the
// integrand's own cosTheta factor, leaving a plain PI * average(L) — see
// e.g. https://learnopengl.com/PBR/IBL/Diffuse-irradiance's importance-
// sampled variant.
vec3 convolveIrradiance(vec3 N) {
    vec3 up = (abs(N.z) < 0.999) ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up, N));
    vec3 bitangent = cross(N, tangent);

    vec3 irradiance = vec3(0.0);
    const uint SAMPLE_COUNT = 16384u;
    for (uint i = 0u; i < SAMPLE_COUNT; ++i) {
        vec2 xi = hammersley(i, SAMPLE_COUNT);
        float phi = 2.0 * PI * xi.x;
        float cosTheta = sqrt(1.0 - xi.y);
        float sinTheta = sqrt(xi.y);
        vec3 localDir = vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
        vec3 sampleDir = localDir.x * tangent + localDir.y * bitangent + localDir.z * N;
        irradiance += textureLod(samplerCube(srcEnvMap, srcEnvSampler), sampleDir, 0.0).rgb;
    }
    return PI * irradiance / float(SAMPLE_COUNT);
}

void main() {
    if (u.mode == 0) {
        vec2 uv = gl_FragCoord.xy / u.outputSize;
        outColor = vec4(integrateBRDF(clamp(uv.x, 0.001, 1.0), clamp(uv.y, 0.001, 1.0)), 0.0, 1.0);
        return;
    }

    // Modes 1/2 render into one face of a cubemap: convert this fragment's
    // pixel coordinate to a signed [-1,1] uv, then to a world direction via
    // the same per-face basis the CPU-side equirect loader uses.
    vec2 ndc = (gl_FragCoord.xy / u.outputSize) * 2.0 - 1.0;
    vec3 dir = faceDirection(u.faceIndex, ndc.x, ndc.y);

    if (u.mode == 1) {
        outColor = vec4(prefilterEnvironment(dir, u.roughness), 1.0);
    } else {
        outColor = vec4(convolveIrradiance(dir), 1.0);
    }
}
