// The sky for WebGPU: the environment drawn behind the scene (fs_skybox, as
// Metal's skyboxFragment). vs_sky_fullscreen draws one triangle at depth 1.

@vertex
fn vs_sky_fullscreen(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
    var pos = vec2f(-1.0, -1.0);
    if (index == 1u) { pos = vec2f(3.0, -1.0); }
    if (index == 2u) { pos = vec2f(-1.0, 3.0); }
    return vec4f(pos, 1.0, 1.0);
}

// --- Sky ---------------------------------------------------------------------

struct SkyboxUniforms {
    invVP: mat4x4f,
    screenSize: vec2f,
    pad: vec2f,        // x = exposure in EV (as CameraUniforms.viewParams.x)
    cameraPos: vec4f,
}

@group(0) @binding(0) var skyEnvMap: texture_cube<f32>;
@group(0) @binding(1) var skyEnvSampler: sampler;
@group(0) @binding(2) var<uniform> sky: SkyboxUniforms;

fn toneMapKhronosPbrNeutral(colorIn: vec3f) -> vec3f {
    let startCompression = 0.8 - 0.04;
    let desaturation = 0.15;
    var color = colorIn;
    let x = min(color.r, min(color.g, color.b));
    let offset = select(0.04, x - 6.25 * x * x, x < 0.08);
    color -= offset;
    let peak = max(color.r, max(color.g, color.b));
    if (peak < startCompression) {
        return color;
    }
    let d = 1.0 - startCompression;
    let newPeak = 1.0 - d * d / (peak + d - startCompression);
    color *= newPeak / peak;
    let g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
    return mix(color, vec3f(newPeak), g);
}

fn linearToSRGBFast(color: vec3f) -> vec3f {
    return pow(max(color, vec3f(0.0)), vec3f(1.0 / 2.2));
}

@fragment
fn fs_skybox(@builtin(position) position: vec4f) -> @location(0) vec4f {
    let ndc = vec2f((position.x / sky.screenSize.x) * 2.0 - 1.0, (1.0 - position.y / sky.screenSize.y) * 2.0 - 1.0);
    let worldFar = sky.invVP * vec4f(ndc, 1.0, 1.0);
    let worldDir = normalize(worldFar.xyz / worldFar.w - sky.cameraPos.xyz);
    var color = textureSample(skyEnvMap, skyEnvSampler, worldDir).rgb;
    color *= exp2(sky.pad.x);  // exposure
    return vec4f(linearToSRGBFast(toneMapKhronosPbrNeutral(color)), 1.0);
}
