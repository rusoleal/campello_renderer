#version 450

// campello_renderer default Vulkan fragment shader
//
// Shading model: Lambertian diffuse + ambient, fixed directional light.
// Samples the primitive's base color texture via bind group 0,
// multiplied by baseColorFactor from the material uniform buffer.
//
// Descriptor set 0:
//   binding 0 — texture2D  baseColorTexture
//   binding 1 — sampler    baseColorSampler

layout(set = 0, binding = 0) uniform texture2D baseColorTexture;
layout(set = 0, binding = 1) uniform sampler   baseColorSampler;

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec2 fragTexcoord;
layout(location = 2) in vec4 fragBaseColor;
layout(location = 3) in vec3 fragMaterialParams;

layout(location = 0) out vec4 outColor;

void main() {
    // Sample base color texture and multiply by baseColorFactor.
    vec4 texColor = texture(sampler2D(baseColorTexture, baseColorSampler), fragTexcoord);
    
    // Material params: [alphaMode, alphaCutoff, unlit]
    float alphaMode   = fragMaterialParams.x;
    float alphaCutoff = fragMaterialParams.y;
    float unlit       = fragMaterialParams.z;
    
    // Alpha mask: discard fragment when alphaMode == 1 (mask) and alpha < alphaCutoff
    if (alphaMode > 0.5 && alphaMode < 1.5 && texColor.a < alphaCutoff) {
        discard;
    }
    
    vec4 color = texColor * fragBaseColor;

    // KHR_materials_unlit: skip lighting when unlit flag is set
    if (unlit > 0.5) {
        outColor = color;
        return;
    }

    // Directional light pointing from upper-right-front (object space).
    vec3  lightDir = normalize(vec3(1.0, 2.0, 1.0));
    vec3  N        = normalize(fragNormal);

    float diffuse  = max(dot(N, lightDir), 0.0);
    float ambient  = 0.15;
    float light    = ambient + diffuse * (1.0 - ambient);

    outColor = vec4(color.rgb * light, color.a);
}
