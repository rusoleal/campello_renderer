#pragma once

// App-defined shader programs (docs/design/programmable-pipelines.md,
// phase 3). An app registers compiled shaders with
// Renderer::registerProgram() and gets a ProgramHandle; materials select a
// program (GpuMaterial::program) and Renderer::renderProgramPass() draws a
// whole scene with one (e.g. element ids into an r32uint target for GPU
// picking). The built-in PBR shaders are program kPbrProgram.
//
// Shader contract (the same one the built-in shaders follow; see
// inc/campello_renderer/shaders/ for an include file per language):
//  - Vertex attributes at location = VertexSemantic (position 0 ..
//    texcoord1 7, custom0..3 8-11). The renderer places them in compact
//    vertex buffer slots; shaders only see locations.
//  - Per-draw transforms (NodeTransforms {float4x4 mvp; float4x4 model;},
//    row-major on the CPU, uploaded column-major): Metal [[buffer(16)]]
//    (index 0); Vulkan/DirectX per-instance attributes at locations
//    16-19 (mvp columns) and 24-27 (model columns).
//  - Programs use the built-in pipeline layout, so they may also read the
//    material and frame resources at the built-in bindings (see the
//    headers); they don't have to.
//  - Culling follows GpuMaterial::doubleSided; blending follows
//    GpuMaterial::alphaBlend only if ShaderProgramDesc::blending is set.

#include <campello_renderer/vertex_layout.hpp>

#include <campello_gpu/constants/pixel_format.hpp>

#include <array>
#include <cstdint>
#include <string>

namespace systems::leal::campello_renderer {

    using ProgramHandle = uint32_t;
    /// The built-in PBR program (glTF materials).
    inline constexpr ProgramHandle kPbrProgram = 0;
    inline constexpr ProgramHandle kInvalidProgram = 0xFFFFFFFFu;

    /// A compiled shader binary (not copied: the bytes need only live until
    /// registerProgram() returns).
    struct ShaderBinary {
        const void* data = nullptr;
        uint64_t size = 0;
        bool empty() const { return data == nullptr || size == 0; }
    };

    struct ShaderProgramDesc {
        std::string name;
        // Compiled shaders per backend. registerProgram() uses the one for
        // the running backend and fails (kInvalidProgram) if it's missing.
        ShaderBinary metalLibrary;            // .metallib with both entry points
        std::string metalVertexEntryPoint = "vertexMain";
        std::string metalFragmentEntryPoint = "fragmentMain";
        ShaderBinary vulkanVertex;            // SPIR-V
        ShaderBinary vulkanFragment;
        std::string vulkanVertexEntryPoint = "main";
        std::string vulkanFragmentEntryPoint = "main";
        ShaderBinary directxVertex;           // DXIL
        ShaderBinary directxFragment;
        std::string directxVertexEntryPoint = "vertexMain";
        std::string directxFragmentEntryPoint = "pixelMain";
        std::string wgsl;                     // reserved for the WebGPU backend
        /// The vertex attributes the vertex shader reads (semanticBit()s).
        /// A mesh is drawn only if its layout has all of them.
        uint32_t vertexReads = semanticBit(VertexSemantic::position);
        /// Whether alpha-blended materials draw with blending (the color
        /// target must be blendable, e.g. not an integer format).
        bool blending = false;
    };

    /// Renderer::renderProgramPass(): one program for every draw, into a
    /// target of the given format, cleared to clearColor (integer targets
    /// take the values as integers). No skybox, anti-aliasing or
    /// transmission; depth-tested with its own depth buffer.
    struct ProgramPassOptions {
        ProgramHandle program = kPbrProgram;
        systems::leal::campello_gpu::PixelFormat colorFormat =
            systems::leal::campello_gpu::PixelFormat::rgba8unorm;
        std::array<double, 4> clearColor{0.0, 0.0, 0.0, 0.0};
        bool includeTransparent = true;
    };

} // namespace systems::leal::campello_renderer
