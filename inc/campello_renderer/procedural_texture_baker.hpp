#pragma once

#include <gltf/gltf.hpp>
#include <vector>
#include <cstdint>
#include <string>

namespace systems::leal::campello_renderer {

/**
 * Bake a KHR_texture_procedurals graph output into an RGBA8 CPU pixel buffer.
 *
 * @param asset   The glTF asset (used for image sampling and graph definitions).
 * @param graph   The procedural graph to evaluate.
 * @param outputName  Name of the graph output to bake.
 * @param width   Desired bake width in pixels.
 * @param height  Desired bake height in pixels.
 * @return Row-major RGBA8 pixel data (size = width * height * 4).
 */
std::vector<uint8_t> bakeProceduralTexture(
    const systems::leal::gltf::GLTF& asset,
    const systems::leal::gltf::ProceduralGraph& graph,
    const std::string& outputName,
    int width, int height);

} // namespace systems::leal::campello_renderer

// Forward declaration — avoids pulling campello_gpu headers into public API.
namespace systems::leal::campello_gpu {
    class Device;
}

namespace systems::leal::campello_renderer {

/**
 * GPU-accelerated procedural texture baker using a pre-compiled uber-shader.
 *
 * Falls back to CPU baking if the graph contains unsupported nodes (e.g. `image`)
 * or if GPU pipeline creation fails.
 *
 * @param device  campello_gpu device used for compute dispatch.
 * @return Row-major RGBA8 pixel data (size = width * height * 4).
 */
std::vector<uint8_t> bakeProceduralTextureGPU(
    const std::shared_ptr<systems::leal::campello_gpu::Device>& device,
    const systems::leal::gltf::GLTF& asset,
    const systems::leal::gltf::ProceduralGraph& graph,
    const std::string& outputName,
    int width, int height);

} // namespace systems::leal::campello_renderer
