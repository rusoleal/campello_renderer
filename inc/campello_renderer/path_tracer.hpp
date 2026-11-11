#pragma once

// Progressive path tracing of a RenderScene (presentation images): on GPUs
// where campello_gpu reports Feature::raytracing, and the Metal backend for
// now. Each render() adds one sample per pixel and shows the running
// average, tone mapped like the raster renderer; anything that changes the
// image (geometry, materials, lights, camera, size) starts it again.
//
//   PathTracer tracer(device);
//   if (tracer.supported()) {
//       while (tracer.sampleCount() < 256) tracer.render(scene, renderer, w, h);
//       show(tracer.output());
//   }
//
// What it traces: the scene's opaque draws (RenderScene::shadowCasters when
// filled, so geometry off screen still shadows and reflects) and its
// transparent draws, triangle meshes with float positions. Materials use
// their constants (base colour, roughness, metallic, emissive, transmission
// as thin glass, unlit); a base colour texture contributes its average
// colour (textures in the path tracer: later). Light: up to four directional
// lights (soft discs 0.53 degrees across, as the sun) and the environment
// map times the renderer's environment intensity (seen directly at full
// strength, as the skybox); the renderer's exposure applies.

#include <campello_renderer/campello_renderer.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

namespace systems::leal::campello_gpu {
class AccelerationStructure;
class Buffer;
class ComputePipeline;
class RayTracingPipeline;
class BindGroupLayout;
class Texture;
} // namespace systems::leal::campello_gpu

namespace systems::leal::campello_renderer {

class PathTracer {
public:
    explicit PathTracer(std::shared_ptr<systems::leal::campello_gpu::Device> device);
    ~PathTracer();

    /// Whether this device can path trace (ray tracing, a backend with the
    /// path tracing shaders: Metal).
    [[nodiscard]] bool supported() const noexcept {
        return pipeline_ != nullptr;
    }

    /// Adds one sample per pixel of `scene` at `width` x `height` to
    /// output(); starts again when the scene, its materials, lights,
    /// camera or the size changed. `renderer` gives materials, environment,
    /// exposure and clear colour. Returns false if nothing was traced.
    bool render(const RenderScene& scene, const Renderer& renderer, uint32_t width, uint32_t height);

    /// The image so far (rgba8unorm, width x height; display-encoded).
    [[nodiscard]] std::shared_ptr<systems::leal::campello_gpu::Texture> output() const noexcept {
        return output_;
    }
    /// Samples per pixel in output().
    [[nodiscard]] uint32_t sampleCount() const noexcept {
        return samples_;
    }
    /// Starts the image again on the next render().
    void reset() noexcept {
        samples_ = 0;
    }
    /// Light bounces after the first hit (default 4).
    void setMaxBounces(uint32_t bounces) noexcept {
        maxBounces_ = bounces;
        samples_ = 0;
    }

private:
    struct MeshEntry {
        std::shared_ptr<systems::leal::campello_gpu::AccelerationStructure> blas;
        uint32_t triangleCount = 0;
    };
    bool rebuild(const RenderScene& scene, const Renderer& renderer, uint64_t sceneKey);
    std::vector<DrawCall> traced(const RenderScene& scene) const;

    std::shared_ptr<systems::leal::campello_gpu::Device> device_;
    std::shared_ptr<systems::leal::campello_gpu::RayTracingPipeline> pipeline_;
    std::shared_ptr<systems::leal::campello_gpu::ComputePipeline> normalsPipeline_;
    std::shared_ptr<systems::leal::campello_gpu::BindGroupLayout> traceLayout_;
    std::shared_ptr<systems::leal::campello_gpu::BindGroupLayout> normalsLayout_;
    std::map<const GpuMesh*, MeshEntry> meshes_;  // acceleration structure per mesh
    std::shared_ptr<systems::leal::campello_gpu::AccelerationStructure> tlas_;
    std::shared_ptr<systems::leal::campello_gpu::Buffer> instances_;
    std::shared_ptr<systems::leal::campello_gpu::Buffer> materials_;
    std::shared_ptr<systems::leal::campello_gpu::Buffer> normals_;
    std::shared_ptr<systems::leal::campello_gpu::Buffer> uniforms_;
    std::shared_ptr<systems::leal::campello_gpu::Buffer> accum_;
    std::shared_ptr<systems::leal::campello_gpu::Texture> output_;
    uint64_t sceneKey_ = 0;
    uint64_t viewKey_ = 0;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    uint32_t samples_ = 0;
    uint32_t maxBounces_ = 4;
};

} // namespace systems::leal::campello_renderer
