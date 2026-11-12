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
// as thin glass, unlit) and base colour textures (up to 64 distinct ones;
// through TEXCOORD_0 and KHR_texture_transform, or the renderer's world
// mapping; meshes without coordinates, and textures beyond 64, shade with
// the texture's average colour). Light: up to four directional
// lights (soft discs 0.53 degrees across, as the sun) and the environment
// map times the renderer's environment intensity (seen directly at full
// strength, as the skybox); the renderer's exposure applies.
//
// Presentation images: setProjection(Projection::panorama) traces a 360 x 180
// degree equirectangular panorama around the camera (use a 2:1 size);
// denoise() cleans up the image so far with Intel Open Image Denoise where
// the library was built with it (denoiserAvailable()); readOutput() and
// readLinear() read the image back.

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
class Fence;
} // namespace systems::leal::campello_gpu

namespace systems::leal::campello_renderer {

class PathTracer {
public:
    enum class Projection {
        perspective,  ///< the camera's projection
        panorama,     ///< equirectangular, 360 x 180 degrees around the camera position:
                      ///< the image centre straight ahead, the camera's up at the top
    };

    explicit PathTracer(std::shared_ptr<systems::leal::campello_gpu::Device> device);
    ~PathTracer();

    /// Whether this device can path trace (ray tracing, a backend with the
    /// path tracing shaders: Metal).
    [[nodiscard]] bool supported() const noexcept {
        return pipeline_ != nullptr;
    }

    /// Adds one sample per pixel of `scene` at `width` x `height` to
    /// output(); starts again when the scene, its materials, lights,
    /// camera or the size changed. Doesn't wait for the GPU: work submitted
    /// afterwards on the device's queue sees the sample; the next render()
    /// and the read and denoise calls wait for it. `renderer` gives materials, environment,
    /// exposure and clear colour. Returns false if nothing was traced.
    bool render(const RenderScene& scene, const Renderer& renderer, uint32_t width, uint32_t height);

    /// Whether the last render()'s sample is still on the GPU. An
    /// interactive caller can skip render() until it isn't, so it never
    /// waits for the GPU (the next render() would).
    [[nodiscard]] bool sampleInFlight() const;

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
    /// How camera rays leave the camera (default: perspective).
    void setProjection(Projection projection) noexcept {
        if (projection_ != projection) samples_ = 0;
        projection_ = projection;
    }
    [[nodiscard]] Projection projection() const noexcept {
        return projection_;
    }

    /// Whether denoise() can work: built with Intel Open Image Denoise
    /// (CAMPELLO_RENDERER_DENOISER).
    [[nodiscard]] static bool denoiserAvailable() noexcept;
    /// Denoises the image so far, guided by the first surfaces' albedo and
    /// normals, and shows the result in output() until the next render().
    /// Blocks (the CPU, or the GPU where the library supports it). Returns
    /// false without a denoiser or an image.
    bool denoise();
    /// Whether output() shows the denoised image.
    [[nodiscard]] bool denoised() const noexcept {
        return denoised_;
    }

    /// output() read back: width x height rgba8, display-encoded, top row first.
    bool readOutput(std::vector<uint8_t>& rgba) const;
    /// The image so far as linear rgb floats (exposure applied, before tone
    /// mapping), top row first; optionally the denoiser guides: the first
    /// surfaces' albedo (rgb) and world normals (xyz).
    bool readLinear(std::vector<float>& rgb, std::vector<float>* albedo = nullptr, std::vector<float>* normal = nullptr) const;
    /// Tone mapping of the shader (Khronos PBR Neutral, then display gamma):
    /// linear rgb -> rgba8, alpha 255.
    static void toneMap(const float* rgb, size_t pixels, uint8_t* rgba);

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
    void waitForSample() const;  // the last render()'s GPU work
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
    std::shared_ptr<systems::leal::campello_gpu::Buffer> uvs_;  // texture coordinates per triangle
    std::vector<std::shared_ptr<systems::leal::campello_gpu::Texture>> textures_;  // base colour textures bound
    std::shared_ptr<systems::leal::campello_gpu::Texture> blank_;  // the unused texture bindings
    std::shared_ptr<systems::leal::campello_gpu::Buffer> uniforms_;
    std::shared_ptr<systems::leal::campello_gpu::Buffer> accum_;
    std::shared_ptr<systems::leal::campello_gpu::Buffer> aux_;  // albedo, then normal sums
    std::shared_ptr<systems::leal::campello_gpu::Texture> output_;
    mutable std::shared_ptr<systems::leal::campello_gpu::Fence> pending_;  // the last sample, not yet waited for
    uint64_t sceneKey_ = 0;
    uint64_t viewKey_ = 0;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    uint32_t samples_ = 0;
    uint32_t maxBounces_ = 4;
    float exposure_ = 0.0f;  // of the last render(), for readLinear()
    Projection projection_ = Projection::perspective;
    bool denoised_ = false;
    std::shared_ptr<void> denoiser_;  // Open Image Denoise device, filter and buffers (path_tracer.cpp)
};

} // namespace systems::leal::campello_renderer
