#pragma once

// Probe-based global illumination (the Games track), after Majercik et al.
// 2019, "Dynamic Diffuse Global Illumination with Ray-Traced Irradiance
// Fields" (DDGI): a grid of light probes, each tracing a few rays a frame
// with the path tracer (PathTracer::traceProbeRays()), keeping irradiance
// and hit distance per direction in octahedral atlases blended over frames;
// a surface's indirect diffuse light is its 8 surrounding probes' irradiance,
// weighted by position, facing and visibility (no light through walls).
// Much cheaper than per-pixel GI and stable under motion.
//
//   PathTracer tracer(device);                 // its own (or the reflections')
//   DdgiVolume gi(device);
//   gi.setGrid(DdgiVolume::fitting(sceneMin, sceneMax, 1.0));
//   // each frame:
//   gi.update(tracer, scene, renderer);        // the probes
//   renderer.renderGBuffer(scene);
//   gi.resolve(renderer.gBuffer(), scene.camera);
//   renderer.setIndirectDiffuse(gi.irradiance());   // instead of the irradiance map
//   renderer.render(scene, view);
//
// Metal, where PathTracer is supported.

#include <campello_renderer/campello_renderer.hpp>
#include <campello_renderer/gpu_timing.hpp>

#include <array>
#include <cstdint>
#include <memory>

namespace systems::leal::campello_gpu {
class BindGroupLayout;
class Buffer;
class Fence;
class RenderPipeline;
class Texture;
class TextureView;
} // namespace systems::leal::campello_gpu

namespace systems::leal::campello_renderer {

class PathTracer;

class DdgiVolume {
public:
    struct Grid {
        systems::leal::vector_math::Vector3<double> origin{0.0, 0.0, 0.0};   ///< probe (0, 0, 0)
        systems::leal::vector_math::Vector3<double> spacing{1.0, 1.0, 1.0};  ///< between probes
        uint32_t count[3] = {8, 4, 8};                                       ///< probes per axis (2 or more)
        uint32_t raysPerProbe = 64;
        /// The history's weight each frame (0..1): higher is smoother and
        /// slower to follow lighting changes.
        float hysteresis = 0.97f;
        /// Hit distances beyond this count as this (world units; 0: 1.5x the largest spacing).
        float maxDistance = 0.0f;
        /// Where a surface looks up the probes: offset along its normal and
        /// towards the viewer, as fractions of the smallest spacing.
        float normalBias = 0.05f;
        float viewBias = 0.2f;
    };
    /// A grid covering [min, max] with probes about `spacing` apart (at least 2 per axis, at most 32).
    [[nodiscard]] static Grid fitting(const systems::leal::vector_math::Vector3<double>& min,
                                      const systems::leal::vector_math::Vector3<double>& max, double spacing);

    explicit DdgiVolume(std::shared_ptr<systems::leal::campello_gpu::Device> device);
    ~DdgiVolume();

    [[nodiscard]] bool supported() const noexcept {
        return resolvePipeline_ != nullptr;
    }

    /// The probes (the history starts again).
    void setGrid(const Grid& grid);
    [[nodiscard]] const Grid& grid() const noexcept {
        return grid_;
    }

    /// This frame's probe rays, traced with `tracer` (a PathTracer for this
    /// and reflections only: it doesn't keep a progressive image), blended
    /// into the probes. Submitted, not waited for.
    bool update(PathTracer& tracer, const RenderScene& scene, const Renderer& renderer);

    /// The indirect diffuse light at each pixel of `gbuffer` (rendered this
    /// frame, `camera` its camera): irradiance(). Submitted, not waited for.
    bool resolve(const GBufferTargets& gbuffer, const CameraData& camera);

    /// rgba16float, the G-buffer's size: the probes' irradiance at each
    /// surface (as the renderer's irradiance map: what a white diffuse
    /// surface reflects, before exposure); 0 on the background.
    [[nodiscard]] std::shared_ptr<systems::leal::campello_gpu::Texture> irradiance() const noexcept {
        return irradiance_;
    }
    /// The atlases (rgba16float): per probe an 8 x 8 irradiance and a 16 x 16
    /// distance (r mean, g mean squared) octahedral map with a 1 texel border.
    [[nodiscard]] std::shared_ptr<systems::leal::campello_gpu::Texture> irradianceAtlas() const noexcept {
        return irradianceAtlas_[historyIndex_].texture;
    }
    [[nodiscard]] std::shared_ptr<systems::leal::campello_gpu::Texture> distanceAtlas() const noexcept {
        return distanceAtlas_[historyIndex_].texture;
    }
    /// Forgets the probes' history.
    void reset() noexcept {
        historyValid_ = false;
    }
    /// GPU time of the last finished update() and resolve() (ms; the probe
    /// rays are the PathTracer's: its stats()).
    [[nodiscard]] double gpuTimeMs() const {
        return updateTiming_.ms() + resolveTiming_.ms();
    }
    /// update() calls so far.
    [[nodiscard]] uint32_t frames() const noexcept {
        return frame_;
    }

private:
    bool ensureAtlases();
    void uniforms(float* out, const CameraData* camera) const;

    std::shared_ptr<systems::leal::campello_gpu::Device> device_;
    std::shared_ptr<systems::leal::campello_gpu::RenderPipeline> irradiancePipeline_;
    std::shared_ptr<systems::leal::campello_gpu::RenderPipeline> distancePipeline_;
    std::shared_ptr<systems::leal::campello_gpu::RenderPipeline> resolvePipeline_;
    std::shared_ptr<systems::leal::campello_gpu::BindGroupLayout> updateLayout_;
    std::shared_ptr<systems::leal::campello_gpu::BindGroupLayout> resolveLayout_;
    struct Target {
        std::shared_ptr<systems::leal::campello_gpu::Texture> texture;
        std::shared_ptr<systems::leal::campello_gpu::TextureView> view;
    };
    std::array<Target, 2> irradianceAtlas_, distanceAtlas_;
    Target resolveTarget_;
    std::shared_ptr<systems::leal::campello_gpu::Texture> irradiance_;
    std::shared_ptr<systems::leal::campello_gpu::Buffer> updateUniforms_, resolveUniforms_;
    std::shared_ptr<systems::leal::campello_gpu::Fence> pendingUpdate_, pendingResolve_;
    GpuTiming updateTiming_, resolveTiming_;
    Grid grid_;
    float rotation_[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};  // this frame's ray rotation
    uint32_t historyIndex_ = 0;
    uint32_t frame_ = 0;
    bool historyValid_ = false;
};

} // namespace systems::leal::campello_renderer
