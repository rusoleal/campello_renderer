#pragma once

// Ray traced effects on top of the raster renderer (the Games track): rays
// from the G-buffer's first hit, about one per pixel per frame, cleaned up
// by RealtimeDenoiser and fed back to the raster renderer.
//
// RayTracingScene: the acceleration structures of a RenderScene's opaque
// draws (a BLAS per mesh, cached; the instances only when transforms
// change), shared by the effects so they're built once per frame.
//
// RayTracedShadows: soft sun shadows. One shadow ray per pixel towards a
// random point of the sun's disc, new random numbers every frame.
//
// RayTracedAmbientOcclusion: one ray per pixel in a cosine-weighted random
// direction of the surface's hemisphere, as far as a radius.
//
//   auto rtScene = std::make_shared<RayTracingScene>(device);
//   RayTracedShadows shadows(device, rtScene);
//   RayTracedAmbientOcclusion ao(device, rtScene);
//   RealtimeDenoiser shadowDenoiser(device), aoDenoiser(device);  // demodulate = false for both
//   // each frame:
//   renderer.renderGBuffer(scene);
//   shadows.trace(scene, renderer.gBuffer());
//   ao.trace(scene, renderer.gBuffer());
//   shadowDenoiser.denoise(shadows.output(), renderer.gBuffer(), scene.camera.projection);
//   aoDenoiser.denoise(ao.output(), renderer.gBuffer(), scene.camera.projection);
//   renderer.setShadowMask(shadowDenoiser.output());           // instead of the shadow maps
//   renderer.setAmbientOcclusionMask(aoDenoiser.output());     // instead of SSAO
//   renderer.render(scene, view);
//
// Metal, where campello_gpu reports Feature::raytracing.

#include <campello_renderer/campello_renderer.hpp>
#include <campello_renderer/path_tracer.hpp>
#include <campello_renderer/gpu_timing.hpp>

#include <cstdint>
#include <map>
#include <memory>

namespace systems::leal::campello_gpu {
class AccelerationStructure;
class Buffer;
class ComputePipeline;
class Fence;
class RayTracingPipeline;
class BindGroupLayout;
class Texture;
} // namespace systems::leal::campello_gpu

namespace systems::leal::campello_renderer {

/// Skinned draws posed for ray tracing: each frame a compute pass applies a
/// skinned draw's joint matrices (DrawCall::jointMatrixBuffer) to its
/// mesh's positions and normals, as the raster vertex stage does, into a
/// posed copy of the mesh (same indices and other attributes, no joints or
/// weights) that acceleration structures are built from. Joint and weight
/// formats come from the mesh's layout, or from the buffers' sizes for the
/// standard layout (uint8 / uint16 / uint32 joints; float / unorm8 /
/// unorm16 weights). RayTracingScene and PathTracer use one each.
/// Morph targets aren't applied (RenderScene draws don't carry them).
class SkinnedMeshes {
public:
    explicit SkinnedMeshes(std::shared_ptr<systems::leal::campello_gpu::Device> device);
    ~SkinnedMeshes();
    [[nodiscard]] bool supported() const noexcept {
        return pipeline_ != nullptr;
    }
    /// `scene` with every skinned draw's mesh replaced by its posed copy
    /// (and its joints cleared), posed now (submitted; work submitted
    /// later on the device's queue sees it). Other draws are unchanged.
    /// Copies are kept per (mesh, joint buffer) while they're in use.
    RenderScene pose(const RenderScene& scene);
    /// Whether `mesh` is one of the posed copies.
    [[nodiscard]] bool isPosed(const GpuMesh* mesh) const;
    /// Skinned draws posed by the last pose().
    [[nodiscard]] uint32_t posedDraws() const noexcept {
        return posedDraws_;
    }

private:
    struct Posed {
        std::unique_ptr<GpuMesh> mesh;
        bool used = false;
    };
    std::shared_ptr<systems::leal::campello_gpu::Device> device_;
    std::shared_ptr<systems::leal::campello_gpu::BindGroupLayout> layout_;
    std::shared_ptr<systems::leal::campello_gpu::ComputePipeline> pipeline_;
    std::map<std::pair<const GpuMesh*, const void*>, Posed> posed_;
    std::shared_ptr<systems::leal::campello_gpu::Fence> pending_;
    std::vector<std::shared_ptr<systems::leal::campello_gpu::Buffer>> keepAlive_;
    uint32_t posedDraws_ = 0;
};

class RayTracingScene {
public:
    explicit RayTracingScene(std::shared_ptr<systems::leal::campello_gpu::Device> device);

    /// Brings the structures up to date with `scene`'s opaque draws
    /// (RenderScene::shadowCasters when filled), triangle meshes with float
    /// positions; glass and other transparent draws aren't included. Cheap
    /// when nothing changed (effects sharing the scene call it each).
    /// Returns false if there's nothing to trace.
    bool update(const RenderScene& scene);

    [[nodiscard]] std::shared_ptr<systems::leal::campello_gpu::AccelerationStructure> tlas() const noexcept {
        return tlas_;
    }
    /// Rebuilds so far: full (meshes changed) and of the instances only
    /// (transforms changed).
    [[nodiscard]] uint32_t fullRebuilds() const noexcept {
        return fullRebuilds_;
    }
    [[nodiscard]] uint32_t transformRebuilds() const noexcept {
        return transformRebuilds_;
    }
    /// Frames whose skinned draws were posed and their structures rebuilt
    /// (every update() with skinned draws).
    [[nodiscard]] uint32_t skinnedRebuilds() const noexcept {
        return skinnedRebuilds_;
    }
    /// Alpha-masked casters (GpuMaterial::alphaMask): per instance its
    /// cutoff, base alpha, texture and first triangle in alphaUvs(), for
    /// the effects' alpha tests (rt_effects.metal, buffers 10 and 11 and
    /// textures 16 .. 31). Always valid after a successful update().
    [[nodiscard]] const std::shared_ptr<systems::leal::campello_gpu::Buffer>& alphaInstances() const noexcept {
        return alphaInstances_;
    }
    [[nodiscard]] const std::shared_ptr<systems::leal::campello_gpu::Buffer>& alphaUvs() const noexcept {
        return alphaUvs_;
    }
    [[nodiscard]] const std::vector<std::shared_ptr<systems::leal::campello_gpu::Texture>>& alphaTextures() const noexcept {
        return alphaTextures_;
    }
    /// Alpha-masked instances of the current structures.
    [[nodiscard]] uint32_t alphaTestedInstances() const noexcept {
        return alphaTestedInstances_;
    }

private:
    bool rebuild(const RenderScene& scene);

    struct MeshEntry {
        std::shared_ptr<systems::leal::campello_gpu::AccelerationStructure> blas;
        const void* positions = nullptr;  // the buffers it was built from
        const void* indices = nullptr;
    };
    std::shared_ptr<systems::leal::campello_gpu::Device> device_;
    std::map<const GpuMesh*, MeshEntry> meshes_;
    std::shared_ptr<systems::leal::campello_gpu::AccelerationStructure> tlas_;
    uint64_t geometryKey_ = 0;
    uint64_t transformKey_ = 0;
    uint32_t fullRebuilds_ = 0;
    uint32_t transformRebuilds_ = 0;
    uint32_t skinnedRebuilds_ = 0;
    SkinnedMeshes skinned_;
    std::shared_ptr<systems::leal::campello_gpu::Buffer> alphaInstances_;
    std::shared_ptr<systems::leal::campello_gpu::Buffer> alphaUvs_;
    std::vector<std::shared_ptr<systems::leal::campello_gpu::Texture>> alphaTextures_;
    std::shared_ptr<systems::leal::campello_gpu::Texture> alphaPlaceholder_;
    std::shared_ptr<systems::leal::campello_gpu::BindGroupLayout> uvLayout_;
    std::shared_ptr<systems::leal::campello_gpu::ComputePipeline> uvPipeline_;
    uint32_t alphaTestedInstances_ = 0;
};

/// What the effects share: a ray generation pipeline over the scene, the
/// G-buffer's normals and depth, a visibility output.
class RayTracedEffect {
public:
    virtual ~RayTracedEffect();

    [[nodiscard]] bool supported() const noexcept {
        return pipeline_ != nullptr;
    }
    /// Visibility (rgba16float, the G-buffer's size): 1 .. 0, noisy (about
    /// one sample per pixel).
    [[nodiscard]] std::shared_ptr<systems::leal::campello_gpu::Texture> output() const noexcept {
        return output_;
    }
    /// The scene structures it traces (shared or its own).
    [[nodiscard]] const std::shared_ptr<RayTracingScene>& scene() const noexcept {
        return scene_;
    }
    [[nodiscard]] uint32_t fullRebuilds() const noexcept {
        return scene_ ? scene_->fullRebuilds() : 0;
    }
    [[nodiscard]] uint32_t transformRebuilds() const noexcept {
        return scene_ ? scene_->transformRebuilds() : 0;
    }
    /// GPU time of the last finished trace() (ms; 0 before one finishes).
    [[nodiscard]] double gpuTimeMs() const {
        return timing_.ms();
    }

protected:
    RayTracedEffect(std::shared_ptr<systems::leal::campello_gpu::Device> device, std::shared_ptr<RayTracingScene> scene,
                    const char* entryPoint);
    /// One dispatch of the pipeline over `gbuffer` (the scene brought up to
    /// date first), with the effect's four `parameters` in its uniforms.
    bool dispatch(const RenderScene& scene, const GBufferTargets& gbuffer, const float (&parameters)[4]);

    std::shared_ptr<systems::leal::campello_gpu::Device> device_;
    std::shared_ptr<RayTracingScene> scene_;
    std::shared_ptr<systems::leal::campello_gpu::RayTracingPipeline> pipeline_;
    std::shared_ptr<systems::leal::campello_gpu::BindGroupLayout> layout_;
    std::shared_ptr<systems::leal::campello_gpu::Buffer> uniforms_;
    std::shared_ptr<systems::leal::campello_gpu::Texture> output_;
    std::shared_ptr<systems::leal::campello_gpu::Fence> pending_;
    GpuTiming timing_;
    uint32_t frame_ = 0;
};

class RayTracedShadows : public RayTracedEffect {
public:
    /// `scene`: structures shared with other effects (null: its own).
    explicit RayTracedShadows(std::shared_ptr<systems::leal::campello_gpu::Device> device,
                              std::shared_ptr<RayTracingScene> scene = nullptr);

    /// Traces the sun's shadows for `gbuffer` (rendered this frame from
    /// `scene`): the first directional light with castShadows (else the
    /// first directional light). Submitted, not waited for (the next
    /// trace() waits). Returns false without a sun or a G-buffer.
    bool trace(const RenderScene& scene, const GBufferTargets& gbuffer);

    /// The sun's angular diameter in degrees (default 0.53, the real sun;
    /// larger: softer shadows).
    void setSunAngularDiameter(float degrees) noexcept {
        sunDiameterDegrees_ = degrees > 0.0f ? degrees : 0.0f;
    }

private:
    float sunDiameterDegrees_ = 0.53f;
};

class RayTracedAmbientOcclusion : public RayTracedEffect {
public:
    explicit RayTracedAmbientOcclusion(std::shared_ptr<systems::leal::campello_gpu::Device> device,
                                       std::shared_ptr<RayTracingScene> scene = nullptr);

    /// Traces ambient occlusion for `gbuffer` (rendered this frame from
    /// `scene`): 1 where nothing is within the radius in the sampled
    /// direction, 0 where something is. Submitted, not waited for.
    bool trace(const RenderScene& scene, const GBufferTargets& gbuffer);

    /// How far occluders count (world units, default 1).
    void setRadius(float radius) noexcept {
        radius_ = radius > 0.0f ? radius : 1e-3f;
    }

private:
    float radius_ = 1.0f;
};

/// Direct light from many dynamic lights (hundreds to thousands) with one
/// shadow ray per pixel: ReSTIR DI (Bitterli et al. 2020). Each pixel picks
/// a light from candidates drawn by power and resampled by their
/// contribution there, reuses the previous frame's choice along the motion
/// vectors and its neighbours' choices, and traces one shadow ray for the
/// winner. output() is the pixel's direct light (rgba16float, the
/// G-buffer's size, linear, before exposure; noisy), for RealtimeDenoiser
/// and Renderer::setDirectLighting(), which uses it in place of the raster
/// light loop (whose limit is 4 lights).
///
///   RestirDirectLighting restir(device, rtScene);
///   RealtimeDenoiser denoiser(device);  // demodulate on (the default)
///   // each frame:
///   renderer.renderGBuffer(scene);
///   restir.render(scene, renderer.gBuffer());
///   denoiser.denoise(restir.output(), renderer.gBuffer(), scene.camera.projection);
///   renderer.setDirectLighting(denoiser.output());
///   renderer.render(scene, view);
///
/// Lights: RenderScene::lights (directional, point, spot; every one casts
/// shadows). The BRDF is the raster PBR shader's base layer (Lambert +
/// GGX) from the G-buffer; sheen, clearcoat and anisotropy aren't in it.
/// Biased, with the bias kept small: the candidates' visibility is traced
/// before reuse, the history is the temporal pass's reservoirs (the spatial
/// pass's, whose neighbour samples carry no visibility at the pixel, aren't
/// fed back) and neighbours are used only while a pixel's history is
/// short; the final shadow ray keeps shadows exact. Metal,
/// Feature::raytracing.
class RestirDirectLighting {
public:
    struct Settings {
        uint32_t candidates = 32;      ///< Lights considered per pixel and frame.
        bool     temporal = true;      ///< Reuse the previous frame's reservoirs.
        uint32_t temporalCap = 20;     ///< The history's weight, at most this many frames of candidates.
        bool     spatial = true;       ///< Reuse neighbours' reservoirs.
        uint32_t spatialSamples = 4;   ///< Neighbours per pixel.
        /// Neighbours' radius in pixels; 0: automatic, 1% of the G-buffer's
        /// width, at least 2 (about 19 at 1920). Larger reaches further
        /// but blends surfaces lit differently (bias).
        float    spatialRadius = 0.0f;
        /// Neighbours are reused only while a pixel's history is shorter
        /// than this fraction of the temporal cap (0..1; 1: always while
        /// under the cap): a long history is better on its own, as the
        /// neighbours' samples carry no visibility at the pixel (bias).
        float    spatialHistory = 0.25f;
    };

    /// `scene`: structures shared with other effects (null: its own).
    explicit RestirDirectLighting(std::shared_ptr<systems::leal::campello_gpu::Device> device,
                                  std::shared_ptr<RayTracingScene> scene = nullptr);
    ~RestirDirectLighting();

    [[nodiscard]] bool supported() const noexcept {
        return initial_ != nullptr && spatial_ != nullptr && reference_ != nullptr;
    }

    /// One frame for `gbuffer` (rendered this frame from `scene`).
    /// Submitted, not waited for (the next render() waits). Returns false
    /// without lights, a G-buffer or support.
    bool render(const RenderScene& scene, const GBufferTargets& gbuffer);
    /// Ground truth into output(): every light with its own shadow ray (as
    /// many rays per pixel as lights; for tuning and tests). Leaves the
    /// reservoirs alone.
    bool renderReference(const RenderScene& scene, const GBufferTargets& gbuffer);

    /// The direct light of the last render() (see the class comment).
    [[nodiscard]] std::shared_ptr<systems::leal::campello_gpu::Texture> output() const noexcept {
        return radiance_;
    }

    void setSettings(const Settings& settings) noexcept {
        settings_ = settings;
    }
    [[nodiscard]] const Settings& settings() const noexcept {
        return settings_;
    }
    /// Forgets the reservoirs (a camera cut); the next frame starts afresh.
    void reset() noexcept {
        historyValid_ = false;
    }
    /// GPU time of the last finished render() (ms; 0 before one finishes).
    [[nodiscard]] double gpuTimeMs() const {
        return timing_.ms();
    }
    [[nodiscard]] const std::shared_ptr<RayTracingScene>& scene() const noexcept {
        return scene_;
    }

private:
    bool ensureTargets(uint32_t width, uint32_t height);
    bool frame(const RenderScene& scene, const GBufferTargets& gbuffer, bool reference);

    std::shared_ptr<systems::leal::campello_gpu::Device> device_;
    std::shared_ptr<RayTracingScene> scene_;
    std::shared_ptr<systems::leal::campello_gpu::BindGroupLayout> layout_;
    std::shared_ptr<systems::leal::campello_gpu::RayTracingPipeline> initial_;
    std::shared_ptr<systems::leal::campello_gpu::RayTracingPipeline> spatial_;
    std::shared_ptr<systems::leal::campello_gpu::RayTracingPipeline> reference_;
    std::shared_ptr<systems::leal::campello_gpu::Buffer> uniforms_;
    std::shared_ptr<systems::leal::campello_gpu::Buffer> lights_;
    std::shared_ptr<systems::leal::campello_gpu::Texture> reservoirs_[2];  // restirInitial's, this frame's and the last (the history)
    std::shared_ptr<systems::leal::campello_gpu::Texture> final_;          // restirSpatial's (not read back)
    std::shared_ptr<systems::leal::campello_gpu::Texture> surfaces_[2];  // normal + view depth, this frame's and the last
    std::shared_ptr<systems::leal::campello_gpu::Texture> radiance_;
    std::shared_ptr<systems::leal::campello_gpu::Fence> pending_;
    GpuTiming timing_;
    Settings settings_;
    uint32_t frame_ = 0;
    uint32_t surfaceIndex_ = 0;
    bool historyValid_ = false;
};

/// One-bounce indirect light at one path per pixel: ReSTIR GI (Ouyang et
/// al. 2021). Each frame its PathTracer traces one cosine-weighted path
/// from every G-buffer pixel (PathTracer::traceGlobalIllumination(): the
/// hit, its normal, the light leaving it, with the tracer's materials, sun,
/// sky and emitters); the hits are resampled by the irradiance they bring,
/// reused along the motion vectors and, while a pixel's history is short,
/// from neighbours (with the reconnection's Jacobian and a visibility
/// test). output() is the irradiance (rgba16float, the G-buffer's size:
/// what a white diffuse surface reflects, linear, before exposure; noisy),
/// for RealtimeDenoiser (demodulate off) and Renderer::setIndirectDiffuse()
/// (as DdgiVolume's).
///
///   RestirGlobalIllumination gi(device, rtScene);
///   // each frame:
///   renderer.renderGBuffer(scene);
///   gi.render(scene, renderer, renderer.gBuffer());
///   denoiser.denoise(gi.output(), renderer.gBuffer(), scene.camera.projection);
///   renderer.setIndirectDiffuse(denoiser.output());
///
/// Biased in the same, small ways as RestirDirectLighting. Metal,
/// Feature::raytracing.
class RestirGlobalIllumination {
public:
    struct Settings {
        bool     temporal = true;        ///< Reuse the previous frame's reservoirs.
        uint32_t temporalCap = 30;       ///< The history's weight, at most this many frames.
        bool     spatial = true;         ///< Reuse neighbours' reservoirs while a history is short.
        uint32_t spatialSamples = 4;     ///< Neighbours per pixel.
        float    spatialRadius = 0.0f;   ///< In pixels; 0: 1% of the width, at least 2.
        float    spatialHistory = 0.25f; ///< Neighbours only below this fraction of the cap.
        uint32_t bounces = 1;            ///< The tracer's bounces after the hit.
    };

    /// `scene`: structures shared with other effects, for the visibility
    /// tests (null: its own). The paths use the tracer's own structures.
    explicit RestirGlobalIllumination(std::shared_ptr<systems::leal::campello_gpu::Device> device,
                                      std::shared_ptr<RayTracingScene> scene = nullptr);
    ~RestirGlobalIllumination();

    [[nodiscard]] bool supported() const noexcept;

    /// One frame for `gbuffer` (rendered this frame from `scene`).
    /// Submitted, not waited for. Returns false without a G-buffer or
    /// support.
    bool render(const RenderScene& scene, const Renderer& renderer, const GBufferTargets& gbuffer);

    [[nodiscard]] std::shared_ptr<systems::leal::campello_gpu::Texture> output() const noexcept {
        return irradiance_;
    }
    void setSettings(const Settings& settings) noexcept {
        settings_ = settings;
    }
    [[nodiscard]] const Settings& settings() const noexcept {
        return settings_;
    }
    void reset() noexcept {
        historyValid_ = false;
    }
    /// GPU time of the last finished render() (ms; 0 before one finishes):
    /// its paths and its resampling.
    [[nodiscard]] double gpuTimeMs() const {
        return timing_.ms() + (tracer_ ? tracer_->passGpuTimeMs() : 0.0);
    }
    /// The tracer of the samples (its settings: sky and emitter sampling,
    /// clamping, ...).
    [[nodiscard]] PathTracer& tracer() noexcept {
        return *tracer_;
    }

private:
    bool ensureTargets(uint32_t width, uint32_t height);

    std::shared_ptr<systems::leal::campello_gpu::Device> device_;
    std::shared_ptr<RayTracingScene> scene_;
    std::unique_ptr<PathTracer> tracer_;
    std::shared_ptr<systems::leal::campello_gpu::BindGroupLayout> layout_;
    std::shared_ptr<systems::leal::campello_gpu::RayTracingPipeline> initial_;
    std::shared_ptr<systems::leal::campello_gpu::RayTracingPipeline> spatial_;
    std::shared_ptr<systems::leal::campello_gpu::Buffer> uniforms_;
    // Reservoirs (point, radiance, normal): this frame's and the last (the
    // history), and the spatial pass's (not read back); surfaces.
    std::shared_ptr<systems::leal::campello_gpu::Texture> reservoirs_[2][3];
    std::shared_ptr<systems::leal::campello_gpu::Texture> scratch_[3];
    std::shared_ptr<systems::leal::campello_gpu::Texture> surfaces_[2];
    std::shared_ptr<systems::leal::campello_gpu::Texture> irradiance_;
    std::shared_ptr<systems::leal::campello_gpu::Fence> pending_;
    GpuTiming timing_;
    Settings settings_;
    uint32_t frame_ = 0;
    uint32_t index_ = 0;
    bool historyValid_ = false;
};

} // namespace systems::leal::campello_renderer
