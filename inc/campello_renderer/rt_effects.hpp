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

#include <cstdint>
#include <map>
#include <memory>

namespace systems::leal::campello_gpu {
class AccelerationStructure;
class Buffer;
class Fence;
class RayTracingPipeline;
class BindGroupLayout;
class Texture;
} // namespace systems::leal::campello_gpu

namespace systems::leal::campello_renderer {

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

} // namespace systems::leal::campello_renderer
