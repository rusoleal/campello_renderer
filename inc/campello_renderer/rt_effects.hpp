#pragma once

// Ray traced effects on top of the raster renderer (the Games track): rays
// from the G-buffer's first hit, about one per pixel per frame, cleaned up
// by RealtimeDenoiser and fed back to the raster renderer.
//
// RayTracedShadows: soft sun shadows. One shadow ray per pixel towards a
// random point of the sun's disc (its angular size), new random numbers
// every frame.
//
//   RayTracedShadows shadows(device);
//   RealtimeDenoiser denoiser(device);
//   RealtimeDenoiser::Settings s; s.demodulate = false; denoiser.setSettings(s);  // a visibility signal
//   // each frame:
//   renderer.renderGBuffer(scene);
//   shadows.trace(scene, renderer.gBuffer());
//   denoiser.denoise(shadows.output(), renderer.gBuffer(), scene.camera.projection);
//   renderer.setShadowMask(denoiser.output());   // instead of the shadow maps
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

class RayTracedShadows {
public:
    explicit RayTracedShadows(std::shared_ptr<systems::leal::campello_gpu::Device> device);
    ~RayTracedShadows();

    [[nodiscard]] bool supported() const noexcept {
        return pipeline_ != nullptr;
    }

    /// Traces the sun's shadows for `gbuffer` (rendered this frame from
    /// `scene`): the first directional light with castShadows (else the
    /// first directional light). Casters: the scene's opaque draws
    /// (RenderScene::shadowCasters when filled), triangle meshes with float
    /// positions; glass doesn't cast. Submitted, not waited for (the next
    /// trace() waits). Returns false without a sun or a G-buffer.
    bool trace(const RenderScene& scene, const GBufferTargets& gbuffer);

    /// Visibility (rgba16float, the G-buffer's size): 1 lit .. 0 shadowed,
    /// noisy (about one sample per pixel).
    [[nodiscard]] std::shared_ptr<systems::leal::campello_gpu::Texture> output() const noexcept {
        return output_;
    }

    /// The sun's angular diameter in degrees (default 0.53, the real sun;
    /// larger: softer shadows).
    void setSunAngularDiameter(float degrees) noexcept {
        sunDiameterDegrees_ = degrees > 0.0f ? degrees : 0.0f;
    }

    /// Acceleration structure rebuilds so far: full (meshes changed) and of
    /// the instances only (transforms changed).
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
    std::shared_ptr<systems::leal::campello_gpu::RayTracingPipeline> pipeline_;
    std::shared_ptr<systems::leal::campello_gpu::BindGroupLayout> layout_;
    std::map<const GpuMesh*, MeshEntry> meshes_;
    std::shared_ptr<systems::leal::campello_gpu::AccelerationStructure> tlas_;
    std::shared_ptr<systems::leal::campello_gpu::Buffer> uniforms_;
    std::shared_ptr<systems::leal::campello_gpu::Texture> output_;
    std::shared_ptr<systems::leal::campello_gpu::Fence> pending_;
    uint64_t geometryKey_ = 0;
    uint64_t transformKey_ = 0;
    uint32_t frame_ = 0;
    uint32_t fullRebuilds_ = 0;
    uint32_t transformRebuilds_ = 0;
    float sunDiameterDegrees_ = 0.53f;
};

} // namespace systems::leal::campello_renderer
