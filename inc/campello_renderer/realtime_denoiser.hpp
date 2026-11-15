#pragma once

// Real-time denoising of a noisy image (about one sample per pixel of a ray
// traced effect: shadows, reflections, ambient occlusion, global
// illumination, a path traced frame) guided by the renderer's G-buffer:
// SVGF-style (Schied et al. 2017) temporal accumulation along the motion
// vectors with disocclusion rejection, a per-pixel variance, and five
// edge-aware a-trous wavelet passes. For games: a fixed cost per frame and a
// moving camera, unlike PathTracer::denoise() (Open Image Denoise, stills).
//
//   RealtimeDenoiser denoiser(device);
//   // each frame:
//   renderer.renderGBuffer(scene);                 // normals, depth, motion, albedo
//   traceEffect(noisy);                            // rgba16float, the G-buffer's size
//   denoiser.denoise(noisy, renderer.gBuffer(), scene.camera.projection);
//   use(denoiser.output());                        // rgba16float, linear
//
// The input is demodulated by the G-buffer's base colour (filtered as
// illumination, so texture detail isn't blurred) and multiplied back after;
// background pixels (nothing in the G-buffer) pass through. Metal for now.

#include <campello_renderer/campello_renderer.hpp>

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

class RealtimeDenoiser {
public:
    explicit RealtimeDenoiser(std::shared_ptr<systems::leal::campello_gpu::Device> device);
    ~RealtimeDenoiser();

    /// Whether this device can run it (Metal for now).
    [[nodiscard]] bool supported() const noexcept {
        return temporalPipeline_ != nullptr;
    }

    /// Denoises `noisy` (a 2D texture of `gbuffer`'s size, linear rgb;
    /// textureBinding usage) guided by `gbuffer` (Renderer::gBuffer(),
    /// rendered this frame) and the camera's `projection` (for linear
    /// depth). The result is in output(). Submitted, not waited for (the
    /// next denoise() waits for it before reusing its buffers).
    /// Returns false if nothing was denoised (unsupported, sizes differ).
    bool denoise(const std::shared_ptr<systems::leal::campello_gpu::Texture>& noisy, const GBufferTargets& gbuffer,
                 const systems::leal::vector_math::Matrix4<double>& projection);

    /// The last denoised image (rgba16float, linear).
    [[nodiscard]] std::shared_ptr<systems::leal::campello_gpu::Texture> output() const noexcept {
        return output_;
    }

    /// Forgets the history (a camera cut, a new scene).
    void reset() noexcept {
        historyValid_ = false;
    }

    struct Settings {
        /// The new frame's minimum weight in the history (lower: smoother,
        /// slower to follow change).
        float colorBlend = 0.2f;
        float momentsBlend = 0.2f;
        /// History length cap (frames).
        float maxHistory = 32.0f;
        /// Disocclusion: reprojected history is kept where the linear depth
        /// agrees within this fraction (and the normals within ~25 degrees).
        float depthTolerance = 0.05f;
        /// Edge stopping: luminance (in standard deviations), normal (power),
        /// depth (in depth gradients).
        float phiColor = 4.0f;
        float phiNormal = 128.0f;
        float phiDepth = 1.0f;
        /// A-trous passes (steps 1, 2, 4, ...; 1 to 7): 5 cover a 125 pixel footprint.
        uint32_t iterations = 5;
    };
    void setSettings(const Settings& settings) noexcept {
        settings_ = settings;
    }
    [[nodiscard]] const Settings& settings() const noexcept {
        return settings_;
    }

private:
    bool ensureTargets(uint32_t width, uint32_t height);

    std::shared_ptr<systems::leal::campello_gpu::Device> device_;
    std::shared_ptr<systems::leal::campello_gpu::RenderPipeline> temporalPipeline_;
    std::shared_ptr<systems::leal::campello_gpu::RenderPipeline> variancePipeline_;
    std::shared_ptr<systems::leal::campello_gpu::RenderPipeline> atrousPipeline_;
    std::shared_ptr<systems::leal::campello_gpu::RenderPipeline> atrousFirstPipeline_;
    std::shared_ptr<systems::leal::campello_gpu::RenderPipeline> modulatePipeline_;
    std::shared_ptr<systems::leal::campello_gpu::BindGroupLayout> temporalLayout_;
    std::shared_ptr<systems::leal::campello_gpu::BindGroupLayout> varianceLayout_;
    std::shared_ptr<systems::leal::campello_gpu::BindGroupLayout> atrousLayout_;
    std::shared_ptr<systems::leal::campello_gpu::BindGroupLayout> atrousFirstLayout_;
    std::shared_ptr<systems::leal::campello_gpu::BindGroupLayout> modulateLayout_;

    // Textures (all width_ x height_): the history (colour + length,
    // moments, geometry) alternates between two; the frame's accumulation
    // and the a-trous ping-pong; the output.
    struct Target {
        std::shared_ptr<systems::leal::campello_gpu::Texture> texture;
        std::shared_ptr<systems::leal::campello_gpu::TextureView> view;
    };
    std::array<Target, 2> historyColor_, historyMoments_, geometry_;
    Target accumulated_, pingA_, pingB_, outputTarget_;
    std::shared_ptr<systems::leal::campello_gpu::Texture> output_;
    std::array<std::shared_ptr<systems::leal::campello_gpu::Buffer>, 8> uniforms_;  // per pass of a frame
    std::shared_ptr<systems::leal::campello_gpu::Fence> pending_;                 // the last frame's work
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    uint32_t historyIndex_ = 0;  // the history being read this frame
    bool historyValid_ = false;
    Settings settings_;
};

} // namespace systems::leal::campello_renderer
