#pragma once

// RealtimeEffects: the Games track's effects run as one chain with quality
// presets, per-effect GPU timings and dynamic resolution.
//
// Each frame render() draws the G-buffer, runs the enabled ray traced
// effects with their denoisers, hands their textures to the renderer
// (setShadowMask(), setAmbientOcclusionMask(), setReflections(),
// setDirectLighting(), setIndirectDiffuse()) and draws the frame with TAA at
// the render scale. Presets (EffectQuality) choose the effects and the
// render scale; Settings changes them one by one. Without
// Feature::raytracing the ray traced effects stay off (raster shadows,
// SSAO, the environment).
//
//   RealtimeEffects effects(renderer);
//   effects.setQuality(EffectQuality::high);
//   effects.setFrameBudget(16.0);  // dynamic resolution
//   // each frame:
//   effects.render(scene, view);
//   const auto t = effects.timings();  // last finished frame's GPU ms per effect
//
// The same chain by hand: docs/using-the-renderer.md.

#include <campello_renderer/campello_renderer.hpp>

#include <memory>

namespace systems::leal::campello_renderer {

class PathTracer;
class RayTracingScene;
class RayTracedShadows;
class RayTracedAmbientOcclusion;
class RestirDirectLighting;
class RestirGlobalIllumination;
class RealtimeDenoiser;

/// Presets, from raster only to every ray traced effect.
enum class EffectQuality {
    off,     ///< Raster only: shadow maps, SSAO if on, the environment; TAA, full resolution.
    low,     ///< Ray traced sun shadows; half resolution upscaled.
    medium,  ///< + ray traced ambient occlusion.
    high,    ///< + ray traced reflections; ReSTIR DI for every light (instead of the shadows' mask).
    ultra    ///< + ReSTIR GI (one-bounce indirect light).
};

class RealtimeEffects {
public:
    struct Settings {
        bool shadows = false;             ///< RayTracedShadows -> setShadowMask()
        bool ambientOcclusion = false;    ///< RayTracedAmbientOcclusion -> setAmbientOcclusionMask()
        bool reflections = false;         ///< PathTracer::traceReflections() -> setReflections()
        bool directLighting = false;      ///< RestirDirectLighting -> setDirectLighting()
        bool globalIllumination = false;  ///< RestirGlobalIllumination -> setIndirectDiffuse()
        float renderScale = 1.0f;         ///< The internal resolution (Renderer::setRenderScale()).
        bool temporalAntiAliasing = true; ///< TAA, the upscaler below a scale of 1.
        Upscaler upscaler = Upscaler::builtin;
        uint32_t restirCandidates = 32;   ///< RestirDirectLighting::Settings::candidates
        uint32_t reflectionBounces = 1;
        uint32_t globalIlluminationBounces = 1;
    };
    /// A preset's settings (the ray traced effects whether or not the
    /// device has them; setQuality() turns them off where it hasn't). The
    /// render scales (1, 0.5, 0.58, 0.67, 0.75) are starting points: a
    /// frame budget adjusts them.
    [[nodiscard]] static Settings settingsFor(EffectQuality quality);

    /// GPU milliseconds of the last finished frame, per effect (0: off or
    /// not measured yet). denoise: all the denoisers; raster: the frame
    /// itself (Renderer); total: their sum.
    struct Timings {
        double gbuffer = 0.0;
        double shadows = 0.0;
        double ambientOcclusion = 0.0;
        double reflections = 0.0;
        double directLighting = 0.0;
        double globalIllumination = 0.0;
        double denoise = 0.0;
        double raster = 0.0;
        double total = 0.0;
    };

    /// Works on `renderer` (createDefaultPipelines() and resize() done;
    /// it must outlive this). Starts at EffectQuality::off.
    explicit RealtimeEffects(Renderer& renderer);
    ~RealtimeEffects();
    RealtimeEffects(const RealtimeEffects&) = delete;
    RealtimeEffects& operator=(const RealtimeEffects&) = delete;

    /// Whether the ray traced effects can run (Feature::raytracing and the
    /// G-buffer).
    [[nodiscard]] bool raytracingSupported() const;

    /// settingsFor(quality), without the ray traced effects where they
    /// can't run, and MetalFX as the upscaler where it's supported.
    void setQuality(EffectQuality quality);
    [[nodiscard]] EffectQuality quality() const noexcept {
        return quality_;
    }
    /// Effects one by one (a preset's, changed). Ray traced effects the
    /// device can't run are ignored.
    void setSettings(const Settings& settings);
    [[nodiscard]] const Settings& settings() const noexcept {
        return settings_;
    }

    /// Dynamic resolution: keeps the frame's GPU time (timings().total)
    /// under `milliseconds` by lowering the render scale, down to
    /// `minScale`, and raising it back towards the settings' while there's
    /// room (steps of 0.05, at most every 8 frames). 0: off (the settings'
    /// scale).
    void setFrameBudget(double milliseconds, float minScale = 0.5f);
    /// The render scale in use (the settings', or the budget's).
    [[nodiscard]] float renderScale() const noexcept {
        return scale_;
    }

    /// One frame of `scene` into `view` (see the header comment).
    void render(const RenderScene& scene, std::shared_ptr<systems::leal::campello_gpu::TextureView> view);

    /// The last finished frame's GPU times (not waiting for the GPU).
    [[nodiscard]] Timings timings() const;

    /// A camera cut: the effects' histories (reservoirs, denoisers, TAA)
    /// start again.
    void reset();

private:
    void ensureEffects();
    void clearRendererInputs();

    Renderer& renderer_;
    EffectQuality quality_ = EffectQuality::off;
    Settings settings_;
    double budgetMs_ = 0.0;
    float minScale_ = 0.5f;
    float scale_ = 1.0f;
    uint32_t framesSinceScaleChange_ = 0;
    std::shared_ptr<RayTracingScene> rtScene_;
    std::unique_ptr<RayTracedShadows> shadows_;
    std::unique_ptr<RayTracedAmbientOcclusion> ao_;
    std::unique_ptr<PathTracer> reflectionTracer_;
    std::unique_ptr<RestirDirectLighting> direct_;
    std::unique_ptr<RestirGlobalIllumination> gi_;
    std::unique_ptr<RealtimeDenoiser> shadowDenoiser_, aoDenoiser_, reflectionDenoiser_, directDenoiser_, giDenoiser_;
};

} // namespace systems::leal::campello_renderer
