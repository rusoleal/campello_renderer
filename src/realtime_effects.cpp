#include <campello_renderer/realtime_effects.hpp>
#include <campello_renderer/path_tracer.hpp>
#include <campello_renderer/realtime_denoiser.hpp>
#include <campello_renderer/rt_effects.hpp>

#include <campello_gpu/constants/feature.hpp>
#include <campello_gpu/device.hpp>
#include <campello_gpu/texture_view.hpp>

#include <algorithm>
#include <cmath>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

// The ray traced effects (RayTracedShadows, ..., PathTracer) have shaders on
// this backend: Metal only so far (rt_effects.cpp, path_tracer.cpp). Elsewhere
// a device can report Feature::raytracing (DirectX on WARP does) with no
// effect able to run.
#if defined(__APPLE__) && !TARGET_OS_SIMULATOR
#define CAMPELLO_RENDERER_RT_EFFECTS 1
#else
#define CAMPELLO_RENDERER_RT_EFFECTS 0
#endif

namespace systems::leal::campello_renderer {

namespace GPU = systems::leal::campello_gpu;

RealtimeEffects::Settings RealtimeEffects::settingsFor(EffectQuality quality) {
    Settings s;
    switch (quality) {
        case EffectQuality::off:
            s.renderScale = 1.0f;
            break;
        case EffectQuality::low:
            s.shadows = true;
            s.renderScale = 0.5f;
            break;
        case EffectQuality::medium:
            s.shadows = true;
            s.ambientOcclusion = true;
            s.renderScale = 0.58f;
            break;
        case EffectQuality::high:
            // ReSTIR DI shadows every light (the sun's mask isn't needed).
            s.ambientOcclusion = true;
            s.reflections = true;
            s.directLighting = true;
            s.renderScale = 0.67f;
            break;
        case EffectQuality::ultra:
            s.ambientOcclusion = true;
            s.reflections = true;
            s.directLighting = true;
            s.globalIllumination = true;
            s.renderScale = 0.75f;
            break;
    }
    return s;
}

RealtimeEffects::RealtimeEffects(Renderer& renderer) : renderer_(renderer) {
    setQuality(EffectQuality::off);
}

RealtimeEffects::~RealtimeEffects() {
    clearRendererInputs();
}

bool RealtimeEffects::raytracingSupported() const {
    const auto device = renderer_.getDevice();
    return CAMPELLO_RENDERER_RT_EFFECTS && device && renderer_.supportsGBuffer() &&
           device->getFeatures().count(GPU::Feature::raytracing) != 0;
}

void RealtimeEffects::setQuality(EffectQuality quality) {
    quality_ = quality;
    Settings s = settingsFor(quality);
    if (renderer_.supportsUpscaler(Upscaler::metalFX)) s.upscaler = Upscaler::metalFX;
    setSettings(s);
}

void RealtimeEffects::setSettings(const Settings& settings) {
    settings_ = settings;
    if (!raytracingSupported()) {
        settings_.shadows = settings_.ambientOcclusion = settings_.reflections = false;
        settings_.directLighting = settings_.globalIllumination = false;
    }
    settings_.renderScale = std::clamp(settings_.renderScale, 0.25f, 1.0f);
    // Under a budget: no higher than the new settings' scale.
    scale_ = budgetMs_ > 0.0 ? std::min(scale_, settings_.renderScale) : settings_.renderScale;
    framesSinceScaleChange_ = 0;
    ensureEffects();
}

void RealtimeEffects::setFrameBudget(double milliseconds, float minScale) {
    budgetMs_ = std::max(0.0, milliseconds);
    minScale_ = std::clamp(minScale, 0.25f, 1.0f);
    if (budgetMs_ <= 0.0) scale_ = settings_.renderScale;
    framesSinceScaleChange_ = 0;
}

void RealtimeEffects::ensureEffects() {
    const auto device = renderer_.getDevice();
    if (!device) return;
    const bool rayTraced = settings_.shadows || settings_.ambientOcclusion || settings_.directLighting ||
                           settings_.globalIllumination;
    if (rayTraced && !rtScene_) rtScene_ = std::make_shared<RayTracingScene>(device);
    const auto denoiser = [&](std::unique_ptr<RealtimeDenoiser>& d, bool demodulate) {
        if (d) return;
        d = std::make_unique<RealtimeDenoiser>(device);
        RealtimeDenoiser::Settings s;
        s.demodulate = demodulate;
        d->setSettings(s);
    };
    if (settings_.shadows) {
        if (!shadows_) shadows_ = std::make_unique<RayTracedShadows>(device, rtScene_);
        denoiser(shadowDenoiser_, false);  // visibility
    }
    if (settings_.ambientOcclusion) {
        if (!ao_) ao_ = std::make_unique<RayTracedAmbientOcclusion>(device, rtScene_);
        denoiser(aoDenoiser_, false);
    }
    if (settings_.reflections) {
        if (!reflectionTracer_) reflectionTracer_ = std::make_unique<PathTracer>(device);
        denoiser(reflectionDenoiser_, false);  // radiance, not modulated by the albedo
    }
    if (settings_.directLighting) {
        if (!direct_) direct_ = std::make_unique<RestirDirectLighting>(device, rtScene_);
        denoiser(directDenoiser_, true);  // diffuse light, modulated by the albedo
    }
    if (settings_.globalIllumination) {
        if (!gi_) gi_ = std::make_unique<RestirGlobalIllumination>(device, rtScene_);
        denoiser(giDenoiser_, false);  // irradiance
    }
}

void RealtimeEffects::clearRendererInputs() {
    renderer_.setShadowMask(nullptr);
    renderer_.setAmbientOcclusionMask(nullptr);
    renderer_.setReflections(nullptr);
    renderer_.setDirectLighting(nullptr);
    renderer_.setIndirectDiffuse(nullptr);
}

void RealtimeEffects::render(const RenderScene& scene, std::shared_ptr<GPU::TextureView> view) {
    // Dynamic resolution from the last finished frame's GPU time.
    if (budgetMs_ > 0.0 && ++framesSinceScaleChange_ >= 8) {
        const double total = timings().total;
        if (total > 0.0) {
            float next = scale_;
            if (total > budgetMs_) {
                next = std::max(minScale_, scale_ - 0.05f);
            } else if (total < 0.8 * budgetMs_) {
                next = std::min(settings_.renderScale, scale_ + 0.05f);
            }
            if (next != scale_) {
                scale_ = next;
                framesSinceScaleChange_ = 0;
            }
        }
    }
    renderer_.setUpscaler(settings_.upscaler);
    renderer_.setTemporalAntiAliasing(settings_.temporalAntiAliasing);
    renderer_.setRenderScale(scale_);

    const bool rayTraced = settings_.shadows || settings_.ambientOcclusion || settings_.reflections ||
                           settings_.directLighting || settings_.globalIllumination;
    if (rayTraced) {
        renderer_.renderGBuffer(scene);
        const auto& g = renderer_.gBuffer();
        const auto& projection = scene.camera.projection;
        // Each effect, denoised, into the renderer; or off.
        const auto run = [&](bool on, bool traced, RealtimeDenoiser* denoiser, const auto& noisy, const auto& set) {
            if (on && traced && denoiser && denoiser->denoise(noisy(), g, projection)) {
                set(denoiser->output());
            } else {
                set(nullptr);
            }
        };
        run(settings_.shadows, settings_.shadows && shadows_ && shadows_->trace(scene, g), shadowDenoiser_.get(),
            [&] { return shadows_->output(); }, [&](auto t) { renderer_.setShadowMask(t); });
        run(settings_.ambientOcclusion, settings_.ambientOcclusion && ao_ && ao_->trace(scene, g), aoDenoiser_.get(),
            [&] { return ao_->output(); }, [&](auto t) { renderer_.setAmbientOcclusionMask(t); });
        if (settings_.reflections && reflectionTracer_) reflectionTracer_->setReflectionBounces(settings_.reflectionBounces);
        run(settings_.reflections,
            settings_.reflections && reflectionTracer_ && reflectionTracer_->traceReflections(scene, renderer_, g),
            reflectionDenoiser_.get(), [&] { return reflectionTracer_->reflections(); },
            [&](auto t) { renderer_.setReflections(t); });
        if (settings_.directLighting && direct_) {
            auto s = direct_->settings();
            s.candidates = settings_.restirCandidates;
            direct_->setSettings(s);
        }
        run(settings_.directLighting, settings_.directLighting && direct_ && direct_->render(scene, g), directDenoiser_.get(),
            [&] { return direct_->output(); }, [&](auto t) { renderer_.setDirectLighting(t); });
        if (settings_.globalIllumination && gi_) {
            auto s = gi_->settings();
            s.bounces = settings_.globalIlluminationBounces;
            gi_->setSettings(s);
        }
        run(settings_.globalIllumination, settings_.globalIllumination && gi_ && gi_->render(scene, renderer_, g),
            giDenoiser_.get(), [&] { return gi_->output(); }, [&](auto t) { renderer_.setIndirectDiffuse(t); });
    } else {
        clearRendererInputs();
    }
    renderer_.render(scene, std::move(view));
}

RealtimeEffects::Timings RealtimeEffects::timings() const {
    Timings t;
    const auto stats = renderer_.getLastFrameStats();
    t.raster = stats.gpuFrameTimeMs;
    const bool rayTraced = settings_.shadows || settings_.ambientOcclusion || settings_.reflections ||
                           settings_.directLighting || settings_.globalIllumination;
    t.gbuffer = rayTraced ? stats.gbufferGpuTimeMs : 0.0;
    if (settings_.shadows && shadows_) t.shadows = shadows_->gpuTimeMs();
    if (settings_.ambientOcclusion && ao_) t.ambientOcclusion = ao_->gpuTimeMs();
    if (settings_.reflections && reflectionTracer_) t.reflections = reflectionTracer_->passGpuTimeMs();
    if (settings_.directLighting && direct_) t.directLighting = direct_->gpuTimeMs();
    if (settings_.globalIllumination && gi_) t.globalIllumination = gi_->gpuTimeMs();
    const auto denoiseTime = [](bool on, const std::unique_ptr<RealtimeDenoiser>& d) { return on && d ? d->gpuTimeMs() : 0.0; };
    t.denoise = denoiseTime(settings_.shadows, shadowDenoiser_) + denoiseTime(settings_.ambientOcclusion, aoDenoiser_) +
                denoiseTime(settings_.reflections, reflectionDenoiser_) + denoiseTime(settings_.directLighting, directDenoiser_) +
                denoiseTime(settings_.globalIllumination, giDenoiser_);
    t.total = t.gbuffer + t.shadows + t.ambientOcclusion + t.reflections + t.directLighting + t.globalIllumination + t.denoise +
              t.raster;
    return t;
}

void RealtimeEffects::reset() {
    for (auto* d : {shadowDenoiser_.get(), aoDenoiser_.get(), reflectionDenoiser_.get(), directDenoiser_.get(), giDenoiser_.get()}) {
        if (d) d->reset();
    }
    if (direct_) direct_->reset();
    if (gi_) gi_->reset();
    renderer_.resetMotionHistory();
}

} // namespace systems::leal::campello_renderer
