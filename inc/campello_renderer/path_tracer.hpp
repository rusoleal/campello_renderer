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
// strength, as the skybox), also sampled as a light by its brightness
// (setEnvironmentSampling()) and through light portals, windows the sky
// lights interiors through (setPortals(), portalsFromGlass()); the
// renderer's exposure applies.
//
// Noise and speed: Owen-scrambled Sobol random numbers (setSampler()),
// clamping of indirect light against fireflies (setIndirectClamp()), fewer
// bounces or a reduced resolution for an image's first samples
// (setProgressiveBounces(), setPreviewResolution()), several samples per
// render() or a GPU time budget (setSamplesPerRender(), setFrameBudget());
// stats()
// times the samples, rebuilds and denoising.
//
// Presentation images: setProjection(Projection::panorama) traces a 360 x 180
// degree equirectangular panorama around the camera (use a 2:1 size);
// denoise() cleans up the image so far with Intel Open Image Denoise where
// the library was built with it (denoiserAvailable()); readOutput() and
// readLinear() read the image back.

#include <campello_renderer/campello_renderer.hpp>

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace systems::leal::campello_gpu {
class AccelerationStructure;
class Buffer;
class CommandBuffer;
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

    /// Adds samples per pixel of `scene` at `width` x `height` to output()
    /// (one, or setSamplesPerRender() / setFrameBudget()'s); starts again when the scene, its materials, lights,
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
    /// Blocks until done (on the CPU, or the denoiser's GPU device).
    /// Returns false without a denoiser or an image.
    bool denoise();

    enum class DenoiserDevice {
        automatic,  ///< a GPU device of Open Image Denoise where it has one (Metal on Apple
                    ///< silicon, CUDA, HIP, SYCL), else the CPU (default)
        cpu,        ///< the CPU
        gpu,        ///< a GPU device only: denoise() fails without one
    };
    /// Where denoise() runs. A GPU device filters many times faster; the
    /// image still goes through host memory (shared on Apple silicon).
    /// The library loads device modules next to its core library
    /// (OpenImageDenoise_device_metal, ..._cuda), which apps bundle too.
    void setDenoiserDevice(DenoiserDevice device);
    enum class DenoiserQuality {
        fast,      ///< fastest (interactive previews, early milestones)
        balanced,  ///< between the two (interactive rendering)
        high,      ///< best (final images; the default)
    };
    /// Changing it (or the prefilter) makes the filters again on the next
    /// denoise(): one slower call (their weights load).
    void setDenoiserQuality(DenoiserQuality quality) noexcept {
        denoiserQuality_ = quality;
    }
    /// Denoises the albedo and normal guides first, each with its own
    /// filter, and tells the main filter they're clean: noise in the
    /// guides (surfaces seen through glass or mirrors, fine texture detail
    /// at few samples) then isn't kept as detail. About three filter
    /// passes instead of one; for stills. Default off.
    void setDenoiserPrefilter(bool enabled) noexcept {
        denoiserPrefilter_ = enabled;
    }

    /// The device denoise() used: "cpu", "metal", "cuda", "hip" or "sycl";
    /// empty before the first denoise().
    [[nodiscard]] std::string denoiserDeviceType() const;
    /// Ray traced reflections for the raster renderer (the Games track):
    /// for each pixel of `gbuffer` (Renderer::renderGBuffer(), this frame,
    /// of `scene`), one path from its surface in a direction of its GGX
    /// lobe (its roughness), shaded as this tracer shades (materials,
    /// textures, the sun, the sky, emitters, setReflectionBounces() more
    /// bounces). reflections() is the radiance arriving along it, about one
    /// sample per pixel: denoise it (RealtimeDenoiser, demodulate off) and
    /// give it to Renderer::setReflections(). Use a PathTracer of its own
    /// for this (it doesn't keep the progressive image's state). Submitted,
    /// not waited for (the next call waits).
    bool traceReflections(const RenderScene& scene, const Renderer& renderer, const GBufferTargets& gbuffer);
    /// rgba16float, the G-buffer's size: linear radiance before exposure, 0 on the background.
    [[nodiscard]] std::shared_ptr<systems::leal::campello_gpu::Texture> reflections() const noexcept {
        return reflectionTarget_;
    }
    /// Indirect light samples for RestirGlobalIllumination (the Games
    /// track): for each pixel of `gbuffer`, one path from its surface in a
    /// cosine-weighted direction of its hemisphere, shaded as this tracer
    /// shades (setGlobalIlluminationBounces() bounces after the hit).
    /// globalIlluminationSamples() is 2 x the G-buffer's width: on the
    /// left, rgb the radiance leaving the hit towards the surface (linear,
    /// before exposure) and a the hit's distance (65000: the sky); on the
    /// right, octahedral unit vectors: xy the direction, zw the hit's
    /// geometric normal (facing the surface; the sky: the reverse
    /// direction). Use a PathTracer of its own. Submitted, not waited for.
    bool traceGlobalIllumination(const RenderScene& scene, const Renderer& renderer, const GBufferTargets& gbuffer);
    [[nodiscard]] std::shared_ptr<systems::leal::campello_gpu::Texture> globalIlluminationSamples() const noexcept {
        return giTarget_;
    }
    /// Bounces after the hit (default 1: its direct light and one more).
    void setGlobalIlluminationBounces(uint32_t bounces) noexcept {
        giBounces_ = bounces;
    }
    /// Probe rays (DdgiVolume): from each probe of a grid, `raysPerProbe`
    /// rays along spherical Fibonacci directions turned by `rotation`
    /// (column-major 4 x 4, its 3 x 3 part), shaded as this tracer shades
    /// (setProbeBounces() bounces after the first hit). probeRays(): x a
    /// ray, y a probe (x + nx (y + ny z)); rgb the radiance arriving along
    /// it, a the first hit's distance (65000: a miss). Use a PathTracer of
    /// its own (or the reflections' one). Submitted, not waited for.
    struct ProbeRays {
        systems::leal::vector_math::Vector3<double> origin;
        systems::leal::vector_math::Vector3<double> spacing;
        uint32_t count[3] = {1, 1, 1};
        uint32_t raysPerProbe = 64;
        float rotation[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    };
    bool traceProbeRays(const RenderScene& scene, const Renderer& renderer, const ProbeRays& rays);
    /// rgba16float, raysPerProbe x probes.
    [[nodiscard]] std::shared_ptr<systems::leal::campello_gpu::Texture> probeRays() const noexcept {
        return probeRayTarget_;
    }
    void setProbeBounces(uint32_t bounces) noexcept {
        probeBounces_ = bounces;
    }

    /// Bounces after the reflected surface (default 1: its direct light and one more).
    void setReflectionBounces(uint32_t bounces) noexcept {
        reflectionBounces_ = bounces;
    }

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
    enum class ToneMapping {
        pbrNeutral,  ///< Khronos PBR Neutral (default; as the raster renderer): true colours
        aces,        ///< ACES filmic (Stephen Hill's fit): contrasty, saturated highlights roll off
        agx,         ///< AgX: highlights desaturate towards white, as film; wide dynamic range
    };
    /// How output() and denoise() map the image to the display. Doesn't
    /// start the image again (it applies to the average when shown).
    void setToneMapping(ToneMapping toneMapping) noexcept {
        toneMapping_ = toneMapping;
    }
    /// Tone mapping of the shader (`toneMapping`, then display gamma):
    /// linear rgb -> rgba8, alpha 255.
    static void toneMap(const float* rgb, size_t pixels, uint8_t* rgba, ToneMapping toneMapping = ToneMapping::pbrNeutral);

    /// Depth of field: a thin lens of `apertureRadius` (world units; 0: a
    /// pinhole, everything sharp, the default) focused at `focusDistance`
    /// along the view direction. Perspective cameras only.
    void setDepthOfField(float apertureRadius, float focusDistance) noexcept {
        apertureRadius_ = apertureRadius > 0.0f ? apertureRadius : 0.0f;
        focusDistance_ = focusDistance > 0.0f ? focusDistance : 1.0f;
        samples_ = 0;
    }
    /// A physical camera's exposure for Renderer::setExposure() (EV, the
    /// image scaled by 2^EV): f-number, shutter time (s) and ISO, with the
    /// usual calibration (1.2 / 2^EV100, Lagarde & de Rousiers 2014), for
    /// scenes lit in physical units (sun about 100000 lux).
    [[nodiscard]] static float exposureForCamera(float fNumber, float shutterSeconds, float iso) noexcept;
    /// The aperture radius of a lens of `focalLength` (world units, e.g.
    /// 0.05 for 50 mm in metres) at `fNumber`, for setDepthOfField().
    [[nodiscard]] static float apertureRadius(float focalLength, float fNumber) noexcept {
        return fNumber > 0.0f ? focalLength / (2.0f * fNumber) : 0.0f;
    }

    /// Light bounces after the first hit (default 4).
    void setMaxBounces(uint32_t bounces) noexcept {
        maxBounces_ = bounces;
        samples_ = 0;
    }

    /// Fewer bounces for the first `samples` samples of every image (a
    /// faster first preview), at most setMaxBounces(); the full count
    /// afterwards. Those first samples stay in the average: they darken
    /// light that needs more bounces by a share that falls as samples are
    /// added (2 of 512 samples: under 0.4%). 0 samples: off (the default).
    void setProgressiveBounces(uint32_t bounces, uint32_t samples) noexcept {
        progressiveBounces_ = bounces;
        progressiveSamples_ = samples;
        samples_ = 0;
    }

    /// Samples per pixel each render() adds, traced in one GPU dispatch
    /// (default 1). More per render() keeps the GPU busy between frames of
    /// a fast scene.
    void setSamplesPerRender(uint32_t samples) noexcept {
        samplesPerRender_ = samples > 0 ? samples : 1;
    }
    /// A GPU time budget per render(), in milliseconds: the samples each
    /// render() adds are chosen from the last measured GPU time per sample
    /// and pixel to fill it (1 to 64). render() still doesn't wait: an
    /// interactive caller skips frames while sampleInFlight(). 0: off,
    /// setSamplesPerRender()'s count (the default).
    void setFrameBudget(double milliseconds) noexcept {
        frameBudgetMs_ = milliseconds > 0.0 ? milliseconds : 0.0;
    }

    /// Adaptive sampling: a pixel stops taking samples once the relative
    /// standard error of its average (luminance; against a floor of 0.05
    /// after exposure, so near-black pixels don't run forever) is under
    /// `relativeError`, after at least `minSamples`. Flat walls and sky
    /// stop early, noisy pixels get the samples. sampleCount() is then
    /// the most any pixel has; stats() counts the converged pixels, a
    /// noise target to stop at instead of a sample count. 0: off (the
    /// default). The estimate is the pixel's own: a few pixels stop a
    /// little early (a small bias, as in every adaptive sampler).
    void setNoiseTarget(float relativeError, uint32_t minSamples = 16) noexcept {
        noiseTarget_ = relativeError > 0.0f ? relativeError : 0.0f;
        noiseMinSamples_ = std::max<uint32_t>(minSamples, 2);
        samples_ = 0;
    }

    /// A faster first image: the first `samples` samples of every image
    /// are traced at 1 / `divisor` of the resolution in each direction
    /// (divisor^2 times fewer rays) and shown upscaled, then the image
    /// starts again at full resolution (sampleCount() from 0). denoise()
    /// during the preview filters the small image and upscales the result.
    /// 0 samples or divisor 1: off (the default).
    void setPreviewResolution(uint32_t divisor, uint32_t samples) noexcept {
        previewDivisor_ = divisor > 0 ? divisor : 1;
        previewSamples_ = samples;
        samples_ = 0;
    }
    /// Whether output() shows the reduced resolution preview.
    [[nodiscard]] bool previewing() const noexcept {
        return previewActive_ && samples_ > 0;
    }

    /// Firefly control: light that reached the camera after two or more
    /// diffuse or glossy bounces is clamped to this radiance (its largest
    /// colour component, before exposure), keeping its hue. Rare bright
    /// paths then don't leave spots that a denoiser can't remove, at the
    /// cost of some indirect energy. 0: no clamp (the default; for stills).
    void setIndirectClamp(float maxRadiance) noexcept {
        indirectClamp_ = maxRadiance > 0.0f ? maxRadiance : 0.0f;
        samples_ = 0;
    }

    enum class Sampler {
        sobol,   ///< Owen-scrambled Sobol (default): less noise for the same samples
        random,  ///< independent random numbers (PCG), for comparisons
    };
    void setSampler(Sampler sampler) noexcept {
        if (sampler_ != sampler) samples_ = 0;
        sampler_ = sampler;
    }

    /// Whether the sky is also sampled as a light, by its brightness, with
    /// shadow rays (multiple importance sampling with the BSDF's
    /// directions). Default on; off finds the sky only by bounces that hit it.
    void setEnvironmentSampling(bool enabled) noexcept {
        if (environmentSampling_ != enabled) samples_ = 0;
        environmentSampling_ = enabled;
    }

    /// A light portal: a rectangle (a window, a glass opening) that the
    /// sky's light comes through, from `corner` along its two sides
    /// `edge1` and `edge2` (world space; edge2 is made perpendicular to
    /// edge1). Two-sided.
    struct Portal {
        systems::leal::vector_math::Vector3<double> corner;
        systems::leal::vector_math::Vector3<double> edge1;
        systems::leal::vector_math::Vector3<double> edge2;
    };
    static constexpr size_t MAX_PORTALS = 32;
    /// Light portals: part of the sky's light samples (setEnvironmentSampling())
    /// aim at the portals in sight instead of at the bright sky, so rooms
    /// lit through windows are found by every path rather than by the few
    /// whose bounces happen to leave through one: much less noise in
    /// interiors. The rest still sample the sky by brightness (sky that
    /// isn't behind a portal, and views from outside). Up to MAX_PORTALS
    /// (the rest are ignored); portals smaller than about 1e-4 sr from a
    /// point aren't aimed at from there. Empty: none (the default).
    void setPortals(std::vector<Portal> portals);
    [[nodiscard]] const std::vector<Portal>& portals() const noexcept {
        return portals_;
    }
    /// Portals for the scene's flat glass: every transparent draw whose
    /// material transmits (glass, as the tracer traces it) and whose mesh
    /// bounds are thin along one axis (under a tenth of the other two), as
    /// the rectangle through the middle of those bounds. Window panes, not
    /// curved glass or one mesh holding a whole building's glass.
    [[nodiscard]] static std::vector<Portal> portalsFromGlass(const RenderScene& scene, const Renderer& renderer);

    /// Whether emissive surfaces (luminaires) are also sampled as lights:
    /// a triangle by its power, a point on it, a shadow ray, with multiple
    /// importance sampling against the BSDF. Default on; off finds them
    /// only by bounces that hit them. Up to 2^20 emissive triangles.
    void setEmissiveSampling(bool enabled) noexcept {
        if (emissiveSampling_ != enabled) samples_ = 0;
        emissiveSampling_ = enabled;
    }

    /// Timings, the baseline for performance work. Milliseconds; 0 while
    /// not measured yet.
    struct Stats {
        double sampleGpuMs = 0.0;         ///< GPU time per sample of the last finished render()
        double averageSampleGpuMs = 0.0;  ///< mean GPU time per sample of the current image
        double rebuildMs = 0.0;           ///< last scene rebuild (acceleration structures, buffers, sky table; wall clock, waits for the GPU)
        double denoiseMs = 0.0;           ///< last denoise() (wall clock: readback, filter, upload)
        double firstDenoisedMs = 0.0;     ///< current image: from its first sample to the end of its first denoise()
        uint32_t triangles = 0;           ///< traced triangles (each mesh once)
        uint32_t instances = 0;           ///< traced instances
        uint32_t samplesPerRender = 0;    ///< samples the last render() added
        uint32_t convergedPixels = 0;     ///< pixels under the noise target in the last finished render() (setNoiseTarget())
        uint32_t tracedPixels = 0;        ///< pixels of that render() (the preview's when previewing)
        uint32_t fullRebuilds = 0;        ///< scene rebuilds so far (meshes or materials changed)
        uint32_t transformRebuilds = 0;   ///< rebuilds of the instances only (only transforms changed)
    };
    /// Doesn't wait for the GPU: a sample still in flight is counted once it finishes.
    [[nodiscard]] Stats stats() const;

private:
    struct MeshEntry {
        std::shared_ptr<systems::leal::campello_gpu::AccelerationStructure> blas;
        uint32_t triangleCount = 0;
        const void* positions = nullptr;  // the buffers it was built from
        const void* indices = nullptr;
    };
    // The scene structure; transformsOnly: the instances and the scene's
    // structure only (the same draws, meshes and materials as the last).
    bool rebuild(const RenderScene& scene, const Renderer& renderer, bool transformsOnly);
    void waitForSample() const;  // the last render()'s GPU work
    // The accumulated image at the traced size (traceWidth_ x traceHeight_).
    bool readTraced(std::vector<float>& rgb, std::vector<float>* albedo, std::vector<float>* normal) const;
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
    std::shared_ptr<systems::leal::campello_gpu::Buffer> vertexNormals_;  // the corners' normals per triangle
    std::shared_ptr<systems::leal::campello_gpu::Buffer> tangents_;       // a tangent per triangle (normal maps)
    // Emissive triangles as lights: world-space triangles (emissiveTriangles
    // in path_tracer.metal) and their cumulative distribution by power.
    std::shared_ptr<systems::leal::campello_gpu::ComputePipeline> emissivePipeline_;
    std::shared_ptr<systems::leal::campello_gpu::BindGroupLayout> emissiveLayout_;
    std::shared_ptr<systems::leal::campello_gpu::Buffer> emissives_;
    std::shared_ptr<systems::leal::campello_gpu::Buffer> lightTable_;
    uint32_t emissiveCount_ = 0;
    bool emissiveSampling_ = true;
    std::vector<std::shared_ptr<systems::leal::campello_gpu::Texture>> textures_;  // base colour textures bound
    std::shared_ptr<systems::leal::campello_gpu::Texture> blank_;  // the unused texture bindings
    std::shared_ptr<systems::leal::campello_gpu::Buffer> uniforms_;
    std::shared_ptr<systems::leal::campello_gpu::Buffer> accum_;
    std::shared_ptr<systems::leal::campello_gpu::Buffer> aux_;  // albedo, then normal sums
    std::shared_ptr<systems::leal::campello_gpu::Buffer> moments_;    // per pixel: sum of squared luminance
    std::shared_ptr<systems::leal::campello_gpu::Buffer> converged_;  // pixels under the noise target, a counter
    float noiseTarget_ = 0.0f;
    uint32_t noiseMinSamples_ = 16;
    mutable uint32_t convergedPixels_ = 0;
    mutable uint32_t convergedOf_ = 0;
    std::shared_ptr<systems::leal::campello_gpu::Texture> output_;
    mutable std::shared_ptr<systems::leal::campello_gpu::Fence> pending_;  // the last sample, not yet waited for
    uint64_t geometryKey_ = 0;   // draws' meshes and materials: a full rebuild
    uint64_t transformKey_ = 0;  // their transforms: the instances and the scene's structure
    uint64_t lightingKey_ = 0;   // lights and lighting settings: the image starts again
    uint64_t viewKey_ = 0;
    uint32_t fullRebuilds_ = 0;
    uint32_t transformRebuilds_ = 0;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    uint32_t traceWidth_ = 0;  // the accumulation's size: width_ x height_, or a preview's
    uint32_t traceHeight_ = 0;
    uint32_t previewDivisor_ = 1;
    uint32_t previewSamples_ = 0;
    bool previewActive_ = false;
    uint32_t samples_ = 0;
    uint32_t maxBounces_ = 4;
    float exposure_ = 0.0f;  // of the last render(), for readLinear()
    Projection projection_ = Projection::perspective;
    bool denoised_ = false;
    // Reflections (traceReflections()): the G-buffer of the call in
    // progress, its output, placeholders for the other mode, its sample index.
    const GBufferTargets* reflectionGBuffer_ = nullptr;
    std::shared_ptr<systems::leal::campello_gpu::Texture> reflectionTarget_;
    std::shared_ptr<systems::leal::campello_gpu::Texture> placeholderDepth_;
    std::shared_ptr<systems::leal::campello_gpu::Texture> placeholderReflection_;
    uint32_t reflectionFrame_ = 0;
    uint32_t reflectionBounces_ = 1;
    // Indirect light samples (traceGlobalIllumination()): with
    // reflectionGBuffer_, mode 4; its output.
    bool giRequest_ = false;
    std::shared_ptr<systems::leal::campello_gpu::Texture> giTarget_;
    uint32_t giBounces_ = 1;
    // Probe rays (traceProbeRays()): the request in progress, its output.
    const ProbeRays* probeRequest_ = nullptr;
    std::shared_ptr<systems::leal::campello_gpu::Texture> probeRayTarget_;
    uint32_t probeBounces_ = 1;
    std::shared_ptr<void> denoiser_;  // Open Image Denoise device, filter and buffers (path_tracer.cpp)
    DenoiserDevice denoiserDevice_ = DenoiserDevice::automatic;
    DenoiserQuality denoiserQuality_ = DenoiserQuality::high;
    bool denoiserPrefilter_ = false;

    // Sampling settings.
    uint32_t progressiveBounces_ = 2;
    uint32_t progressiveSamples_ = 0;
    float indirectClamp_ = 0.0f;
    Sampler sampler_ = Sampler::sobol;
    bool environmentSampling_ = true;
    ToneMapping toneMapping_ = ToneMapping::pbrNeutral;
    float apertureRadius_ = 0.0f;
    float focusDistance_ = 1.0f;

    // The sky as a light: a cumulative distribution over a low resolution cube
    // (environmentLuminance in path_tracer.metal), for this environment.
    bool buildEnvironmentTable(const std::shared_ptr<systems::leal::campello_gpu::Texture>& environment);
    std::shared_ptr<systems::leal::campello_gpu::RayTracingPipeline> envPipeline_;
    std::shared_ptr<systems::leal::campello_gpu::BindGroupLayout> envLayout_;
    std::shared_ptr<systems::leal::campello_gpu::Buffer> envTable_;
    std::vector<Portal> portals_;
    std::shared_ptr<systems::leal::campello_gpu::Buffer> portalBuffer_;  // as path_tracer.metal's Portal
    uint32_t portalCount_ = 0;                                           // uploaded
    bool portalsDirty_ = false;
    const void* envTableSource_ = nullptr;  // the environment envTable_ was built from
    uint32_t envResolution_ = 0;            // per cube face; 0: no table

    // Timings (stats()).
    void collectTiming() const;  // the finished sample's GPU time
    mutable std::shared_ptr<systems::leal::campello_gpu::CommandBuffer> pendingCommands_;
    uint32_t pendingSamples_ = 1;            // samples of pendingCommands_
    uint64_t pendingPixels_ = 0;             // and its traced pixels
    mutable double nsPerPixelSample_ = 0.0;  // measured: GPU time per sample and pixel
    uint32_t samplesPerRender_ = 1;
    double frameBudgetMs_ = 0.0;
    uint32_t lastRenderSamples_ = 0;
    mutable double lastSampleGpuMs_ = 0.0;
    mutable double imageGpuMs_ = 0.0;
    mutable uint32_t imageTimedSamples_ = 0;
    double rebuildMs_ = 0.0;
    double denoiseMs_ = 0.0;
    double firstDenoisedMs_ = 0.0;
    int64_t imageStartNs_ = 0;  // steady clock, the current image's first sample
    uint32_t triangles_ = 0;
    uint32_t instanceCount_ = 0;
};

} // namespace systems::leal::campello_renderer
