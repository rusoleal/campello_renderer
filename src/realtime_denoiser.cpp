#include <campello_renderer/realtime_denoiser.hpp>

#include <campello_gpu/bind_group.hpp>
#include <campello_gpu/bind_group_layout.hpp>
#include <campello_gpu/buffer.hpp>
#include <campello_gpu/command_buffer.hpp>
#include <campello_gpu/command_encoder.hpp>
#include <campello_gpu/descriptors/begin_render_pass_descriptor.hpp>
#include <campello_gpu/descriptors/bind_group_descriptor.hpp>
#include <campello_gpu/descriptors/bind_group_layout_descriptor.hpp>
#include <campello_gpu/descriptors/pipeline_layout_descriptor.hpp>
#include <campello_gpu/descriptors/render_pipeline_descriptor.hpp>
#include <campello_gpu/device.hpp>
#include <campello_gpu/fence.hpp>
#include <campello_gpu/render_pass_encoder.hpp>
#include <campello_gpu/render_pipeline.hpp>
#include <campello_gpu/texture.hpp>
#include <campello_gpu/texture_view.hpp>

#include <algorithm>
#include <vector>

#if defined(__APPLE__)
#include "shaders/metal_denoiser.h"
#endif

namespace systems::leal::campello_renderer {

namespace GPU = systems::leal::campello_gpu;

namespace {

// As DenoiseUniforms in denoiser.metal.
struct DenoiseUniforms {
    float params[4];  // colour blend floor, moments blend floor, history valid, max history
    float depth[4];   // projection depth terms (x: P22, y: P23), relative depth tolerance
    float filter[4];  // step, phi colour, phi normal, phi depth
};

enum class Slot { color, depth };

// Textures at bindings 0..n-1 (fragment, 2D), then optionally a uniform buffer at n.
std::shared_ptr<GPU::BindGroupLayout> makeLayout(GPU::Device& device, const std::vector<Slot>& textures, bool uniforms) {
    GPU::BindGroupLayoutDescriptor desc{};
    for (uint32_t i = 0; i < textures.size(); ++i) {
        GPU::EntryObject t{};
        t.binding = i;
        t.visibility = GPU::ShaderStage::fragment;
        t.type = GPU::EntryObjectType::texture;
        t.data.texture.multisampled = false;
        t.data.texture.sampleType =
            textures[i] == Slot::depth ? GPU::EntryObjectTextureType::ttDepth : GPU::EntryObjectTextureType::ttUnfilterableFloat;
        t.data.texture.viewDimension = GPU::TextureType::tt2d;
        desc.entries.push_back(t);
    }
    if (uniforms) {
        GPU::EntryObject b{};
        b.binding = static_cast<uint32_t>(textures.size());
        b.visibility = GPU::ShaderStage::fragment;
        b.type = GPU::EntryObjectType::buffer;
        b.data.buffer.hasDinamicOffaset = false;
        b.data.buffer.minBindingSize = sizeof(DenoiseUniforms);
        b.data.buffer.type = GPU::EntryObjectBufferType::uniform;
        desc.entries.push_back(b);
    }
    return device.createBindGroupLayout(desc);
}

std::shared_ptr<GPU::RenderPipeline> makePipeline(GPU::Device& device, const std::shared_ptr<GPU::ShaderModule>& module,
                                                  const char* fragment,
                                                  const std::shared_ptr<GPU::BindGroupLayout>& layout,
                                                  const std::vector<GPU::PixelFormat>& targets) {
    if (!layout) return nullptr;
    GPU::PipelineLayoutDescriptor pl{};
    pl.bindGroupLayouts = {layout};
    GPU::RenderPipelineDescriptor d{};
    d.layout = device.createPipelineLayout(pl);
    d.vertex.module = module;
    d.vertex.entryPoint = "denoiseVertex";
    d.topology = GPU::PrimitiveTopology::triangleList;
    d.cullMode = GPU::CullMode::none;
    d.frontFace = GPU::FrontFace::ccw;
    GPU::FragmentDescriptor frag{};
    frag.module = module;
    frag.entryPoint = fragment;
    for (GPU::PixelFormat format : targets) {
        GPU::ColorState target{};
        target.format = format;
        target.writeMask = GPU::ColorWrite::all;
        frag.targets.push_back(target);
    }
    d.fragment = frag;
    return device.createRenderPipeline(d);
}

} // namespace

RealtimeDenoiser::RealtimeDenoiser(std::shared_ptr<GPU::Device> device)
    : device_(std::move(device)) {
#if defined(__APPLE__)
    if (!device_) return;
    auto module = device_->createShaderModule(shaders::kDenoiserMetalShader, shaders::kDenoiserMetalShaderSize);
    if (!module) return;
    auto& d = *device_;
    temporalLayout_ = makeLayout(d,
                                 {Slot::color, Slot::color, Slot::color, Slot::color, Slot::depth, Slot::color,
                                  Slot::color, Slot::color},
                                 true);
    varianceLayout_ = makeLayout(d, {Slot::color, Slot::color, Slot::color}, true);
    atrousLayout_ = makeLayout(d, {Slot::color, Slot::color}, true);
    atrousFirstLayout_ = makeLayout(d, {Slot::color, Slot::color, Slot::color}, true);
    modulateLayout_ = makeLayout(d, {Slot::color, Slot::color, Slot::color, Slot::color}, true);
    const auto half4 = GPU::PixelFormat::rgba16float;
    const auto float4 = GPU::PixelFormat::rgba32float;
    variancePipeline_ = makePipeline(d, module, "denoiseVariance", varianceLayout_, {half4});
    atrousPipeline_ = makePipeline(d, module, "denoiseAtrous", atrousLayout_, {half4});
    atrousFirstPipeline_ = makePipeline(d, module, "denoiseAtrousFirst", atrousFirstLayout_, {half4, half4});
    modulatePipeline_ = makePipeline(d, module, "denoiseModulate", modulateLayout_, {half4});
    if (variancePipeline_ && atrousPipeline_ && atrousFirstPipeline_ && modulatePipeline_) {
        temporalPipeline_ = makePipeline(d, module, "denoiseTemporal", temporalLayout_, {half4, half4, float4});
    }
#endif
}

RealtimeDenoiser::~RealtimeDenoiser() {
    if (pending_) pending_->wait();  // the GPU may still be using the textures
}

bool RealtimeDenoiser::ensureTargets(uint32_t width, uint32_t height) {
    if (width == width_ && height == height_ && output_) return true;
    const auto usage = (GPU::TextureUsage)(uint32_t(GPU::TextureUsage::renderTarget) |
                                           uint32_t(GPU::TextureUsage::textureBinding) |
                                           uint32_t(GPU::TextureUsage::copySrc));
    const auto make = [&](Target& target, GPU::PixelFormat format) {
        target.texture = device_->createTexture(GPU::TextureType::tt2d, format, width, height, 1, 1, 1, usage);
        target.view = target.texture ? target.texture->createView(format, 1, GPU::Aspect::all, 0, 0, GPU::TextureType::tt2d)
                                     : nullptr;
        return target.view != nullptr;
    };
    bool ok = true;
    for (int k = 0; k < 2; ++k) {
        ok = make(historyColor_[k], GPU::PixelFormat::rgba16float) && ok;
        ok = make(historyMoments_[k], GPU::PixelFormat::rgba16float) && ok;
        ok = make(geometry_[k], GPU::PixelFormat::rgba32float) && ok;
    }
    ok = make(accumulated_, GPU::PixelFormat::rgba16float) && ok;
    ok = make(pingA_, GPU::PixelFormat::rgba16float) && ok;
    ok = make(pingB_, GPU::PixelFormat::rgba16float) && ok;
    ok = make(outputTarget_, GPU::PixelFormat::rgba16float) && ok;
    width_ = ok ? width : 0;
    height_ = ok ? height : 0;
    output_ = ok ? outputTarget_.texture : nullptr;
    historyValid_ = false;
    return ok;
}

bool RealtimeDenoiser::denoise(const std::shared_ptr<GPU::Texture>& noisy, const GBufferTargets& gbuffer,
                               const systems::leal::vector_math::Matrix4<double>& projection) {
    if (!supported() || !noisy || !gbuffer.normalRoughness || !gbuffer.albedoMetallic || !gbuffer.motion || !gbuffer.depth) {
        return false;
    }
    const uint32_t w = gbuffer.width, h = gbuffer.height;
    if (w == 0 || h == 0 || noisy->getWidth() != w || noisy->getHeight() != h) return false;
    if (pending_) {
        pending_->wait();  // its uniforms are rewritten below
        pending_.reset();
    }
    if (!ensureTargets(w, h)) return false;

    const uint32_t iterations = std::clamp<uint32_t>(settings_.iterations, 1u, 7u);
    // Uniforms: [0] the temporal and variance passes, [1..] each a-trous step.
    for (uint32_t i = 0; i <= iterations; ++i) {
        if (!uniforms_[i]) uniforms_[i] = device_->createBuffer(sizeof(DenoiseUniforms), GPU::BufferUsage::uniform);
        if (!uniforms_[i]) return false;
        DenoiseUniforms u{};
        u.params[0] = settings_.colorBlend;
        u.params[1] = settings_.momentsBlend;
        u.params[2] = historyValid_ ? 1.0f : 0.0f;
        u.params[3] = std::max(settings_.maxHistory, 1.0f);
        u.depth[0] = static_cast<float>(projection.data[10]);
        u.depth[1] = static_cast<float>(projection.data[11]);
        u.depth[2] = settings_.depthTolerance;
        u.depth[3] = settings_.demodulate ? 1.0f : 0.0f;
        u.filter[0] = i == 0 ? 1.0f : static_cast<float>(1u << (i - 1));
        u.filter[1] = settings_.phiColor;
        u.filter[2] = settings_.phiNormal;
        u.filter[3] = settings_.phiDepth;
        uniforms_[i]->upload(0, sizeof(u), &u);
    }

    const uint32_t read = historyIndex_, write = 1u - historyIndex_;

    auto encoder = device_->createCommandEncoder();
    if (!encoder) return false;
    std::vector<std::shared_ptr<GPU::BindGroup>> groups;
    const auto pass = [&](const std::shared_ptr<GPU::RenderPipeline>& pipeline,
                          const std::shared_ptr<GPU::BindGroupLayout>& layout,
                          std::vector<GPU::BindGroupEntryDescriptor> entries,
                          const std::vector<std::shared_ptr<GPU::TextureView>>& targets) {
        GPU::BindGroupDescriptor desc{};
        desc.layout = layout;
        desc.entries = std::move(entries);
        auto group = device_->createBindGroup(desc);
        if (!group) return false;
        groups.push_back(group);
        GPU::BeginRenderPassDescriptor rp{};
        for (const auto& target : targets) {
            GPU::ColorAttachment ca{};
            ca.view = target;
            ca.loadOp = GPU::LoadOp::clear;
            ca.storeOp = GPU::StoreOp::store;
            ca.depthSlice = 0;
            rp.colorAttachments.push_back(ca);
        }
        auto rpe = encoder->beginRenderPass(rp);
        if (!rpe) return false;
        rpe->setViewport(0.0f, 0.0f, float(w), float(h), 0.0f, 1.0f);
        rpe->setScissorRect(0.0f, 0.0f, float(w), float(h));
        rpe->setPipeline(pipeline);
        rpe->setBindGroup(0, group);
        rpe->draw(3);
        rpe->end();
        return true;
    };
    const auto uniform = [&](uint32_t binding, uint32_t index) {
        return GPU::BindGroupEntryDescriptor{binding, GPU::BufferBinding{uniforms_[index], 0, sizeof(DenoiseUniforms)}};
    };

    bool ok = pass(temporalPipeline_, temporalLayout_,
                   {{0, noisy},
                    {1, historyColor_[read].texture},
                    {2, historyMoments_[read].texture},
                    {3, gbuffer.normalRoughness},
                    {4, gbuffer.depth},
                    {5, gbuffer.motion},
                    {6, geometry_[read].texture},
                    {7, gbuffer.albedoMetallic},
                    uniform(8, 0)},
                   {accumulated_.view, historyMoments_[write].view, geometry_[write].view});
    ok = ok && pass(variancePipeline_, varianceLayout_,
                    {{0, accumulated_.texture}, {1, historyMoments_[write].texture}, {2, geometry_[write].texture}, uniform(3, 0)},
                    {pingA_.view});
    // A-trous: A -> B (and the next history), then alternating.
    ok = ok && pass(atrousFirstPipeline_, atrousFirstLayout_,
                    {{0, pingA_.texture}, {1, geometry_[write].texture}, {2, historyMoments_[write].texture}, uniform(3, 1)},
                    {pingB_.view, historyColor_[write].view});
    Target* source = &pingB_;
    Target* destination = &pingA_;
    for (uint32_t i = 1; ok && i < iterations; ++i) {
        ok = pass(atrousPipeline_, atrousLayout_, {{0, source->texture}, {1, geometry_[write].texture}, uniform(2, i + 1)},
                  {destination->view});
        std::swap(source, destination);
    }
    ok = ok && pass(modulatePipeline_, modulateLayout_,
                    {{0, source->texture}, {1, gbuffer.albedoMetallic}, {2, noisy}, {3, geometry_[write].texture}, uniform(4, 0)},
                    {outputTarget_.view});
    if (!ok) return false;

    auto fence = device_->createFence();
    auto commands = encoder->finish();
    device_->submit(commands, fence);
    timing_.begin();
    timing_.submitted(commands, fence);
    pending_ = fence;
    historyIndex_ = write;
    historyValid_ = true;
    return true;
}

} // namespace systems::leal::campello_renderer
