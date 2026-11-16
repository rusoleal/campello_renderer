#include <campello_renderer/ddgi.hpp>
#include <campello_renderer/path_tracer.hpp>

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
#include <cmath>
#include <cstring>
#include <vector>

#if defined(__APPLE__)
#include "shaders/metal_ddgi.h"
#endif

namespace systems::leal::campello_renderer {

namespace GPU = systems::leal::campello_gpu;
namespace VM = systems::leal::vector_math;

// (Named: the unity build joins this file with others' anonymous helpers.)
namespace ddgi_detail {

constexpr uint32_t IRRADIANCE_RES = 8;  // as ddgi.metal
constexpr uint32_t DISTANCE_RES = 16;

// As DdgiUniforms in ddgi.metal.
struct DdgiUniforms {
    float origin[4];
    float spacing[4];
    uint32_t count[4];
    float rotation[16];
    float invViewProj[16];
    float eye[4];
    float bias[4];
};

std::shared_ptr<GPU::BindGroupLayout> layout(GPU::Device& device, const std::vector<GPU::EntryObjectTextureType>& textures) {
    GPU::BindGroupLayoutDescriptor desc{};
    for (uint32_t i = 0; i < textures.size(); ++i) {
        GPU::EntryObject t{};
        t.binding = i;
        t.visibility = GPU::ShaderStage::fragment;
        t.type = GPU::EntryObjectType::texture;
        t.data.texture.multisampled = false;
        t.data.texture.sampleType = textures[i];
        t.data.texture.viewDimension = GPU::TextureType::tt2d;
        desc.entries.push_back(t);
    }
    GPU::EntryObject b{};
    b.binding = static_cast<uint32_t>(textures.size());
    b.visibility = GPU::ShaderStage::fragment;
    b.type = GPU::EntryObjectType::buffer;
    b.data.buffer.hasDinamicOffaset = false;
    b.data.buffer.minBindingSize = sizeof(DdgiUniforms);
    b.data.buffer.type = GPU::EntryObjectBufferType::uniform;
    desc.entries.push_back(b);
    return device.createBindGroupLayout(desc);
}

std::shared_ptr<GPU::RenderPipeline> pipeline(GPU::Device& device, const std::shared_ptr<GPU::ShaderModule>& module,
                                              const char* fragment, const std::shared_ptr<GPU::BindGroupLayout>& layout) {
    if (!layout) return nullptr;
    GPU::PipelineLayoutDescriptor pl{};
    pl.bindGroupLayouts = {layout};
    GPU::RenderPipelineDescriptor d{};
    d.layout = device.createPipelineLayout(pl);
    d.vertex.module = module;
    d.vertex.entryPoint = "ddgiVertex";
    d.topology = GPU::PrimitiveTopology::triangleList;
    d.cullMode = GPU::CullMode::none;
    d.frontFace = GPU::FrontFace::ccw;
    GPU::FragmentDescriptor frag{};
    frag.module = module;
    frag.entryPoint = fragment;
    GPU::ColorState target{};
    target.format = GPU::PixelFormat::rgba16float;
    target.writeMask = GPU::ColorWrite::all;
    frag.targets.push_back(target);
    d.fragment = frag;
    return device.createRenderPipeline(d);
}

uint32_t hashU(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// A uniformly random rotation (Shoemake's quaternion), column-major 4 x 4.
void randomRotation(uint32_t seed, float* out) {
    const double u1 = double(hashU(seed * 3u + 1u) >> 8) / 16777216.0;
    const double u2 = double(hashU(seed * 3u + 2u) >> 8) / 16777216.0;
    const double u3 = double(hashU(seed * 3u + 3u) >> 8) / 16777216.0;
    const double pi2 = 2.0 * 3.14159265358979323846;
    const double x = std::sqrt(1.0 - u1) * std::sin(pi2 * u2), y = std::sqrt(1.0 - u1) * std::cos(pi2 * u2);
    const double z = std::sqrt(u1) * std::sin(pi2 * u3), w = std::sqrt(u1) * std::cos(pi2 * u3);
    const double m[3][3] = {{1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)},
                            {2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)},
                            {2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)}};
    for (int i = 0; i < 16; ++i) out[i] = (i % 5 == 0) ? 1.0f : 0.0f;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) out[c * 4 + r] = static_cast<float>(m[r][c]);
    }
}

} // namespace ddgi_detail

DdgiVolume::Grid DdgiVolume::fitting(const VM::Vector3<double>& min, const VM::Vector3<double>& max, double spacing) {
    Grid grid;
    spacing = std::max(spacing, 1e-3);
    const double lo[3] = {min.x(), min.y(), min.z()}, hi[3] = {max.x(), max.y(), max.z()};
    double s[3];
    for (int a = 0; a < 3; ++a) {
        const double extent = std::max(hi[a] - lo[a], 0.0);
        grid.count[a] = static_cast<uint32_t>(std::clamp(std::ceil(extent / spacing) + 1.0, 2.0, 32.0));
        s[a] = std::max(extent / double(grid.count[a] - 1), 1e-3);
    }
    grid.origin = VM::Vector3<double>(lo[0], lo[1], lo[2]);
    grid.spacing = VM::Vector3<double>(s[0], s[1], s[2]);
    return grid;
}

DdgiVolume::DdgiVolume(std::shared_ptr<GPU::Device> device)
    : device_(std::move(device)) {
#if defined(__APPLE__)
    if (!device_) return;
    auto module = device_->createShaderModule(shaders::kDdgiMetalShader, shaders::kDdgiMetalShaderSize);
    if (!module) return;
    using TT = GPU::EntryObjectTextureType;
    updateLayout_ = ddgi_detail::layout(*device_, {TT::ttUnfilterableFloat, TT::ttUnfilterableFloat});
    resolveLayout_ = ddgi_detail::layout(*device_, {TT::ttFloat, TT::ttFloat, TT::ttUnfilterableFloat, TT::ttDepth});
    irradiancePipeline_ = ddgi_detail::pipeline(*device_, module, "ddgiUpdateIrradiance", updateLayout_);
    distancePipeline_ = ddgi_detail::pipeline(*device_, module, "ddgiUpdateDistance", updateLayout_);
    if (irradiancePipeline_ && distancePipeline_) {
        resolvePipeline_ = ddgi_detail::pipeline(*device_, module, "ddgiResolve", resolveLayout_);
    }
#endif
}

DdgiVolume::~DdgiVolume() {
    if (pendingUpdate_) pendingUpdate_->wait();
    if (pendingResolve_) pendingResolve_->wait();
}

void DdgiVolume::setGrid(const Grid& grid) {
    grid_ = grid;
    for (uint32_t& c : grid_.count) c = std::max(c, 2u);
    grid_.raysPerProbe = std::clamp(grid_.raysPerProbe, 8u, 1024u);
    historyValid_ = false;
    irradianceAtlas_ = {};
    distanceAtlas_ = {};
}

bool DdgiVolume::ensureAtlases() {
    if (irradianceAtlas_[0].texture) return true;
    const uint32_t tilesPerRow = grid_.count[0] * grid_.count[1], rows = grid_.count[2];
    const auto usage = (GPU::TextureUsage)(uint32_t(GPU::TextureUsage::renderTarget) |
                                           uint32_t(GPU::TextureUsage::textureBinding) |
                                           uint32_t(GPU::TextureUsage::copySrc));
    const auto make = [&](Target& target, uint32_t res) {
        const uint32_t tile = res + 2;
        target.texture = device_->createTexture(GPU::TextureType::tt2d, GPU::PixelFormat::rgba16float, tilesPerRow * tile,
                                                rows * tile, 1, 1, 1, usage);
        target.view = target.texture ? target.texture->createView(GPU::PixelFormat::rgba16float, 1, GPU::Aspect::all, 0, 0,
                                                                  GPU::TextureType::tt2d)
                                     : nullptr;
        return target.view != nullptr;
    };
    bool ok = true;
    for (int k = 0; k < 2; ++k) {
        ok = make(irradianceAtlas_[k], ddgi_detail::IRRADIANCE_RES) && ok;
        ok = make(distanceAtlas_[k], ddgi_detail::DISTANCE_RES) && ok;
    }
    if (!ok) {
        irradianceAtlas_ = {};
        distanceAtlas_ = {};
    }
    historyValid_ = false;
    return ok;
}

void DdgiVolume::uniforms(float* out, const CameraData* camera) const {
    ddgi_detail::DdgiUniforms u{};
    u.origin[0] = static_cast<float>(grid_.origin.x());
    u.origin[1] = static_cast<float>(grid_.origin.y());
    u.origin[2] = static_cast<float>(grid_.origin.z());
    u.origin[3] = historyValid_ ? grid_.hysteresis : 0.0f;
    u.spacing[0] = static_cast<float>(grid_.spacing.x());
    u.spacing[1] = static_cast<float>(grid_.spacing.y());
    u.spacing[2] = static_cast<float>(grid_.spacing.z());
    const double largest = std::max(grid_.spacing.x(), std::max(grid_.spacing.y(), grid_.spacing.z()));
    u.spacing[3] = grid_.maxDistance > 0.0f ? grid_.maxDistance : static_cast<float>(1.5 * largest);
    for (int k = 0; k < 3; ++k) u.count[k] = grid_.count[k];
    u.count[3] = grid_.raysPerProbe;
    std::copy(std::begin(rotation_), std::end(rotation_), u.rotation);
    if (camera) {
        const auto inverse = (camera->projection * camera->view).inverted();
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) u.invViewProj[c * 4 + r] = static_cast<float>(inverse.data[r * 4 + c]);
        }
        const auto eye = camera->view.inverted() * VM::Vector4<double>(0.0, 0.0, 0.0, 1.0);
        u.eye[0] = static_cast<float>(eye.x() / eye.w());
        u.eye[1] = static_cast<float>(eye.y() / eye.w());
        u.eye[2] = static_cast<float>(eye.z() / eye.w());
    }
    u.eye[3] = historyValid_ ? 1.0f : 0.0f;
    u.bias[0] = grid_.normalBias;
    u.bias[1] = grid_.viewBias;
    std::memcpy(out, &u, sizeof(u));
}

bool DdgiVolume::update(PathTracer& tracer, const RenderScene& scene, const Renderer& renderer) {
    if (!supported() || !ensureAtlases()) return false;
    if (pendingUpdate_) {
        pendingUpdate_->wait();  // its uniforms are rewritten below
        pendingUpdate_.reset();
    }
    // This frame's rays: turned by a new random rotation, so the probes see
    // every direction over frames.
    ddgi_detail::randomRotation(frame_, rotation_);
    PathTracer::ProbeRays request;
    request.origin = grid_.origin;
    request.spacing = grid_.spacing;
    for (int k = 0; k < 3; ++k) request.count[k] = grid_.count[k];
    request.raysPerProbe = grid_.raysPerProbe;
    std::copy(std::begin(rotation_), std::end(rotation_), request.rotation);
    if (!tracer.traceProbeRays(scene, renderer, request) || !tracer.probeRays()) return false;

    float data[sizeof(ddgi_detail::DdgiUniforms) / sizeof(float)];
    uniforms(data, nullptr);
    if (!updateUniforms_) updateUniforms_ = device_->createBuffer(sizeof(data), GPU::BufferUsage::uniform);
    if (!updateUniforms_) return false;
    updateUniforms_->upload(0, sizeof(data), data);

    const uint32_t read = historyIndex_, write = 1u - historyIndex_;
    auto encoder = device_->createCommandEncoder();
    if (!encoder) return false;
    std::vector<std::shared_ptr<GPU::BindGroup>> groups;
    const auto pass = [&](const std::shared_ptr<GPU::RenderPipeline>& pipeline, const Target& previous, const Target& target) {
        GPU::BindGroupDescriptor desc{};
        desc.layout = updateLayout_;
        desc.entries = {{0, tracer.probeRays()},
                        {1, previous.texture},
                        {2, GPU::BufferBinding{updateUniforms_, 0, sizeof(ddgi_detail::DdgiUniforms)}}};
        auto group = device_->createBindGroup(desc);
        if (!group) return false;
        groups.push_back(group);
        GPU::BeginRenderPassDescriptor rp{};
        GPU::ColorAttachment ca{};
        ca.view = target.view;
        ca.loadOp = GPU::LoadOp::clear;
        ca.storeOp = GPU::StoreOp::store;
        ca.depthSlice = 0;
        rp.colorAttachments.push_back(ca);
        auto rpe = encoder->beginRenderPass(rp);
        if (!rpe) return false;
        const float w = float(target.texture->getWidth()), h = float(target.texture->getHeight());
        rpe->setViewport(0.0f, 0.0f, w, h, 0.0f, 1.0f);
        rpe->setScissorRect(0.0f, 0.0f, w, h);
        rpe->setPipeline(pipeline);
        rpe->setBindGroup(0, group);
        rpe->draw(3);
        rpe->end();
        return true;
    };
    if (!pass(irradiancePipeline_, irradianceAtlas_[read], irradianceAtlas_[write]) ||
        !pass(distancePipeline_, distanceAtlas_[read], distanceAtlas_[write])) {
        return false;
    }
    auto fence = device_->createFence();
    auto commands = encoder->finish();
    device_->submit(commands, fence);
    updateTiming_.begin();
    updateTiming_.submitted(commands, fence);
    pendingUpdate_ = fence;
    historyIndex_ = write;
    historyValid_ = true;
    ++frame_;
    return true;
}

bool DdgiVolume::resolve(const GBufferTargets& gbuffer, const CameraData& camera) {
    if (!supported() || !irradianceAtlas_[0].texture || !gbuffer.normalRoughness || !gbuffer.depth || gbuffer.width == 0) {
        return false;
    }
    if (pendingResolve_) {
        pendingResolve_->wait();
        pendingResolve_.reset();
    }
    const uint32_t w = gbuffer.width, h = gbuffer.height;
    if (!resolveTarget_.texture || resolveTarget_.texture->getWidth() != w || resolveTarget_.texture->getHeight() != h) {
        resolveTarget_.texture = device_->createTexture(GPU::TextureType::tt2d, GPU::PixelFormat::rgba16float, w, h, 1, 1, 1,
                                                        (GPU::TextureUsage)(uint32_t(GPU::TextureUsage::renderTarget) |
                                                                            uint32_t(GPU::TextureUsage::textureBinding) |
                                                                            uint32_t(GPU::TextureUsage::copySrc)));
        resolveTarget_.view = resolveTarget_.texture ? resolveTarget_.texture->createView(GPU::PixelFormat::rgba16float, 1,
                                                                                          GPU::Aspect::all, 0, 0,
                                                                                          GPU::TextureType::tt2d)
                                                     : nullptr;
        if (!resolveTarget_.view) return false;
    }
    float data[sizeof(ddgi_detail::DdgiUniforms) / sizeof(float)];
    uniforms(data, &camera);
    if (!resolveUniforms_) resolveUniforms_ = device_->createBuffer(sizeof(data), GPU::BufferUsage::uniform);
    if (!resolveUniforms_) return false;
    resolveUniforms_->upload(0, sizeof(data), data);
    GPU::BindGroupDescriptor desc{};
    desc.layout = resolveLayout_;
    desc.entries = {{0, irradianceAtlas_[historyIndex_].texture},
                    {1, distanceAtlas_[historyIndex_].texture},
                    {2, gbuffer.normalRoughness},
                    {3, gbuffer.depth},
                    {4, GPU::BufferBinding{resolveUniforms_, 0, sizeof(ddgi_detail::DdgiUniforms)}}};
    auto group = device_->createBindGroup(desc);
    auto encoder = device_->createCommandEncoder();
    if (!group || !encoder) return false;
    GPU::BeginRenderPassDescriptor rp{};
    GPU::ColorAttachment ca{};
    ca.view = resolveTarget_.view;
    ca.loadOp = GPU::LoadOp::clear;
    ca.storeOp = GPU::StoreOp::store;
    ca.depthSlice = 0;
    rp.colorAttachments.push_back(ca);
    if (auto rpe = encoder->beginRenderPass(rp)) {
        rpe->setViewport(0.0f, 0.0f, float(w), float(h), 0.0f, 1.0f);
        rpe->setScissorRect(0.0f, 0.0f, float(w), float(h));
        rpe->setPipeline(resolvePipeline_);
        rpe->setBindGroup(0, group);
        rpe->draw(3);
        rpe->end();
    }
    auto fence = device_->createFence();
    auto commands = encoder->finish();
    device_->submit(commands, fence);
    resolveTiming_.begin();
    resolveTiming_.submitted(commands, fence);
    pendingResolve_ = fence;
    irradiance_ = resolveTarget_.texture;
    return true;
}

} // namespace systems::leal::campello_renderer
