#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <campello_gpu/command_encoder.hpp>
#include <campello_gpu/descriptors/begin_render_pass_descriptor.hpp>
#include <campello_gpu/descriptors/render_pipeline_descriptor.hpp>
#include <campello_gpu/device.hpp>
#include <campello_gpu/render_pass_encoder.hpp>
#include <campello_gpu/render_pipeline.hpp>

#include "overlay_shaders.h"

// Draws lines of text in the top-left corner, inset to stay inside a TV's safe
// area. Each lit pixel of a 5x7 bitmap font is one quad, so it needs no textures.
class Overlay {
public:
    Overlay(const std::shared_ptr<systems::leal::campello_gpu::Device>& device,
            uint32_t width, uint32_t height)
        : width(width), height(height) {
        namespace GPU = systems::leal::campello_gpu;

        GPU::RenderPipelineDescriptor desc{};
        desc.vertex.module = device->createShaderModule(kOverlayVertexShader, sizeof(kOverlayVertexShader));
        desc.vertex.entryPoint = "vertexMain";

        GPU::VertexAttribute position{};
        position.componentType = GPU::ComponentType::ctFloat;
        position.accessorType = GPU::AccessorType::acVec2;
        position.offset = 0;
        position.shaderLocation = 0;
        GPU::VertexAttribute color{};
        color.componentType = GPU::ComponentType::ctFloat;
        color.accessorType = GPU::AccessorType::acVec3;
        color.offset = 8;
        color.shaderLocation = 1;

        GPU::VertexLayout layout{};
        layout.arrayStride = sizeof(Vertex);
        layout.stepMode = GPU::StepMode::vertex;
        layout.attributes = {position, color};
        desc.vertex.buffers.push_back(layout);

        desc.topology = GPU::PrimitiveTopology::triangleList;
        desc.cullMode = GPU::CullMode::none;
        desc.frontFace = GPU::FrontFace::ccw;

        GPU::ColorState colorState{};
        colorState.format = device->getSwapchainPixelFormat();
        colorState.writeMask = GPU::ColorWrite::all;
        GPU::FragmentDescriptor fragment{};
        fragment.module = device->createShaderModule(kOverlayPixelShader, sizeof(kOverlayPixelShader));
        fragment.entryPoint = "pixelMain";
        fragment.targets.push_back(colorState);
        desc.fragment = fragment;

        pipeline = device->createRenderPipeline(desc);
        for (auto& buffer : vertexBuffers) {
            buffer = device->createBuffer(kMaxVertices * sizeof(Vertex), GPU::BufferUsage::vertex);
        }
    }

    bool isReady() const { return pipeline && vertexBuffers[0]; }

    void setLines(const std::vector<std::string>& lines) {
        vertices.clear();

        const float scale = static_cast<float>(height) / 360.0f;
        const float marginX = static_cast<float>(width) * 0.05f;
        const float marginY = static_cast<float>(height) * 0.05f;
        const float advanceX = 6.0f * scale;
        const float advanceY = 9.0f * scale;

        size_t longest = 0;
        for (const auto& line : lines) longest = std::max(longest, line.size());
        addQuad(marginX - scale, marginY - scale,
                static_cast<float>(longest) * advanceX + scale,
                static_cast<float>(lines.size()) * advanceY, 0.0f, 0.0f, 0.0f);

        float y = marginY;
        for (const auto& line : lines) {
            float x = marginX;
            for (const char c : line) {
                const auto& rows = glyph(c);
                for (int row = 0; row < 7; ++row) {
                    for (int column = 0; column < 5; ++column) {
                        if (rows[row] & (0x10 >> column)) {
                            addQuad(x + static_cast<float>(column) * scale,
                                    y + static_cast<float>(row) * scale, scale, scale, 0.4f, 1.0f, 0.4f);
                        }
                    }
                }
                x += advanceX;
            }
            y += advanceY;
        }
    }

    // frameIndex picks a buffer the GPU isn't still reading from an earlier frame.
    void draw(const std::shared_ptr<systems::leal::campello_gpu::CommandEncoder>& encoder,
              const std::shared_ptr<systems::leal::campello_gpu::TextureView>& colorView,
              uint32_t frameIndex) {
        namespace GPU = systems::leal::campello_gpu;
        if (vertices.empty()) return;

        auto& buffer = vertexBuffers[frameIndex % vertexBuffers.size()];
        buffer->upload(0, vertices.size() * sizeof(Vertex), vertices.data());

        GPU::ColorAttachment attachment{};
        attachment.view = colorView;
        attachment.loadOp = GPU::LoadOp::load;
        attachment.storeOp = GPU::StoreOp::store;
        attachment.depthSlice = 0;
        GPU::BeginRenderPassDescriptor pass{};
        pass.colorAttachments = {attachment};

        auto rpe = encoder->beginRenderPass(pass);
        if (!rpe) return;
        const auto w = static_cast<float>(width);
        const auto h = static_cast<float>(height);
        rpe->setViewport(0.0f, 0.0f, w, h, 0.0f, 1.0f);
        rpe->setScissorRect(0.0f, 0.0f, w, h);
        rpe->setPipeline(pipeline);
        rpe->setVertexBuffer(0, buffer, 0);
        rpe->draw(static_cast<uint32_t>(vertices.size()));
        rpe->end();
    }

private:
    struct Vertex {
        float x, y;
        float r, g, b;
    };

    // About 40 characters of 35 pixels, plus the background.
    static constexpr size_t kMaxVertices = 8500;
    static constexpr size_t kFramesInFlight = 3; // Renderer::kMaxFramesInFlight

    void addQuad(float x, float y, float w, float h, float r, float g, float b) {
        if (vertices.size() + 6 > kMaxVertices) return;
        const auto clipX = [&](float value) { return value / static_cast<float>(width) * 2.0f - 1.0f; };
        const auto clipY = [&](float value) { return 1.0f - value / static_cast<float>(height) * 2.0f; };
        const Vertex topLeft{clipX(x), clipY(y), r, g, b};
        const Vertex topRight{clipX(x + w), clipY(y), r, g, b};
        const Vertex bottomLeft{clipX(x), clipY(y + h), r, g, b};
        const Vertex bottomRight{clipX(x + w), clipY(y + h), r, g, b};
        vertices.insert(vertices.end(), {topLeft, topRight, bottomLeft, bottomLeft, topRight, bottomRight});
    }

    // One byte per row, top to bottom, using the low 5 bits. Only the characters
    // the stats need; anything else is blank.
    static const std::array<uint8_t, 7>& glyph(char c) {
        static const std::array<uint8_t, 7> digits[10] = {
            {{0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E}}, {{0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E}},
            {{0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F}}, {{0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E}},
            {{0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02}}, {{0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E}},
            {{0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E}}, {{0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08}},
            {{0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E}}, {{0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C}},
        };
        static const std::array<uint8_t, 7> blank = {{0, 0, 0, 0, 0, 0, 0}};
        static const std::array<uint8_t, 7> dot = {{0, 0, 0, 0, 0, 0x0C, 0x0C}};
        static const std::array<uint8_t, 7> f = {{0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10}};
        static const std::array<uint8_t, 7> g = {{0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F}};
        static const std::array<uint8_t, 7> m = {{0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11}};
        static const std::array<uint8_t, 7> u = {{0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E}};
        static const std::array<uint8_t, 7> p = {{0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10}};
        static const std::array<uint8_t, 7> s = {{0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E}};
        static const std::array<uint8_t, 7> x = {{0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11}};

        if (c >= '0' && c <= '9') return digits[c - '0'];
        switch (c) {
            case '.': return dot;
            case 'F': return f;
            case 'G': return g;
            case 'M': return m;
            case 'P': return p;
            case 'U': return u;
            case 'S': return s;
            case 'X': return x;
            default:  return blank;
        }
    }

    uint32_t width;
    uint32_t height;
    std::shared_ptr<systems::leal::campello_gpu::RenderPipeline> pipeline;
    std::array<std::shared_ptr<systems::leal::campello_gpu::Buffer>, kFramesInFlight> vertexBuffers;
    std::vector<Vertex> vertices;
};
