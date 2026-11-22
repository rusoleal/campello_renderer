#include <campello_renderer/procedural_texture_baker.hpp>

#include "procedural_bridge.hpp"

#include <campello_image/image.hpp>

#include <cstring>
#include <iostream>

#if defined(__EMSCRIPTEN_PTHREADS__)
#include <emscripten/threading.h>  // emscripten_is_main_browser_thread()
#endif

namespace systems::leal::campello_renderer {

namespace cp = systems::leal::campello_procedural;
namespace gl = systems::leal::gltf;

namespace procedural {

namespace {

cp::Input toInput(const gl::ProceduralNodeInput& in) {
    cp::Input input;
    input.name = in.name;
    input.type = cp::valueTypeFromString(in.type).value_or(cp::ValueType::none);
    if (in.value) {
        cp::Value value;
        value.type = input.type;
        if (input.type == cp::ValueType::string || input.type == cp::ValueType::filename) {
            value.text = in.value->stringValue;
        } else if (input.type == cp::ValueType::boolean && in.value->values.empty()) {
            value.data[0] = in.value->boolValue ? 1.0f : 0.0f;
        } else {
            for (size_t i = 0; i < in.value->values.size() && i < value.data.size(); ++i) {
                value.data[i] = static_cast<float>(in.value->values[i]);
            }
        }
        input.value = value;
    }
    input.node = static_cast<int>(in.nodeIndex);
    input.output = in.nodeOutput;
    input.graphInput = in.graphInput;
    input.uiName = in.uiname;
    input.uiFolder = in.uifolder;
    input.doc = in.doc;
    return input;
}

}  // namespace

cp::Graph toGraph(const gl::ProceduralGraph& graph) {
    cp::Graph out;
    out.name = graph.name;
    out.type = cp::valueTypeFromString(graph.type).value_or(cp::ValueType::none);
    for (const auto& input : graph.inputs) out.inputs.push_back(toInput(input));
    for (const auto& output : graph.outputs) {
        out.outputs.push_back({output.name, cp::valueTypeFromString(output.type).value_or(cp::ValueType::none),
                               static_cast<int>(output.nodeIndex), output.nodeOutput});
    }
    for (const auto& node : graph.nodes) {
        cp::Node n;
        n.name = node.name;
        n.category = node.nodetype;
        n.type = cp::valueTypeFromString(node.type).value_or(cp::ValueType::none);
        for (const auto& input : node.inputs) n.inputs.push_back(toInput(input));
        for (const auto& output : node.outputs) {
            n.outputs.push_back({output.name, cp::valueTypeFromString(output.type).value_or(cp::ValueType::none), -1, ""});
        }
        out.nodes.push_back(std::move(n));
    }
    return out;
}

cp::ImageResolver imageResolver(const gl::GLTF& asset) {
    return [&asset](const cp::ImageRef& ref) -> std::shared_ptr<const cp::Image> {
        if (ref.filename.empty()) return nullptr;
        std::shared_ptr<systems::leal::campello_image::Image> img;
        if (asset.images) {
            for (const auto& gltfImage : *asset.images) {
                if (gltfImage.uri == ref.filename && !gltfImage.data.empty()) {
                    img = systems::leal::campello_image::Image::fromMemory(gltfImage.data.data(), gltfImage.data.size());
                    break;
                }
            }
        }
        if (!img) img = systems::leal::campello_image::Image::fromFile(ref.filename.c_str());
        if (!img) return nullptr;
        auto out = std::make_shared<cp::Image>();
        out->width = img->getWidth();
        out->height = img->getHeight();
        const size_t count = size_t(out->width) * out->height * 4;
        out->rgba.resize(count);
        using IF = systems::leal::campello_image::ImageFormat;
        switch (img->getFormat()) {
            case IF::rgba8: {
                const auto* src = static_cast<const uint8_t*>(img->getData());
                for (size_t i = 0; i < count; ++i) out->rgba[i] = float(src[i]) / 255.0f;
                break;
            }
            case IF::rgba32f:
                std::memcpy(out->rgba.data(), img->getData(), count * sizeof(float));
                break;
            default:
                return nullptr;  // rgba16f: not supported by image nodes yet
        }
        return out;
    };
}

bool canWaitForThreads() {
#if defined(__EMSCRIPTEN_PTHREADS__)
    return !emscripten_is_main_browser_thread();
#else
    return true;
#endif
}

}  // namespace procedural

std::vector<uint8_t> bakeProceduralTexture(const gl::GLTF& asset, const gl::ProceduralGraph& graph,
                                           const std::string& outputName, int width, int height) {
    cp::BakeOptions options;
    options.width = static_cast<uint32_t>(std::max(1, width));
    options.height = static_cast<uint32_t>(std::max(1, height));
    options.outputs = {outputName};
    options.images = procedural::imageResolver(asset);
    if (!procedural::canWaitForThreads()) options.threads = 1;
    const auto result = cp::bake(procedural::toGraph(graph), options);
    if (!result.ok() || result.images.empty()) {
        std::cerr << "[bakeProceduralTexture] graph \"" << graph.name << "\": " << result.error << std::endl;
        return std::vector<uint8_t>(size_t(options.width) * options.height * 4, 0);
    }
    return cp::toRGBA8(result.images[0], cp::Encoding::linear);
}

std::vector<uint8_t> bakeProceduralTextureGPU(const std::shared_ptr<systems::leal::campello_gpu::Device>&,
                                              const gl::GLTF& asset, const gl::ProceduralGraph& graph,
                                              const std::string& outputName, int width, int height) {
    return bakeProceduralTexture(asset, graph, outputName, width, height);
}

}  // namespace systems::leal::campello_renderer
