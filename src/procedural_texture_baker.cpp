#include <campello_renderer/procedural_texture_baker.hpp>
#include <algorithm>
#include <cmath>
#include <optional>
#include <campello_image/image.hpp>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include <campello_gpu/device.hpp>
#include <campello_gpu/buffer.hpp>
#include <campello_gpu/texture.hpp>
#include <campello_gpu/command_encoder.hpp>
#include <campello_gpu/compute_pass_encoder.hpp>
#include <campello_gpu/compute_pipeline.hpp>
#include <campello_gpu/pipeline_layout.hpp>
#include <campello_gpu/bind_group.hpp>
#include <campello_gpu/bind_group_layout.hpp>
#include <campello_gpu/constants/buffer_usage.hpp>
#include <campello_gpu/constants/texture_usage.hpp>
#include <campello_gpu/constants/pixel_format.hpp>
#include <campello_gpu/constants/shader_stage.hpp>
#include <campello_gpu/descriptors/compute_pipeline_descriptor.hpp>
#include <campello_gpu/descriptors/pipeline_layout_descriptor.hpp>
#include <campello_gpu/descriptors/bind_group_layout_descriptor.hpp>
#include <campello_gpu/descriptors/bind_group_descriptor.hpp>
#include <campello_gpu/shader_module.hpp>
#if defined(__APPLE__)
#include "shaders/metal_default.h"
#elif defined(__ANDROID__) || defined(__linux__)
#include "shaders/vulkan_default.h"
#elif defined(_WIN32)
#include "shaders/directx_default.h"
#endif

using namespace systems::leal::gltf;

namespace systems::leal::campello_renderer {

namespace {

struct TypedValue {
    std::string type;
    std::vector<float> data;
    std::string stringValue;

    static TypedValue makeFloat(float v) {
        TypedValue tv; tv.type = "float"; tv.data = {v}; return tv;
    }
    static TypedValue makeVec2(float x, float y) {
        TypedValue tv; tv.type = "vector2"; tv.data = {x, y}; return tv;
    }
    static TypedValue makeVec3(float x, float y, float z) {
        TypedValue tv; tv.type = "vector3"; tv.data = {x, y, z}; return tv;
    }
    static TypedValue makeVec4(float x, float y, float z, float w) {
        TypedValue tv; tv.type = "vector4"; tv.data = {x, y, z, w}; return tv;
    }
    static TypedValue makeColor3(float r, float g, float b) {
        TypedValue tv; tv.type = "color3"; tv.data = {r, g, b}; return tv;
    }
    static TypedValue makeColor4(float r, float g, float b, float a) {
        TypedValue tv; tv.type = "color4"; tv.data = {r, g, b, a}; return tv;
    }
    static TypedValue makeInteger(int v) {
        TypedValue tv; tv.type = "integer"; tv.data = {static_cast<float>(v)}; return tv;
    }

    float asFloat() const {
        return data.empty() ? 0.0f : data[0];
    }
};

static TypedValue valueFromProceduralValue(const ProceduralValue& pv) {
    if (pv.type == "float" && !pv.values.empty()) return TypedValue::makeFloat(pv.values[0]);
    if (pv.type == "integer" && !pv.values.empty()) return TypedValue::makeInteger(static_cast<int>(pv.values[0]));
    if (pv.type == "vector2" && pv.values.size() >= 2) return TypedValue::makeVec2(pv.values[0], pv.values[1]);
    if (pv.type == "vector3" && pv.values.size() >= 3) return TypedValue::makeVec3(pv.values[0], pv.values[1], pv.values[2]);
    if (pv.type == "vector4" && pv.values.size() >= 4) return TypedValue::makeVec4(pv.values[0], pv.values[1], pv.values[2], pv.values[3]);
    if (pv.type == "color3" && pv.values.size() >= 3) return TypedValue::makeColor3(pv.values[0], pv.values[1], pv.values[2]);
    if (pv.type == "color4" && pv.values.size() >= 4) return TypedValue::makeColor4(pv.values[0], pv.values[1], pv.values[2], pv.values[3]);
    if ((pv.type == "string" || pv.type == "filename") && !pv.stringValue.empty()) {
        TypedValue tv; tv.type = pv.type; tv.stringValue = pv.stringValue; return tv;
    }
    return TypedValue::makeFloat(0.0f);
}

// --- Noise helpers ---
static inline float fade(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

static inline float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

static const int kPerm[512] = {
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
    8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
    35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,
    134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
    55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,
    18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,
    250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,
    189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,
    172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,
    228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,
    107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
    138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
    8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
    35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,
    134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
    55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,
    18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,
    250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,
    189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,
    172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,
    228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,
    107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
    138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
};

static inline float grad(int hash, float x, float y) {
    int h = hash & 7;
    float u = h < 4 ? x : -x;
    float v = h < 2 || h == 5 || h == 6 ? y : -y;
    return u + v;
}

static float perlin2D(float x, float y) {
    int X = static_cast<int>(std::floor(x)) & 255;
    int Y = static_cast<int>(std::floor(y)) & 255;
    x -= std::floor(x);
    y -= std::floor(y);
    float u = fade(x);
    float v = fade(y);
    int A = kPerm[X] + Y;
    int B = kPerm[X + 1] + Y;
    return lerp(v, lerp(u, grad(kPerm[A], x, y), grad(kPerm[B], x - 1.0f, y)),
                   lerp(u, grad(kPerm[A + 1], x, y - 1.0f), grad(kPerm[B + 1], x - 1.0f, y - 1.0f)));
}

static float noise2D(float x, float y) {
    return (perlin2D(x, y) + 1.0f) * 0.5f;
}

static float fbm2D(float x, float y, int octaves, float lacunarity, float gain) {
    float total = 0.0f;
    float amplitude = 1.0f;
    float frequency = 1.0f;
    float maxValue = 0.0f;
    for (int i = 0; i < octaves; ++i) {
        total += noise2D(x * frequency, y * frequency) * amplitude;
        maxValue += amplitude;
        amplitude *= gain;
        frequency *= lacunarity;
    }
    return total / maxValue;
}

// --- Image helpers ---
struct LoadedImage {
    std::shared_ptr<systems::leal::campello_image::Image> image;
    std::vector<float> floatPixels;
    uint32_t width = 0;
    uint32_t height = 0;
};
struct ImageCache {
    std::unordered_map<std::string, LoadedImage> images;
    std::mutex mutex;
};

static LoadedImage loadImage(const std::string& filename,
                             const systems::leal::gltf::GLTF& asset,
                             ImageCache& cache) {
    {
        std::lock_guard<std::mutex> lock(cache.mutex);
        auto it = cache.images.find(filename);
        if (it != cache.images.end()) return it->second;
    }

    std::shared_ptr<systems::leal::campello_image::Image> img;

    // Try to find in glTF images by URI match
    if (asset.images) {
        for (const auto& gltfImg : *asset.images) {
            if (gltfImg.uri == filename && !gltfImg.data.empty()) {
                img = systems::leal::campello_image::Image::fromMemory(
                    gltfImg.data.data(), gltfImg.data.size());
                break;
            }
        }
    }

    // Fall back to filesystem
    if (!img) {
        img = systems::leal::campello_image::Image::fromFile(filename.c_str());
    }

    LoadedImage result;
    if (img) {
        result.image = img;
        result.width = img->getWidth();
        result.height = img->getHeight();
        auto fmt = img->getFormat();
        const void* data = img->getData();
        size_t pixelCount = static_cast<size_t>(result.width) * result.height;
        result.floatPixels.resize(pixelCount * 4);

        if (fmt == systems::leal::campello_image::ImageFormat::rgba8) {
            const uint8_t* src = static_cast<const uint8_t*>(data);
            for (size_t i = 0; i < pixelCount * 4; ++i) {
                result.floatPixels[i] = src[i] / 255.0f;
            }
        } else if (fmt == systems::leal::campello_image::ImageFormat::rgba32f) {
            const float* src = static_cast<const float*>(data);
            std::memcpy(result.floatPixels.data(), src, pixelCount * 4 * sizeof(float));
        } else if (fmt == systems::leal::campello_image::ImageFormat::rgba16f) {
            const uint16_t* src = static_cast<const uint16_t*>(data);
            for (size_t i = 0; i < pixelCount * 4; ++i) {
                result.floatPixels[i] = src[i] / 65535.0f;
            }
        }
    }

    {
        std::lock_guard<std::mutex> lock(cache.mutex);
        cache.images[filename] = result;
    }
    return result;
}

static TypedValue sampleImage(const LoadedImage& img, float u, float v) {
    if (!img.image || img.width == 0 || img.height == 0) {
        return TypedValue::makeColor4(0.0f, 0.0f, 0.0f, 1.0f);
    }

    u = std::fmod(u, 1.0f);
    v = std::fmod(v, 1.0f);
    if (u < 0.0f) u += 1.0f;
    if (v < 0.0f) v += 1.0f;

    float fx = u * (img.width - 1);
    float fy = v * (img.height - 1);
    int x0 = static_cast<int>(std::floor(fx));
    int y0 = static_cast<int>(std::floor(fy));
    int x1 = std::min(x0 + 1, static_cast<int>(img.width) - 1);
    int y1 = std::min(y0 + 1, static_cast<int>(img.height) - 1);
    float tX = fx - x0;
    float tY = fy - y0;

    size_t i00 = (static_cast<size_t>(y0) * img.width + x0) * 4;
    size_t i10 = (static_cast<size_t>(y0) * img.width + x1) * 4;
    size_t i01 = (static_cast<size_t>(y1) * img.width + x0) * 4;
    size_t i11 = (static_cast<size_t>(y1) * img.width + x1) * 4;

    float out[4];
    for (int i = 0; i < 4; ++i) {
        float a = img.floatPixels[i00 + i] + (img.floatPixels[i10 + i] - img.floatPixels[i00 + i]) * tX;
        float b = img.floatPixels[i01 + i] + (img.floatPixels[i11 + i] - img.floatPixels[i01 + i]) * tX;
        out[i] = a + (b - a) * tY;
    }
    return TypedValue::makeColor4(out[0], out[1], out[2], out[3]);
}

static TypedValue evaluateGraphInput(const ProceduralGraph& graph, const std::string& name) {
    for (const auto& input : graph.inputs) {
        if (input.name == name && input.value) {
            return valueFromProceduralValue(*input.value);
        }
    }
    return TypedValue::makeFloat(0.0f);
}

static const ProceduralNodeInput* findInput(const ProceduralNode& node, const std::string& name) {
    for (const auto& in : node.inputs) {
        if (in.name == name) return &in;
    }
    return nullptr;
}

static TypedValue evaluateNode(const ProceduralGraph& graph, int nodeIndex, float u, float v,
                                std::vector<std::optional<TypedValue>>& cache,
                                const systems::leal::gltf::GLTF& asset, ImageCache& imageCache);

static TypedValue evaluateNodeInput(const ProceduralGraph& graph, const ProceduralNodeInput& input,
                                     float u, float v, std::vector<std::optional<TypedValue>>& cache,
                                     const systems::leal::gltf::GLTF& asset, ImageCache& imageCache) {
    if (input.value) {
        return valueFromProceduralValue(*input.value);
    }
    if (input.nodeIndex >= 0) {
        // Multi-output nodes: nodeOutput field would select the specific output.
        // MVP: single-output assumption — ignore nodeOutput.
        return evaluateNode(graph, static_cast<int>(input.nodeIndex), u, v, cache, asset, imageCache);
    }
    if (!input.graphInput.empty()) {
        return evaluateGraphInput(graph, input.graphInput);
    }
    return TypedValue::makeFloat(0.0f);
}

static TypedValue evaluateNode(const ProceduralGraph& graph, int nodeIndex, float u, float v,
                                std::vector<std::optional<TypedValue>>& cache,
                                const systems::leal::gltf::GLTF& asset, ImageCache& imageCache) {
    if (nodeIndex < 0 || nodeIndex >= static_cast<int>(graph.nodes.size())) return TypedValue::makeFloat(0.0f);
    if (cache[nodeIndex].has_value()) return *cache[nodeIndex];

    const auto& node = graph.nodes[nodeIndex];
    TypedValue result;

    if (node.nodetype == "constant") {
        auto in = findInput(node, "value");
        result = in ? evaluateNodeInput(graph, *in, u, v, cache, asset, imageCache) : TypedValue::makeFloat(0.0f);
    }
    else if (node.nodetype == "texcoord") {
        auto in = findInput(node, "index");
        // TEXCOORD index is ignored for baking — we always provide the bake-space UVs.
        (void)in;
        result = TypedValue::makeVec2(u, v);
    }
    else if (node.nodetype == "add") {
        auto in1p = findInput(node, "in1");
        auto in2p = findInput(node, "in2");
        if (in1p && in2p) {
            auto v1 = evaluateNodeInput(graph, *in1p, u, v, cache, asset, imageCache);
            auto v2 = evaluateNodeInput(graph, *in2p, u, v, cache, asset, imageCache);
            result.type = v1.type;
            size_t n = std::min(v1.data.size(), v2.data.size());
            result.data.resize(n);
            for (size_t i = 0; i < n; ++i) result.data[i] = v1.data[i] + v2.data[i];
        } else {
            result = TypedValue::makeFloat(0.0f);
        }
    }
    else if (node.nodetype == "subtract") {
        auto in1p = findInput(node, "in1");
        auto in2p = findInput(node, "in2");
        if (in1p && in2p) {
            auto v1 = evaluateNodeInput(graph, *in1p, u, v, cache, asset, imageCache);
            auto v2 = evaluateNodeInput(graph, *in2p, u, v, cache, asset, imageCache);
            result.type = v1.type;
            size_t n = std::min(v1.data.size(), v2.data.size());
            result.data.resize(n);
            for (size_t i = 0; i < n; ++i) result.data[i] = v1.data[i] - v2.data[i];
        } else {
            result = TypedValue::makeFloat(0.0f);
        }
    }
    else if (node.nodetype == "multiply") {
        auto in1p = findInput(node, "in1");
        auto in2p = findInput(node, "in2");
        if (in1p && in2p) {
            auto v1 = evaluateNodeInput(graph, *in1p, u, v, cache, asset, imageCache);
            auto v2 = evaluateNodeInput(graph, *in2p, u, v, cache, asset, imageCache);
            result.type = v1.type;
            size_t n = std::min(v1.data.size(), v2.data.size());
            result.data.resize(n);
            for (size_t i = 0; i < n; ++i) result.data[i] = v1.data[i] * v2.data[i];
        } else {
            result = TypedValue::makeFloat(0.0f);
        }
    }
    else if (node.nodetype == "divide") {
        auto in1p = findInput(node, "in1");
        auto in2p = findInput(node, "in2");
        if (in1p && in2p) {
            auto v1 = evaluateNodeInput(graph, *in1p, u, v, cache, asset, imageCache);
            auto v2 = evaluateNodeInput(graph, *in2p, u, v, cache, asset, imageCache);
            result.type = v1.type;
            size_t n = std::min(v1.data.size(), v2.data.size());
            result.data.resize(n);
            for (size_t i = 0; i < n; ++i) {
                float div = v2.data[i];
                if (div == 0.0f) div = 1e-6f;
                result.data[i] = v1.data[i] / div;
            }
        } else {
            result = TypedValue::makeFloat(0.0f);
        }
    }
    else if (node.nodetype == "floor") {
        auto inp = findInput(node, "in");
        if (inp) {
            result = evaluateNodeInput(graph, *inp, u, v, cache, asset, imageCache);
            for (auto& v : result.data) v = std::floor(v);
        } else {
            result = TypedValue::makeFloat(0.0f);
        }
    }
    else if (node.nodetype == "modulo") {
        auto in1p = findInput(node, "in1");
        auto in2p = findInput(node, "in2");
        if (in1p && in2p) {
            auto v1 = evaluateNodeInput(graph, *in1p, u, v, cache, asset, imageCache);
            auto v2 = evaluateNodeInput(graph, *in2p, u, v, cache, asset, imageCache);
            result.type = v1.type;
            size_t n = std::min(v1.data.size(), v2.data.size());
            result.data.resize(n);
            for (size_t i = 0; i < n; ++i) {
                float div = v2.data[i];
                if (div == 0.0f) div = 1e-6f;
                result.data[i] = std::fmod(v1.data[i], div);
                if (result.data[i] < 0.0f) result.data[i] += std::abs(div);
            }
        } else {
            result = TypedValue::makeFloat(0.0f);
        }
    }
    else if (node.nodetype == "dotproduct") {
        auto in1p = findInput(node, "in1");
        auto in2p = findInput(node, "in2");
        if (in1p && in2p) {
            auto v1 = evaluateNodeInput(graph, *in1p, u, v, cache, asset, imageCache);
            auto v2 = evaluateNodeInput(graph, *in2p, u, v, cache, asset, imageCache);
            size_t n = std::min(v1.data.size(), v2.data.size());
            float sum = 0.0f;
            for (size_t i = 0; i < n; ++i) sum += v1.data[i] * v2.data[i];
            result = TypedValue::makeFloat(sum);
        } else {
            result = TypedValue::makeFloat(0.0f);
        }
    }
    else if (node.nodetype == "mix") {
        auto fgp = findInput(node, "fg");
        auto bgp = findInput(node, "bg");
        auto mixp = findInput(node, "mix");
        if (fgp && bgp && mixp) {
            auto fg = evaluateNodeInput(graph, *fgp, u, v, cache, asset, imageCache);
            auto bg = evaluateNodeInput(graph, *bgp, u, v, cache, asset, imageCache);
            float t = evaluateNodeInput(graph, *mixp, u, v, cache, asset, imageCache).asFloat();
            result.type = fg.type;
            size_t n = std::min(fg.data.size(), bg.data.size());
            result.data.resize(n);
            for (size_t i = 0; i < n; ++i) {
                result.data[i] = fg.data[i] * t + bg.data[i] * (1.0f - t);
            }
        } else {
            result = TypedValue::makeFloat(0.0f);
        }
    }
    else if (node.nodetype == "noise2d") {
        auto posp = findInput(node, "position");
        auto octp = findInput(node, "octaves");
        auto lacp = findInput(node, "lacunarity");
        auto gainp = findInput(node, "gain");
        auto scalep = findInput(node, "scale");
        float u_in = 0.0f, v_in = 0.0f;
        if (posp) {
            auto pos = evaluateNodeInput(graph, *posp, u, v, cache, asset, imageCache);
            if (pos.data.size() >= 2) { u_in = pos.data[0]; v_in = pos.data[1]; }
            else if (!pos.data.empty()) { u_in = v_in = pos.data[0]; }
        }
        int octaves = 1;
        if (octp) octaves = static_cast<int>(evaluateNodeInput(graph, *octp, u, v, cache, asset, imageCache).asFloat());
        if (octaves < 1) octaves = 1;
        if (octaves > 8) octaves = 8;
        float lacunarity = 2.0f;
        if (lacp) lacunarity = evaluateNodeInput(graph, *lacp, u, v, cache, asset, imageCache).asFloat();
        float gain = 0.5f;
        if (gainp) gain = evaluateNodeInput(graph, *gainp, u, v, cache, asset, imageCache).asFloat();
        float scale = 1.0f;
        if (scalep) scale = evaluateNodeInput(graph, *scalep, u, v, cache, asset, imageCache).asFloat();
        float val = fbm2D(u_in * scale, v_in * scale, octaves, lacunarity, gain);
        if (node.type == "color3" || node.type == "vector3") {
            result = TypedValue::makeColor3(val, val, val);
        } else if (node.type == "color4" || node.type == "vector4") {
            result = TypedValue::makeColor4(val, val, val, 1.0f);
        } else if (node.type == "vector2") {
            result = TypedValue::makeVec2(val, val);
        } else {
            result = TypedValue::makeFloat(val);
        }
    }
    else if (node.nodetype == "checkerboard") {
        auto texp = findInput(node, "texcoord");
        auto c1p = findInput(node, "color1");
        auto c2p = findInput(node, "color2");
        auto tilp = findInput(node, "uvtiling");
        float u_in = u, v_in = v;
        if (texp) {
            auto tc = evaluateNodeInput(graph, *texp, u, v, cache, asset, imageCache);
            if (tc.data.size() >= 2) { u_in = tc.data[0]; v_in = tc.data[1]; }
        }
        float tx = 1.0f, ty = 1.0f;
        if (tilp) {
            auto t = evaluateNodeInput(graph, *tilp, u, v, cache, asset, imageCache);
            if (t.data.size() >= 2) { tx = t.data[0]; ty = t.data[1]; }
            else if (!t.data.empty()) { tx = ty = t.data[0]; }
        }
        bool check = (static_cast<int>(std::floor(u_in * tx)) + static_cast<int>(std::floor(v_in * ty))) % 2 == 0;
        TypedValue v1, v2;
        if (node.type == "color3" || node.type == "vector3") {
            v1 = TypedValue::makeColor3(1.0f, 1.0f, 1.0f);
            v2 = TypedValue::makeColor3(0.0f, 0.0f, 0.0f);
        } else if (node.type == "color4" || node.type == "vector4") {
            v1 = TypedValue::makeColor4(1.0f, 1.0f, 1.0f, 1.0f);
            v2 = TypedValue::makeColor4(0.0f, 0.0f, 0.0f, 1.0f);
        } else if (node.type == "vector2") {
            v1 = TypedValue::makeVec2(1.0f, 1.0f);
            v2 = TypedValue::makeVec2(0.0f, 0.0f);
        } else {
            v1 = TypedValue::makeFloat(1.0f);
            v2 = TypedValue::makeFloat(0.0f);
        }
        if (c1p) v1 = evaluateNodeInput(graph, *c1p, u, v, cache, asset, imageCache);
        if (c2p) v2 = evaluateNodeInput(graph, *c2p, u, v, cache, asset, imageCache);
        result.type = v1.type;
        size_t n = std::min(v1.data.size(), v2.data.size());
        result.data.resize(n);
        if (check) {
            for (size_t i = 0; i < n; ++i) result.data[i] = v1.data[i];
        } else {
            for (size_t i = 0; i < n; ++i) result.data[i] = v2.data[i];
        }
    }
    else if (node.nodetype == "place2d") {
        auto texp = findInput(node, "texcoord");
        auto offp = findInput(node, "offset");
        auto scalep = findInput(node, "scale");
        auto rotp = findInput(node, "rotate");
        auto pivp = findInput(node, "pivot");
        float u_in = u, v_in = v;
        if (texp) {
            auto tc = evaluateNodeInput(graph, *texp, u, v, cache, asset, imageCache);
            if (tc.data.size() >= 2) { u_in = tc.data[0]; v_in = tc.data[1]; }
        }
        float ox = 0.0f, oy = 0.0f;
        if (offp) {
            auto o = evaluateNodeInput(graph, *offp, u, v, cache, asset, imageCache);
            if (o.data.size() >= 2) { ox = o.data[0]; oy = o.data[1]; }
            else if (!o.data.empty()) { ox = oy = o.data[0]; }
        }
        float sx = 1.0f, sy = 1.0f;
        if (scalep) {
            auto s = evaluateNodeInput(graph, *scalep, u, v, cache, asset, imageCache);
            if (s.data.size() >= 2) { sx = s.data[0]; sy = s.data[1]; }
            else if (!s.data.empty()) { sx = sy = s.data[0]; }
        }
        float rot = 0.0f;
        if (rotp) rot = evaluateNodeInput(graph, *rotp, u, v, cache, asset, imageCache).asFloat();
        float px = 0.5f, py = 0.5f;
        if (pivp) {
            auto p = evaluateNodeInput(graph, *pivp, u, v, cache, asset, imageCache);
            if (p.data.size() >= 2) { px = p.data[0]; py = p.data[1]; }
            else if (!p.data.empty()) { px = py = p.data[0]; }
        }
        float x = u_in - px;
        float y = v_in - py;
        x *= sx;
        y *= sy;
        if (rot != 0.0f) {
            float rad = rot * 3.14159265f / 180.0f;
            float c = std::cos(rad);
            float s = std::sin(rad);
            float rx = x * c - y * s;
            float ry = x * s + y * c;
            x = rx;
            y = ry;
        }
        x += px + ox;
        y += py + oy;
        result = TypedValue::makeVec2(x, y);
    }
    else if (node.nodetype == "sin") {
        auto inp = findInput(node, "in");
        if (inp) {
            result = evaluateNodeInput(graph, *inp, u, v, cache, asset, imageCache);
            for (auto& val : result.data) val = std::sin(val);
        } else {
            result = TypedValue::makeFloat(0.0f);
        }
    }
    else if (node.nodetype == "cos") {
        auto inp = findInput(node, "in");
        if (inp) {
            result = evaluateNodeInput(graph, *inp, u, v, cache, asset, imageCache);
            for (auto& val : result.data) val = std::cos(val);
        } else {
            result = TypedValue::makeFloat(1.0f);
        }
    }
    else if (node.nodetype == "abs") {
        auto inp = findInput(node, "in");
        if (inp) {
            result = evaluateNodeInput(graph, *inp, u, v, cache, asset, imageCache);
            for (auto& val : result.data) val = std::abs(val);
        } else {
            result = TypedValue::makeFloat(0.0f);
        }
    }
    else if (node.nodetype == "clamp") {
        auto inp = findInput(node, "in");
        auto lowp = findInput(node, "low");
        auto highp = findInput(node, "high");
        if (inp) {
            result = evaluateNodeInput(graph, *inp, u, v, cache, asset, imageCache);
            float low = lowp ? evaluateNodeInput(graph, *lowp, u, v, cache, asset, imageCache).asFloat() : 0.0f;
            float high = highp ? evaluateNodeInput(graph, *highp, u, v, cache, asset, imageCache).asFloat() : 1.0f;
            for (auto& val : result.data) {
                val = val < low ? low : (val > high ? high : val);
            }
        } else {
            result = TypedValue::makeFloat(0.0f);
        }
    }
    else if (node.nodetype == "power") {
        auto in1p = findInput(node, "in1");
        auto in2p = findInput(node, "in2");
        if (in1p && in2p) {
            auto v1 = evaluateNodeInput(graph, *in1p, u, v, cache, asset, imageCache);
            auto v2 = evaluateNodeInput(graph, *in2p, u, v, cache, asset, imageCache);
            result.type = v1.type;
            size_t n = std::min(v1.data.size(), v2.data.size());
            result.data.resize(n);
            for (size_t i = 0; i < n; ++i) result.data[i] = std::pow(v1.data[i], v2.data[i]);
        } else {
            result = TypedValue::makeFloat(0.0f);
        }
    }
    else if (node.nodetype == "sqrt") {
        auto inp = findInput(node, "in");
        if (inp) {
            result = evaluateNodeInput(graph, *inp, u, v, cache, asset, imageCache);
            for (auto& val : result.data) val = std::sqrt(std::max(val, 0.0f));
        } else {
            result = TypedValue::makeFloat(0.0f);
        }
    }
    else if (node.nodetype == "max") {
        auto in1p = findInput(node, "in1");
        auto in2p = findInput(node, "in2");
        if (in1p && in2p) {
            auto v1 = evaluateNodeInput(graph, *in1p, u, v, cache, asset, imageCache);
            auto v2 = evaluateNodeInput(graph, *in2p, u, v, cache, asset, imageCache);
            result.type = v1.type;
            size_t n = std::min(v1.data.size(), v2.data.size());
            result.data.resize(n);
            for (size_t i = 0; i < n; ++i) result.data[i] = std::max(v1.data[i], v2.data[i]);
        } else {
            result = TypedValue::makeFloat(0.0f);
        }
    }
    else if (node.nodetype == "min") {
        auto in1p = findInput(node, "in1");
        auto in2p = findInput(node, "in2");
        if (in1p && in2p) {
            auto v1 = evaluateNodeInput(graph, *in1p, u, v, cache, asset, imageCache);
            auto v2 = evaluateNodeInput(graph, *in2p, u, v, cache, asset, imageCache);
            result.type = v1.type;
            size_t n = std::min(v1.data.size(), v2.data.size());
            result.data.resize(n);
            for (size_t i = 0; i < n; ++i) result.data[i] = std::min(v1.data[i], v2.data[i]);
        } else {
            result = TypedValue::makeFloat(0.0f);
        }
    }
    else if (node.nodetype == "image") {
        auto filep = findInput(node, "file");
        auto texp = findInput(node, "texcoord");
        auto defp = findInput(node, "default");
        float u_in = u, v_in = v;
        if (texp) {
            auto tc = evaluateNodeInput(graph, *texp, u, v, cache, asset, imageCache);
            if (tc.data.size() >= 2) { u_in = tc.data[0]; v_in = tc.data[1]; }
        }
        std::string filename;
        if (filep) {
            auto fileVal = evaluateNodeInput(graph, *filep, u, v, cache, asset, imageCache);
            if (!fileVal.stringValue.empty()) {
                filename = fileVal.stringValue;
            } else if (filep->value) {
                filename = filep->value->stringValue;
            }
        }
        if (!filename.empty()) {
            auto loaded = loadImage(filename, asset, imageCache);
            if (loaded.image) {
                result = sampleImage(loaded, u_in, v_in);
            }
        }
        if (result.data.empty()) {
            if (defp) {
                result = evaluateNodeInput(graph, *defp, u, v, cache, asset, imageCache);
            } else {
                result = TypedValue::makeColor4(0.0f, 0.0f, 0.0f, 1.0f);
            }
        }
    }
    else if (node.nodetype == "swizzle") {
        auto inp = findInput(node, "in");
        auto chp = findInput(node, "channels");
        std::string channels;
        if (chp) {
            auto chVal = evaluateNodeInput(graph, *chp, u, v, cache, asset, imageCache);
            if (!chVal.stringValue.empty()) channels = chVal.stringValue;
        }
        int outSize = 4;
        if (node.type == "float") outSize = 1;
        else if (node.type == "vector2" || node.type == "color2") outSize = 2;
        else if (node.type == "vector3" || node.type == "color3") outSize = 3;
        else if (node.type == "vector4" || node.type == "color4") outSize = 4;
        if (channels.empty()) {
            channels = (outSize == 1) ? "x" : (outSize == 2) ? "xy" : (outSize == 3) ? "xyz" : "xyzw";
        }
        TypedValue inVal;
        if (inp) inVal = evaluateNodeInput(graph, *inp, u, v, cache, asset, imageCache);
        std::vector<float> outData(outSize, 0.0f);
        for (int i = 0; i < outSize; ++i) {
            char c = (i < static_cast<int>(channels.size())) ? channels[i] : ('x' + i);
            int idx = -1;
            if (c == 'x' || c == 'r' || c == '0') idx = 0;
            else if (c == 'y' || c == 'g' || c == '1') idx = 1;
            else if (c == 'z' || c == 'b' || c == '2') idx = 2;
            else if (c == 'w' || c == 'a' || c == '3') idx = 3;
            if (idx >= 0 && idx < static_cast<int>(inVal.data.size())) {
                outData[i] = inVal.data[idx];
            } else if (i < static_cast<int>(inVal.data.size())) {
                outData[i] = inVal.data[i];
            }
        }
        if (node.type == "float") result = TypedValue::makeFloat(outData[0]);
        else if (node.type == "vector2") result = TypedValue::makeVec2(outData[0], outData[1]);
        else if (node.type == "vector3") result = TypedValue::makeVec3(outData[0], outData[1], outData[2]);
        else if (node.type == "vector4") result = TypedValue::makeVec4(outData[0], outData[1], outData[2], outData[3]);
        else if (node.type == "color3") result = TypedValue::makeColor3(outData[0], outData[1], outData[2]);
        else if (node.type == "color4") result = TypedValue::makeColor4(outData[0], outData[1], outData[2], outData[3]);
        else result = TypedValue::makeFloat(outData[0]);
    }
    else if (node.nodetype == "combine") {
        auto in1p = findInput(node, "in1");
        auto in2p = findInput(node, "in2");
        auto in3p = findInput(node, "in3");
        auto in4p = findInput(node, "in4");
        float vals[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (in1p) vals[0] = evaluateNodeInput(graph, *in1p, u, v, cache, asset, imageCache).asFloat();
        if (in2p) vals[1] = evaluateNodeInput(graph, *in2p, u, v, cache, asset, imageCache).asFloat();
        if (in3p) vals[2] = evaluateNodeInput(graph, *in3p, u, v, cache, asset, imageCache).asFloat();
        if (in4p) vals[3] = evaluateNodeInput(graph, *in4p, u, v, cache, asset, imageCache).asFloat();
        if (node.type == "vector2") result = TypedValue::makeVec2(vals[0], vals[1]);
        else if (node.type == "vector3") result = TypedValue::makeVec3(vals[0], vals[1], vals[2]);
        else if (node.type == "vector4") result = TypedValue::makeVec4(vals[0], vals[1], vals[2], vals[3]);
        else if (node.type == "color3") result = TypedValue::makeColor3(vals[0], vals[1], vals[2]);
        else if (node.type == "color4") result = TypedValue::makeColor4(vals[0], vals[1], vals[2], vals[3]);
        else result = TypedValue::makeVec4(vals[0], vals[1], vals[2], vals[3]);
    }
    else if (node.nodetype == "extract") {
        auto inp = findInput(node, "in");
        auto idxp = findInput(node, "index");
        int idx = 0;
        if (idxp) idx = static_cast<int>(evaluateNodeInput(graph, *idxp, u, v, cache, asset, imageCache).asFloat());
        float val = 0.0f;
        if (inp) {
            auto inVal = evaluateNodeInput(graph, *inp, u, v, cache, asset, imageCache);
            if (idx >= 0 && idx < static_cast<int>(inVal.data.size())) val = inVal.data[idx];
        }
        result = TypedValue::makeFloat(val);
    }
    else if (node.nodetype == "ifgreater") {
        auto v1p = findInput(node, "value1");
        auto v2p = findInput(node, "value2");
        auto in1p = findInput(node, "in1");
        auto in2p = findInput(node, "in2");
        float val1 = v1p ? evaluateNodeInput(graph, *v1p, u, v, cache, asset, imageCache).asFloat() : 0.0f;
        float val2 = v2p ? evaluateNodeInput(graph, *v2p, u, v, cache, asset, imageCache).asFloat() : 0.0f;
        if (val1 > val2) {
            result = in1p ? evaluateNodeInput(graph, *in1p, u, v, cache, asset, imageCache) : TypedValue::makeFloat(0.0f);
        } else {
            result = in2p ? evaluateNodeInput(graph, *in2p, u, v, cache, asset, imageCache) : TypedValue::makeFloat(0.0f);
        }
    }
    else if (node.nodetype == "ifequal") {
        auto v1p = findInput(node, "value1");
        auto v2p = findInput(node, "value2");
        auto in1p = findInput(node, "in1");
        auto in2p = findInput(node, "in2");
        float val1 = v1p ? evaluateNodeInput(graph, *v1p, u, v, cache, asset, imageCache).asFloat() : 0.0f;
        float val2 = v2p ? evaluateNodeInput(graph, *v2p, u, v, cache, asset, imageCache).asFloat() : 0.0f;
        if (val1 == val2) {
            result = in1p ? evaluateNodeInput(graph, *in1p, u, v, cache, asset, imageCache) : TypedValue::makeFloat(0.0f);
        } else {
            result = in2p ? evaluateNodeInput(graph, *in2p, u, v, cache, asset, imageCache) : TypedValue::makeFloat(0.0f);
        }
    }
    else if (node.nodetype == "length") {
        auto inp = findInput(node, "in");
        if (inp) {
            auto inVal = evaluateNodeInput(graph, *inp, u, v, cache, asset, imageCache);
            float sum = 0.0f;
            for (auto& val : inVal.data) sum += val * val;
            result = TypedValue::makeFloat(std::sqrt(sum));
        } else {
            result = TypedValue::makeFloat(0.0f);
        }
    }
    else if (node.nodetype == "distance") {
        auto in1p = findInput(node, "in1");
        auto in2p = findInput(node, "in2");
        if (in1p && in2p) {
            auto v1 = evaluateNodeInput(graph, *in1p, u, v, cache, asset, imageCache);
            auto v2 = evaluateNodeInput(graph, *in2p, u, v, cache, asset, imageCache);
            size_t n = std::min(v1.data.size(), v2.data.size());
            float sum = 0.0f;
            for (size_t i = 0; i < n; ++i) {
                float d = v1.data[i] - v2.data[i];
                sum += d * d;
            }
            result = TypedValue::makeFloat(std::sqrt(sum));
        } else {
            result = TypedValue::makeFloat(0.0f);
        }
    }
    else if (node.nodetype == "crossproduct") {
        auto in1p = findInput(node, "in1");
        auto in2p = findInput(node, "in2");
        if (in1p && in2p) {
            auto v1 = evaluateNodeInput(graph, *in1p, u, v, cache, asset, imageCache);
            auto v2 = evaluateNodeInput(graph, *in2p, u, v, cache, asset, imageCache);
            float x1 = v1.data.size() > 0 ? v1.data[0] : 0.0f;
            float y1 = v1.data.size() > 1 ? v1.data[1] : 0.0f;
            float z1 = v1.data.size() > 2 ? v1.data[2] : 0.0f;
            float x2 = v2.data.size() > 0 ? v2.data[0] : 0.0f;
            float y2 = v2.data.size() > 1 ? v2.data[1] : 0.0f;
            float z2 = v2.data.size() > 2 ? v2.data[2] : 0.0f;
            result = TypedValue::makeVec3(
                y1 * z2 - z1 * y2,
                z1 * x2 - x1 * z2,
                x1 * y2 - y1 * x2);
        } else {
            result = TypedValue::makeFloat(0.0f);
        }
    }
    else if (node.nodetype == "normalize") {
        auto inp = findInput(node, "in");
        if (inp) {
            auto inVal = evaluateNodeInput(graph, *inp, u, v, cache, asset, imageCache);
            float sum = 0.0f;
            for (auto& val : inVal.data) sum += val * val;
            float len = std::sqrt(sum);
            if (len == 0.0f) len = 1.0f;
            result.type = inVal.type;
            result.data.resize(inVal.data.size());
            for (size_t i = 0; i < inVal.data.size(); ++i) result.data[i] = inVal.data[i] / len;
        } else {
            result = TypedValue::makeFloat(0.0f);
        }
    }
    else {
        // Unsupported node type — return magenta for visibility during development.
        result = TypedValue::makeColor4(1.0f, 0.0f, 1.0f, 1.0f);
    }

    cache[nodeIndex] = result;
    return result;
}

} // anonymous namespace

std::vector<uint8_t> bakeProceduralTexture(
    const systems::leal::gltf::GLTF& asset,
    const systems::leal::gltf::ProceduralGraph& graph,
    const std::string& outputName,
    int width, int height)
{
    std::vector<uint8_t> pixels(width * height * 4, 0);
    ImageCache imageCache;

    // Find target output once
    int targetNodeIndex = -1;
    for (const auto& out : graph.outputs) {
        if (out.name == outputName) {
            targetNodeIndex = static_cast<int>(out.nodeIndex);
            break;
        }
    }

    auto bakeRows = [&](int yStart, int yEnd) {
        for (int y = yStart; y < yEnd; ++y) {
            for (int x = 0; x < width; ++x) {
                float u = (x + 0.5f) / width;
                float v = (y + 0.5f) / height;
                std::vector<std::optional<TypedValue>> cache(graph.nodes.size());

                TypedValue finalValue = TypedValue::makeColor4(0.0f, 0.0f, 0.0f, 1.0f);
                if (targetNodeIndex >= 0) {
                    finalValue = evaluateNode(graph, targetNodeIndex, u, v, cache, asset, imageCache);
                }

                uint8_t r = 0, g = 0, b = 0, a = 255;
                if (!finalValue.data.empty()) {
                    r = static_cast<uint8_t>(std::clamp(finalValue.data[0] * 255.0f, 0.0f, 255.0f));
                    if (finalValue.data.size() > 1) g = static_cast<uint8_t>(std::clamp(finalValue.data[1] * 255.0f, 0.0f, 255.0f));
                    if (finalValue.data.size() > 2) b = static_cast<uint8_t>(std::clamp(finalValue.data[2] * 255.0f, 0.0f, 255.0f));
                    if (finalValue.data.size() > 3) a = static_cast<uint8_t>(std::clamp(finalValue.data[3] * 255.0f, 0.0f, 255.0f));
                }

                int idx = (y * width + x) * 4;
                pixels[idx + 0] = r;
                pixels[idx + 1] = g;
                pixels[idx + 2] = b;
                pixels[idx + 3] = a;
            }
        }
    };

    int pixelCount = width * height;
    if (pixelCount < 256) {
        // Small bakes: single-threaded to avoid thread overhead
        bakeRows(0, height);
    } else {
        int numThreads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
        std::vector<std::thread> threads;
        threads.reserve(numThreads);
        for (int t = 0; t < numThreads; ++t) {
            int yStart = t * height / numThreads;
            int yEnd = (t + 1) * height / numThreads;
            if (yStart < yEnd) {
                threads.emplace_back(bakeRows, yStart, yEnd);
            }
        }
        for (auto& t : threads) {
            t.join();
        }
    }

    return pixels;
}

// ============================================================================
// GPU uber-shader baker
// ============================================================================

namespace {

enum ProceduralOp : uint32_t {
    OP_CONSTANT = 0,
    OP_TEXCOORD = 1,
    OP_ADD = 2,
    OP_SUBTRACT = 3,
    OP_MULTIPLY = 4,
    OP_DIVIDE = 5,
    OP_FLOOR = 6,
    OP_MODULO = 7,
    OP_ABS = 8,
    OP_CLAMP = 9,
    OP_POWER = 10,
    OP_SQRT = 11,
    OP_MAX = 12,
    OP_MIN = 13,
    OP_SIN = 14,
    OP_COS = 15,
    OP_DOTPRODUCT = 16,
    OP_CROSSPRODUCT = 17,
    OP_LENGTH = 18,
    OP_DISTANCE = 19,
    OP_NORMALIZE = 20,
    OP_MIX = 21,
    OP_NOISE2D = 22,
    OP_CHECKERBOARD = 23,
    OP_PLACE2D = 24,
    OP_SWIZZLE = 25,
    OP_COMBINE = 26,
    OP_EXTRACT = 27,
    OP_IFGREATER = 28,
    OP_IFEQUAL = 29,
};

// Presence flags occupy bits 8-11 so they don't collide with swizzle
// (bits 0-7), extract (bits 0-1), or combine count (bits 0-3).
static const uint32_t FLAG_HAS_A = 0x0100;
static const uint32_t FLAG_HAS_B = 0x0200;
static const uint32_t FLAG_HAS_C = 0x0400;
static const uint32_t FLAG_HAS_D = 0x0800;

// Must match ProceduralNode in Metal exactly (48 bytes).
// float4 value is placed first so both C++ and Metal agree on layout
// (Metal aligns float4 to 16 bytes, which would insert padding after
// six uint32_t fields if value were later in the struct).
struct ProceduralNodeGPU {
    float    value[4];
    uint32_t op;
    uint32_t outReg;
    uint32_t inA;
    uint32_t inB;
    uint32_t inC;
    uint32_t inD;
    uint32_t flags;
    uint32_t pad;
};

// Must match ProceduralBakeUniforms in Metal exactly.
struct ProceduralBakeUniforms {
    uint32_t width;
    uint32_t height;
    uint32_t nodeCount;
    uint32_t outputReg;
    uint32_t outputComponents;
    uint32_t pad[3];
};

static bool isNodeGPUSupported(const std::string& nodetype) {
    static const std::unordered_set<std::string> supported = {
        "constant", "texcoord", "add", "subtract", "multiply", "divide",
        "floor", "modulo", "abs", "clamp", "power", "sqrt", "max", "min",
        "sin", "cos", "dotproduct", "crossproduct", "length", "distance",
        "normalize", "mix", "noise2d", "checkerboard", "place2d",
        "swizzle", "combine", "extract", "ifgreater", "ifequal"
    };
    return supported.count(nodetype) > 0;
}

static float getFloatInput(const ProceduralNodeInput* in) {
    if (!in || !in->value || in->value->values.empty()) return 0.0f;
    return in->value->values[0];
}

static int getIntInput(const ProceduralNodeInput* in) {
    if (!in || !in->value || in->value->values.empty()) return 0;
    return static_cast<int>(in->value->values[0]);
}

static std::string getStringInput(const ProceduralNodeInput* in) {
    if (!in || !in->value) return "";
    return in->value->stringValue;
}

static const ProceduralNodeInput* findNodeInput(const ProceduralNode& node, const std::string& name) {
    for (const auto& in : node.inputs) {
        if (in.name == name) return &in;
    }
    return nullptr;
}

static uint32_t componentCountFromType(const std::string& type) {
    if (type == "float" || type == "integer") return 1;
    if (type == "vector2") return 2;
    if (type == "vector3" || type == "color3") return 3;
    if (type == "vector4" || type == "color4") return 4;
    return 4;
}

// ---------------------------------------------------------------------------
// Graph flattening: topological sort + register allocation + bytecode encode.
// ---------------------------------------------------------------------------
static std::optional<std::vector<ProceduralNodeGPU>> flattenGraph(
    const ProceduralGraph& graph,
    const std::string& outputName,
    uint32_t& outOutputReg,
    uint32_t& outOutputComponents)
{
    // Find the output node.
    int outputNodeIndex = -1;
    for (const auto& out : graph.outputs) {
        if (out.name == outputName) {
            outputNodeIndex = static_cast<int>(out.nodeIndex);
            break;
        }
    }
    if (outputNodeIndex < 0 || outputNodeIndex >= (int)graph.nodes.size())
        return std::nullopt;

    // Check all nodes are GPU-supported.
    for (const auto& node : graph.nodes) {
        if (!isNodeGPUSupported(node.nodetype)) return std::nullopt;
    }

    // Build dependency graph.
    size_t N = graph.nodes.size();
    std::vector<std::vector<int>> dependents(N);
    std::vector<int> inDegree(N, 0);
    for (int i = 0; i < (int)N; ++i) {
        for (const auto& in : graph.nodes[i].inputs) {
            if (in.nodeIndex >= 0 && in.nodeIndex < (int)N) {
                dependents[in.nodeIndex].push_back(i);
                inDegree[i]++;
            }
        }
    }

    // Kahn's topological sort.
    std::vector<int> order;
    order.reserve(N);
    std::vector<int> queue;
    for (int i = 0; i < (int)N; ++i) {
        if (inDegree[i] == 0) queue.push_back(i);
    }
    while (!queue.empty()) {
        int u = queue.back();
        queue.pop_back();
        order.push_back(u);
        for (int v : dependents[u]) {
            if (--inDegree[v] == 0) queue.push_back(v);
        }
    }
    if (order.size() != N) return std::nullopt; // cycle detected

    // Map original node index -> register index.
    std::vector<int> nodeToReg(N, -1);
    uint32_t nextReg = 0;

    std::vector<ProceduralNodeGPU> bytecode;
    bytecode.reserve(N * 2);

    auto addConstantNode = [&](const float* vals, uint32_t count) -> uint32_t {
        ProceduralNodeGPU node{};
        node.op = OP_CONSTANT;
        node.outReg = nextReg++;
        for (uint32_t i = 0; i < count && i < 4; ++i) node.value[i] = vals[i];
        bytecode.push_back(node);
        return node.outReg;
    };

    auto getInputReg = [&](const ProceduralNodeInput* in) -> std::pair<uint32_t, bool> {
        if (!in) return {0, false};
        if (in->nodeIndex >= 0 && in->nodeIndex < (int)N) {
            int reg = nodeToReg[in->nodeIndex];
            if (reg < 0) return {0, false};
            return {static_cast<uint32_t>(reg), true};
        }
        if (in->value && !in->value->values.empty()) {
            float v[4] = {0,0,0,0};
            for (size_t i = 0; i < in->value->values.size() && i < 4; ++i) v[i] = in->value->values[i];
            uint32_t reg = addConstantNode(v, static_cast<uint32_t>(in->value->values.size()));
            return {reg, true};
        }
        return {0, false};
    };

    for (int nodeIdx : order) {
        const ProceduralNode& node = graph.nodes[nodeIdx];
        ProceduralNodeGPU gpuNode{};
        gpuNode.op = OP_CONSTANT; // placeholder

        auto mapOp = [&]() {
            const std::string& t = node.nodetype;
            if (t == "constant") return OP_CONSTANT;
            if (t == "texcoord") return OP_TEXCOORD;
            if (t == "add") return OP_ADD;
            if (t == "subtract") return OP_SUBTRACT;
            if (t == "multiply") return OP_MULTIPLY;
            if (t == "divide") return OP_DIVIDE;
            if (t == "floor") return OP_FLOOR;
            if (t == "modulo") return OP_MODULO;
            if (t == "abs") return OP_ABS;
            if (t == "clamp") return OP_CLAMP;
            if (t == "power") return OP_POWER;
            if (t == "sqrt") return OP_SQRT;
            if (t == "max") return OP_MAX;
            if (t == "min") return OP_MIN;
            if (t == "sin") return OP_SIN;
            if (t == "cos") return OP_COS;
            if (t == "dotproduct") return OP_DOTPRODUCT;
            if (t == "crossproduct") return OP_CROSSPRODUCT;
            if (t == "length") return OP_LENGTH;
            if (t == "distance") return OP_DISTANCE;
            if (t == "normalize") return OP_NORMALIZE;
            if (t == "mix") return OP_MIX;
            if (t == "noise2d") return OP_NOISE2D;
            if (t == "checkerboard") return OP_CHECKERBOARD;
            if (t == "place2d") return OP_PLACE2D;
            if (t == "swizzle") return OP_SWIZZLE;
            if (t == "combine") return OP_COMBINE;
            if (t == "extract") return OP_EXTRACT;
            if (t == "ifgreater") return OP_IFGREATER;
            if (t == "ifequal") return OP_IFEQUAL;
            return OP_CONSTANT;
        };

        gpuNode.op = static_cast<uint32_t>(mapOp());

        // Handle constant inline value for OP_CONSTANT.
        if (gpuNode.op == OP_CONSTANT) {
            auto valIn = findNodeInput(node, "value");
            if (valIn && valIn->value) {
                for (size_t i = 0; i < valIn->value->values.size() && i < 4; ++i)
                    gpuNode.value[i] = valIn->value->values[i];
            }
        }

        // Handle generic inputs.
        auto wireInput = [&](const std::string& name) -> uint32_t {
            auto in = findNodeInput(node, name);
            auto [reg, ok] = getInputReg(in);
            if (ok) return reg;
            return 0;
        };

        auto hasInput = [&](const std::string& name) -> bool {
            auto in = findNodeInput(node, name);
            if (!in) return false;
            return (in->nodeIndex >= 0) || (in->value && !in->value->values.empty());
        };

        uint32_t flagMask = 0;

        if (gpuNode.op == OP_CONSTANT) {
            // already handled
        }
        else if (gpuNode.op == OP_TEXCOORD) {
            // no inputs
        }
        else if (gpuNode.op == OP_ADD || gpuNode.op == OP_SUBTRACT || gpuNode.op == OP_MULTIPLY ||
                 gpuNode.op == OP_DIVIDE || gpuNode.op == OP_MAX || gpuNode.op == OP_MIN ||
                 gpuNode.op == OP_DOTPRODUCT || gpuNode.op == OP_CROSSPRODUCT ||
                 gpuNode.op == OP_DISTANCE) {
            gpuNode.inA = wireInput("in1");
            gpuNode.inB = wireInput("in2");
            flagMask |= FLAG_HAS_A | FLAG_HAS_B;
        }
        else if (gpuNode.op == OP_FLOOR || gpuNode.op == OP_ABS || gpuNode.op == OP_SQRT ||
                 gpuNode.op == OP_SIN || gpuNode.op == OP_COS || gpuNode.op == OP_LENGTH ||
                 gpuNode.op == OP_NORMALIZE) {
            gpuNode.inA = wireInput("in");
            flagMask |= FLAG_HAS_A;
        }
        else if (gpuNode.op == OP_CLAMP) {
            gpuNode.inA = wireInput("in");
            gpuNode.inB = wireInput("min");
            gpuNode.inC = wireInput("max");
            flagMask |= FLAG_HAS_A | FLAG_HAS_B | FLAG_HAS_C;
        }
        else if (gpuNode.op == OP_POWER) {
            gpuNode.inA = wireInput("in1");
            gpuNode.inB = wireInput("in2");
            flagMask |= FLAG_HAS_A | FLAG_HAS_B;
        }
        else if (gpuNode.op == OP_MIX) {
            gpuNode.inA = wireInput("fg");
            gpuNode.inB = wireInput("bg");
            gpuNode.inC = wireInput("mix");
            flagMask |= FLAG_HAS_A | FLAG_HAS_B | FLAG_HAS_C;
        }
        else if (gpuNode.op == OP_NOISE2D) {
            if (hasInput("position")) {
                gpuNode.inA = wireInput("position");
                flagMask |= FLAG_HAS_A;
            }
            gpuNode.value[0] = getFloatInput(findNodeInput(node, "octaves"));
            if (gpuNode.value[0] < 1.0f) gpuNode.value[0] = 1.0f;
            if (gpuNode.value[0] > 8.0f) gpuNode.value[0] = 8.0f;
            gpuNode.value[1] = getFloatInput(findNodeInput(node, "lacunarity"));
            if (gpuNode.value[1] == 0.0f) gpuNode.value[1] = 2.0f;
            gpuNode.value[2] = getFloatInput(findNodeInput(node, "gain"));
            if (gpuNode.value[2] == 0.0f) gpuNode.value[2] = 0.5f;
            gpuNode.value[3] = getFloatInput(findNodeInput(node, "scale"));
            if (gpuNode.value[3] == 0.0f) gpuNode.value[3] = 1.0f;
        }
        else if (gpuNode.op == OP_CHECKERBOARD) {
            if (hasInput("texcoord")) {
                gpuNode.inA = wireInput("texcoord");
                flagMask |= FLAG_HAS_A;
            }
            if (hasInput("color1")) {
                gpuNode.inB = wireInput("color1");
                flagMask |= FLAG_HAS_B;
            } else {
                // Default white based on node type.
                float w[4] = {1,1,1,1};
                gpuNode.inB = addConstantNode(w, 4);
                flagMask |= FLAG_HAS_B;
            }
            if (hasInput("color2")) {
                gpuNode.inC = wireInput("color2");
                flagMask |= FLAG_HAS_C;
            } else {
                // Default black.
                float k[4] = {0,0,0,1};
                gpuNode.inC = addConstantNode(k, 4);
                flagMask |= FLAG_HAS_C;
            }
            auto tilIn = findNodeInput(node, "uvtiling");
            if (tilIn && tilIn->value && tilIn->value->values.size() >= 2) {
                gpuNode.value[0] = tilIn->value->values[0];
                gpuNode.value[1] = tilIn->value->values[1];
            } else if (tilIn && tilIn->value && tilIn->value->values.size() == 1) {
                gpuNode.value[0] = gpuNode.value[1] = tilIn->value->values[0];
            } else {
                gpuNode.value[0] = gpuNode.value[1] = 1.0f;
            }
        }
        else if (gpuNode.op == OP_PLACE2D) {
            if (hasInput("texcoord")) {
                gpuNode.inA = wireInput("texcoord");
                flagMask |= FLAG_HAS_A;
            }
            if (hasInput("rotate")) {
                gpuNode.inB = wireInput("rotate");
                flagMask |= FLAG_HAS_B;
            }
            auto offIn = findNodeInput(node, "offset");
            if (offIn && offIn->value && offIn->value->values.size() >= 2) {
                gpuNode.value[0] = offIn->value->values[0];
                gpuNode.value[1] = offIn->value->values[1];
            }
            auto scIn = findNodeInput(node, "scale");
            if (scIn && scIn->value && scIn->value->values.size() >= 2) {
                gpuNode.value[2] = scIn->value->values[0];
                gpuNode.value[3] = scIn->value->values[1];
            } else if (scIn && scIn->value && scIn->value->values.size() == 1) {
                gpuNode.value[2] = gpuNode.value[3] = scIn->value->values[0];
            } else {
                gpuNode.value[2] = gpuNode.value[3] = 1.0f;
            }
        }
        else if (gpuNode.op == OP_SWIZZLE) {
            gpuNode.inA = wireInput("in");
            flagMask |= FLAG_HAS_A;
            std::string ch = getStringInput(findNodeInput(node, "channels"));
            auto compIdx = [](char c) -> uint32_t {
                if (c == 'r' || c == 'x') return 0;
                if (c == 'g' || c == 'y') return 1;
                if (c == 'b' || c == 'z') return 2;
                if (c == 'a' || c == 'w') return 3;
                return 0;
            };
            for (size_t i = 0; i < 4 && i < ch.size(); ++i) {
                flagMask |= (compIdx(ch[i]) << (i * 2));
            }
        }
        else if (gpuNode.op == OP_COMBINE) {
            uint32_t count = 0;
            if (hasInput("in1")) { gpuNode.inA = wireInput("in1"); flagMask |= FLAG_HAS_A; count++; }
            if (hasInput("in2")) { gpuNode.inB = wireInput("in2"); flagMask |= FLAG_HAS_B; count++; }
            if (hasInput("in3")) { gpuNode.inC = wireInput("in3"); flagMask |= FLAG_HAS_C; count++; }
            if (hasInput("in4")) { gpuNode.inD = wireInput("in4"); flagMask |= FLAG_HAS_D; count++; }
            flagMask |= count;
        }
        else if (gpuNode.op == OP_EXTRACT) {
            gpuNode.inA = wireInput("in");
            flagMask |= FLAG_HAS_A;
            int idx = getIntInput(findNodeInput(node, "index"));
            flagMask |= static_cast<uint32_t>(idx) & 3;
        }
        else if (gpuNode.op == OP_IFGREATER || gpuNode.op == OP_IFEQUAL) {
            gpuNode.inA = wireInput("value1");
            gpuNode.inB = wireInput("value2");
            gpuNode.inC = wireInput("in1");
            gpuNode.inD = wireInput("in2");
            flagMask |= FLAG_HAS_A | FLAG_HAS_B | FLAG_HAS_C | FLAG_HAS_D;
        }

        gpuNode.flags = flagMask;
        gpuNode.outReg = nextReg++;
        nodeToReg[nodeIdx] = static_cast<int>(gpuNode.outReg);
        bytecode.push_back(gpuNode);
    }

    outOutputReg = static_cast<uint32_t>(nodeToReg[outputNodeIndex]);
    outOutputComponents = componentCountFromType(graph.nodes[outputNodeIndex].type);
    return bytecode;
}

} // anonymous namespace

std::vector<uint8_t> bakeProceduralTextureGPU(
    const std::shared_ptr<systems::leal::campello_gpu::Device>& device,
    const systems::leal::gltf::GLTF& asset,
    const systems::leal::gltf::ProceduralGraph& graph,
    const std::string& outputName,
    int width, int height)
{
    using namespace systems::leal::campello_gpu;

    if (!device) {
        return bakeProceduralTexture(asset, graph, outputName, width, height);
    }

    uint32_t outputReg = 0;
    uint32_t outputComponents = 4;
    auto bytecodeOpt = flattenGraph(graph, outputName, outputReg, outputComponents);
    if (!bytecodeOpt) {
        return bakeProceduralTexture(asset, graph, outputName, width, height);
    }
    const auto& bytecode = *bytecodeOpt;

    // Load platform-specific pre-compiled compute shader.
    using namespace systems::leal::campello_renderer::shaders;
    const uint8_t* shaderData = nullptr;
    unsigned int shaderSize = 0;
    const char* entryPoint = nullptr;

#if defined(__APPLE__)
    shaderData = kDefaultMetalShader;
    shaderSize = kDefaultMetalShaderSize;
    entryPoint = "proceduralBakeKernel";
#elif defined(__ANDROID__) || defined(__linux__)
    shaderData = kDefaultVulkanProceduralBakeShader;
    shaderSize = kDefaultVulkanProceduralBakeShaderSize;
    entryPoint = "main";
#elif defined(_WIN32)
    shaderData = kDefaultDirectXProceduralBakeShader;
    shaderSize = kDefaultDirectXProceduralBakeShaderSize;
    entryPoint = "proceduralBakeKernel";
#endif

    if (!shaderData || shaderSize == 0 || !entryPoint) {
        return bakeProceduralTexture(asset, graph, outputName, width, height);
    }

    auto shaderModule = device->createShaderModule(shaderData, shaderSize);
    if (!shaderModule) {
        return bakeProceduralTexture(asset, graph, outputName, width, height);
    }

    // Bind group layout: 3 storage buffers.
    BindGroupLayoutDescriptor bglDesc{};
    bglDesc.entries = {
        { 0, ShaderStage::compute, EntryObjectType::buffer,
          {{ false, sizeof(ProceduralBakeUniforms), EntryObjectBufferType::uniform }} },
        { 1, ShaderStage::compute, EntryObjectType::buffer,
          {{ false, 0, EntryObjectBufferType::readOnlyStorage }} },
        { 2, ShaderStage::compute, EntryObjectType::buffer,
          {{ false, 0, EntryObjectBufferType::storage }} },
    };
    auto bindGroupLayout = device->createBindGroupLayout(bglDesc);
    if (!bindGroupLayout) {
        return bakeProceduralTexture(asset, graph, outputName, width, height);
    }

    PipelineLayoutDescriptor plDesc{};
    plDesc.bindGroupLayouts = { bindGroupLayout };
    auto pipelineLayout = device->createPipelineLayout(plDesc);
    if (!pipelineLayout) {
        return bakeProceduralTexture(asset, graph, outputName, width, height);
    }

    ComputePipelineDescriptor cpDesc{};
    cpDesc.compute.module = shaderModule;
    cpDesc.compute.entryPoint = entryPoint;
    cpDesc.layout = pipelineLayout;
    auto computePipeline = device->createComputePipeline(cpDesc);
    if (!computePipeline) {
        return bakeProceduralTexture(asset, graph, outputName, width, height);
    }

    // Upload uniforms.
    ProceduralBakeUniforms uniforms{};
    uniforms.width = static_cast<uint32_t>(width);
    uniforms.height = static_cast<uint32_t>(height);
    uniforms.nodeCount = static_cast<uint32_t>(bytecode.size());
    uniforms.outputReg = outputReg;
    uniforms.outputComponents = outputComponents;

    auto uniformBuffer = device->createBuffer(sizeof(ProceduralBakeUniforms),
                                              BufferUsage::uniform, &uniforms);

    // Upload bytecode.
    size_t bytecodeBytes = bytecode.size() * sizeof(ProceduralNodeGPU);
    auto bytecodeBuffer = device->createBuffer(bytecodeBytes,
                                               BufferUsage::storage, (void*)bytecode.data());

    // Output buffer.
    size_t pixelCount = static_cast<size_t>(width) * height;
    size_t outputBytes = pixelCount * sizeof(float) * 4;
    auto outputBuffer = device->createBuffer(outputBytes,
                                             static_cast<BufferUsage>(static_cast<int>(BufferUsage::storage) | static_cast<int>(BufferUsage::copySrc)));

    if (!uniformBuffer || !bytecodeBuffer || !outputBuffer) {
        return bakeProceduralTexture(asset, graph, outputName, width, height);
    }

    // Bind group.
    BindGroupDescriptor bgDesc{};
    bgDesc.layout = bindGroupLayout;
    bgDesc.entries = {
        { 0, BufferBinding{ uniformBuffer, 0, sizeof(ProceduralBakeUniforms) } },
        { 1, BufferBinding{ bytecodeBuffer, 0, bytecodeBytes } },
        { 2, BufferBinding{ outputBuffer, 0, outputBytes } },
    };
    auto bindGroup = device->createBindGroup(bgDesc);
    if (!bindGroup) {
        return bakeProceduralTexture(asset, graph, outputName, width, height);
    }

    // Dispatch.
    auto encoder = device->createCommandEncoder();
    if (!encoder) {
        return bakeProceduralTexture(asset, graph, outputName, width, height);
    }

    auto computePass = encoder->beginComputePass();
    computePass->setPipeline(computePipeline);
    computePass->setBindGroup(0, bindGroup, {}, 0, 0);

    // campello_gpu's Metal backend sets threadgroup size to threadExecutionWidth
    // (e.g. 8 or 32) and dispatches 1-D workgroups. Our shader maps each
    // workgroup to exactly one pixel by dividing the flat thread index by
    // threads-per-threadgroup, so we dispatch one workgroup per pixel.
    computePass->dispatchWorkgroups(static_cast<uint32_t>(width),
                                    static_cast<uint32_t>(height), 1);
    computePass->end();

    // Readback: copy output buffer to CPU-visible buffer.
    auto readbackBuffer = device->createBuffer(outputBytes, static_cast<BufferUsage>(static_cast<int>(BufferUsage::copyDst) | static_cast<int>(BufferUsage::mapRead)));
    if (!readbackBuffer) {
        return bakeProceduralTexture(asset, graph, outputName, width, height);
    }

    encoder->copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputBytes);
    auto cmdBuffer = encoder->finish();
    device->submit(cmdBuffer);
    device->waitForIdle();

    // Download.
    std::vector<float> floatPixels(pixelCount * 4);
    if (!readbackBuffer->download(0, outputBytes, floatPixels.data())) {
        return bakeProceduralTexture(asset, graph, outputName, width, height);
    }

    // Convert float4 -> RGBA8.
    std::vector<uint8_t> pixels(pixelCount * 4);
    for (size_t i = 0; i < pixelCount * 4; ++i) {
        pixels[i] = static_cast<uint8_t>(std::clamp(floatPixels[i] * 255.0f, 0.0f, 255.0f));
    }

    return pixels;
}

} // namespace systems::leal::campello_renderer
