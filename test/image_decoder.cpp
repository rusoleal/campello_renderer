// Renderer::imageDecoder(): images decoded on another thread, then only
// uploaded by setAsset(asset, decoded).

#include <campello_renderer/campello_renderer.hpp>
#include <campello_gpu/device.hpp>
#include <campello_gpu/texture.hpp>
#include <gltf/gltf.hpp>

#include <campello_image/texture_encoder.hpp>
#include <gtest/gtest.h>

#include <array>
#include <future>
#include <thread>

using namespace systems::leal::campello_renderer;
using systems::leal::gltf::GLTF;

namespace image_decoder_test {

// A 2x2 quad facing +Z, unlit, textured with a 2x2 red PNG (a data: URI).
// The renderer's default camera looks at the origin from (0, 0, 5).
const char* kRedQuad = R"({
    "asset": {"version": "2.0"},
    "extensionsUsed": ["KHR_materials_unlit"],
    "scene": 0,
    "scenes": [{"nodes": [0]}],
    "nodes": [{"mesh": 0}],
    "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 2, "material": 0}]}],
    "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
                   "extensions": {"KHR_materials_unlit": {}}}],
    "textures": [{"source": 0}],
    "images": [{"uri": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAEUlEQVR4nGP4z8DwH4QZYAwAR8oH+WdZbrcAAAAASUVORK5CYII="}],
    "accessors": [
        {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3", "min": [-1, -1, 0], "max": [1, 1, 0]},
        {"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC2"},
        {"bufferView": 2, "componentType": 5123, "count": 6, "type": "SCALAR"}
    ],
    "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 48}, {"buffer": 0, "byteOffset": 48, "byteLength": 32},
                    {"buffer": 0, "byteOffset": 80, "byteLength": 12}],
    "buffers": [{"uri": "data:application/octet-stream;base64,AACAvwAAgL8AAAAAAACAPwAAgL8AAAAAAACAPwAAgD8AAAAAAACAvwAAgD8AAAAAAAAAAAAAgD8AAIA/AACAPwAAgD8AAAAAAAAAAAAAAAAAAAEAAgAAAAIAAwA=", "byteLength": 92}]
})";

std::shared_ptr<GLTF> loadRedQuad() {
    return GLTF::loadGLTF(kRedQuad, [](const std::string&) {
        return std::async(std::launch::deferred, [] { return std::vector<uint8_t>{}; });
    });
}

// The centre pixel of the asset rendered at 32x32 (black clear colour),
// with its images decoded beforehand on another thread or not.
// (Without its texture the quad would be white.)
std::array<uint8_t, 4> renderCentre(const std::shared_ptr<systems::leal::campello_gpu::Device>& device,
                                    const std::shared_ptr<GLTF>& asset, bool decodeFirst) {
    using namespace systems::leal::campello_gpu;
    constexpr uint32_t kSize = 32;
    Renderer renderer(device);
    if (decodeFirst) {
        auto decoder = renderer.imageDecoder();
        std::shared_ptr<DecodedImages> decoded;
        std::thread worker([&] { decoded = decoder(*asset); });
        worker.join();
        EXPECT_NE(decoded, nullptr);
        renderer.setAsset(asset, decoded);
    } else {
        renderer.setAsset(asset);
    }
    renderer.resize(kSize, kSize);
    renderer.createDefaultPipelines(PixelFormat::rgba8unorm);
    renderer.setSkyboxEnabled(false);
    renderer.setClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    auto target = device->createTexture(TextureType::tt2d, PixelFormat::rgba8unorm, kSize, kSize, 1, 1, 1,
                                        static_cast<TextureUsage>(static_cast<uint32_t>(TextureUsage::renderTarget) |
                                                                  static_cast<uint32_t>(TextureUsage::copySrc)));
    EXPECT_NE(target, nullptr);
    if (!target) return {};
    auto view = target->createView(PixelFormat::rgba8unorm, 1);
    renderer.render(view);
    device->waitForIdle();
    std::vector<uint8_t> pixels(kSize * kSize * 4);
    EXPECT_TRUE(target->download(0, 0, pixels.data(), pixels.size()));
    const size_t at = ((kSize / 2) * kSize + kSize / 2) * 4;
    return {pixels[at], pixels[at + 1], pixels[at + 2], pixels[at + 3]};
}

TEST(ImageDecoderTest, ImagesDecodedOnAnotherThreadRenderAsWhenDecodedBySetAsset) {
    auto device = systems::leal::campello_gpu::Device::createDefaultDevice(nullptr);
    if (!device) GTEST_SKIP() << "no GPU device";
    const auto ahead = renderCentre(device, loadRedQuad(), true);
    const auto here = renderCentre(device, loadRedQuad(), false);
    EXPECT_GT(ahead[0], 200) << "red expected";
    EXPECT_LT(ahead[1], 40);
    EXPECT_LT(ahead[2], 40);
    EXPECT_EQ(ahead, here);
}

TEST(ImageDecoderTest, DecodesWithoutADevice) {
    // A renderer without a device still decodes (KTX2/Basis to RGBA8).
    Renderer renderer(nullptr);
    auto asset = loadRedQuad();
    ASSERT_NE(asset, nullptr);
    auto decoded = renderer.imageDecoder()(*asset);
    EXPECT_NE(decoded, nullptr);
    renderer.setAsset(asset, decoded);  // no device: nothing is uploaded, nothing breaks
    renderer.setAsset(nullptr, decoded);
    EXPECT_EQ(renderer.getAsset(), nullptr);
}

// The red quad with its texture as a KTX2 file and only KHR_texture_basisu
// naming it (no "source", as KTX2-only assets have).
std::string base64(const std::vector<uint8_t>& bytes) {
    static const char* abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    for (size_t i = 0; i < bytes.size(); i += 3) {
        const uint32_t n = (uint32_t)bytes[i] << 16 | (i + 1 < bytes.size() ? (uint32_t)bytes[i + 1] << 8 : 0) |
                           (i + 2 < bytes.size() ? bytes[i + 2] : 0);
        out += abc[n >> 18 & 63];
        out += abc[n >> 12 & 63];
        out += i + 1 < bytes.size() ? abc[n >> 6 & 63] : '=';
        out += i + 2 < bytes.size() ? abc[n & 63] : '=';
    }
    return out;
}

std::shared_ptr<GLTF> loadRedKtx2Quad() {
    std::vector<uint8_t> red(4 * 4 * 4);
    for (size_t i = 0; i < red.size(); i += 4) {
        red[i] = 255;
        red[i + 3] = 255;
    }
    systems::leal::campello_image::EncodeOptions options;
    options.threaded = false;
    const auto ktx2 = systems::leal::campello_image::encodeKtx2(red.data(), 4, 4, options);
    if (ktx2.empty()) return nullptr;
    std::string json = kRedQuad;
    const auto at = json.find("\"textures\": [{\"source\": 0}]");
    json.replace(at, std::string("\"textures\": [{\"source\": 0}]").size(),
                 "\"textures\": [{\"extensions\": {\"KHR_texture_basisu\": {\"source\": 0}}}]");
    const auto image = json.find("data:image/png;base64,");
    const auto end = json.find('"', image);
    json.replace(image, end - image, "data:image/ktx2;base64," + base64(ktx2));
    json.replace(json.find("\"KHR_materials_unlit\"]"), std::string("\"KHR_materials_unlit\"]").size(),
                 "\"KHR_materials_unlit\", \"KHR_texture_basisu\"]");
    return GLTF::loadGLTF(json, [](const std::string&) {
        return std::async(std::launch::deferred, [] { return std::vector<uint8_t>{}; });
    });
}

TEST(ImageDecoderTest, Ktx2OnlyTexturesAreSampled) {
    if (!systems::leal::campello_image::canEncode()) GTEST_SKIP() << "no KTX2 encoder";
    auto device = systems::leal::campello_gpu::Device::createDefaultDevice(nullptr);
    if (!device) GTEST_SKIP() << "no GPU device";
    for (const bool decodeFirst : {false, true}) {
        auto asset = loadRedKtx2Quad();
        ASSERT_NE(asset, nullptr);
        const auto centre = renderCentre(device, asset, decodeFirst);
        // Without its KTX2 texture the quad would be white.
        EXPECT_GT(centre[0], 200) << "decodeFirst " << decodeFirst;
        EXPECT_LT(centre[1], 40) << "decodeFirst " << decodeFirst;
        EXPECT_LT(centre[2], 40) << "decodeFirst " << decodeFirst;
    }
}

}  // namespace image_decoder_test
