// Renderer::imageDecoder(): images decoded on another thread, then only
// uploaded by setAsset(asset, decoded).

#include <campello_renderer/campello_renderer.hpp>
#include <campello_gpu/device.hpp>
#include <campello_gpu/texture.hpp>
#include <gltf/gltf.hpp>

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
std::array<uint8_t, 4> renderCentre(const std::shared_ptr<systems::leal::campello_gpu::Device>& device, bool decodeFirst) {
    using namespace systems::leal::campello_gpu;
    constexpr uint32_t kSize = 32;
    Renderer renderer(device);
    auto asset = loadRedQuad();
    EXPECT_NE(asset, nullptr);
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
    const auto ahead = renderCentre(device, true);
    const auto here = renderCentre(device, false);
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

}  // namespace image_decoder_test
