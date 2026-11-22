// KHR_interactivity support: node visibility, picking (KHR_node_selectability)
// and the object model host (InteractivityObjectModel).

#include <campello_renderer/campello_renderer.hpp>
#include <campello_renderer/interactivity.hpp>
#include <campello_gpu/device.hpp>
#include <campello_gpu/texture.hpp>
#include <gltf/gltf.hpp>

#include <gtest/gtest.h>

#include <future>

using namespace systems::leal::campello_renderer;
using systems::leal::gltf::GLTF;

namespace interactivity_test {

using Sig = systems::leal::gltf::InteractivityTypeSignature;
using Value = systems::leal::gltf::InteractivityRuntimeValue;

// Two emissive 2x2 quads facing +Z: "Back" (node 0) at z=0 and its child
// "Front" (node 1) at z=0.5, half the size. Node 2 has an authored matrix.
// The renderer's default camera looks at the origin from (0, 0, 5).
const char* kQuads = R"({
    "asset": {"version": "2.0"},
    "extensionsUsed": ["KHR_texture_transform", "KHR_node_visibility", "KHR_node_selectability"],
    "scene": 0,
    "scenes": [{"nodes": [0, 2]}],
    "nodes": [
        {"name": "Back", "mesh": 0, "children": [1]},
        {"name": "Front", "mesh": 0, "translation": [0, 0, 0.5], "scale": [0.5, 0.5, 1]},
        {"name": "Matrix", "matrix": [1,0,0,0, 0,1,0,0, 0,0,1,0, 9,0,0,1]}
    ],
    "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]}],
    "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [1, 0, 0, 1],
                                            "baseColorTexture": {"index": 0}},
                   "emissiveFactor": [1, 0, 0]}],
    "textures": [{}],
    "accessors": [
        {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3", "min": [-1, -1, 0], "max": [1, 1, 0]},
        {"bufferView": 1, "componentType": 5123, "count": 6, "type": "SCALAR"}
    ],
    "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 48}, {"buffer": 0, "byteOffset": 48, "byteLength": 12}],
    "buffers": [{"uri": "data:application/octet-stream;base64,AACAvwAAgL8AAAAAAACAPwAAgL8AAAAAAACAPwAAgD8AAAAAAACAvwAAgD8AAAAAAAABAAIAAAACAAMA", "byteLength": 60}]
})";

std::shared_ptr<GLTF> loadQuads() {
    return GLTF::loadGLTF(kQuads, [](const std::string&) {
        return std::async(std::launch::deferred, [] { return std::vector<uint8_t>{}; });
    });
}

class InteractivityRenderTest : public testing::Test {
protected:
    std::shared_ptr<systems::leal::campello_gpu::Device> device;
    std::unique_ptr<Renderer> renderer;
    std::shared_ptr<GLTF> asset;
    std::shared_ptr<systems::leal::campello_gpu::Texture> target;
    std::shared_ptr<systems::leal::campello_gpu::TextureView> view;
    static constexpr uint32_t kSize = 64;

    void SetUp() override {
        using namespace systems::leal::campello_gpu;
        device = Device::createDefaultDevice(nullptr);
        if (!device) return;
        renderer = std::make_unique<Renderer>(device);
        asset = loadQuads();
        ASSERT_NE(asset, nullptr);
        renderer->setAsset(asset);
        renderer->resize(kSize, kSize);
        renderer->createDefaultPipelines(PixelFormat::rgba8unorm);
        renderer->setSkyboxEnabled(false);
        renderer->setClearColor(0.0f, 0.0f, 1.0f, 1.0f);
        target = device->createTexture(TextureType::tt2d, PixelFormat::rgba8unorm, kSize, kSize, 1, 1, 1,
                                       static_cast<TextureUsage>(static_cast<uint32_t>(TextureUsage::renderTarget) |
                                                                 static_cast<uint32_t>(TextureUsage::copySrc)));
        ASSERT_NE(target, nullptr);
        view = target->createView(PixelFormat::rgba8unorm, 1);
    }

    void TearDown() override {
        if (device) device->waitForIdle();
    }

    // The rendered pixel at (x, y) as RGBA.
    std::array<uint8_t, 4> renderAndRead(uint32_t x, uint32_t y) {
        renderer->render(view);
        device->waitForIdle();
        std::vector<uint8_t> pixels(kSize * kSize * 4);
        EXPECT_TRUE(target->download(0, 0, pixels.data(), pixels.size()));
        const size_t at = (y * kSize + x) * 4;
        return {pixels[at], pixels[at + 1], pixels[at + 2], pixels[at + 3]};
    }
};

bool isClearColor(const std::array<uint8_t, 4>& p) { return p[0] < 20 && p[1] < 20 && p[2] > 230; }

TEST_F(InteractivityRenderTest, HiddenNodeHidesItsSubtree) {
    if (!device) GTEST_SKIP() << "No GPU device available";
    EXPECT_FALSE(isClearColor(renderAndRead(kSize / 2, kSize / 2)));
    (*asset->nodes)[0].visible = false;  // Back, Front's parent
    EXPECT_TRUE(isClearColor(renderAndRead(kSize / 2, kSize / 2)));
    (*asset->nodes)[0].visible = true;
    (*asset->nodes)[1].visible = false;  // Front only: Back still covers the centre
    EXPECT_FALSE(isClearColor(renderAndRead(kSize / 2, kSize / 2)));
}

TEST_F(InteractivityRenderTest, PickFindsTheNearestSelectableNode) {
    if (!device) GTEST_SKIP() << "No GPU device available";
    renderAndRead(0, 0);  // pick() uses the last frame's camera and transforms

    auto hit = renderer->pick(0.5, 0.5);
    EXPECT_EQ(hit.node, 1);  // Front, nearer
    EXPECT_NEAR(hit.point[2], 0.5, 1e-4);
    EXPECT_NEAR(hit.rayOrigin[2], 5.0, 1e-3);
    EXPECT_NEAR(hit.distance, 4.5, 1e-3);

    // Outside Front (half size) but on Back.
    EXPECT_EQ(renderer->pick(0.63, 0.5).node, 0);
    // Outside both.
    EXPECT_EQ(renderer->pick(0.95, 0.5).node, -1);

    // Unselectable: the ray goes through it.
    (*asset->nodes)[1].selectable = false;
    EXPECT_EQ(renderer->pick(0.5, 0.5).node, 0);
    // An unselectable parent makes its subtree unselectable.
    (*asset->nodes)[1].selectable = true;
    (*asset->nodes)[0].selectable = false;
    EXPECT_EQ(renderer->pick(0.5, 0.5).node, -1);
    // Visibility doesn't matter (KHR_node_selectability).
    (*asset->nodes)[0].selectable = true;
    (*asset->nodes)[1].visible = false;
    EXPECT_EQ(renderer->pick(0.5, 0.5).node, 1);
}

TEST_F(InteractivityRenderTest, ObjectModelReadsAndWritesTheAsset) {
    if (!device) GTEST_SKIP() << "No GPU device available";
    renderAndRead(0, 0);
    InteractivityObjectModel model(*renderer);
    Value v;

    // Node TRS and flags.
    ASSERT_TRUE(model.getPointer("/nodes/1/translation", Sig::Float3, v));
    EXPECT_DOUBLE_EQ(v.f[2], 0.5);
    EXPECT_FALSE(model.getPointer("/nodes/1/translation", Sig::Float2, v));  // wrong type
    const double t[3] = {1, 2, 3};
    EXPECT_TRUE(model.setPointer("/nodes/1/translation", Value::ofFloats(Sig::Float3, t)));
    EXPECT_DOUBLE_EQ((*asset->nodes)[1].translation.data[1], 2.0);
    EXPECT_FALSE(model.setPointer("/nodes/2/translation", Value::ofFloats(Sig::Float3, t)));  // authored matrix
    EXPECT_FALSE(model.setPointer("/nodes/9/translation", Value::ofFloats(Sig::Float3, t)));
    EXPECT_TRUE(model.setPointer("/nodes/0/extensions/KHR_node_visibility/visible", Value::ofBool(false)));
    EXPECT_FALSE((*asset->nodes)[0].visible);
    ASSERT_TRUE(model.getPointer("/nodes/0/extensions/KHR_node_selectability/selectable", Sig::Bool, v));
    EXPECT_TRUE(v.b);

    // Matrices are column-major: the translation is f[12..14].
    ASSERT_TRUE(model.getPointer("/nodes/2/matrix", Sig::Float4x4, v));
    EXPECT_DOUBLE_EQ(v.f[12], 9.0);
    ASSERT_TRUE(model.getPointer("/nodes/2/globalMatrix", Sig::Float4x4, v));
    EXPECT_DOUBLE_EQ(v.f[12], 9.0);
    EXPECT_FALSE(model.setPointer("/nodes/2/matrix", v));  // read-only

    // Material factors and KHR_texture_transform (created on first write).
    const std::string offset = "/materials/0/pbrMetallicRoughness/baseColorTexture/extensions/KHR_texture_transform/offset";
    ASSERT_TRUE(model.getPointer(offset, Sig::Float2, v));
    EXPECT_DOUBLE_EQ(v.f[0], 0.0);
    const double o[2] = {0.25, 0.5};
    EXPECT_TRUE(model.setPointer(offset, Value::ofFloats(Sig::Float2, o)));
    auto& base = *(*asset->materials)[0].pbrMetallicRoughness->baseColorTexture;
    ASSERT_NE(base.khrTextureTransform, nullptr);
    EXPECT_DOUBLE_EQ(base.khrTextureTransform->offset.data[1], 0.5);
    EXPECT_DOUBLE_EQ(base.khrTextureTransform->scale.data[0], 1.0);
    EXPECT_TRUE(model.setPointer("/materials/0/pbrMetallicRoughness/metallicFactor", Value::ofFloat(0.25)));
    EXPECT_FALSE(model.setPointer("/materials/0/emissiveTexture/extensions/KHR_texture_transform/offset",
                                  Value::ofFloats(Sig::Float2, o)));  // no emissive texture
    EXPECT_NO_THROW(model.flush());
    EXPECT_NO_THROW(renderAndRead(0, 0));

    // The active camera: the default camera at (0, 0, 5), unrotated.
    ASSERT_TRUE(model.getPointer("/extensions/KHR_interactivity/activeCamera/position", Sig::Float3, v));
    EXPECT_NEAR(v.f[2], 5.0, 1e-6);
    ASSERT_TRUE(model.getPointer("/extensions/KHR_interactivity/activeCamera/rotation", Sig::Float4, v));
    EXPECT_NEAR(std::fabs(v.f[3]), 1.0, 1e-6);
}

}  // namespace interactivity_test
