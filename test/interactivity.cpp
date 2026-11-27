// KHR_interactivity support: node visibility, picking (KHR_node_selectability)
// and the object model host (InteractivityObjectModel).

#include <campello_renderer/campello_renderer.hpp>
#include <campello_renderer/interactivity.hpp>
#include <campello_gpu/device.hpp>
#include <campello_gpu/texture.hpp>
#include <gltf/gltf.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <future>
#include <limits>

using namespace systems::leal::campello_renderer;
using systems::leal::gltf::GLTF;

namespace interactivity_test {

using Sig = systems::leal::gltf::InteractivityTypeSignature;
using Value = systems::leal::gltf::InteractivityRuntimeValue;

// Two emissive 2x2 quads facing +Z: "Back" (node 0) at z=0 and its child
// "Front" (node 1) at z=0.5, half the size. Node 2 has an authored matrix.
// Two cameras no node uses: a perspective one without aspectRatio and zfar
// (camera 0) and an orthographic one (camera 1).
// Two lights no node uses: a spot (light 0) and a point light (light 1).
// The renderer's default camera looks at the origin from (0, 0, 5).
// Animation 0 moves Back from x=0 to x=2 in 2 s.
const char* kQuads = R"({
    "asset": {"version": "2.0"},
    "extensionsUsed": ["KHR_texture_transform", "KHR_node_visibility", "KHR_node_selectability", "KHR_lights_punctual",
                       "KHR_materials_clearcoat"],
    "extensions": {"KHR_lights_punctual": {"lights": [
        {"type": "spot", "spot": {"innerConeAngle": 0.25, "outerConeAngle": 0.5}},
        {"type": "point"}]}},
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
                   "emissiveFactor": [1, 0, 0],
                   "extensions": {"KHR_materials_clearcoat": {"clearcoatFactor": 0.5, "clearcoatTexture": {"index": 0},
                                                                "clearcoatNormalTexture": {"index": 0, "scale": 2}}}}],
    "cameras": [{"type": "perspective", "perspective": {"yfov": 0.5, "znear": 0.1}},
                {"type": "orthographic", "orthographic": {"xmag": 2, "ymag": 1, "znear": 0.5, "zfar": 50}}],
    "textures": [{}],
    "accessors": [
        {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3", "min": [-1, -1, 0], "max": [1, 1, 0]},
        {"bufferView": 1, "componentType": 5123, "count": 6, "type": "SCALAR"},
        {"bufferView": 2, "componentType": 5126, "count": 2, "type": "SCALAR", "min": [0], "max": [2]},
        {"bufferView": 3, "componentType": 5126, "count": 2, "type": "VEC3"}
    ],
    "animations": [{"channels": [{"sampler": 0, "target": {"node": 0, "path": "translation"}}],
                    "samplers": [{"input": 2, "output": 3}]}],
    "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 48}, {"buffer": 0, "byteOffset": 48, "byteLength": 12},
                    {"buffer": 0, "byteOffset": 60, "byteLength": 8}, {"buffer": 0, "byteOffset": 68, "byteLength": 24}],
    "buffers": [{"uri": "data:application/octet-stream;base64,AACAvwAAgL8AAAAAAACAPwAAgL8AAAAAAACAPwAAgD8AAAAAAACAvwAAgD8AAAAAAAABAAIAAAACAAMAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAA=", "byteLength": 92}]
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

    // Hover picking reads KHR_node_hoverability instead.
    (*asset->nodes)[1].hoverable = false;
    EXPECT_EQ(renderer->pick(0.5, 0.5, Renderer::PickFilter::hoverable).node, 0);
    EXPECT_EQ(renderer->pick(0.5, 0.5).node, 1);
}

TEST_F(InteractivityRenderTest, ObjectModelPosesAnimationsAndReleasesThem) {
    if (!device) GTEST_SKIP() << "No GPU device available";
    InteractivityObjectModel model(*renderer);
    auto x = [&] { return (*asset->nodes)[0].translation.data[0]; };
    model.applyAnimation(0, 1.0);  // animation/start, on a tick
    EXPECT_NEAR(x(), 1.0, 1e-6);
    renderer->update(0.5);  // held, not advanced
    EXPECT_NEAR(x(), 1.0, 1e-6);
    model.applyAnimation(0, 2.0);
    EXPECT_NEAR(x(), 2.0, 1e-6);
    model.animationStopped(0);  // the properties keep their values
    renderer->update(0.5);
    EXPECT_NEAR(x(), 2.0, 1e-6);
    EXPECT_FALSE(renderer->isAnimationPlaying(0));
    EXPECT_NO_THROW(renderAndRead(0, 0));
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
    // globalMatrix follows a parent's change before the next frame.
    const double up[3] = {0, 4, 0};
    EXPECT_TRUE(model.setPointer("/nodes/0/translation", Value::ofFloats(Sig::Float3, up)));
    ASSERT_TRUE(model.getPointer("/nodes/1/globalMatrix", Sig::Float4x4, v));
    EXPECT_DOUBLE_EQ(v.f[12], 1.0);  // Front's own (1, 2, 3) plus Back's (0, 4, 0)
    EXPECT_DOUBLE_EQ(v.f[13], 6.0);
    EXPECT_DOUBLE_EQ(v.f[14], 3.0);

    // Spot cone angles: spot lights only.
    const std::string spot = "/extensions/KHR_lights_punctual/lights/0/spot/";
    ASSERT_TRUE(model.getPointer(spot + "outerConeAngle", Sig::Float, v));
    EXPECT_DOUBLE_EQ(v.f[0], 0.5);
    EXPECT_TRUE(model.setPointer(spot + "innerConeAngle", Value::ofFloat(0.125)));
    EXPECT_DOUBLE_EQ((*asset->khrLightsPunctual)[0].innerConeAngle, 0.125);
    EXPECT_FALSE(model.getPointer("/extensions/KHR_lights_punctual/lights/1/spot/outerConeAngle", Sig::Float, v));
    EXPECT_FALSE(model.setPointer("/extensions/KHR_lights_punctual/lights/1/spot/innerConeAngle", Value::ofFloat(0.1)));

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
    // KHR_materials_* factors and their textures' transforms.
    const std::string clearcoat = "/materials/0/extensions/KHR_materials_clearcoat/";
    ASSERT_TRUE(model.getPointer(clearcoat + "clearcoatFactor", Sig::Float, v));
    EXPECT_DOUBLE_EQ(v.f[0], 0.5);
    EXPECT_TRUE(model.setPointer(clearcoat + "clearcoatFactor", Value::ofFloat(0.75)));
    EXPECT_DOUBLE_EQ((*asset->materials)[0].khrMaterialsClearcoat->clearcoatFactor, 0.75);
    EXPECT_TRUE(model.setPointer(clearcoat + "clearcoatTexture/extensions/KHR_texture_transform/rotation", Value::ofFloat(1.5)));
    EXPECT_DOUBLE_EQ((*asset->materials)[0].khrMaterialsClearcoat->clearcoatTexture->khrTextureTransform->rotation, 1.5);
    EXPECT_FALSE(model.setPointer(clearcoat + "clearcoatRoughnessTexture/extensions/KHR_texture_transform/rotation",
                                  Value::ofFloat(1.5)));  // no such texture
    EXPECT_FALSE(model.getPointer("/materials/0/extensions/KHR_materials_sheen/sheenRoughnessFactor", Sig::Float, v));
    ASSERT_TRUE(model.getPointer(clearcoat + "clearcoatNormalTexture/scale", Sig::Float, v));
    EXPECT_DOUBLE_EQ(v.f[0], 2.0);
    EXPECT_TRUE(model.setPointer(clearcoat + "clearcoatNormalTexture/scale", Value::ofFloat(0.5)));
    EXPECT_DOUBLE_EQ((*asset->materials)[0].khrMaterialsClearcoat->clearcoatNormalTexture->scale, 0.5);
    EXPECT_FALSE(model.setPointer(clearcoat + "clearcoatNormalTexture/scale", Value::ofFloats(Sig::Float2, o)));

    // Cameras: of their own type; an undefined aspectRatio doesn't resolve
    // until set, an undefined zfar is infinite (and setting it infinite
    // undefines it again).
    auto* persp = dynamic_cast<systems::leal::gltf::PerspectiveCamera*>((*asset->cameras)[0].get());
    auto* orthoCam = dynamic_cast<systems::leal::gltf::OrthographicCamera*>((*asset->cameras)[1].get());
    ASSERT_NE(persp, nullptr);
    ASSERT_NE(orthoCam, nullptr);
    ASSERT_TRUE(model.getPointer("/cameras/0/perspective/yfov", Sig::Float, v));
    EXPECT_DOUBLE_EQ(v.f[0], 0.5);
    EXPECT_TRUE(model.setPointer("/cameras/0/perspective/yfov", Value::ofFloat(1.0)));
    EXPECT_DOUBLE_EQ(persp->yFov, 1.0);
    EXPECT_FALSE(model.getPointer("/cameras/0/perspective/aspectRatio", Sig::Float, v));
    EXPECT_TRUE(model.setPointer("/cameras/0/perspective/aspectRatio", Value::ofFloat(1.5)));
    ASSERT_TRUE(model.getPointer("/cameras/0/perspective/aspectRatio", Sig::Float, v));
    EXPECT_DOUBLE_EQ(v.f[0], 1.5);
    ASSERT_TRUE(model.getPointer("/cameras/0/perspective/zfar", Sig::Float, v));
    EXPECT_TRUE(std::isinf(v.f[0]));
    EXPECT_TRUE(model.setPointer("/cameras/0/perspective/zfar", Value::ofFloat(100)));
    ASSERT_TRUE(persp->zFar.has_value());
    EXPECT_DOUBLE_EQ(*persp->zFar, 100.0);
    EXPECT_TRUE(model.setPointer("/cameras/0/perspective/zfar",
                                 Value::ofFloat(std::numeric_limits<double>::infinity())));
    EXPECT_FALSE(persp->zFar.has_value());
    EXPECT_FALSE(model.getPointer("/cameras/0/orthographic/xmag", Sig::Float, v));  // a perspective camera
    EXPECT_FALSE(model.getPointer("/cameras/0/perspective/yfov", Sig::Float2, v));
    EXPECT_TRUE(model.setPointer("/cameras/1/orthographic/ymag", Value::ofFloat(3)));
    EXPECT_DOUBLE_EQ(orthoCam->yMag, 3.0);
    ASSERT_TRUE(model.getPointer("/cameras/1/orthographic/zfar", Sig::Float, v));
    EXPECT_DOUBLE_EQ(v.f[0], 50.0);
    EXPECT_FALSE(model.getPointer("/cameras/1/perspective/yfov", Sig::Float, v));
    EXPECT_FALSE(model.getPointer("/cameras/2/perspective/yfov", Sig::Float, v));
    EXPECT_NO_THROW(model.flush());
    EXPECT_NO_THROW(renderAndRead(0, 0));

    // The active camera: the default camera at (0, 0, 5), unrotated.
    ASSERT_TRUE(model.getPointer("/extensions/KHR_interactivity/activeCamera/position", Sig::Float3, v));
    EXPECT_NEAR(v.f[2], 5.0, 1e-6);
    ASSERT_TRUE(model.getPointer("/extensions/KHR_interactivity/activeCamera/rotation", Sig::Float4, v));
    EXPECT_NEAR(std::fabs(v.f[3]), 1.0, 1e-6);

    // Its projection: the default camera's perspective (60 degrees, the
    // square target's aspect, 0.1..1000); orthographic values are NaN.
    const char* cam = "/extensions/KHR_interactivity/activeCamera/";
    auto get = [&](const std::string& p) {
        Value f;
        EXPECT_TRUE(model.getPointer(cam + p, Sig::Float, f)) << p;
        return f.f[0];
    };
    EXPECT_NEAR(get("perspective/yfov"), 60.0 * std::acos(-1.0) / 180.0, 1e-6);
    EXPECT_NEAR(get("perspective/aspectRatio"), 1.0, 1e-6);
    EXPECT_NEAR(get("perspective/znear"), 0.1, 1e-6);
    EXPECT_NEAR(get("perspective/zfar"), 1000.0, 1e-3);
    EXPECT_TRUE(std::isnan(get("orthographic/xmag")));
    EXPECT_FALSE(model.getPointer(std::string(cam) + "perspective/fov", Sig::Float, v));

    // An orthographic camera set by the application (column-major, WebGPU
    // depth: 4 x 2 wide, 1..11 deep, looking down -Z).
    const float view[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    const float oL = -2, oR = 2, oB = -1, oT = 1, zn = 1, zf = 11;
    const float ortho[16] = {2 / (oR - oL), 0, 0, 0, 0, 2 / (oT - oB), 0, 0, 0, 0, -1 / (zf - zn), 0,
                             0, 0, -zn / (zf - zn), 1};
    renderer->setCameraMatrices(view, ortho);
    renderAndRead(0, 0);
    EXPECT_NEAR(get("orthographic/xmag"), 2.0, 1e-6);
    EXPECT_NEAR(get("orthographic/ymag"), 1.0, 1e-6);
    EXPECT_NEAR(get("orthographic/znear"), 1.0, 1e-5);
    EXPECT_NEAR(get("orthographic/zfar"), 11.0, 1e-5);
    EXPECT_TRUE(std::isnan(get("perspective/yfov")));
}

}  // namespace interactivity_test
