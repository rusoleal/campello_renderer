#include <gtest/gtest.h>
#include <campello_renderer/campello_renderer.hpp>
#include <campello_renderer/procedural_texture_baker.hpp>
#include <gltf/gltf.hpp>
#include <future>
#include <vector>
#include <fstream>
#include <cmath>

#if defined(__APPLE__)
#include <campello_gpu/device.hpp>
#endif

using namespace systems::leal::campello_renderer;
using namespace systems::leal::gltf;

// No-op resource loader for embedded GLTF data (no external resources).
static auto kNoOpLoader = [](const std::string&) {
    return std::async(std::launch::deferred, []() {
        return std::vector<uint8_t>{};
    });
};

// ---------------------------------------------------------------------------
// Helpers — minimal valid GLTF JSON strings
// ---------------------------------------------------------------------------

static const char *kGltfNoScenes = R"({
    "asset": { "version": "2.0" },
    "scenes": [],
    "scene": -1
})";

static const char *kGltfOneEmptyScene = R"({
    "asset": { "version": "2.0" },
    "scene": 0,
    "scenes": [{ "name": "Scene", "nodes": [] }]
})";

static const char *kGltfTwoEmptyScenes = R"({
    "asset": { "version": "2.0" },
    "scene": 0,
    "scenes": [
        { "name": "Scene0", "nodes": [] },
        { "name": "Scene1", "nodes": [] }
    ]
})";

static const char *kGltfWithCamera = R"({
    "asset": { "version": "2.0" },
    "scene": 0,
    "cameras": [{
        "name": "Camera",
        "type": "perspective",
        "perspective": { "yfov": 0.785, "znear": 0.01 }
    }],
    "nodes": [{ "name": "CameraNode", "camera": 0 }],
    "scenes": [{ "name": "Scene", "nodes": [0] }]
})";

static const char *kGltfWithMesh = R"({
    "asset": { "version": "2.0" },
    "scene": 0,
    "accessors": [
        { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
          "max": [1,1,0], "min": [-1,-1,0] },
        { "bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR" }
    ],
    "bufferViews": [
        { "buffer": 0, "byteOffset": 0,  "byteLength": 36 },
        { "buffer": 0, "byteOffset": 36, "byteLength": 6  }
    ],
    "buffers": [{ "byteLength": 44 }],
    "meshes": [{
        "name": "Triangle",
        "primitives": [{
            "attributes": { "POSITION": 0 },
            "indices": 1
        }]
    }],
    "nodes": [{ "name": "Triangle", "mesh": 0 }],
    "scenes": [{ "name": "Scene", "nodes": [0] }]
})";

static const char *kGltfWithBasisuTexture = R"({
    "asset": { "version": "2.0" },
    "extensionsUsed": ["KHR_texture_basisu"],
    "scene": 0,
    "images": [
        { "uri": "fallback.png" },
        { "uri": "compressed.ktx2" }
    ],
    "textures": [{
        "source": 0,
        "extensions": {
            "KHR_texture_basisu": { "source": 1 }
        }
    }],
    "materials": [{
        "pbrMetallicRoughness": {
            "baseColorTexture": { "index": 0 }
        }
    }],
    "meshes": [{
        "primitives": [{
            "attributes": { "POSITION": 0 },
            "material": 0
        }]
    }],
    "accessors": [
        { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3" }
    ],
    "bufferViews": [
        { "buffer": 0, "byteOffset": 0, "byteLength": 36 }
    ],
    "buffers": [{ "byteLength": 36 }],
    "nodes": [{ "mesh": 0 }],
    "scenes": [{ "nodes": [0] }]
})";

// ---------------------------------------------------------------------------
// Version
// ---------------------------------------------------------------------------

TEST(VersionTest, ReturnsExpectedVersion) {
    EXPECT_EQ(systems::leal::campello_renderer::getVersion(), "0.7.0");
}

TEST(VersionTest, VersionIsNonEmpty) {
    EXPECT_FALSE(systems::leal::campello_renderer::getVersion().empty());
}

TEST(VersionTest, VersionHasMajorMinorPatch) {
    auto v = systems::leal::campello_renderer::getVersion();
    int count = 0;
    for (char c : v) { if (c == '.') count++; }
    EXPECT_EQ(count, 2);
}

// ---------------------------------------------------------------------------
// Renderer construction
// ---------------------------------------------------------------------------

TEST(RendererConstructionTest, NullDeviceDoesNotCrash) {
    EXPECT_NO_THROW(Renderer renderer(nullptr));
}

TEST(RendererConstructionTest, InitialAssetIsNull) {
    Renderer renderer(nullptr);
    EXPECT_EQ(renderer.getAsset(), nullptr);
}

// ---------------------------------------------------------------------------
// setAsset / getAsset — null
// ---------------------------------------------------------------------------

TEST(SetAssetTest, SetNullClearsAsset) {
    Renderer renderer(nullptr);
    renderer.setAsset(nullptr);
    EXPECT_EQ(renderer.getAsset(), nullptr);
}

TEST(SetAssetTest, SetNullIsIdempotent) {
    Renderer renderer(nullptr);
    renderer.setAsset(nullptr);
    renderer.setAsset(nullptr);
    EXPECT_EQ(renderer.getAsset(), nullptr);
}

// ---------------------------------------------------------------------------
// setAsset / getAsset — real GLTF assets (no GPU device needed)
// ---------------------------------------------------------------------------

TEST(SetAssetTest, SetAssetWithNoScenes) {
    auto asset = GLTF::loadGLTF(kGltfNoScenes, kNoOpLoader);
    ASSERT_NE(asset, nullptr);
    Renderer renderer(nullptr);
    renderer.setAsset(asset);
    EXPECT_EQ(renderer.getAsset(), asset);
}

TEST(SetAssetTest, SetAssetWithOneEmptyScene) {
    auto asset = GLTF::loadGLTF(kGltfOneEmptyScene, kNoOpLoader);
    ASSERT_NE(asset, nullptr);
    Renderer renderer(nullptr);
    EXPECT_NO_THROW(renderer.setAsset(asset));
    EXPECT_EQ(renderer.getAsset(), asset);
}

TEST(SetAssetTest, SetAssetWithTwoScenes) {
    auto asset = GLTF::loadGLTF(kGltfTwoEmptyScenes, kNoOpLoader);
    ASSERT_NE(asset, nullptr);
    Renderer renderer(nullptr);
    EXPECT_NO_THROW(renderer.setAsset(asset));
    EXPECT_EQ(renderer.getAsset(), asset);
}

TEST(SetAssetTest, SetAssetWithCameraNode) {
    auto asset = GLTF::loadGLTF(kGltfWithCamera, kNoOpLoader);
    ASSERT_NE(asset, nullptr);
    Renderer renderer(nullptr);
    EXPECT_NO_THROW(renderer.setAsset(asset));
    EXPECT_EQ(renderer.getAsset(), asset);
}

TEST(SetAssetTest, SetAssetWithMeshNode) {
    auto asset = GLTF::loadGLTF(kGltfWithMesh, kNoOpLoader);
    ASSERT_NE(asset, nullptr);
    Renderer renderer(nullptr);
    EXPECT_NO_THROW(renderer.setAsset(asset));
    EXPECT_EQ(renderer.getAsset(), asset);
}

TEST(SetAssetTest, SetAssetTwiceReplacesAsset) {
    auto asset1 = GLTF::loadGLTF(kGltfOneEmptyScene, kNoOpLoader);
    auto asset2 = GLTF::loadGLTF(kGltfTwoEmptyScenes, kNoOpLoader);
    Renderer renderer(nullptr);
    renderer.setAsset(asset1);
    renderer.setAsset(asset2);
    EXPECT_EQ(renderer.getAsset(), asset2);
}

TEST(SetAssetTest, SetAssetThenClearWithNull) {
    auto asset = GLTF::loadGLTF(kGltfOneEmptyScene, kNoOpLoader);
    Renderer renderer(nullptr);
    renderer.setAsset(asset);
    EXPECT_EQ(renderer.getAsset(), asset);
    renderer.setAsset(nullptr);
    EXPECT_EQ(renderer.getAsset(), nullptr);
}

TEST(SetAssetTest, SetAssetThenClearThenSetAgain) {
    auto asset = GLTF::loadGLTF(kGltfOneEmptyScene, kNoOpLoader);
    Renderer renderer(nullptr);
    renderer.setAsset(asset);
    renderer.setAsset(nullptr);
    renderer.setAsset(asset);
    EXPECT_EQ(renderer.getAsset(), asset);
}

TEST(SetAssetTest, SetAssetWithBasisuTextureDoesNotCrash) {
    auto asset = GLTF::loadGLTF(kGltfWithBasisuTexture, kNoOpLoader);
    ASSERT_NE(asset, nullptr);
    ASSERT_TRUE(asset->textures != nullptr);
    ASSERT_EQ(asset->textures->size(), 1u);
    EXPECT_EQ((*asset->textures)[0].khr_texture_basisu, 1);
    Renderer renderer(nullptr);
    EXPECT_NO_THROW(renderer.setAsset(asset));
    EXPECT_EQ(renderer.getAsset(), asset);
}

// ---------------------------------------------------------------------------
// setScene
// ---------------------------------------------------------------------------

TEST(SetSceneTest, NullAssetDoesNotCrash) {
    Renderer renderer(nullptr);
    EXPECT_NO_THROW(renderer.setScene(0));
}

TEST(SetSceneTest, ValidAssetValidScene) {
    auto asset = GLTF::loadGLTF(kGltfOneEmptyScene, kNoOpLoader);
    Renderer renderer(nullptr);
    renderer.setAsset(asset);
    EXPECT_NO_THROW(renderer.setScene(0));
}

TEST(SetSceneTest, OutOfBoundsSceneIndexDoesNotCrash) {
    auto asset = GLTF::loadGLTF(kGltfOneEmptyScene, kNoOpLoader);
    Renderer renderer(nullptr);
    renderer.setAsset(asset);
    EXPECT_NO_THROW(renderer.setScene(99));
}

TEST(SetSceneTest, SwitchBetweenScenes) {
    auto asset = GLTF::loadGLTF(kGltfTwoEmptyScenes, kNoOpLoader);
    Renderer renderer(nullptr);
    renderer.setAsset(asset);
    EXPECT_NO_THROW(renderer.setScene(0));
    EXPECT_NO_THROW(renderer.setScene(1));
    EXPECT_NO_THROW(renderer.setScene(0));
}

TEST(SetSceneTest, SetSceneAfterClearingAsset) {
    auto asset = GLTF::loadGLTF(kGltfOneEmptyScene, kNoOpLoader);
    Renderer renderer(nullptr);
    renderer.setAsset(asset);
    renderer.setAsset(nullptr);
    EXPECT_NO_THROW(renderer.setScene(0));
}

TEST(SetSceneTest, SetSceneWithMeshNode) {
    auto asset = GLTF::loadGLTF(kGltfWithMesh, kNoOpLoader);
    Renderer renderer(nullptr);
    renderer.setAsset(asset);
    EXPECT_NO_THROW(renderer.setScene(0));
}

// ---------------------------------------------------------------------------
// setCamera
// ---------------------------------------------------------------------------

TEST(SetCameraTest, IndexZeroDoesNotCrash) {
    Renderer renderer(nullptr);
    EXPECT_NO_THROW(renderer.setCamera(0));
}

TEST(SetCameraTest, LargeIndexDoesNotCrash) {
    Renderer renderer(nullptr);
    EXPECT_NO_THROW(renderer.setCamera(UINT32_MAX));
}

TEST(SetCameraTest, SetCameraMultipleTimes) {
    Renderer renderer(nullptr);
    for (uint32_t i = 0; i < 10; ++i) {
        EXPECT_NO_THROW(renderer.setCamera(i));
    }
}

TEST(SetCameraTest, SetCameraWithAssetLoaded) {
    auto asset = GLTF::loadGLTF(kGltfWithCamera, kNoOpLoader);
    Renderer renderer(nullptr);
    renderer.setAsset(asset);
    EXPECT_NO_THROW(renderer.setCamera(0));
}

// ---------------------------------------------------------------------------
// render / update (stubs)
// ---------------------------------------------------------------------------

TEST(RenderTest, RenderDoesNotCrash) {
    Renderer renderer(nullptr);
    EXPECT_NO_THROW(renderer.render());
}

TEST(RenderTest, RenderWithAsset) {
    auto asset = GLTF::loadGLTF(kGltfOneEmptyScene, kNoOpLoader);
    Renderer renderer(nullptr);
    renderer.setAsset(asset);
    EXPECT_NO_THROW(renderer.render());
}

TEST(RenderTest, RenderCalledMultipleTimes) {
    Renderer renderer(nullptr);
    for (int i = 0; i < 10; ++i) {
        EXPECT_NO_THROW(renderer.render());
    }
}

TEST(UpdateTest, ZeroDeltaDoesNotCrash) {
    Renderer renderer(nullptr);
    EXPECT_NO_THROW(renderer.update(0.0));
}

TEST(UpdateTest, TypicalFrameDeltaDoesNotCrash) {
    Renderer renderer(nullptr);
    EXPECT_NO_THROW(renderer.update(0.016));
}

TEST(UpdateTest, LargeDeltaDoesNotCrash) {
    Renderer renderer(nullptr);
    EXPECT_NO_THROW(renderer.update(1000.0));
}

TEST(UpdateTest, NegativeDeltaDoesNotCrash) {
    Renderer renderer(nullptr);
    EXPECT_NO_THROW(renderer.update(-1.0));
}

TEST(UpdateTest, UpdateCalledMultipleTimes) {
    Renderer renderer(nullptr);
    for (int i = 0; i < 60; ++i) {
        EXPECT_NO_THROW(renderer.update(0.016));
    }
}

// ---------------------------------------------------------------------------
// Sequences — multi-step interaction
// ---------------------------------------------------------------------------

TEST(SequenceTest, FullLifecycleWithAsset) {
    auto asset = GLTF::loadGLTF(kGltfWithCamera, kNoOpLoader);
    Renderer renderer(nullptr);
    renderer.setAsset(asset);
    renderer.setScene(0);
    renderer.setCamera(0);
    EXPECT_NO_THROW(renderer.render());
    EXPECT_NO_THROW(renderer.update(0.016));
    EXPECT_EQ(renderer.getAsset(), asset);
}

TEST(SequenceTest, ReplaceAssetAndRender) {
    auto asset1 = GLTF::loadGLTF(kGltfOneEmptyScene, kNoOpLoader);
    auto asset2 = GLTF::loadGLTF(kGltfWithMesh, kNoOpLoader);
    Renderer renderer(nullptr);
    renderer.setAsset(asset1);
    renderer.setScene(0);
    renderer.setAsset(asset2);
    renderer.setScene(0);
    EXPECT_NO_THROW(renderer.render());
    EXPECT_EQ(renderer.getAsset(), asset2);
}

TEST(SequenceTest, SimulateFrameLoop) {
    auto asset = GLTF::loadGLTF(kGltfWithCamera, kNoOpLoader);
    Renderer renderer(nullptr);
    renderer.setAsset(asset);
    renderer.setScene(0);
    renderer.setCamera(0);
    const double dt = 1.0 / 60.0;
    for (int i = 0; i < 120; ++i) {
        EXPECT_NO_THROW(renderer.update(dt));
        EXPECT_NO_THROW(renderer.render());
    }
}

TEST(SequenceTest, ClearAssetBetweenRenders) {
    auto asset = GLTF::loadGLTF(kGltfOneEmptyScene, kNoOpLoader);
    Renderer renderer(nullptr);
    renderer.setAsset(asset);
    renderer.render();
    renderer.setAsset(nullptr);
    EXPECT_NO_THROW(renderer.render());
    EXPECT_EQ(renderer.getAsset(), nullptr);
}

// ---------------------------------------------------------------------------
// Procedural texture baker
// ---------------------------------------------------------------------------

TEST(ProceduralBakerTest, ConstantColor3) {
    using namespace systems::leal::gltf;
    ProceduralGraph graph;
    graph.name = "constant_red";
    graph.nodetype = "nodegraph";
    graph.type = "color3";

    ProceduralNode node;
    node.name = "red";
    node.nodetype = "constant";
    node.type = "color3";
    ProceduralNodeInput valIn;
    valIn.name = "value";
    valIn.type = "color3";
    valIn.value = std::make_shared<ProceduralValue>("color3");
    valIn.value->values = {1.0f, 0.0f, 0.0f};
    node.inputs.push_back(valIn);
    ProceduralNodeOutput out;
    out.name = "out";
    out.type = "color3";
    node.outputs.push_back(out);
    graph.nodes.push_back(node);

    ProceduralGraphOutput gout;
    gout.name = "result";
    gout.type = "color3";
    gout.nodeIndex = 0;
    graph.outputs.push_back(gout);

    auto dummyAsset = GLTF::loadGLTF(R"({"asset":{"version":"2.0"}})", kNoOpLoader);
    auto pixels = systems::leal::campello_renderer::bakeProceduralTexture(
        *dummyAsset, graph, "result", 4, 4);

    ASSERT_EQ(pixels.size(), 4u * 4u * 4u);
    // Center pixel should be red
    size_t idx = (2 * 4 + 2) * 4;
    EXPECT_EQ(pixels[idx + 0], 255);
    EXPECT_EQ(pixels[idx + 1], 0);
    EXPECT_EQ(pixels[idx + 2], 0);
    EXPECT_EQ(pixels[idx + 3], 255);
}

TEST(ProceduralBakerTest, MixNode) {
    using namespace systems::leal::gltf;
    ProceduralGraph graph;
    graph.name = "mix_test";
    graph.nodetype = "nodegraph";
    graph.type = "color3";

    // Node 0: constant red
    ProceduralNode redNode;
    redNode.name = "red";
    redNode.nodetype = "constant";
    redNode.type = "color3";
    ProceduralNodeInput redVal;
    redVal.name = "value";
    redVal.type = "color3";
    redVal.value = std::make_shared<ProceduralValue>("color3");
    redVal.value->values = {1.0f, 0.0f, 0.0f};
    redNode.inputs.push_back(redVal);
    redNode.outputs.push_back({"out", "color3"});
    graph.nodes.push_back(redNode);

    // Node 1: constant blue
    ProceduralNode blueNode;
    blueNode.name = "blue";
    blueNode.nodetype = "constant";
    blueNode.type = "color3";
    ProceduralNodeInput blueVal;
    blueVal.name = "value";
    blueVal.type = "color3";
    blueVal.value = std::make_shared<ProceduralValue>("color3");
    blueVal.value->values = {0.0f, 0.0f, 1.0f};
    blueNode.inputs.push_back(blueVal);
    blueNode.outputs.push_back({"out", "color3"});
    graph.nodes.push_back(blueNode);

    // Node 2: mix with t=0.5
    ProceduralNode mixNode;
    mixNode.name = "mix";
    mixNode.nodetype = "mix";
    mixNode.type = "color3";
    ProceduralNodeInput fg;
    fg.name = "fg";
    fg.type = "color3";
    fg.nodeIndex = 0;
    mixNode.inputs.push_back(fg);
    ProceduralNodeInput bg;
    bg.name = "bg";
    bg.type = "color3";
    bg.nodeIndex = 1;
    mixNode.inputs.push_back(bg);
    ProceduralNodeInput mixVal;
    mixVal.name = "mix";
    mixVal.type = "float";
    mixVal.value = std::make_shared<ProceduralValue>("float");
    mixVal.value->values = {0.5f};
    mixNode.inputs.push_back(mixVal);
    mixNode.outputs.push_back({"out", "color3"});
    graph.nodes.push_back(mixNode);

    ProceduralGraphOutput gout;
    gout.name = "result";
    gout.type = "color3";
    gout.nodeIndex = 2;
    graph.outputs.push_back(gout);

    auto dummyAsset = GLTF::loadGLTF(R"({"asset":{"version":"2.0"}})", kNoOpLoader);
    auto pixels = systems::leal::campello_renderer::bakeProceduralTexture(
        *dummyAsset, graph, "result", 4, 4);

    ASSERT_EQ(pixels.size(), 4u * 4u * 4u);
    size_t idx = (2 * 4 + 2) * 4;
    EXPECT_EQ(pixels[idx + 0], 127); // 0.5 * 255 = 127.5 → 127
    EXPECT_EQ(pixels[idx + 1], 0);
    EXPECT_EQ(pixels[idx + 2], 127); // 0.5 * 255 = 127.5 → 127
    EXPECT_EQ(pixels[idx + 3], 255);
}

TEST(ProceduralBakerTest, Noise2DNode) {
    using namespace systems::leal::gltf;
    ProceduralGraph graph;
    graph.name = "noise_test";
    graph.nodetype = "nodegraph";
    graph.type = "float";

    // Node 0: texcoord
    ProceduralNode texNode;
    texNode.name = "uv";
    texNode.nodetype = "texcoord";
    texNode.type = "vector2";
    texNode.outputs.push_back({"out", "vector2"});
    graph.nodes.push_back(texNode);

    // Node 1: noise2d
    ProceduralNode noiseNode;
    noiseNode.name = "noise";
    noiseNode.nodetype = "noise2d";
    noiseNode.type = "float";
    ProceduralNodeInput posIn;
    posIn.name = "position";
    posIn.type = "vector2";
    posIn.nodeIndex = 0;
    noiseNode.inputs.push_back(posIn);
    noiseNode.outputs.push_back({"out", "float"});
    graph.nodes.push_back(noiseNode);

    ProceduralGraphOutput gout;
    gout.name = "result";
    gout.type = "float";
    gout.nodeIndex = 1;
    graph.outputs.push_back(gout);

    auto dummyAsset = GLTF::loadGLTF(R"({"asset":{"version":"2.0"}})", kNoOpLoader);
    auto pixels = systems::leal::campello_renderer::bakeProceduralTexture(
        *dummyAsset, graph, "result", 8, 8);

    ASSERT_EQ(pixels.size(), 8u * 8u * 4u);
    // Verify not all magenta and has variation
    bool hasNonMagenta = false;
    bool hasVariation = false;
    uint8_t firstR = pixels[0];
    for (size_t i = 0; i < pixels.size(); i += 4) {
        if (pixels[i] != 255 || pixels[i+1] != 0 || pixels[i+2] != 255) {
            hasNonMagenta = true;
        }
        if (pixels[i] != firstR) hasVariation = true;
    }
    EXPECT_TRUE(hasNonMagenta);
    EXPECT_TRUE(hasVariation);
}

TEST(ProceduralBakerTest, CheckerboardNode) {
    using namespace systems::leal::gltf;
    ProceduralGraph graph;
    graph.name = "checker_test";
    graph.nodetype = "nodegraph";
    graph.type = "color3";

    // Node 0: texcoord
    ProceduralNode texNode;
    texNode.name = "uv";
    texNode.nodetype = "texcoord";
    texNode.type = "vector2";
    texNode.outputs.push_back({"out", "vector2"});
    graph.nodes.push_back(texNode);

    // Node 1: uvtiling constant 2,2
    ProceduralNode tilNode;
    tilNode.name = "tiling";
    tilNode.nodetype = "constant";
    tilNode.type = "vector2";
    ProceduralNodeInput tilVal;
    tilVal.name = "value";
    tilVal.type = "vector2";
    tilVal.value = std::make_shared<ProceduralValue>("vector2");
    tilVal.value->values = {2.0f, 2.0f};
    tilNode.inputs.push_back(tilVal);
    tilNode.outputs.push_back({"out", "vector2"});
    graph.nodes.push_back(tilNode);

    // Node 2: checkerboard
    ProceduralNode checkNode;
    checkNode.name = "checker";
    checkNode.nodetype = "checkerboard";
    checkNode.type = "color3";
    ProceduralNodeInput tcIn;
    tcIn.name = "texcoord";
    tcIn.type = "vector2";
    tcIn.nodeIndex = 0;
    checkNode.inputs.push_back(tcIn);
    ProceduralNodeInput tilIn;
    tilIn.name = "uvtiling";
    tilIn.type = "vector2";
    tilIn.nodeIndex = 1;
    checkNode.inputs.push_back(tilIn);
    checkNode.outputs.push_back({"out", "color3"});
    graph.nodes.push_back(checkNode);

    ProceduralGraphOutput gout;
    gout.name = "result";
    gout.type = "color3";
    gout.nodeIndex = 2;
    graph.outputs.push_back(gout);

    auto dummyAsset = GLTF::loadGLTF(R"({"asset":{"version":"2.0"}})", kNoOpLoader);
    auto pixels = systems::leal::campello_renderer::bakeProceduralTexture(
        *dummyAsset, graph, "result", 4, 4);

    ASSERT_EQ(pixels.size(), 4u * 4u * 4u);
    // With 2x2 tiling on 4x4, top-left quadrant should be white (color1 default),
    // top-right and bottom-left should be black (color2 default).
    size_t tl = (0 * 4 + 0) * 4; // white
    size_t tr = (0 * 4 + 2) * 4; // black
    size_t bl = (2 * 4 + 0) * 4; // black
    size_t br = (2 * 4 + 2) * 4; // white
    EXPECT_EQ(pixels[tl + 0], 255);
    EXPECT_EQ(pixels[tl + 1], 255);
    EXPECT_EQ(pixels[tl + 2], 255);
    EXPECT_EQ(pixels[tr + 0], 0);
    EXPECT_EQ(pixels[tr + 1], 0);
    EXPECT_EQ(pixels[tr + 2], 0);
    EXPECT_EQ(pixels[bl + 0], 0);
    EXPECT_EQ(pixels[bl + 1], 0);
    EXPECT_EQ(pixels[bl + 2], 0);
    EXPECT_EQ(pixels[br + 0], 255);
    EXPECT_EQ(pixels[br + 1], 255);
    EXPECT_EQ(pixels[br + 2], 255);
}

TEST(ProceduralBakerTest, SinNode) {
    using namespace systems::leal::gltf;
    ProceduralGraph graph;
    graph.name = "sin_test";
    graph.nodetype = "nodegraph";
    graph.type = "float";

    // Node 0: constant pi/2
    ProceduralNode valNode;
    valNode.name = "val";
    valNode.nodetype = "constant";
    valNode.type = "float";
    ProceduralNodeInput valIn;
    valIn.name = "value";
    valIn.type = "float";
    valIn.value = std::make_shared<ProceduralValue>("float");
    valIn.value->values = {1.57079633f};
    valNode.inputs.push_back(valIn);
    valNode.outputs.push_back({"out", "float"});
    graph.nodes.push_back(valNode);

    // Node 1: sin
    ProceduralNode sinNode;
    sinNode.name = "sine";
    sinNode.nodetype = "sin";
    sinNode.type = "float";
    ProceduralNodeInput in;
    in.name = "in";
    in.type = "float";
    in.nodeIndex = 0;
    sinNode.inputs.push_back(in);
    sinNode.outputs.push_back({"out", "float"});
    graph.nodes.push_back(sinNode);

    ProceduralGraphOutput gout;
    gout.name = "result";
    gout.type = "float";
    gout.nodeIndex = 1;
    graph.outputs.push_back(gout);

    auto dummyAsset = GLTF::loadGLTF(R"({"asset":{"version":"2.0"}})", kNoOpLoader);
    auto pixels = systems::leal::campello_renderer::bakeProceduralTexture(
        *dummyAsset, graph, "result", 4, 4);

    ASSERT_EQ(pixels.size(), 4u * 4u * 4u);
    size_t idx = (2 * 4 + 2) * 4;
    EXPECT_EQ(pixels[idx + 0], 255); // sin(pi/2) = 1.0
    EXPECT_EQ(pixels[idx + 1], 0);
    EXPECT_EQ(pixels[idx + 2], 0);
    EXPECT_EQ(pixels[idx + 3], 255);
}

TEST(ProceduralBakerTest, Place2DNode) {
    using namespace systems::leal::gltf;
    ProceduralGraph graph;
    graph.name = "place2d_test";
    graph.nodetype = "nodegraph";
    graph.type = "vector2";

    // Node 0: texcoord
    ProceduralNode texNode;
    texNode.name = "uv";
    texNode.nodetype = "texcoord";
    texNode.type = "vector2";
    texNode.outputs.push_back({"out", "vector2"});
    graph.nodes.push_back(texNode);

    // Node 1: offset constant 0.25, 0.25
    ProceduralNode offNode;
    offNode.name = "off";
    offNode.nodetype = "constant";
    offNode.type = "vector2";
    ProceduralNodeInput offVal;
    offVal.name = "value";
    offVal.type = "vector2";
    offVal.value = std::make_shared<ProceduralValue>("vector2");
    offVal.value->values = {0.25f, 0.25f};
    offNode.inputs.push_back(offVal);
    offNode.outputs.push_back({"out", "vector2"});
    graph.nodes.push_back(offNode);

    // Node 2: place2d
    ProceduralNode placeNode;
    placeNode.name = "place";
    placeNode.nodetype = "place2d";
    placeNode.type = "vector2";
    ProceduralNodeInput tcIn;
    tcIn.name = "texcoord";
    tcIn.type = "vector2";
    tcIn.nodeIndex = 0;
    placeNode.inputs.push_back(tcIn);
    ProceduralNodeInput offIn;
    offIn.name = "offset";
    offIn.type = "vector2";
    offIn.nodeIndex = 1;
    placeNode.inputs.push_back(offIn);
    placeNode.outputs.push_back({"out", "vector2"});
    graph.nodes.push_back(placeNode);

    ProceduralGraphOutput gout;
    gout.name = "result";
    gout.type = "vector2";
    gout.nodeIndex = 2;
    graph.outputs.push_back(gout);

    auto dummyAsset = GLTF::loadGLTF(R"({"asset":{"version":"2.0"}})", kNoOpLoader);
    auto pixels = systems::leal::campello_renderer::bakeProceduralTexture(
        *dummyAsset, graph, "result", 4, 4);

    ASSERT_EQ(pixels.size(), 4u * 4u * 4u);
    // Center pixel (x=2,y=2) uv = (2.5/4, 2.5/4) = (0.625,0.625)
    // place2d: (0.625,0.625) - pivot(0.5,0.5) + pivot(0.5,0.5) + offset(0.25,0.25) = (0.875,0.875)
    size_t idx = (2 * 4 + 2) * 4;
    EXPECT_EQ(pixels[idx + 0], 223); // R = 0.875 * 255 = 223.125 -> 223
    EXPECT_EQ(pixels[idx + 1], 223); // G = 0.875 * 255
    EXPECT_EQ(pixels[idx + 2], 0);   // B unused for vector2
    EXPECT_EQ(pixels[idx + 3], 255);
}

TEST(ProceduralBakerTest, ImageNodeFileNotFound) {
    using namespace systems::leal::gltf;
    ProceduralGraph graph;
    graph.name = "image_missing";
    graph.nodetype = "nodegraph";
    graph.type = "color3";

    // Node 0: image with missing file and green default
    ProceduralNode imgNode;
    imgNode.name = "img";
    imgNode.nodetype = "image";
    imgNode.type = "color3";
    ProceduralNodeInput fileIn;
    fileIn.name = "file";
    fileIn.type = "filename";
    fileIn.value = std::make_shared<ProceduralValue>("filename");
    fileIn.value->stringValue = "nonexistent_file.png";
    imgNode.inputs.push_back(fileIn);
    ProceduralNodeInput defIn;
    defIn.name = "default";
    defIn.type = "color3";
    defIn.value = std::make_shared<ProceduralValue>("color3");
    defIn.value->values = {0.0f, 1.0f, 0.0f};
    imgNode.inputs.push_back(defIn);
    imgNode.outputs.push_back({"out", "color3"});
    graph.nodes.push_back(imgNode);

    ProceduralGraphOutput gout;
    gout.name = "result";
    gout.type = "color3";
    gout.nodeIndex = 0;
    graph.outputs.push_back(gout);

    auto dummyAsset = GLTF::loadGLTF(R"({"asset":{"version":"2.0"}})", kNoOpLoader);
    auto pixels = systems::leal::campello_renderer::bakeProceduralTexture(
        *dummyAsset, graph, "result", 4, 4);

    ASSERT_EQ(pixels.size(), 4u * 4u * 4u);
    size_t idx = (2 * 4 + 2) * 4;
    EXPECT_EQ(pixels[idx + 0], 0);
    EXPECT_EQ(pixels[idx + 1], 255);
    EXPECT_EQ(pixels[idx + 2], 0);
    EXPECT_EQ(pixels[idx + 3], 255);
}

TEST(ProceduralBakerTest, ImageNode) {
    using namespace systems::leal::gltf;

    // Write a 1x1 red TGA test image
    {
        std::vector<uint8_t> tga = {
            0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 32, 8,
            0, 0, 255, 255
        };
        std::ofstream f("test_image.tga", std::ios::binary);
        ASSERT_TRUE(f.is_open());
        f.write(reinterpret_cast<const char*>(tga.data()), tga.size());
    }

    ProceduralGraph graph;
    graph.name = "image_test";
    graph.nodetype = "nodegraph";
    graph.type = "color3";

    // Node 0: texcoord
    ProceduralNode texNode;
    texNode.name = "uv";
    texNode.nodetype = "texcoord";
    texNode.type = "vector2";
    texNode.outputs.push_back({"out", "vector2"});
    graph.nodes.push_back(texNode);

    // Node 1: image
    ProceduralNode imgNode;
    imgNode.name = "img";
    imgNode.nodetype = "image";
    imgNode.type = "color3";
    ProceduralNodeInput fileIn;
    fileIn.name = "file";
    fileIn.type = "filename";
    fileIn.value = std::make_shared<ProceduralValue>("filename");
    fileIn.value->stringValue = "test_image.tga";
    imgNode.inputs.push_back(fileIn);
    ProceduralNodeInput tcIn;
    tcIn.name = "texcoord";
    tcIn.type = "vector2";
    tcIn.nodeIndex = 0;
    imgNode.inputs.push_back(tcIn);
    imgNode.outputs.push_back({"out", "color3"});
    graph.nodes.push_back(imgNode);

    ProceduralGraphOutput gout;
    gout.name = "result";
    gout.type = "color3";
    gout.nodeIndex = 1;
    graph.outputs.push_back(gout);

    auto dummyAsset = GLTF::loadGLTF(R"({"asset":{"version":"2.0"}})", kNoOpLoader);
    auto pixels = systems::leal::campello_renderer::bakeProceduralTexture(
        *dummyAsset, graph, "result", 4, 4);

    ASSERT_EQ(pixels.size(), 4u * 4u * 4u);
    // All pixels should be red since the source image is 1x1 red
    for (size_t i = 0; i < pixels.size(); i += 4) {
        EXPECT_EQ(pixels[i + 0], 255);
        EXPECT_EQ(pixels[i + 1], 0);
        EXPECT_EQ(pixels[i + 2], 0);
        EXPECT_EQ(pixels[i + 3], 255);
    }
}

TEST(ProceduralBakerTest, SwizzleNode) {
    using namespace systems::leal::gltf;
    ProceduralGraph graph;
    graph.name = "swizzle_test";
    graph.nodetype = "nodegraph";
    graph.type = "color3";

    // Node 0: constant color3 (1, 0.5, 0.25)
    ProceduralNode constNode;
    constNode.name = "c";
    constNode.nodetype = "constant";
    constNode.type = "color3";
    ProceduralNodeInput valIn;
    valIn.name = "value";
    valIn.type = "color3";
    valIn.value = std::make_shared<ProceduralValue>("color3");
    valIn.value->values = {1.0f, 0.5f, 0.25f};
    constNode.inputs.push_back(valIn);
    constNode.outputs.push_back({"out", "color3"});
    graph.nodes.push_back(constNode);

    // Node 1: swizzle channels = "bbr"
    ProceduralNode swizNode;
    swizNode.name = "swiz";
    swizNode.nodetype = "swizzle";
    swizNode.type = "color3";
    ProceduralNodeInput in;
    in.name = "in";
    in.type = "color3";
    in.nodeIndex = 0;
    swizNode.inputs.push_back(in);
    ProceduralNodeInput chIn;
    chIn.name = "channels";
    chIn.type = "string";
    chIn.value = std::make_shared<ProceduralValue>("string");
    chIn.value->stringValue = "bbr";
    swizNode.inputs.push_back(chIn);
    swizNode.outputs.push_back({"out", "color3"});
    graph.nodes.push_back(swizNode);

    ProceduralGraphOutput gout;
    gout.name = "result";
    gout.type = "color3";
    gout.nodeIndex = 1;
    graph.outputs.push_back(gout);

    auto dummyAsset = GLTF::loadGLTF(R"({"asset":{"version":"2.0"}})", kNoOpLoader);
    auto pixels = systems::leal::campello_renderer::bakeProceduralTexture(
        *dummyAsset, graph, "result", 4, 4);

    ASSERT_EQ(pixels.size(), 4u * 4u * 4u);
    size_t idx = (2 * 4 + 2) * 4;
    // bbr: blue=0.25, blue=0.25, red=1.0 -> (63, 63, 255) because 0.25*255=63.75 truncates to 63
    EXPECT_EQ(pixels[idx + 0], 63);
    EXPECT_EQ(pixels[idx + 1], 63);
    EXPECT_EQ(pixels[idx + 2], 255);
    EXPECT_EQ(pixels[idx + 3], 255);
}

TEST(ProceduralBakerTest, CombineNode) {
    using namespace systems::leal::gltf;
    ProceduralGraph graph;
    graph.name = "combine_test";
    graph.nodetype = "nodegraph";
    graph.type = "color3";

    // Node 0: constant 1.0
    ProceduralNode n1;
    n1.name = "one";
    n1.nodetype = "constant";
    n1.type = "float";
    ProceduralNodeInput v1;
    v1.name = "value";
    v1.type = "float";
    v1.value = std::make_shared<ProceduralValue>("float");
    v1.value->values = {1.0f};
    n1.inputs.push_back(v1);
    n1.outputs.push_back({"out", "float"});
    graph.nodes.push_back(n1);

    // Node 1: constant 0.0
    ProceduralNode n2;
    n2.name = "zero";
    n2.nodetype = "constant";
    n2.type = "float";
    ProceduralNodeInput v2;
    v2.name = "value";
    v2.type = "float";
    v2.value = std::make_shared<ProceduralValue>("float");
    v2.value->values = {0.0f};
    n2.inputs.push_back(v2);
    n2.outputs.push_back({"out", "float"});
    graph.nodes.push_back(n2);

    // Node 2: combine (1, 0, 0) -> red
    ProceduralNode combNode;
    combNode.name = "comb";
    combNode.nodetype = "combine";
    combNode.type = "color3";
    ProceduralNodeInput c1;
    c1.name = "in1";
    c1.type = "float";
    c1.nodeIndex = 0;
    combNode.inputs.push_back(c1);
    ProceduralNodeInput c2;
    c2.name = "in2";
    c2.type = "float";
    c2.nodeIndex = 1;
    combNode.inputs.push_back(c2);
    ProceduralNodeInput c3;
    c3.name = "in3";
    c3.type = "float";
    c3.nodeIndex = 1;
    combNode.inputs.push_back(c3);
    combNode.outputs.push_back({"out", "color3"});
    graph.nodes.push_back(combNode);

    ProceduralGraphOutput gout;
    gout.name = "result";
    gout.type = "color3";
    gout.nodeIndex = 2;
    graph.outputs.push_back(gout);

    auto dummyAsset = GLTF::loadGLTF(R"({"asset":{"version":"2.0"}})", kNoOpLoader);
    auto pixels = systems::leal::campello_renderer::bakeProceduralTexture(
        *dummyAsset, graph, "result", 4, 4);

    ASSERT_EQ(pixels.size(), 4u * 4u * 4u);
    size_t idx = (2 * 4 + 2) * 4;
    EXPECT_EQ(pixels[idx + 0], 255);
    EXPECT_EQ(pixels[idx + 1], 0);
    EXPECT_EQ(pixels[idx + 2], 0);
    EXPECT_EQ(pixels[idx + 3], 255);
}

TEST(ProceduralBakerTest, ExtractNode) {
    using namespace systems::leal::gltf;
    ProceduralGraph graph;
    graph.name = "extract_test";
    graph.nodetype = "nodegraph";
    graph.type = "float";

    // Node 0: constant color3 (1, 0.5, 0.25)
    ProceduralNode constNode;
    constNode.name = "c";
    constNode.nodetype = "constant";
    constNode.type = "color3";
    ProceduralNodeInput valIn;
    valIn.name = "value";
    valIn.type = "color3";
    valIn.value = std::make_shared<ProceduralValue>("color3");
    valIn.value->values = {1.0f, 0.5f, 0.25f};
    constNode.inputs.push_back(valIn);
    constNode.outputs.push_back({"out", "color3"});
    graph.nodes.push_back(constNode);

    // Node 1: extract index 1 -> 0.5
    ProceduralNode extNode;
    extNode.name = "ext";
    extNode.nodetype = "extract";
    extNode.type = "float";
    ProceduralNodeInput in;
    in.name = "in";
    in.type = "color3";
    in.nodeIndex = 0;
    extNode.inputs.push_back(in);
    ProceduralNodeInput idxIn;
    idxIn.name = "index";
    idxIn.type = "integer";
    idxIn.value = std::make_shared<ProceduralValue>("integer");
    idxIn.value->values = {1.0f};
    extNode.inputs.push_back(idxIn);
    extNode.outputs.push_back({"out", "float"});
    graph.nodes.push_back(extNode);

    ProceduralGraphOutput gout;
    gout.name = "result";
    gout.type = "float";
    gout.nodeIndex = 1;
    graph.outputs.push_back(gout);

    auto dummyAsset = GLTF::loadGLTF(R"({"asset":{"version":"2.0"}})", kNoOpLoader);
    auto pixels = systems::leal::campello_renderer::bakeProceduralTexture(
        *dummyAsset, graph, "result", 4, 4);

    ASSERT_EQ(pixels.size(), 4u * 4u * 4u);
    size_t idx = (2 * 4 + 2) * 4;
    EXPECT_EQ(pixels[idx + 0], 127); // 0.5 * 255 = 127.5 -> 127
    EXPECT_EQ(pixels[idx + 1], 0);
    EXPECT_EQ(pixels[idx + 2], 0);
    EXPECT_EQ(pixels[idx + 3], 255);
}

TEST(ProceduralBakerTest, IfGreaterNode) {
    using namespace systems::leal::gltf;
    ProceduralGraph graph;
    graph.name = "ifgreater_test";
    graph.nodetype = "nodegraph";
    graph.type = "color3";

    // Node 0: constant red
    ProceduralNode redNode;
    redNode.name = "red";
    redNode.nodetype = "constant";
    redNode.type = "color3";
    ProceduralNodeInput redVal;
    redVal.name = "value";
    redVal.type = "color3";
    redVal.value = std::make_shared<ProceduralValue>("color3");
    redVal.value->values = {1.0f, 0.0f, 0.0f};
    redNode.inputs.push_back(redVal);
    redNode.outputs.push_back({"out", "color3"});
    graph.nodes.push_back(redNode);

    // Node 1: constant blue
    ProceduralNode blueNode;
    blueNode.name = "blue";
    blueNode.nodetype = "constant";
    blueNode.type = "color3";
    ProceduralNodeInput blueVal;
    blueVal.name = "value";
    blueVal.type = "color3";
    blueVal.value = std::make_shared<ProceduralValue>("color3");
    blueVal.value->values = {0.0f, 0.0f, 1.0f};
    blueNode.inputs.push_back(blueVal);
    blueNode.outputs.push_back({"out", "color3"});
    graph.nodes.push_back(blueNode);

    // Node 2: ifgreater (1.0 > 0.5) -> red
    ProceduralNode ifNode;
    ifNode.name = "ifg";
    ifNode.nodetype = "ifgreater";
    ifNode.type = "color3";
    ProceduralNodeInput v1;
    v1.name = "value1";
    v1.type = "float";
    v1.value = std::make_shared<ProceduralValue>("float");
    v1.value->values = {1.0f};
    ifNode.inputs.push_back(v1);
    ProceduralNodeInput v2;
    v2.name = "value2";
    v2.type = "float";
    v2.value = std::make_shared<ProceduralValue>("float");
    v2.value->values = {0.5f};
    ifNode.inputs.push_back(v2);
    ProceduralNodeInput in1;
    in1.name = "in1";
    in1.type = "color3";
    in1.nodeIndex = 0;
    ifNode.inputs.push_back(in1);
    ProceduralNodeInput in2;
    in2.name = "in2";
    in2.type = "color3";
    in2.nodeIndex = 1;
    ifNode.inputs.push_back(in2);
    ifNode.outputs.push_back({"out", "color3"});
    graph.nodes.push_back(ifNode);

    ProceduralGraphOutput gout;
    gout.name = "result";
    gout.type = "color3";
    gout.nodeIndex = 2;
    graph.outputs.push_back(gout);

    auto dummyAsset = GLTF::loadGLTF(R"({"asset":{"version":"2.0"}})", kNoOpLoader);
    auto pixels = systems::leal::campello_renderer::bakeProceduralTexture(
        *dummyAsset, graph, "result", 4, 4);

    ASSERT_EQ(pixels.size(), 4u * 4u * 4u);
    size_t idx = (2 * 4 + 2) * 4;
    EXPECT_EQ(pixels[idx + 0], 255);
    EXPECT_EQ(pixels[idx + 1], 0);
    EXPECT_EQ(pixels[idx + 2], 0);
    EXPECT_EQ(pixels[idx + 3], 255);
}

TEST(ProceduralBakerTest, LengthNode) {
    using namespace systems::leal::gltf;
    ProceduralGraph graph;
    graph.name = "length_test";
    graph.nodetype = "nodegraph";
    graph.type = "float";

    // Node 0: constant vector3 (3, 4, 0)
    ProceduralNode vecNode;
    vecNode.name = "vec";
    vecNode.nodetype = "constant";
    vecNode.type = "vector3";
    ProceduralNodeInput vecVal;
    vecVal.name = "value";
    vecVal.type = "vector3";
    vecVal.value = std::make_shared<ProceduralValue>("vector3");
    vecVal.value->values = {3.0f, 4.0f, 0.0f};
    vecNode.inputs.push_back(vecVal);
    vecNode.outputs.push_back({"out", "vector3"});
    graph.nodes.push_back(vecNode);

    // Node 1: length -> 5.0
    ProceduralNode lenNode;
    lenNode.name = "len";
    lenNode.nodetype = "length";
    lenNode.type = "float";
    ProceduralNodeInput in;
    in.name = "in";
    in.type = "vector3";
    in.nodeIndex = 0;
    lenNode.inputs.push_back(in);
    lenNode.outputs.push_back({"out", "float"});
    graph.nodes.push_back(lenNode);

    ProceduralGraphOutput gout;
    gout.name = "result";
    gout.type = "float";
    gout.nodeIndex = 1;
    graph.outputs.push_back(gout);

    auto dummyAsset = GLTF::loadGLTF(R"({"asset":{"version":"2.0"}})", kNoOpLoader);
    auto pixels = systems::leal::campello_renderer::bakeProceduralTexture(
        *dummyAsset, graph, "result", 4, 4);

    ASSERT_EQ(pixels.size(), 4u * 4u * 4u);
    size_t idx = (2 * 4 + 2) * 4;
    EXPECT_EQ(pixels[idx + 0], 255); // length = 5.0 -> 255
    EXPECT_EQ(pixels[idx + 1], 0);
    EXPECT_EQ(pixels[idx + 2], 0);
    EXPECT_EQ(pixels[idx + 3], 255);
}

TEST(ProceduralBakerTest, DistanceNode) {
    using namespace systems::leal::gltf;
    ProceduralGraph graph;
    graph.name = "distance_test";
    graph.nodetype = "nodegraph";
    graph.type = "float";

    // Node 0: constant (1, 0, 0)
    ProceduralNode v1;
    v1.name = "a";
    v1.nodetype = "constant";
    v1.type = "vector3";
    ProceduralNodeInput v1val;
    v1val.name = "value";
    v1val.type = "vector3";
    v1val.value = std::make_shared<ProceduralValue>("vector3");
    v1val.value->values = {1.0f, 0.0f, 0.0f};
    v1.inputs.push_back(v1val);
    v1.outputs.push_back({"out", "vector3"});
    graph.nodes.push_back(v1);

    // Node 1: constant (0, 1, 0)
    ProceduralNode v2;
    v2.name = "b";
    v2.nodetype = "constant";
    v2.type = "vector3";
    ProceduralNodeInput v2val;
    v2val.name = "value";
    v2val.type = "vector3";
    v2val.value = std::make_shared<ProceduralValue>("vector3");
    v2val.value->values = {0.0f, 1.0f, 0.0f};
    v2.inputs.push_back(v2val);
    v2.outputs.push_back({"out", "vector3"});
    graph.nodes.push_back(v2);

    // Node 2: distance -> sqrt(2) ~ 1.414
    ProceduralNode distNode;
    distNode.name = "dist";
    distNode.nodetype = "distance";
    distNode.type = "float";
    ProceduralNodeInput in1;
    in1.name = "in1";
    in1.type = "vector3";
    in1.nodeIndex = 0;
    distNode.inputs.push_back(in1);
    ProceduralNodeInput in2;
    in2.name = "in2";
    in2.type = "vector3";
    in2.nodeIndex = 1;
    distNode.inputs.push_back(in2);
    distNode.outputs.push_back({"out", "float"});
    graph.nodes.push_back(distNode);

    ProceduralGraphOutput gout;
    gout.name = "result";
    gout.type = "float";
    gout.nodeIndex = 2;
    graph.outputs.push_back(gout);

    auto dummyAsset = GLTF::loadGLTF(R"({"asset":{"version":"2.0"}})", kNoOpLoader);
    auto pixels = systems::leal::campello_renderer::bakeProceduralTexture(
        *dummyAsset, graph, "result", 4, 4);

    ASSERT_EQ(pixels.size(), 4u * 4u * 4u);
    size_t idx = (2 * 4 + 2) * 4;
    // sqrt(2) = 1.4142 -> clamped to 1.0 -> 255
    EXPECT_EQ(pixels[idx + 0], 255);
    EXPECT_EQ(pixels[idx + 1], 0);
    EXPECT_EQ(pixels[idx + 2], 0);
    EXPECT_EQ(pixels[idx + 3], 255);
}

TEST(ProceduralBakerTest, CrossProductNode) {
    using namespace systems::leal::gltf;
    ProceduralGraph graph;
    graph.name = "cross_test";
    graph.nodetype = "nodegraph";
    graph.type = "vector3";

    // Node 0: constant (1, 0, 0)
    ProceduralNode v1;
    v1.name = "a";
    v1.nodetype = "constant";
    v1.type = "vector3";
    ProceduralNodeInput v1val;
    v1val.name = "value";
    v1val.type = "vector3";
    v1val.value = std::make_shared<ProceduralValue>("vector3");
    v1val.value->values = {1.0f, 0.0f, 0.0f};
    v1.inputs.push_back(v1val);
    v1.outputs.push_back({"out", "vector3"});
    graph.nodes.push_back(v1);

    // Node 1: constant (0, 1, 0)
    ProceduralNode v2;
    v2.name = "b";
    v2.nodetype = "constant";
    v2.type = "vector3";
    ProceduralNodeInput v2val;
    v2val.name = "value";
    v2val.type = "vector3";
    v2val.value = std::make_shared<ProceduralValue>("vector3");
    v2val.value->values = {0.0f, 1.0f, 0.0f};
    v2.inputs.push_back(v2val);
    v2.outputs.push_back({"out", "vector3"});
    graph.nodes.push_back(v2);

    // Node 2: crossproduct -> (0, 0, 1)
    ProceduralNode crossNode;
    crossNode.name = "cross";
    crossNode.nodetype = "crossproduct";
    crossNode.type = "vector3";
    ProceduralNodeInput in1;
    in1.name = "in1";
    in1.type = "vector3";
    in1.nodeIndex = 0;
    crossNode.inputs.push_back(in1);
    ProceduralNodeInput in2;
    in2.name = "in2";
    in2.type = "vector3";
    in2.nodeIndex = 1;
    crossNode.inputs.push_back(in2);
    crossNode.outputs.push_back({"out", "vector3"});
    graph.nodes.push_back(crossNode);

    ProceduralGraphOutput gout;
    gout.name = "result";
    gout.type = "vector3";
    gout.nodeIndex = 2;
    graph.outputs.push_back(gout);

    auto dummyAsset = GLTF::loadGLTF(R"({"asset":{"version":"2.0"}})", kNoOpLoader);
    auto pixels = systems::leal::campello_renderer::bakeProceduralTexture(
        *dummyAsset, graph, "result", 4, 4);

    ASSERT_EQ(pixels.size(), 4u * 4u * 4u);
    size_t idx = (2 * 4 + 2) * 4;
    // (0, 0, 1) -> R=0, G=0, B=255
    EXPECT_EQ(pixels[idx + 0], 0);
    EXPECT_EQ(pixels[idx + 1], 0);
    EXPECT_EQ(pixels[idx + 2], 255);
    EXPECT_EQ(pixels[idx + 3], 255);
}

TEST(ProceduralBakerTest, NormalizeNode) {
    using namespace systems::leal::gltf;
    ProceduralGraph graph;
    graph.name = "normalize_test";
    graph.nodetype = "nodegraph";
    graph.type = "vector3";

    // Node 0: constant (1, 1, 0)
    ProceduralNode vecNode;
    vecNode.name = "vec";
    vecNode.nodetype = "constant";
    vecNode.type = "vector3";
    ProceduralNodeInput vecVal;
    vecVal.name = "value";
    vecVal.type = "vector3";
    vecVal.value = std::make_shared<ProceduralValue>("vector3");
    vecVal.value->values = {1.0f, 1.0f, 0.0f};
    vecNode.inputs.push_back(vecVal);
    vecNode.outputs.push_back({"out", "vector3"});
    graph.nodes.push_back(vecNode);

    // Node 1: normalize -> (0.707, 0.707, 0)
    ProceduralNode normNode;
    normNode.name = "norm";
    normNode.nodetype = "normalize";
    normNode.type = "vector3";
    ProceduralNodeInput in;
    in.name = "in";
    in.type = "vector3";
    in.nodeIndex = 0;
    normNode.inputs.push_back(in);
    normNode.outputs.push_back({"out", "vector3"});
    graph.nodes.push_back(normNode);

    ProceduralGraphOutput gout;
    gout.name = "result";
    gout.type = "vector3";
    gout.nodeIndex = 1;
    graph.outputs.push_back(gout);

    auto dummyAsset = GLTF::loadGLTF(R"({"asset":{"version":"2.0"}})", kNoOpLoader);
    auto pixels = systems::leal::campello_renderer::bakeProceduralTexture(
        *dummyAsset, graph, "result", 4, 4);

    ASSERT_EQ(pixels.size(), 4u * 4u * 4u);
    size_t idx = (2 * 4 + 2) * 4;
    // 0.707 * 255 = 180.2 -> 180
    EXPECT_EQ(pixels[idx + 0], 180);
    EXPECT_EQ(pixels[idx + 1], 180);
    EXPECT_EQ(pixels[idx + 2], 0);
    EXPECT_EQ(pixels[idx + 3], 255);
}

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// GPU procedural baker tests — compare GPU output against CPU reference.
// ---------------------------------------------------------------------------

class ProceduralBakerGPUTest : public testing::Test {
protected:
    std::shared_ptr<systems::leal::campello_gpu::Device> device;

    void SetUp() override {
        device = systems::leal::campello_gpu::Device::createDefaultDevice(nullptr);
    }
};

static bool comparePixels(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, int tolerance = 2) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs((int)a[i] - (int)b[i]) > tolerance) return false;
    }
    return true;
}

TEST_F(ProceduralBakerGPUTest, ConstantColor3MatchesCPU) {
    if (!device) GTEST_SKIP() << "No GPU device available";
    using namespace systems::leal::gltf;
    ProceduralGraph graph;
    graph.name = "gpu_const_test";
    graph.nodetype = "nodegraph";
    graph.type = "color3";

    ProceduralNode constNode;
    constNode.name = "c";
    constNode.nodetype = "constant";
    constNode.type = "color3";
    ProceduralNodeInput valIn;
    valIn.name = "value";
    valIn.type = "color3";
    valIn.value = std::make_shared<ProceduralValue>("color3");
    valIn.value->values = {1.0f, 0.5f, 0.25f};
    constNode.inputs.push_back(valIn);
    constNode.outputs.push_back({"out", "color3"});
    graph.nodes.push_back(constNode);

    ProceduralGraphOutput gout;
    gout.name = "result";
    gout.type = "color3";
    gout.nodeIndex = 0;
    graph.outputs.push_back(gout);

    auto dummyAsset = GLTF::loadGLTF(R"({"asset":{"version":"2.0"}})", kNoOpLoader);
    auto cpuPixels = systems::leal::campello_renderer::bakeProceduralTexture(
        *dummyAsset, graph, "result", 4, 4);
    auto gpuPixels = systems::leal::campello_renderer::bakeProceduralTextureGPU(
        device, *dummyAsset, graph, "result", 4, 4);

    EXPECT_TRUE(comparePixels(cpuPixels, gpuPixels));
}

TEST_F(ProceduralBakerGPUTest, MixNodeMatchesCPU) {
    if (!device) GTEST_SKIP() << "No GPU device available";
    using namespace systems::leal::gltf;
    ProceduralGraph graph;
    graph.name = "gpu_mix_test";
    graph.nodetype = "nodegraph";
    graph.type = "color3";

    // Node 0: constant red
    ProceduralNode red;
    red.name = "red";
    red.nodetype = "constant";
    red.type = "color3";
    ProceduralNodeInput redVal;
    redVal.name = "value";
    redVal.type = "color3";
    redVal.value = std::make_shared<ProceduralValue>("color3");
    redVal.value->values = {1.0f, 0.0f, 0.0f};
    red.inputs.push_back(redVal);
    red.outputs.push_back({"out", "color3"});
    graph.nodes.push_back(red);

    // Node 1: constant blue
    ProceduralNode blue;
    blue.name = "blue";
    blue.nodetype = "constant";
    blue.type = "color3";
    ProceduralNodeInput blueVal;
    blueVal.name = "value";
    blueVal.type = "color3";
    blueVal.value = std::make_shared<ProceduralValue>("color3");
    blueVal.value->values = {0.0f, 0.0f, 1.0f};
    blue.inputs.push_back(blueVal);
    blue.outputs.push_back({"out", "color3"});
    graph.nodes.push_back(blue);

    // Node 2: mix(0.5)
    ProceduralNode mixNode;
    mixNode.name = "mix";
    mixNode.nodetype = "mix";
    mixNode.type = "color3";
    ProceduralNodeInput fg;
    fg.name = "fg";
    fg.type = "color3";
    fg.nodeIndex = 0;
    mixNode.inputs.push_back(fg);
    ProceduralNodeInput bg;
    bg.name = "bg";
    bg.type = "color3";
    bg.nodeIndex = 1;
    mixNode.inputs.push_back(bg);
    ProceduralNodeInput mixVal;
    mixVal.name = "mix";
    mixVal.type = "float";
    mixVal.value = std::make_shared<ProceduralValue>("float");
    mixVal.value->values = {0.5f};
    mixNode.inputs.push_back(mixVal);
    mixNode.outputs.push_back({"out", "color3"});
    graph.nodes.push_back(mixNode);

    ProceduralGraphOutput gout;
    gout.name = "result";
    gout.type = "color3";
    gout.nodeIndex = 2;
    graph.outputs.push_back(gout);

    auto dummyAsset = GLTF::loadGLTF(R"({"asset":{"version":"2.0"}})", kNoOpLoader);
    auto cpuPixels = systems::leal::campello_renderer::bakeProceduralTexture(
        *dummyAsset, graph, "result", 4, 4);
    auto gpuPixels = systems::leal::campello_renderer::bakeProceduralTextureGPU(
        device, *dummyAsset, graph, "result", 4, 4);

    EXPECT_TRUE(comparePixels(cpuPixels, gpuPixels));
}

TEST_F(ProceduralBakerGPUTest, Noise2DNodeMatchesCPU) {
    if (!device) GTEST_SKIP() << "No GPU device available";
    using namespace systems::leal::gltf;
    ProceduralGraph graph;
    graph.name = "gpu_noise_test";
    graph.nodetype = "nodegraph";
    graph.type = "float";

    // Node 0: texcoord
    ProceduralNode tc;
    tc.name = "tc";
    tc.nodetype = "texcoord";
    tc.type = "vector2";
    tc.outputs.push_back({"out", "vector2"});
    graph.nodes.push_back(tc);

    // Node 1: noise2d
    ProceduralNode noise;
    noise.name = "noise";
    noise.nodetype = "noise2d";
    noise.type = "float";
    ProceduralNodeInput posIn;
    posIn.name = "position";
    posIn.type = "vector2";
    posIn.nodeIndex = 0;
    noise.inputs.push_back(posIn);
    noise.outputs.push_back({"out", "float"});
    graph.nodes.push_back(noise);

    ProceduralGraphOutput gout;
    gout.name = "result";
    gout.type = "float";
    gout.nodeIndex = 1;
    graph.outputs.push_back(gout);

    auto dummyAsset = GLTF::loadGLTF(R"({"asset":{"version":"2.0"}})", kNoOpLoader);
    auto cpuPixels = systems::leal::campello_renderer::bakeProceduralTexture(
        *dummyAsset, graph, "result", 8, 8);
    auto gpuPixels = systems::leal::campello_renderer::bakeProceduralTextureGPU(
        device, *dummyAsset, graph, "result", 8, 8);

    EXPECT_TRUE(comparePixels(cpuPixels, gpuPixels, 3));
}

TEST_F(ProceduralBakerGPUTest, CheckerboardNodeMatchesCPU) {
    if (!device) GTEST_SKIP() << "No GPU device available";
    using namespace systems::leal::gltf;
    ProceduralGraph graph;
    graph.name = "gpu_checker_test";
    graph.nodetype = "nodegraph";
    graph.type = "color3";

    // Node 0: texcoord
    ProceduralNode tc;
    tc.name = "tc";
    tc.nodetype = "texcoord";
    tc.type = "vector2";
    tc.outputs.push_back({"out", "vector2"});
    graph.nodes.push_back(tc);

    // Node 1: checkerboard
    ProceduralNode cb;
    cb.name = "cb";
    cb.nodetype = "checkerboard";
    cb.type = "color3";
    ProceduralNodeInput texIn;
    texIn.name = "texcoord";
    texIn.type = "vector2";
    texIn.nodeIndex = 0;
    cb.inputs.push_back(texIn);
    cb.outputs.push_back({"out", "color3"});
    graph.nodes.push_back(cb);

    ProceduralGraphOutput gout;
    gout.name = "result";
    gout.type = "color3";
    gout.nodeIndex = 1;
    graph.outputs.push_back(gout);

    auto dummyAsset = GLTF::loadGLTF(R"({"asset":{"version":"2.0"}})", kNoOpLoader);
    auto cpuPixels = systems::leal::campello_renderer::bakeProceduralTexture(
        *dummyAsset, graph, "result", 4, 4);
    auto gpuPixels = systems::leal::campello_renderer::bakeProceduralTextureGPU(
        device, *dummyAsset, graph, "result", 4, 4);

    EXPECT_TRUE(comparePixels(cpuPixels, gpuPixels));
}

TEST_F(ProceduralBakerGPUTest, SwizzleNodeMatchesCPU) {
    if (!device) GTEST_SKIP() << "No GPU device available";
    using namespace systems::leal::gltf;
    ProceduralGraph graph;
    graph.name = "gpu_swizzle_test";
    graph.nodetype = "nodegraph";
    graph.type = "color3";

    // Node 0: constant color3 (1, 0.5, 0.25)
    ProceduralNode constNode;
    constNode.name = "c";
    constNode.nodetype = "constant";
    constNode.type = "color3";
    ProceduralNodeInput valIn;
    valIn.name = "value";
    valIn.type = "color3";
    valIn.value = std::make_shared<ProceduralValue>("color3");
    valIn.value->values = {1.0f, 0.5f, 0.25f};
    constNode.inputs.push_back(valIn);
    constNode.outputs.push_back({"out", "color3"});
    graph.nodes.push_back(constNode);

    // Node 1: swizzle "bbr"
    ProceduralNode swizNode;
    swizNode.name = "swiz";
    swizNode.nodetype = "swizzle";
    swizNode.type = "color3";
    ProceduralNodeInput in;
    in.name = "in";
    in.type = "color3";
    in.nodeIndex = 0;
    swizNode.inputs.push_back(in);
    ProceduralNodeInput chIn;
    chIn.name = "channels";
    chIn.type = "string";
    chIn.value = std::make_shared<ProceduralValue>("string");
    chIn.value->stringValue = "bbr";
    swizNode.inputs.push_back(chIn);
    swizNode.outputs.push_back({"out", "color3"});
    graph.nodes.push_back(swizNode);

    ProceduralGraphOutput gout;
    gout.name = "result";
    gout.type = "color3";
    gout.nodeIndex = 1;
    graph.outputs.push_back(gout);

    auto dummyAsset = GLTF::loadGLTF(R"({"asset":{"version":"2.0"}})", kNoOpLoader);
    auto cpuPixels = systems::leal::campello_renderer::bakeProceduralTexture(
        *dummyAsset, graph, "result", 4, 4);
    auto gpuPixels = systems::leal::campello_renderer::bakeProceduralTextureGPU(
        device, *dummyAsset, graph, "result", 4, 4);

    EXPECT_TRUE(comparePixels(cpuPixels, gpuPixels));
}

TEST_F(ProceduralBakerGPUTest, IfGreaterNodeMatchesCPU) {
    if (!device) GTEST_SKIP() << "No GPU device available";
    using namespace systems::leal::gltf;
    ProceduralGraph graph;
    graph.name = "gpu_if_test";
    graph.nodetype = "nodegraph";
    graph.type = "color3";

    // Node 0: constant 0.5
    ProceduralNode half;
    half.name = "half";
    half.nodetype = "constant";
    half.type = "float";
    ProceduralNodeInput halfVal;
    halfVal.name = "value";
    halfVal.type = "float";
    halfVal.value = std::make_shared<ProceduralValue>("float");
    halfVal.value->values = {0.5f};
    half.inputs.push_back(halfVal);
    half.outputs.push_back({"out", "float"});
    graph.nodes.push_back(half);

    // Node 1: constant red
    ProceduralNode red;
    red.name = "red";
    red.nodetype = "constant";
    red.type = "color3";
    ProceduralNodeInput redVal;
    redVal.name = "value";
    redVal.type = "color3";
    redVal.value = std::make_shared<ProceduralValue>("color3");
    redVal.value->values = {1.0f, 0.0f, 0.0f};
    red.inputs.push_back(redVal);
    red.outputs.push_back({"out", "color3"});
    graph.nodes.push_back(red);

    // Node 2: constant green
    ProceduralNode green;
    green.name = "green";
    green.nodetype = "constant";
    green.type = "color3";
    ProceduralNodeInput greenVal;
    greenVal.name = "value";
    greenVal.type = "color3";
    greenVal.value = std::make_shared<ProceduralValue>("color3");
    greenVal.value->values = {0.0f, 1.0f, 0.0f};
    green.inputs.push_back(greenVal);
    green.outputs.push_back({"out", "color3"});
    graph.nodes.push_back(green);

    // Node 3: ifgreater(0.5 > 0.0) -> red
    ProceduralNode ifNode;
    ifNode.name = "if";
    ifNode.nodetype = "ifgreater";
    ifNode.type = "color3";
    ProceduralNodeInput v1;
    v1.name = "value1";
    v1.type = "float";
    v1.nodeIndex = 0;
    ifNode.inputs.push_back(v1);
    ProceduralNodeInput v2;
    v2.name = "value2";
    v2.type = "float";
    v2.value = std::make_shared<ProceduralValue>("float");
    v2.value->values = {0.0f};
    ifNode.inputs.push_back(v2);
    ProceduralNodeInput in1;
    in1.name = "in1";
    in1.type = "color3";
    in1.nodeIndex = 1;
    ifNode.inputs.push_back(in1);
    ProceduralNodeInput in2;
    in2.name = "in2";
    in2.type = "color3";
    in2.nodeIndex = 2;
    ifNode.inputs.push_back(in2);
    ifNode.outputs.push_back({"out", "color3"});
    graph.nodes.push_back(ifNode);

    ProceduralGraphOutput gout;
    gout.name = "result";
    gout.type = "color3";
    gout.nodeIndex = 3;
    graph.outputs.push_back(gout);

    auto dummyAsset = GLTF::loadGLTF(R"({"asset":{"version":"2.0"}})", kNoOpLoader);
    auto cpuPixels = systems::leal::campello_renderer::bakeProceduralTexture(
        *dummyAsset, graph, "result", 4, 4);
    auto gpuPixels = systems::leal::campello_renderer::bakeProceduralTextureGPU(
        device, *dummyAsset, graph, "result", 4, 4);

    EXPECT_TRUE(comparePixels(cpuPixels, gpuPixels));
}

TEST_F(ProceduralBakerGPUTest, ImageNodeFallsBackToCPU) {
    if (!device) GTEST_SKIP() << "No GPU device available";
    using namespace systems::leal::gltf;
    ProceduralGraph graph;
    graph.name = "gpu_image_fallback";
    graph.nodetype = "nodegraph";
    graph.type = "color3";

    // Node 0: image (unsupported on GPU)
    ProceduralNode imgNode;
    imgNode.name = "img";
    imgNode.nodetype = "image";
    imgNode.type = "color3";
    ProceduralNodeInput fileIn;
    fileIn.name = "file";
    fileIn.type = "filename";
    fileIn.value = std::make_shared<ProceduralValue>("filename");
    fileIn.value->stringValue = "nonexistent.png";
    imgNode.inputs.push_back(fileIn);
    imgNode.outputs.push_back({"out", "color3"});
    graph.nodes.push_back(imgNode);

    ProceduralGraphOutput gout;
    gout.name = "result";
    gout.type = "color3";
    gout.nodeIndex = 0;
    graph.outputs.push_back(gout);

    auto dummyAsset = GLTF::loadGLTF(R"({"asset":{"version":"2.0"}})", kNoOpLoader);
    // Should not crash and should fall back to CPU (producing magenta for missing image).
    auto gpuPixels = systems::leal::campello_renderer::bakeProceduralTextureGPU(
        device, *dummyAsset, graph, "result", 4, 4);
    EXPECT_EQ(gpuPixels.size(), 4u * 4u * 4u);
}

// ---------------------------------------------------------------------------
// Offscreen rendering tests
// ---------------------------------------------------------------------------

// Minimal glTF with a single triangle using an embedded data-uri buffer.
static const char *kGltfTriangleWithData = R"({
    "asset": {"version": "2.0"},
    "scene": 0,
    "scenes": [{"nodes": [0]}],
    "nodes": [{"mesh": 0}],
    "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
    "accessors": [
        {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
         "max": [0.5, 0.5, 0], "min": [-0.5, -0.5, 0]},
        {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}
    ],
    "bufferViews": [
        {"buffer": 0, "byteOffset": 0, "byteLength": 36},
        {"buffer": 0, "byteOffset": 36, "byteLength": 6}
    ],
    "buffers": [{"uri": "data:application/octet-stream;base64,AAAAvwAAAL8AAAAAAAAAPwAAAL8AAAAAAAAAAAAAAD8AAAAAAAABAAIA", "byteLength": 42}]
})";

class OffscreenRenderTest : public testing::Test {
protected:
    std::shared_ptr<systems::leal::campello_gpu::Device> device;
    std::unique_ptr<Renderer> renderer;

    void SetUp() override {
        device = systems::leal::campello_gpu::Device::createDefaultDevice(nullptr);
        if (device) {
            renderer = std::make_unique<Renderer>(device);
        }
    }

    std::shared_ptr<systems::leal::campello_gpu::Texture> makeOffscreenTexture(
        uint32_t w, uint32_t h,
        systems::leal::campello_gpu::PixelFormat fmt =
            systems::leal::campello_gpu::PixelFormat::rgba8unorm)
    {
        using namespace systems::leal::campello_gpu;
        return device->createTexture(
            TextureType::tt2d, fmt, w, h, 1, 1, 1,
            static_cast<TextureUsage>(
                static_cast<uint32_t>(TextureUsage::renderTarget) |
                static_cast<uint32_t>(TextureUsage::copySrc)));
    }

    std::vector<uint8_t> readBackPixels(
        std::shared_ptr<systems::leal::campello_gpu::Texture> tex)
    {
        using namespace systems::leal::campello_gpu;
        uint32_t w = tex->getWidth();
        uint32_t h = tex->getHeight();
        uint64_t size = w * h * 4; // assume RGBA8
        std::vector<uint8_t> pixels(size);
        device->waitForIdle();
        EXPECT_TRUE(tex->download(0, 0, pixels.data(), size));
        return pixels;
    }
};

TEST_F(OffscreenRenderTest, BasicOffscreenRenderDoesNotCrash) {
    if (!device) GTEST_SKIP() << "No GPU device available";
    auto asset = GLTF::loadGLTF(kGltfOneEmptyScene, kNoOpLoader);
    ASSERT_NE(asset, nullptr);
    renderer->setAsset(asset);
    renderer->resize(64, 64);
    renderer->createDefaultPipelines(systems::leal::campello_gpu::PixelFormat::rgba8unorm);

    auto tex = makeOffscreenTexture(64, 64);
    ASSERT_NE(tex, nullptr);
    auto view = tex->createView(systems::leal::campello_gpu::PixelFormat::rgba8unorm, 1);
    ASSERT_NE(view, nullptr);

    EXPECT_NO_THROW(renderer->render(view));
}

TEST_F(OffscreenRenderTest, ClearColorIsApplied) {
    if (!device) GTEST_SKIP() << "No GPU device available";
    auto asset = GLTF::loadGLTF(kGltfOneEmptyScene, kNoOpLoader);
    ASSERT_NE(asset, nullptr);
    renderer->setAsset(asset);
    renderer->resize(64, 64);
    renderer->createDefaultPipelines(systems::leal::campello_gpu::PixelFormat::rgba8unorm);
    renderer->setClearColor(1.0f, 0.0f, 0.0f, 1.0f);

    auto tex = makeOffscreenTexture(64, 64);
    ASSERT_NE(tex, nullptr);
    auto view = tex->createView(systems::leal::campello_gpu::PixelFormat::rgba8unorm, 1);
    ASSERT_NE(view, nullptr);

    EXPECT_NO_THROW(renderer->render(view));

    auto pixels = readBackPixels(tex);
    ASSERT_EQ(pixels.size(), 64u * 64u * 4u);

    // Sample a few pixels from the center and corners — all should be red.
    for (size_t y : {0u, 31u, 63u}) {
        for (size_t x : {0u, 31u, 63u}) {
            size_t idx = (y * 64 + x) * 4;
            EXPECT_NEAR(pixels[idx + 0], 255, 5) << "R at (" << x << "," << y << ")";
            EXPECT_NEAR(pixels[idx + 1], 0,   5) << "G at (" << x << "," << y << ")";
            EXPECT_NEAR(pixels[idx + 2], 0,   5) << "B at (" << x << "," << y << ")";
            EXPECT_NEAR(pixels[idx + 3], 255, 5) << "A at (" << x << "," << y << ")";
        }
    }
}

TEST_F(OffscreenRenderTest, MultipleConsecutiveRenders) {
    if (!device) GTEST_SKIP() << "No GPU device available";
    auto asset = GLTF::loadGLTF(kGltfOneEmptyScene, kNoOpLoader);
    ASSERT_NE(asset, nullptr);
    renderer->setAsset(asset);
    renderer->resize(64, 64);
    renderer->createDefaultPipelines(systems::leal::campello_gpu::PixelFormat::rgba8unorm);

    auto tex = makeOffscreenTexture(64, 64);
    ASSERT_NE(tex, nullptr);
    auto view = tex->createView(systems::leal::campello_gpu::PixelFormat::rgba8unorm, 1);
    ASSERT_NE(view, nullptr);

    for (int i = 0; i < 10; ++i) {
        EXPECT_NO_THROW(renderer->render(view));
    }
}

TEST_F(OffscreenRenderTest, ResizeOffscreenTarget) {
    if (!device) GTEST_SKIP() << "No GPU device available";
    auto asset = GLTF::loadGLTF(kGltfOneEmptyScene, kNoOpLoader);
    ASSERT_NE(asset, nullptr);
    renderer->setAsset(asset);
    renderer->resize(64, 64);
    renderer->createDefaultPipelines(systems::leal::campello_gpu::PixelFormat::rgba8unorm);

    auto tex64 = makeOffscreenTexture(64, 64);
    ASSERT_NE(tex64, nullptr);
    auto view64 = tex64->createView(systems::leal::campello_gpu::PixelFormat::rgba8unorm, 1);
    ASSERT_NE(view64, nullptr);
    EXPECT_NO_THROW(renderer->render(view64));

    renderer->resize(128, 128);
    auto tex128 = makeOffscreenTexture(128, 128);
    ASSERT_NE(tex128, nullptr);
    auto view128 = tex128->createView(systems::leal::campello_gpu::PixelFormat::rgba8unorm, 1);
    ASSERT_NE(view128, nullptr);
    EXPECT_NO_THROW(renderer->render(view128));
}

TEST_F(OffscreenRenderTest, MeshRendersNonClearPixels) {
    if (!device) GTEST_SKIP() << "No GPU device available";
    auto asset = GLTF::loadGLTF(kGltfTriangleWithData, kNoOpLoader);
    ASSERT_NE(asset, nullptr);
    renderer->setAsset(asset);
    renderer->resize(64, 64);
    renderer->createDefaultPipelines(systems::leal::campello_gpu::PixelFormat::rgba8unorm);
    renderer->setClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    auto tex = makeOffscreenTexture(64, 64);
    ASSERT_NE(tex, nullptr);
    auto view = tex->createView(systems::leal::campello_gpu::PixelFormat::rgba8unorm, 1);
    ASSERT_NE(view, nullptr);

    EXPECT_NO_THROW(renderer->render(view));

    auto pixels = readBackPixels(tex);
    ASSERT_EQ(pixels.size(), 64u * 64u * 4u);

    // Check that at least one pixel is not black (clear color) — i.e. the triangle was drawn.
    bool foundNonBlack = false;
    for (size_t i = 0; i < pixels.size(); i += 4) {
        if (pixels[i] > 20 || pixels[i+1] > 20 || pixels[i+2] > 20) {
            foundNonBlack = true;
            break;
        }
    }
    EXPECT_TRUE(foundNonBlack) << "All pixels are black — mesh did not render";
}

TEST_F(OffscreenRenderTest, ECSPathRendersToOffscreen) {
    if (!device) GTEST_SKIP() << "No GPU device available";
    auto asset = GLTF::loadGLTF(kGltfTriangleWithData, kNoOpLoader);
    ASSERT_NE(asset, nullptr);

    renderer->setAsset(asset);
    renderer->resize(64, 64);
    renderer->createDefaultPipelines(systems::leal::campello_gpu::PixelFormat::rgba8unorm);
    renderer->setClearColor(0.0f, 0.0f, 1.0f, 1.0f); // blue clear

    // Upload mesh and material manually.
    ASSERT_TRUE(asset->meshes && !asset->meshes->empty());
    ASSERT_FALSE(asset->meshes->front().primitives.empty());
    GpuMesh* mesh = renderer->uploadMesh(asset->meshes->front().primitives.front(), *asset);

    namespace GLTF_NS = systems::leal::gltf;
    GLTF_NS::Material defaultMat(
        "", nullptr, nullptr, nullptr, nullptr,
        systems::leal::vector_math::Vector3<double>(0.0, 0.0, 0.0),
        GLTF_NS::AlphaMode::opaque,
        0.5, false, false, 1.0, 1.5, 0.0);
    GpuMaterial* material = renderer->uploadMaterial(defaultMat, *asset);
    ASSERT_NE(mesh, nullptr);
    ASSERT_NE(material, nullptr);

    auto tex = makeOffscreenTexture(64, 64);
    ASSERT_NE(tex, nullptr);
    auto view = tex->createView(systems::leal::campello_gpu::PixelFormat::rgba8unorm, 1);
    ASSERT_NE(view, nullptr);

    namespace VM = systems::leal::vector_math;
    RenderScene scene;
    scene.camera.view = VM::Matrix4<double>::lookAt(
        VM::Vector3<double>(0.0, 0.0, 5.0),
        VM::Vector3<double>(0.0, 0.0, 0.0),
        VM::Vector3<double>(0.0, 1.0, 0.0));
    scene.camera.projection = VM::Matrix4<double>::perspective(
        60.0 * 3.14159265358979323846 / 180.0, 1.0, 0.1, 1000.0);
    scene.camera.position = VM::Vector3<double>(0.0, 0.0, 5.0);

    DrawCall draw;
    draw.mesh = mesh;
    draw.material = material;
    draw.worldTransform = VM::Matrix4<double>::identity();
    draw.instanceCount = 1;
    scene.opaque.push_back(draw);

    EXPECT_NO_THROW(renderer->render(scene, view));

    auto pixels = readBackPixels(tex);
    ASSERT_EQ(pixels.size(), 64u * 64u * 4u);

    // Verify clear color (blue) is present in at least some pixels.
    bool foundBlue = false;
    for (size_t i = 0; i < pixels.size(); i += 4) {
        if (pixels[i+2] > 200 && pixels[i] < 50 && pixels[i+1] < 50) {
            foundBlue = true;
            break;
        }
    }
    EXPECT_TRUE(foundBlue) << "No blue clear pixels found";

    // Verify some non-blue pixels exist (the triangle).
    bool foundNonBlue = false;
    for (size_t i = 0; i < pixels.size(); i += 4) {
        if (pixels[i+2] < 200) {
            foundNonBlue = true;
            break;
        }
    }
    EXPECT_TRUE(foundNonBlue) << "All pixels are blue — mesh did not render via ECS path";
}

TEST_F(OffscreenRenderTest, DifferentPixelFormatBGRA8) {
    if (!device) GTEST_SKIP() << "No GPU device available";
    auto asset = GLTF::loadGLTF(kGltfOneEmptyScene, kNoOpLoader);
    ASSERT_NE(asset, nullptr);
    renderer->setAsset(asset);
    renderer->resize(64, 64);
    renderer->createDefaultPipelines(systems::leal::campello_gpu::PixelFormat::bgra8unorm);

    auto tex = makeOffscreenTexture(64, 64, systems::leal::campello_gpu::PixelFormat::bgra8unorm);
    if (!tex) {
        GTEST_SKIP() << "bgra8unorm not supported on this device";
    }
    auto view = tex->createView(systems::leal::campello_gpu::PixelFormat::bgra8unorm, 1);
    ASSERT_NE(view, nullptr);
    EXPECT_NO_THROW(renderer->render(view));
}

// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
