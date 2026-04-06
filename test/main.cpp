#include <gtest/gtest.h>
#include <campello_renderer/campello_renderer.hpp>
#include <gltf/gltf.hpp>

using namespace systems::leal::campello_renderer;
using namespace systems::leal::gltf;

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

// ---------------------------------------------------------------------------
// Version
// ---------------------------------------------------------------------------

TEST(VersionTest, ReturnsExpectedVersion) {
    EXPECT_EQ(systems::leal::campello_renderer::getVersion(), "0.1.2");
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
    auto asset = GLTF::loadGLTF(kGltfNoScenes);
    ASSERT_NE(asset, nullptr);
    Renderer renderer(nullptr);
    renderer.setAsset(asset);
    EXPECT_EQ(renderer.getAsset(), asset);
}

TEST(SetAssetTest, SetAssetWithOneEmptyScene) {
    auto asset = GLTF::loadGLTF(kGltfOneEmptyScene);
    ASSERT_NE(asset, nullptr);
    Renderer renderer(nullptr);
    EXPECT_NO_THROW(renderer.setAsset(asset));
    EXPECT_EQ(renderer.getAsset(), asset);
}

TEST(SetAssetTest, SetAssetWithTwoScenes) {
    auto asset = GLTF::loadGLTF(kGltfTwoEmptyScenes);
    ASSERT_NE(asset, nullptr);
    Renderer renderer(nullptr);
    EXPECT_NO_THROW(renderer.setAsset(asset));
    EXPECT_EQ(renderer.getAsset(), asset);
}

TEST(SetAssetTest, SetAssetWithCameraNode) {
    auto asset = GLTF::loadGLTF(kGltfWithCamera);
    ASSERT_NE(asset, nullptr);
    Renderer renderer(nullptr);
    EXPECT_NO_THROW(renderer.setAsset(asset));
    EXPECT_EQ(renderer.getAsset(), asset);
}

TEST(SetAssetTest, SetAssetWithMeshNode) {
    auto asset = GLTF::loadGLTF(kGltfWithMesh);
    ASSERT_NE(asset, nullptr);
    Renderer renderer(nullptr);
    EXPECT_NO_THROW(renderer.setAsset(asset));
    EXPECT_EQ(renderer.getAsset(), asset);
}

TEST(SetAssetTest, SetAssetTwiceReplacesAsset) {
    auto asset1 = GLTF::loadGLTF(kGltfOneEmptyScene);
    auto asset2 = GLTF::loadGLTF(kGltfTwoEmptyScenes);
    Renderer renderer(nullptr);
    renderer.setAsset(asset1);
    renderer.setAsset(asset2);
    EXPECT_EQ(renderer.getAsset(), asset2);
}

TEST(SetAssetTest, SetAssetThenClearWithNull) {
    auto asset = GLTF::loadGLTF(kGltfOneEmptyScene);
    Renderer renderer(nullptr);
    renderer.setAsset(asset);
    EXPECT_EQ(renderer.getAsset(), asset);
    renderer.setAsset(nullptr);
    EXPECT_EQ(renderer.getAsset(), nullptr);
}

TEST(SetAssetTest, SetAssetThenClearThenSetAgain) {
    auto asset = GLTF::loadGLTF(kGltfOneEmptyScene);
    Renderer renderer(nullptr);
    renderer.setAsset(asset);
    renderer.setAsset(nullptr);
    renderer.setAsset(asset);
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
    auto asset = GLTF::loadGLTF(kGltfOneEmptyScene);
    Renderer renderer(nullptr);
    renderer.setAsset(asset);
    EXPECT_NO_THROW(renderer.setScene(0));
}

TEST(SetSceneTest, OutOfBoundsSceneIndexDoesNotCrash) {
    auto asset = GLTF::loadGLTF(kGltfOneEmptyScene);
    Renderer renderer(nullptr);
    renderer.setAsset(asset);
    EXPECT_NO_THROW(renderer.setScene(99));
}

TEST(SetSceneTest, SwitchBetweenScenes) {
    auto asset = GLTF::loadGLTF(kGltfTwoEmptyScenes);
    Renderer renderer(nullptr);
    renderer.setAsset(asset);
    EXPECT_NO_THROW(renderer.setScene(0));
    EXPECT_NO_THROW(renderer.setScene(1));
    EXPECT_NO_THROW(renderer.setScene(0));
}

TEST(SetSceneTest, SetSceneAfterClearingAsset) {
    auto asset = GLTF::loadGLTF(kGltfOneEmptyScene);
    Renderer renderer(nullptr);
    renderer.setAsset(asset);
    renderer.setAsset(nullptr);
    EXPECT_NO_THROW(renderer.setScene(0));
}

TEST(SetSceneTest, SetSceneWithMeshNode) {
    auto asset = GLTF::loadGLTF(kGltfWithMesh);
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
    auto asset = GLTF::loadGLTF(kGltfWithCamera);
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
    auto asset = GLTF::loadGLTF(kGltfOneEmptyScene);
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
    auto asset = GLTF::loadGLTF(kGltfWithCamera);
    Renderer renderer(nullptr);
    renderer.setAsset(asset);
    renderer.setScene(0);
    renderer.setCamera(0);
    EXPECT_NO_THROW(renderer.render());
    EXPECT_NO_THROW(renderer.update(0.016));
    EXPECT_EQ(renderer.getAsset(), asset);
}

TEST(SequenceTest, ReplaceAssetAndRender) {
    auto asset1 = GLTF::loadGLTF(kGltfOneEmptyScene);
    auto asset2 = GLTF::loadGLTF(kGltfWithMesh);
    Renderer renderer(nullptr);
    renderer.setAsset(asset1);
    renderer.setScene(0);
    renderer.setAsset(asset2);
    renderer.setScene(0);
    EXPECT_NO_THROW(renderer.render());
    EXPECT_EQ(renderer.getAsset(), asset2);
}

TEST(SequenceTest, SimulateFrameLoop) {
    auto asset = GLTF::loadGLTF(kGltfWithCamera);
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
    auto asset = GLTF::loadGLTF(kGltfOneEmptyScene);
    Renderer renderer(nullptr);
    renderer.setAsset(asset);
    renderer.render();
    renderer.setAsset(nullptr);
    EXPECT_NO_THROW(renderer.render());
    EXPECT_EQ(renderer.getAsset(), nullptr);
}

// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
