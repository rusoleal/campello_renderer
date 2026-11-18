// Frame benchmark: the CPU cost of the renderer's frame paths and the frame
// time they sustain, to compare changes against a baseline.
//
//   campello_renderer_bench [scenario] [frames] [--general] [--no-ibl]
//
// On Android: build it for arm64 (CAMPELLO_RENDERER_BUILD_BENCH, the NDK
// toolchain as run_android_tests.sh), adb push it with the libraries, run
// it from /data/local/tmp with LD_LIBRARY_PATH=.
//
// Scenarios (all by default):
//   draws    render(RenderScene): a grid of cubes, one DrawCall each, over
//            64 materials (sorting, transforms, per-draw binding).
//   gltf     render(view): the same grid as glTF nodes (node transforms,
//            material uniforms, renderPrimitive()).
//   shadows  render(RenderScene) with a third-person view of one corner of
//            the grid and one cube moving, so the sun's shadow map is
//            redrawn every frame (shadow caster culling).
//   fill     one opaque quad filling a 3840x2160 view: fragment shading
//            (material specialization) with almost no CPU work.
//   gbuffer  renderGBuffer() alone, the grid.
//   denoise  renderGBuffer() + RealtimeDenoiser::denoise() every frame.
//
// Per scenario: "cpu" is the time spent inside the calls (encoding and
// submission; it includes waits for the GPU on earlier frames), "frame" the
// wall time per frame over the run once the GPU has finished, which is the
// GPU time when the GPU is the bottleneck. Offscreen 1280x720, no vsync.

#include <campello_renderer/campello_renderer.hpp>
#include <campello_renderer/realtime_denoiser.hpp>
#include <campello_gpu/device.hpp>
#include <campello_gpu/texture.hpp>
#include <gltf/gltf.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <future>
#include <string>
#include <vector>

using namespace systems::leal::campello_renderer;
namespace GPU = systems::leal::campello_gpu;
namespace gltf = systems::leal::gltf;
namespace VM = systems::leal::vector_math;
using Clock = std::chrono::steady_clock;

namespace {

constexpr uint32_t kWidth = 1280, kHeight = 720;
constexpr int kGrid = 70;  // 70 x 70 = 4900 cubes
constexpr int kMaterials = 64;
constexpr int kWarmup = 20;

std::string base64(const std::vector<uint8_t>& bytes) {
    static const char* table = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    for (size_t i = 0; i < bytes.size(); i += 3) {
        uint32_t v = uint32_t(bytes[i]) << 16;
        if (i + 1 < bytes.size()) v |= uint32_t(bytes[i + 1]) << 8;
        if (i + 2 < bytes.size()) v |= bytes[i + 2];
        out += table[(v >> 18) & 63];
        out += table[(v >> 12) & 63];
        out += i + 1 < bytes.size() ? table[(v >> 6) & 63] : '=';
        out += i + 2 < bytes.size() ? table[v & 63] : '=';
    }
    return out;
}

// A unit cube (24 vertices, positions and normals, 36 uint16 indices), kGrid^2
// nodes on a grid in the XY plane, each using the cube mesh with one of
// kMaterials materials (one mesh per material: a glTF mesh owns its material).
std::string gridGltf(bool withNodes) {
    std::vector<float> positions, normals;
    std::vector<uint16_t> indices;
    const float n[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    for (int f = 0; f < 6; ++f) {
        const float* N = n[f];
        // Two axes spanning the face.
        float u[3] = {N[1], N[2], N[0]}, v[3];
        v[0] = N[1] * u[2] - N[2] * u[1];
        v[1] = N[2] * u[0] - N[0] * u[2];
        v[2] = N[0] * u[1] - N[1] * u[0];
        const uint16_t base = uint16_t(positions.size() / 3);
        for (int c = 0; c < 4; ++c) {
            const float su = (c == 1 || c == 2) ? 0.4f : -0.4f, sv = c >= 2 ? 0.4f : -0.4f;
            for (int k = 0; k < 3; ++k) {
                positions.push_back(0.4f * N[k] + su * u[k] + sv * v[k]);
                normals.push_back(N[k]);
            }
        }
        for (uint16_t i : {0, 1, 2, 0, 2, 3}) indices.push_back(uint16_t(base + i));
    }
    std::vector<uint8_t> buffer(positions.size() * 4 + normals.size() * 4 + indices.size() * 2);
    std::memcpy(buffer.data(), positions.data(), positions.size() * 4);
    std::memcpy(buffer.data() + positions.size() * 4, normals.data(), normals.size() * 4);
    std::memcpy(buffer.data() + positions.size() * 8, indices.data(), indices.size() * 2);

    std::string materials, meshes, nodes, roots;
    for (int m = 0; m < kMaterials; ++m) {
        char text[256];
        std::snprintf(text, sizeof(text),
                      R"(%s{"pbrMetallicRoughness": {"baseColorFactor": [%.3f, %.3f, %.3f, 1], "metallicFactor": %.2f, "roughnessFactor": %.2f}})",
                      m ? "," : "", 0.2 + 0.8 * ((m * 37) % 64) / 64.0, 0.2 + 0.8 * ((m * 11) % 64) / 64.0,
                      0.2 + 0.8 * ((m * 23) % 64) / 64.0, (m % 4) / 4.0, 0.2 + 0.8 * (m % 8) / 8.0);
        materials += text;
        std::snprintf(text, sizeof(text),
                      R"(%s{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2, "material": %d}]})",
                      m ? "," : "", m);
        meshes += text;
    }
    if (withNodes) {
        for (int i = 0; i < kGrid * kGrid; ++i) {
            char text[160];
            const double x = (i % kGrid) - (kGrid - 1) * 0.5, y = (i / kGrid) - (kGrid - 1) * 0.5;
            std::snprintf(text, sizeof(text), R"(%s{"mesh": %d, "translation": [%.1f, %.1f, 0]})", i ? "," : "",
                          i % kMaterials, x, y);
            nodes += text;
            roots += (i ? "," : "") + std::to_string(i);
        }
    }
    char accessors[1024];
    std::snprintf(accessors, sizeof(accessors),
                  R"("accessors": [
        {"bufferView": 0, "componentType": 5126, "count": 24, "type": "VEC3", "max": [0.4, 0.4, 0.4], "min": [-0.4, -0.4, -0.4]},
        {"bufferView": 1, "componentType": 5126, "count": 24, "type": "VEC3"},
        {"bufferView": 2, "componentType": 5123, "count": 36, "type": "SCALAR"}],
    "bufferViews": [
        {"buffer": 0, "byteOffset": 0, "byteLength": 288},
        {"buffer": 0, "byteOffset": 288, "byteLength": 288},
        {"buffer": 0, "byteOffset": 576, "byteLength": 72}],)");
    return std::string(R"({"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [)") + roots + "]}], " +
           R"("nodes": [)" + nodes + "], " + R"("materials": [)" + materials + "], " + R"("meshes": [)" + meshes + "], " +
           accessors + R"( "buffers": [{"uri": "data:application/octet-stream;base64,)" + base64(buffer) +
           R"(", "byteLength": )" + std::to_string(buffer.size()) + "}]}";
}

std::shared_ptr<gltf::GLTF> load(const std::string& json) {
    return gltf::GLTF::loadGLTF(json, [](const std::string&) {
        return std::async(std::launch::deferred, [] { return std::vector<uint8_t>{}; });
    });
}

void cameraLooking(RenderScene& scene) {
    const VM::Vector3<double> eye(0.0, -10.0, 75.0);
    scene.camera.view = VM::Matrix4<double>::lookAt(eye, VM::Vector3<double>(0.0, 0.0, 0.0), VM::Vector3<double>(0.0, 1.0, 0.0));
    scene.camera.projection =
        VM::Matrix4<double>::perspective(55.0 * 3.14159265358979323846 / 180.0, double(kWidth) / kHeight, 0.5, 500.0);
    scene.camera.position = eye;
}

struct Result {
    double cpuMs = 0, frameMs = 0;
};

// frame() once per frame: kWarmup frames, then `frames` timed ones.
Result run(const std::shared_ptr<GPU::Device>& device, int frames, const std::function<void()>& frame) {
    for (int i = 0; i < kWarmup; ++i) frame();
    device->waitForIdle();
    std::vector<double> cpu;
    cpu.reserve(frames);
    const auto start = Clock::now();
    for (int i = 0; i < frames; ++i) {
        const auto t0 = Clock::now();
        frame();
        cpu.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    }
    device->waitForIdle();
    const double total = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    std::sort(cpu.begin(), cpu.end());
    return {cpu[cpu.size() / 2], total / frames};
}

void report(const char* name, const Result& r) {
    std::printf("%-8s cpu %7.3f ms (median)   frame %7.3f ms (%6.1f fps)\n", name, r.cpuMs, r.frameMs, 1000.0 / r.frameMs);
    std::fflush(stdout);
}

// The render targets, kept for the run: on Vulkan a view doesn't keep its
// texture alive (a view of a released texture is a dangling image).
std::vector<std::shared_ptr<GPU::Texture>> gTargets;

std::shared_ptr<GPU::TextureView> target(const std::shared_ptr<GPU::Device>& device, uint32_t width = kWidth,
                                         uint32_t height = kHeight) {
    auto texture = device->createTexture(GPU::TextureType::tt2d, GPU::PixelFormat::rgba8unorm, width, height, 1, 1, 1,
                                         GPU::TextureUsage::renderTarget);
    if (!texture) return nullptr;
    gTargets.push_back(texture);
    return texture->createView(GPU::PixelFormat::rgba8unorm, 1);
}

// The grid as DrawCalls: one cube mesh, the asset's materials.
struct DrawScene {
    std::shared_ptr<gltf::GLTF> asset;
    RenderScene scene;
};

// --no-ibl: pipelines are created after setAsset(), which then has none to
// bake the environment's IBL with (as the tests do). The bake hangs some
// Adreno drivers; shading then has no image-based light.
bool gBakeIbl = true;
bool gSpecialize = true;

void createPipelines(Renderer& renderer) {
    renderer.setMaterialSpecialization(gSpecialize);
    renderer.createDefaultPipelines(GPU::PixelFormat::rgba8unorm);
}

bool buildDrawScene(Renderer& renderer, DrawScene& out) {
    out.asset = load(gridGltf(false));
    if (!out.asset) return false;
    const auto setupStart = Clock::now();
    renderer.setAsset(out.asset);  // with pipelines: bakes the environment's IBL
    std::printf("         setAsset %.0f ms\n", std::chrono::duration<double, std::milli>(Clock::now() - setupStart).count());
    std::fflush(stdout);
    if (!gBakeIbl) createPipelines(renderer);
    GpuMesh* mesh = renderer.uploadMesh((*out.asset->meshes)[0].primitives[0], *out.asset);
    if (!mesh) return false;
    std::vector<GpuMaterial*> materials;
    for (const auto& material : *out.asset->materials) materials.push_back(renderer.uploadMaterial(material, *out.asset));
    cameraLooking(out.scene);
    for (int i = 0; i < kGrid * kGrid; ++i) {
        DrawCall draw;
        draw.mesh = mesh;
        draw.material = materials[size_t(i % kMaterials)];
        draw.worldTransform = VM::Matrix4<double>::translate(
            VM::Vector3<double>((i % kGrid) - (kGrid - 1) * 0.5, (i / kGrid) - (kGrid - 1) * 0.5, 0.0));
        out.scene.opaque.push_back(draw);
    }
    LightData sun;
    sun.type = 0;
    sun.direction = VM::Vector3<double>(-0.3, -0.5, -1.0);
    sun.castShadows = true;
    out.scene.lights.push_back(sun);
    return true;
}

// Fragment-bound: one opaque quad filling a 3840x2160 view (8.3M shaded
// fragments a frame, one draw), so the frame time is the fragment shader's.
constexpr uint32_t kFillWidth = 3840, kFillHeight = 2160;

struct FillScene {
    std::shared_ptr<gltf::GLTF> asset;
    GpuMesh quad;
    RenderScene scene;
};

bool buildFillScene(const std::shared_ptr<GPU::Device>& device, Renderer& renderer, FillScene& out) {
    out.asset = load(R"({"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": []}],
        "materials": [{"pbrMetallicRoughness":
            {"baseColorFactor": [0.8, 0.5, 0.3, 1], "metallicFactor": 0.1, "roughnessFactor": 0.5}}]})");
    if (!out.asset) return false;
    renderer.setAsset(out.asset);
    if (!gBakeIbl) createPipelines(renderer);
    GpuMaterial* material = renderer.uploadMaterial((*out.asset->materials)[0], *out.asset);
    if (!material) return false;
    // A unit quad facing +z, scaled per layer to fill the view.
    const float positions[] = {-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0};
    const float normals[] = {0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1};
    const float tangents[] = {1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1};
    const float uvs[] = {0, 1, 1, 1, 1, 0, 0, 0};
    const uint32_t indices[] = {0, 1, 2, 0, 2, 3};
    auto& q = out.quad;
    q.vertexBuffers[Renderer::VERTEX_SLOT_POSITION] =
        device->createBuffer(sizeof(positions), GPU::BufferUsage::vertex, (void*)positions);
    q.vertexBuffers[Renderer::VERTEX_SLOT_NORMAL] = device->createBuffer(sizeof(normals), GPU::BufferUsage::vertex, (void*)normals);
    q.vertexBuffers[Renderer::VERTEX_SLOT_TANGENT] = device->createBuffer(sizeof(tangents), GPU::BufferUsage::vertex, (void*)tangents);
    q.vertexBuffers[Renderer::VERTEX_SLOT_TEXCOORD0] = device->createBuffer(sizeof(uvs), GPU::BufferUsage::vertex, (void*)uvs);
    q.indexBuffer = device->createBuffer(sizeof(indices), GPU::BufferUsage::index, (void*)indices);
    q.indexCount = 6;
    q.indexFormat = GPU::IndexFormat::uint32;
    q.vertexCount = 4;
    q.layout = nullptr;
    q.setBoundsFromPositions(positions, 4);

    const VM::Vector3<double> eye(0.0, 0.0, 0.0);
    out.scene.camera.view = VM::Matrix4<double>::lookAt(eye, VM::Vector3<double>(0.0, 0.0, -1.0), VM::Vector3<double>(0.0, 1.0, 0.0));
    out.scene.camera.projection =
        VM::Matrix4<double>::perspective(55.0 * 3.14159265358979323846 / 180.0, double(kFillWidth) / kFillHeight, 0.1, 100.0);
    out.scene.camera.position = eye;
    DrawCall draw;  // 2 away, wider than the view (55 deg, 16:9) there
    draw.mesh = &q;
    draw.material = material;
    draw.worldTransform = VM::Matrix4<double>::translate(VM::Vector3<double>(0.0, 0.0, -2.0)) *
                          VM::Matrix4<double>::scale(VM::Vector3<double>(2.4, 2.4, 1.0));
    out.scene.opaque.push_back(draw);
    LightData sun;
    sun.type = 0;
    sun.direction = VM::Vector3<double>(-0.3, -0.5, -1.0);
    out.scene.lights.push_back(sun);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    // [scenario] [frames], and --general: the general PBR shader
    // (setMaterialSpecialization(false)), to compare with specialization.
    std::vector<std::string> args;
    bool specialize = true;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--general") specialize = false;
        else if (std::string(argv[i]) == "--no-ibl") gBakeIbl = false;
        else args.push_back(argv[i]);
    }
    gSpecialize = specialize;
    const std::string only = args.size() > 0 ? args[0] : "";
    const int frames = args.size() > 1 ? std::max(1, std::atoi(args[1].c_str())) : 200;
    auto device = GPU::Device::createDefaultDevice(nullptr);
    if (!device) {
        std::fprintf(stderr, "no GPU device\n");
        return 1;
    }
    std::printf("campello_renderer %s, %s, %dx%d, %d draws, %d materials, %d frames, %s shader\n", getVersion().c_str(),
                device->getName().c_str(), kWidth, kHeight, kGrid * kGrid, kMaterials, frames,
                specialize ? "specialized" : "general");

    if (only.empty() || only == "draws") {
        Renderer renderer(device);
        renderer.resize(kWidth, kHeight);
        if (gBakeIbl) createPipelines(renderer);
        renderer.setSkyboxEnabled(false);
        DrawScene draws;
        auto view = target(device);
        if (view && buildDrawScene(renderer, draws)) {
            report("draws", run(device, frames, [&] { renderer.render(draws.scene, view); }));

        }
    }
    if (only.empty() || only == "shadows") {
        // A third-person view: looking down at one corner of the grid, one cube
        // moving, so the sun's shadow map is redrawn every frame.
        Renderer renderer(device);
        renderer.resize(kWidth, kHeight);
        if (gBakeIbl) createPipelines(renderer);
        renderer.setSkyboxEnabled(false);
        DrawScene draws;
        auto view = target(device);
        if (view && buildDrawScene(renderer, draws)) {
            const VM::Vector3<double> eye(-30.0, -30.0, 6.0);
            draws.scene.camera.view = VM::Matrix4<double>::lookAt(eye, VM::Vector3<double>(-27.0, -27.0, 0.0),
                                                                 VM::Vector3<double>(0.0, 0.0, 1.0));
            draws.scene.camera.position = eye;
            const auto rest = draws.scene.opaque[0].worldTransform;
            int frame = 0;
            report("shadows", run(device, frames, [&] {
                       draws.scene.opaque[0].worldTransform =
                           VM::Matrix4<double>::translate(VM::Vector3<double>(0.0, 0.0, 0.01 * (frame++ % 2))) * rest;
                       renderer.render(draws.scene, view);
                   }));
            const auto stats = renderer.getLastFrameStats();
            std::printf("         shadow casters drawn %u of %u\n", stats.shadowCastersDrawn, stats.shadowCasterCount);
        }
    }
    if (only.empty() || only == "fill") {
        Renderer renderer(device);
        renderer.resize(kFillWidth, kFillHeight);
        if (gBakeIbl) createPipelines(renderer);
        renderer.setSkyboxEnabled(false);
        FillScene fill;
        auto view = target(device, kFillWidth, kFillHeight);
        if (view && buildFillScene(device, renderer, fill)) {
            report("fill", run(device, frames, [&] { renderer.render(fill.scene, view); }));
            // That it drew: the centre pixel of one more frame (not the clear colour).
            auto readable = device->createTexture(GPU::TextureType::tt2d, GPU::PixelFormat::rgba8unorm, kFillWidth,
                                                  kFillHeight, 1, 1, 1,
                                                  (GPU::TextureUsage)(uint32_t(GPU::TextureUsage::renderTarget) |
                                                                      uint32_t(GPU::TextureUsage::copySrc)));
            if (readable) {
                renderer.render(fill.scene, readable->createView(GPU::PixelFormat::rgba8unorm, 1));
                device->waitForIdle();
                std::vector<uint8_t> pixels(size_t(kFillWidth) * kFillHeight * 4);
                if (readable->download(0, 0, pixels.data(), pixels.size())) {
                    const size_t at = (size_t(kFillHeight / 2) * kFillWidth + kFillWidth / 2) * 4;
                    std::printf("         centre pixel %u %u %u\n", pixels[at], pixels[at + 1], pixels[at + 2]);
                }
            }
        }
    }
    if (only.empty() || only == "gltf") {
        Renderer renderer(device);
        auto asset = load(gridGltf(true));
        auto view = target(device);
        if (asset && view) {
            renderer.setAsset(asset);
            renderer.resize(kWidth, kHeight);
            createPipelines(renderer);  // after setAsset(): the glTF scenario never bakes IBL
            renderer.setSkyboxEnabled(false);
            RenderScene camera;
            cameraLooking(camera);
            float viewMatrix[16], projection[16];
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) {
                    viewMatrix[c * 4 + r] = (float)camera.camera.view.data[r * 4 + c];
                    projection[c * 4 + r] = (float)camera.camera.projection.data[r * 4 + c];
                }
            }
            renderer.setCameraMatrices(viewMatrix, projection);
            report("gltf", run(device, frames, [&] { renderer.render(view); }));
        }
    }
    if (only.empty() || only == "gbuffer") {
        Renderer renderer(device);
        renderer.resize(kWidth, kHeight);
        if (gBakeIbl) createPipelines(renderer);
        DrawScene draws;
        if (!renderer.supportsGBuffer()) {
            std::printf("gbuffer  (not supported here)\n");
        } else if (buildDrawScene(renderer, draws)) {
            report("gbuffer", run(device, frames, [&] { renderer.renderGBuffer(draws.scene); }));
        }
    }
    if (only.empty() || only == "denoise") {
        Renderer renderer(device);
        renderer.resize(kWidth, kHeight);
        if (gBakeIbl) createPipelines(renderer);
        RealtimeDenoiser denoiser(device);
        DrawScene draws;
        auto noisy = device->createTexture(GPU::TextureType::tt2d, GPU::PixelFormat::rgba16float, kWidth, kHeight, 1, 1, 1,
                                           GPU::TextureUsage::textureBinding);
        if (!renderer.supportsGBuffer() || !denoiser.supported()) {
            std::printf("denoise  (not supported here)\n");
        } else if (noisy && buildDrawScene(renderer, draws)) {
            report("denoise", run(device, frames, [&] {
                       renderer.renderGBuffer(draws.scene);
                       denoiser.denoise(noisy, renderer.gBuffer(), draws.scene.camera.projection);
                   }));
        }
    }
    gTargets.clear();  // before the device goes
    return 0;
}
