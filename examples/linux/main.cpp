// campello_renderer Linux example — GLFW window + Vulkan (via campello_gpu).
//
// Responsibilities:
//   - Creates a GLFW window (Wayland) and a campello_gpu Device that owns its swapchain.
//   - Loads a GLTF/GLB file dropped onto the window (or passed as argv[1]) and delegates
//     to Renderer; dropping an HDR/EXR/PNG/JPEG/WebP file loads it as an environment map.
//   - Calls Renderer::render() each frame, targeting the device's own swapchain.
//   - Handles mouse drag/scroll for orbit/pan/zoom camera control.
//   - Mirrors examples/macos/ViewController.mm + AppDelegate.mm's feature set; the macOS
//     menu bar's Cmd+<key> actions become Ctrl+<key> here (there's no native menu bar
//     concept on Linux) — see printHelp() below for the full list.

// No OpenGL/Vulkan headers needed from GLFW itself — campello_gpu owns the
// Vulkan surface/device entirely; GL/gl.h isn't even installed on this machine.
#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_WAYLAND
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <campello_gpu/device.hpp>
#include <campello_gpu/texture_view.hpp>
#include <campello_gpu/constants/pixel_format.hpp>
#include <campello_gpu/platform/linux_surface.hpp>

#include <campello_renderer/campello_renderer.hpp>

#include <gltf/gltf.hpp>

#include "Camera.hpp"

#include <algorithm>
#include <chrono>
#include <thread>
#include <cstdio>
#include <fstream>
#include <future>
#include <string>
#include <vector>

namespace gpu  = systems::leal::campello_gpu;
namespace gltf = systems::leal::gltf;
namespace rend = systems::leal::campello_renderer;

// ---------------------------------------------------------------------------
// ViewMode name/hotkey table — mirrors examples/macos/ViewController.mm exactly
// (same plain-key layout), just keyed by GLFW_KEY_* instead of NSEvent characters.
// ---------------------------------------------------------------------------

static const char *ViewModeName(rend::ViewMode mode) {
    switch (mode) {
        case rend::ViewMode::normal: return "normal";
        case rend::ViewMode::worldNormal: return "worldNormal";
        case rend::ViewMode::baseColor: return "baseColor";
        case rend::ViewMode::metallic: return "metallic";
        case rend::ViewMode::roughness: return "roughness";
        case rend::ViewMode::occlusion: return "occlusion";
        case rend::ViewMode::emissive: return "emissive";
        case rend::ViewMode::alpha: return "alpha";
        case rend::ViewMode::uv0: return "uv0";
        case rend::ViewMode::specularFactor: return "specularFactor";
        case rend::ViewMode::specularColor: return "specularColor";
        case rend::ViewMode::sheenColor: return "sheenColor";
        case rend::ViewMode::sheenRoughness: return "sheenRoughness";
        case rend::ViewMode::clearcoat: return "clearcoat";
        case rend::ViewMode::clearcoatRoughness: return "clearcoatRoughness";
        case rend::ViewMode::clearcoatNormal: return "clearcoatNormal";
        case rend::ViewMode::transmission: return "transmission";
        case rend::ViewMode::environment: return "environment";
        case rend::ViewMode::iridescence: return "iridescence";
        case rend::ViewMode::anisotropy: return "anisotropy";
        case rend::ViewMode::dispersion: return "dispersion";
    }
    return "unknown";
}

static bool ViewModeForKey(int key, rend::ViewMode &outMode) {
    switch (key) {
        case GLFW_KEY_0: outMode = rend::ViewMode::normal; return true;
        case GLFW_KEY_1: outMode = rend::ViewMode::worldNormal; return true;
        case GLFW_KEY_2: outMode = rend::ViewMode::baseColor; return true;
        case GLFW_KEY_3: outMode = rend::ViewMode::metallic; return true;
        case GLFW_KEY_4: outMode = rend::ViewMode::roughness; return true;
        case GLFW_KEY_5: outMode = rend::ViewMode::occlusion; return true;
        case GLFW_KEY_6: outMode = rend::ViewMode::emissive; return true;
        case GLFW_KEY_7: outMode = rend::ViewMode::alpha; return true;
        case GLFW_KEY_8: outMode = rend::ViewMode::uv0; return true;
        case GLFW_KEY_9: outMode = rend::ViewMode::specularFactor; return true;
        case GLFW_KEY_Q: outMode = rend::ViewMode::specularColor; return true;
        case GLFW_KEY_W: outMode = rend::ViewMode::sheenColor; return true;
        case GLFW_KEY_E: outMode = rend::ViewMode::sheenRoughness; return true;
        case GLFW_KEY_R: outMode = rend::ViewMode::clearcoat; return true;
        case GLFW_KEY_T: outMode = rend::ViewMode::clearcoatRoughness; return true;
        case GLFW_KEY_Y: outMode = rend::ViewMode::clearcoatNormal; return true;
        case GLFW_KEY_U: outMode = rend::ViewMode::transmission; return true;
        case GLFW_KEY_I: outMode = rend::ViewMode::environment; return true;
        case GLFW_KEY_O: outMode = rend::ViewMode::iridescence; return true;
        case GLFW_KEY_P: outMode = rend::ViewMode::anisotropy; return true;
        case GLFW_KEY_Z: outMode = rend::ViewMode::dispersion; return true;
        default: return false;
    }
}

static bool hasExtension(const std::string &path, std::initializer_list<const char *> exts) {
    auto dot = path.find_last_of('.');
    if (dot == std::string::npos) return false;
    std::string ext = path.substr(dot + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    for (auto *e : exts) {
        if (ext == e) return true;
    }
    return false;
}

static bool isModelFile(const std::string &path) { return hasExtension(path, {"glb", "gltf"}); }
static bool isImageFile(const std::string &path) {
    return hasExtension(path, {"hdr", "exr", "png", "jpg", "jpeg", "webp"});
}

static bool readFile(const std::string &path, std::vector<uint8_t> &out) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return false;
    std::streamsize size = file.tellg();
    if (size < 0) return false;
    file.seekg(0, std::ios::beg);
    out.resize((size_t)size);
    if (size > 0 && !file.read(reinterpret_cast<char *>(out.data()), size)) return false;
    return true;
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

struct App {
    GLFWwindow *window = nullptr;

    std::shared_ptr<gpu::Device>    device;
    std::shared_ptr<rend::Renderer> renderer;

    Camera camera;
    double lastMouseX = 0.0, lastMouseY = 0.0;
    bool   leftDown = false, rightDown = false;
    int    fbWidth = 1280, fbHeight = 720;
    int    ssaaIndex = 0; // 0=off, 1=1.5x, 2=2.0x — see Ctrl+S in printHelp().

    // A file passed as argv[1] is loaded a few frames into the run instead of
    // before the loop starts, so it exercises the exact same "replace the
    // active asset while frames are already in flight" path a real drag-drop
    // hits (nobody drops a file before the window has even rendered once) —
    // rather than the comparatively trivial "load into an otherwise-idle
    // renderer" case that startup-time loading would otherwise only ever test.
    std::string pendingModelPath;
    int         pendingModelDelayFrames = 0;

    std::chrono::steady_clock::time_point lastTick;

    bool init();
    void loadModel(const std::string &path);
    void loadEnvironmentMapFile(const std::string &path);
    void frame();
    void printHelp() const;
};

// ---------------------------------------------------------------------------
// GLFW callbacks
// ---------------------------------------------------------------------------

static void framebufferSizeCallback(GLFWwindow *window, int width, int height) {
    auto *app = static_cast<App *>(glfwGetWindowUserPointer(window));
    if (!app || !app->renderer) return;
    if (width == 0 || height == 0) return; // minimized
    app->renderer->resize((uint32_t)width, (uint32_t)height);
    app->fbWidth  = width;
    app->fbHeight = height;
}

static void dropCallback(GLFWwindow *window, int count, const char **paths) {
    auto *app = static_cast<App *>(glfwGetWindowUserPointer(window));
    if (!app) return;
    for (int i = 0; i < count; ++i) {
        if (isModelFile(paths[i])) {
            app->loadModel(paths[i]);
            return;
        }
    }
    for (int i = 0; i < count; ++i) {
        if (isImageFile(paths[i])) {
            app->loadEnvironmentMapFile(paths[i]);
            return;
        }
    }
    fprintf(stderr, "campello_renderer_linux: dropped file(s) not a recognized .gltf/.glb "
                     "or .hdr/.exr/.png/.jpg/.jpeg/.webp\n");
}

static void mouseButtonCallback(GLFWwindow *window, int button, int action, int /*mods*/) {
    auto *app = static_cast<App *>(glfwGetWindowUserPointer(window));
    if (!app) return;
    glfwGetCursorPos(window, &app->lastMouseX, &app->lastMouseY);
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        app->leftDown = (action == GLFW_PRESS);
    } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        app->rightDown = (action == GLFW_PRESS);
    }
}

static void cursorPosCallback(GLFWwindow *window, double x, double y) {
    auto *app = static_cast<App *>(glfwGetWindowUserPointer(window));
    if (!app || (!app->leftDown && !app->rightDown)) return;

    float dx = (float)(x - app->lastMouseX);
    // GLFW's cursor Y grows downward; examples/macos/Camera.hpp's orbit/pan signs were
    // tuned against Cocoa's bottom-up NSPoint Y, so this negation is the two conventions'
    // algebraic difference, not an arbitrary flip — see ViewController.mm's mouseDragged:/
    // rightMouseDragged: for the original (dx, dy) signs this mirrors.
    float dy = -(float)(y - app->lastMouseY);
    app->lastMouseX = x;
    app->lastMouseY = y;

    bool shift = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                 glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;

    if (app->leftDown) {
        if (shift) app->camera.pan(dx, dy);
        else       app->camera.orbit(-dx * 0.005f, dy * 0.005f);
    } else if (app->rightDown) {
        app->camera.pan(dx, dy);
    }
}

static void scrollCallback(GLFWwindow *window, double /*xoffset*/, double yoffset) {
    auto *app = static_cast<App *>(glfwGetWindowUserPointer(window));
    if (!app) return;
    app->camera.zoom((float)yoffset);
}

static void keyCallback(GLFWwindow *window, int key, int /*scancode*/, int action, int mods) {
    if (action != GLFW_PRESS) return;
    auto *app = static_cast<App *>(glfwGetWindowUserPointer(window));
    if (!app || !app->renderer) return;

    if (key == GLFW_KEY_ESCAPE) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
        return;
    }

    if (mods & GLFW_MOD_CONTROL) {
        auto &r = *app->renderer;
        switch (key) {
            case GLFW_KEY_Q:
                glfwSetWindowShouldClose(window, GLFW_TRUE);
                return;
            case GLFW_KEY_D: {
                auto next = r.getViewMode() == rend::ViewMode::worldNormal
                    ? rend::ViewMode::normal : rend::ViewMode::worldNormal;
                r.setViewMode(next);
                printf("View mode: %s\n", ViewModeName(next));
                return;
            }
            case GLFW_KEY_L:
                if (mods & GLFW_MOD_SHIFT) {
                    bool enabled = !r.isDefaultLightEnabled();
                    r.setDefaultLightEnabled(enabled);
                    printf("Default light: %s\n", enabled ? "ON" : "OFF");
                } else {
                    bool enabled = !r.isPunctualLightsEnabled();
                    r.setPunctualLightsEnabled(enabled);
                    printf("Punctual lights: %s\n", enabled ? "ON" : "OFF");
                }
                return;
            case GLFW_KEY_B: {
                bool enabled = !r.isSkyboxEnabled();
                r.setSkyboxEnabled(enabled);
                printf("Skybox: %s\n", enabled ? "ON" : "OFF");
                return;
            }
            case GLFW_KEY_I: {
                bool enabled = !r.isIBLEnabled();
                r.setIBLEnabled(enabled);
                printf("IBL: %s\n", enabled ? "ON" : "OFF");
                return;
            }
            case GLFW_KEY_A: {
                bool enabled = !r.isFxaaEnabled();
                r.setFxaaEnabled(enabled);
                printf("FXAA: %s\n", enabled ? "ON" : "OFF");
                return;
            }
            case GLFW_KEY_S: {
                app->ssaaIndex = (app->ssaaIndex + 1) % 3;
                float scale = app->ssaaIndex == 0 ? 1.0f : app->ssaaIndex == 1 ? 1.5f : 2.0f;
                r.setSsaaScale(scale);
                printf("SSAA: %s\n", app->ssaaIndex == 0 ? "OFF" : app->ssaaIndex == 1 ? "1.5x" : "2.0x");
                return;
            }
            case GLFW_KEY_1: r.setClearColor(0.08f, 0.08f, 0.10f, 1.0f); printf("Background: Dark\n"); return;
            case GLFW_KEY_2: r.setClearColor(0.35f, 0.35f, 0.38f, 1.0f); printf("Background: Gray\n"); return;
            case GLFW_KEY_3: r.setClearColor(0.78f, 0.82f, 0.88f, 1.0f); printf("Background: Light\n"); return;
            case GLFW_KEY_4:
                r.setSkyboxEnabled(false);
                r.setIBLEnabled(false);
                r.setClearColor(0.08f, 0.08f, 0.10f, 1.0f);
                printf("Background mode: Solid Color (Dark)\n");
                return;
            case GLFW_KEY_5:
                r.setSkyboxEnabled(true);
                r.setIBLEnabled(false);
                printf("Background mode: Skybox\n");
                return;
            case GLFW_KEY_6:
                r.setSkyboxEnabled(true);
                r.setIBLEnabled(true);
                printf("Background mode: Skybox + IBL\n");
                return;
            default:
                return;
        }
    }

    rend::ViewMode mode;
    if (ViewModeForKey(key, mode)) {
        app->renderer->setViewMode(mode);
        printf("View mode: %s\n", ViewModeName(mode));
    }
}

// ---------------------------------------------------------------------------
// App implementation
// ---------------------------------------------------------------------------

void App::printHelp() const {
    printf("campello_renderer -- Linux example\n");
    printf("Drag a .gltf/.glb file onto the window to load it (or pass a path as argv[1]).\n");
    printf("Drag an .hdr/.exr/.png/.jpg/.jpeg/.webp file to load it as an environment map.\n\n");
    printf("Camera: left-drag orbit, shift+left-drag / right-drag pan, scroll zoom.\n\n");
    printf("ViewMode hotkeys: 0 normal, 1 worldNormal, 2 baseColor, 3 metallic, 4 roughness,\n"
           "  5 occlusion, 6 emissive, 7 alpha, 8 uv0, 9 specularFactor, q specularColor,\n"
           "  w sheenColor, e sheenRoughness, r clearcoat, t clearcoatRoughness, y clearcoatNormal,\n"
           "  u transmission, i environment, o iridescence, p anisotropy, z dispersion.\n\n");
    printf("Ctrl+D world-normal debug view, Ctrl+L punctual lights, Ctrl+Shift+L default light,\n"
           "  Ctrl+B skybox, Ctrl+I IBL, Ctrl+A FXAA, Ctrl+S cycle SSAA,\n"
           "  Ctrl+1/2/3 background dark/gray/light, Ctrl+4/5/6 background solid/skybox/skybox+IBL,\n"
           "  Ctrl+Q or Esc quit.\n");
}

static void glfwErrorCallback(int error, const char *description) {
    fprintf(stderr, "GLFW error %d: %s\n", error, description);
}

bool App::init() {
    glfwSetErrorCallback(glfwErrorCallback);
    if (!glfwInit()) {
        fprintf(stderr, "campello_renderer_linux: failed to initialize GLFW\n");
        return false;
    }

    // No OpenGL/GLES context — campello_gpu owns the Vulkan surface/swapchain itself.
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    window = glfwCreateWindow(1280, 720, "campello_renderer -- Linux", nullptr, nullptr);
    if (!window) {
        fprintf(stderr, "campello_renderer_linux: failed to create window\n");
        glfwTerminate();
        return false;
    }

    glfwGetFramebufferSize(window, &fbWidth, &fbHeight);

    gpu::LinuxSurfaceInfo surfaceInfo{};
    surfaceInfo.display = glfwGetWaylandDisplay();
    surfaceInfo.window  = glfwGetWaylandWindow(window);
    surfaceInfo.api     = gpu::LinuxWindowApi::wayland;
    surfaceInfo.width   = (uint32_t)fbWidth;
    surfaceInfo.height  = (uint32_t)fbHeight;

    device = gpu::Device::createDefaultDevice(&surfaceInfo);
    if (!device) {
        fprintf(stderr, "campello_gpu: failed to create device\n");
        return false;
    }
    printf("device: %s  engine: %s\n",
           device->getName().c_str(), gpu::Device::getEngineVersion().c_str());

    renderer = std::make_shared<rend::Renderer>(device);
    renderer->createDefaultPipelines(device->getSwapchainPixelFormat());
    renderer->resize((uint32_t)fbWidth, (uint32_t)fbHeight);

    lastTick = std::chrono::steady_clock::now();

    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetDropCallback(window, dropCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetKeyCallback(window, keyCallback);

    printHelp();
    return true;
}

void App::loadModel(const std::string &path) {
    std::vector<uint8_t> bytes;
    if (!readFile(path, bytes)) {
        fprintf(stderr, "campello_renderer_linux: could not read %s\n", path.c_str());
        return;
    }

    std::shared_ptr<gltf::GLTF> asset;
    if (hasExtension(path, {"glb"})) {
        asset = gltf::GLTF::loadGLB(bytes.data(), bytes.size());
    } else {
        // Base directory of the GLTF file, for resolving relative buffer/image URIs.
        std::string baseDir;
        auto slash = path.find_last_of('/');
        if (slash != std::string::npos) baseDir = path.substr(0, slash + 1);

        asset = gltf::GLTF::loadGLTF(
            std::string(bytes.begin(), bytes.end()),
            [baseDir](const std::string &uri) -> std::future<std::vector<uint8_t>> {
                return std::async(std::launch::deferred, [baseDir, uri]() {
                    std::vector<uint8_t> result;
                    readFile(baseDir + uri, result);
                    return result;
                });
            });
    }

    if (!asset) {
        fprintf(stderr, "campello_renderer_linux: failed to parse %s\n", path.c_str());
        return;
    }

    printf("campello_renderer_linux: loaded %s\n", path.c_str());

    renderer->setAsset(asset);
    camera.fitBounds(renderer->getBoundsRadius());

    uint32_t animCount = renderer->getAnimationCount();
    printf("Animations found: %u\n", animCount);
    for (uint32_t i = 0; i < animCount; ++i) {
        printf("  Animation %u: %s (duration: %.2fs)\n",
               i, renderer->getAnimationName(i).c_str(), renderer->getAnimationDuration(i));
    }
    if (animCount > 0) {
        renderer->playAnimation(0);
        printf("Auto-playing animation: %s\n", renderer->getAnimationName(0).c_str());
    }
}

void App::loadEnvironmentMapFile(const std::string &path) {
    auto tex = renderer->loadEquirectangularEnvironmentMap(path, 0);
    if (tex) {
        renderer->setEnvironmentMap(tex);
        renderer->setSkyboxEnabled(true);
        renderer->setIBLEnabled(true);
        printf("Environment map loaded: %s\n", path.c_str());
    } else {
        fprintf(stderr, "campello_renderer_linux: failed to load environment map %s\n", path.c_str());
    }
}

void App::frame() {
    if (!pendingModelPath.empty() && --pendingModelDelayFrames == 0) {
        std::string path = pendingModelPath;
        pendingModelPath.clear();
        loadModel(path);
    }

    auto now = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(now - lastTick).count();
    lastTick = now;
    if (dt < 0.0) dt = 0.0;
    if (dt > 0.25) dt = 0.25;

    renderer->update(dt);

    int w, h;
    glfwGetFramebufferSize(window, &w, &h);
    if (w == 0 || h == 0) return; // minimized
    if (w != fbWidth || h != fbHeight) {
        renderer->resize((uint32_t)w, (uint32_t)h);
        fbWidth  = w;
        fbHeight = h;
    }

    float aspect = (float)w / (float)h;
    float viewMat[16], projMat[16];
    camera.viewMatrix(viewMat);
    camera.projectionMatrix(aspect, projMat);
    renderer->setCameraMatrices(viewMat, projMat);

    // Targets the device's own swapchain — campello_gpu handles acquire/present
    // internally (see Renderer::render()'s doc comment).
    renderer->render();
}

int main(int argc, char **argv) {
    // Line-buffer stdout so status/help output (device name, loaded model,
    // hotkey feedback) shows up promptly even when not attached to a TTY,
    // instead of sitting in a fully-buffered pipe until exit.
    setvbuf(stdout, nullptr, _IOLBF, 0);

    App app;
    if (!app.init()) return 1;

    if (argc > 1) {
        // See pendingModelPath's doc comment — deliberately not loaded here,
        // before the render loop has produced a single frame.
        app.pendingModelPath = argv[1];
        app.pendingModelDelayFrames = 120; // ~2s at the 60fps cap below
    }

    // Cap to ~60 FPS. campello_gpu's Vulkan present doesn't actually block on
    // vsync here (observed submitting several thousand frames/sec unthrottled),
    // and hammering the 3-deep frame-in-flight ring that much faster than the
    // GPU retires it is what was tripping command-buffer/query-pool-still-in-use
    // validation errors (and an eventual crash) — this keeps CPU submission rate
    // sane regardless of whether campello_gpu's own pacing gets fixed later.
    const auto targetFrameTime = std::chrono::duration<double>(1.0 / 60.0);
    while (!glfwWindowShouldClose(app.window)) {
        auto frameStart = std::chrono::steady_clock::now();
        glfwPollEvents();
        app.frame();
        auto elapsed = std::chrono::steady_clock::now() - frameStart;
        auto remaining = targetFrameTime - elapsed;
        if (remaining > std::chrono::steady_clock::duration::zero()) {
            std::this_thread::sleep_for(remaining);
        }
    }

    // Release the Renderer/Device (and the VkSurfaceKHR/swapchain they own,
    // which wrap the window's Wayland wl_surface) before tearing down GLFW.
    // glfwDestroyWindow()/glfwTerminate() destroy the underlying wl_surface
    // and disconnect from the Wayland display; destroying the Vulkan surface
    // afterwards crashes inside libwayland-client on the now-freed objects.
    app.renderer.reset();
    app.device.reset();

    glfwDestroyWindow(app.window);
    glfwTerminate();
    return 0;
}
