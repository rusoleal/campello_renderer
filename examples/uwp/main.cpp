// campello_renderer on UWP, for a Windows PC or an Xbox in Developer Mode.
//
// Loads DamagedHelmet.glb and an HDR environment from the package and renders
// them with a slowly orbiting camera.
//
// There is no console on UWP, so everything goes to LocalState:
//   app.log    this file's messages (also sent to the debugger's Output window)
//   stdout.log, stderr.log  output of the campello libraries

#include <unknwn.h>

#include <winrt/Windows.ApplicationModel.Core.h>
#include <winrt/Windows.ApplicationModel.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Gaming.Input.h>
#include <winrt/Windows.Storage.h>
#include <winrt/Windows.System.Profile.h>
#include <winrt/Windows.System.h>
#include <winrt/Windows.UI.Core.h>

#include <windows.h>
#include <crtdbg.h>

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <iterator>
#include <exception>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <campello_gpu/device.hpp>
#include <campello_renderer/campello_renderer.hpp>
#include <gltf/gltf.hpp>

#include "Camera.hpp"
#include "Overlay.hpp"

using namespace winrt;
using namespace winrt::Windows::ApplicationModel::Core;
using namespace winrt::Windows::UI::Core;

namespace gpu = systems::leal::campello_gpu;
namespace rend = systems::leal::campello_renderer;
namespace gltf = systems::leal::gltf;

namespace {

constexpr float kOrbitRadiansPerSecond = 0.4f;
constexpr float kStickRadiansPerSecond = 1.5f;
constexpr float kStickDeadZone = 0.15f;

struct NamedViewMode {
    rend::ViewMode mode;
    const char* name;
};
// Button A cycles through these, to see which stage of the shader shows a problem.
constexpr NamedViewMode kViewModes[] = {
    {rend::ViewMode::normal, "normal"},
    {rend::ViewMode::baseColor, "baseColor"},
    {rend::ViewMode::worldNormal, "worldNormal"},
    {rend::ViewMode::uv0, "uv0"},
    {rend::ViewMode::metallic, "metallic"},
    {rend::ViewMode::roughness, "roughness"},
    {rend::ViewMode::occlusion, "occlusion"},
    {rend::ViewMode::emissive, "emissive"},
};

std::ofstream gLogFile;

void logLine(const char* fmt, ...) {
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    std::string line = std::string("[uwp] ") + buf + "\n";
    OutputDebugStringA(line.c_str());
    if (gLogFile.is_open()) {
        gLogFile << line;
        gLogFile.flush(); // survive a crash
    }
}

// Logs a fatal exception's code and location: on the console there's no debugger
// attached, so app.log is the only evidence of a crash. Resolve the offset with
// the module's PDB.
LONG WINAPI logFatalException(PEXCEPTION_POINTERS info) {
    const DWORD code = info->ExceptionRecord->ExceptionCode;
    const bool fatal = code == EXCEPTION_ACCESS_VIOLATION || code == EXCEPTION_ILLEGAL_INSTRUCTION ||
                       code == EXCEPTION_STACK_OVERFLOW || code == EXCEPTION_INT_DIVIDE_BY_ZERO ||
                       code == EXCEPTION_BREAKPOINT || code == 0xC0000374 /* heap corruption */ ||
                       code == 0xC0000409 /* fail-fast */;
    if (fatal) {
        const auto address = reinterpret_cast<uintptr_t>(info->ExceptionRecord->ExceptionAddress);
        MEMORY_BASIC_INFORMATION region = {};
        VirtualQuery(info->ExceptionRecord->ExceptionAddress, &region, sizeof(region));
        char module[MAX_PATH] = "?";
        GetModuleFileNameA(static_cast<HMODULE>(region.AllocationBase), module, MAX_PATH);
        logLine("CRASH: exception 0x%08X at %s+0x%llX", static_cast<unsigned>(code), module,
                static_cast<unsigned long long>(address - reinterpret_cast<uintptr_t>(region.AllocationBase)));
        if (code == EXCEPTION_ACCESS_VIOLATION && info->ExceptionRecord->NumberParameters >= 2) {
            logLine("CRASH: %s address 0x%llX",
                    info->ExceptionRecord->ExceptionInformation[0] == 0 ? "reading" : "writing or executing",
                    static_cast<unsigned long long>(info->ExceptionRecord->ExceptionInformation[1]));
        }
    }
    return EXCEPTION_CONTINUE_SEARCH; // let the system terminate the app
}

// CRT failures end the process without reaching the filter above.
void installCrashLogging() {
    SetUnhandledExceptionFilter(logFatalException);
    std::set_terminate([] {
        logLine("CRASH: std::terminate");
        std::abort();
    });
    std::signal(SIGABRT, [](int) { logLine("CRASH: abort"); });
    _CrtSetReportHook([](int, char* message, int*) -> int {
        logLine("CRT report: %s", message);
        return 0;
    });
    _set_invalid_parameter_handler([](const wchar_t*, const wchar_t*, const wchar_t*, unsigned, uintptr_t) {
        logLine("CRASH: invalid parameter passed to a CRT function");
    });
}

bool readFile(const std::string& path, std::vector<uint8_t>& out) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return false;
    const std::streamsize size = file.tellg();
    if (size <= 0) return false;
    file.seekg(0, std::ios::beg);
    out.resize(static_cast<size_t>(size));
    return static_cast<bool>(file.read(reinterpret_cast<char*>(out.data()), size));
}

double secondsSince(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

void openLogs() {
    std::wstring dir{winrt::Windows::Storage::ApplicationData::Current().LocalFolder().Path()};
    gLogFile.open(dir + L"\\app.log", std::ios::out | std::ios::trunc);
    FILE* ignored = nullptr;
    // Separate files: a second freopen of the same file fails with EACCES, and a
    // stderr without a valid handle aborts the Debug CRT on its first write.
    const errno_t outResult = _wfreopen_s(&ignored, (dir + L"\\stdout.log").c_str(), L"w", stdout);
    const errno_t errResult = _wfreopen_s(&ignored, (dir + L"\\stderr.log").c_str(), L"w", stderr);
    logLine("stdio redirect: stdout errno=%d, stderr errno=%d", outResult, errResult);
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);
}

} // namespace

struct App : implements<App, IFrameworkViewSource, IFrameworkView> {
    IFrameworkView CreateView() { return *this; }
    void Initialize(CoreApplicationView const&) {}
    void Load(hstring const&) {}
    void Uninitialize() {}

    void SetWindow(CoreWindow const& window) {
        window.Closed([this](auto&&, auto&&) { closed = true; });
    }

    void Run() {
        CoreWindow window = CoreWindow::GetForCurrentThread();
        window.Activate();
        CoreDispatcher dispatcher = window.Dispatcher();

        openLogs();
        installCrashLogging();
        bool ready = false;
        try {
            ready = start(window);
        } catch (const std::exception& e) {
            logLine("FAILED: exception during startup: %s", e.what());
        } catch (const hresult_error& e) {
            logLine("FAILED: WinRT error during startup: hr=0x%08X", static_cast<unsigned>(e.code().value));
        }

        while (!closed) {
            if (ready) {
                dispatcher.ProcessEvents(CoreProcessEventsOption::ProcessAllIfPresent);
                renderFrame();
            } else {
                // Keep the app alive so the log can be read.
                dispatcher.ProcessEvents(CoreProcessEventsOption::ProcessOneAndAllPending);
            }
        }

        renderer.reset();
        overlay.reset();
        device.reset();
    }

private:
    // 4K only on an Xbox in Game mode, which has the memory and GPU time for it. The
    // swapchain stretches to fill the window, so this is the render resolution.
    void chooseResolution() {
        const bool isXbox = winrt::Windows::System::Profile::AnalyticsInfo::VersionInfo().DeviceFamily() == L"Windows.Xbox";
        const bool isGameMode = winrt::Windows::System::MemoryManager::AppMemoryUsageLimit() > 2048ull * 1024ull * 1024ull;
        if (isXbox && isGameMode) {
            width = 3840;
            height = 2160;
        }
    }

    bool start(CoreWindow const& window) {
        const auto startTime = std::chrono::steady_clock::now();
        logLine("app memory limit: %llu MB",
                winrt::Windows::System::MemoryManager::AppMemoryUsageLimit() / (1024ull * 1024ull));

        logLine("creating device");
        chooseResolution();
        logLine("render resolution: %u x %u", width, height);

        device = gpu::Device::createDefaultDeviceForCoreWindow(winrt::get_unknown(window), width, height);
        if (!device) {
            logLine("FAILED: createDefaultDeviceForCoreWindow returned null");
            return false;
        }
        logLine("device: %s, engine: %s", device->getName().c_str(), gpu::Device::getEngineVersion().c_str());

        renderer = std::make_shared<rend::Renderer>(device);
        renderer->createDefaultPipelines(device->getSwapchainPixelFormat());
        renderer->resize(width, height);
        logLine("pipelines ready at %.2f s", secondsSince(startTime));

        const std::string assets =
            to_string(winrt::Windows::ApplicationModel::Package::Current().InstalledLocation().Path()) + "\\Assets\\";

        std::vector<uint8_t> bytes;
        if (!readFile(assets + "DamagedHelmet.glb", bytes)) {
            logLine("FAILED: could not read DamagedHelmet.glb from %s", assets.c_str());
            return false;
        }
        auto asset = gltf::GLTF::loadGLB(bytes.data(), bytes.size());
        if (!asset) {
            logLine("FAILED: could not parse DamagedHelmet.glb");
            return false;
        }
        renderer->setAsset(asset);
        logLine("asset ready at %.2f s", secondsSince(startTime));

        float cx = 0.0f, cy = 0.0f, cz = 0.0f;
        renderer->getBoundsCenter(&cx, &cy, &cz);
        camera.fitBounds(renderer->getBoundsRadius(), Vec3{cx, cy, cz});

        auto environment = renderer->loadEquirectangularEnvironmentMap(assets + "default_environment.hdr", 0);
        if (environment) {
            renderer->setEnvironmentMap(environment);
            renderer->setSkyboxEnabled(true);
            renderer->setIBLEnabled(true);
            logLine("environment ready at %.2f s", secondsSince(startTime));
        } else {
            logLine("WARNING: environment map failed to load");
        }

        overlay = std::make_unique<Overlay>(device, width, height);
        if (overlay->isReady()) {
            renderer->setOverlayCallback([this](const auto& encoder, const auto& colorView, uint32_t frameIndex) {
                overlay->draw(encoder, colorView, frameIndex);
            });
        } else {
            logLine("WARNING: overlay failed to initialize");
        }

        lastFrame = std::chrono::steady_clock::now();
        statsStart = lastFrame;
        return true;
    }

    // A: next view mode, X: FXAA, Y: IBL and skybox, right bumper: pause the orbit,
    // left stick: orbit, triggers: zoom. Every change is written to app.log.
    void handleGamepad(float dt) {
        using namespace winrt::Windows::Gaming::Input;
        auto gamepads = Gamepad::Gamepads();
        if (gamepads.Size() == 0) return;
        const GamepadReading reading = gamepads.GetAt(0).GetCurrentReading();

        const auto pressed = [&](GamepadButtons button) {
            return (reading.Buttons & button) == button && (previousButtons & button) != button;
        };
        if (pressed(GamepadButtons::A)) {
            viewModeIndex = (viewModeIndex + 1) % std::size(kViewModes);
            renderer->setViewMode(kViewModes[viewModeIndex].mode);
            logLine("view mode: %s", kViewModes[viewModeIndex].name);
        }
        if (pressed(GamepadButtons::X)) {
            renderer->setFxaaEnabled(!renderer->isFxaaEnabled());
            logLine("FXAA: %s", renderer->isFxaaEnabled() ? "on" : "off");
        }
        if (pressed(GamepadButtons::Y)) {
            const bool enabled = !renderer->isIBLEnabled();
            renderer->setIBLEnabled(enabled);
            renderer->setSkyboxEnabled(enabled);
            logLine("IBL and skybox: %s", enabled ? "on" : "off");
        }
        if (pressed(GamepadButtons::RightShoulder)) {
            orbitPaused = !orbitPaused;
            logLine("orbit: %s", orbitPaused ? "paused" : "running");
        }
        previousButtons = reading.Buttons;

        const auto stick = [](double value) {
            return std::abs(value) > kStickDeadZone ? static_cast<float>(value) : 0.0f;
        };
        camera.orbit(stick(reading.LeftThumbstickX) * kStickRadiansPerSecond * dt,
                     stick(reading.LeftThumbstickY) * kStickRadiansPerSecond * dt);
        camera.zoom(static_cast<float>(reading.RightTrigger - reading.LeftTrigger) * dt * 10.0f);
    }

    // The fixed-size swapchain stretches to fill the window, so the projection
    // uses the window's aspect ratio, not the swapchain's, to cancel that out.
    float windowAspect() {
        const auto bounds = CoreWindow::GetForCurrentThread().Bounds();
        if (bounds.Width <= 0.0f || bounds.Height <= 0.0f) return 1.0f;
        if (bounds.Width != loggedWidth || bounds.Height != loggedHeight) {
            logLine("window bounds: %.0f x %.0f", bounds.Width, bounds.Height);
            loggedWidth = bounds.Width;
            loggedHeight = bounds.Height;
        }
        return bounds.Width / bounds.Height;
    }

    void updateOverlay() {
        ++statsFrames;
        const double elapsed = secondsSince(statsStart);
        if (elapsed < 0.5 || !overlay) return;

        const auto bounds = CoreWindow::GetForCurrentThread().Bounds();
        char line[64];
        snprintf(line, sizeof(line), "%uX%u %.1f FPS", static_cast<uint32_t>(bounds.Width),
                 static_cast<uint32_t>(bounds.Height), statsFrames / elapsed);
        overlay->setLines({line});

        statsFrames = 0;
        statsStart = std::chrono::steady_clock::now();
    }

    void renderFrame() {
        const auto now = std::chrono::steady_clock::now();
        const double dt = std::min(std::chrono::duration<double>(now - lastFrame).count(), 0.25);
        lastFrame = now;

        handleGamepad(static_cast<float>(dt));
        if (!orbitPaused) camera.orbit(kOrbitRadiansPerSecond * static_cast<float>(dt), 0.0f);
        float view[16];
        float projection[16];
        camera.viewMatrix(view);
        camera.projectionMatrix(windowAspect(), projection);

        renderer->update(dt);
        renderer->setCameraMatrices(view, projection);
        updateOverlay();
        renderer->render();

        // The first frame includes lazy setup, so it's timed on its own.
        if (frames == 0) {
            logLine("first frame submitted");
            fpsStart = std::chrono::steady_clock::now();
        } else if (frames % 300 == 0) {
            logLine("frame %u: %.1f fps", frames, 300.0 / secondsSince(fpsStart));
            fpsStart = std::chrono::steady_clock::now();
        }
        ++frames;
    }

    bool closed = false;
    std::shared_ptr<gpu::Device> device;
    std::shared_ptr<rend::Renderer> renderer;
    std::unique_ptr<Overlay> overlay;
    uint32_t width = 1920;
    uint32_t height = 1080;
    std::chrono::steady_clock::time_point statsStart;
    unsigned statsFrames = 0;
    Camera camera;
    std::chrono::steady_clock::time_point lastFrame;
    std::chrono::steady_clock::time_point fpsStart;
    unsigned frames = 0;
    float loggedWidth = 0.0f;
    float loggedHeight = 0.0f;
    size_t viewModeIndex = 0;
    bool orbitPaused = false;
    winrt::Windows::Gaming::Input::GamepadButtons previousButtons{};
};

int __stdcall wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
    init_apartment();
    CoreApplication::Run(make<App>());
    return 0;
}
