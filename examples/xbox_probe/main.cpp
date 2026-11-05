// xbox_probe -- minimal UWP app that answers one question before any porting
// work starts: can campello_gpu's D3D12 backend, and the renderer's embedded
// DXIL, work on an Xbox Series X in Developer Mode?
//
// It deliberately links neither campello_gpu nor campello_renderer (neither
// builds for UWP yet). It repeats the D3D12 calls those libraries make, with
// the same parameters, and logs each result.
//
// Results go to the VS Output window (OutputDebugString) and to
// LocalState\probe.log. The screen colour is the summary:
//   splash screen stays  no D3D12 device or no swapchain, read the log
//   green                everything the backend needs worked at FL 11_0
//   amber                device + present work, a backend requirement failed
//   magenta              running on the WARP software rasterizer

#include <unknwn.h>

#include <winrt/Windows.ApplicationModel.Core.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Security.ExchangeActiveSyncProvisioning.h>
#include <winrt/Windows.Storage.h>
#include <winrt/Windows.System.h>
#include <winrt/Windows.System.Profile.h>
#include <winrt/Windows.UI.Core.h>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <windows.h>

#include <chrono>
#include <climits>
#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

// The exact DXIL campello_renderer ships, so a pass here is about the real shaders.
#include "../../src/shaders/directx_default.h"

using namespace winrt;
using namespace winrt::Windows::ApplicationModel::Core;
using namespace winrt::Windows::UI::Core;
namespace shaders = systems::leal::campello_renderer::shaders;

namespace {

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------
std::ofstream gLogFile;
int gFailures = 0;

void logLine(const char* fmt, ...) {
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    std::string line = std::string("[probe] ") + buf + "\n";
    OutputDebugStringA(line.c_str());
    if (gLogFile.is_open()) {
        gLogFile << line;
        gLogFile.flush(); // survive a crash or termination mid-probe
    }
}

// A check the backend depends on: a failure here turns the screen amber.
bool require(const char* what, HRESULT hr) {
    logLine("%-52s %s (hr=0x%08X)", what, SUCCEEDED(hr) ? "PASS" : "FAIL",
            static_cast<unsigned>(hr));
    if (FAILED(hr)) ++gFailures;
    return SUCCEEDED(hr);
}

// Informational only: failing is an expected, useful answer.
bool report(const char* what, HRESULT hr) {
    logLine("%-52s %s (hr=0x%08X)", what, SUCCEEDED(hr) ? "yes" : "no",
            static_cast<unsigned>(hr));
    return SUCCEEDED(hr);
}

const char* featureLevelName(D3D_FEATURE_LEVEL level) {
    switch (level) {
        case D3D_FEATURE_LEVEL_12_2: return "12_2";
        case D3D_FEATURE_LEVEL_12_1: return "12_1";
        case D3D_FEATURE_LEVEL_12_0: return "12_0";
        case D3D_FEATURE_LEVEL_11_1: return "11_1";
        case D3D_FEATURE_LEVEL_11_0: return "11_0";
        default:                     return "?";
    }
}

// ---------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------
void probeEnvironment() {
    logLine("--- environment ---");
    try {
        using namespace winrt::Windows::System::Profile;
        using winrt::Windows::Security::ExchangeActiveSyncProvisioning::EasClientDeviceInformation;

        auto version = AnalyticsInfo::VersionInfo();
        logLine("device family:   %s", to_string(version.DeviceFamily()).c_str());
        logLine("family version:  %s", to_string(version.DeviceFamilyVersion()).c_str());

        EasClientDeviceInformation eas;
        logLine("product name:    %s", to_string(eas.SystemProductName()).c_str());
    } catch (const hresult_error& e) {
        logLine("environment query failed: hr=0x%08X", static_cast<unsigned>(e.code().value));
    }
    // There is no API for "am I in Game mode". This limit is the best indicator,
    // so compare it between a run in App mode and a run in Game mode.
    logLine("app memory limit: %llu MB",
            winrt::Windows::System::MemoryManager::AppMemoryUsageLimit() / (1024ull * 1024ull));
}

// ---------------------------------------------------------------------------
// Adapter selection
// ---------------------------------------------------------------------------
com_ptr<IDXGIAdapter1> probeAdapters(const com_ptr<IDXGIFactory4>& factory) {
    logLine("--- adapters ---");
    com_ptr<IDXGIAdapter1> first;
    com_ptr<IDXGIAdapter1> adapter;
    for (UINT i = 0; factory->EnumAdapters1(i, adapter.put()) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc = {};
        adapter->GetDesc1(&desc);
        logLine("adapter %u: %s vendor=0x%04X device=0x%04X vram=%llu MB shared=%llu MB%s", i,
                to_string(desc.Description).c_str(), desc.VendorId, desc.DeviceId,
                static_cast<unsigned long long>(desc.DedicatedVideoMemory) / (1024ull * 1024ull),
                static_cast<unsigned long long>(desc.SharedSystemMemory) / (1024ull * 1024ull),
                (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) ? " [SOFTWARE]" : "");
        if (!first) first = adapter;
        adapter = nullptr;
    }

    // campello_gpu's selectDefaultAdapter() prefers IDXGIFactory6.
    auto factory6 = factory.try_as<IDXGIFactory6>();
    report("IDXGIFactory6 available", factory6 ? S_OK : E_NOINTERFACE);
    if (factory6) {
        com_ptr<IDXGIAdapter1> preferred;
        HRESULT hr = factory6->EnumAdapterByGpuPreference(
            0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(preferred.put()));
        report("EnumAdapterByGpuPreference(HIGH_PERFORMANCE)", hr);
        if (SUCCEEDED(hr)) return preferred;
    }
    return first;
}

// ---------------------------------------------------------------------------
// Feature levels: campello_gpu calls D3D12CreateDevice with 12_0 today
// (device.cpp:208, adapter.cpp:25). This is the main question of the probe.
// ---------------------------------------------------------------------------
com_ptr<ID3D12Device> probeFeatureLevels(IDXGIAdapter1* adapter) {
    logLine("--- D3D12CreateDevice per feature level ---");
    const D3D_FEATURE_LEVEL levels[] = {
        D3D_FEATURE_LEVEL_12_2, D3D_FEATURE_LEVEL_12_1, D3D_FEATURE_LEVEL_12_0,
        D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0,
    };
    for (D3D_FEATURE_LEVEL level : levels) {
        com_ptr<ID3D12Device> device;
        HRESULT hr = D3D12CreateDevice(adapter, level, IID_PPV_ARGS(device.put()));
        char label[64];
        snprintf(label, sizeof(label), "D3D12CreateDevice at FL %s", featureLevelName(level));
        report(label, hr);
    }

    // The rest of the probe uses the lowest level, which is what the port would request.
    com_ptr<ID3D12Device> device;
    HRESULT hr = D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(device.put()));
    require("device for the remaining checks (FL 11_0)", hr);
    return SUCCEEDED(hr) ? device : nullptr;
}

// ---------------------------------------------------------------------------
// Capabilities
// ---------------------------------------------------------------------------
void probeCapabilities(ID3D12Device* device) {
    logLine("--- capabilities ---");

    D3D12_FEATURE_DATA_FEATURE_LEVELS levels = {};
    const D3D_FEATURE_LEVEL requested[] = {
        D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_12_0,
        D3D_FEATURE_LEVEL_12_1, D3D_FEATURE_LEVEL_12_2,
    };
    levels.NumFeatureLevels = static_cast<UINT>(std::size(requested));
    levels.pFeatureLevelsRequested = requested;
    if (SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FEATURE_LEVELS, &levels, sizeof(levels))))
        logLine("max feature level: %s", featureLevelName(levels.MaxSupportedFeatureLevel));

    // Ask from the top down: a runtime that doesn't know a model returns E_INVALIDARG.
    const int models[] = {0x67, 0x66, 0x65, 0x64, 0x63, 0x62, 0x61, 0x60, 0x51};
    bool foundModel = false;
    for (int model : models) {
        D3D12_FEATURE_DATA_SHADER_MODEL sm = {static_cast<D3D_SHADER_MODEL>(model)};
        if (SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm, sizeof(sm)))) {
            logLine("highest shader model: %X.%X", (sm.HighestShaderModel >> 4) & 0xF,
                    sm.HighestShaderModel & 0xF);
            // The renderer's DXIL is compiled as vs/ps/cs_6_0.
            require("shader model >= 6.0 (renderer's DXIL)",
                    sm.HighestShaderModel >= 0x60 ? S_OK : E_FAIL);
            foundModel = true;
            break;
        }
    }
    if (!foundModel) require("shader model query", E_FAIL);

    D3D12_FEATURE_DATA_ROOT_SIGNATURE rootSig = {D3D_ROOT_SIGNATURE_VERSION_1_1};
    if (SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &rootSig, sizeof(rootSig))))
        logLine("root signature version: 0x%X", rootSig.HighestVersion);
    else
        logLine("root signature version: query failed (treat as 1.0)");

    D3D12_FEATURE_DATA_D3D12_OPTIONS o = {};
    if (SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &o, sizeof(o)))) {
        logLine("resource binding tier: %d", o.ResourceBindingTier);
        logLine("resource heap tier: %d", o.ResourceHeapTier);
        logLine("tiled resources tier: %d", o.TiledResourcesTier);
        logLine("conservative raster tier: %d", o.ConservativeRasterizationTier);
        logLine("ROVs: %d, typed UAV load extra formats: %d, double precision: %d",
                o.ROVsSupported, o.TypedUAVLoadAdditionalFormats, o.DoublePrecisionFloatShaderOps);
    }

    D3D12_FEATURE_DATA_D3D12_OPTIONS1 o1 = {};
    if (SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &o1, sizeof(o1))))
        logLine("wave ops: %d (lanes %u-%u)", o1.WaveOps, o1.WaveLaneCountMin, o1.WaveLaneCountMax);

    D3D12_FEATURE_DATA_D3D12_OPTIONS4 o4 = {};
    if (SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS4, &o4, sizeof(o4))))
        logLine("native 16-bit shader ops: %d", o4.Native16BitShaderOpsSupported);

    D3D12_FEATURE_DATA_D3D12_OPTIONS5 o5 = {};
    if (SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &o5, sizeof(o5))))
        logLine("raytracing tier: %d, render passes tier: %d", o5.RaytracingTier, o5.RenderPassesTier);

    D3D12_FEATURE_DATA_D3D12_OPTIONS7 o7 = {};
    if (SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &o7, sizeof(o7))))
        logLine("mesh shader tier: %d, sampler feedback tier: %d", o7.MeshShaderTier, o7.SamplerFeedbackTier);

    D3D12_FEATURE_DATA_ARCHITECTURE1 arch = {};
    if (SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE1, &arch, sizeof(arch))))
        logLine("UMA: %d, cache coherent UMA: %d", arch.UMA, arch.CacheCoherentUMA);

    // Interfaces the backend queries at runtime.
    com_ptr<ID3D12Device> ref;
    ref.copy_from(device);
    report("ID3D12Device5 (raytracing paths)", ref.try_as<ID3D12Device5>() ? S_OK : E_NOINTERFACE);
}

// ---------------------------------------------------------------------------
// Formats the renderer creates (PixelFormat:: uses in campello_renderer.cpp)
// ---------------------------------------------------------------------------
void probeFormats(ID3D12Device* device) {
    logLine("--- formats ---");
    struct Entry {
        DXGI_FORMAT format;
        const char* name;
        D3D12_FORMAT_SUPPORT1 needed;
    };
    const auto tex = D3D12_FORMAT_SUPPORT1_TEXTURE2D;
    const auto sample = D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE;
    const auto rt = D3D12_FORMAT_SUPPORT1_RENDER_TARGET;
    const Entry entries[] = {
        {DXGI_FORMAT_R8G8B8A8_UNORM, "rgba8unorm", tex | sample | rt},
        {DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, "rgba8unorm_srgb", tex | sample},
        {DXGI_FORMAT_R16G16B16A16_FLOAT, "rgba16float", tex | sample | rt},
        // Not required to be filterable at 11_0: only needs to exist. Check the
        // SHADER_SAMPLE bit in the log if an HDR environment map looks wrong.
        {DXGI_FORMAT_R32G32B32A32_FLOAT, "rgba32float", tex},
        {DXGI_FORMAT_D32_FLOAT, "depth32float", tex | D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL},
        {DXGI_FORMAT_BC7_UNORM, "bc7unorm (compressed textures)", tex | sample},
    };
    for (const Entry& entry : entries) {
        D3D12_FEATURE_DATA_FORMAT_SUPPORT support = {entry.format};
        HRESULT hr = device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support));
        if (SUCCEEDED(hr) && (support.Support1 & entry.needed) != entry.needed) hr = E_FAIL;
        char label[96];
        snprintf(label, sizeof(label), "%s [s1=0x%08X s2=0x%08X]", entry.name,
                 static_cast<unsigned>(support.Support1), static_cast<unsigned>(support.Support2));
        require(label, hr);
    }

    const DXGI_FORMAT msaaFormats[] = {DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R16G16B16A16_FLOAT,
                                       DXGI_FORMAT_D32_FLOAT};
    for (DXGI_FORMAT format : msaaFormats) {
        D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS msaa = {};
        msaa.Format = format;
        msaa.SampleCount = 4;
        if (SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &msaa, sizeof(msaa))))
            logLine("4x MSAA, DXGI format %d: %u quality levels", format, msaa.NumQualityLevels);
    }
}

// ---------------------------------------------------------------------------
// Descriptor heaps, same sizes as createDeviceData() in device.cpp
// ---------------------------------------------------------------------------
void probeHeaps(ID3D12Device* device) {
    logLine("--- descriptor heaps ---");
    D3D12_DESCRIPTOR_HEAP_DESC desc = {};

    desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    desc.NumDescriptors = 65536;
    desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    com_ptr<ID3D12DescriptorHeap> srvHeap;
    require("shader-visible CBV/SRV/UAV heap, 65536",
            device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(srvHeap.put())));

    desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
    desc.NumDescriptors = 2048;
    com_ptr<ID3D12DescriptorHeap> samplerHeap;
    require("shader-visible sampler heap, 2048",
            device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(samplerHeap.put())));
}

// ---------------------------------------------------------------------------
// The real PBR pipeline: same bindings as ensureDirectXPbrBindGroupLayout(),
// same root signature shape as createRootSignature() in device.cpp.
// ---------------------------------------------------------------------------
com_ptr<ID3D12RootSignature> createPbrRootSignature(ID3D12Device* device) {
    auto makeRange = [](D3D12_DESCRIPTOR_RANGE_TYPE type, UINT binding) {
        D3D12_DESCRIPTOR_RANGE1 range = {};
        range.RangeType = type;
        range.NumDescriptors = 1;
        range.BaseShaderRegister = binding;
        range.RegisterSpace = 0;
        range.Flags = D3D12_DESCRIPTOR_RANGE_FLAG_NONE;
        range.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
        return range;
    };

    std::vector<D3D12_DESCRIPTOR_RANGE1> resources;
    std::vector<D3D12_DESCRIPTOR_RANGE1> samplers;
    const UINT srvBindings[] = {0,  2,  4,  6,  8,  10, 12, 14, 15, 16, 17, 18,
                                19, 20, 21, 22, 23, 32, 34, 36, 37};
    const UINT samplerBindings[] = {1, 3, 5, 7, 9, 11, 13, 33, 35, 38};
    const UINT cbvBindings[] = {24, 30, 31};
    for (UINT b : srvBindings) resources.push_back(makeRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, b));
    for (UINT b : cbvBindings) resources.push_back(makeRange(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, b));
    for (UINT b : samplerBindings) samplers.push_back(makeRange(D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, b));

    D3D12_ROOT_PARAMETER1 params[2] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].DescriptorTable.NumDescriptorRanges = static_cast<UINT>(resources.size());
    params[0].DescriptorTable.pDescriptorRanges = resources.data();
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = static_cast<UINT>(samplers.size());
    params[1].DescriptorTable.pDescriptorRanges = samplers.data();
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc = {};
    desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    desc.Desc_1_1.NumParameters = 2;
    desc.Desc_1_1.pParameters = params;
    desc.Desc_1_1.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

    com_ptr<ID3DBlob> blob;
    com_ptr<ID3DBlob> error;
    HRESULT hr = D3D12SerializeVersionedRootSignature(&desc, blob.put(), error.put());
    require("serialize PBR root signature (v1.1)", hr);
    if (error) logLine("serializer: %s", static_cast<const char*>(error->GetBufferPointer()));
    if (FAILED(hr)) return nullptr;

    com_ptr<ID3D12RootSignature> rootSignature;
    hr = device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                     IID_PPV_ARGS(rootSignature.put()));
    require("create PBR root signature (39 bindings, s0-s38)", hr);
    return SUCCEEDED(hr) ? rootSignature : nullptr;
}

void probePipeline(ID3D12Device* device) {
    logLine("--- renderer's PBR pipeline from embedded DXIL ---");
    auto rootSignature = createPbrRootSignature(device);
    if (!rootSignature) return;

    auto element = [](UINT location, DXGI_FORMAT format, UINT slot, UINT offset, bool instanced) {
        D3D12_INPUT_ELEMENT_DESC e = {};
        e.SemanticName = "TEXCOORD"; // campello_gpu: semantic index = shader location
        e.SemanticIndex = location;
        e.Format = format;
        e.InputSlot = slot;
        e.AlignedByteOffset = offset;
        e.InputSlotClass = instanced ? D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA
                                     : D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA;
        e.InstanceDataStepRate = instanced ? 1 : 0;
        return e;
    };
    const DXGI_FORMAT f2 = DXGI_FORMAT_R32G32_FLOAT;
    const DXGI_FORMAT f3 = DXGI_FORMAT_R32G32B32_FLOAT;
    const DXGI_FORMAT f4 = DXGI_FORMAT_R32G32B32A32_FLOAT;
    // VertexIn in shaders/directx/default.hlsl.
    const D3D12_INPUT_ELEMENT_DESC layout[] = {
        element(0, f3, 0, 0, false),   element(1, f3, 1, 0, false),
        element(2, f2, 2, 0, false),   element(3, f4, 3, 0, false),
        element(6, f4, 6, 0, false),   element(7, f2, 7, 0, false),
        element(16, f4, 16, 0, true),  element(17, f4, 16, 16, true),
        element(18, f4, 16, 32, true), element(19, f4, 16, 48, true),
        element(24, f4, 16, 64, true), element(25, f4, 16, 80, true),
        element(26, f4, 16, 96, true), element(27, f4, 16, 112, true),
    };

    struct Variant {
        const char* name;
        DXGI_FORMAT color;
        UINT samples;
    };
    const Variant variants[] = {
        {"PBR PSO, rgba8 + depth32, 1x", DXGI_FORMAT_R8G8B8A8_UNORM, 1},
        {"PBR PSO, rgba16float + depth32, 1x", DXGI_FORMAT_R16G16B16A16_FLOAT, 1},
        {"PBR PSO, rgba8 + depth32, 4x MSAA", DXGI_FORMAT_R8G8B8A8_UNORM, 4},
    };
    for (const Variant& variant : variants) {
        D3D12_GRAPHICS_PIPELINE_STATE_DESC desc = {};
        desc.pRootSignature = rootSignature.get();
        desc.VS = {shaders::kDefaultDirectXVertShader, shaders::kDefaultDirectXVertShaderSize};
        desc.PS = {shaders::kDefaultDirectXPixelShader, shaders::kDefaultDirectXPixelShaderSize};
        desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        desc.BlendState.RenderTarget[0].SrcBlend = D3D12_BLEND_ONE;
        desc.BlendState.RenderTarget[0].DestBlend = D3D12_BLEND_ZERO;
        desc.BlendState.RenderTarget[0].BlendOp = D3D12_BLEND_OP_ADD;
        desc.BlendState.RenderTarget[0].SrcBlendAlpha = D3D12_BLEND_ONE;
        desc.BlendState.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_ZERO;
        desc.BlendState.RenderTarget[0].BlendOpAlpha = D3D12_BLEND_OP_ADD;
        desc.SampleMask = UINT_MAX;
        desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        desc.RasterizerState.CullMode = D3D12_CULL_MODE_BACK;
        desc.RasterizerState.FrontCounterClockwise = TRUE;
        desc.RasterizerState.DepthClipEnable = TRUE;
        desc.DepthStencilState.DepthEnable = TRUE;
        desc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
        desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
        desc.InputLayout = {layout, static_cast<UINT>(std::size(layout))};
        desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        desc.NumRenderTargets = 1;
        desc.RTVFormats[0] = variant.color;
        desc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
        desc.SampleDesc.Count = variant.samples;

        com_ptr<ID3D12PipelineState> pso;
        HRESULT hr = device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(pso.put()));
        // MSAA is optional for the renderer, so it doesn't count against the verdict.
        if (variant.samples == 1) require(variant.name, hr);
        else report(variant.name, hr);
    }
}

void probeMemoryBudget(IDXGIAdapter1* adapter) {
    logLine("--- memory budget ---");
    com_ptr<IDXGIAdapter1> ref;
    ref.copy_from(adapter);
    auto adapter3 = ref.try_as<IDXGIAdapter3>();
    if (!adapter3) {
        logLine("IDXGIAdapter3 unavailable");
        return;
    }
    DXGI_QUERY_VIDEO_MEMORY_INFO info = {};
    if (SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info)))
        logLine("local video memory budget: %llu MB",
                static_cast<unsigned long long>(info.Budget) / (1024ull * 1024ull));
}

bool adapterIsSoftware(IDXGIAdapter1* adapter) {
    DXGI_ADAPTER_DESC1 desc = {};
    adapter->GetDesc1(&desc);
    return (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0;
}

} // namespace

// ---------------------------------------------------------------------------
// The app: runs the probe once, then clears and presents every frame, which
// proves the CoreWindow swapchain path the port will need.
// ---------------------------------------------------------------------------
struct ProbeApp : implements<ProbeApp, IFrameworkViewSource, IFrameworkView> {
    static constexpr UINT kFrameCount = 2; // DeviceData::kFrameCount in campello_gpu

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

        openLog();
        const bool canRender = runProbe(window);

        while (!closed) {
            if (canRender && renderFrame()) {
                dispatcher.ProcessEvents(CoreProcessEventsOption::ProcessAllIfPresent);
            } else {
                // Nothing to draw, or the device was lost: just stay alive to be closed.
                dispatcher.ProcessEvents(CoreProcessEventsOption::ProcessOneAndAllPending);
            }
        }
        waitForGpu();
    }

private:
    void openLog() {
        try {
            std::wstring path{winrt::Windows::Storage::ApplicationData::Current().LocalFolder().Path()};
            path += L"\\probe.log";
            gLogFile.open(path, std::ios::out | std::ios::trunc);
            logLine("log file: %s", to_string(path).c_str());
        } catch (const hresult_error& e) {
            logLine("could not open log file: hr=0x%08X", static_cast<unsigned>(e.code().value));
        }
    }

    bool runProbe(CoreWindow const& window) {
        probeEnvironment();

        com_ptr<IDXGIFactory4> factory;
        if (!require("CreateDXGIFactory2", CreateDXGIFactory2(0, IID_PPV_ARGS(factory.put())))) return false;

        auto adapter = probeAdapters(factory);
        if (!adapter) {
            logLine("VERDICT: no DXGI adapter");
            return false;
        }

        device = probeFeatureLevels(adapter.get());
        if (!device) {
            logLine("VERDICT: BLOCKED, no D3D12 device at FL 11_0");
            return false;
        }

        software = adapterIsSoftware(adapter.get());
        logLine("running on: %s", software ? "SOFTWARE ADAPTER (WARP)" : "hardware adapter");

        probeCapabilities(device.get());
        probeFormats(device.get());
        probeHeaps(device.get());
        probePipeline(device.get());
        probeMemoryBudget(adapter.get());

        if (!createSwapChain(factory.get(), window)) {
            logLine("VERDICT: BLOCKED, no CoreWindow swapchain");
            return false;
        }

        logLine("--- result ---");
        logLine("VERDICT: %s, %d required check(s) failed%s",
                gFailures == 0 ? "OK" : "PROBLEMS", gFailures, software ? ", on WARP" : "");
        return true;
    }

    bool createSwapChain(IDXGIFactory4* factory, CoreWindow const& window) {
        logLine("--- CoreWindow swapchain ---");
        D3D12_COMMAND_QUEUE_DESC queueDesc = {};
        queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        if (!require("create direct command queue",
                     device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(queue.put()))))
            return false;

        // Same format, count and swap effect as createDeviceData() in device.cpp.
        // A CoreWindow swapchain needs an explicit size and stretch scaling. 4K is
        // what Series X should use; 1080p is what the UWP window reports.
        DXGI_SWAP_CHAIN_DESC1 desc = {};
        desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        desc.BufferCount = kFrameCount;
        desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        desc.SampleDesc.Count = 1;
        desc.Scaling = DXGI_SCALING_STRETCH;

        const UINT sizes[][2] = {{3840, 2160}, {1920, 1080}};
        com_ptr<IDXGISwapChain1> swapChain1;
        for (const auto& size : sizes) {
            desc.Width = size[0];
            desc.Height = size[1];
            char label[64];
            snprintf(label, sizeof(label), "CreateSwapChainForCoreWindow %ux%u", size[0], size[1]);
            HRESULT hr = factory->CreateSwapChainForCoreWindow(queue.get(), winrt::get_unknown(window), &desc,
                                                               nullptr, swapChain1.put());
            report(label, hr);
            if (SUCCEEDED(hr)) break;
            swapChain1 = nullptr;
        }
        if (!swapChain1) {
            ++gFailures;
            return false;
        }
        swapChain = swapChain1.try_as<IDXGISwapChain3>();
        if (!swapChain) {
            require("IDXGISwapChain3", E_NOINTERFACE);
            return false;
        }

        D3D12_DESCRIPTOR_HEAP_DESC rtvDesc = {};
        rtvDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        rtvDesc.NumDescriptors = kFrameCount;
        if (!require("create RTV heap", device->CreateDescriptorHeap(&rtvDesc, IID_PPV_ARGS(rtvHeap.put()))))
            return false;
        rtvStride = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

        D3D12_CPU_DESCRIPTOR_HANDLE rtv = rtvHeap->GetCPUDescriptorHandleForHeapStart();
        for (UINT i = 0; i < kFrameCount; ++i) {
            if (!require("get swapchain buffer", swapChain->GetBuffer(i, IID_PPV_ARGS(backBuffers[i].put()))))
                return false;
            device->CreateRenderTargetView(backBuffers[i].get(), nullptr, rtv);
            rtv.ptr += rtvStride;
        }

        if (!require("create command allocator",
                     device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                    IID_PPV_ARGS(allocator.put()))))
            return false;
        if (!require("create command list",
                     device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.get(), nullptr,
                                               IID_PPV_ARGS(commandList.put()))))
            return false;
        commandList->Close();
        if (!require("create fence", device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence.put()))))
            return false;
        fenceEvent = CreateEventExW(nullptr, nullptr, 0, EVENT_ALL_ACCESS);
        return fenceEvent != nullptr;
    }

    bool renderFrame() {
        const UINT index = swapChain->GetCurrentBackBufferIndex();

        allocator->Reset();
        commandList->Reset(allocator.get(), nullptr);

        auto barrier = [&](D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
            D3D12_RESOURCE_BARRIER b = {};
            b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            b.Transition.pResource = backBuffers[index].get();
            b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            b.Transition.StateBefore = before;
            b.Transition.StateAfter = after;
            commandList->ResourceBarrier(1, &b);
        };

        barrier(D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
        D3D12_CPU_DESCRIPTOR_HANDLE rtv = rtvHeap->GetCPUDescriptorHandleForHeapStart();
        rtv.ptr += static_cast<SIZE_T>(index) * rtvStride;
        const FLOAT green[4] = {0.05f, 0.65f, 0.15f, 1.0f};
        const FLOAT amber[4] = {0.90f, 0.55f, 0.05f, 1.0f};
        const FLOAT magenta[4] = {0.80f, 0.05f, 0.70f, 1.0f};
        commandList->ClearRenderTargetView(rtv, software ? magenta : (gFailures == 0 ? green : amber), 0, nullptr);
        barrier(D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
        commandList->Close();

        ID3D12CommandList* lists[] = {commandList.get()};
        queue->ExecuteCommandLists(1, lists);

        HRESULT hr = swapChain->Present(1, 0);
        if (FAILED(hr)) {
            logLine("Present failed: hr=0x%08X, device removed reason=0x%08X",
                    static_cast<unsigned>(hr), static_cast<unsigned>(device->GetDeviceRemovedReason()));
            ++gFailures;
            return false;
        }
        waitForGpu();

        // Log once so the log shows frames really reach the display, and at what rate.
        using clock = std::chrono::steady_clock;
        if (frames == 0) firstFrame = clock::now();
        if (++frames == 301) {
            const double seconds = std::chrono::duration<double>(clock::now() - firstFrame).count();
            logLine("presented 300 frames in %.2f s (%.1f fps)", seconds, 300.0 / seconds);
        }
        return true;
    }

    void waitForGpu() {
        if (!queue || !fence || !fenceEvent) return;
        const UINT64 value = ++fenceValue;
        if (FAILED(queue->Signal(fence.get(), value))) return;
        if (fence->GetCompletedValue() < value && SUCCEEDED(fence->SetEventOnCompletion(value, fenceEvent)))
            WaitForSingleObjectEx(fenceEvent, 2000, FALSE);
    }

    bool closed = false;
    bool software = false;
    com_ptr<ID3D12Device> device;
    com_ptr<ID3D12CommandQueue> queue;
    com_ptr<IDXGISwapChain3> swapChain;
    com_ptr<ID3D12DescriptorHeap> rtvHeap;
    UINT rtvStride = 0;
    com_ptr<ID3D12Resource> backBuffers[kFrameCount];
    com_ptr<ID3D12CommandAllocator> allocator;
    com_ptr<ID3D12GraphicsCommandList> commandList;
    com_ptr<ID3D12Fence> fence;
    UINT64 fenceValue = 0;
    HANDLE fenceEvent = nullptr;
    unsigned frames = 0;
    std::chrono::steady_clock::time_point firstFrame;
};

int __stdcall wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
    init_apartment();
    CoreApplication::Run(make<ProbeApp>());
    return 0;
}
