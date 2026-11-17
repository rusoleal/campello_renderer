#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHADER_DIR="$SCRIPT_DIR/shaders/vulkan"

# ---------------------------------------------------------------------------
# Locate glslc — prefer NDK bundled version, fall back to PATH.
# ---------------------------------------------------------------------------
if [ -n "$ANDROID_NDK" ]; then
    case "$(uname -s)" in
        Darwin) GLSLC="$ANDROID_NDK/shader-tools/darwin-x86_64/glslc" ;;
        Linux)  GLSLC="$ANDROID_NDK/shader-tools/linux-x86_64/glslc"  ;;
        *)      GLSLC="glslc" ;;
    esac
else
    GLSLC="glslc"
fi

if ! command -v "$GLSLC" &>/dev/null && [ ! -x "$GLSLC" ]; then
    if command -v glslangValidator &>/dev/null; then
        # Same GLSL frontend as glslc (shaderc wraps glslang); -V = Vulkan SPIR-V.
        GLSLC_WRAPPER="$(mktemp)"
        printf '#!/bin/bash\nexec glslangValidator -V "$1" -o "$3" > /dev/null\n' > "$GLSLC_WRAPPER"
        chmod +x "$GLSLC_WRAPPER"
        GLSLC="$GLSLC_WRAPPER"
    else
        echo "Error: neither glslc nor glslangValidator found. Set ANDROID_NDK or install one."
        exit 1
    fi
fi

echo "Using glslc: $GLSLC"

# ---------------------------------------------------------------------------
# Compile GLSL → SPIR-V and embed it. Each entry: source, .spv, C symbol
# (the header has <symbol>[] and <symbol>Size), and the header under src/shaders/.
# ---------------------------------------------------------------------------
SHADERS=(
    "default.vert               default_vert.spv               kDefaultVulkanVertShader               vulkan_default.h"
    "default_skinned.vert       default_skinned_vert.spv       kDefaultVulkanSkinnedVertShader        vulkan_default.h"
    "default.frag               default_frag.spv               kDefaultVulkanFragShader               vulkan_default.h"
    "default_lean.vert          default_lean_vert.spv          kDefaultVulkanLeanVertShader           vulkan_default.h"
    "default_line.vert          default_line_vert.spv          kDefaultVulkanLineVertShader           vulkan_default.h"
    "fxaa.vert                  fxaa_vert.spv                  kDefaultVulkanFxaaVertShader           vulkan_default.h"
    "fxaa.frag                  fxaa_frag.spv                  kDefaultVulkanFxaaFragShader           vulkan_default.h"
    "downsample.frag            downsample_frag.spv            kDefaultVulkanDownsampleFragShader     vulkan_default.h"
    "procedural_bake.comp       procedural_bake.spv            kDefaultVulkanProceduralBakeShader     vulkan_default.h"
    "ibl_bake.frag              ibl_bake_frag.spv              kDefaultVulkanIblBakeFragShader        vulkan_default.h"
    "skybox.frag                skybox_frag.spv                kDefaultVulkanSkyboxFragShader         vulkan_default.h"
    "ssao.frag                  ssao_frag.spv                  kDefaultVulkanSsaoFragShader           vulkan_default.h"
    "ssao_apply.frag            ssao_apply_frag.spv            kDefaultVulkanSsaoApplyFragShader      vulkan_default.h"
    "gbuffer.vert               gbuffer_vert.spv               kDefaultVulkanGBufferVertShader        vulkan_default.h"
    "gbuffer_skinned.vert       gbuffer_skinned_vert.spv       kDefaultVulkanGBufferSkinnedVertShader vulkan_default.h"
    "gbuffer_lean.vert          gbuffer_lean_vert.spv          kDefaultVulkanGBufferLeanVertShader    vulkan_default.h"
    "gbuffer.frag               gbuffer_frag.spv               kDefaultVulkanGBufferFragShader        vulkan_default.h"
    "taa.frag                   taa_frag.spv                   kDefaultVulkanTaaFragShader            vulkan_default.h"
    "fxaa.vert                  fxaa_vert.spv                  kDenoiserVulkanVertShader              vulkan_denoiser.h"
    "denoise_temporal.frag      denoise_temporal_frag.spv      kDenoiserVulkanTemporalShader          vulkan_denoiser.h"
    "denoise_variance.frag      denoise_variance_frag.spv      kDenoiserVulkanVarianceShader          vulkan_denoiser.h"
    "denoise_atrous.frag        denoise_atrous_frag.spv        kDenoiserVulkanAtrousShader            vulkan_denoiser.h"
    "denoise_atrous_first.frag  denoise_atrous_first_frag.spv  kDenoiserVulkanAtrousFirstShader       vulkan_denoiser.h"
    "denoise_modulate.frag      denoise_modulate_frag.spv      kDenoiserVulkanModulateShader          vulkan_denoiser.h"
)

ENTRIES=()
for entry in "${SHADERS[@]}"; do
    read -r src spv symbol header <<< "$entry"
    "$GLSLC" "$SHADER_DIR/$src" -o "$SHADER_DIR/$spv"
    echo "Compiled: $src → $spv"
    ENTRIES+=("$SHADER_DIR/$spv" "$symbol" "$SCRIPT_DIR/src/shaders/$header")
done

python3 - "${ENTRIES[@]}" <<'PYEOF'
import sys

triples = list(zip(sys.argv[1::3], sys.argv[2::3], sys.argv[3::3]))

def spv_to_hex_lines(path):
    data = open(path, "rb").read()
    lines = []
    for i in range(0, len(data), 12):
        chunk = data[i:i+12]
        lines.append("    " + ", ".join(f"0x{b:02x}" for b in chunk) + ",")
    return "\n".join(lines), len(data)

headers = {}
for path, symbol, header in triples:
    headers.setdefault(header, []).append((path, symbol))
for header, shaders in headers.items():
    parts = ["""#pragma once

// Auto-generated by build_vulkan_shaders.sh — do not edit.
// Rebuild with: ./build_vulkan_shaders.sh
//
// glslc is available at:
//   $ANDROID_NDK/shader-tools/darwin-x86_64/glslc   (macOS host)
//   $ANDROID_NDK/shader-tools/linux-x86_64/glslc    (Linux host)

namespace systems::leal::campello_renderer::shaders {
"""]
    for path, symbol in shaders:
        hex_lines, size = spv_to_hex_lines(path)
        parts.append(f"static const unsigned char {symbol}[] = {{\n{hex_lines}\n}};\n"
                     f"static const unsigned int {symbol}Size = {size};\n")
        print(f"  {symbol}: {size} bytes")
    parts.append("} // namespace systems::leal::campello_renderer::shaders\n")
    open(header, "w").write("\n".join(parts))
    print(f"Written: {header}")
PYEOF
