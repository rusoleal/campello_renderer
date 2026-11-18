#Requires -Version 5.1
<#
Compiles shaders/directx/default.hlsl to DXIL and regenerates
src/shaders/directx_default.h — the DirectX analogue of
build_vulkan_shaders.sh / build_metal_shaders.sh.

Usage (from repo root, in PowerShell):
    ./build_directx_shaders.ps1

Requires dxc.exe (DirectX Shader Compiler) — ships with the Windows SDK at
  C:\Program Files (x86)\Windows Kits\10\bin\<version>\x64\dxc.exe
or via `winget install Microsoft.DirectXShaderCompiler` / the standalone
DXC release. This script searches the Windows SDK's Bin directory (newest
version first) before falling back to PATH. Header generation is pure
PowerShell (no Python dependency) so this script has no other prerequisites.
#>

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ShaderDir = Join-Path $ScriptDir "shaders\directx"

# ---------------------------------------------------------------------------
# Locate dxc.exe — newest Windows Kits SDK version first, then PATH.
# ---------------------------------------------------------------------------
function Find-Dxc {
    $sdkBinRoot = "C:\Program Files (x86)\Windows Kits\10\bin"
    if (Test-Path $sdkBinRoot) {
        $candidate = Get-ChildItem $sdkBinRoot -Directory -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending |
            ForEach-Object { Join-Path $_.FullName "x64\dxc.exe" } |
            Where-Object { Test-Path $_ } |
            Select-Object -First 1
        if ($candidate) { return $candidate }
    }
    $onPath = Get-Command dxc.exe -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }
    throw "dxc.exe not found. Install the Windows SDK (provides DirectX Shader Compiler) or the standalone DXC release, or add dxc.exe to PATH."
}

$Dxc = Find-Dxc
Write-Host "Using dxc: $Dxc"

# ---------------------------------------------------------------------------
# Compile HLSL -> DXIL, one invocation per entry point (dxc only accepts one
# -E/-T pair per run; each pulls only the globals that entry point actually
# references from the shared default.hlsl file — see that file's header
# comment on why entry points can safely reuse register numbers across
# unrelated pipelines).
# ---------------------------------------------------------------------------
#
# Each entry: the .dxil name, profile, entry point, C symbol (the header has
# <symbol>[] and <symbol>Size), the HLSL source and the header under src/shaders/.
$Entries = @(
    @{ Name = "default_vs";      Profile = "vs_6_0"; Entry = "vertexMain";           CArray = "kDefaultDirectXVertShader";             Source = "default.hlsl";  Header = "directx_default.h" },
    @{ Name = "default_ps";      Profile = "ps_6_0"; Entry = "pixelMain";             CArray = "kDefaultDirectXPixelShader";            Source = "default.hlsl";  Header = "directx_default.h" },
    @{ Name = "fxaa_vs";         Profile = "vs_6_0"; Entry = "fxaaVertex";            CArray = "kDefaultDirectXFxaaVertShader";         Source = "default.hlsl";  Header = "directx_default.h" },
    @{ Name = "fxaa_ps";         Profile = "ps_6_0"; Entry = "fxaaPixel";             CArray = "kDefaultDirectXFxaaPixelShader";        Source = "default.hlsl";  Header = "directx_default.h" },
    @{ Name = "downsample_ps";   Profile = "ps_6_0"; Entry = "downsamplePixel";       CArray = "kDefaultDirectXDownsamplePixelShader";  Source = "default.hlsl";  Header = "directx_default.h" },
    @{ Name = "skybox_ps";       Profile = "ps_6_0"; Entry = "skyboxPixel";           CArray = "kDefaultDirectXSkyboxPixelShader";      Source = "default.hlsl";  Header = "directx_default.h" },
    @{ Name = "iblbake_ps";      Profile = "ps_6_0"; Entry = "iblBakePixel";          CArray = "kDefaultDirectXIblBakePixelShader";     Source = "default.hlsl";  Header = "directx_default.h" },
    @{ Name = "procedural_bake"; Profile = "cs_6_0"; Entry = "proceduralBakeKernel";  CArray = "kDefaultDirectXProceduralBakeShader";   Source = "default.hlsl";  Header = "directx_default.h" },
    @{ Name = "gbuffer_vs";      Profile = "vs_6_0"; Entry = "gbufferVertex";         CArray = "kDefaultDirectXGBufferVertShader";      Source = "default.hlsl";  Header = "directx_default.h" },
    @{ Name = "skinned_vs";      Profile = "vs_6_0"; Entry = "vertexSkinned";         CArray = "kDefaultDirectXSkinnedVertShader";      Source = "default.hlsl";  Header = "directx_default.h" },
    @{ Name = "gbuffer_skinned_vs"; Profile = "vs_6_0"; Entry = "gbufferSkinnedVertex"; CArray = "kDefaultDirectXGBufferSkinnedVertShader"; Source = "default.hlsl";  Header = "directx_default.h" },
    @{ Name = "lean_vs";        Profile = "vs_6_0"; Entry = "leanVertex";            CArray = "kDefaultDirectXLeanVertShader";         Source = "default.hlsl";  Header = "directx_default.h" },
    @{ Name = "shadow_vs";       Profile = "vs_6_0"; Entry = "shadowVertex";          CArray = "kDefaultDirectXShadowVertShader";       Source = "default.hlsl";  Header = "directx_default.h" },
    @{ Name = "shadow_skinned_vs"; Profile = "vs_6_0"; Entry = "shadowSkinnedVertex"; CArray = "kDefaultDirectXShadowSkinnedVertShader"; Source = "default.hlsl";  Header = "directx_default.h" },
    @{ Name = "line_vs";        Profile = "vs_6_0"; Entry = "lineVertex";            CArray = "kDefaultDirectXLineVertShader";         Source = "default.hlsl";  Header = "directx_default.h" },
    @{ Name = "gbuffer_lean_vs";Profile = "vs_6_0"; Entry = "gbufferLeanVertex";     CArray = "kDefaultDirectXGBufferLeanVertShader";  Source = "default.hlsl";  Header = "directx_default.h" },
    @{ Name = "gbuffer_ps";     Profile = "ps_6_0"; Entry = "gbufferPixel";          CArray = "kDefaultDirectXGBufferPixelShader";     Source = "default.hlsl";  Header = "directx_default.h" },
    @{ Name = "ssao_ps";         Profile = "ps_6_0"; Entry = "ssaoPixel";             CArray = "kDefaultDirectXSsaoPixelShader";        Source = "default.hlsl";  Header = "directx_default.h" },
    @{ Name = "ssao_apply_ps";   Profile = "ps_6_0"; Entry = "ssaoApplyPixel";        CArray = "kDefaultDirectXSsaoApplyPixelShader";   Source = "default.hlsl";  Header = "directx_default.h" },
    @{ Name = "taa_ps";         Profile = "ps_6_0"; Entry = "taaPixel";              CArray = "kDefaultDirectXTaaPixelShader";         Source = "default.hlsl";  Header = "directx_default.h" },
    @{ Name = "denoise_vs";      Profile = "vs_6_0"; Entry = "denoiseVertex";         CArray = "kDenoiserDirectXVertShader";            Source = "denoiser.hlsl"; Header = "directx_denoiser.h" },
    @{ Name = "denoise_temporal_ps";     Profile = "ps_6_0"; Entry = "denoiseTemporal";    CArray = "kDenoiserDirectXTemporalShader";    Source = "denoiser.hlsl"; Header = "directx_denoiser.h" },
    @{ Name = "denoise_variance_ps";     Profile = "ps_6_0"; Entry = "denoiseVariance";    CArray = "kDenoiserDirectXVarianceShader";    Source = "denoiser.hlsl"; Header = "directx_denoiser.h" },
    @{ Name = "denoise_atrous_ps";       Profile = "ps_6_0"; Entry = "denoiseAtrous";      CArray = "kDenoiserDirectXAtrousShader";      Source = "denoiser.hlsl"; Header = "directx_denoiser.h" },
    @{ Name = "denoise_atrous_first_ps"; Profile = "ps_6_0"; Entry = "denoiseAtrousFirst"; CArray = "kDenoiserDirectXAtrousFirstShader"; Source = "denoiser.hlsl"; Header = "directx_denoiser.h" },
    @{ Name = "denoise_modulate_ps";     Profile = "ps_6_0"; Entry = "denoiseModulate";    CArray = "kDenoiserDirectXModulateShader";    Source = "denoiser.hlsl"; Header = "directx_denoiser.h" }
)

foreach ($e in $Entries) {
    $outPath = Join-Path $ShaderDir "$($e.Name).dxil"
    $HlslPath = Join-Path $ShaderDir $e.Source
    & $Dxc -T $e.Profile -E $e.Entry $HlslPath -Fo $outPath
    if ($LASTEXITCODE -ne 0) {
        throw "dxc failed compiling entry point '$($e.Entry)' ($($e.Profile))"
    }
    Write-Host "Compiled: $($e.Entry) ($($e.Profile)) -> $($e.Name).dxil"
    $e.OutPath = $outPath
}

# ---------------------------------------------------------------------------
# Generate src/shaders/directx_default.h — plain PowerShell, 12 bytes/line to
# match the formatting build_vulkan_shaders.sh/build_metal_shaders.sh use.
# ---------------------------------------------------------------------------
function ConvertTo-CByteArray {
    param([byte[]]$Bytes)
    $sb = New-Object System.Text.StringBuilder
    for ($i = 0; $i -lt $Bytes.Length; $i += 12) {
        $chunk = $Bytes[$i..([Math]::Min($i + 11, $Bytes.Length - 1))]
        $hex = ($chunk | ForEach-Object { "0x{0:x2}" -f $_ }) -join ", "
        [void]$sb.AppendLine("    $hex,")
    }
    return $sb.ToString()
}

# One header per distinct Header value, its entries in list order.
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
foreach ($header in ($Entries | ForEach-Object { $_.Header } | Select-Object -Unique)) {
    $OutHeader = Join-Path $ScriptDir "src\shaders\$header"
    $sb = New-Object System.Text.StringBuilder
    [void]$sb.AppendLine("#pragma once")
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("// Auto-generated by build_directx_shaders.ps1 -- do not edit.")
    [void]$sb.AppendLine("// Rebuild with: ./build_directx_shaders.ps1")
    [void]$sb.AppendLine("//")
    [void]$sb.AppendLine("// dxc.exe ships with the Windows SDK:")
    [void]$sb.AppendLine("//   C:\Program Files (x86)\Windows Kits\10\bin\<version>\x64\dxc.exe")
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("namespace systems::leal::campello_renderer::shaders {")
    [void]$sb.AppendLine("")

    foreach ($e in ($Entries | Where-Object { $_.Header -eq $header })) {
        $bytes = [System.IO.File]::ReadAllBytes($e.OutPath)
        [void]$sb.AppendLine("static const unsigned char $($e.CArray)[] = {")
        [void]$sb.Append((ConvertTo-CByteArray -Bytes $bytes))
        [void]$sb.AppendLine("};")
        [void]$sb.AppendLine("static const unsigned int $($e.CArray)Size = $($bytes.Length);")
        [void]$sb.AppendLine("")
        Write-Host "  $($e.CArray): $($bytes.Length) bytes"
    }

    [void]$sb.AppendLine("} // namespace systems::leal::campello_renderer::shaders")

    # UTF8 without BOM, matching the other generated shader headers.
    [System.IO.File]::WriteAllText($OutHeader, $sb.ToString(), $utf8NoBom)
    Write-Host "Written: $OutHeader"
}
