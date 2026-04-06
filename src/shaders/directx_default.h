#pragma once

// PLACEHOLDER — arrays are empty until the shader is compiled on Windows.
//
// To populate this file, compile shaders/directx/default.hlsl on Windows with DXC:
//
//   dxc -T vs_6_0 -E vertexMain default.hlsl -Fo default_vs.dxil
//   dxc -T ps_6_0 -E pixelMain  default.hlsl -Fo default_ps.dxil
//
// Then generate this header (PowerShell):
//   python3 gen_directx_header.py default_vs.dxil default_ps.dxil > directx_default.h
//
// Or using the CMake custom target added for WIN32 builds (see CMakeLists.txt TODO).

namespace systems::leal::campello_renderer::shaders {

static const unsigned char kDefaultDirectXVertShader[1] = {0};
static const unsigned int  kDefaultDirectXVertShaderSize = 0;

static const unsigned char kDefaultDirectXPixelShader[1] = {0};
static const unsigned int  kDefaultDirectXPixelShaderSize = 0;

} // namespace systems::leal::campello_renderer::shaders
