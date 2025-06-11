#pragma once

namespace systems::leal::gpu
{

    enum class Feature
    {
        raytracing,
        msaa32bit,
        bcTextureCompression,
        depth24Stencil8PixelFormat,
        geometryShader
    };
}