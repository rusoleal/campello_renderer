#pragma once

#include <gpu/pixel_format.hpp>
#include <gpu/texture_usage.hpp>

namespace systems::leal::gpu
{

    class Device;

    class Texture
    {
    private:
        friend class Device;
        void *native;

        Texture(void *pd);

    public:
        ~Texture();

        PixelFormat getPixelFormat();
        uint64_t getWidth();
        uint64_t getHeight();
        TextureUsage getUsageMode();
    };
}