#pragma once

#include <memory>
#include <gpu/texture.hpp>

namespace systems::leal::gpu {

    class Context {
    public:
        std::shared_ptr<Texture> createTexture();
    };
}