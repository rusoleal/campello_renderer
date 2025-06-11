#include "Metal.hpp"
#include <gpu/texture.hpp>

using namespace systems::leal::gpu;

Texture::Texture(void *pd) {
    native = pd;
}

Texture::~Texture() {
    if (native != nullptr) {
        ((MTL::Texture *)native)->release();
    }
}

PixelFormat Texture::getPixelFormat() {
    return (PixelFormat)((MTL::Texture *)native)->pixelFormat();
}

uint64_t Texture::getWidth() {
    return ((MTL::Texture *)native)->width();
}

uint64_t Texture::getHeight() {
    return ((MTL::Texture *)native)->height();
}

TextureUsage Texture::getUsageMode() {
    return TextureUsage::renderTarget;
}
