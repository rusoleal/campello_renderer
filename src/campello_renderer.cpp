#include <campello_renderer/campello_renderer.hpp>
#include "campello_renderer_config.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace systems::leal::campello_renderer;

Renderer::Renderer(std::shared_ptr<systems::leal::campello_gpu::Device> device) {
    this->device = device;
}

void Renderer::setAsset(std::shared_ptr<systems::leal::gltf::GLTF> asset) {
    this->asset = asset;
    if (asset == nullptr) {
        images.clear();
        return;
    }

    images = std::vector<std::shared_ptr<Image>>(asset->images->size(),nullptr);

    gpuBuffers = std::vector<std::shared_ptr<systems::leal::campello_gpu::Buffer>>(asset->buffers->size(), nullptr);
    gpuTextures = std::vector<std::shared_ptr<systems::leal::campello_gpu::Texture>>(asset->images->size(), nullptr);

    if (asset->scenes->size() > 0) {
        if (asset->scene == -1) {
            setScene(0);
        } else {
            setScene(asset->scene);
        }
    }
}

std::shared_ptr<systems::leal::gltf::GLTF> Renderer::getAsset() {
    return asset;
}

void Renderer::setCamera(uint32_t index) {
    cameraIndex = index;
}

/*void Renderer::setCamera(const Camera &camera) {

}*/

void Renderer::setScene(uint32_t index) {

    if (asset == nullptr) {
        return;
    }

    auto info = asset->getRuntimeInfo(index);
    if (info != nullptr) {
        sceneIndex = index;
        
        for (int a=0; a<info->images.size(); a++) {
            if (info->images[a]) {
                if (gpuTextures[a] == nullptr) {
                    auto &image = (*asset->images)[a];
                    if (image.data.size() > 0) {
                        // populated data from data:uri

                    } else if (image.bufferView != -1) {
                        // image embedded in a bufferView as glb files.

                        auto &bufferView = (*asset->bufferViews)[image.bufferView];
                        auto &buffer = (*asset->buffers)[bufferView.buffer];
                        if (buffer.data.size() > 0) {
                            uint8_t *data = &buffer.data[0] + bufferView.byteOffset;
                            int x,y,comp;
                            void *img = stbi_load_from_memory(data, bufferView.byteLength, &x, &y, &comp, 4);
                            if (img != nullptr) {
                                auto texture = device->createTexture(
                                    systems::leal::campello_gpu::StorageMode::devicePrivate,
                                    x,
                                    y,
                                    systems::leal::campello_gpu::PixelFormat::rgba8uint,
                                    systems::leal::campello_gpu::TextureUsage::shaderRead
                                );
                                gpuTextures[a] = texture;
                            }
                        }
                    }
                }
            } else {
                gpuTextures[a] = nullptr;
            }
        }
    }

}

void Renderer::render() {

}

void Renderer::update(double dt) {

}

std::string systems::leal::campello_renderer::getVersion() {
    return std::to_string(campello_renderer_VERSION_MAJOR) + "." + std::to_string(campello_renderer_VERSION_MINOR) + "." + std::to_string(campello_renderer_VERSION_PATCH);
}
