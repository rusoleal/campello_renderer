#pragma once

#include <memory>
#include <gltf/gltf.hpp>
#include <campello_gpu/device.hpp>
#include <campello_gpu/texture.hpp>
#include <campello_gpu/buffer.hpp>
#include <campello_renderer/image.hpp>
//#include <gpu/device.hpp>

namespace systems::leal::campello_renderer {

    class Renderer {
    private:
        std::shared_ptr<systems::leal::campello_gpu::Device> device;
        std::shared_ptr<systems::leal::gltf::GLTF> asset;
        uint32_t sceneIndex;
        uint32_t cameraIndex;
        std::vector<std::shared_ptr<Image>> images;
        //Camera camera;

        std::vector<std::shared_ptr<systems::leal::campello_gpu::Buffer>> gpuBuffers;
        std::vector<std::shared_ptr<systems::leal::campello_gpu::Texture>> gpuTextures;

    public:
        Renderer(std::shared_ptr<systems::leal::campello_gpu::Device> device);

        void setAsset(std::shared_ptr<systems::leal::gltf::GLTF> asset);
        std::shared_ptr<systems::leal::gltf::GLTF> getAsset();

        void setCamera(uint32_t index);
        //void setCamera(const Camera &camera);

        void setScene(uint32_t index);

        void render();
        void update(double dt);

    };

    std::string getVersion();
    
}