#pragma once

#include <memory>
#include <gltf/gltf.hpp>
//#include <gpu/device.hpp>

namespace systems::leal::renderer {

    class Campello {
    private:
        //std::shared_ptr<Device> device;
        std::shared_ptr<systems::leal::gltf::GLTF> asset;
        uint32_t sceneIndex;
        uint32_t cameraIndex;
        //Camera camera;

    public:
        Campello();

        void setAsset(std::shared_ptr<systems::leal::gltf::GLTF> asset);
        std::shared_ptr<systems::leal::gltf::GLTF> getAsset();

        void setCamera(uint32_t index);
        //void setCamera(const Camera &camera);

        void setScene(uint32_t index);

        void render();
        void update(double dt);

    };
    
}