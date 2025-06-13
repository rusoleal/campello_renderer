#include <campello/campello.hpp>

using namespace systems::leal::renderer;

Campello::Campello() {
}

void Campello::setAsset(std::shared_ptr<systems::leal::gltf::GLTF> asset) {
    this->asset = asset;
}

std::shared_ptr<systems::leal::gltf::GLTF> Campello::getAsset() {
    return asset;
}

void Campello::setCamera(uint32_t index) {
    cameraIndex = index;
}

/*void Campello::setCamera(const Camera &camera) {

}*/

void Campello::setScene(uint32_t index) {
    sceneIndex = index;
}

void Campello::render() {

}

void Campello::update(double dt) {

}
