#include <campello_renderer/campello_renderer.hpp>

using namespace systems::leal::campello_renderer;

CampelloRenderer::CampelloRenderer() {
}

void CampelloRenderer::setAsset(std::shared_ptr<systems::leal::gltf::GLTF> asset) {
    this->asset = asset;
}

std::shared_ptr<systems::leal::gltf::GLTF> CampelloRenderer::getAsset() {
    return asset;
}

void CampelloRenderer::setCamera(uint32_t index) {
    cameraIndex = index;
}

/*void Campello::setCamera(const Camera &camera) {

}*/

void CampelloRenderer::setScene(uint32_t index) {
    sceneIndex = index;
}

void CampelloRenderer::render() {

}

void CampelloRenderer::update(double dt) {

}
