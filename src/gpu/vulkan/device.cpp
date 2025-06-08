#include <vulkan/vulkan.h>
#include <gpu/device.hpp>
#include <iostream>

using namespace systems::leal::gpu;

Device::Device(void *pd) {
    native = pd;
}

Device::~Device() {
    if (native != nullptr) {
        std::cout << "Device::~Device()" << std::endl;
    }
}

std::shared_ptr<Device> Device::getDefaultDevice() {
    return nullptr;
}

std::vector<std::shared_ptr<Device>> Device::getDevices() {
    std::vector<std::shared_ptr<Device>> toReturn;

    return toReturn;
}


std::shared_ptr<Texture> Device::createTexture(
    StorageMode storageMode, 
    uint32_t width, 
    uint32_t height, 
    PixelFormat pixelFormat,
    //TextureCoordinateSystem textureCoordinateSystem,
    UsageMode usageMode) {

    return nullptr;
}

std::shared_ptr<Buffer> Device::createBuffer(uint64_t size, StorageMode storageMode) {

    return nullptr;
}

std::string Device::getName() {
    return "unknown";
}

std::set<Feature> Device::getFeatures() {
    std::set<Feature> toReturn;

    return toReturn;
}

std::string Device::getEngineVersion() {
    return "unknown";
}
