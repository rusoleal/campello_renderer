#include <vulkan/vulkan.h>
#include <gpu/device.hpp>
#include <android/log.h>

using namespace systems::leal::gpu;

VkInstance *instance = nullptr;

VkInstance *getInstance() {

    VkApplicationInfo appInfo;
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "test";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "campello";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;
    appInfo.pNext = nullptr;

    VkInstanceCreateInfo instanceInfo;
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pNext = nullptr;
    instanceInfo.flags = 0;
    instanceInfo.enabledExtensionCount = 0;
    instanceInfo.ppEnabledExtensionNames = nullptr;
    instanceInfo.enabledLayerCount = 0;
    instanceInfo.ppEnabledLayerNames = nullptr;
    instanceInfo.pApplicationInfo = &appInfo;

    VkInstance ins;
    auto res = vkCreateInstance(&instanceInfo, nullptr, &ins);
    if (res != VK_SUCCESS) {
        __android_log_print(ANDROID_LOG_DEBUG,"Campello","vkCreateInstance failed with error: %d", res);
        return nullptr;
    }
    return new VkInstance(ins);
}

Device::Device(void *pd) {
    native = pd;
    __android_log_print(ANDROID_LOG_DEBUG,"Campello","Device::Device()");
}

Device::~Device() {
    if (native != nullptr) {
        vkDestroyDevice((VkDevice)native, nullptr);
        native = nullptr;
        __android_log_print(ANDROID_LOG_DEBUG,"Campello","Device::~Device()");
    }
}

std::shared_ptr<Device> Device::getDefaultDevice() {
    auto devices = getDevices();
    if (devices.size() > 0) {
        return devices[0];
    }
    return nullptr;
}

std::vector<std::shared_ptr<Device>> Device::getDevices() {

    if (instance == nullptr) {
        instance = getInstance();
    }

    std::vector<std::shared_ptr<Device>> toReturn;

    uint32_t gpuCount = 0;

    // First call: Get the number of GPUs
    vkEnumeratePhysicalDevices(*instance, &gpuCount, nullptr);

    __android_log_print(ANDROID_LOG_DEBUG,"Campello","gpuCount=%d", gpuCount);

    // Allocate memory for the GPU handles
    std::vector<VkPhysicalDevice> gpus(gpuCount);

    // Second call: Get the GPU handles
    vkEnumeratePhysicalDevices(*instance, &gpuCount, gpus.data());

    // Now, 'gpus' contains a list of VkPhysicalDevice handles
    for (const auto& gpu : gpus) {
        Device *device = new Device(gpu);
        toReturn.push_back(std::shared_ptr<Device>(device));
    }

    return toReturn;
}


std::shared_ptr<Texture> Device::createTexture(
    StorageMode storageMode, 
    uint32_t width, 
    uint32_t height, 
    PixelFormat pixelFormat,
    //TextureCoordinateSystem textureCoordinateSystem,
    TextureUsage usageMode) {

    return nullptr;
}

std::shared_ptr<Buffer> Device::createBuffer(uint64_t size, StorageMode storageMode) {

    return nullptr;
}

std::string Device::getName() {
    return "unknown";
}

std::set<Feature> Device::getFeatures() {

    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties((VkPhysicalDevice)native, &deviceProperties);

    VkPhysicalDeviceFeatures deviceFeatures;
    vkGetPhysicalDeviceFeatures((VkPhysicalDevice)native, &deviceFeatures);

    std::set<Feature> toReturn;

    if (deviceFeatures.textureCompressionBC) {
        toReturn.insert(Feature::bcTextureCompression);
    }

    if (deviceFeatures.geometryShader) {
        toReturn.insert(Feature::geometryShader);
    }

    return toReturn;
}

std::string Device::getEngineVersion() {
    return "unknown";
}
