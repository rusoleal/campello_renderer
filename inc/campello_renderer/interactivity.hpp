#pragma once

// KHR_interactivity support: the live glTF Asset Object Model a behavior
// graph reads and writes (gltf's InteractivityRuntime executes the graph;
// this is its host, on the renderer's asset).

#include <gltf/extensions/khr_interactivity_runtime.hpp>

#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace systems::leal::campello_renderer {

class Renderer;

/**
 * pointer/get and pointer/set on the renderer's asset (Renderer::getAsset()),
 * which is changed in place: the next rendered frame shows the change.
 *
 * Supported pointers (types as the glTF Asset Object Model):
 * - /nodes/{}/translation (float3), rotation (float4), scale (float3): nodes
 *   without an authored matrix; matrix and globalMatrix (float4x4, read-only;
 *   globalMatrix from the nodes' current transforms)
 * - /nodes/{}/weights/{} (float): one morph target weight
 * - /nodes/{}/extensions/KHR_node_visibility/visible,
 *   KHR_node_selectability/selectable, KHR_node_hoverability/hoverable (bool)
 * - /materials/{}/alphaCutoff, emissiveFactor, pbrMetallicRoughness/
 *   baseColorFactor, metallicFactor, roughnessFactor, normalTexture/scale,
 *   occlusionTexture/strength, extensions/KHR_materials_emissive_strength/
 *   emissiveStrength, KHR_materials_ior/ior, KHR_materials_dispersion/
 *   dispersion, and the factors of KHR_materials_transmission, clearcoat,
 *   sheen, specular, volume, iridescence, anisotropy and diffuse_transmission
 *   (of a material that uses the extension)
 * - KHR_texture_transform offset (float2), rotation (float), scale (float2)
 *   of a material's baseColorTexture, metallicRoughnessTexture,
 *   normalTexture, occlusionTexture and emissiveTexture, and of its
 *   KHR_materials_* extensions' textures
 * - /extensions/KHR_lights_punctual/lights/{}/color, intensity, range, and
 *   spot/innerConeAngle, spot/outerConeAngle of spot lights
 * - /extensions/KHR_interactivity/activeCamera/position (float3), rotation
 *   (float4), read-only: the last rendered frame's camera
 *
 * animation/start plays glTF animations with Renderer::poseAnimation() at the
 * time the runtime decides (the application must not play them itself).
 *
 * Material changes are re-uploaded by flush(): call it after the runtime's
 * start(), tick(), select() or receive() (once per frame is enough).
 */
class InteractivityObjectModel : public systems::leal::gltf::InteractivityHost {
public:
    explicit InteractivityObjectModel(Renderer& renderer);

    bool getPointer(const std::string& pointer, systems::leal::gltf::InteractivityTypeSignature type,
                    systems::leal::gltf::InteractivityRuntimeValue& value) override;
    bool setPointer(const std::string& pointer, const systems::leal::gltf::InteractivityRuntimeValue& value) override;
    void log(const std::string& message) override {
        if (onLog) onLog(message);
    }
    void eventSent(int64_t eventIndex,
                   const std::map<std::string, systems::leal::gltf::InteractivityRuntimeValue>& values) override {
        if (onEventSent) onEventSent(eventIndex, values);
    }

    /// animation/start: the animation at `time`, through Renderer::poseAnimation().
    void applyAnimation(int64_t animation, double time) override;
    /// The graph stopped it: Renderer::releaseAnimation().
    void animationStopped(int64_t animation) override;

    /// Re-uploads the materials pointer/set changed since the last flush().
    void flush();

    /// debug/log messages.
    std::function<void(const std::string&)> onLog;
    /// event/send: custom events the graph sends to the application.
    std::function<void(int64_t, const std::map<std::string, systems::leal::gltf::InteractivityRuntimeValue>&)> onEventSent;

private:
    /// Reads (set == nullptr) or writes a property; false when unresolved or mistyped.
    bool access(const std::string& pointer, systems::leal::gltf::InteractivityTypeSignature type,
                systems::leal::gltf::InteractivityRuntimeValue* get,
                const systems::leal::gltf::InteractivityRuntimeValue* set);

    /// Each node's parent (-1 for roots), for globalMatrix.
    const std::vector<int64_t>& parentsOf(const systems::leal::gltf::GLTF& asset);

    Renderer& renderer_;
    std::set<uint64_t> dirtyMaterials_;
    const systems::leal::gltf::GLTF* parentsAsset_ = nullptr;
    std::vector<int64_t> parents_;
};

}  // namespace systems::leal::campello_renderer
