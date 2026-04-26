#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <gltf/gltf.hpp>
#include <vector_math/vector_math.hpp>

namespace systems::leal::campello_renderer {

class GltfAnimator {
public:
    struct AnimationState {
        double time = 0.0;
        bool playing = false;
        bool loop = true;
        double duration = 0.0;
    };

    struct AnimatedTRS {
        bool hasTranslation = false;
        bool hasRotation = false;
        bool hasScale = false;
        systems::leal::vector_math::Vector3<double> translation;
        systems::leal::vector_math::Quaternion<double> rotation;
        systems::leal::vector_math::Vector3<double> scale;
    };

    GltfAnimator();
    explicit GltfAnimator(std::shared_ptr<systems::leal::gltf::GLTF> asset);

    void setAsset(std::shared_ptr<systems::leal::gltf::GLTF> asset);

    // Advance all playing animations and sample them.
    void update(double dt);

    // Sample a single animation at a specific time (seconds).
    void sampleAnimation(int32_t animIndex, float time);

    // Clear all animated node values.
    void clearAnimatedNodes();

    // Returns the map of animated node transforms.
    const std::unordered_map<uint64_t, AnimatedTRS>& getAnimatedNodes() const;

    // Returns the map of animated pointer values (KHR_animation_pointer).
    const std::unordered_map<std::string, std::vector<float>>& getAnimatedPointers() const;

    // Resolve all animated pointer values into the glTF asset.
    // Returns the set of material indices that were modified.
    std::unordered_set<uint64_t> applyAnimatedPointers();

    // Playback controls.
    void playAnimation(uint32_t animationIndex);
    void pauseAnimation(uint32_t animationIndex);
    void stopAnimation(uint32_t animationIndex);
    void stopAllAnimations();

    bool isAnimationPlaying(uint32_t animationIndex) const;
    void setAnimationLoop(uint32_t animationIndex, bool loop);
    bool isAnimationLooping(uint32_t animationIndex) const;

    void setAnimationTime(uint32_t animationIndex, double time);
    double getAnimationTime(uint32_t animationIndex) const;

    // Queries.
    uint32_t getAnimationCount() const;
    std::string getAnimationName(uint32_t animationIndex) const;
    double getAnimationDuration(uint32_t animationIndex) const;

private:
    std::shared_ptr<systems::leal::gltf::GLTF> asset;
    std::unordered_map<int32_t, AnimationState> animationStates;
    std::unordered_map<uint64_t, AnimatedTRS> animatedNodes;
    std::unordered_map<std::string, std::vector<float>> animatedPointers;

    void initDuration(int32_t animIndex);
};

} // namespace systems::leal::campello_renderer
