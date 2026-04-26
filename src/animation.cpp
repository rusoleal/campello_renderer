#include <campello_renderer/animation.hpp>
#include <cmath>

namespace systems::leal::campello_renderer {

using namespace systems::leal;

// Forward declarations for KHR_animation_pointer helpers.
static std::vector<std::string> splitJsonPointer(const std::string& pointer);
static bool sampleChannelRaw(const systems::leal::gltf::GLTF& asset,
                             const systems::leal::gltf::AnimationSampler& sampler,
                             float time,
                             std::vector<float>& outValue);

GltfAnimator::GltfAnimator() = default;

GltfAnimator::GltfAnimator(std::shared_ptr<gltf::GLTF> asset)
    : asset(std::move(asset)) {}

void GltfAnimator::setAsset(std::shared_ptr<gltf::GLTF> newAsset) {
    asset = std::move(newAsset);
    animationStates.clear();
    animatedNodes.clear();
}

void GltfAnimator::initDuration(int32_t animIndex) {
    if (!asset || !asset->animations) return;
    if (animIndex < 0 || (size_t)animIndex >= asset->animations->size()) return;
    auto &state = animationStates[animIndex];
    if (state.duration > 0.0) return;
    auto &anim = (*asset->animations)[(size_t)animIndex];
    // Determine duration from the max keyframe time across all channels.
    double maxTime = 0.0;
    if (asset->accessors) {
        for (auto &channel : anim.channels) {
            if (channel.sampler < 0 || (size_t)channel.sampler >= anim.samplers.size()) continue;
            auto &sampler = anim.samplers[channel.sampler];
            if (sampler.input < 0 || (size_t)sampler.input >= asset->accessors->size()) continue;
            auto &inputAcc = (*asset->accessors)[(size_t)sampler.input];
            if (inputAcc.bufferView < 0 || !asset->bufferViews ||
                (size_t)inputAcc.bufferView >= asset->bufferViews->size()) continue;
            auto &inputBV = (*asset->bufferViews)[(size_t)inputAcc.bufferView];
            if (!asset->buffers || (size_t)inputBV.buffer >= asset->buffers->size()) continue;
            auto &buf = (*asset->buffers)[(size_t)inputBV.buffer];
            const float *times = reinterpret_cast<const float*>(
                buf.data.data() + inputBV.byteOffset + inputAcc.byteOffset);
            for (size_t i = 0; i < inputAcc.count; ++i) {
                if (times[i] > maxTime) maxTime = times[i];
            }
        }
    }
    state.duration = maxTime;
}

void GltfAnimator::update(double dt) {
    if (!asset || !asset->animations) return;
    if (animationStates.empty()) return;

    animatedNodes.clear();
    animatedPointers.clear();

    for (auto &pair : animationStates) {
        int32_t animIndex = pair.first;
        AnimationState &state = pair.second;

        if (animIndex < 0 || (size_t)animIndex >= asset->animations->size()) continue;
        if (!state.playing) continue;

        state.time += dt;

        if (state.time > state.duration) {
            if (state.loop) {
                state.time = fmod(state.time, state.duration);
            } else {
                state.time = state.duration;
                state.playing = false;
            }
        }

        sampleAnimation(animIndex, (float)state.time);
    }
}

void GltfAnimator::sampleAnimation(int32_t animIndex, float time) {
    if (!asset || !asset->animations) return;
    if (animIndex < 0 || (size_t)animIndex >= asset->animations->size()) return;

    auto &animation = (*asset->animations)[(size_t)animIndex];

    for (auto &channel : animation.channels) {
        // KHR_animation_pointer
        if (channel.target.path == "pointer") {
            if (!channel.target.khrAnimationPointer) continue;
            if (channel.sampler < 0 || (size_t)channel.sampler >= animation.samplers.size()) continue;
            auto& sampler = animation.samplers[channel.sampler];
            std::vector<float> value;
            if (!sampleChannelRaw(*asset, sampler, time, value)) continue;

            auto segs = splitJsonPointer(channel.target.khrAnimationPointer->pointer);
            if (!segs.empty() && segs[0] == "nodes" && segs.size() >= 3) {
                uint64_t nodeIdx = std::stoull(segs[1]);
                const std::string& prop = segs[2];
                namespace VM = systems::leal::vector_math;
                auto& trs = animatedNodes[nodeIdx];
                if (prop == "translation" && value.size() >= 3) {
                    trs.hasTranslation = true;
                    trs.translation = VM::Vector3<double>(value[0], value[1], value[2]);
                } else if (prop == "rotation" && value.size() >= 4) {
                    trs.hasRotation = true;
                    // sampleChannelRaw linearly interpolates; for rotation we normalize
                    // the result quaternion. True slerp would require re-sampling raw keyframes.
                    auto n = VM::Quaternion<double>(value[0], value[1], value[2], value[3]).normalized();
                    trs.rotation = VM::Quaternion<double>(n.data[0], n.data[1], n.data[2], n.data[3]);
                } else if (prop == "scale" && value.size() >= 3) {
                    trs.hasScale = true;
                    trs.scale = VM::Vector3<double>(value[0], value[1], value[2]);
                }
            } else {
                animatedPointers[channel.target.khrAnimationPointer->pointer] = std::move(value);
            }
            continue;
        }

        if (channel.target.node < 0) continue;
        if (!asset->nodes || (size_t)channel.target.node >= asset->nodes->size()) continue;
        if (channel.sampler < 0 || (size_t)channel.sampler >= animation.samplers.size()) continue;

        auto &sampler = animation.samplers[channel.sampler];
        if (!asset->accessors) continue;
        if (sampler.input < 0 || (size_t)sampler.input >= asset->accessors->size()) continue;
        if (sampler.output < 0 || (size_t)sampler.output >= asset->accessors->size()) continue;

        auto &inputAcc = (*asset->accessors)[(size_t)sampler.input];
        auto &outputAcc = (*asset->accessors)[(size_t)sampler.output];
        if (inputAcc.bufferView < 0 || outputAcc.bufferView < 0) continue;
        if (!asset->bufferViews) continue;
        if ((size_t)inputAcc.bufferView >= asset->bufferViews->size()) continue;
        if ((size_t)outputAcc.bufferView >= asset->bufferViews->size()) continue;

        auto &inputBV = (*asset->bufferViews)[(size_t)inputAcc.bufferView];
        auto &outputBV = (*asset->bufferViews)[(size_t)outputAcc.bufferView];
        if (!asset->buffers) continue;
        if ((size_t)inputBV.buffer >= asset->buffers->size()) continue;
        if ((size_t)outputBV.buffer >= asset->buffers->size()) continue;

        auto &inputBuf = (*asset->buffers)[(size_t)inputBV.buffer];
        auto &outputBuf = (*asset->buffers)[(size_t)outputBV.buffer];

        const float *times = reinterpret_cast<const float*>(
            inputBuf.data.data() + inputBV.byteOffset + inputAcc.byteOffset);
        uint32_t keyframeCount = (uint32_t)inputAcc.count;
        if (keyframeCount == 0) continue;

        const uint8_t *values = outputBuf.data.data() + outputBV.byteOffset + outputAcc.byteOffset;

        uint32_t kf0 = 0, kf1 = 0;
        float t = 0.0f;

        if (time <= times[0]) {
            kf0 = kf1 = 0;
            t = 0.0f;
        } else if (time >= times[keyframeCount - 1]) {
            kf0 = kf1 = keyframeCount - 1;
            t = 0.0f;
        } else {
            for (uint32_t i = 0; i < keyframeCount - 1; ++i) {
                if (time >= times[i] && time < times[i + 1]) {
                    kf0 = i;
                    kf1 = i + 1;
                    float span = times[kf1] - times[kf0];
                    t = (span > 0.0f) ? (time - times[kf0]) / span : 0.0f;
                    break;
                }
            }
        }

        namespace VM = systems::leal::vector_math;
        uint64_t nodeIdx = (uint64_t)channel.target.node;

        if (channel.target.path == "translation") {
            auto &trs = animatedNodes[nodeIdx];
            trs.hasTranslation = true;
            const float *v0 = reinterpret_cast<const float*>(values) + kf0 * 3;
            const float *v1 = reinterpret_cast<const float*>(values) + kf1 * 3;
            if (sampler.interpolation == gltf::AnimationInterpolation::aiStep) {
                trs.translation = VM::Vector3<double>(v0[0], v0[1], v0[2]);
            } else {
                trs.translation = VM::Vector3<double>(
                    v0[0] + (v1[0] - v0[0]) * t,
                    v0[1] + (v1[1] - v0[1]) * t,
                    v0[2] + (v1[2] - v0[2]) * t);
            }
        } else if (channel.target.path == "rotation") {
            auto &trs = animatedNodes[nodeIdx];
            trs.hasRotation = true;
            const float *v0 = reinterpret_cast<const float*>(values) + kf0 * 4;
            const float *v1 = reinterpret_cast<const float*>(values) + kf1 * 4;
            VM::Quaternion<double> q0(v0[0], v0[1], v0[2], v0[3]);
            VM::Quaternion<double> q1(v1[0], v1[1], v1[2], v1[3]);
            if (sampler.interpolation == gltf::AnimationInterpolation::aiStep) {
                trs.rotation = q0;
            } else {
                trs.rotation = VM::Quaternion<double>::slerp(q0, q1, (double)t);
            }
        } else if (channel.target.path == "scale") {
            auto &trs = animatedNodes[nodeIdx];
            trs.hasScale = true;
            const float *v0 = reinterpret_cast<const float*>(values) + kf0 * 3;
            const float *v1 = reinterpret_cast<const float*>(values) + kf1 * 3;
            if (sampler.interpolation == gltf::AnimationInterpolation::aiStep) {
                trs.scale = VM::Vector3<double>(v0[0], v0[1], v0[2]);
            } else {
                trs.scale = VM::Vector3<double>(
                    v0[0] + (v1[0] - v0[0]) * t,
                    v0[1] + (v1[1] - v0[1]) * t,
                    v0[2] + (v1[2] - v0[2]) * t);
            }
        }
    }
}

void GltfAnimator::clearAnimatedNodes() {
    animatedNodes.clear();
}

const std::unordered_map<uint64_t, GltfAnimator::AnimatedTRS>& GltfAnimator::getAnimatedNodes() const {
    return animatedNodes;
}

void GltfAnimator::playAnimation(uint32_t animationIndex) {
    if (!asset || !asset->animations || animationIndex >= asset->animations->size()) return;
    auto &state = animationStates[(int32_t)animationIndex];
    state.playing = true;
    initDuration((int32_t)animationIndex);
}

void GltfAnimator::pauseAnimation(uint32_t animationIndex) {
    auto it = animationStates.find((int32_t)animationIndex);
    if (it != animationStates.end()) {
        it->second.playing = false;
    }
}

void GltfAnimator::stopAnimation(uint32_t animationIndex) {
    auto it = animationStates.find((int32_t)animationIndex);
    if (it != animationStates.end()) {
        it->second.playing = false;
        it->second.time = 0.0;
    }
    bool anyPlaying = false;
    for (auto &pair : animationStates) {
        if (pair.second.playing) {
            anyPlaying = true;
            break;
        }
    }
    if (!anyPlaying) {
        animatedNodes.clear();
        animatedPointers.clear();
    }
}

void GltfAnimator::stopAllAnimations() {
    for (auto &pair : animationStates) {
        pair.second.playing = false;
        pair.second.time = 0.0;
    }
    animatedNodes.clear();
    animatedPointers.clear();
}

bool GltfAnimator::isAnimationPlaying(uint32_t animationIndex) const {
    auto it = animationStates.find((int32_t)animationIndex);
    if (it != animationStates.end()) {
        return it->second.playing;
    }
    return false;
}

void GltfAnimator::setAnimationLoop(uint32_t animationIndex, bool loop) {
    if (!asset || !asset->animations || animationIndex >= asset->animations->size()) return;
    animationStates[(int32_t)animationIndex].loop = loop;
}

bool GltfAnimator::isAnimationLooping(uint32_t animationIndex) const {
    auto it = animationStates.find((int32_t)animationIndex);
    if (it != animationStates.end()) {
        return it->second.loop;
    }
    return true;
}

void GltfAnimator::setAnimationTime(uint32_t animationIndex, double time) {
    if (!asset || !asset->animations || animationIndex >= asset->animations->size()) return;
    auto &state = animationStates[(int32_t)animationIndex];
    state.time = std::max(0.0, std::min(time, state.duration));
}

double GltfAnimator::getAnimationTime(uint32_t animationIndex) const {
    auto it = animationStates.find((int32_t)animationIndex);
    if (it != animationStates.end()) {
        return it->second.time;
    }
    return 0.0;
}

uint32_t GltfAnimator::getAnimationCount() const {
    if (!asset || !asset->animations) return 0;
    return (uint32_t)asset->animations->size();
}

std::string GltfAnimator::getAnimationName(uint32_t animationIndex) const {
    if (!asset || !asset->animations || animationIndex >= asset->animations->size()) return "";
    return (*asset->animations)[animationIndex].name;
}

double GltfAnimator::getAnimationDuration(uint32_t animationIndex) const {
    if (!asset || !asset->animations || animationIndex >= asset->animations->size()) return 0.0;
    auto it = animationStates.find((int32_t)animationIndex);
    if (it != animationStates.end()) {
        return it->second.duration;
    }
    return 0.0;
}

const std::unordered_map<std::string, std::vector<float>>& GltfAnimator::getAnimatedPointers() const {
    return animatedPointers;
}

// ---------------------------------------------------------------------------
// KHR_animation_pointer helpers
// ---------------------------------------------------------------------------

static std::vector<std::string> splitJsonPointer(const std::string& pointer) {
    std::vector<std::string> segments;
    size_t i = 0;
    if (i < pointer.size() && pointer[i] == '/') i++;
    while (i < pointer.size()) {
        size_t j = pointer.find('/', i);
        if (j == std::string::npos) j = pointer.size();
        std::string seg = pointer.substr(i, j - i);
        std::string decoded;
        decoded.reserve(seg.size());
        for (size_t k = 0; k < seg.size(); ++k) {
            if (seg[k] == '~' && k + 1 < seg.size()) {
                if (seg[k+1] == '1') { decoded += '/'; k++; }
                else if (seg[k+1] == '0') { decoded += '~'; k++; }
                else decoded += seg[k];
            } else {
                decoded += seg[k];
            }
        }
        segments.push_back(decoded);
        i = j + 1;
    }
    return segments;
}

static bool sampleChannelRaw(const systems::leal::gltf::GLTF& asset,
                             const systems::leal::gltf::AnimationSampler& sampler,
                             float time,
                             std::vector<float>& outValue) {
    if (!asset.accessors) return false;
    if (sampler.input < 0 || (size_t)sampler.input >= asset.accessors->size()) return false;
    if (sampler.output < 0 || (size_t)sampler.output >= asset.accessors->size()) return false;

    auto& inputAcc = (*asset.accessors)[(size_t)sampler.input];
    auto& outputAcc = (*asset.accessors)[(size_t)sampler.output];
    if (inputAcc.bufferView < 0 || outputAcc.bufferView < 0) return false;
    if (!asset.bufferViews) return false;
    if ((size_t)inputAcc.bufferView >= asset.bufferViews->size()) return false;
    if ((size_t)outputAcc.bufferView >= asset.bufferViews->size()) return false;

    auto& inputBV = (*asset.bufferViews)[(size_t)inputAcc.bufferView];
    auto& outputBV = (*asset.bufferViews)[(size_t)outputAcc.bufferView];
    if (!asset.buffers) return false;
    if ((size_t)inputBV.buffer >= asset.buffers->size()) return false;
    if ((size_t)outputBV.buffer >= asset.buffers->size()) return false;

    auto& inputBuf = (*asset.buffers)[(size_t)inputBV.buffer];
    auto& outputBuf = (*asset.buffers)[(size_t)outputBV.buffer];

    const float* times = reinterpret_cast<const float*>(
        inputBuf.data.data() + inputBV.byteOffset + inputAcc.byteOffset);
    uint32_t keyframeCount = (uint32_t)inputAcc.count;
    if (keyframeCount == 0) return false;

    const uint8_t* values = outputBuf.data.data() + outputBV.byteOffset + outputAcc.byteOffset;

    uint32_t kf0 = 0, kf1 = 0;
    float t = 0.0f;
    if (time <= times[0]) {
        kf0 = kf1 = 0; t = 0.0f;
    } else if (time >= times[keyframeCount - 1]) {
        kf0 = kf1 = keyframeCount - 1; t = 0.0f;
    } else {
        for (uint32_t i = 0; i < keyframeCount - 1; ++i) {
            if (time >= times[i] && time < times[i + 1]) {
                kf0 = i; kf1 = i + 1;
                float span = times[kf1] - times[kf0];
                t = (span > 0.0f) ? (time - times[kf0]) / span : 0.0f;
                break;
            }
        }
    }

    uint32_t components = 1;
    if (outputAcc.type == systems::leal::gltf::AccessorType::acVec2) components = 2;
    else if (outputAcc.type == systems::leal::gltf::AccessorType::acVec3) components = 3;
    else if (outputAcc.type == systems::leal::gltf::AccessorType::acVec4) components = 4;

    outValue.resize(components);
    const float* v0 = reinterpret_cast<const float*>(values) + kf0 * components;

    if (sampler.interpolation == systems::leal::gltf::AnimationInterpolation::aiStep || kf0 == kf1) {
        for (uint32_t i = 0; i < components; ++i) outValue[i] = v0[i];
    } else {
        const float* v1 = reinterpret_cast<const float*>(values) + kf1 * components;
        for (uint32_t i = 0; i < components; ++i) {
            outValue[i] = v0[i] + (v1[i] - v0[i]) * t;
        }
    }
    return true;
}

std::unordered_set<uint64_t> GltfAnimator::applyAnimatedPointers() {
    std::unordered_set<uint64_t> modifiedMaterials;
    if (!asset) return modifiedMaterials;

    namespace VM = systems::leal::vector_math;

    for (auto& pair : animatedPointers) {
        const std::string& ptr = pair.first;
        const std::vector<float>& value = pair.second;
        if (value.empty()) continue;

        auto segs = splitJsonPointer(ptr);
        if (segs.empty()) continue;

        if (segs[0] == "materials" && segs.size() >= 3 && asset->materials) {
            int idx = std::stoi(segs[1]);
            if (idx < 0 || (size_t)idx >= asset->materials->size()) continue;
            auto& mat = (*asset->materials)[idx];
            modifiedMaterials.insert((uint64_t)idx);
            const std::string& prop = segs[2];

            if (prop == "pbrMetallicRoughness" && segs.size() >= 4 && mat.pbrMetallicRoughness) {
                const std::string& sub = segs[3];
                if (sub == "baseColorFactor" && value.size() >= 4) {
                    mat.pbrMetallicRoughness->baseColorFactor = VM::Vector4<double>(value[0], value[1], value[2], value[3]);
                } else if (sub == "metallicFactor" && value.size() >= 1) {
                    mat.pbrMetallicRoughness->metallicFactor = value[0];
                } else if (sub == "roughnessFactor" && value.size() >= 1) {
                    mat.pbrMetallicRoughness->roughnessFactor = value[0];
                }
            } else if (prop == "emissiveFactor" && value.size() >= 3) {
                mat.emissiveFactor = VM::Vector3<double>(value[0], value[1], value[2]);
            } else if (prop == "alphaCutoff" && value.size() >= 1) {
                mat.alphaCutoff = value[0];
            } else if (prop == "normalTexture" && segs.size() >= 4 && segs[3] == "scale" && value.size() >= 1) {
                if (mat.normalTexture) mat.normalTexture->scale = value[0];
            } else if (prop == "occlusionTexture" && segs.size() >= 4 && segs[3] == "strength" && value.size() >= 1) {
                if (mat.occlusionTexture) mat.occlusionTexture->strength = value[0];
            } else if (prop == "extensions" && segs.size() >= 4) {
                const std::string& ext = segs[3];
                if (ext == "KHR_materials_ior" && segs.size() >= 5 && segs[4] == "ior" && value.size() >= 1) {
                    mat.khrMaterialsIor = value[0];
                } else if (ext == "KHR_materials_transmission" && segs.size() >= 5 && segs[4] == "transmissionFactor" && value.size() >= 1) {
                    if (mat.khrMaterialsTransmission) mat.khrMaterialsTransmission->transmissionFactor = value[0];
                } else if (ext == "KHR_materials_volume" && segs.size() >= 5 && segs[4] == "thicknessFactor" && value.size() >= 1) {
                    if (mat.khrMaterialsVolume) mat.khrMaterialsVolume->thicknessFactor = value[0];
                } else if (ext == "KHR_materials_clearcoat" && segs.size() >= 5) {
                    if (!mat.khrMaterialsClearcoat) continue;
                    if (segs[4] == "clearcoatFactor" && value.size() >= 1) mat.khrMaterialsClearcoat->clearcoatFactor = value[0];
                    else if (segs[4] == "clearcoatRoughnessFactor" && value.size() >= 1) mat.khrMaterialsClearcoat->clearcoatRoughnessFactor = value[0];
                } else if (ext == "KHR_materials_sheen" && segs.size() >= 5) {
                    if (!mat.khrMaterialsSheen) continue;
                    if (segs[4] == "sheenColorFactor" && value.size() >= 3) {
                        mat.khrMaterialsSheen->sheenColorFactor = VM::Vector3<double>(value[0], value[1], value[2]);
                    } else if (segs[4] == "sheenRoughnessFactor" && value.size() >= 1) {
                        mat.khrMaterialsSheen->sheenRoughnessFactor = value[0];
                    }
                } else if (ext == "KHR_materials_anisotropy" && segs.size() >= 5) {
                    if (!mat.khrMaterialsAnisotropy) continue;
                    if (segs[4] == "anisotropyStrength" && value.size() >= 1) mat.khrMaterialsAnisotropy->anisotropyStrength = value[0];
                    else if (segs[4] == "anisotropyRotation" && value.size() >= 1) mat.khrMaterialsAnisotropy->anisotropyRotation = value[0];
                } else if (ext == "KHR_materials_iridescence" && segs.size() >= 5 && segs[4] == "iridescenceFactor" && value.size() >= 1) {
                    if (mat.khrMaterialsIridescence) mat.khrMaterialsIridescence->iridescenceFactor = value[0];
                } else if (ext == "KHR_materials_dispersion" && segs.size() >= 5 && segs[4] == "dispersion" && value.size() >= 1) {
                    mat.khrMaterialsDispersion = value[0];
                }
            }
        } else if (segs[0] == "extensions" && segs.size() >= 5 && segs[1] == "KHR_lights_punctual" && segs[2] == "lights" && asset->khrLightsPunctual) {
            int idx = std::stoi(segs[3]);
            if (idx < 0 || (size_t)idx >= asset->khrLightsPunctual->size()) continue;
            const std::string& prop = segs[4];
            if (prop == "color" && value.size() >= 3) {
                (*asset->khrLightsPunctual)[idx].color = VM::Vector3<double>(value[0], value[1], value[2]);
            } else if (prop == "intensity" && value.size() >= 1) {
                (*asset->khrLightsPunctual)[idx].intensity = value[0];
            }
        }
    }

    animatedPointers.clear();
    return modifiedMaterials;
}

} // namespace systems::leal::campello_renderer
