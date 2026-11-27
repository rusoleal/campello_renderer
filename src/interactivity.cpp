// KHR_interactivity host: the glTF Asset Object Model on the renderer's asset
// (see interactivity.hpp for the supported pointers).

#include <campello_renderer/interactivity.hpp>
#include <campello_renderer/campello_renderer.hpp>

#include <gltf/gltf.hpp>

#include <cmath>
#include <limits>
#include <vector>

namespace systems::leal::campello_renderer {

namespace {

namespace G = systems::leal::gltf;
namespace VM = systems::leal::vector_math;
using Sig = G::InteractivityTypeSignature;
using Value = G::InteractivityRuntimeValue;

// "/a/b~1c" -> {"a", "b/c"}.
std::vector<std::string> segments(const std::string& pointer) {
    std::vector<std::string> out;
    if (pointer.empty() || pointer[0] != '/') return out;
    size_t i = 1;
    for (;;) {
        const size_t j = pointer.find('/', i);
        const std::string raw = pointer.substr(i, j == std::string::npos ? std::string::npos : j - i);
        std::string seg;
        for (size_t k = 0; k < raw.size(); ++k) {
            if (raw[k] == '~' && k + 1 < raw.size() && (raw[k + 1] == '0' || raw[k + 1] == '1')) {
                seg += raw[k + 1] == '1' ? '/' : '~';
                ++k;
            } else {
                seg += raw[k];
            }
        }
        out.push_back(seg);
        if (j == std::string::npos) break;
        i = j + 1;
    }
    return out;
}

// A JSON Pointer array index ("0", "12"; no sign, no leading zeros).
bool index(const std::string& s, uint64_t& out) {
    if (s.empty() || s.size() > 10 || s.find_first_not_of("0123456789") != std::string::npos) return false;
    if (s.size() > 1 && s[0] == '0') return false;
    out = std::stoull(s);
    return true;
}

// A float property of `prop` components stored contiguously at `data`.
template <typename T>
bool floats(Sig type, Sig prop, T* data, Value* get, const Value* set) {
    if (type != prop) return false;
    const int n = Value::componentCount(prop);
    if (get) {
        *get = Value::defaultOf(prop);
        for (int k = 0; k < n; ++k) get->f[k] = (double)data[k];
    }
    if (set) {
        for (int k = 0; k < n; ++k) data[k] = (T)set->f[k];
    }
    return true;
}

bool boolean(Sig type, bool& b, Value* get, const Value* set) {
    if (type != Sig::Bool) return false;
    if (get) *get = Value::ofBool(b);
    if (set) b = set->b;
    return true;
}

// A read-only float4x4 (row-major vector_math matrix -> column-major value).
template <typename T>
bool matrix(Sig type, const VM::Matrix4<T>& m, Value* get, const Value* set) {
    if (type != Sig::Float4x4 || set) return false;
    *get = Value::defaultOf(Sig::Float4x4);
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) get->f[c * 4 + r] = (double)m.data[r * 4 + c];
    return true;
}

template <typename T>
bool authoredMatrix(const VM::Matrix4<T>& m) {
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            if (m.data[r * 4 + c] != (r == c ? T(1) : T(0))) return true;
    return false;
}

// A node's local transform: its authored matrix, else its TRS.
VM::Matrix4<double> localMatrix(const G::Node& node) {
    if (authoredMatrix(node.matrix)) return node.matrix;
    return VM::Matrix4<double>::compose(node.translation, node.rotation, node.scale);
}

// The unit quaternion (x, y, z, w) of a rotation matrix m[row][col].
void quaternionFromMatrix(const double m[3][3], double q[4]) {
    const double trace = m[0][0] + m[1][1] + m[2][2];
    if (trace > 0) {
        const double t = std::sqrt(trace + 1.0) * 2;
        q[0] = (m[2][1] - m[1][2]) / t; q[1] = (m[0][2] - m[2][0]) / t; q[2] = (m[1][0] - m[0][1]) / t; q[3] = 0.25 * t;
    } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
        const double t = std::sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2;
        q[0] = 0.25 * t; q[1] = (m[0][1] + m[1][0]) / t; q[2] = (m[0][2] + m[2][0]) / t; q[3] = (m[2][1] - m[1][2]) / t;
    } else if (m[1][1] > m[2][2]) {
        const double t = std::sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2;
        q[0] = (m[0][1] + m[1][0]) / t; q[1] = 0.25 * t; q[2] = (m[1][2] + m[2][1]) / t; q[3] = (m[0][2] - m[2][0]) / t;
    } else {
        const double t = std::sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2;
        q[0] = (m[0][2] + m[2][0]) / t; q[1] = (m[1][2] + m[2][1]) / t; q[2] = 0.25 * t; q[3] = (m[1][0] - m[0][1]) / t;
    }
}

// KHR_texture_transform of a texture slot (path: [...]/extensions/KHR_texture_transform/<prop>).
bool textureTransform(G::TextureInfo* info, const std::string& prop, Sig type, Value* get, const Value* set) {
    if (!info) return false;
    if (prop != "offset" && prop != "rotation" && prop != "scale") return false;
    const Sig propType = prop == "rotation" ? Sig::Float : Sig::Float2;
    if (type != propType) return false;
    using Real = GLTF_REAL_NUMBER_TYPE;
    if (!info->khrTextureTransform) {
        if (get) {
            // The extension's defaults.
            const double d[2] = {prop == "scale" ? 1.0 : 0.0, prop == "scale" ? 1.0 : 0.0};
            *get = Value::ofFloats(propType, d);
            return true;
        }
        info->khrTextureTransform = std::make_shared<G::KHRTextureTransform>(VM::Vector2<Real>(0, 0), Real(0),
                                                                              VM::Vector2<Real>(1, 1), -1);
    }
    auto& t = *info->khrTextureTransform;
    if (prop == "offset") return floats(type, propType, t.offset.data, get, set);
    if (prop == "scale") return floats(type, propType, t.scale.data, get, set);
    return floats(type, propType, &t.rotation, get, set);
}

// A property of one of a material's KHR_materials_* extensions: a factor
// (`data`, of `type`) or a texture slot. Empty when the material doesn't use
// the extension or it has no such property.
struct MaterialExtensionProperty {
    GLTF_REAL_NUMBER_TYPE* data = nullptr;
    Sig type = Sig::Float;
    G::TextureInfo* texture = nullptr;
};

MaterialExtensionProperty materialExtensionProperty(G::Material& m, const std::string& extension, const std::string& name) {
    MaterialExtensionProperty out;
    auto factor = [&](GLTF_REAL_NUMBER_TYPE* data, Sig type = Sig::Float) {
        out.data = data;
        out.type = type;
    };
    auto texture = [&](const auto& slot) { out.texture = slot.get(); };
    if (extension == "KHR_materials_transmission" && m.khrMaterialsTransmission) {
        auto& e = *m.khrMaterialsTransmission;
        if (name == "transmissionFactor") factor(&e.transmissionFactor);
        else if (name == "transmissionTexture") texture(e.transmissionTexture);
    } else if (extension == "KHR_materials_clearcoat" && m.khrMaterialsClearcoat) {
        auto& e = *m.khrMaterialsClearcoat;
        if (name == "clearcoatFactor") factor(&e.clearcoatFactor);
        else if (name == "clearcoatRoughnessFactor") factor(&e.clearcoatRoughnessFactor);
        else if (name == "clearcoatTexture") texture(e.clearcoatTexture);
        else if (name == "clearcoatRoughnessTexture") texture(e.clearcoatRoughnessTexture);
        else if (name == "clearcoatNormalTexture") texture(e.clearcoatNormalTexture);
    } else if (extension == "KHR_materials_sheen" && m.khrMaterialsSheen) {
        auto& e = *m.khrMaterialsSheen;
        if (name == "sheenColorFactor") factor(e.sheenColorFactor.data, Sig::Float3);
        else if (name == "sheenRoughnessFactor") factor(&e.sheenRoughnessFactor);
        else if (name == "sheenColorTexture") texture(e.sheenColorTexture);
        else if (name == "sheenRoughnessTexture") texture(e.sheenRoughnessTexture);
    } else if (extension == "KHR_materials_specular" && m.khrMaterialsSpecular) {
        auto& e = *m.khrMaterialsSpecular;
        if (name == "specularFactor") factor(&e.specularFactor);
        else if (name == "specularColorFactor") factor(e.specularColorFactor.data, Sig::Float3);
        else if (name == "specularTexture") texture(e.specularTexture);
        else if (name == "specularColorTexture") texture(e.specularColorTexture);
    } else if (extension == "KHR_materials_volume" && m.khrMaterialsVolume) {
        auto& e = *m.khrMaterialsVolume;
        if (name == "thicknessFactor") factor(&e.thicknessFactor);
        else if (name == "attenuationDistance") factor(&e.attenuationDistance);
        else if (name == "attenuationColor") factor(e.attenuationColor.data, Sig::Float3);
        else if (name == "thicknessTexture") texture(e.thicknessTexture);
    } else if (extension == "KHR_materials_iridescence" && m.khrMaterialsIridescence) {
        auto& e = *m.khrMaterialsIridescence;
        if (name == "iridescenceFactor") factor(&e.iridescenceFactor);
        else if (name == "iridescenceIor") factor(&e.iridescenceIor);
        else if (name == "iridescenceThicknessMinimum") factor(&e.iridescenceThicknessMinimum);
        else if (name == "iridescenceThicknessMaximum") factor(&e.iridescenceThicknessMaximum);
        else if (name == "iridescenceTexture") texture(e.iridescenceTexture);
        else if (name == "iridescenceThicknessTexture") texture(e.iridescenceThicknessTexture);
    } else if (extension == "KHR_materials_anisotropy" && m.khrMaterialsAnisotropy) {
        auto& e = *m.khrMaterialsAnisotropy;
        if (name == "anisotropyStrength") factor(&e.anisotropyStrength);
        else if (name == "anisotropyRotation") factor(&e.anisotropyRotation);
        else if (name == "anisotropyTexture") texture(e.anisotropyTexture);
    } else if (extension == "KHR_materials_diffuse_transmission" && m.khrMaterialsDiffuseTransmission) {
        auto& e = *m.khrMaterialsDiffuseTransmission;
        if (name == "diffuseTransmissionFactor") factor(&e.diffuseTransmissionFactor);
        else if (name == "diffuseTransmissionColorFactor") factor(e.diffuseTransmissionColorFactor.data, Sig::Float3);
        else if (name == "diffuseTransmissionTexture") texture(e.diffuseTransmissionTexture);
        else if (name == "diffuseTransmissionColorTexture") texture(e.diffuseTransmissionColorTexture);
    }
    return out;
}

}  // namespace

InteractivityObjectModel::InteractivityObjectModel(Renderer& renderer) : renderer_(renderer) {}

bool InteractivityObjectModel::getPointer(const std::string& pointer, Sig type, Value& value) {
    return access(pointer, type, &value, nullptr);
}

bool InteractivityObjectModel::setPointer(const std::string& pointer, const Value& value) {
    return access(pointer, value.type, nullptr, &value);
}

void InteractivityObjectModel::applyAnimation(int64_t animation, double time) {
    if (animation >= 0) renderer_.poseAnimation((uint32_t)animation, time);
}

void InteractivityObjectModel::animationStopped(int64_t animation) {
    if (animation >= 0) renderer_.releaseAnimation((uint32_t)animation);
}

void InteractivityObjectModel::flush() {
    for (const uint64_t m : dirtyMaterials_) renderer_.materialChanged(m);
    dirtyMaterials_.clear();
}

const std::vector<int64_t>& InteractivityObjectModel::parentsOf(const G::GLTF& asset) {
    const size_t count = asset.nodes ? asset.nodes->size() : 0;
    if (parentsAsset_ != &asset || parents_.size() != count) {
        parentsAsset_ = &asset;
        parents_.assign(count, -1);
        for (size_t p = 0; p < count; ++p)
            for (const auto c : (*asset.nodes)[p].children)
                if (c < count) parents_[c] = (int64_t)p;
    }
    return parents_;
}

bool InteractivityObjectModel::access(const std::string& pointer, Sig type, Value* get, const Value* set) {
    auto asset = renderer_.getAsset();
    if (!asset) return false;
    const auto seg = segments(pointer);
    const size_t n = seg.size();
    uint64_t i = 0;

    // /nodes/{}/...
    if (n >= 3 && seg[0] == "nodes" && index(seg[1], i)) {
        if (!asset->nodes || i >= asset->nodes->size()) return false;
        auto& node = (*asset->nodes)[i];
        const std::string& p = seg[2];
        if (n == 3) {
            if (p == "translation" || p == "rotation" || p == "scale") {
                if (set && authoredMatrix(node.matrix)) return false;  // TRS of a node with a matrix
                if (p == "translation") return floats(type, Sig::Float3, node.translation.data, get, set);
                if (p == "rotation") return floats(type, Sig::Float4, node.rotation.data, get, set);
                return floats(type, Sig::Float3, node.scale.data, get, set);
            }
            if (p == "matrix") {
                if (!get) return false;
                return matrix(type, localMatrix(node), get, set);
            }
            if (p == "globalMatrix") {
                // From the nodes' current transforms, not the last frame's: a
                // pointer/set of an ancestor shows at once.
                if (!get) return false;
                const auto& parents = parentsOf(*asset);
                VM::Matrix4<double> world = localMatrix(node);
                size_t steps = 0;
                for (int64_t up = parents[i]; up >= 0 && steps < parents.size(); up = parents[(size_t)up], ++steps)
                    world = localMatrix((*asset->nodes)[(size_t)up]) * world;
                return matrix(type, world, get, set);
            }
            return false;
        }
        // One morph target weight: the node's (its mesh's until written).
        uint64_t k = 0;
        if (n == 4 && p == "weights" && index(seg[3], k)) {
            if (node.mesh < 0 || !asset->meshes || (size_t)node.mesh >= asset->meshes->size()) return false;
            const auto& mesh = (*asset->meshes)[(size_t)node.mesh];
            const size_t targets = mesh.primitives.empty() ? 0 : mesh.primitives[0].targets.size();
            if (k >= targets) return false;
            if (node.weights.size() < targets) {
                std::vector<GLTF_REAL_NUMBER_TYPE> w(targets, 0);
                for (size_t t = 0; t < targets; ++t) {
                    if (t < node.weights.size()) w[t] = node.weights[t];
                    else if (t < mesh.weights.size()) w[t] = mesh.weights[t];
                }
                node.weights = std::move(w);
            }
            return floats(type, Sig::Float, &node.weights[k], get, set);
        }
        if (n == 5 && p == "extensions") {
            if (seg[3] == "KHR_node_visibility" && seg[4] == "visible") return boolean(type, node.visible, get, set);
            if (seg[3] == "KHR_node_selectability" && seg[4] == "selectable") return boolean(type, node.selectable, get, set);
            if (seg[3] == "KHR_node_hoverability" && seg[4] == "hoverable") return boolean(type, node.hoverable, get, set);
        }
        return false;
    }

    // /materials/{}/...
    if (n >= 3 && seg[0] == "materials" && index(seg[1], i)) {
        if (!asset->materials || i >= asset->materials->size()) return false;
        auto& m = (*asset->materials)[i];
        bool ok = false;
        const std::string& p = seg[2];
        G::TextureInfo* slot = nullptr;
        size_t rest = 0;  // where ".../extensions/KHR_texture_transform/<prop>" starts
        if (p == "pbrMetallicRoughness" && n >= 4 && m.pbrMetallicRoughness) {
            auto& pbr = *m.pbrMetallicRoughness;
            if (n == 4 && seg[3] == "baseColorFactor") ok = floats(type, Sig::Float4, pbr.baseColorFactor.data, get, set);
            else if (n == 4 && seg[3] == "metallicFactor") ok = floats(type, Sig::Float, &pbr.metallicFactor, get, set);
            else if (n == 4 && seg[3] == "roughnessFactor") ok = floats(type, Sig::Float, &pbr.roughnessFactor, get, set);
            else if (seg[3] == "baseColorTexture") { slot = pbr.baseColorTexture.get(); rest = 4; }
            else if (seg[3] == "metallicRoughnessTexture") { slot = pbr.metallicRoughnessTexture.get(); rest = 4; }
        } else if (n == 3 && p == "alphaCutoff") {
            ok = floats(type, Sig::Float, &m.alphaCutoff, get, set);
        } else if (n == 3 && p == "emissiveFactor") {
            ok = floats(type, Sig::Float3, m.emissiveFactor.data, get, set);
        } else if (p == "normalTexture" && m.normalTexture) {
            if (n == 4 && seg[3] == "scale") ok = floats(type, Sig::Float, &m.normalTexture->scale, get, set);
            else { slot = m.normalTexture.get(); rest = 3; }
        } else if (p == "occlusionTexture" && m.occlusionTexture) {
            if (n == 4 && seg[3] == "strength") ok = floats(type, Sig::Float, &m.occlusionTexture->strength, get, set);
            else { slot = m.occlusionTexture.get(); rest = 3; }
        } else if (p == "emissiveTexture") {
            slot = m.emissiveTexture.get();
            rest = 3;
        } else if (n == 5 && p == "extensions") {
            if (seg[3] == "KHR_materials_emissive_strength" && seg[4] == "emissiveStrength")
                ok = floats(type, Sig::Float, &m.khrMaterialsEmissiveStrength, get, set);
            else if (seg[3] == "KHR_materials_ior" && seg[4] == "ior")
                ok = floats(type, Sig::Float, &m.khrMaterialsIor, get, set);
            else if (seg[3] == "KHR_materials_dispersion" && seg[4] == "dispersion")
                ok = floats(type, Sig::Float, &m.khrMaterialsDispersion, get, set);
            else {
                const MaterialExtensionProperty prop = materialExtensionProperty(m, seg[3], seg[4]);
                if (prop.data) ok = floats(type, prop.type, prop.data, get, set);
            }
        } else if (n == 6 && p == "extensions" && seg[3] == "KHR_materials_clearcoat" &&
                   seg[4] == "clearcoatNormalTexture" && seg[5] == "scale") {
            if (m.khrMaterialsClearcoat && m.khrMaterialsClearcoat->clearcoatNormalTexture)
                ok = floats(type, Sig::Float, &m.khrMaterialsClearcoat->clearcoatNormalTexture->scale, get, set);
        } else if (n == 8 && p == "extensions" && seg[5] == "extensions" && seg[6] == "KHR_texture_transform") {
            // .../extensions/KHR_materials_<x>/<x>Texture/extensions/KHR_texture_transform/<prop>
            const MaterialExtensionProperty prop = materialExtensionProperty(m, seg[3], seg[4]);
            if (prop.texture) ok = textureTransform(prop.texture, seg[7], type, get, set);
        }
        if (slot && rest && n == rest + 3 && seg[rest] == "extensions" && seg[rest + 1] == "KHR_texture_transform")
            ok = textureTransform(slot, seg[rest + 2], type, get, set);
        if (ok && set) dirtyMaterials_.insert(i);
        return ok;
    }

    // /cameras/{}/perspective/... and /cameras/{}/orthographic/...: the frame
    // after a pointer/set renders with the new value (when it's the camera in use).
    if (n == 4 && seg[0] == "cameras" && index(seg[1], i)) {
        if (!asset->cameras || i >= asset->cameras->size() || type != Sig::Float) return false;
        auto* camera = (*asset->cameras)[i].get();
        const std::string& prop = seg[3];
        if (seg[2] == "perspective") {
            auto* persp = dynamic_cast<G::PerspectiveCamera*>(camera);
            if (!persp) return false;
            if (prop == "yfov") return floats(type, Sig::Float, &persp->yFov, get, set);
            if (prop == "znear") return floats(type, Sig::Float, &persp->zNear, get, set);
            if (prop == "aspectRatio" || prop == "zfar") {
                // Optional: an undefined aspectRatio (the viewport's) doesn't
                // resolve until written; an undefined zfar is infinite.
                auto& field = prop == "zfar" ? persp->zFar : persp->aspectRatio;
                if (get) {
                    if (!field && prop == "aspectRatio") return false;
                    *get = Value::ofFloat(field ? (double)*field : std::numeric_limits<double>::infinity());
                }
                if (set) {
                    if (prop == "zfar" && !std::isfinite(set->f[0])) field.reset();  // infinite
                    else field = (GLTF_REAL_NUMBER_TYPE)set->f[0];
                }
                return true;
            }
            return false;
        }
        if (seg[2] == "orthographic") {
            auto* ortho = dynamic_cast<G::OrthographicCamera*>(camera);
            if (!ortho) return false;
            if (prop == "xmag") return floats(type, Sig::Float, &ortho->xMag, get, set);
            if (prop == "ymag") return floats(type, Sig::Float, &ortho->yMag, get, set);
            if (prop == "znear") return floats(type, Sig::Float, &ortho->zNear, get, set);
            if (prop == "zfar") return floats(type, Sig::Float, &ortho->zFar, get, set);
        }
        return false;
    }

    // /extensions/...
    if (n >= 4 && seg[0] == "extensions") {
        if (n == 5 && seg[1] == "KHR_lights_punctual" && seg[2] == "lights" && index(seg[3], i)) {
            if (!asset->khrLightsPunctual || i >= asset->khrLightsPunctual->size()) return false;
            auto& light = (*asset->khrLightsPunctual)[i];
            if (seg[4] == "color") return floats(type, Sig::Float3, light.color.data, get, set);
            if (seg[4] == "intensity") return floats(type, Sig::Float, &light.intensity, get, set);
            if (seg[4] == "range") return floats(type, Sig::Float, &light.range, get, set);
            return false;
        }
        // /extensions/KHR_lights_punctual/lights/{}/spot/innerConeAngle, outerConeAngle: spot lights only.
        if (n == 6 && seg[1] == "KHR_lights_punctual" && seg[2] == "lights" && index(seg[3], i) && seg[4] == "spot") {
            if (!asset->khrLightsPunctual || i >= asset->khrLightsPunctual->size()) return false;
            auto& light = (*asset->khrLightsPunctual)[i];
            if (light.type != G::KHRLightPunctualType::spot) return false;
            if (seg[5] == "innerConeAngle") return floats(type, Sig::Float, &light.innerConeAngle, get, set);
            if (seg[5] == "outerConeAngle") return floats(type, Sig::Float, &light.outerConeAngle, get, set);
            return false;
        }
        if (n == 5 && seg[1] == "KHR_interactivity" && seg[2] == "activeCamera" && get && type == Sig::Float &&
            (seg[3] == "perspective" || seg[3] == "orthographic")) {
            // From the last frame's projection (row-major, depth 0..1, either
            // forward axis; reversed depth too). The other projection's values
            // are NaN, as are all of them before the first frame.
            const auto& p = renderer_.getLastProjectionMatrix().data;
            const double nan = std::numeric_limits<double>::quiet_NaN();
            const bool perspective = p[15] == 0.0 && p[14] != 0.0;
            const bool orthographic = p[15] != 0.0 && p[14] == 0.0 && p[0] != 0.0;
            double value = nan;
            if (seg[3] == "perspective" && perspective && p[0] != 0.0 && p[5] != 0.0) {
                // Depth d ahead: z = s*d, w = d, ndc = p10*s + p11/d.
                const double s = p[14], a = p[10] * s, b = p[11];
                const double d0 = a != 0.0 ? -b / a : nan;                         // ndc 0
                const double d1 = 1.0 - a != 0.0 ? b / (1.0 - a) : std::numeric_limits<double>::infinity();  // ndc 1
                if (seg[4] == "yfov") value = 2.0 * std::atan(1.0 / std::abs(p[5]));
                else if (seg[4] == "aspectRatio") value = std::abs(p[5] / p[0]);
                else if (seg[4] == "znear") value = std::isnan(d0) ? nan : std::min(std::abs(d0), std::abs(d1));
                else if (seg[4] == "zfar") value = std::isnan(d0) ? nan : std::max(std::abs(d0), std::abs(d1));
                else return false;
            } else if (seg[3] == "orthographic" && orthographic && p[10] != 0.0 && p[5] != 0.0) {
                // ndc = p10*z + p11 (w = 1).
                const double z0 = -p[11] / p[10], z1 = (1.0 - p[11]) / p[10];
                if (seg[4] == "xmag") value = 1.0 / std::abs(p[0]);
                else if (seg[4] == "ymag") value = 1.0 / std::abs(p[5]);
                else if (seg[4] == "znear") value = std::min(std::abs(z0), std::abs(z1));
                else if (seg[4] == "zfar") value = std::max(std::abs(z0), std::abs(z1));
                else return false;
            } else if (!(seg[4] == "yfov" || seg[4] == "aspectRatio" || seg[4] == "xmag" || seg[4] == "ymag" ||
                         seg[4] == "znear" || seg[4] == "zfar")) {
                return false;
            }
            *get = Value::ofFloat(value);
            return true;
        }
        if (n == 4 && seg[1] == "KHR_interactivity" && seg[2] == "activeCamera" && get) {
            const auto camera = renderer_.getLastViewMatrix().inverted();
            if (seg[3] == "position" && type == Sig::Float3) {
                const double pos[3] = {camera.data[3], camera.data[7], camera.data[11]};
                *get = Value::ofFloats(Sig::Float3, pos);
                return true;
            }
            if (seg[3] == "rotation" && type == Sig::Float4) {
                double m[3][3];
                for (int c = 0; c < 3; ++c) {
                    const double len = std::sqrt(camera.data[c] * camera.data[c] + camera.data[4 + c] * camera.data[4 + c] +
                                                 camera.data[8 + c] * camera.data[8 + c]);
                    for (int r = 0; r < 3; ++r) m[r][c] = len > 0 ? camera.data[r * 4 + c] / len : 0.0;
                }
                // A view that keeps "forward" as +Z (vector_math's lookAt) is a
                // reflection: flip Z, as glTF cameras look down -Z.
                const double det = m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2]) -
                                   m[0][1] * (m[1][0] * m[2][2] - m[2][0] * m[1][2]) +
                                   m[0][2] * (m[1][0] * m[2][1] - m[2][0] * m[1][1]);
                if (det < 0)
                    for (int r = 0; r < 3; ++r) m[r][2] = -m[r][2];
                double q[4];
                quaternionFromMatrix(m, q);
                *get = Value::ofFloats(Sig::Float4, q);
                return true;
            }
        }
    }
    return false;
}

}  // namespace systems::leal::campello_renderer
