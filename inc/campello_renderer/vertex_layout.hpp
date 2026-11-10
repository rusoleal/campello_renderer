#pragma once

// Vertex layouts described by data (docs/design/programmable-pipelines.md,
// phase 1): a mesh says which attributes it has and in which formats, and
// the renderer builds (and caches) pipelines for it, instead of one
// hardcoded pipeline member per vertex format.

#include <campello_gpu/descriptors/vertex_descriptor.hpp>

#include <cstdint>
#include <vector>

namespace systems::leal::campello_renderer {

    /// What a vertex attribute means. The value is the attribute's vertex
    /// buffer slot and shader location in the built-in shaders (phase 1 keeps
    /// today's slot convention: Renderer::VERTEX_SLOT_*).
    enum class VertexSemantic : uint8_t {
        position  = 0,
        normal    = 1,
        texcoord0 = 2,
        tangent   = 3,
        joints    = 4,
        weights   = 5,
        color0    = 6,
        texcoord1 = 7,
        // App-defined per-vertex data for custom shader programs (e.g. a
        // uint32 element id). Shader locations 8-11. The built-in PBR
        // shaders don't read them, so a mesh may carry them and still be
        // drawn by the built-in program.
        custom0   = 8,
        custom1   = 9,
        custom2   = 10,
        custom3   = 11,
    };
    inline constexpr uint32_t kVertexSemanticCount = 12;
    /// The glTF semantics (position .. texcoord1): what VertexLayout::standard() has.
    inline constexpr uint32_t kStandardSemanticMask = 0xFFu;

    constexpr uint32_t semanticBit(VertexSemantic semantic) {
        return 1u << static_cast<uint32_t>(semantic);
    }

    /// Format of one attribute. Each attribute lives in its own tightly
    /// packed buffer (non-interleaved, as GpuMesh::vertexBuffers).
    struct VertexAttributeFormat {
        VertexSemantic semantic = VertexSemantic::position;
        systems::leal::campello_gpu::ComponentType componentType =
            systems::leal::campello_gpu::ComponentType::ctFloat;
        systems::leal::campello_gpu::AccessorType accessorType =
            systems::leal::campello_gpu::AccessorType::acVec3;
        bool normalized = false;

        bool operator==(const VertexAttributeFormat&) const = default;

        /// Bytes per element: tightly packed, rounded up to 4 bytes (glTF
        /// vertex attribute alignment; Metal requires it), e.g. 4 for a
        /// normalized unsigned-byte vec3.
        uint32_t stride() const {
            using CT = systems::leal::campello_gpu::ComponentType;
            using AT = systems::leal::campello_gpu::AccessorType;
            uint32_t size = 4;
            switch (componentType) {
                case CT::ctByte: case CT::ctUnsignedByte: size = 1; break;
                case CT::ctShort: case CT::ctUnsignedShort: size = 2; break;
                case CT::ctUnsignedInt: case CT::ctFloat: size = 4; break;
            }
            uint32_t count = 1;
            switch (accessorType) {
                case AT::acScalar: count = 1; break;
                case AT::acVec2: count = 2; break;
                case AT::acVec3: count = 3; break;
                case AT::acVec4: count = 4; break;
                case AT::acMat2: count = 4; break;
                case AT::acMat3: count = 9; break;
                case AT::acMat4: count = 16; break;
            }
            return (size * count + 3u) & ~3u;
        }
    };

    /// The attributes a mesh has, in ascending semantic order, at most one
    /// per semantic.
    struct VertexLayout {
        std::vector<VertexAttributeFormat> attributes;

        bool operator==(const VertexLayout&) const = default;

        /// One bit per semantic present (semanticBit()).
        uint32_t semanticMask() const {
            uint32_t mask = 0;
            for (const auto& a : attributes) mask |= semanticBit(a.semantic);
            return mask;
        }

        const VertexAttributeFormat* find(VertexSemantic semantic) const {
            for (const auto& a : attributes) {
                if (a.semantic == semantic) return &a;
            }
            return nullptr;
        }

        /// Stable 64-bit hash (FNV-1a over the formats), used as a pipeline
        /// cache key.
        uint64_t hash() const {
            uint64_t h = 1469598103934665603ull;
            const auto mix = [&h](uint64_t v) {
                h ^= v;
                h *= 1099511628211ull;
            };
            for (const auto& a : attributes) {
                mix(static_cast<uint64_t>(a.semantic));
                mix(static_cast<uint64_t>(a.componentType));
                mix(static_cast<uint64_t>(a.accessorType));
                mix(a.normalized ? 1u : 0u);
            }
            return h;
        }

        /// The glTF/PBR layout: float position, normal, texcoords, tangent
        /// and color; joints and weights in the given formats.
        static VertexLayout standard(
            systems::leal::campello_gpu::ComponentType joints =
                systems::leal::campello_gpu::ComponentType::ctUnsignedByte,
            systems::leal::campello_gpu::ComponentType weights =
                systems::leal::campello_gpu::ComponentType::ctFloat,
            bool weightsNormalized = false) {
            using CT = systems::leal::campello_gpu::ComponentType;
            using AT = systems::leal::campello_gpu::AccessorType;
            VertexLayout layout;
            layout.attributes = {
                {VertexSemantic::position, CT::ctFloat, AT::acVec3, false},
                {VertexSemantic::normal, CT::ctFloat, AT::acVec3, false},
                {VertexSemantic::texcoord0, CT::ctFloat, AT::acVec2, false},
                {VertexSemantic::tangent, CT::ctFloat, AT::acVec4, false},
                {VertexSemantic::joints, joints, AT::acVec4, false},
                {VertexSemantic::weights, weights, AT::acVec4, weightsNormalized},
                {VertexSemantic::color0, CT::ctFloat, AT::acVec4, false},
                {VertexSemantic::texcoord1, CT::ctFloat, AT::acVec2, false},
            };
            return layout;
        }

        /// Float positions only. Drawn by the built-in shaders with the flat
        /// face normal derived in the fragment shader (screen-space
        /// derivatives of the world position), so vertices can be shared
        /// between faces of different orientation (CAD/BIM geometry).
        static const VertexLayout& positionOnly() {
            static const VertexLayout layout{{{VertexSemantic::position,
                                               systems::leal::campello_gpu::ComponentType::ctFloat,
                                               systems::leal::campello_gpu::AccessorType::acVec3,
                                               false}}};
            return layout;
        }
    };

} // namespace systems::leal::campello_renderer
