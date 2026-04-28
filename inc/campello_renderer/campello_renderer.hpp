#pragma once

#include <memory>
#include <array>
#include <string>
#include <unordered_map>
#include <vector>
#include <gltf/gltf.hpp>
#include <campello_gpu/device.hpp>
#include <campello_gpu/render_pass_encoder.hpp>
#include <campello_gpu/texture.hpp>
#include <campello_gpu/texture_view.hpp>
#include <campello_gpu/buffer.hpp>
#include <campello_gpu/bind_group.hpp>
#include <campello_gpu/bind_group_layout.hpp>
#include <campello_renderer/image.hpp>
#include <campello_renderer/animation.hpp>
#include <campello_renderer/procedural_texture_baker.hpp>
#include <vector_math/vector_math.hpp>

namespace systems::leal::campello_renderer {

    struct GpuMesh {
        std::shared_ptr<systems::leal::campello_gpu::Buffer> indexBuffer;
        uint32_t indexCount = 0;
        systems::leal::campello_gpu::IndexFormat indexFormat =
            systems::leal::campello_gpu::IndexFormat::uint32;
        std::array<std::shared_ptr<systems::leal::campello_gpu::Buffer>, 6> vertexBuffers; // POSITION, NORMAL, UV, TANGENT, JOINTS, WEIGHTS
        uint32_t vertexCount = 0;
        // topology, bounds — reserved for future phases
    };

    struct GpuMaterial {
        std::shared_ptr<systems::leal::campello_gpu::BindGroup> bindGroup;     // textures + samplers
        std::shared_ptr<systems::leal::campello_gpu::BindGroup> flatBindGroup; // fallback for flat pipeline
        uint32_t uniformSlot = 0; // index into the global material uniform buffer
        bool doubleSided = false;
        bool alphaBlend = false;
        bool alphaMask = false;
        bool transmission = false; // true if KHR_materials_transmission is active
    };

    struct DrawCall {
        GpuMesh* mesh = nullptr;
        GpuMaterial* material = nullptr;
        systems::leal::vector_math::Matrix4<double> worldTransform;
        std::shared_ptr<systems::leal::campello_gpu::Buffer> jointMatrixBuffer; // nullptr if unskinned
        uint32_t jointCount = 0;
        uint32_t instanceCount = 1;
    };

    struct CameraData {
        systems::leal::vector_math::Matrix4<double> view;
        systems::leal::vector_math::Matrix4<double> projection;
        systems::leal::vector_math::Vector3<double> position;
    };

    struct LightData {
        systems::leal::vector_math::Vector3<double> direction;
        systems::leal::vector_math::Vector3<double> position;
        systems::leal::vector_math::Vector3<double> color;
        float intensity = 0.0f;
        float range = 0.0f;
        float innerConeAngle = 0.0f;
        float outerConeAngle = 0.0f;
        int type = 0; // 0 = directional, 1 = point, 2 = spot
    };

    struct RenderScene {
        CameraData camera;
        std::vector<LightData> lights;
        std::vector<DrawCall> opaque;
        std::vector<DrawCall> transparent;
    };


    enum class ViewMode : uint32_t {
        normal = 0,
        worldNormal = 1,
        baseColor = 2,
        metallic = 3,
        roughness = 4,
        occlusion = 5,
        emissive = 6,
        alpha = 7,
        uv0 = 8,
        specularFactor = 9,
        specularColor = 10,
        sheenColor = 11,
        sheenRoughness = 12,
        clearcoat = 13,
        clearcoatRoughness = 14,
        clearcoatNormal = 15,
        transmission = 16,
        environment = 17,
        iridescence = 18,
        anisotropy = 19,
        dispersion = 20,
    };

    struct RenderStats {
        uint32_t opaqueDrawCount = 0;
        uint32_t transparentDrawCount = 0;
        uint32_t totalDrawCount = 0;
        uint32_t culledNodeCount = 0;
        uint32_t visibleNodeCount = 0;
        uint32_t instanceCount = 0;
        double cpuFrameTimeMs = 0.0;
    };

    class Renderer {
    public:
        uint32_t getRenderWidth() const  { return renderWidth; }
        uint32_t getRenderHeight() const { return renderHeight; }

        // Vertex buffer slot convention — must match the pipeline's vertex descriptor
        // and the shader bindings.
        static constexpr uint32_t VERTEX_SLOT_POSITION   = 0;  // float3 POSITION
        static constexpr uint32_t VERTEX_SLOT_NORMAL     = 1;  // float3 NORMAL
        static constexpr uint32_t VERTEX_SLOT_TEXCOORD0  = 2;  // float2 TEXCOORD_0
        static constexpr uint32_t VERTEX_SLOT_TANGENT    = 3;  // float4 TANGENT
        static constexpr uint32_t VERTEX_SLOT_JOINTS     = 4;  // uint4  JOINTS_0  (joint indices)
        static constexpr uint32_t VERTEX_SLOT_WEIGHTS    = 5;  // float4 WEIGHTS_0 (joint weights)
        static constexpr uint32_t VERTEX_SLOT_MVP        = 16; // float4x4, row-major MVP matrix (per-node)
        static constexpr uint32_t VERTEX_SLOT_MATERIAL   = 17; // MaterialUniforms (112 bytes, 256 stride)
        static constexpr uint32_t VERTEX_SLOT_CAMERA     = 18; // CameraPosition (float3 world space)
        static constexpr uint32_t VERTEX_SLOT_LIGHTS     = 15; // LightsUniforms - use slot 15 to avoid conflict
        static constexpr uint32_t VERTEX_SLOT_INSTANCE_MATRIX = 19; // float4x4 per-instance transform (EXT_mesh_gpu_instancing)
        static constexpr uint32_t VERTEX_SLOT_JOINT_MATRICES  = 20; // float4x4[] joint matrix palette (per-skin)
                                                               //   [0..15]  baseColorFactor
                                                               //   [16..31] uvTransformRow0 (.w = hasTransform)
                                                               //   [32..47] uvTransformRow1
                                                               //   [48..51] metallicFactor
                                                               //   [52..55] roughnessFactor
                                                               //   [56..59] normalScale
                                                               //   [60..63] alphaMode (0=opaque,1=mask,2=blend)
                                                               //   [64..67] alphaCutoff
                                                               //   [68..71] unlit (0=lit,1=unlit)
                                                               //   [72..75] hasNormalTexture (0=no,1=yes)

    private:
        std::shared_ptr<systems::leal::campello_gpu::Device> device;
        std::shared_ptr<systems::leal::gltf::GLTF> asset;
        std::string assetBasePath;
        uint32_t sceneIndex  = 0;
        uint32_t cameraIndex = 0;
        std::vector<std::shared_ptr<Image>> images;

        std::vector<std::shared_ptr<systems::leal::campello_gpu::Buffer>>  gpuBuffers;
        std::vector<std::shared_ptr<systems::leal::campello_gpu::Texture>> gpuTextures;

        // Per-GLTF-sampler GPU samplers and per-material bind groups.
        std::vector<std::shared_ptr<systems::leal::campello_gpu::Sampler>>        gpuSamplers;
        std::vector<std::shared_ptr<systems::leal::campello_gpu::BindGroup>>      materialBindGroups;

        // Engine-native resource pools — owned by the renderer, independent of glTF lifetime.
        std::vector<std::unique_ptr<GpuMesh>>     meshPool;
        std::vector<std::unique_ptr<GpuMaterial>> materialPool;

        // Look-up caches from glTF indices/handles to engine-native handles.
        std::unordered_map<const systems::leal::gltf::Primitive*, GpuMesh*>     meshCache;
        std::unordered_map<int64_t, GpuMaterial*>                               materialCache;

        // Minimal bind groups for flat pipeline variants (no textures/samplers).
        // Only contains the material uniform buffer at binding 17 so the fragment
        // shader can read mat.baseColorFactor without triggering unused-binding
        // asserts for texture/sampler slots.
        std::shared_ptr<systems::leal::campello_gpu::BindGroup>        defaultFlatBindGroup;
        std::vector<std::shared_ptr<systems::leal::campello_gpu::BindGroup>>      flatMaterialBindGroups;

        // Draco-compressed primitive GPU buffers (primitive pointer → index buffer + attribute buffers).
        // Populated in setScene() after calling GLTF::decompressDraco().
        struct DracoBuffers {
            std::shared_ptr<systems::leal::campello_gpu::Buffer> indexBuffer;
            std::unordered_map<std::string, std::shared_ptr<systems::leal::campello_gpu::Buffer>> attributeBuffers;
            uint32_t indexCount = 0;  // Number of indices for drawing
        };
        std::unordered_map<const systems::leal::gltf::Primitive *, DracoBuffers> dracoPrimitiveBuffers;

        // Deinterleaved vertex attribute buffers for accessors in buffer views with
        // byteStride > 0. Key is accessor index; value is a contiguous GPU buffer.
        std::unordered_map<int64_t, std::shared_ptr<systems::leal::campello_gpu::Buffer>> deinterleavedBuffers;

        // Shared resources created on first setScene() call.
        std::shared_ptr<systems::leal::campello_gpu::Sampler>          defaultSampler;
        std::shared_ptr<systems::leal::campello_gpu::Texture>          defaultTexture;        // White (1,1,1,1) for baseColor
        std::shared_ptr<systems::leal::campello_gpu::Texture>          defaultMetallicRoughnessTexture; // (0,1,1,1) - G=roughness, B=metallic
        std::shared_ptr<systems::leal::campello_gpu::Texture>          defaultNormalTexture;            // (0.5,0.5,1,1) - flat normal
        std::shared_ptr<systems::leal::campello_gpu::Texture>          defaultEmissiveTexture;          // (0,0,0,1) - black (no emission)
        std::shared_ptr<systems::leal::campello_gpu::Texture>          defaultOcclusionTexture;         // (1,1,1,1) - white (no occlusion)
        std::shared_ptr<systems::leal::campello_gpu::Texture>          defaultSpecularTexture;          // (1,1,1,1) - white alpha = full specularFactor passthrough
        std::shared_ptr<systems::leal::campello_gpu::Texture>          defaultSpecularColorTexture;     // (1,1,1,1) sRGB - white = no specular color tint
        std::shared_ptr<systems::leal::campello_gpu::Texture>          defaultSheenColorTexture;        // (0,0,0,1) sRGB - black = no sheen (default sheenColor=[0,0,0])
        std::shared_ptr<systems::leal::campello_gpu::Texture>          defaultSheenRoughnessTexture;    // (1,1,1,1) linear - white R = factor passes through unchanged
        std::shared_ptr<systems::leal::campello_gpu::Texture>          defaultClearcoatTexture;         // (1,1,1,1) linear - white R = factor passes through (default factor=0)
        std::shared_ptr<systems::leal::campello_gpu::Texture>          defaultClearcoatRoughnessTexture; // (1,1,1,1) linear - white G = factor passes through
        std::shared_ptr<systems::leal::campello_gpu::Texture>          defaultClearcoatNormalTexture;   // (0.5,0.5,1,1) linear - flat normal
        
        // Default identity matrix for non-instanced rendering (EXT_mesh_gpu_instancing).
        // Always bound to slot 19; instanced objects override with their own buffer.
        std::shared_ptr<systems::leal::campello_gpu::Buffer>           defaultInstanceMatrixBuffer;
        
        std::shared_ptr<systems::leal::campello_gpu::BindGroupLayout>  bindGroupLayout;
        std::shared_ptr<systems::leal::campello_gpu::BindGroup>        defaultBindGroup;

        // Built-in pipeline variants created by createDefaultPipelines().
        std::shared_ptr<systems::leal::campello_gpu::RenderPipeline> pipelineFlat;
        std::shared_ptr<systems::leal::campello_gpu::RenderPipeline> pipelineTextured;
        std::shared_ptr<systems::leal::campello_gpu::RenderPipeline> pipelineDebug;  // Flat shaded, no lighting
        
        // Double-sided pipeline variants (no back-face culling).
        std::shared_ptr<systems::leal::campello_gpu::RenderPipeline> pipelineFlatDoubleSided;
        std::shared_ptr<systems::leal::campello_gpu::RenderPipeline> pipelineTexturedDoubleSided;
        
        // Alpha-blend pipeline variants (transparent materials).
        std::shared_ptr<systems::leal::campello_gpu::RenderPipeline> pipelineFlatBlend;
        std::shared_ptr<systems::leal::campello_gpu::RenderPipeline> pipelineTexturedBlend;
        std::shared_ptr<systems::leal::campello_gpu::RenderPipeline> pipelineFlatBlendDoubleSided;
        std::shared_ptr<systems::leal::campello_gpu::RenderPipeline> pipelineTexturedBlendDoubleSided;

        // Skybox pipeline — fullscreen triangle that samples an environment cubemap.
        std::shared_ptr<systems::leal::campello_gpu::RenderPipeline> pipelineSkybox;
        std::shared_ptr<systems::leal::campello_gpu::BindGroupLayout> skyboxBindGroupLayout;

        // FXAA post-process pipeline — fullscreen triangle that samples scene color texture.
        std::shared_ptr<systems::leal::campello_gpu::RenderPipeline> pipelineFxaa;
        std::shared_ptr<systems::leal::campello_gpu::BindGroupLayout> fxaaBindGroupLayout;

        // SSAA downsample pipeline — bilinear downsample from scaled scene texture.
        std::shared_ptr<systems::leal::campello_gpu::RenderPipeline> pipelineDownsample;
        std::shared_ptr<systems::leal::campello_gpu::BindGroupLayout> downsampleBindGroupLayout;

        // Environment map / IBL
        std::shared_ptr<systems::leal::campello_gpu::Texture>          environmentMap;
        std::shared_ptr<systems::leal::campello_gpu::TextureView>      environmentMapView;
        std::shared_ptr<systems::leal::campello_gpu::Sampler>          environmentSampler;
        std::shared_ptr<systems::leal::campello_gpu::BindGroup>        environmentBindGroup;
        bool skyboxEnabled = false;
        bool iblEnabled = true;
        float environmentIntensity = 1.0f;
        bool fxaaEnabled = false;
        float ssaaScale = 1.0f;

        // Active shading / inspection mode.
        ViewMode viewMode = ViewMode::normal;

        // Cached color format from createDefaultPipelines() — used to build
        // quantized pipeline variants on demand in setScene().
        systems::leal::campello_gpu::PixelFormat cachedColorFormat =
            systems::leal::campello_gpu::PixelFormat::invalid;

        // Detected component types for skinning attributes. Used to create matching
        // vertex descriptors in pipeline creation. Defaults are the most common case
        // (UNSIGNED_BYTE for JOINTS_0, FLOAT for WEIGHTS_0).
        systems::leal::campello_gpu::ComponentType jointsComponentType =
            systems::leal::campello_gpu::ComponentType::ctUnsignedByte;
        systems::leal::campello_gpu::ComponentType weightsComponentType =
            systems::leal::campello_gpu::ComponentType::ctFloat;
        bool weightsNormalized = false;

        // Pipeline variants for KHR_mesh_quantization (non-float vertex formats).
        // Key is wantedVariant (1–9). Created lazily in setScene() when quantized
        // accessors are detected.
        std::unordered_map<int, std::shared_ptr<systems::leal::campello_gpu::RenderPipeline>> quantizedPipelines;

        // Per-draw buffers: material uniforms (baseColorFactor per material slot)
        // and a fallback zero-UV buffer for primitives without TEXCOORD_0.
        std::shared_ptr<systems::leal::campello_gpu::Buffer> fallbackUVBuffer;
        std::shared_ptr<systems::leal::campello_gpu::Buffer> fallbackTangentBuffer;

        // Frame-in-flight ring buffer (3 frames) — prevents CPU overwriting GPU data.
        static constexpr uint32_t kMaxFramesInFlight = 3;
        // Per-frame uniform buffer for skybox: invVP (64) + screenSize (8) + cameraPos (12) + pad = 96 bytes.
        std::shared_ptr<systems::leal::campello_gpu::Buffer>           skyboxUniformBuffer[kMaxFramesInFlight];
        std::shared_ptr<systems::leal::campello_gpu::BindGroup>        skyboxBindGroup[kMaxFramesInFlight];

        // FXAA / SSAA resources — intermediate scene texture + per-frame bind groups.
        std::shared_ptr<systems::leal::campello_gpu::Texture>          sceneColorTexture;
        std::shared_ptr<systems::leal::campello_gpu::TextureView>      sceneColorView;
        std::shared_ptr<systems::leal::campello_gpu::Sampler>          fxaaSampler;
        std::shared_ptr<systems::leal::campello_gpu::Buffer>           fxaaUniformBuffer[kMaxFramesInFlight];
        std::shared_ptr<systems::leal::campello_gpu::BindGroup>        fxaaBindGroup[kMaxFramesInFlight];
        std::shared_ptr<systems::leal::campello_gpu::BindGroup>        downsampleBindGroup[kMaxFramesInFlight];

        // Screen-space refraction resources.
        std::shared_ptr<systems::leal::campello_gpu::Texture>          opaqueSceneTexture;
        std::shared_ptr<systems::leal::campello_gpu::TextureView>      opaqueSceneView;
        std::shared_ptr<systems::leal::campello_gpu::BindGroup>        copyBindGroup[kMaxFramesInFlight];
        struct FrameResources {
            std::shared_ptr<systems::leal::campello_gpu::Buffer> transformBuffer;
            std::shared_ptr<systems::leal::campello_gpu::Buffer> cameraPositionBuffer;
            std::shared_ptr<systems::leal::campello_gpu::Buffer> lightsUniformBuffer;
            std::shared_ptr<systems::leal::campello_gpu::Buffer> jointMatrixBuffer;
            std::shared_ptr<systems::leal::campello_gpu::Fence>  fence;
        };
        FrameResources frameResources[kMaxFramesInFlight];
        uint32_t currentFrameIndex = 0;

        // Per-frame bind groups for lights (10) and camera (18).
        // Bound at setBindGroup index 1; textures/material remain at index 0.
        std::shared_ptr<systems::leal::campello_gpu::BindGroup> frameBindGroup[kMaxFramesInFlight];

        // Aliases to the current frame's buffers — updated at the top of each render().
        // These exist so existing code referencing e.g. transformBuffer continues to work.
        std::shared_ptr<systems::leal::campello_gpu::Buffer> transformBuffer;
        std::shared_ptr<systems::leal::campello_gpu::Buffer> materialUniformBuffer;
        std::shared_ptr<systems::leal::campello_gpu::Buffer> cameraPositionBuffer;
        std::shared_ptr<systems::leal::campello_gpu::Buffer> lightsUniformBuffer;

        // EXT_mesh_gpu_instancing: per-node instance data.
        // Each node with the extension gets a GPU buffer containing per-instance
        // transform matrices (float4x4 per instance, column-major).
        struct InstanceData {
            std::shared_ptr<systems::leal::campello_gpu::Buffer> matrixBuffer;
            uint32_t instanceCount = 0;
            std::vector<systems::leal::vector_math::Matrix4<double>> cpuMatrices;
            std::vector<float> visibleMatrices;
            uint32_t visibleCount = 0;
        };
        std::unordered_map<uint64_t, InstanceData> nodeInstanceData;

        // Skeletal mesh skinning data.
        struct SkinData {
            std::vector<float> inverseBindMatrices; // float4x4 per joint, column-major for shader
            std::vector<uint64_t> jointNodeIndices; // node indices for each joint
            uint64_t jointCount = 0;
            uint64_t gpuOffset = 0; // byte offset into jointMatrixBuffer (256-byte aligned)
        };
        std::vector<SkinData> skinData;
        std::vector<int64_t> nodeSkinIndex; // nodeIndex -> skinIndex, -1 if none
        uint64_t totalJointMatrixBytes = 0;
        std::vector<float> jointMatrixData; // CPU-side flattened joint matrices for current frame

        // Fallback buffers for primitives without JOINTS_0 / WEIGHTS_0.
        std::shared_ptr<systems::leal::campello_gpu::Buffer> fallbackJointBuffer;
        std::shared_ptr<systems::leal::campello_gpu::Buffer> fallbackWeightBuffer;
        // Default identity joint matrix for non-skinned draws (bound to slot 20).
        std::shared_ptr<systems::leal::campello_gpu::Buffer> defaultJointMatrixBuffer;

        uint32_t renderWidth  = 0;
        uint32_t renderHeight = 0;

        // Depth buffer — recreated in resize().
        std::shared_ptr<systems::leal::campello_gpu::Texture>     depthTexture;
        std::shared_ptr<systems::leal::campello_gpu::TextureView> depthView;

        // Per-node matrices uploaded to the GPU at the start of each render().
        // Each node gets 32 floats: 16 for MVP (clip space), 16 for Model (world space).
        std::vector<float> nodeTransforms; // 32 floats per node

        // Lights uniform buffer — uploaded each frame with active punctual lights from KHR_lights_punctual.
        // Supports up to 4 lights (directional, point, or spot). Bound via bind group at binding 10.

        // Camera override — set by setCameraMatrices(), cleared by clearCameraOverride().
        bool hasCameraOverride = false;
        systems::leal::vector_math::Matrix4<double> overrideView;
        systems::leal::vector_math::Matrix4<double> overrideProj;

        // View-projection matrix recomputed from the active camera each frame.
        systems::leal::vector_math::Matrix4<double> vpMatrix;

        // Approximate bounding radius of the current scene (computed in setScene()).
        float boundsRadius = 1.0f;

        struct Bounds {
            bool valid = false;
            systems::leal::vector_math::Vector3<double> min;
            systems::leal::vector_math::Vector3<double> max;
        };

        // Cached local-space bounds per primitive and per-node bounds used by
        // visibility systems. World bounds are refreshed during transform updates.
        std::unordered_map<const systems::leal::gltf::Primitive *, Bounds> primitiveBounds;
        std::vector<Bounds> nodeMeshLocalBounds;
        std::vector<Bounds> nodeLocalBounds;
        std::vector<Bounds> nodeWorldBounds;
        std::vector<systems::leal::vector_math::Matrix4<double>> nodeWorldMatrices;
        std::vector<uint8_t> visibleNodeMask;

        struct Plane {
            systems::leal::vector_math::Vector3<double> normal;
            double distance = 0.0;
        };
        std::array<Plane, 6> frustumPlanes{};
        bool hasFrustumPlanes = false;

        // Tracks which pipeline variant is currently bound within a render pass
        // to avoid redundant setPipeline() calls.
        // 0 = none, 1 = flat, 2 = textured, 3 = debug, 4 = flat double-sided, 5 = textured double-sided
        int currentPipelineVariant = 0;

        // Tracks last-bound vertex buffer per slot within a render pass to avoid
        // redundant setVertexBuffer() calls (Metal debug layer asserts on these).
        struct BoundVertexBuffer {
            const systems::leal::campello_gpu::Buffer *buffer = nullptr;
            uint64_t offset = 0;
        };
        std::array<BoundVertexBuffer, 32> lastBoundVertexBuffers{};

        // Camera world position extracted each frame — used to depth-sort transparent draws.
        float cameraWorldPos[3] = {0.0f, 0.0f, 3.0f};

        // Transparent (BLEND) draws collected during opaque traversal, sorted and drawn after.
        // Internal DrawCall — carries legacy glTF fields until Phase 3.
        // Shadows the public DrawCall defined above.
        struct DrawCall {
            GpuMesh* mesh = nullptr;
            GpuMaterial* material = nullptr;
            systems::leal::vector_math::Matrix4<double> worldTransform;
            std::shared_ptr<systems::leal::campello_gpu::Buffer> jointMatrixBuffer;
            uint32_t jointCount = 0;
            uint32_t instanceCount = 1;
            // Legacy fields — will be removed in Phase 3 when renderPrimitive() is refactored.
            const systems::leal::gltf::Primitive* primitive = nullptr;
            uint64_t nodeIndex = 0;
            int64_t materialIndex = -1;
            bool transparent = false;
        };
        std::vector<DrawCall> opaqueQueue;
        std::vector<DrawCall> transparentQueue;

        // Active KHR_materials_variants index. -1 = use each primitive's default material.
        int32_t activeVariant = -1;

        // ------------------------------------------------------------------
        // Scene presentation controls
        // ------------------------------------------------------------------
        float clearColor[4] = {0.08f, 0.08f, 0.10f, 1.0f};
        bool punctualLightsEnabled = true;

        RenderStats lastFrameStats;
        bool defaultLightEnabled = true;

        // KHR_texture_procedurals — load-time baked textures.
        // Key: "graph:<index>:output:<name>"  Value: baked GPU texture.
        std::unordered_map<std::string, std::shared_ptr<systems::leal::campello_gpu::Texture>> proceduralBakedTextures;
        int proceduralBakeSize = 1024;

        // Standalone animation helper — extracted in Phase 5.
        std::unique_ptr<GltfAnimator> animator;

        // Animation sampling helpers (legacy, kept for internal use).
        void applyAnimatedTRS(uint64_t nodeIndex);

        static systems::leal::vector_math::Matrix4<double> nodeLocalMatrix(
            const systems::leal::gltf::Node &node);

        bool findCameraNode(
            uint64_t nodeIndex,
            const systems::leal::vector_math::Matrix4<double> &parentWorld,
            uint32_t camIndex,
            systems::leal::vector_math::Matrix4<double> &outWorld);

        void computeNodeTransform(
            uint64_t nodeIndex,
            const systems::leal::vector_math::Matrix4<double> &parentWorld);

        void computeSceneBounds(
            uint64_t nodeIndex,
            const systems::leal::vector_math::Matrix4<double> &parentWorld);

        void gatherVisibleDraws(
            uint64_t nodeIndex);

        void renderPrimitive(
            const std::shared_ptr<systems::leal::campello_gpu::RenderPassEncoder> &rpe,
            const systems::leal::gltf::Primitive &primitive,
            uint64_t nodeIndex);

        // New ECS-path primitive renderer.
        void renderPrimitive(
            const std::shared_ptr<systems::leal::campello_gpu::RenderPassEncoder> &rpe,
            const systems::leal::campello_renderer::DrawCall& draw,
            uint64_t transformOffset);

        void setVertexBufferIfChanged(
            const std::shared_ptr<systems::leal::campello_gpu::RenderPassEncoder> &rpe,
            uint32_t slot,
            const std::shared_ptr<systems::leal::campello_gpu::Buffer> &buffer,
            uint64_t offset);

        int64_t resolvePrimitiveMaterial(const systems::leal::gltf::Primitive &primitive) const;
        bool isTransparentMaterial(int64_t materialIndex) const;
        Bounds computePrimitiveBounds(const systems::leal::gltf::Primitive &primitive) const;
        Bounds transformBounds(const Bounds &bounds,
                               const systems::leal::vector_math::Matrix4<double> &world) const;
        static Bounds mergeBounds(const Bounds &a, const Bounds &b);
        void updateFrustumPlanes();
        bool isBoundsVisible(const Bounds &bounds) const;
        void uploadVisibleNodeTransforms();
        void updateVisibleInstances(uint64_t nodeIndex);

        // Skeletal mesh skinning.
        void computeSkinningTransforms();
        void uploadJointMatrices();

        // Core render implementation — shared by both render() overloads.
        void renderToTarget(
            std::shared_ptr<systems::leal::campello_gpu::TextureView> colorView);

        // Upload Draco-decompressed primitive data to GPU buffers.
        void uploadDracoBuffers(std::shared_ptr<systems::leal::gltf::RuntimeInfo> &info);

    public:
        Renderer(std::shared_ptr<systems::leal::campello_gpu::Device> device);

        void setAsset(std::shared_ptr<systems::leal::gltf::GLTF> asset);
        std::shared_ptr<systems::leal::gltf::GLTF> getAsset();

        // Sets the base directory for resolving relative external image URIs
        // in .gltf files (e.g. "textures/baseColor.png"). Must be called before
        // setScene() or uploadMaterial() for external textures to load.
        void setAssetBasePath(const std::string& path);
        std::string getAssetBasePath() const;

        // Upload a single primitive's geometry and return an engine-native handle.
        GpuMesh* uploadMesh(const systems::leal::gltf::Primitive& primitive,
                            const systems::leal::gltf::GLTF& asset);

        // Upload a material's textures/uniforms and return an engine-native handle.
        GpuMaterial* uploadMaterial(const systems::leal::gltf::Material& material,
                                    const systems::leal::gltf::GLTF& asset);

        // Re-upload a single material slot to the GPU material uniform buffer.
        // Used by KHR_animation_pointer to sync animated material properties.
        void reuploadMaterialSlot(uint32_t uniformSlot,
                                  const systems::leal::gltf::Material& material,
                                  const systems::leal::gltf::GLTF& asset);

        void setCamera(uint32_t index);
        void setScene(uint32_t index);

        // Notify the renderer of the current swapchain dimensions.
        // Must be called once after creation and whenever the surface resizes.
        void resize(uint32_t width, uint32_t height);

        // Creates all built-in shader pipeline variants for the given color format.
        // Must be called once before render(). Replaces createDefaultPipeline().
        void createDefaultPipelines(systems::leal::campello_gpu::PixelFormat colorFormat);

        // Creates quantized pipeline variants when KHR_mesh_quantization accessors
        // are detected in setScene(). Called lazily — only when needed.
        void createQuantizedPipelinesIfNeeded();

        // Render to the device's swapchain (Android / platforms where the device
        // owns the surface).
        void render();

        // Render to an externally provided color target (e.g., an MTKView drawable
        // texture on macOS). The renderer's own depth buffer (from resize()) is used.
        void render(std::shared_ptr<systems::leal::campello_gpu::TextureView> colorView);

        // Render a fully-prepared scene description (ECS-driven path).
        void render(const RenderScene& scene,
                    std::shared_ptr<systems::leal::campello_gpu::TextureView> colorView);

        void update(double dt);

        // Override the camera used by render(). Both matrices are column-major
        // float[16] (the standard layout for simd_float4x4 and most graphics APIs).
        // Overrides the GLTF camera until clearCameraOverride() is called.
        void setCameraMatrices(const float *viewColMajor16, const float *projColMajor16);
        void clearCameraOverride();

        // Returns the approximate bounding radius of the current scene,
        // computed from node world-space positions in setScene().
        float getBoundsRadius() const;

        void setViewMode(ViewMode mode);
        ViewMode getViewMode() const;

        // Compatibility wrapper for the old debug toggle.
        // `true` maps to `ViewMode::worldNormal`, `false` maps to `ViewMode::normal`.
        void setDebugMode(bool enabled);
        bool isDebugModeEnabled() const;

        // KHR_materials_variants — switch the active named material variant.
        // variantIndex selects an entry from asset->khrMaterialsVariants.
        // Pass -1 to restore each primitive's default material (the asset default).
        void setMaterialVariant(int32_t variantIndex);
        uint32_t getMaterialVariantCount() const;
        std::string getMaterialVariantName(uint32_t variantIndex) const;

        // Animation control — multi-animation support.
        // Multiple animations can play simultaneously on different node sets.
        uint32_t getAnimationCount() const;
        std::string getAnimationName(uint32_t animationIndex) const;
        double getAnimationDuration(uint32_t animationIndex) const;

        // Per-animation playback control.
        void playAnimation(uint32_t animationIndex);            // Start/resume specific animation
        void pauseAnimation(uint32_t animationIndex);           // Pause specific animation
        void stopAnimation(uint32_t animationIndex);            // Stop and reset specific animation to time 0
        void stopAllAnimations();                               // Stop all animations

        // Per-animation state queries.
        bool isAnimationPlaying(uint32_t animationIndex) const; // Get playback state
        void setAnimationLoop(uint32_t animationIndex, bool loop);  // Set loop mode
        bool isAnimationLooping(uint32_t animationIndex) const;   // Get loop mode

        // Per-animation time control.
        void setAnimationTime(uint32_t animationIndex, double time);  // Seek to specific time
        double getAnimationTime(uint32_t animationIndex) const;       // Get current time

        // Scene presentation controls (clear color, lighting layers).
        void setClearColor(float r, float g, float b, float a);
        void getClearColor(float *outR, float *outG, float *outB, float *outA) const;

        void setPunctualLightsEnabled(bool enabled);
        bool isPunctualLightsEnabled() const;

        void setDefaultLightEnabled(bool enabled);
        bool isDefaultLightEnabled() const;

        // Environment map / IBL controls.
        void setEnvironmentMap(std::shared_ptr<systems::leal::campello_gpu::Texture> cubemap);

        // Load 6 face images (+X, -X, +Y, -Y, +Z, -Z) into a cube texture.
        // Supports PNG, JPEG, WebP, HDR, EXR. Returns nullptr on failure.
        std::shared_ptr<systems::leal::campello_gpu::Texture> loadEnvironmentMap(
            const std::string &px, const std::string &nx,
            const std::string &py, const std::string &ny,
            const std::string &pz, const std::string &nz);

        // Load a single equirectangular image (2:1 aspect ratio) and convert
        // it to a cubemap on the CPU. Supports PNG, JPEG, HDR, EXR.
        // faceSize is the desired size of each cubemap face; 0 means auto
        // (half the equirectangular height). Returns nullptr on failure.
        std::shared_ptr<systems::leal::campello_gpu::Texture> loadEquirectangularEnvironmentMap(
            const std::string &path, uint32_t faceSize = 0);

        void setSkyboxEnabled(bool enabled);
        void setIBLEnabled(bool enabled);
        void setEnvironmentIntensity(float intensity);
        bool isSkyboxEnabled() const;
        bool isIBLEnabled() const;
        float getEnvironmentIntensity() const;

        void setFxaaEnabled(bool enabled);
        bool isFxaaEnabled() const;

        void setSsaaScale(float scale);
        float getSsaaScale() const;

        // KHR_texture_procedurals bake controls.
        // Sets the resolution for load-time procedural texture baking.
        // Default is 1024. Must be called before setScene() to take effect.
        void setProceduralBakeSize(int size);
        int getProceduralBakeSize() const;

        std::shared_ptr<systems::leal::campello_gpu::Device> getDevice() const { return device; }

        // Returns statistics from the last completed frame.
        RenderStats getLastFrameStats() const;

    private:
        void ensureSceneColorTexture();

        // Accessors for uploaded GPU resources.
        std::shared_ptr<systems::leal::campello_gpu::Buffer>  getGpuBuffer(uint32_t index) const;
        std::shared_ptr<systems::leal::campello_gpu::Texture> getGpuTexture(uint32_t index) const;
        uint32_t getGpuBufferCount()  const;
        uint32_t getGpuTextureCount() const;

        std::shared_ptr<systems::leal::campello_gpu::BindGroup> getBindGroup(uint32_t index) const;
        uint32_t getBindGroupCount() const;
        std::shared_ptr<systems::leal::campello_gpu::BindGroup> getDefaultBindGroup() const;
    };

    std::string getVersion();

}
