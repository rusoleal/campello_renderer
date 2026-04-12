#pragma once

#include <memory>
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

namespace systems::leal::campello_renderer {

    class Renderer {
    public:
        // Vertex buffer slot convention — must match the pipeline's vertex descriptor
        // and the shader bindings.
        static constexpr uint32_t VERTEX_SLOT_POSITION   = 0;  // float3 POSITION
        static constexpr uint32_t VERTEX_SLOT_NORMAL     = 1;  // float3 NORMAL
        static constexpr uint32_t VERTEX_SLOT_TEXCOORD0  = 2;  // float2 TEXCOORD_0
        static constexpr uint32_t VERTEX_SLOT_TANGENT    = 3;  // float4 TANGENT
        static constexpr uint32_t VERTEX_SLOT_MVP        = 16; // float4x4, row-major MVP matrix (per-node)
        static constexpr uint32_t VERTEX_SLOT_MATERIAL   = 17; // MaterialUniforms (112 bytes, 256 stride)
        static constexpr uint32_t VERTEX_SLOT_CAMERA     = 18; // CameraPosition (float3 world space)
        static constexpr uint32_t VERTEX_SLOT_LIGHTS     = 15; // LightsUniforms - use slot 15 to avoid conflict
        static constexpr uint32_t VERTEX_SLOT_INSTANCE_MATRIX = 19; // float4x4 per-instance transform (EXT_mesh_gpu_instancing)
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
        uint32_t sceneIndex  = 0;
        uint32_t cameraIndex = 0;
        std::vector<std::shared_ptr<Image>> images;

        std::vector<std::shared_ptr<systems::leal::campello_gpu::Buffer>>  gpuBuffers;
        std::vector<std::shared_ptr<systems::leal::campello_gpu::Texture>> gpuTextures;

        // Per-GLTF-sampler GPU samplers and per-material bind groups.
        std::vector<std::shared_ptr<systems::leal::campello_gpu::Sampler>>        gpuSamplers;
        std::vector<std::shared_ptr<systems::leal::campello_gpu::BindGroup>>      materialBindGroups;

        // Draco-compressed primitive GPU buffers (primitive pointer → index buffer + attribute buffers).
        // Populated in setScene() after calling GLTF::decompressDraco().
        struct DracoBuffers {
            std::shared_ptr<systems::leal::campello_gpu::Buffer> indexBuffer;
            std::unordered_map<std::string, std::shared_ptr<systems::leal::campello_gpu::Buffer>> attributeBuffers;
            uint32_t indexCount = 0;  // Number of indices for drawing
        };
        std::unordered_map<const systems::leal::gltf::Primitive *, DracoBuffers> dracoPrimitiveBuffers;

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

        // Debug rendering mode.
        bool debugModeEnabled = false;

        // Per-draw buffers: material uniforms (baseColorFactor per material slot)
        // and a fallback zero-UV buffer for primitives without TEXCOORD_0.
        std::shared_ptr<systems::leal::campello_gpu::Buffer> materialUniformBuffer;
        std::shared_ptr<systems::leal::campello_gpu::Buffer> fallbackUVBuffer;

        // EXT_mesh_gpu_instancing: per-node instance data.
        // Each node with the extension gets a GPU buffer containing per-instance
        // transform matrices (float4x4 per instance, column-major).
        struct InstanceData {
            std::shared_ptr<systems::leal::campello_gpu::Buffer> matrixBuffer;
            uint32_t instanceCount = 0;
        };
        std::unordered_map<uint64_t, InstanceData> nodeInstanceData;

        uint32_t renderWidth  = 0;
        uint32_t renderHeight = 0;

        // Depth buffer — recreated in resize().
        std::shared_ptr<systems::leal::campello_gpu::Texture>     depthTexture;
        std::shared_ptr<systems::leal::campello_gpu::TextureView> depthView;

        // Per-node matrices uploaded to the GPU at the start of each render().
        // Each node gets 32 floats: 16 for MVP (clip space), 16 for Model (world space).
        std::shared_ptr<systems::leal::campello_gpu::Buffer> transformBuffer;
        std::vector<float> nodeTransforms; // 32 floats per node

        // Camera position buffer — uploaded each frame for specular lighting.
        // Contains float3 cameraPositionWorld.
        std::shared_ptr<systems::leal::campello_gpu::Buffer> cameraPositionBuffer;
        
        // Lights uniform buffer — uploaded each frame with active punctual lights from KHR_lights_punctual.
        // Supports up to 4 lights (directional, point, or spot). Bound via bind group at binding 10.
        std::shared_ptr<systems::leal::campello_gpu::Buffer> lightsUniformBuffer;

        // Camera override — set by setCameraMatrices(), cleared by clearCameraOverride().
        bool hasCameraOverride = false;
        systems::leal::vector_math::Matrix4<double> overrideView;
        systems::leal::vector_math::Matrix4<double> overrideProj;

        // View-projection matrix recomputed from the active camera each frame.
        systems::leal::vector_math::Matrix4<double> vpMatrix;

        // Approximate bounding radius of the current scene (computed in setScene()).
        float boundsRadius = 1.0f;

        // Tracks which pipeline variant is currently bound within a render pass
        // to avoid redundant setPipeline() calls.
        // 0 = none, 1 = flat, 2 = textured, 3 = debug, 4 = flat double-sided, 5 = textured double-sided
        int currentPipelineVariant = 0;

        // Camera world position extracted each frame — used to depth-sort transparent draws.
        float cameraWorldPos[3] = {0.0f, 0.0f, 3.0f};

        // Transparent (BLEND) draws collected during opaque traversal, sorted and drawn after.
        struct TransparentDraw {
            const systems::leal::gltf::Primitive *primitive;
            uint64_t nodeIndex;
        };
        std::vector<TransparentDraw> transparentQueue;

        // Active KHR_materials_variants index. -1 = use each primitive's default material.
        int32_t activeVariant = -1;

        // Per-animation state for multi-animation support.
        struct AnimationState {
            double time = 0.0;           // Current animation time in seconds
            bool playing = false;        // true = playing, false = paused
            bool loop = true;            // true = loop, false = stop at end
            double duration = 0.0;       // Duration from keyframe data
        };
        std::unordered_map<int32_t, AnimationState> animationStates;

        // Animated node transforms — merged from all playing animations.
        // Maps node index to animated TRS values.
        struct AnimatedTRS {
            bool hasTranslation = false;
            bool hasRotation = false;
            bool hasScale = false;
            systems::leal::vector_math::Vector3<double> translation;
            systems::leal::vector_math::Quaternion<double> rotation;
            systems::leal::vector_math::Vector3<double> scale;
        };
        std::unordered_map<uint64_t, AnimatedTRS> animatedNodes;

        // Animation sampling helpers.
        void sampleAnimation(int32_t animIndex, float time);
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

        void renderNode(
            const std::shared_ptr<systems::leal::campello_gpu::RenderPassEncoder> &rpe,
            uint64_t nodeIndex);

        void renderPrimitive(
            const std::shared_ptr<systems::leal::campello_gpu::RenderPassEncoder> &rpe,
            const systems::leal::gltf::Primitive &primitive,
            uint64_t nodeIndex);

        // Core render implementation — shared by both render() overloads.
        void renderToTarget(
            std::shared_ptr<systems::leal::campello_gpu::TextureView> colorView);

        // Upload Draco-decompressed primitive data to GPU buffers.
        void uploadDracoBuffers(std::shared_ptr<systems::leal::gltf::RuntimeInfo> &info);

    public:
        Renderer(std::shared_ptr<systems::leal::campello_gpu::Device> device);

        void setAsset(std::shared_ptr<systems::leal::gltf::GLTF> asset);
        std::shared_ptr<systems::leal::gltf::GLTF> getAsset();

        void setCamera(uint32_t index);
        void setScene(uint32_t index);

        // Notify the renderer of the current swapchain dimensions.
        // Must be called once after creation and whenever the surface resizes.
        void resize(uint32_t width, uint32_t height);

        // Creates all built-in shader pipeline variants for the given color format.
        // Must be called once before render(). Replaces createDefaultPipeline().
        void createDefaultPipelines(systems::leal::campello_gpu::PixelFormat colorFormat);

        // Render to the device's swapchain (Android / platforms where the device
        // owns the surface).
        void render();

        // Render to an externally provided color target (e.g., an MTKView drawable
        // texture on macOS). The renderer's own depth buffer (from resize()) is used.
        void render(std::shared_ptr<systems::leal::campello_gpu::TextureView> colorView);

        void update(double dt);

        // Override the camera used by render(). Both matrices are column-major
        // float[16] (the standard layout for simd_float4x4 and most graphics APIs).
        // Overrides the GLTF camera until clearCameraOverride() is called.
        void setCameraMatrices(const float *viewColMajor16, const float *projColMajor16);
        void clearCameraOverride();

        // Returns the approximate bounding radius of the current scene,
        // computed from node world-space positions in setScene().
        float getBoundsRadius() const;

        // Debug rendering mode.
        // When enabled, renders geometry with flat shading (no textures, no lighting)
        // to verify geometry loading and camera setup.
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
