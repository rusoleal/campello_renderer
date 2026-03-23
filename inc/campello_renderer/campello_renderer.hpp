#pragma once

#include <memory>
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
        static constexpr uint32_t VERTEX_SLOT_POSITION  = 0;  // float3 POSITION
        static constexpr uint32_t VERTEX_SLOT_NORMAL    = 1;  // float3 NORMAL
        static constexpr uint32_t VERTEX_SLOT_TEXCOORD0 = 2;  // float2 TEXCOORD_0
        static constexpr uint32_t VERTEX_SLOT_TANGENT   = 3;  // float4 TANGENT
        static constexpr uint32_t VERTEX_SLOT_MVP       = 16; // float4x4, row-major MVP matrix
        static constexpr uint32_t VERTEX_SLOT_MATERIAL  = 17; // MaterialUniforms (baseColorFactor)

    private:
        std::shared_ptr<systems::leal::campello_gpu::Device> device;
        std::shared_ptr<systems::leal::gltf::GLTF> asset;
        uint32_t sceneIndex  = 0;
        uint32_t cameraIndex = 0;
        std::vector<std::shared_ptr<Image>> images;

        std::vector<std::shared_ptr<systems::leal::campello_gpu::Buffer>>  gpuBuffers;
        std::vector<std::shared_ptr<systems::leal::campello_gpu::Texture>> gpuTextures;

        // Per-GLTF-sampler GPU samplers and per-GLTF-texture bind groups.
        std::vector<std::shared_ptr<systems::leal::campello_gpu::Sampler>>        gpuSamplers;
        std::vector<std::shared_ptr<systems::leal::campello_gpu::BindGroup>>      gpuBindGroups;

        // Shared resources created on first setScene() call.
        std::shared_ptr<systems::leal::campello_gpu::Sampler>          defaultSampler;
        std::shared_ptr<systems::leal::campello_gpu::Texture>          defaultTexture;
        std::shared_ptr<systems::leal::campello_gpu::BindGroupLayout>  bindGroupLayout;
        std::shared_ptr<systems::leal::campello_gpu::BindGroup>        defaultBindGroup;

        // Built-in pipeline variants created by createDefaultPipelines().
        std::shared_ptr<systems::leal::campello_gpu::RenderPipeline> pipelineFlat;
        std::shared_ptr<systems::leal::campello_gpu::RenderPipeline> pipelineTextured;

        // Per-draw buffers: material uniforms (baseColorFactor per material slot)
        // and a fallback zero-UV buffer for primitives without TEXCOORD_0.
        std::shared_ptr<systems::leal::campello_gpu::Buffer> materialUniformBuffer;
        std::shared_ptr<systems::leal::campello_gpu::Buffer> fallbackUVBuffer;

        uint32_t renderWidth  = 0;
        uint32_t renderHeight = 0;

        // Depth buffer — recreated in resize().
        std::shared_ptr<systems::leal::campello_gpu::Texture>     depthTexture;
        std::shared_ptr<systems::leal::campello_gpu::TextureView> depthView;

        // Per-node MVP matrices uploaded to the GPU at the start of each render().
        std::shared_ptr<systems::leal::campello_gpu::Buffer> transformBuffer;
        std::vector<float> nodeTransforms;

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
        // 0 = none, 1 = flat, 2 = textured
        int currentPipelineVariant = 0;

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
