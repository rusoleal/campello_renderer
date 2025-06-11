#pragma once

namespace systems::leal::gpu
{

    enum class TextureUsage
    {

        /**
         * (WebGPU)
         * The texture can be used as the source of a copy operation, for example the source
         * argument of a copyTextureToBuffer() call.
         */
        copySrc = 0x01,

        /**
         * (WebGPU)
         * The texture can be used as the destination of a copy/write operation, for example the
         * destination argument of a copyBufferToTexture() call.
         */
        copyDst = 0x02,

        /**
         * The texture can be readed in a shader.
         */
        shaderRead = 0x04,

        /**
         * The texture can be written in a shader.
         */
        shaderWrite = 0x08,

        /**
         * (WebGPU)
         * The texture can be used as a color or depth/stencil attachment in a render pass,
         * for example as the view property of the descriptor object in a beginRenderPass() call.
         */
        renderTarget = 0xA0,
    };

}