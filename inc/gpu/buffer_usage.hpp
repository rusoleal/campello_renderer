#pragma once

namespace systems::leal::gpu
{

    enum class BufferUsage
    {

        copySrc = 0x0004,

        copyDst = 0x0008,

        index = 0x0010,

        indirect = 0x0100,

        mapRead = 0x0001,

        mapWrite = 0x0002,

        queryResolve = 0x0200,

        storage = 0x0080,

        uniform = 0x0040,

        vertex = 0x0020

    };

}