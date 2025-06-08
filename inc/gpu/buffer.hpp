#pragma once

namespace systems::leal::gpu
{

    class Device;

    class Buffer
    {
        friend class Device;
        void *native;

        Buffer(void *pd);

        uint64_t getLength();
        void upload(uint64_t offset, uint64_t length, void *data);

    public:
        ~Buffer();
    };

}