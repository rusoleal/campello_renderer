#include "Metal.hpp"
#include <gpu/buffer.hpp>

using namespace systems::leal::gpu;

Buffer::Buffer(void *pd) {
    native = pd;
}

Buffer::~Buffer() {
    if (native != nullptr) {
        ((MTL::Buffer *)native)->release();
    }
}

uint64_t Buffer::getLength() {
    return ((MTL::Buffer *)native)->length();
}

void Buffer::upload(uint64_t offset, uint64_t size, void *data) {

    uint8_t *dst = (uint8_t *)((MTL::Buffer *)native)->contents();
    memcpy(dst+offset,data, size);
    
    ((MTL::Buffer *)native)->didModifyRange(NS::Range::Make(offset, size));
}


