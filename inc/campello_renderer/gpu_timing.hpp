#pragma once

// GpuTiming: the GPU execution time of an effect's last finished work
// (campello_gpu CommandBuffer::getGPUExecutionTime()), read without
// waiting: work still in flight is counted once its fence has signalled.
// The effects (RayTracedShadows, RestirDirectLighting, RealtimeDenoiser,
// DdgiVolume, ...) keep one and report it as gpuTimeMs().

#include <campello_gpu/command_buffer.hpp>
#include <campello_gpu/fence.hpp>

#include <memory>
#include <vector>

namespace systems::leal::campello_renderer {

class GpuTiming {
public:
    /// A call's command buffers start: the next submitted() ones are its.
    void begin() {
        collect();
        call_.clear();
    }
    /// One of the current call's command buffers, submitted with `fence`.
    void submitted(std::shared_ptr<systems::leal::campello_gpu::CommandBuffer> commands,
                   std::shared_ptr<systems::leal::campello_gpu::Fence> fence) {
        if (commands && fence) call_.push_back({std::move(commands), std::move(fence)});
    }
    /// The last finished call's GPU time (ms; all its command buffers); 0
    /// before one finishes or where the backend doesn't measure.
    [[nodiscard]] double ms() const {
        collect();
        return ms_;
    }

private:
    struct Entry {
        std::shared_ptr<systems::leal::campello_gpu::CommandBuffer> commands;
        std::shared_ptr<systems::leal::campello_gpu::Fence> fence;
    };
    void collect() const {
        if (call_.empty()) return;
        for (const auto& e : call_) {
            if (!e.fence->isSignaled()) return;  // not finished: keep the last value
        }
        double total = 0.0;
        for (const auto& e : call_) total += static_cast<double>(e.commands->getGPUExecutionTime()) * 1e-6;
        ms_ = total;
        call_.clear();
    }
    mutable std::vector<Entry> call_;
    mutable double ms_ = 0.0;
};

} // namespace systems::leal::campello_renderer
