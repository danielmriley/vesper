#include <vesper/core/stream.h>
#include <stdexcept>

#if USE_HIP_BACKEND
#include <hip/hip_runtime.h>
#endif

#if USE_CUDA_BACKEND
#include <cuda_runtime.h>
#endif

namespace vesper {

Stream::Stream(Device device) : device_(device), own_handle_(true) {
    if (device == Device::CPU) {
        handle_ = nullptr; // CPU streams not really implemented, effectively synchronous
    } else if (device == Device::HIP) {
#if USE_HIP_BACKEND
        hipStream_t stream;
        if (hipStreamCreate(&stream) != hipSuccess) {
            throw std::runtime_error("Failed to create HIP stream.");
        }
        handle_ = static_cast<void*>(stream);
#else
        throw std::runtime_error("HIP backend not enabled.");
#endif
    } else if (device == Device::CUDA) {
#if USE_CUDA_BACKEND
        cudaStream_t stream;
        if (cudaStreamCreate(&stream) != cudaSuccess) {
            throw std::runtime_error("Failed to create CUDA stream.");
        }
        handle_ = static_cast<void*>(stream);
#else
        throw std::runtime_error("CUDA backend not enabled.");
#endif
    }
}

Stream::Stream(Device device, void* handle, bool own) 
    : device_(device), handle_(handle), own_handle_(own) {}

Stream::~Stream() {
    if (own_handle_ && handle_) {
        if (device_ == Device::HIP) {
#if USE_HIP_BACKEND
            hipStreamDestroy(static_cast<hipStream_t>(handle_));
#endif
        } else if (device_ == Device::CUDA) {
#if USE_CUDA_BACKEND
            cudaStreamDestroy(static_cast<cudaStream_t>(handle_));
#endif
        }
    }
}

Stream::Stream(Stream&& other) noexcept
    : device_(other.device_), handle_(other.handle_), own_handle_(other.own_handle_) {
    other.handle_ = nullptr;
    other.own_handle_ = false;
}

Stream& Stream::operator=(Stream&& other) noexcept {
    if (this != &other) {
        if (own_handle_ && handle_) {
            // Clean up current
             if (device_ == Device::HIP) {
#if USE_HIP_BACKEND
                hipStreamDestroy(static_cast<hipStream_t>(handle_));
#endif
            } else if (device_ == Device::CUDA) {
#if USE_CUDA_BACKEND
                cudaStreamDestroy(static_cast<cudaStream_t>(handle_));
#endif
            }
        }
        device_ = other.device_;
        handle_ = other.handle_;
        own_handle_ = other.own_handle_;
        other.handle_ = nullptr;
        other.own_handle_ = false;
    }
    return *this;
}

void Stream::synchronize() const {
    if (device_ == Device::CPU) return;
    
    if (device_ == Device::HIP) {
#if USE_HIP_BACKEND
        hipStreamSynchronize(static_cast<hipStream_t>(handle_));
#endif
    } else if (device_ == Device::CUDA) {
#if USE_CUDA_BACKEND
        cudaStreamSynchronize(static_cast<cudaStream_t>(handle_));
#endif
    }
}

void* Stream::raw_handle() const {
    return handle_;
}

Stream& Stream::default_stream(Device device) {
    static Stream cpu_default(Device::CPU, nullptr, false);
#if USE_HIP_BACKEND
    static Stream hip_default(Device::HIP, nullptr, false); // nullptr is default stream (0)
#endif
#if USE_CUDA_BACKEND
    static Stream cuda_default(Device::CUDA, nullptr, false);
#endif

    switch(device) {
        case Device::CPU: return cpu_default;
        case Device::HIP:
#if USE_HIP_BACKEND
            return hip_default;
#else
            throw std::runtime_error("HIP backend not enabled.");
#endif
        case Device::CUDA:
#if USE_CUDA_BACKEND
            return cuda_default;
#else
            throw std::runtime_error("CUDA backend not enabled.");
#endif
        default: throw std::runtime_error("Unknown device.");
    }
}

// Thread-local current streams
static thread_local Stream* current_cpu_stream = nullptr;
static thread_local Stream* current_hip_stream = nullptr;
static thread_local Stream* current_cuda_stream = nullptr;

void Stream::set_current(const Stream& stream) {
    // We store a pointer to the Stream object.
    // WARNING: Caller must ensure Stream object stays alive.
    // Ideally we would store raw handle or shared_ptr, but Stream is not copyable.
    // Storing raw handle in a wrapper might be safer if we had a StreamHandle type.
    // For now, we trust the user (or RAII guard scope).
    
    // But wait, Stream::current returns Stream&.
    // If we store Stream*, we can return *Stream.
    // But we can't easily construct a Stream from raw handle without making a new object.
    // Let's change design: Stream::current returns reference to thread-local wrapper?
    // No, just return the one set.
    
    switch(stream.device()) {
        case Device::CPU: current_cpu_stream = const_cast<Stream*>(&stream); break;
        case Device::HIP: current_hip_stream = const_cast<Stream*>(&stream); break;
        case Device::CUDA: current_cuda_stream = const_cast<Stream*>(&stream); break;
    }
}

Stream& Stream::current(Device device) {
    switch(device) {
        case Device::CPU: 
            return current_cpu_stream ? *current_cpu_stream : default_stream(Device::CPU);
        case Device::HIP: 
            return current_hip_stream ? *current_hip_stream : default_stream(Device::HIP);
        case Device::CUDA: 
            return current_cuda_stream ? *current_cuda_stream : default_stream(Device::CUDA);
        default: throw std::runtime_error("Unknown device.");
    }
}

} // namespace vesper
