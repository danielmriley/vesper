#include <vesper/ops/random.h>
#include <random>
#include <stdexcept>
#include <mutex>
#include <optional>

namespace vesper::ops {

// ============================================================================
// Global RNG State Management
// ============================================================================

namespace {
    // Thread-safe global RNG state
    std::mutex g_rng_mutex;
    std::optional<uint64_t> g_manual_seed_value;
    std::mt19937 g_rng;
    bool g_rng_initialized = false;
}

// Internal functions to access global RNG (for sampling_ops.cpp)
// NOTE: Caller must hold the mutex when using the returned reference
std::mt19937& get_global_rng() {
    // Initialize RNG on first use (caller must already hold the lock)
    if (!g_rng_initialized) {
        if (g_manual_seed_value.has_value()) {
            g_rng.seed(static_cast<std::mt19937::result_type>(g_manual_seed_value.value()));
        } else {
            std::random_device rd;
            g_rng.seed(rd());
        }
        g_rng_initialized = true;
    }
    return g_rng;
}

std::mutex& get_global_rng_mutex() {
    return g_rng_mutex;
}

namespace {
    std::mt19937& get_rng() {
        // For internal use - caller does NOT hold the lock yet
        std::lock_guard<std::mutex> lock(g_rng_mutex);
        return get_global_rng();
    }
    
    // Use this for generation - takes the lock and initializes if needed
    std::mt19937& get_rng_locked() {
        return get_global_rng();
    }
}

void manual_seed(uint64_t seed) {
    std::lock_guard<std::mutex> lock(g_rng_mutex);
    g_manual_seed_value = seed;
    g_rng.seed(static_cast<std::mt19937::result_type>(seed));
    g_rng_initialized = true;
}

void uniform_cpu_dispatch(Tensor& tensor, float min, float max) {
    if (tensor.dtype() != DType::Float32) {
        throw std::runtime_error("uniform_ only supports Float32");
    }

    std::mt19937& rng = get_rng();
    std::uniform_real_distribution<float> dist(min, max);
    
    float* data = tensor.data_ptr<float>();
    size_t n = tensor.numel();
    
    // Lock RNG during generation for thread safety
    std::lock_guard<std::mutex> lock(g_rng_mutex);
    for (size_t i = 0; i < n; ++i) {
        data[i] = dist(rng);
    }
}

void normal_cpu_dispatch(Tensor& tensor, float mean, float std) {
    if (tensor.dtype() != DType::Float32) {
        throw std::runtime_error("normal_ only supports Float32");
    }

    std::mt19937& rng = get_rng();
    std::normal_distribution<float> dist(mean, std);
    
    float* data = tensor.data_ptr<float>();
    size_t n = tensor.numel();
    
    // Lock RNG during generation for thread safety
    std::lock_guard<std::mutex> lock(g_rng_mutex);
    for (size_t i = 0; i < n; ++i) {
        data[i] = dist(rng);
    }
}

void uniform_(Tensor& tensor, float min, float max) {
    if (tensor.device() == Device::CPU) {
        uniform_cpu_dispatch(tensor, min, max);
    } else if (tensor.device() == Device::HIP) {
#if USE_HIP_BACKEND
        uniform_hip_dispatch(tensor, min, max);
#else
        throw std::runtime_error("HIP backend not enabled.");
#endif
    } else if (tensor.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        uniform_cuda_dispatch(tensor, min, max);
#else
        throw std::runtime_error("CUDA backend not enabled.");
#endif
    } else {
        throw std::runtime_error("Device not supported for uniform_");
    }
}

void normal_(Tensor& tensor, float mean, float std) {
    if (tensor.device() == Device::CPU) {
        normal_cpu_dispatch(tensor, mean, std);
    } else if (tensor.device() == Device::HIP) {
#if USE_HIP_BACKEND
        normal_hip_dispatch(tensor, mean, std);
#else
        throw std::runtime_error("HIP backend not enabled.");
#endif
    } else if (tensor.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        normal_cuda_dispatch(tensor, mean, std);
#else
        throw std::runtime_error("CUDA backend not enabled.");
#endif
    } else {
        throw std::runtime_error("Device not supported for normal_");
    }
}

} // namespace vesper::ops
