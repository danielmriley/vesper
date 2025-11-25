#include <vesper/ops/random.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace vesper::ops {

// A simple hash-based PRNG to generate a random float from an index
// Based on PCG-like hashing or SplitMix
__device__ float random_float(unsigned int idx, unsigned int seed, float min, float max) {
    // Simple hash function
    unsigned int state = idx * 747796405u + seed * 2891336453u;
    unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    unsigned int result = (word >> 22u) ^ word;
    
    // Normalize to [0, 1]
    // 0xFFFFFFFF is max uint (4294967295)
    float r = static_cast<float>(result) / 4294967295.0f;
    
    return min + r * (max - min);
}

__global__ void uniform_kernel(float* data, size_t n, float min, float max, unsigned int seed) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = random_float(idx, seed, min, max);
    }
}

void uniform_cuda_dispatch(Tensor& tensor, float min, float max) {
    if (tensor.dtype() != DType::Float32) {
        throw std::runtime_error("uniform_ only supports Float32");
    }
    
    const int threads = 256;
    const int blocks = (tensor.numel() + threads - 1) / threads;
    
    // Simple seed generation
    // In a real system, we might want to pass a generator state or use a better seed source
    static unsigned int seed_counter = 0;
    seed_counter++;
    unsigned int seed = 123456789 + seed_counter; 

    uniform_kernel<<<dim3(blocks), dim3(threads), 0, 0>>>(
        tensor.data_ptr<float>(), tensor.numel(), min, max, seed);
}

__global__ void normal_kernel(float* data, size_t n, float mean, float std, unsigned int seed1, unsigned int seed2) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float u1 = random_float(idx, seed1, 1e-7f, 1.0f);
        float u2 = random_float(idx, seed2, 0.0f, 1.0f);
        
        // Box-Muller transform
        float r = sqrtf(-2.0f * logf(u1));
        float theta = 2.0f * 3.14159265359f * u2;
        float z = r * cosf(theta);
        
        data[idx] = mean + z * std;
    }
}

void normal_cuda_dispatch(Tensor& tensor, float mean, float std) {
    if (tensor.dtype() != DType::Float32) {
        throw std::runtime_error("normal_ only supports Float32");
    }
    
    const int threads = 256;
    const int blocks = (tensor.numel() + threads - 1) / threads;
    
    static unsigned int seed_counter = 0;
    seed_counter += 2;
    unsigned int seed1 = 123456789 + seed_counter; 
    unsigned int seed2 = 987654321 + seed_counter;

    normal_kernel<<<dim3(blocks), dim3(threads), 0, 0>>>(
        tensor.data_ptr<float>(), tensor.numel(), mean, std, seed1, seed2);
}

} // namespace vesper::ops
