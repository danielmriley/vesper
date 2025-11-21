#include <vesper/ops/random.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

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

    uniform_kernel<<<blocks, threads>>>(
        tensor.data_ptr<float>(), 
        tensor.numel(), 
        min, 
        max, 
        seed
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }
}

} // namespace vesper::ops
