#include <vesper/ops/embedding.h>
#include <cuda_runtime.h>
#include <vesper/core/stream.h>

namespace vesper::ops {

template <typename IndexType>
__global__ void embedding_forward_kernel(
    const IndexType* indices, 
    const float* weight, 
    float* out,
    int64_t num_indices, 
    int64_t embedding_dim,
    int64_t num_embeddings,
    int64_t padding_idx) 
{
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_indices * embedding_dim) {
        // i goes from 0 to num_indices * dim - 1
        // row = i / dim
        // col = i % dim
        int64_t row = i / embedding_dim;
        int64_t col = i % embedding_dim;
        
        int64_t idx = static_cast<int64_t>(indices[row]);
        
        if (idx >= 0 && idx < num_embeddings) {
            if (idx == padding_idx) {
                out[i] = 0.0f;
            } else {
                out[i] = weight[idx * embedding_dim + col];
            }
        } else {
            // Out of bounds: Output zero (safe default)
            out[i] = 0.0f;
        }
    }
}

void embedding_cuda_dispatch(const Tensor& input, const Tensor& weight, int64_t padding_idx, float max_norm, Tensor& out) {
    if (max_norm > 0.0f) {
        throw std::runtime_error("Embedding max_norm not yet supported on CUDA backend.");
    }

    int64_t num_indices = input.numel();
    int64_t embedding_dim = weight.shape()[1];
    int64_t total_elements = num_indices * embedding_dim;
    int64_t num_embeddings = weight.shape()[0];
    
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    cudaStream_t stream = static_cast<cudaStream_t>(Stream::current(Device::CUDA).raw_handle());
    
    if (input.dtype() == DType::Int32) {
        embedding_forward_kernel<<<blocks, threads, 0, stream>>>(
            input.data_ptr<int32_t>(),
            weight.data_ptr<float>(),
            out.data_ptr<float>(),
            num_indices,
            embedding_dim,
            num_embeddings,
            padding_idx
        );
    } else {
        // Float indices support (cast)
        embedding_forward_kernel<<<blocks, threads, 0, stream>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            out.data_ptr<float>(),
            num_indices,
            embedding_dim,
            num_embeddings,
            padding_idx
        );
    }
}

template <typename IndexType>
__global__ void embedding_backward_kernel(
    const float* grad_output,
    const IndexType* indices,
    float* grad_weight,
    int64_t num_indices,
    int64_t embedding_dim,
    int64_t num_embeddings,
    int64_t padding_idx)
{
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    // Parallelize over grad_output elements
    if (i < num_indices * embedding_dim) {
        int64_t row = i / embedding_dim;
        int64_t col = i % embedding_dim;
        
        int64_t idx = static_cast<int64_t>(indices[row]);
        if (idx >= 0 && idx < num_embeddings && idx != padding_idx) {
            float grad = grad_output[i];
            // Scatter Add: grad_weight[idx, col] += grad
            // Using atomicAdd for float
            atomicAdd(&grad_weight[idx * embedding_dim + col], grad);
        }
    }
}

void embedding_backward_cuda_dispatch(const Tensor& grad_output, const Tensor& input, int64_t num_embeddings, int64_t padding_idx, Tensor& grad_weight) {
    int64_t num_indices = input.numel();
    int64_t embedding_dim = grad_weight.shape()[1];
    int64_t total_elements = num_indices * embedding_dim;
    
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    cudaStream_t stream = static_cast<cudaStream_t>(Stream::current(Device::CUDA).raw_handle());
    
    if (input.dtype() == DType::Int32) {
        embedding_backward_kernel<<<blocks, threads, 0, stream>>>(
            grad_output.data_ptr<float>(),
            input.data_ptr<int32_t>(),
            grad_weight.data_ptr<float>(),
            num_indices,
            embedding_dim,
            num_embeddings,
            padding_idx
        );
    } else {
        embedding_backward_kernel<<<blocks, threads, 0, stream>>>(
            grad_output.data_ptr<float>(),
            input.data_ptr<float>(),
            grad_weight.data_ptr<float>(),
            num_indices,
            embedding_dim,
            num_embeddings,
            padding_idx
        );
    }
}

} // namespace vesper::ops
