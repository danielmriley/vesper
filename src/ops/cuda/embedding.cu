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
            out[i] = 0.0f;
        }
    }
}

void embedding_cuda_dispatch(const Tensor& input, const Tensor& weight, int64_t padding_idx, float max_norm, float norm_type, Tensor& out) {
    if (max_norm > 0.0f) {
        throw std::runtime_error("Embedding max_norm not yet supported on CUDA backend.");
    }

    int64_t num_indices = input.numel();
    int64_t embedding_dim = weight.shape()[1];
    int64_t num_embeddings = weight.shape()[0];
    int64_t total_elements = num_indices * embedding_dim;

    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    cudaStream_t stream = static_cast<cudaStream_t>(Stream::current(Device::CUDA).raw_handle());

    if (input.dtype() == DType::Int32) {
        embedding_forward_kernel<<<dim3(blocks), dim3(threads), 0, stream>>>(
            input.data_ptr<int32_t>(),
            weight.data_ptr<float>(),
            out.data_ptr<float>(),
            num_indices,
            embedding_dim,
            num_embeddings,
            padding_idx
        );
    } else {
        embedding_forward_kernel<<<dim3(blocks), dim3(threads), 0, stream>>>(
            input.data_ptr<int64_t>(),
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
    if (i < num_indices * embedding_dim) {
        int64_t row = i / embedding_dim;
        int64_t col = i % embedding_dim;
        
        int64_t idx = static_cast<int64_t>(indices[row]);
        if (idx >= 0 && idx < num_embeddings && idx != padding_idx) {
            float grad = grad_output[i];
            atomicAdd(&grad_weight[idx * embedding_dim + col], grad);
        }
    }
}

// Kernel to count frequency of each embedding index
template <typename IndexType>
__global__ void compute_frequencies_kernel(
    const IndexType* indices,
    int* frequencies,
    int64_t num_indices,
    int64_t num_embeddings)
{
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_indices) {
        int64_t idx = static_cast<int64_t>(indices[i]);
        if (idx >= 0 && idx < num_embeddings) {
            atomicAdd(&frequencies[idx], 1);
        }
    }
}

// Backward kernel with scale_grad_by_freq support
template <typename IndexType>
__global__ void embedding_backward_scaled_kernel(
    const float* grad_output,
    const IndexType* indices,
    const int* frequencies,
    float* grad_weight,
    int64_t num_indices,
    int64_t embedding_dim,
    int64_t num_embeddings,
    int64_t padding_idx)
{
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_indices * embedding_dim) {
        int64_t row = i / embedding_dim;
        int64_t col = i % embedding_dim;
        
        int64_t idx = static_cast<int64_t>(indices[row]);
        if (idx >= 0 && idx < num_embeddings && idx != padding_idx) {
            int freq = frequencies[idx];
            float scale = (freq > 0) ? (1.0f / static_cast<float>(freq)) : 0.0f;
            float grad = grad_output[i] * scale;
            atomicAdd(&grad_weight[idx * embedding_dim + col], grad);
        }
    }
}

void embedding_backward_cuda_dispatch(const Tensor& grad_output, const Tensor& input, int64_t num_embeddings, int64_t padding_idx, bool scale_grad_by_freq, Tensor& grad_weight) {
    int64_t num_indices = input.numel();
    int64_t embedding_dim = grad_weight.shape()[1];
    int64_t total_elements = num_indices * embedding_dim;
    
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    const int freq_blocks = (num_indices + threads - 1) / threads;
    
    cudaStream_t stream = static_cast<cudaStream_t>(Stream::current(Device::CUDA).raw_handle());
    
    if (scale_grad_by_freq) {
        // Allocate and zero-initialize frequency array
        int* frequencies;
        cudaMalloc(&frequencies, num_embeddings * sizeof(int));
        cudaMemsetAsync(frequencies, 0, num_embeddings * sizeof(int), stream);
        
        // Compute frequencies
        if (input.dtype() == DType::Int32) {
            compute_frequencies_kernel<<<dim3(freq_blocks), dim3(threads), 0, stream>>>(
                input.data_ptr<int32_t>(),
                frequencies,
                num_indices,
                num_embeddings
            );
            
            // Run backward with scaling
            embedding_backward_scaled_kernel<<<dim3(blocks), dim3(threads), 0, stream>>>(
                grad_output.data_ptr<float>(),
                input.data_ptr<int32_t>(),
                frequencies,
                grad_weight.data_ptr<float>(),
                num_indices,
                embedding_dim,
                num_embeddings,
                padding_idx
            );
        } else {
            compute_frequencies_kernel<<<dim3(freq_blocks), dim3(threads), 0, stream>>>(
                input.data_ptr<int64_t>(),
                frequencies,
                num_indices,
                num_embeddings
            );
            
            // Run backward with scaling
            embedding_backward_scaled_kernel<<<dim3(blocks), dim3(threads), 0, stream>>>(
                grad_output.data_ptr<float>(),
                input.data_ptr<int64_t>(),
                frequencies,
                grad_weight.data_ptr<float>(),
                num_indices,
                embedding_dim,
                num_embeddings,
                padding_idx
            );
        }
        
        cudaFree(frequencies);
    } else {
        if (input.dtype() == DType::Int32) {
            embedding_backward_kernel<<<dim3(blocks), dim3(threads), 0, stream>>>(
                grad_output.data_ptr<float>(),
                input.data_ptr<int32_t>(),
                grad_weight.data_ptr<float>(),
                num_indices,
                embedding_dim,
                num_embeddings,
                padding_idx
            );
        } else {
            embedding_backward_kernel<<<dim3(blocks), dim3(threads), 0, stream>>>(
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
}

} // namespace vesper::ops
