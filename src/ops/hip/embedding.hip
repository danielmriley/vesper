#include <vesper/ops/embedding.h>
#include <hip/hip_runtime.h>
#include <vesper/core/stream.h>
#include <vesper/core/factories.h>

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

void embedding_hip_dispatch(const Tensor& input, const Tensor& weight, int64_t padding_idx, float max_norm, float norm_type, Tensor& out) {
    if (max_norm > 0.0f) {
        throw std::runtime_error("Embedding max_norm not yet supported on HIP backend.");
    }

    int64_t num_indices = input.numel();
    int64_t embedding_dim = weight.shape()[1];
    int64_t num_embeddings = weight.shape()[0];
    int64_t total_elements = num_indices * embedding_dim;

    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    hipStream_t stream = static_cast<hipStream_t>(Stream::current(Device::HIP).raw_handle());

    if (input.dtype() == DType::Int32) {
        hipLaunchKernelGGL(embedding_forward_kernel, dim3(blocks), dim3(threads), 0, stream,
            input.data_ptr<int32_t>(),
            weight.data_ptr<float>(),
            out.data_ptr<float>(),
            num_indices,
            embedding_dim,
            num_embeddings,
            padding_idx
        );
    } else {
        hipLaunchKernelGGL(embedding_forward_kernel, dim3(blocks), dim3(threads), 0, stream,
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
__global__ void compute_frequencies_kernel(
    int64_t* frequencies,     // [num_embeddings]
    const IndexType* indices,    // [num_indices]
    int64_t num_indices,
    int64_t padding_idx
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_indices) return;
    
    IndexType idx = indices[i];
    if (idx != static_cast<IndexType>(padding_idx)) {
        atomicAdd(reinterpret_cast<unsigned long long*>(&frequencies[idx]), 1ULL);
    }
}

__global__ void scale_grad_by_freq_kernel(
    float* grad_weight,
    const int64_t* frequencies,
    int64_t num_embeddings,
    int64_t embedding_dim
) {
    int64_t idx = blockIdx.y;
    int64_t d = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_embeddings || d >= embedding_dim) return;
    
    int64_t freq = frequencies[idx];
    if (freq > 0) {
        grad_weight[idx * embedding_dim + d] /= static_cast<float>(freq);
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

void embedding_backward_hip_dispatch(const Tensor& grad_output, const Tensor& input, int64_t num_embeddings, int64_t padding_idx, bool scale_grad_by_freq, Tensor& grad_weight) {
    int64_t num_indices = input.numel();
    int64_t embedding_dim = grad_weight.shape()[1];
    int64_t total_elements = num_indices * embedding_dim;
    
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    hipStream_t stream = static_cast<hipStream_t>(Stream::current(Device::HIP).raw_handle());
    
    if (input.dtype() == DType::Int32) {
        hipLaunchKernelGGL(embedding_backward_kernel, dim3(blocks), dim3(threads), 0, stream,
            grad_output.data_ptr<float>(),
            input.data_ptr<int32_t>(),
            grad_weight.data_ptr<float>(),
            num_indices,
            embedding_dim,
            num_embeddings,
            padding_idx
        );
    } else {
        hipLaunchKernelGGL(embedding_backward_kernel, dim3(blocks), dim3(threads), 0, stream,
            grad_output.data_ptr<float>(),
            input.data_ptr<float>(),
            grad_weight.data_ptr<float>(),
            num_indices,
            embedding_dim,
            num_embeddings,
            padding_idx
        );
    }

    if (scale_grad_by_freq) {
        // Compute frequencies
        Tensor frequencies = zeros({num_embeddings}, DType::Int64, grad_weight.device());
        
        int freq_threads = 256;
        int freq_blocks = (num_indices + freq_threads - 1) / freq_threads;
        
        if (input.dtype() == DType::Int32) {
            hipLaunchKernelGGL(compute_frequencies_kernel, dim3(freq_blocks), dim3(freq_threads), 0, stream,
                frequencies.data_ptr<int64_t>(),
                input.data_ptr<int32_t>(),
                num_indices,
                padding_idx
            );
        } else {
            hipLaunchKernelGGL(compute_frequencies_kernel, dim3(freq_blocks), dim3(freq_threads), 0, stream,
                frequencies.data_ptr<int64_t>(),
                input.data_ptr<int64_t>(),
                num_indices,
                padding_idx
            );
        }
        
        // Scale gradients
        dim3 scale_block(256);
        dim3 scale_grid(
            (embedding_dim + scale_block.x - 1) / scale_block.x,
            num_embeddings
        );
        
        hipLaunchKernelGGL(scale_grad_by_freq_kernel, dim3(scale_grid), dim3(scale_block), 0, stream,
            grad_weight.data_ptr<float>(),
            frequencies.data_ptr<int64_t>(),
            num_embeddings,
            embedding_dim
        );
    }
}

} // namespace vesper::ops
