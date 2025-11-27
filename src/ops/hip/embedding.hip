#include <vesper/ops/embedding.h>
#include <hip/hip_runtime.h>
#include <vesper/core/stream.h>
#include <vesper/core/factories.h>
#include <cmath>

namespace vesper::ops {

// Kernel to compute p-norm of each embedding row and renormalize if > max_norm
// This modifies weights in-place before the lookup
template <typename IndexType>
__global__ void embedding_max_norm_kernel(
    float* weight,           // [num_embeddings, embedding_dim] - modified in-place
    const IndexType* indices,
    int64_t num_indices,
    int64_t embedding_dim,
    int64_t num_embeddings,
    float max_norm,
    float norm_type)
{
    // Each block handles one index
    int64_t idx_pos = blockIdx.x;
    if (idx_pos >= num_indices) return;
    
    int64_t emb_idx = static_cast<int64_t>(indices[idx_pos]);
    if (emb_idx < 0 || emb_idx >= num_embeddings) return;
    
    float* row = weight + emb_idx * embedding_dim;
    
    // Compute p-norm using shared memory reduction
    __shared__ float partial_sums[256];
    
    float local_sum = 0.0f;
    for (int64_t d = threadIdx.x; d < embedding_dim; d += blockDim.x) {
        float val = row[d];
        local_sum += powf(fabsf(val), norm_type);
    }
    
    partial_sums[threadIdx.x] = local_sum;
    __syncthreads();
    
    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // Thread 0 computes final norm and broadcasts scale
    __shared__ float scale;
    if (threadIdx.x == 0) {
        float norm = powf(partial_sums[0], 1.0f / norm_type);
        if (norm > max_norm) {
            scale = max_norm / (norm + 1e-7f);
        } else {
            scale = 1.0f;
        }
    }
    __syncthreads();
    
    // Apply scaling if needed
    if (scale != 1.0f) {
        for (int64_t d = threadIdx.x; d < embedding_dim; d += blockDim.x) {
            row[d] *= scale;
        }
    }
}

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
    int64_t num_indices = input.numel();
    int64_t embedding_dim = weight.shape()[1];
    int64_t num_embeddings = weight.shape()[0];
    int64_t total_elements = num_indices * embedding_dim;

    hipStream_t stream = static_cast<hipStream_t>(Stream::current(Device::HIP).raw_handle());

    // Apply max_norm renormalization if requested
    if (max_norm > 0.0f) {
        // We need to modify weights in-place, so cast away const
        float* weight_ptr = const_cast<float*>(weight.data_ptr<float>());
        
        // Launch one block per index, each block handles one embedding row
        int threads_per_block = 256;
        int num_blocks = num_indices;
        
        if (input.dtype() == DType::Int32) {
            hipLaunchKernelGGL(embedding_max_norm_kernel, dim3(num_blocks), dim3(threads_per_block), 0, stream,
                weight_ptr,
                input.data_ptr<int32_t>(),
                num_indices,
                embedding_dim,
                num_embeddings,
                max_norm,
                norm_type
            );
        } else {
            hipLaunchKernelGGL(embedding_max_norm_kernel, dim3(num_blocks), dim3(threads_per_block), 0, stream,
                weight_ptr,
                input.data_ptr<int64_t>(),
                num_indices,
                embedding_dim,
                num_embeddings,
                max_norm,
                norm_type
            );
        }
    }

    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

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
