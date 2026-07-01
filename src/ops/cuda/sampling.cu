#include <vesper/ops/sampling_ops.h>
#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <cuda_runtime.h>
#include <limits>
#include <cstdint>

namespace vesper::ops {

// =============================================================================
// Random Number Generation (device functions)
// =============================================================================

/// PCG-like hash for random number generation
__device__ __forceinline__ uint32_t pcg_hash(uint32_t input) {
    uint32_t state = input * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

/// Generate random float in [0, 1)
__device__ __forceinline__ float random_float(uint32_t idx, uint32_t seed) {
    uint32_t hash = pcg_hash(idx ^ seed);
    return static_cast<float>(hash) / 4294967296.0f;  // 2^32
}

/// Atomic max on a float stored via int reinterpretation.
/// Correct for negative values (plain int atomicMax breaks on negatives).
__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
    int* a = reinterpret_cast<int*>(addr);
    int old = *a, assumed;
    do {
        assumed = old;
        float m = fmaxf(__int_as_float(assumed), value);
        old = atomicCAS(a, assumed, __float_as_int(m));
    } while (assumed != old);
    return __int_as_float(old);
}

// =============================================================================
// Argmax Kernel
// =============================================================================

__global__ void argmax_kernel(const float* __restrict__ input, 
                               int32_t* __restrict__ output,
                               int batch_size, int vocab_size) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const float* batch_input = input + batch_idx * vocab_size;
    
    extern __shared__ char shared_mem[];
    float* s_max = reinterpret_cast<float*>(shared_mem);
    int* s_idx = reinterpret_cast<int*>(s_max + blockDim.x);
    
    // Each thread finds local max
    float local_max = -INFINITY;
    int local_idx = 0;
    
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float val = batch_input[i];
        if (val > local_max) {
            local_max = val;
            local_idx = i;
        }
    }
    
    s_max[threadIdx.x] = local_max;
    s_idx[threadIdx.x] = local_idx;
    __syncthreads();
    
    // Block reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (s_max[threadIdx.x + stride] > s_max[threadIdx.x]) {
                s_max[threadIdx.x] = s_max[threadIdx.x + stride];
                s_idx[threadIdx.x] = s_idx[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        output[batch_idx] = s_idx[0];
    }
}

void argmax_cuda_dispatch(const Tensor& input, Tensor& output) {
    int64_t batch_size = input.shape()[0];
    int64_t vocab_size = input.shape()[1];
    
    int threads = 256;
    int blocks = batch_size;
    size_t shared_mem = threads * (sizeof(float) + sizeof(int));
    
    argmax_kernel<<<blocks, threads, shared_mem>>>(
        input.data_ptr<const float>(),
        output.data_ptr<int32_t>(),
        batch_size, vocab_size);
}

// =============================================================================
// Top-K Filter Kernel
// =============================================================================

__global__ void top_k_filter_kernel(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int batch_size, int vocab_size, int k) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const float* batch_input = input + batch_idx * vocab_size;
    float* batch_output = output + batch_idx * vocab_size;
    
    extern __shared__ char shared_mem[];
    float* s_vals = reinterpret_cast<float*>(shared_mem);
    int* s_idx = reinterpret_cast<int*>(s_vals + k);

    // Initialize output to -inf (parallel)
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        batch_output[i] = -INFINITY;
    }
    __syncthreads();

    // Single thread does the top-k selection to avoid race conditions
    // This is simple and correct; for large vocab, use the large kernel
    if (threadIdx.x == 0) {
        // Initialize top-k candidates with -inf
        for (int j = 0; j < k; ++j) {
            s_vals[j] = -INFINITY;
            s_idx[j] = -1;
        }

        // Find top-k values
        for (int i = 0; i < vocab_size; ++i) {
            float val = batch_input[i];

            // Find minimum in current top-k
            float min_val = s_vals[0];
            int min_pos = 0;
            for (int j = 1; j < k; ++j) {
                if (s_vals[j] < min_val) {
                    min_val = s_vals[j];
                    min_pos = j;
                }
            }

            // Update if this value is larger than the minimum
            if (val > min_val) {
                s_vals[min_pos] = val;
                s_idx[min_pos] = i;
            }
        }

        // Copy top-k values to their original positions in output
        for (int j = 0; j < k; ++j) {
            int idx = s_idx[j];
            if (idx >= 0 && idx < vocab_size) {
                batch_output[idx] = s_vals[j];
            }
        }
    }
}

__global__ void top_k_filter_large_kernel(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           int batch_size, int vocab_size, int k) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const float* batch_input = input + batch_idx * vocab_size;
    float* batch_output = output + batch_idx * vocab_size;
    
    extern __shared__ float s_samples[];
    
    int num_samples = min(blockDim.x, vocab_size);
    float sample_val = -INFINITY;
    if (threadIdx.x < num_samples) {
        int sample_idx = (vocab_size * threadIdx.x) / num_samples;
        sample_val = batch_input[sample_idx];
    }
    s_samples[threadIdx.x] = sample_val;
    __syncthreads();
    
    if (threadIdx.x == 0) {
        for (int i = 0; i < num_samples - 1; ++i) {
            for (int j = 0; j < num_samples - i - 1; ++j) {
                if (s_samples[j] < s_samples[j + 1]) {
                    float tmp = s_samples[j];
                    s_samples[j] = s_samples[j + 1];
                    s_samples[j + 1] = tmp;
                }
            }
        }
    }
    __syncthreads();
    
    int sample_k = min(k - 1, num_samples - 1);
    float threshold = s_samples[sample_k];
    
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float val = batch_input[i];
        batch_output[i] = (val >= threshold) ? val : -INFINITY;
    }
}

void top_k_filter_cuda_dispatch(const Tensor& logits, Tensor& output, int64_t k) {
    int64_t batch_size = logits.shape()[0];
    int64_t vocab_size = logits.shape()[1];
    
    int threads = 256;
    int blocks = batch_size;
    
    if (k <= 64) {
        size_t shared_mem = k * (sizeof(float) + sizeof(int));
        top_k_filter_kernel<<<blocks, threads, shared_mem>>>(
            logits.data_ptr<const float>(),
            output.data_ptr<float>(),
            batch_size, vocab_size, k);
    } else {
        size_t shared_mem = threads * sizeof(float);
        top_k_filter_large_kernel<<<blocks, threads, shared_mem>>>(
            logits.data_ptr<const float>(),
            output.data_ptr<float>(),
            batch_size, vocab_size, k);
    }
}

// =============================================================================
// Top-P (Nucleus) Filter Kernel
// =============================================================================

__global__ void top_p_filter_opt_kernel(const float* __restrict__ input,
                                         float* __restrict__ output,
                                         int batch_size, int vocab_size, float p) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const float* batch_input = input + batch_idx * vocab_size;
    float* batch_output = output + batch_idx * vocab_size;
    
    // Find max
    float max_val = -INFINITY;
    int max_idx = 0;
    
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float val = batch_input[i];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }
    
    __shared__ float s_max_val;
    __shared__ int s_max_idx;
    if (threadIdx.x == 0) {
        s_max_val = -INFINITY;
        s_max_idx = 0;
    }
    __syncthreads();
    
    float old = atomicMaxFloat(&s_max_val, max_val);
    if (old < max_val) {
        s_max_idx = max_idx;
    }
    __syncthreads();
    
    // Compute softmax sum
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        local_sum += expf(batch_input[i] - s_max_val);
    }
    
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();
    atomicAdd(&s_sum, local_sum);
    __syncthreads();
    
    float max_prob = expf(batch_input[s_max_idx] - s_max_val) / s_sum;
    
    __shared__ float threshold_prob;
    if (threadIdx.x == 0) {
        if (max_prob >= p) {
            threshold_prob = max_prob - 1e-6f;
        } else {
            float cumsum = 0.0f;
            threshold_prob = 1.0f;
            
            for (float thresh = 0.9f; thresh >= 0.0f; thresh -= 0.1f) {
                cumsum = 0.0f;
                for (int i = 0; i < vocab_size; ++i) {
                    float prob = expf(batch_input[i] - s_max_val) / s_sum;
                    if (prob >= thresh) cumsum += prob;
                }
                if (cumsum >= p) {
                    threshold_prob = thresh;
                    break;
                }
            }
        }
    }
    __syncthreads();
    
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float prob = expf(batch_input[i] - s_max_val) / s_sum;
        batch_output[i] = (prob >= threshold_prob) ? batch_input[i] : -INFINITY;
    }
}

void top_p_filter_cuda_dispatch(const Tensor& logits, Tensor& output, float p) {
    int64_t batch_size = logits.shape()[0];
    int64_t vocab_size = logits.shape()[1];
    
    int threads = 256;
    int blocks = batch_size;
    
    top_p_filter_opt_kernel<<<blocks, threads>>>(
        logits.data_ptr<const float>(),
        output.data_ptr<float>(),
        batch_size, vocab_size, p);
}

// =============================================================================
// Repetition Penalty Kernel
// =============================================================================

__global__ void repetition_penalty_kernel(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           const int32_t* __restrict__ token_ids,
                                           int batch_size, int vocab_size, 
                                           int seq_len, float penalty) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const float* batch_input = input + batch_idx * vocab_size;
    float* batch_output = output + batch_idx * vocab_size;
    const int32_t* batch_ids = token_ids + batch_idx * seq_len;
    
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        batch_output[i] = batch_input[i];
    }
    __syncthreads();
    
    for (int s = threadIdx.x; s < seq_len; s += blockDim.x) {
        int32_t token_id = batch_ids[s];
        if (token_id >= 0 && token_id < vocab_size) {
            float logit = batch_input[token_id];
            float penalized;
            if (logit > 0) {
                penalized = logit / penalty;
            } else {
                penalized = logit * penalty;
            }
            atomicExch(&batch_output[token_id], penalized);
        }
    }
}

void repetition_penalty_cuda_dispatch(const Tensor& logits, Tensor& output,
                                       const Tensor& input_ids, float penalty) {
    int64_t batch_size = logits.shape()[0];
    int64_t vocab_size = logits.shape()[1];
    int64_t seq_len = input_ids.shape()[1];
    
    int threads = 256;
    int blocks = batch_size;
    
    repetition_penalty_kernel<<<blocks, threads>>>(
        logits.data_ptr<const float>(),
        output.data_ptr<float>(),
        input_ids.data_ptr<const int32_t>(),
        batch_size, vocab_size, seq_len, penalty);
}

// =============================================================================
// Multinomial Sampling Kernel
// =============================================================================

__global__ void multinomial_kernel(const float* __restrict__ probs,
                                    int32_t* __restrict__ output,
                                    int batch_size, int vocab_size,
                                    int num_samples, uint64_t base_seed) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const float* batch_probs = probs + batch_idx * vocab_size;
    int32_t* batch_output = output + batch_idx * num_samples;
    
    for (int s = threadIdx.x; s < num_samples; s += blockDim.x) {
        uint32_t seed = static_cast<uint32_t>(base_seed + batch_idx * 1000 + s);
        float r = random_float(s, seed);
        
        float cumsum = 0.0f;
        int selected = vocab_size - 1;
        
        for (int i = 0; i < vocab_size; ++i) {
            cumsum += batch_probs[i];
            if (r < cumsum) {
                selected = i;
                break;
            }
        }
        
        batch_output[s] = selected;
    }
}

__global__ void multinomial_cdf_kernel(const float* __restrict__ probs,
                                        int32_t* __restrict__ output,
                                        int batch_size, int vocab_size,
                                        int num_samples, uint64_t base_seed) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const float* batch_probs = probs + batch_idx * vocab_size;
    int32_t* batch_output = output + batch_idx * num_samples;
    
    extern __shared__ float cdf[];
    
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        cdf[i] = batch_probs[i];
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        float cumsum = 0.0f;
        for (int i = 0; i < vocab_size; ++i) {
            cumsum += cdf[i];
            cdf[i] = cumsum;
        }
    }
    __syncthreads();
    
    for (int s = threadIdx.x; s < num_samples; s += blockDim.x) {
        uint32_t seed = static_cast<uint32_t>(base_seed + batch_idx * 10000 + s * 7);
        float r = random_float(s, seed);
        
        int lo = 0, hi = vocab_size - 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (cdf[mid] < r) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        
        batch_output[s] = lo;
    }
}

void multinomial_cuda_dispatch(const Tensor& probs, Tensor& output,
                                int64_t num_samples, uint64_t seed) {
    int64_t batch_size = probs.shape()[0];
    int64_t vocab_size = probs.shape()[1];
    
    int threads = 256;
    int blocks = batch_size;
    
    if (vocab_size <= 8192) {
        size_t shared_mem = vocab_size * sizeof(float);
        multinomial_cdf_kernel<<<blocks, threads, shared_mem>>>(
            probs.data_ptr<const float>(),
            output.data_ptr<int32_t>(),
            batch_size, vocab_size, num_samples, seed);
    } else {
        multinomial_kernel<<<blocks, threads>>>(
            probs.data_ptr<const float>(),
            output.data_ptr<int32_t>(),
            batch_size, vocab_size, num_samples, seed);
    }
}

// =============================================================================
// TopK with Values and Indices
// =============================================================================

__global__ void topk_kernel(const float* __restrict__ input,
                             float* __restrict__ values,
                             int32_t* __restrict__ indices,
                             int batch_size, int n, int k) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const float* batch_input = input + batch_idx * n;
    float* batch_values = values + batch_idx * k;
    int32_t* batch_indices = indices + batch_idx * k;
    
    extern __shared__ char smem[];
    float* s_vals = reinterpret_cast<float*>(smem);
    int* s_idx = reinterpret_cast<int*>(s_vals + k);
    
    if (threadIdx.x < k) {
        s_vals[threadIdx.x] = -INFINITY;
        s_idx[threadIdx.x] = -1;
    }
    __syncthreads();
    
    // Pad the loop bound so every thread reaches the barrier each iteration,
    // even when n is not a multiple of blockDim.x (e.g. vocab 50257).
    int n_padded = ((n + blockDim.x - 1) / blockDim.x) * blockDim.x;
    for (int i = threadIdx.x; i < n_padded; i += blockDim.x) {
        if (i < n) {
            float val = batch_input[i];

            int min_pos = 0;
            float min_val = s_vals[0];
            for (int j = 1; j < k; ++j) {
                if (s_vals[j] < min_val) {
                    min_val = s_vals[j];
                    min_pos = j;
                }
            }

            if (val > min_val) {
                s_vals[min_pos] = val;
                s_idx[min_pos] = i;
            }
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        for (int i = 0; i < k - 1; ++i) {
            for (int j = 0; j < k - i - 1; ++j) {
                if (s_vals[j] < s_vals[j + 1]) {
                    float tmp_v = s_vals[j];
                    s_vals[j] = s_vals[j + 1];
                    s_vals[j + 1] = tmp_v;
                    
                    int tmp_i = s_idx[j];
                    s_idx[j] = s_idx[j + 1];
                    s_idx[j + 1] = tmp_i;
                }
            }
        }
    }
    __syncthreads();
    
    if (threadIdx.x < k) {
        batch_values[threadIdx.x] = s_vals[threadIdx.x];
        batch_indices[threadIdx.x] = s_idx[threadIdx.x];
    }
}

void topk_cuda_dispatch(const Tensor& input, int64_t k, Tensor& values, Tensor& indices) {
    int64_t batch_size = input.shape()[0];
    int64_t n = input.shape()[1];
    
    int threads = 256;
    int blocks = batch_size;
    size_t shared_mem = k * (sizeof(float) + sizeof(int));
    
    topk_kernel<<<blocks, threads, shared_mem>>>(
        input.data_ptr<const float>(),
        values.data_ptr<float>(),
        indices.data_ptr<int32_t>(),
        batch_size, n, k);
}

// =============================================================================
// Softmax 2D Kernel
// =============================================================================

__global__ void softmax_2d_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int batch_size, int n) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const float* batch_input = input + batch_idx * n;
    float* batch_output = output + batch_idx * n;
    
    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        local_max = fmaxf(local_max, batch_input[i]);
    }
    
    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = -INFINITY;
    __syncthreads();
    atomicMaxFloat(&s_max, local_max);
    __syncthreads();
    
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float exp_val = expf(batch_input[i] - s_max);
        batch_output[i] = exp_val;
        local_sum += exp_val;
    }
    
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();
    atomicAdd(&s_sum, local_sum);
    __syncthreads();
    
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        batch_output[i] /= s_sum;
    }
}

void softmax_2d_cuda_dispatch(const Tensor& input, Tensor& output) {
    int64_t batch_size = input.shape()[0];
    int64_t n = input.shape()[1];
    
    int threads = 256;
    int blocks = batch_size;
    
    softmax_2d_kernel<<<blocks, threads>>>(
        input.data_ptr<const float>(),
        output.data_ptr<float>(),
        batch_size, n);
}

// =============================================================================
// Stop Token Checking Kernel
// =============================================================================

__global__ void check_stop_tokens_kernel(const int32_t* __restrict__ tokens,
                                          int32_t* __restrict__ should_stop,
                                          const int64_t* __restrict__ stop_ids,
                                          int batch_size, int num_stop_ids,
                                          int64_t min_generated, int64_t generated_count) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;
    
    if (generated_count < min_generated) {
        should_stop[batch_idx] = 0;
        return;
    }
    
    int32_t token = tokens[batch_idx];
    int32_t stop = 0;
    
    for (int i = 0; i < num_stop_ids; ++i) {
        if (token == static_cast<int32_t>(stop_ids[i])) {
            stop = 1;
            break;
        }
    }
    
    should_stop[batch_idx] = stop;
}

void check_stop_tokens_cuda_dispatch(const Tensor& tokens, Tensor& output,
                                      const int64_t* stop_ids, int64_t num_stop_ids,
                                      int64_t min_generated, int64_t generated_count) {
    int64_t batch_size = tokens.numel();
    
    int64_t* d_stop_ids = nullptr;
    if (num_stop_ids > 0) {
        cudaMalloc(&d_stop_ids, num_stop_ids * sizeof(int64_t));
        cudaMemcpy(d_stop_ids, stop_ids, num_stop_ids * sizeof(int64_t), cudaMemcpyHostToDevice);
    }
    
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    
    check_stop_tokens_kernel<<<blocks, threads>>>(
        tokens.data_ptr<const int32_t>(),
        output.data_ptr<int32_t>(),
        d_stop_ids,
        batch_size, num_stop_ids,
        min_generated, generated_count);
    
    cudaDeviceSynchronize();
    
    if (d_stop_ids) {
        cudaFree(d_stop_ids);
    }
}

// =============================================================================
// All True Reduction
// =============================================================================

__global__ void all_true_kernel(const int32_t* __restrict__ mask,
                                 int* __restrict__ result,
                                 int n) {
    extern __shared__ int s_all[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int local_all = 1;
    if (idx < n) {
        local_all = (mask[idx] != 0) ? 1 : 0;
    }
    s_all[tid] = local_all;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_all[tid] = s_all[tid] & s_all[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAnd(result, s_all[0]);
    }
}

bool all_true_cuda_dispatch(const Tensor& mask) {
    int64_t n = mask.numel();
    if (n == 0) return true;
    
    int* d_result;
    cudaMalloc(&d_result, sizeof(int));
    int initial = 1;
    cudaMemcpy(d_result, &initial, sizeof(int), cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    size_t shared_mem = threads * sizeof(int);
    
    all_true_kernel<<<blocks, threads, shared_mem>>>(
        mask.data_ptr<const int32_t>(),
        d_result,
        n);
    
    cudaDeviceSynchronize();
    
    int result;
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    
    return result != 0;
}

} // namespace vesper::ops
