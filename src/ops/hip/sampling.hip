#include <vesper/ops/sampling_ops.h>
#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <hip/hip_runtime.h>
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

// =============================================================================
// Warp-level Primitives
// =============================================================================

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down(val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset);
    }
    return val;
}

__device__ __forceinline__ int warp_reduce_max_idx(float val, int idx) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_val = __shfl_down(val, offset);
        int other_idx = __shfl_down(idx, offset);
        if (other_val > val) {
            val = other_val;
            idx = other_idx;
        }
    }
    return idx;
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

void argmax_hip_dispatch(const Tensor& input, Tensor& output) {
    int64_t batch_size = input.shape()[0];
    int64_t vocab_size = input.shape()[1];
    
    int threads = 256;
    int blocks = batch_size;
    size_t shared_mem = threads * (sizeof(float) + sizeof(int));
    
    hipLaunchKernelGGL(argmax_kernel, dim3(blocks), dim3(threads), shared_mem, 0,
                       input.data_ptr<const float>(),
                       output.data_ptr<int32_t>(),
                       batch_size, vocab_size);
}

// =============================================================================
// Top-K Filter Kernel
// =============================================================================

/// Parallel bitonic sort for small K values
/// For larger K, we use a partial sort approach
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
    
    // Initialize top-k candidates with -inf
    if (threadIdx.x < k) {
        s_vals[threadIdx.x] = -INFINITY;
        s_idx[threadIdx.x] = -1;
    }
    __syncthreads();
    
    // Each thread processes multiple elements
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float val = batch_input[i];
        
        // Check if this value should be in top-k
        // Find minimum in current top-k
        float min_val = s_vals[0];
        int min_pos = 0;
        for (int j = 1; j < k; ++j) {
            if (s_vals[j] < min_val) {
                min_val = s_vals[j];
                min_pos = j;
            }
        }
        
        // Atomic-like update (simplified - one thread at a time per batch)
        if (val > min_val) {
            // This is not fully thread-safe, but works for demonstration
            // For production, use atomic operations or better algorithms
            s_vals[min_pos] = val;
            s_idx[min_pos] = i;
        }
        __syncthreads();
    }
    
    // Now we have top-k values and indices in shared memory
    // Initialize output to -inf
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        batch_output[i] = -INFINITY;
    }
    __syncthreads();
    
    // Copy top-k values to their original positions
    if (threadIdx.x < k) {
        int idx = s_idx[threadIdx.x];
        if (idx >= 0 && idx < vocab_size) {
            batch_output[idx] = s_vals[threadIdx.x];
        }
    }
}

/// More efficient top-k using radix select for larger vocabularies
__global__ void top_k_filter_large_kernel(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           int batch_size, int vocab_size, int k) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const float* batch_input = input + batch_idx * vocab_size;
    float* batch_output = output + batch_idx * vocab_size;
    
    // Strategy: Find the k-th largest value, then filter
    // Step 1: Find approximate k-th value using sampling
    extern __shared__ float s_samples[];
    
    // Sample some values
    int num_samples = min(blockDim.x, vocab_size);
    float sample_val = -INFINITY;
    if (threadIdx.x < num_samples) {
        int sample_idx = (vocab_size * threadIdx.x) / num_samples;
        sample_val = batch_input[sample_idx];
    }
    s_samples[threadIdx.x] = sample_val;
    __syncthreads();
    
    // Sort samples (bubble sort for small arrays in shared mem)
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
    
    // Use k-th sample as threshold estimate
    int sample_k = min(k - 1, num_samples - 1);
    float threshold = s_samples[sample_k];
    
    // Count values above threshold
    int count_above = 0;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        if (batch_input[i] > threshold) {
            atomicAdd(&count_above, 1);
        }
    }
    __syncthreads();
    
    // Adjust threshold if needed (simplified - in production, iterate)
    // For now, just use the threshold as-is
    
    // Set output: keep values above threshold (approximately k values)
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float val = batch_input[i];
        // Include if above threshold or if we need more values
        batch_output[i] = (val >= threshold) ? val : -INFINITY;
    }
}

void top_k_filter_hip_dispatch(const Tensor& logits, Tensor& output, int64_t k) {
    int64_t batch_size = logits.shape()[0];
    int64_t vocab_size = logits.shape()[1];
    
    int threads = 256;
    int blocks = batch_size;
    
    if (k <= 64) {
        // Use simple approach for small k
        size_t shared_mem = k * (sizeof(float) + sizeof(int));
        hipLaunchKernelGGL(top_k_filter_kernel, dim3(blocks), dim3(threads), shared_mem, 0,
                           logits.data_ptr<const float>(),
                           output.data_ptr<float>(),
                           batch_size, vocab_size, k);
    } else {
        // Use sampling-based approach for large k
        size_t shared_mem = threads * sizeof(float);
        hipLaunchKernelGGL(top_k_filter_large_kernel, dim3(blocks), dim3(threads), shared_mem, 0,
                           logits.data_ptr<const float>(),
                           output.data_ptr<float>(),
                           batch_size, vocab_size, k);
    }
}

// =============================================================================
// Top-P (Nucleus) Filter Kernel
// =============================================================================

__global__ void top_p_filter_kernel(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int batch_size, int vocab_size, float p) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const float* batch_input = input + batch_idx * vocab_size;
    float* batch_output = output + batch_idx * vocab_size;
    
    extern __shared__ float shared[];
    float* s_probs = shared;
    float* s_sorted_probs = s_probs + vocab_size;
    int* s_sorted_idx = reinterpret_cast<int*>(s_sorted_probs + vocab_size);
    
    // Step 1: Compute softmax (find max, compute exp, sum, normalize)
    // Find max
    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        local_max = fmaxf(local_max, batch_input[i]);
    }
    
    __shared__ float block_max;
    if (threadIdx.x == 0) block_max = -INFINITY;
    __syncthreads();
    
    // Reduce max
    atomicMax((int*)&block_max, __float_as_int(local_max));
    __syncthreads();
    
    // Compute exp and sum
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float exp_val = expf(batch_input[i] - block_max);
        s_probs[i] = exp_val;
        local_sum += exp_val;
    }
    
    __shared__ float block_sum;
    if (threadIdx.x == 0) block_sum = 0.0f;
    __syncthreads();
    
    atomicAdd(&block_sum, local_sum);
    __syncthreads();
    
    // Normalize
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        s_probs[i] /= block_sum;
    }
    __syncthreads();
    
    // Step 2: Find threshold for top-p
    // For simplicity, use a histogram-based approach
    // Count how many tokens are needed to reach probability p
    
    // Find the k-th largest probability that gives cumsum > p
    // This is expensive on GPU - use approximate method
    
    // Simple approach: iterate through probability buckets
    __shared__ float threshold;
    if (threadIdx.x == 0) {
        // Serial computation on one thread (not ideal but correct)
        // In production, use parallel sorting
        float cumsum = 0.0f;
        threshold = 0.0f;
        
        // Find max prob first
        float max_prob = 0.0f;
        for (int i = 0; i < vocab_size; ++i) {
            if (s_probs[i] > max_prob) max_prob = s_probs[i];
        }
        
        // Iterate from high to low probability
        // This is O(V^2) worst case - fine for small vocab, use sorting for large
        float current_thresh = max_prob;
        while (cumsum < p && current_thresh > 1e-10f) {
            float next_thresh = 0.0f;
            for (int i = 0; i < vocab_size; ++i) {
                float prob = s_probs[i];
                if (prob >= current_thresh && prob < threshold + 1e-10f) {
                    // Not yet counted
                    cumsum += prob;
                }
                if (prob < current_thresh && prob > next_thresh) {
                    next_thresh = prob;
                }
            }
            threshold = current_thresh;
            current_thresh = next_thresh;
        }
    }
    __syncthreads();
    
    // Step 3: Set output
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float prob = s_probs[i];
        batch_output[i] = (prob >= threshold) ? batch_input[i] : -INFINITY;
    }
}

/// Optimized top-p using parallel prefix sum
__global__ void top_p_filter_opt_kernel(const float* __restrict__ input,
                                         float* __restrict__ output,
                                         int batch_size, int vocab_size, float p) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const float* batch_input = input + batch_idx * vocab_size;
    float* batch_output = output + batch_idx * vocab_size;
    
    // Step 1: Online softmax and find max value/index
    float max_val = -INFINITY;
    int max_idx = 0;
    
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float val = batch_input[i];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }
    
    // Block reduce max
    __shared__ float s_max_val;
    __shared__ int s_max_idx;
    if (threadIdx.x == 0) {
        s_max_val = -INFINITY;
        s_max_idx = 0;
    }
    __syncthreads();
    
    // Use atomicMax with float trick
    int old = atomicMax((int*)&s_max_val, __float_as_int(max_val));
    if (__int_as_float(old) < max_val) {
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
    
    // Get max probability
    float max_prob = expf(batch_input[s_max_idx] - s_max_val) / s_sum;
    
    // If max_prob >= p, only keep max
    __shared__ float threshold_prob;
    if (threadIdx.x == 0) {
        if (max_prob >= p) {
            threshold_prob = max_prob - 1e-6f;  // Only max
        } else {
            // Need to find proper threshold
            // Use iterative refinement
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
    
    // Set output
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float prob = expf(batch_input[i] - s_max_val) / s_sum;
        batch_output[i] = (prob >= threshold_prob) ? batch_input[i] : -INFINITY;
    }
}

void top_p_filter_hip_dispatch(const Tensor& logits, Tensor& output, float p) {
    int64_t batch_size = logits.shape()[0];
    int64_t vocab_size = logits.shape()[1];
    
    int threads = 256;
    int blocks = batch_size;
    
    // Use optimized kernel
    hipLaunchKernelGGL(top_p_filter_opt_kernel, dim3(blocks), dim3(threads), 0, 0,
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
    
    // Copy input to output first
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        batch_output[i] = batch_input[i];
    }
    __syncthreads();
    
    // Apply penalty to tokens in sequence
    // Each thread handles a subset of the sequence
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
            // Use atomicExch to handle race conditions
            // (if same token appears multiple times, last write wins - same result)
            atomicExch(&batch_output[token_id], penalized);
        }
    }
}

void repetition_penalty_hip_dispatch(const Tensor& logits, Tensor& output,
                                      const Tensor& input_ids, float penalty) {
    int64_t batch_size = logits.shape()[0];
    int64_t vocab_size = logits.shape()[1];
    int64_t seq_len = input_ids.shape()[1];
    
    int threads = 256;
    int blocks = batch_size;
    
    hipLaunchKernelGGL(repetition_penalty_kernel, dim3(blocks), dim3(threads), 0, 0,
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
    
    // Each thread handles one sample
    for (int s = threadIdx.x; s < num_samples; s += blockDim.x) {
        // Generate random number in [0, 1)
        uint32_t seed = static_cast<uint32_t>(base_seed + batch_idx * 1000 + s);
        float r = random_float(s, seed);
        
        // Binary search for the token (cumulative sum method)
        float cumsum = 0.0f;
        int selected = vocab_size - 1;  // Default to last token
        
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

/// Optimized multinomial using precomputed CDF
__global__ void multinomial_cdf_kernel(const float* __restrict__ probs,
                                        int32_t* __restrict__ output,
                                        int batch_size, int vocab_size,
                                        int num_samples, uint64_t base_seed) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const float* batch_probs = probs + batch_idx * vocab_size;
    int32_t* batch_output = output + batch_idx * num_samples;
    
    // Compute CDF in shared memory
    extern __shared__ float cdf[];
    
    // Parallel prefix sum for CDF
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        cdf[i] = batch_probs[i];
    }
    __syncthreads();
    
    // Simple sequential CDF (for correctness; can be optimized with scan)
    if (threadIdx.x == 0) {
        float cumsum = 0.0f;
        for (int i = 0; i < vocab_size; ++i) {
            cumsum += cdf[i];
            cdf[i] = cumsum;
        }
    }
    __syncthreads();
    
    // Sample using binary search on CDF
    for (int s = threadIdx.x; s < num_samples; s += blockDim.x) {
        uint32_t seed = static_cast<uint32_t>(base_seed + batch_idx * 10000 + s * 7);
        float r = random_float(s, seed);
        
        // Binary search
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

void multinomial_hip_dispatch(const Tensor& probs, Tensor& output,
                               int64_t num_samples, uint64_t seed) {
    int64_t batch_size = probs.shape()[0];
    int64_t vocab_size = probs.shape()[1];
    
    int threads = 256;
    int blocks = batch_size;
    
    if (vocab_size <= 8192) {
        // Use CDF-based method for smaller vocabularies
        size_t shared_mem = vocab_size * sizeof(float);
        hipLaunchKernelGGL(multinomial_cdf_kernel, dim3(blocks), dim3(threads), shared_mem, 0,
                           probs.data_ptr<const float>(),
                           output.data_ptr<int32_t>(),
                           batch_size, vocab_size, num_samples, seed);
    } else {
        // Use simpler method for large vocabularies
        hipLaunchKernelGGL(multinomial_kernel, dim3(blocks), dim3(threads), 0, 0,
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
    
    // Use shared memory for top-k heap
    extern __shared__ char smem[];
    float* s_vals = reinterpret_cast<float*>(smem);
    int* s_idx = reinterpret_cast<int*>(s_vals + k);
    
    // Initialize with -inf
    if (threadIdx.x < k) {
        s_vals[threadIdx.x] = -INFINITY;
        s_idx[threadIdx.x] = -1;
    }
    __syncthreads();
    
    // Find top-k (simple insertion sort approach for small k)
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float val = batch_input[i];
        
        // Check if this should be in top-k
        // Find min in current top-k
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
        __syncthreads();
    }
    
    // Sort the top-k (bubble sort for small k)
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
    
    // Copy to output
    if (threadIdx.x < k) {
        batch_values[threadIdx.x] = s_vals[threadIdx.x];
        batch_indices[threadIdx.x] = s_idx[threadIdx.x];
    }
}

void topk_hip_dispatch(const Tensor& input, int64_t k, Tensor& values, Tensor& indices) {
    int64_t batch_size = input.shape()[0];
    int64_t n = input.shape()[1];
    
    int threads = 256;
    int blocks = batch_size;
    size_t shared_mem = k * (sizeof(float) + sizeof(int));
    
    hipLaunchKernelGGL(topk_kernel, dim3(blocks), dim3(threads), shared_mem, 0,
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
    
    // Find max
    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        local_max = fmaxf(local_max, batch_input[i]);
    }
    
    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = -INFINITY;
    __syncthreads();
    atomicMax((int*)&s_max, __float_as_int(local_max));
    __syncthreads();
    
    // Compute exp and sum
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
    
    // Normalize
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        batch_output[i] /= s_sum;
    }
}

void softmax_2d_hip_dispatch(const Tensor& input, Tensor& output) {
    int64_t batch_size = input.shape()[0];
    int64_t n = input.shape()[1];
    
    int threads = 256;
    int blocks = batch_size;
    
    hipLaunchKernelGGL(softmax_2d_kernel, dim3(blocks), dim3(threads), 0, 0,
                       input.data_ptr<const float>(),
                       output.data_ptr<float>(),
                       batch_size, n);
}

// =============================================================================
// Stop Token Checking Kernel
// =============================================================================

/// Check if each token matches any stop token ID
/// Returns a mask (stored as int32_t: 1 = should stop, 0 = continue)
__global__ void check_stop_tokens_kernel(const int32_t* __restrict__ tokens,
                                          int32_t* __restrict__ should_stop,
                                          const int64_t* __restrict__ stop_ids,
                                          int batch_size, int num_stop_ids,
                                          int64_t min_generated, int64_t generated_count) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;
    
    // If we haven't generated minimum tokens yet, don't stop
    if (generated_count < min_generated) {
        should_stop[batch_idx] = 0;
        return;
    }
    
    int32_t token = tokens[batch_idx];
    int32_t stop = 0;
    
    // Check against all stop tokens
    for (int i = 0; i < num_stop_ids; ++i) {
        if (token == static_cast<int32_t>(stop_ids[i])) {
            stop = 1;
            break;
        }
    }
    
    should_stop[batch_idx] = stop;
}

void check_stop_tokens_hip_dispatch(const Tensor& tokens, Tensor& output,
                                     const int64_t* stop_ids, int64_t num_stop_ids,
                                     int64_t min_generated, int64_t generated_count) {
    int64_t batch_size = tokens.numel();  // [Batch] or [Batch, 1]
    
    // Copy stop IDs to device
    int64_t* d_stop_ids = nullptr;
    if (num_stop_ids > 0) {
        hipMalloc(&d_stop_ids, num_stop_ids * sizeof(int64_t));
        hipMemcpy(d_stop_ids, stop_ids, num_stop_ids * sizeof(int64_t), hipMemcpyHostToDevice);
    }
    
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    
    hipLaunchKernelGGL(check_stop_tokens_kernel, dim3(blocks), dim3(threads), 0, 0,
                       tokens.data_ptr<const int32_t>(),
                       output.data_ptr<int32_t>(),
                       d_stop_ids,
                       batch_size, num_stop_ids,
                       min_generated, generated_count);
    
    hipDeviceSynchronize();
    
    if (d_stop_ids) {
        hipFree(d_stop_ids);
    }
}

// =============================================================================
// All True Reduction
// =============================================================================

/// Reduce boolean mask to check if all elements are true (non-zero)
/// Uses atomic AND reduction
__global__ void all_true_kernel(const int32_t* __restrict__ mask,
                                 int* __restrict__ result,
                                 int n) {
    extern __shared__ int s_all[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load and check
    int local_all = 1;
    if (idx < n) {
        local_all = (mask[idx] != 0) ? 1 : 0;
    }
    s_all[tid] = local_all;
    __syncthreads();
    
    // Block reduction with AND
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_all[tid] = s_all[tid] & s_all[tid + stride];
        }
        __syncthreads();
    }
    
    // Atomic AND to global result
    if (tid == 0) {
        atomicAnd(result, s_all[0]);
    }
}

bool all_true_hip_dispatch(const Tensor& mask) {
    int64_t n = mask.numel();
    if (n == 0) return true;
    
    // Allocate result on device (initialized to 1 = true)
    int* d_result;
    hipMalloc(&d_result, sizeof(int));
    int initial = 1;
    hipMemcpy(d_result, &initial, sizeof(int), hipMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    size_t shared_mem = threads * sizeof(int);
    
    hipLaunchKernelGGL(all_true_kernel, dim3(blocks), dim3(threads), shared_mem, 0,
                       mask.data_ptr<const int32_t>(),
                       d_result,
                       n);
    
    hipDeviceSynchronize();
    
    int result;
    hipMemcpy(&result, d_result, sizeof(int), hipMemcpyDeviceToHost);
    hipFree(d_result);
    
    return result != 0;
}

} // namespace vesper::ops
