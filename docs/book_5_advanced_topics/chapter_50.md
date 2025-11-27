# Chapter 50: GPU-Native Beam Search

## Overview

This chapter documents the upgrade path from CPU-based beam search candidate selection to a fully GPU-native implementation. The current implementation in `src/generation/beam_search.cpp` uses `std::vector` and `std::priority_queue` for simplicity, but this requires CPU-GPU synchronization at each generation step.

PyTorch/HuggingFace Transformers demonstrates that full GPU beam search is achievable using tensor-based operations. This chapter provides a detailed roadmap for implementing this optimization in Vesper.

## Current Architecture Analysis

### What We Have Now

```cpp
// Current: CPU-based data structures
std::vector<std::vector<float>> beam_scores(batch_size, std::vector<float>(num_beams));
std::vector<std::vector<std::vector<int32_t>>> beam_tokens(batch_size);
std::vector<std::vector<bool>> beam_done(batch_size, std::vector<bool>(num_beams));

// CPU priority queue for candidate selection
std::priority_queue<Candidate> candidates;
for (int64_t beam = 0; beam < num_beams; ++beam) {
    for (int64_t v = 0; v < vocab_size; ++v) {
        candidates.push({score, beam, token});
    }
}
```

### Problems with Current Approach

1. **CPU-GPU Synchronization**: Every step requires `log_probs.to(Device::CPU)` transfer
2. **Sequential Selection**: Priority queue operations are O(n log k), run on CPU
3. **Dynamic Memory**: `std::vector` resizing and copying
4. **Branching Logic**: Complex conditionals that don't map well to GPU

## Target Architecture: Tensor-Based Beam Search

### Core Insight from PyTorch

PyTorch keeps everything as fixed-size tensors and uses GPU-native operations:

```python
# PyTorch approach (pseudocode)
running_sequences = torch.full((batch, num_beams, max_len), pad_token)
beam_scores = torch.zeros((batch, num_beams))
is_finished = torch.zeros((batch, num_beams), dtype=torch.bool)

# Selection via GPU topk - no CPU involved!
accumulated_scores = beam_scores[:, :, None] + log_probs  # [B, beams, vocab]
accumulated_scores = accumulated_scores.view(batch, num_beams * vocab_size)
topk_scores, topk_indices = torch.topk(accumulated_scores, k=beams_to_keep)

# Recover beam and token indices via integer math
beam_indices = topk_indices // vocab_size
token_indices = topk_indices % vocab_size
```

## Implementation Plan

### Phase 1: New GPU Operations

#### 1.1 GPU Top-K Operation

Create `include/vesper/ops/topk.h`:

```cpp
#pragma once
#include <vesper/core/tensor.h>

namespace vesper::ops {

/// GPU-accelerated top-k selection
/// Returns (values, indices) tensors
/// @param input Input tensor of shape [..., N]
/// @param k Number of top elements to select
/// @param dim Dimension along which to select (default: -1)
/// @param largest If true, return largest elements (default: true)
/// @param sorted If true, return in sorted order (default: true)
std::pair<Tensor, Tensor> topk(
    const Tensor& input,
    int64_t k,
    int64_t dim = -1,
    bool largest = true,
    bool sorted = true
);

// HIP/CUDA dispatch functions
std::pair<Tensor, Tensor> topk_hip_dispatch(
    const Tensor& input, int64_t k, int64_t dim, bool largest, bool sorted);
std::pair<Tensor, Tensor> topk_cuda_dispatch(
    const Tensor& input, int64_t k, int64_t dim, bool largest, bool sorted);

} // namespace vesper::ops
```

#### 1.2 GPU Top-K Kernel

Create `src/ops/hip/topk.hip`:

```cpp
#include <hip/hip_runtime.h>
#include <vesper/ops/topk.h>

namespace vesper::ops {

// Bitonic sort-based top-k for small k (k <= 32)
template<typename T>
__global__ void topk_bitonic_kernel(
    const T* __restrict__ input,
    T* __restrict__ values,
    int64_t* __restrict__ indices,
    int64_t n,
    int64_t k
) {
    extern __shared__ char shared_mem[];
    T* s_vals = reinterpret_cast<T*>(shared_mem);
    int64_t* s_idx = reinterpret_cast<int64_t*>(s_vals + blockDim.x);
    
    int tid = threadIdx.x;
    int batch = blockIdx.x;
    
    const T* batch_input = input + batch * n;
    
    // Load data into shared memory
    T val = (tid < n) ? batch_input[tid] : -INFINITY;
    int64_t idx = tid;
    
    // Parallel reduction to find top-k
    // ... (bitonic sort implementation)
    
    // Write top-k results
    if (tid < k) {
        values[batch * k + tid] = s_vals[tid];
        indices[batch * k + tid] = s_idx[tid];
    }
}

// Radix-based top-k for large k or large vocabularies
template<typename T>
__global__ void topk_radix_kernel(
    const T* __restrict__ input,
    T* __restrict__ values,
    int64_t* __restrict__ indices,
    int64_t n,
    int64_t k
) {
    // Radix selection algorithm:
    // 1. Find k-th largest element via radix partitioning
    // 2. Gather all elements >= k-th element
    // 3. Sort the k elements if needed
    // ... (implementation)
}

std::pair<Tensor, Tensor> topk_hip_dispatch(
    const Tensor& input,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted
) {
    // Normalize dimension
    int64_t ndim = input.ndim();
    if (dim < 0) dim += ndim;
    
    // For now, only support last dimension
    VESPER_CHECK(dim == ndim - 1, "topk currently only supports last dimension");
    
    int64_t n = input.shape()[dim];
    int64_t batch_size = input.numel() / n;
    
    // Allocate output tensors
    std::vector<int64_t> out_shape = input.shape();
    out_shape[dim] = k;
    
    Tensor values = vesper::empty(out_shape, input.dtype(), input.device());
    Tensor indices = vesper::empty(out_shape, DType::Int64, input.device());
    
    // Choose kernel based on k and n
    if (k <= 32 && n <= 4096) {
        // Use bitonic sort for small problems
        int threads = 256;
        size_t shared_size = threads * (sizeof(float) + sizeof(int64_t));
        
        hipLaunchKernelGGL(
            topk_bitonic_kernel<float>,
            dim3(batch_size), dim3(threads), shared_size, 0,
            input.data_ptr<float>(),
            values.data_ptr<float>(),
            indices.data_ptr<int64_t>(),
            n, k
        );
    } else {
        // Use radix selection for large problems
        // ... (launch radix kernel)
    }
    
    return {values, indices};
}

} // namespace vesper::ops
```

#### 1.3 Gather Along Dimension

Create `include/vesper/ops/gather.h`:

```cpp
#pragma once
#include <vesper/core/tensor.h>

namespace vesper::ops {

/// Gather values along a dimension using indices
/// Similar to torch.gather or torch.take_along_dim
Tensor gather(const Tensor& input, int64_t dim, const Tensor& indices);

/// Scatter values along a dimension using indices
Tensor scatter(const Tensor& input, int64_t dim, const Tensor& indices, const Tensor& src);

} // namespace vesper::ops
```

### Phase 2: Tensor-Based Beam State

#### 2.1 New BeamState Structure

```cpp
struct TensorBeamState {
    // All tensors on GPU, fixed size allocated upfront
    Tensor sequences;      // [batch, num_beams, max_length] Int32
    Tensor scores;         // [batch, num_beams] Float32
    Tensor beam_indices;   // [batch, num_beams, max_new_tokens] Int32 - tracks beam origins
    Tensor is_finished;    // [batch, num_beams] Int32 (0 or 1, no bool tensor)
    
    int64_t batch_size;
    int64_t num_beams;
    int64_t max_length;
    int64_t cur_length;
    
    static TensorBeamState create(
        int64_t batch_size,
        int64_t num_beams,
        int64_t prompt_len,
        int64_t max_new_tokens,
        const Tensor& prompt_ids,
        Device device
    ) {
        TensorBeamState state;
        state.batch_size = batch_size;
        state.num_beams = num_beams;
        state.max_length = prompt_len + max_new_tokens;
        state.cur_length = prompt_len;
        
        // Preallocate all tensors
        state.sequences = vesper::zeros(
            {batch_size, num_beams, state.max_length}, DType::Int32, device);
        state.scores = vesper::zeros({batch_size, num_beams}, DType::Float32, device);
        state.beam_indices = vesper::full(
            {batch_size, num_beams, max_new_tokens}, -1, DType::Int32, device);
        state.is_finished = vesper::zeros({batch_size, num_beams}, DType::Int32, device);
        
        // Initialize first beam with prompt, others with -inf score
        // state.sequences[:, 0, :prompt_len] = prompt_ids
        // state.scores[:, 1:] = -1e9
        
        return state;
    }
};
```

### Phase 3: GPU Beam Search Algorithm

#### 3.1 Main Loop (All GPU)

```cpp
Tensor BeamSearcher::search_gpu(const Tensor& prompt_ids, const BeamSearchParams& params) {
    auto state = TensorBeamState::create(
        batch_size, num_beams, prompt_len, params.max_new_tokens, prompt_ids, device);
    
    for (int64_t step = 0; step < params.max_new_tokens; ++step) {
        // 1. Forward pass (already on GPU)
        Tensor logits = model_->forward_with_cache(
            flatten_beams(state.sequences, state.cur_length), state.cur_length - 1);
        
        // 2. Compute log probabilities (GPU)
        Tensor log_probs = nn::functional::log_softmax(logits, -1);
        log_probs = unflatten_beams(log_probs, batch_size, num_beams);
        
        // 3. Accumulate scores: [B, beams, vocab]
        Tensor accumulated = state.scores.unsqueeze(-1) + log_probs;
        accumulated = accumulated.view({batch_size, num_beams * vocab_size});
        
        // 4. GPU top-k selection
        int64_t k = std::min(2 * num_beams, num_beams * vocab_size);
        auto [topk_scores, topk_indices] = ops::topk(accumulated, k);
        
        // 5. Recover beam and token indices (GPU elementwise ops)
        Tensor beam_idx = topk_indices / vocab_size;  // integer division
        Tensor token_idx = topk_indices % vocab_size; // modulo
        
        // 6. Check for EOS tokens (GPU kernel)
        Tensor hits_eos = check_eos_tokens(token_idx, params.eos_token_ids);
        
        // 7. Update finished beams (GPU masking)
        // Mask finished beams with -1e9 so they don't get selected again
        Tensor masked_scores = topk_scores + state.is_finished.gather(1, beam_idx) * -1e9f;
        
        // 8. Select top num_beams for next iteration (GPU topk again)
        auto [next_scores, select_idx] = ops::topk(masked_scores, num_beams);
        
        // 9. Gather selected beams (GPU gather operations)
        state.sequences = gather_beams(state.sequences, beam_idx, select_idx);
        state.sequences[..., state.cur_length] = gather(token_idx, select_idx);
        state.scores = next_scores;
        state.is_finished = update_finished(state.is_finished, hits_eos, select_idx);
        
        // 10. Check termination (single GPU->CPU sync)
        if (ops::all_true(state.is_finished)) break;
        
        // 11. Reorder KV cache for selected beams
        reorder_kv_cache(beam_idx, select_idx);
        
        ++state.cur_length;
    }
    
    // Return best sequences
    return select_best_sequences(state);
}
```

### Phase 4: Supporting Operations

#### 4.1 EOS Checking Kernel

```cpp
// GPU kernel to check if tokens match any EOS token
__global__ void check_eos_kernel(
    const int32_t* tokens,       // [batch * k]
    const int64_t* eos_ids,      // [num_eos]
    int32_t* hits_eos,           // [batch * k] output
    int64_t num_tokens,
    int64_t num_eos
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tokens) return;
    
    int32_t token = tokens[idx];
    int32_t is_eos = 0;
    
    for (int64_t e = 0; e < num_eos; ++e) {
        if (token == eos_ids[e]) {
            is_eos = 1;
            break;
        }
    }
    
    hits_eos[idx] = is_eos;
}
```

#### 4.2 Beam Gathering Kernel

```cpp
// Reorder sequences based on beam selection
__global__ void gather_beams_kernel(
    const int32_t* src_sequences,  // [batch, beams, seq_len]
    const int64_t* beam_indices,   // [batch, k]
    const int64_t* select_indices, // [batch, num_beams]
    int32_t* dst_sequences,        // [batch, num_beams, seq_len]
    int64_t batch_size,
    int64_t num_beams,
    int64_t k,
    int64_t seq_len
) {
    int b = blockIdx.x;
    int beam = blockIdx.y;
    int pos = threadIdx.x;
    
    if (b >= batch_size || beam >= num_beams || pos >= seq_len) return;
    
    // Which of the top-k candidates was selected for this beam?
    int64_t sel = select_indices[b * num_beams + beam];
    // Which original beam did that candidate come from?
    int64_t src_beam = beam_indices[b * k + sel];
    
    // Copy sequence from source beam
    dst_sequences[(b * num_beams + beam) * seq_len + pos] =
        src_sequences[(b * num_beams + src_beam) * seq_len + pos];
}
```

#### 4.3 KV Cache Reordering

```cpp
// Critical: reorder KV cache to match beam selection
void reorder_kv_cache(
    KVCache& cache,
    const Tensor& beam_indices,   // [batch, k]
    const Tensor& select_indices  // [batch, num_beams]
) {
    // For each layer's K and V:
    // new_k[:, beam, :, :] = old_k[:, source_beam, :, :]
    // This requires a gather operation along the beam dimension
    
    for (auto& layer_cache : cache.layers()) {
        layer_cache.keys = ops::gather(layer_cache.keys, /*dim=*/1, final_beam_indices);
        layer_cache.values = ops::gather(layer_cache.values, /*dim=*/1, final_beam_indices);
    }
}
```

## Performance Comparison

### Expected Improvements

| Metric | CPU-Based | GPU-Native | Improvement |
|--------|-----------|------------|-------------|
| Selection latency | ~100μs | ~10μs | 10x |
| Memory transfers/step | 2 (logits + tokens) | 0 | Eliminated |
| Synchronization points | 1/step | 1/search | N× fewer |
| Batch scalability | Limited | Linear | Better |

### When CPU Is Still Acceptable

The current CPU implementation is adequate when:
- `num_beams` is small (≤4)
- `batch_size` is 1
- Latency is dominated by model forward pass
- Simplicity is prioritized over peak performance

## Testing Strategy

### Unit Tests

```cpp
TEST(GPUBeamSearch, TopKCorrectness) {
    // Compare GPU topk with CPU reference
    Tensor input = vesper::randn({4, 32000}, DType::Float32, Device::HIP);
    auto [gpu_vals, gpu_idx] = ops::topk(input, 10);
    
    // CPU reference
    Tensor cpu_input = input.to(Device::CPU);
    auto [cpu_vals, cpu_idx] = topk_cpu_reference(cpu_input, 10);
    
    EXPECT_TRUE(allclose(gpu_vals.to(Device::CPU), cpu_vals, 1e-5));
}

TEST(GPUBeamSearch, BeamGatherCorrectness) {
    // Verify beam reordering produces correct sequences
}

TEST(GPUBeamSearch, EndToEndEquivalence) {
    // Compare GPU beam search output with CPU implementation
    // Should produce identical or near-identical sequences
}
```

### Benchmarks

```cpp
BENCHMARK(BeamSearch_CPU_B1_Beams4);
BENCHMARK(BeamSearch_GPU_B1_Beams4);
BENCHMARK(BeamSearch_CPU_B8_Beams4);
BENCHMARK(BeamSearch_GPU_B8_Beams4);
BENCHMARK(BeamSearch_GPU_B32_Beams8);  // Only feasible with GPU
```

## Migration Path

### Step 1: Add GPU Operations (Non-Breaking)
- Implement `ops::topk()`
- Implement `ops::gather()` 
- Add unit tests
- **Keep existing beam search unchanged**

### Step 2: Create Parallel Implementation
- Add `BeamSearcher::search_gpu()` as new method
- Keep `BeamSearcher::search()` as default
- Add flag to select implementation

### Step 3: Validate Equivalence
- Run both implementations on same inputs
- Verify output sequences match
- Profile performance difference

### Step 4: Gradual Rollout
- Make GPU version default when available
- Deprecate CPU version for GPU devices
- Keep CPU version for CPU-only builds

## Summary

The upgrade from CPU-based to GPU-native beam search requires:

1. **New GPU Kernels**: `topk`, `gather`, `scatter`, EOS checking
2. **Tensor-Based State**: Replace `std::vector` with preallocated tensors
3. **Score Masking**: Replace conditionals with `-1e9` masking
4. **KV Cache Reordering**: GPU-based cache manipulation

This is a significant but well-defined refactor. The current CPU implementation remains correct and should be kept as a fallback. The GPU implementation should be added as an optimization for high-throughput inference scenarios.

## References

- [HuggingFace Transformers Generation Utils](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py)
- [Speculative Decoding Paper](https://arxiv.org/abs/2211.17192)
- [CUDA Top-K Algorithms Survey](https://developer.nvidia.com/blog/fast-top-k-selection-on-gpu/)
- Wu et al., "Google's Neural Machine Translation System" (Length Penalty)
