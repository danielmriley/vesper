# Training Optimization Ideas for Vesper

This document outlines potential optimization strategies to further accelerate training of large transformer models in Vesper. These ideas range from relatively simple changes to more complex architectural modifications.

## Current Baseline

- **Model**: 271M parameter TransformerLM (dim=1024, 16 layers, 16 heads)
- **Hardware**: AMD RX 6950 XT (16GB VRAM, gfx1030)
- **Current Performance**: ~0.5 steps/s, ~267 tok/s (batch_size=1, seq_len=512)
- **Flash Attention**: Tiled implementation with BLOCK_M=64, BLOCK_N=64

---

## 1. Mixed Precision Training (FP16/BF16)

**Priority: HIGH | Estimated Speedup: 1.5-2x | Complexity: MEDIUM**

### Description
Train with half-precision (FP16 or BF16) weights and activations while maintaining FP32 master weights for stable optimizer updates.

### Benefits
- **Memory Reduction**: 2x reduction in model memory footprint
- **Bandwidth Reduction**: Half the memory traffic for tensor operations
- **Batch Size**: Can double or triple batch size with same VRAM
- **Compute**: AMD's matrix cores are optimized for FP16 operations

### Implementation Approach
1. Cast model weights to FP16 after initialization
2. Keep FP32 master weight copies for optimizer updates
3. Use `GradScaler` for loss scaling to prevent gradient underflow
4. Cast inputs to FP16 before forward pass
5. Cast gradients to FP32 before optimizer step, then copy back to FP16

### Code Sketch
```cpp
// Initialization
std::vector<Tensor> master_weights;  // FP32 copies
for (auto& param : model->parameters()) {
    master_weights.push_back(param.clone());  // Keep FP32 copy
    param = ops::cast(param, DType::Float16); // Model in FP16
}
GradScaler scaler;

// Training step
Tensor input_fp16 = ops::cast(input, DType::Float16);
Tensor loss = model->compute_loss(input_fp16, target);
Tensor scaled_loss = scaler.scale(loss);
scaled_loss.backward();

// Update master weights
for (size_t i = 0; i < master_weights.size(); ++i) {
    Tensor grad_fp32 = ops::cast(model_params[i].grad(), DType::Float32);
    master_weights[i].grad() = grad_fp32;
}
optimizer.step();  // Updates master_weights

// Copy back to FP16 model
for (size_t i = 0; i < master_weights.size(); ++i) {
    model_params[i] = ops::cast(master_weights[i], DType::Float16);
}
```

### Considerations
- FP16 has limited dynamic range (±65504), may need loss scaling
- BF16 has same dynamic range as FP32, better for training stability
- RX 6950 XT (RDNA2) may have better FP16 than BF16 support

---

## 2. Gradient Accumulation with Async Data Prefetch

**Priority: HIGH | Estimated Speedup: 1.2-1.5x | Complexity: LOW**

### Description
Accumulate gradients across multiple micro-batches before optimizer step, while overlapping data loading with computation.

### Benefits
- **Effective Batch Size**: Simulate larger batches without more VRAM
- **Better Convergence**: Larger effective batch sizes often converge better
- **GPU Utilization**: Overlap CPU data loading with GPU compute
- **Throughput**: Eliminate data loading stalls

### Implementation Approach
1. Create a background thread for data prefetching
2. Use double buffering: load next batch while processing current
3. Accumulate gradients for N micro-batches before step
4. Scale loss by 1/N for correct gradient magnitudes

### Code Sketch
```cpp
int accumulation_steps = 4;
int64_t effective_batch_size = batch_size * accumulation_steps;

// Prefetch next batch in background
std::future<data::Sample> next_batch = 
    std::async(std::launch::async, [&]() { return loader.next(); });

for (int micro_step = 0; micro_step < accumulation_steps; ++micro_step) {
    auto batch = next_batch.get();
    next_batch = std::async(std::launch::async, [&]() { return loader.next(); });
    
    Tensor loss = model->compute_loss(batch.input, batch.target);
    Tensor scaled_loss = loss / accumulation_steps;  // Scale for accumulation
    scaled_loss.backward();
}

optimizer.step();
optimizer.zero_grad();
```

### Considerations
- Requires thread-safe data loading
- May need HIP streams for true async overlap
- Learning rate should scale with effective batch size

---

## 3. Fused Kernels

**Priority: MEDIUM | Estimated Speedup: 1.2-1.4x | Complexity: MEDIUM-HIGH**

### Description
Combine multiple sequential operations into single GPU kernels to reduce memory traffic and kernel launch overhead.

### Opportunities

#### 3.1 Fused LayerNorm + Linear
Combine layer normalization with the following linear projection.
```
y = W @ LayerNorm(x) + b
```
This saves one full read/write of the tensor between operations.

#### 3.2 Fused SwiGLU
The SwiGLU activation used in the FFN:
```
SwiGLU(x) = (x @ W1) * sigmoid(x @ W1) * (x @ W3)
```
Can be computed with fused elementwise operations after the matmuls.

#### 3.3 Fused Softmax + Dropout
Combine softmax computation with dropout mask application in attention.

#### 3.4 Fused RoPE + Attention QK
Apply rotary position embeddings during Q and K computation rather than as separate operation.

### Implementation Notes
- Each fused kernel reduces global memory round-trips
- Need to maintain backward pass compatibility
- Consider using templated kernels for different configurations

---

## 4. Chunked/Streaming Training

**Priority: LOW | Estimated Speedup: N/A (enables larger sequences) | Complexity: HIGH**

### Description
Process sequences in chunks with hidden state carry-over between chunks, enabling training on longer contexts than fit in VRAM.

### Benefits
- **Long Context**: Train on 4K, 8K, or longer sequences without OOM
- **Memory Constant**: Memory usage depends on chunk size, not total sequence length
- **Better Language Modeling**: Longer context improves coherence

### Implementation Approach
1. Split long sequences into chunks (e.g., 512 tokens each)
2. Process chunks sequentially, carrying KV cache state between chunks
3. Detach hidden states at chunk boundaries to limit backward graph depth
4. Optionally use gradient checkpointing within chunks

### Code Sketch
```cpp
int chunk_size = 512;
int total_length = 4096;

Tensor kv_cache = nullptr;  // Carried between chunks

for (int start = 0; start < total_length; start += chunk_size) {
    Tensor chunk_input = input.slice(start, start + chunk_size);
    Tensor chunk_target = target.slice(start, start + chunk_size);
    
    auto [output, new_cache] = model->forward_with_cache(chunk_input, kv_cache);
    Tensor loss = compute_loss(output, chunk_target);
    loss.backward();
    
    kv_cache = new_cache.detach();  // Detach to limit graph
}
```

### Considerations
- Gradient flow is truncated at chunk boundaries
- Need careful handling of causal masking across chunks
- May need curriculum learning (short → long sequences)

---

## 5. Selective Recomputation (Gradient Checkpointing)

**Priority: MEDIUM | Estimated Speedup: Enables larger models | Complexity: MEDIUM**

### Description
Instead of storing all intermediate activations for backward pass, selectively recompute them during backward. Trade compute for memory.

### Benefits
- **Memory Savings**: O(√N) instead of O(N) activation memory for N layers
- **Larger Models**: Fit larger models in same VRAM
- **Larger Batches**: Use saved memory for bigger batches

### Implementation Approach
1. Mark certain layers/blocks as checkpointed
2. During forward, only store checkpoint boundaries
3. During backward, recompute from nearest checkpoint

### Code Sketch
```cpp
class CheckpointedTransformerBlock : public Module {
    Tensor forward(const Tensor& x) override {
        // During forward: just compute, don't save intermediates
        auto hidden = attention_block_->forward(x);
        hidden = ffn_block_->forward(hidden);
        return hidden;
    }
    
    Tensor backward(const Tensor& grad_output) override {
        // Recompute forward to get intermediates
        auto hidden = attention_block_->forward(saved_input_);
        // Now do backward with fresh intermediates
        // ...
    }
};
```

### Trade-offs
- Increases compute by up to 33% (recomputing forward)
- Significant memory savings for deep networks
- Sweet spot: checkpoint every 2-4 transformer blocks

---

## 6. Token Dropping / Importance Sampling

**Priority: LOW | Estimated Speedup: 1.5-2x | Complexity: HIGH**

### Description
Skip computation for tokens that contribute minimally to the training signal.

### Approaches

#### 6.1 Loss-Based Token Dropping
After computing loss, only backpropagate through tokens with high loss (informative tokens).

#### 6.2 Attention-Based Importance
Use attention patterns to identify important tokens for computation.

#### 6.3 Random Subset Training
Randomly sample a subset of sequence positions for loss computation.

### Code Sketch
```cpp
Tensor logits = model->forward(input);
Tensor per_token_loss = cross_entropy_unreduced(logits, target);

// Keep only top-k highest loss tokens
auto [values, indices] = per_token_loss.topk(k);
Tensor important_loss = values.mean();
important_loss.backward();
```

### Considerations
- May hurt convergence if too aggressive
- Need careful tuning of drop rate
- Can combine with curriculum (drop more as training progresses)

---

## 7. Sparse Attention Patterns

**Priority: LOW | Estimated Speedup: Variable | Complexity: HIGH**

### Description
Replace full O(N²) attention with sparse patterns that scale better.

### Patterns

#### 7.1 Sliding Window Attention
Each token attends only to local window of W tokens.
- Complexity: O(N × W)
- Good for: Local patterns, when global context isn't critical

#### 7.2 Strided Attention
Attend to every k-th token globally + local window.
- Complexity: O(N × (W + N/k))
- Good for: Long-range dependencies with local detail

#### 7.3 Block Sparse Attention
Divide sequence into blocks, attend within and between selected blocks.
- Complexity: O(N × B) where B = block size
- Good for: Structured documents

### Considerations
- Requires custom sparse attention kernels
- Pattern must be known at compile time for efficiency
- May hurt quality for tasks requiring global attention

---

## 8. Kernel Autotuning

**Priority: MEDIUM | Estimated Speedup: 1.1-1.3x | Complexity: LOW**

### Description
Automatically tune kernel parameters (block sizes, shared memory usage) for specific GPU and problem sizes.

### Approach
1. Define parameter search space for each kernel
2. Run micro-benchmarks on target hardware
3. Cache optimal parameters for common shapes
4. Use cached params at runtime

### Parameters to Tune
- Flash attention: BLOCK_M, BLOCK_N, number of warps
- GEMM: tile sizes, vector widths
- Elementwise: block size, elements per thread

---

## Recommended Priority Order

1. **Gradient Accumulation + Async Prefetch** - Low complexity, immediate benefit
2. **Mixed Precision (FP16)** - High impact, moderate complexity
3. **Kernel Autotuning** - Low complexity, steady gains
4. **Fused Kernels** - Medium complexity, good returns
5. **Gradient Checkpointing** - Enables larger models/batches
6. **Chunked Training** - If you need longer sequences
7. **Token Dropping** - Experimental, high potential
8. **Sparse Attention** - For very long sequences

---

## Hardware-Specific Notes (AMD RDNA2 - RX 6950 XT)

- **Matrix Cores**: RDNA2 doesn't have dedicated tensor cores like NVIDIA. FP16 compute is done on standard shader cores but still benefits from reduced bandwidth.
- **L2 Cache**: 96MB Infinity Cache helps with memory-bound operations. Keep tile sizes cache-friendly.
- **Memory Bandwidth**: 576 GB/s. Most ops are memory-bound, so reducing memory traffic (fused kernels, FP16) has outsized impact.
- **Wavefront Size**: 32 threads (vs NVIDIA's 32-warp). Ensure kernel occupancy.
- **LDS (Shared Memory)**: 64KB per workgroup. Flash attention tiles must fit.

---

## Estimated Combined Speedup

Conservative estimates if implementing multiple optimizations:

| Optimization | Standalone | Combined |
|--------------|------------|----------|
| Gradient Accumulation + Prefetch | 1.2-1.5x | 1.2x |
| Mixed Precision (FP16) | 1.5-2x | 1.4x |
| Fused Kernels | 1.2-1.4x | 1.2x |
| **Total** | - | **~2x** |

With all major optimizations, we could potentially achieve **~500-600 tok/s** (from current ~270 tok/s), reducing the ~28 hour ETA to **~12-14 hours**.
