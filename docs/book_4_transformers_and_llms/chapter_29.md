# Chapter 29: The `nn.Embedding` Layer

## 1. Introduction

The `Embedding` layer is the fundamental bridge between discrete symbolic data—such as words, token IDs, or categorical features—and the continuous vector space that neural networks operate in. Every language model, from simple RNNs to massive transformers like GPT-4 and LLaMA, begins with an embedding layer that converts token indices into dense learned representations.

Unlike most neural network layers, embedding is not a matrix multiplication—it's a **lookup table**. Given an integer index, we simply retrieve the corresponding row from a weight matrix. This simplicity belies the layer's importance: the quality of learned embeddings fundamentally determines how well a model captures semantic relationships.

### Why Embeddings Matter

Consider the alternative: one-hot encoding. For a vocabulary of 50,000 tokens, each token would be represented as a 50,000-dimensional sparse vector with a single 1. This is:
- **Memory inefficient**: Wastes storage on zeros
- **Computationally wasteful**: Matrix multiplies with sparse vectors
- **Semantically flat**: All tokens are equidistant (cosine similarity = 0)

Embeddings solve all three problems. A 512-dimensional embedding compresses 50,000 tokens into a dense space where "king" and "queen" are nearby, while "king" and "banana" are distant.

### Key Challenges

1. **GPU Memory Access Patterns**: Random index lookups create scattered memory access, which is inefficient on GPUs designed for coalesced sequential reads.
2. **Backward Pass Atomicity**: When the same token appears multiple times in a batch, gradients must be accumulated atomically to avoid race conditions.
3. **Large Vocabulary Scaling**: With vocabularies of 100K+ tokens, the embedding matrix alone can consume gigabytes of memory.

---

## 2. Mathematical Foundation

### 2.1 Forward Pass: The Gather Operation

Given:
- Vocabulary size $V$ (number of unique tokens)
- Embedding dimension $D$ (size of each vector)
- Weight matrix $W \in \mathbb{R}^{V \times D}$
- Input indices $I \in \mathbb{Z}^{B \times S}$ where $B$ is batch size and $S$ is sequence length

The forward pass computes:

$$E[b, s, :] = W[I[b, s], :]$$

In other words, for each position $(b, s)$ in the input, we look up the row of $W$ corresponding to the index stored at $I[b, s]$.

**Output Shape**: $(B, S, D)$

### 2.2 Backward Pass: Sparse Gradient Accumulation

The embedding layer has only one learnable parameter: the weight matrix $W$. The gradient computation is:

$$\frac{\partial L}{\partial W[i, :]} = \sum_{(b,s) : I[b,s] = i} \frac{\partial L}{\partial E[b, s, :]}$$

This means: for each row $i$ of the weight matrix, we sum up all the gradients from output positions where index $i$ was used.

**Key Insight**: This is a **scatter-add** operation—the inverse of the forward gather.

### 2.3 Padding Index Semantics

When processing variable-length sequences, we pad shorter sequences with a special token (often index 0). The `padding_idx` parameter ensures:

1. **Forward**: The output vector for `padding_idx` is always zeros (regardless of learned weights)
2. **Backward**: The gradient for `padding_idx` row is always zero (weights never update)

Mathematically:
$$E[b, s, :] = \begin{cases} \mathbf{0} & \text{if } I[b,s] = \text{padding\_idx} \\ W[I[b,s], :] & \text{otherwise} \end{cases}$$

### 2.4 Max Norm Constraint

Some applications benefit from constraining embedding vectors to have bounded norm. The `max_norm` parameter clips vectors after each update:

$$W[i, :] \leftarrow \begin{cases} W[i, :] \cdot \frac{\text{max\_norm}}{\|W[i, :]\|_2} & \text{if } \|W[i, :]\|_2 > \text{max\_norm} \\ W[i, :] & \text{otherwise} \end{cases}$$

This prevents embeddings from growing unboundedly during training, which can destabilize learning.

---

## 3. Implementation

### 3.1 Header Definition

```cpp
// include/vesper/nn/modules/embedding.h
#pragma once

#include "vesper/nn/module.h"
#include "vesper/core/tensor.h"

namespace vesper {
namespace nn {

/**
 * @brief Embedding layer: maps integer indices to dense vectors.
 * 
 * This is a lookup table that stores embeddings of a fixed vocabulary.
 * The input is a tensor of indices, and the output is the corresponding
 * embedding vectors.
 * 
 * @param num_embeddings Size of the vocabulary (number of unique tokens)
 * @param embedding_dim Dimension of each embedding vector
 * @param padding_idx If specified, entries at this index are zeroed and
 *                    don't contribute to gradients (default: -1 = disabled)
 * @param max_norm If specified, embeddings are renormalized to have norm
 *                 at most this value (default: -1.0 = disabled)
 * @param norm_type The p of the p-norm for max_norm (default: 2.0)
 * @param scale_grad_by_freq If true, scale gradients by inverse frequency
 *                           of the words in the mini-batch (default: false)
 * @param sparse If true, use sparse gradient updates (default: false)
 */
class Embedding : public Module {
public:
    Embedding(int64_t num_embeddings, 
              int64_t embedding_dim,
              int64_t padding_idx = -1,
              float max_norm = -1.0f,
              float norm_type = 2.0f,
              bool scale_grad_by_freq = false,
              bool sparse = false);
    
    /**
     * @brief Forward pass: lookup embeddings for input indices.
     * @param input Tensor of shape (*, ) containing indices in [0, num_embeddings)
     * @return Tensor of shape (*, embedding_dim)
     */
    Tensor forward(const Tensor& input) override;
    
    /**
     * @brief Initialize weights from a pre-trained embedding matrix.
     * @param embeddings Tensor of shape (num_embeddings, embedding_dim)
     * @param freeze If true, embeddings are not updated during training
     */
    void from_pretrained(const Tensor& embeddings, bool freeze = false);
    
    /**
     * @brief Reset embedding for padding_idx to zeros.
     */
    void reset_padding_idx();
    
    // Accessors
    int64_t num_embeddings() const { return num_embeddings_; }
    int64_t embedding_dim() const { return embedding_dim_; }
    int64_t padding_idx() const { return padding_idx_; }

    Tensor weight;  // Shape: [num_embeddings, embedding_dim]

private:
    int64_t num_embeddings_;
    int64_t embedding_dim_;
    int64_t padding_idx_;
    float max_norm_;
    float norm_type_;
    bool scale_grad_by_freq_;
    bool sparse_;
    
    void validate_indices(const Tensor& input) const;
    void apply_max_norm(const Tensor& indices);
};

}  // namespace nn
}  // namespace vesper
```

### 3.2 CPU Implementation

```cpp
// src/nn/modules/embedding.cpp
#include "vesper/nn/modules/embedding.h"
#include "vesper/core/error.h"
#include "vesper/ops/factory.h"
#include <cmath>

namespace vesper {
namespace nn {

Embedding::Embedding(int64_t num_embeddings,
                     int64_t embedding_dim,
                     int64_t padding_idx,
                     float max_norm,
                     float norm_type,
                     bool scale_grad_by_freq,
                     bool sparse)
    : num_embeddings_(num_embeddings),
      embedding_dim_(embedding_dim),
      padding_idx_(padding_idx),
      max_norm_(max_norm),
      norm_type_(norm_type),
      scale_grad_by_freq_(scale_grad_by_freq),
      sparse_(sparse) {
    
    VESPER_CHECK(num_embeddings > 0, 
                 "num_embeddings must be positive, got ", num_embeddings);
    VESPER_CHECK(embedding_dim > 0,
                 "embedding_dim must be positive, got ", embedding_dim);
    VESPER_CHECK(padding_idx < num_embeddings,
                 "padding_idx must be less than num_embeddings");
    
    // Initialize weights from N(0, 1)
    weight = ops::randn({num_embeddings, embedding_dim});
    register_parameter("weight", weight);
    
    // Zero out padding index if specified
    if (padding_idx_ >= 0) {
        reset_padding_idx();
    }
}

void Embedding::reset_padding_idx() {
    if (padding_idx_ >= 0) {
        // Zero the padding row
        auto weight_data = weight.data_ptr<float>();
        for (int64_t d = 0; d < embedding_dim_; ++d) {
            weight_data[padding_idx_ * embedding_dim_ + d] = 0.0f;
        }
    }
}

void Embedding::validate_indices(const Tensor& input) const {
    VESPER_CHECK(input.dtype() == DType::Int64 || input.dtype() == DType::Int32,
                 "Embedding indices must be integer type");
    
    // In debug mode, check all indices are in bounds
    #ifdef VESPER_DEBUG
    auto* data = input.data_ptr<int64_t>();
    int64_t numel = input.numel();
    for (int64_t i = 0; i < numel; ++i) {
        VESPER_CHECK(data[i] >= 0 && data[i] < num_embeddings_,
                     "Index ", data[i], " out of range [0, ", num_embeddings_, ")");
    }
    #endif
}

Tensor Embedding::forward(const Tensor& input) {
    validate_indices(input);
    
    // Compute output shape: input_shape + [embedding_dim]
    std::vector<int64_t> output_shape = input.shape();
    output_shape.push_back(embedding_dim_);
    
    // Flatten input for easier indexing
    int64_t num_indices = input.numel();
    Tensor flat_input = input.view({num_indices});
    
    // Allocate output
    Tensor output = ops::empty(output_shape, weight.dtype(), weight.device());
    Tensor flat_output = output.view({num_indices, embedding_dim_});
    
    if (input.device().is_cpu()) {
        // CPU implementation: simple gather
        auto* indices = flat_input.data_ptr<int64_t>();
        auto* weight_ptr = weight.data_ptr<float>();
        auto* out_ptr = flat_output.data_ptr<float>();
        
        for (int64_t i = 0; i < num_indices; ++i) {
            int64_t idx = indices[i];
            
            if (idx == padding_idx_) {
                // Zero output for padding
                for (int64_t d = 0; d < embedding_dim_; ++d) {
                    out_ptr[i * embedding_dim_ + d] = 0.0f;
                }
            } else {
                // Copy row from weight matrix
                const float* src = weight_ptr + idx * embedding_dim_;
                float* dst = out_ptr + i * embedding_dim_;
                std::memcpy(dst, src, embedding_dim_ * sizeof(float));
            }
        }
    } else {
        // GPU: dispatch to CUDA/HIP kernel
        ops::embedding_forward(flat_output, weight, flat_input, padding_idx_);
    }
    
    // Apply max_norm if specified
    if (max_norm_ > 0) {
        apply_max_norm(flat_input);
    }
    
    // Reshape to proper output shape (unflatten)
    return output;
}

void Embedding::apply_max_norm(const Tensor& indices) {
    // Renormalize the embeddings that were accessed
    auto* idx_ptr = indices.data_ptr<int64_t>();
    auto* weight_ptr = weight.data_ptr<float>();
    int64_t num_indices = indices.numel();
    
    for (int64_t i = 0; i < num_indices; ++i) {
        int64_t idx = idx_ptr[i];
        if (idx == padding_idx_) continue;
        
        // Compute norm
        float norm = 0.0f;
        for (int64_t d = 0; d < embedding_dim_; ++d) {
            float val = weight_ptr[idx * embedding_dim_ + d];
            norm += std::pow(std::abs(val), norm_type_);
        }
        norm = std::pow(norm, 1.0f / norm_type_);
        
        // Clip if necessary
        if (norm > max_norm_) {
            float scale = max_norm_ / norm;
            for (int64_t d = 0; d < embedding_dim_; ++d) {
                weight_ptr[idx * embedding_dim_ + d] *= scale;
            }
        }
    }
}

void Embedding::from_pretrained(const Tensor& embeddings, bool freeze) {
    VESPER_CHECK(embeddings.dim() == 2, "Pretrained embeddings must be 2D");
    VESPER_CHECK(embeddings.size(0) == num_embeddings_,
                 "Embedding size mismatch: expected ", num_embeddings_,
                 ", got ", embeddings.size(0));
    VESPER_CHECK(embeddings.size(1) == embedding_dim_,
                 "Embedding dim mismatch: expected ", embedding_dim_,
                 ", got ", embeddings.size(1));
    
    weight.copy_(embeddings);
    
    if (freeze) {
        weight.set_requires_grad(false);
    }
    
    if (padding_idx_ >= 0) {
        reset_padding_idx();
    }
}

}  // namespace nn
}  // namespace vesper
```

### 3.3 GPU Forward Kernel

The forward kernel must handle scattered memory access efficiently. We assign one thread per output element to maximize parallelism.

```cpp
// src/ops/kernels/embedding_kernels.hip
#include <hip/hip_runtime.h>
#include "vesper/core/macros.h"

namespace vesper {
namespace ops {
namespace kernels {

/**
 * @brief Embedding forward kernel: gather rows from weight matrix.
 * 
 * Each thread handles one element of the output tensor.
 * Thread (idx, d) computes output[idx, d] = weight[indices[idx], d]
 * 
 * Memory access pattern:
 * - indices: coalesced read (threads in a warp read consecutive indices)
 * - weight: scattered read (indices determine which rows to access)
 * - output: coalesced write (threads write to consecutive positions)
 */
template<typename T, typename IndexT>
__global__ void embedding_forward_kernel(
    T* __restrict__ output,              // [num_indices, embedding_dim]
    const T* __restrict__ weight,        // [num_embeddings, embedding_dim]
    const IndexT* __restrict__ indices,  // [num_indices]
    int64_t num_indices,
    int64_t embedding_dim,
    int64_t padding_idx
) {
    // 2D grid: x = element within embedding, y = which index
    int64_t d = blockIdx.x * blockDim.x + threadIdx.x;  // dimension
    int64_t i = blockIdx.y * blockDim.y + threadIdx.y;  // index position
    
    if (d >= embedding_dim || i >= num_indices) return;
    
    IndexT idx = indices[i];
    
    if (idx == static_cast<IndexT>(padding_idx)) {
        // Zero output for padding token
        output[i * embedding_dim + d] = T(0);
    } else {
        // Gather from weight matrix
        output[i * embedding_dim + d] = weight[idx * embedding_dim + d];
    }
}

/**
 * @brief Optimized embedding forward using vectorized loads.
 * 
 * Uses float4 to load 4 elements at once, reducing memory transactions.
 * Requires embedding_dim to be divisible by 4.
 */
template<typename T>
__global__ void embedding_forward_vectorized_kernel(
    T* __restrict__ output,
    const T* __restrict__ weight,
    const int64_t* __restrict__ indices,
    int64_t num_indices,
    int64_t embedding_dim,
    int64_t padding_idx
) {
    // Each thread handles 4 consecutive elements
    int64_t vec_dim = embedding_dim / 4;
    int64_t d = blockIdx.x * blockDim.x + threadIdx.x;  // vector position
    int64_t i = blockIdx.y;  // index position (one block per index)
    
    if (d >= vec_dim || i >= num_indices) return;
    
    int64_t idx = indices[i];
    
    float4* out_vec = reinterpret_cast<float4*>(output + i * embedding_dim);
    
    if (idx == padding_idx) {
        out_vec[d] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    } else {
        const float4* weight_vec = reinterpret_cast<const float4*>(
            weight + idx * embedding_dim);
        out_vec[d] = weight_vec[d];
    }
}

}  // namespace kernels
}  // namespace ops
}  // namespace vesper
```

### 3.4 GPU Backward Kernel with Atomic Operations

The backward pass requires accumulating gradients into the weight matrix. When the same index appears multiple times, we must use atomic operations to avoid race conditions.

```cpp
// src/ops/kernels/embedding_backward_kernels.hip
#include <hip/hip_runtime.h>

namespace vesper {
namespace ops {
namespace kernels {

/**
 * @brief Naive embedding backward using atomicAdd.
 * 
 * Each thread atomically adds its gradient to the appropriate weight row.
 * This is simple but can be slow for frequent tokens (many collisions).
 */
template<typename T, typename IndexT>
__global__ void embedding_backward_atomic_kernel(
    T* __restrict__ grad_weight,           // [num_embeddings, embedding_dim]
    const T* __restrict__ grad_output,     // [num_indices, embedding_dim]
    const IndexT* __restrict__ indices,    // [num_indices]
    int64_t num_indices,
    int64_t embedding_dim,
    int64_t padding_idx
) {
    int64_t d = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t i = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (d >= embedding_dim || i >= num_indices) return;
    
    IndexT idx = indices[i];
    
    // Skip padding index
    if (idx == static_cast<IndexT>(padding_idx)) return;
    
    T grad = grad_output[i * embedding_dim + d];
    atomicAdd(&grad_weight[idx * embedding_dim + d], grad);
}

/**
 * @brief Segmented embedding backward for better performance.
 * 
 * First sorts indices, then uses warp-level reduction for same-index gradients.
 * Reduces atomic contention when tokens repeat frequently.
 * 
 * This is a two-pass algorithm:
 * Pass 1: Sort (index, position) pairs by index
 * Pass 2: Reduce gradients within each segment
 */
template<typename T>
__global__ void embedding_backward_segmented_kernel(
    T* __restrict__ grad_weight,
    const T* __restrict__ grad_output,
    const int64_t* __restrict__ sorted_indices,    // sorted indices
    const int64_t* __restrict__ original_positions, // original positions
    const int64_t* __restrict__ segment_offsets,    // start of each segment
    int64_t num_segments,
    int64_t embedding_dim
) {
    // Each block handles one segment (one vocabulary index)
    int64_t seg = blockIdx.y;
    if (seg >= num_segments) return;
    
    int64_t seg_start = segment_offsets[seg];
    int64_t seg_end = segment_offsets[seg + 1];
    int64_t vocab_idx = sorted_indices[seg_start];
    
    // Each thread handles one dimension
    int64_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= embedding_dim) return;
    
    // Sum gradients for this dimension across all occurrences
    T sum = T(0);
    for (int64_t p = seg_start; p < seg_end; ++p) {
        int64_t orig_pos = original_positions[p];
        sum += grad_output[orig_pos * embedding_dim + d];
    }
    
    // Single atomic write per segment
    atomicAdd(&grad_weight[vocab_idx * embedding_dim + d], sum);
}

/**
 * @brief Compute word frequencies for gradient scaling.
 * 
 * Used when scale_grad_by_freq=true to scale gradients by 1/frequency.
 */
template<typename IndexT>
__global__ void compute_frequencies_kernel(
    int64_t* __restrict__ frequencies,     // [num_embeddings]
    const IndexT* __restrict__ indices,    // [num_indices]
    int64_t num_indices
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_indices) return;
    
    IndexT idx = indices[i];
    atomicAdd(reinterpret_cast<unsigned long long*>(&frequencies[idx]), 1ULL);
}

/**
 * @brief Scale gradients by inverse frequency.
 */
template<typename T>
__global__ void scale_grad_by_freq_kernel(
    T* __restrict__ grad_weight,
    const int64_t* __restrict__ frequencies,
    int64_t num_embeddings,
    int64_t embedding_dim
) {
    int64_t idx = blockIdx.y;
    int64_t d = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_embeddings || d >= embedding_dim) return;
    
    int64_t freq = frequencies[idx];
    if (freq > 0) {
        grad_weight[idx * embedding_dim + d] /= static_cast<T>(freq);
    }
}

}  // namespace kernels
}  // namespace ops
}  // namespace vesper
```

### 3.5 Autograd Function

```cpp
// src/autograd/functions/embedding_function.cpp
#include "vesper/autograd/function.h"
#include "vesper/ops/embedding_ops.h"

namespace vesper {
namespace autograd {

class EmbeddingFunction : public Function {
public:
    static Tensor forward(AutogradContext& ctx,
                          const Tensor& weight,
                          const Tensor& indices,
                          int64_t padding_idx,
                          bool scale_grad_by_freq) {
        // Save for backward
        ctx.save_for_backward({indices});
        ctx.saved_data["num_embeddings"] = weight.size(0);
        ctx.saved_data["embedding_dim"] = weight.size(1);
        ctx.saved_data["padding_idx"] = padding_idx;
        ctx.saved_data["scale_grad_by_freq"] = scale_grad_by_freq;
        
        return ops::embedding_forward(weight, indices, padding_idx);
    }
    
    static std::vector<Tensor> backward(AutogradContext& ctx,
                                        const std::vector<Tensor>& grad_outputs) {
        auto saved = ctx.get_saved_tensors();
        Tensor indices = saved[0];
        
        int64_t num_embeddings = ctx.saved_data["num_embeddings"].toInt();
        int64_t embedding_dim = ctx.saved_data["embedding_dim"].toInt();
        int64_t padding_idx = ctx.saved_data["padding_idx"].toInt();
        bool scale_grad_by_freq = ctx.saved_data["scale_grad_by_freq"].toBool();
        
        Tensor grad_weight = ops::zeros({num_embeddings, embedding_dim},
                                        grad_outputs[0].dtype(),
                                        grad_outputs[0].device());
        
        ops::embedding_backward(grad_weight, 
                                grad_outputs[0], 
                                indices,
                                padding_idx,
                                scale_grad_by_freq);
        
        // No gradient for indices (discrete values)
        return {grad_weight, Tensor()};
    }
};

}  // namespace autograd
}  // namespace vesper
```

### 3.6 Kernel Dispatch

```cpp
// src/ops/embedding_ops.cpp
#include "vesper/ops/embedding_ops.h"
#include "vesper/ops/kernels/embedding_kernels.h"

namespace vesper {
namespace ops {

Tensor embedding_forward(const Tensor& weight,
                         const Tensor& indices,
                         int64_t padding_idx) {
    int64_t num_indices = indices.numel();
    int64_t embedding_dim = weight.size(1);
    
    // Compute output shape
    std::vector<int64_t> output_shape = indices.shape();
    output_shape.push_back(embedding_dim);
    
    Tensor output = empty(output_shape, weight.dtype(), weight.device());
    
    if (weight.device().is_cuda()) {
        // Choose between vectorized and scalar kernel
        if (embedding_dim % 4 == 0 && embedding_dim >= 64) {
            // Vectorized kernel: one block per index
            int64_t vec_dim = embedding_dim / 4;
            dim3 block(256);
            dim3 grid((vec_dim + block.x - 1) / block.x, num_indices);
            
            kernels::embedding_forward_vectorized_kernel<<<grid, block>>>(
                output.data_ptr<float>(),
                weight.data_ptr<float>(),
                indices.data_ptr<int64_t>(),
                num_indices,
                embedding_dim,
                padding_idx
            );
        } else {
            // Scalar kernel
            dim3 block(16, 16);
            dim3 grid(
                (embedding_dim + block.x - 1) / block.x,
                (num_indices + block.y - 1) / block.y
            );
            
            kernels::embedding_forward_kernel<<<grid, block>>>(
                output.data_ptr<float>(),
                weight.data_ptr<float>(),
                indices.data_ptr<int64_t>(),
                num_indices,
                embedding_dim,
                padding_idx
            );
        }
        
        HIP_CHECK(hipGetLastError());
    }
    
    return output;
}

void embedding_backward(Tensor& grad_weight,
                        const Tensor& grad_output,
                        const Tensor& indices,
                        int64_t padding_idx,
                        bool scale_grad_by_freq) {
    int64_t num_indices = indices.numel();
    int64_t embedding_dim = grad_weight.size(1);
    int64_t num_embeddings = grad_weight.size(0);
    
    if (grad_weight.device().is_cuda()) {
        dim3 block(16, 16);
        dim3 grid(
            (embedding_dim + block.x - 1) / block.x,
            (num_indices + block.y - 1) / block.y
        );
        
        kernels::embedding_backward_atomic_kernel<<<grid, block>>>(
            grad_weight.data_ptr<float>(),
            grad_output.view({num_indices, embedding_dim}).data_ptr<float>(),
            indices.view({num_indices}).data_ptr<int64_t>(),
            num_indices,
            embedding_dim,
            padding_idx
        );
        
        if (scale_grad_by_freq) {
            // Compute frequencies
            Tensor frequencies = zeros({num_embeddings}, DType::Int64, 
                                        grad_weight.device());
            
            int threads = 256;
            int blocks = (num_indices + threads - 1) / threads;
            
            kernels::compute_frequencies_kernel<<<blocks, threads>>>(
                frequencies.data_ptr<int64_t>(),
                indices.data_ptr<int64_t>(),
                num_indices
            );
            
            // Scale gradients
            dim3 scale_block(256);
            dim3 scale_grid(
                (embedding_dim + scale_block.x - 1) / scale_block.x,
                num_embeddings
            );
            
            kernels::scale_grad_by_freq_kernel<<<scale_grid, scale_block>>>(
                grad_weight.data_ptr<float>(),
                frequencies.data_ptr<int64_t>(),
                num_embeddings,
                embedding_dim
            );
        }
        
        HIP_CHECK(hipGetLastError());
    }
}

}  // namespace ops
}  // namespace vesper
```

---

## 4. Usage Examples

### 4.1 Basic Usage

```cpp
#include "vesper/nn/modules/embedding.h"

using namespace vesper;

// Create embedding layer
// Vocabulary of 10000 tokens, embedding dimension 512
auto embedding = nn::Embedding(10000, 512);

// Input: batch of 4 sequences, each of length 128
Tensor input = ops::randint(0, 10000, {4, 128}, DType::Int64);

// Forward pass
Tensor output = embedding.forward(input);
// output shape: [4, 128, 512]

std::cout << "Output shape: " << output.shape() << std::endl;
```

### 4.2 With Padding Index

```cpp
// Create embedding with padding at index 0
auto embedding = nn::Embedding(10000, 512, /*padding_idx=*/0);

// Create input with some padding
Tensor input = ops::tensor({
    {1, 5, 0, 0},   // Sequence 1: tokens 1, 5, then padding
    {3, 7, 2, 0}    // Sequence 2: tokens 3, 7, 2, then padding
}, DType::Int64);

Tensor output = embedding.forward(input);
// output[0, 2, :] and output[0, 3, :] will be all zeros
// output[1, 3, :] will be all zeros
```

### 4.3 Loading Pre-trained Embeddings

```cpp
// Load GloVe or Word2Vec embeddings
Tensor pretrained = load_glove_embeddings("glove.6B.300d.txt");
// pretrained shape: [400000, 300]

auto embedding = nn::Embedding(400000, 300);
embedding.from_pretrained(pretrained, /*freeze=*/true);  // Don't fine-tune

// Now use in model
Tensor tokens = tokenize("the quick brown fox");
Tensor vectors = embedding.forward(tokens);
```

### 4.4 With Max Norm Constraint

```cpp
// Constrain embeddings to have L2 norm <= 1.0
auto embedding = nn::Embedding(10000, 512, 
                               /*padding_idx=*/-1,
                               /*max_norm=*/1.0f);

// During training, embeddings that exceed norm 1.0 will be renormalized
```

### 4.5 Integration with Transformer

```cpp
class TransformerLM : public nn::Module {
public:
    TransformerLM(int64_t vocab_size, int64_t d_model, int64_t n_heads, 
                  int64_t n_layers) {
        // Token embeddings
        token_emb_ = register_module("token_emb", 
            nn::Embedding(vocab_size, d_model, /*padding_idx=*/0));
        
        // Position embeddings (learned)
        pos_emb_ = register_module("pos_emb",
            nn::Embedding(2048, d_model));  // Max sequence length 2048
        
        // Transformer layers...
    }
    
    Tensor forward(const Tensor& tokens) {
        int64_t seq_len = tokens.size(1);
        
        // Get token embeddings
        Tensor tok_emb = token_emb_->forward(tokens);
        
        // Get position embeddings
        Tensor positions = ops::arange(0, seq_len, DType::Int64);
        Tensor pos_emb = pos_emb_->forward(positions);
        
        // Combine
        Tensor x = tok_emb + pos_emb;
        
        // Continue with transformer layers...
        return x;
    }
    
private:
    std::shared_ptr<nn::Embedding> token_emb_;
    std::shared_ptr<nn::Embedding> pos_emb_;
};
```

---

## 5. Testing Strategy

### 5.1 Unit Tests

```cpp
// tests/nn/test_embedding.cpp
#include <gtest/gtest.h>
#include "vesper/nn/modules/embedding.h"
#include "vesper/ops/factory.h"

class EmbeddingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Seed for reproducibility
        ops::manual_seed(42);
    }
};

TEST_F(EmbeddingTest, BasicForwardShape) {
    auto embed = nn::Embedding(100, 32);
    Tensor input = ops::randint(0, 100, {4, 16}, DType::Int64);
    
    Tensor output = embed.forward(input);
    
    EXPECT_EQ(output.dim(), 3);
    EXPECT_EQ(output.size(0), 4);
    EXPECT_EQ(output.size(1), 16);
    EXPECT_EQ(output.size(2), 32);
}

TEST_F(EmbeddingTest, GatherCorrectness) {
    auto embed = nn::Embedding(10, 4);
    
    // Set known weights
    Tensor weight = ops::tensor({
        {1.0f, 0.0f, 0.0f, 0.0f},  // index 0
        {0.0f, 1.0f, 0.0f, 0.0f},  // index 1
        {0.0f, 0.0f, 1.0f, 0.0f},  // index 2
        {0.0f, 0.0f, 0.0f, 1.0f},  // index 3
        {1.0f, 1.0f, 0.0f, 0.0f},  // index 4
        {0.0f, 1.0f, 1.0f, 0.0f},  // index 5
        {0.0f, 0.0f, 1.0f, 1.0f},  // index 6
        {1.0f, 0.0f, 0.0f, 1.0f},  // index 7
        {1.0f, 1.0f, 1.0f, 0.0f},  // index 8
        {0.0f, 1.0f, 1.0f, 1.0f},  // index 9
    });
    embed.weight.copy_(weight);
    
    Tensor input = ops::tensor({{2, 5, 0}}, DType::Int64);
    Tensor output = embed.forward(input);
    
    // Verify gather operation
    EXPECT_FLOAT_EQ(output[0][0][0].item<float>(), 0.0f);  // weight[2,0]
    EXPECT_FLOAT_EQ(output[0][0][2].item<float>(), 1.0f);  // weight[2,2]
    EXPECT_FLOAT_EQ(output[0][1][1].item<float>(), 1.0f);  // weight[5,1]
    EXPECT_FLOAT_EQ(output[0][2][0].item<float>(), 1.0f);  // weight[0,0]
}

TEST_F(EmbeddingTest, PaddingIndexZeros) {
    auto embed = nn::Embedding(100, 32, /*padding_idx=*/0);
    
    Tensor input = ops::tensor({{0, 5, 0, 10}}, DType::Int64);
    Tensor output = embed.forward(input);
    
    // Padding index outputs should be zeros
    Tensor zeros = ops::zeros({32});
    EXPECT_TRUE(ops::allclose(output[0][0], zeros));
    EXPECT_TRUE(ops::allclose(output[0][2], zeros));
    
    // Non-padding should not be zeros
    EXPECT_FALSE(ops::allclose(output[0][1], zeros));
    EXPECT_FALSE(ops::allclose(output[0][3], zeros));
}

TEST_F(EmbeddingTest, GradientAccumulation) {
    auto embed = nn::Embedding(10, 4);
    embed.weight.set_requires_grad(true);
    
    // Input with repeated index
    Tensor input = ops::tensor({{2, 2, 2}}, DType::Int64);  // index 2 appears 3 times
    
    Tensor output = embed.forward(input);
    Tensor loss = output.sum();
    loss.backward();
    
    // Gradient at index 2 should be 3x the gradient at other indices
    Tensor grad = embed.weight.grad();
    
    // Each occurrence adds 1 to all dimensions, so gradient[2] = [3, 3, 3, 3]
    for (int d = 0; d < 4; ++d) {
        EXPECT_FLOAT_EQ(grad[2][d].item<float>(), 3.0f);
    }
    
    // Other indices should have zero gradient
    for (int d = 0; d < 4; ++d) {
        EXPECT_FLOAT_EQ(grad[0][d].item<float>(), 0.0f);
        EXPECT_FLOAT_EQ(grad[5][d].item<float>(), 0.0f);
    }
}

TEST_F(EmbeddingTest, PaddingGradientIsZero) {
    auto embed = nn::Embedding(10, 4, /*padding_idx=*/0);
    embed.weight.set_requires_grad(true);
    
    // Include padding index
    Tensor input = ops::tensor({{0, 1, 0}}, DType::Int64);
    
    Tensor output = embed.forward(input);
    Tensor loss = output.sum();
    loss.backward();
    
    // Gradient at padding index should be zero
    Tensor grad = embed.weight.grad();
    for (int d = 0; d < 4; ++d) {
        EXPECT_FLOAT_EQ(grad[0][d].item<float>(), 0.0f);
    }
}

TEST_F(EmbeddingTest, MaxNormConstraint) {
    auto embed = nn::Embedding(10, 4, -1, /*max_norm=*/1.0f);
    
    // Set a weight with large norm
    embed.weight[5].fill_(10.0f);  // norm = sqrt(4 * 100) = 20
    
    Tensor input = ops::tensor({{5}}, DType::Int64);
    embed.forward(input);  // Triggers renormalization
    
    // Check norm is now <= 1.0
    float norm = embed.weight[5].norm().item<float>();
    EXPECT_LE(norm, 1.0f + 1e-5f);
}

TEST_F(EmbeddingTest, FromPretrained) {
    Tensor pretrained = ops::randn({100, 50});
    
    auto embed = nn::Embedding(100, 50);
    embed.from_pretrained(pretrained, /*freeze=*/true);
    
    EXPECT_TRUE(ops::allclose(embed.weight, pretrained));
    EXPECT_FALSE(embed.weight.requires_grad());
}

TEST_F(EmbeddingTest, LargeVocabulary) {
    // Test with vocabulary similar to GPT-2 (50257 tokens)
    auto embed = nn::Embedding(50257, 768);
    
    Tensor input = ops::randint(0, 50257, {8, 1024}, DType::Int64);
    Tensor output = embed.forward(input);
    
    EXPECT_EQ(output.size(0), 8);
    EXPECT_EQ(output.size(1), 1024);
    EXPECT_EQ(output.size(2), 768);
}
```

### 5.2 GPU-Specific Tests

```cpp
TEST_F(EmbeddingTest, GPUForwardMatchesCPU) {
    auto embed_cpu = nn::Embedding(1000, 128);
    auto embed_gpu = nn::Embedding(1000, 128);
    embed_gpu.to(Device::CUDA(0));
    
    // Copy weights
    embed_gpu.weight.copy_(embed_cpu.weight);
    
    Tensor input = ops::randint(0, 1000, {32, 64}, DType::Int64);
    
    Tensor out_cpu = embed_cpu.forward(input);
    Tensor out_gpu = embed_gpu.forward(input.to(Device::CUDA(0)));
    
    EXPECT_TRUE(ops::allclose(out_cpu, out_gpu.to(Device::CPU()), 1e-5, 1e-5));
}

TEST_F(EmbeddingTest, GPUGradientMatchesCPU) {
    auto embed_cpu = nn::Embedding(1000, 128);
    auto embed_gpu = nn::Embedding(1000, 128);
    embed_cpu.weight.set_requires_grad(true);
    embed_gpu.weight.set_requires_grad(true);
    embed_gpu.to(Device::CUDA(0));
    embed_gpu.weight.copy_(embed_cpu.weight);
    
    Tensor input = ops::randint(0, 1000, {32, 64}, DType::Int64);
    
    // Forward + backward on CPU
    Tensor out_cpu = embed_cpu.forward(input);
    out_cpu.sum().backward();
    
    // Forward + backward on GPU
    Tensor out_gpu = embed_gpu.forward(input.to(Device::CUDA(0)));
    out_gpu.sum().backward();
    
    // Compare gradients
    EXPECT_TRUE(ops::allclose(
        embed_cpu.weight.grad(),
        embed_gpu.weight.grad().to(Device::CPU()),
        1e-5, 1e-5
    ));
}

TEST_F(EmbeddingTest, AtomicAddCorrectness) {
    // Test case with many repeated indices to stress atomic adds
    auto embed = nn::Embedding(10, 128);
    embed.weight.set_requires_grad(true);
    embed.to(Device::CUDA(0));
    
    // Create input where every position has the same index
    Tensor input = ops::full({256, 256}, 5, DType::Int64).to(Device::CUDA(0));
    
    Tensor output = embed.forward(input);
    output.sum().backward();
    
    // All gradients should accumulate at index 5
    Tensor grad = embed.weight.grad();
    float expected = 256.0f * 256.0f;  // 65536 occurrences
    
    for (int d = 0; d < 128; ++d) {
        EXPECT_FLOAT_EQ(grad[5][d].item<float>(), expected);
    }
}
```

### 5.3 Stress Tests

```cpp
TEST_F(EmbeddingTest, StressTestLargeVocab) {
    // LLaMA-scale vocabulary
    const int64_t vocab_size = 32000;
    const int64_t embed_dim = 4096;
    const int64_t batch_size = 32;
    const int64_t seq_len = 2048;
    
    auto embed = nn::Embedding(vocab_size, embed_dim);
    embed.to(Device::CUDA(0));
    embed.weight.set_requires_grad(true);
    
    Tensor input = ops::randint(0, vocab_size, {batch_size, seq_len}, 
                                DType::Int64).to(Device::CUDA(0));
    
    // Time forward pass
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100; ++i) {
        Tensor output = embed.forward(input);
        output.sum().backward();
        embed.weight.grad().zero_();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "100 forward+backward passes: " << duration.count() << " ms" << std::endl;
    std::cout << "Average: " << duration.count() / 100.0 << " ms per iteration" << std::endl;
    
    // Memory usage
    size_t weight_bytes = vocab_size * embed_dim * sizeof(float);
    size_t output_bytes = batch_size * seq_len * embed_dim * sizeof(float);
    std::cout << "Weight memory: " << weight_bytes / 1e9 << " GB" << std::endl;
    std::cout << "Output memory: " << output_bytes / 1e9 << " GB" << std::endl;
}

TEST_F(EmbeddingTest, StressTestFrequentTokens) {
    // Simulate realistic token distribution (Zipf's law)
    auto embed = nn::Embedding(50000, 768);
    embed.to(Device::CUDA(0));
    embed.weight.set_requires_grad(true);
    
    // Generate Zipf-distributed indices (many repeats of common tokens)
    std::vector<int64_t> indices(32 * 512);
    std::mt19937 gen(42);
    
    // Simple power-law distribution approximating Zipf
    for (size_t i = 0; i < indices.size(); ++i) {
        // Bias towards lower indices (more frequent tokens)
        double u = std::uniform_real_distribution<double>(0, 1)(gen);
        indices[i] = static_cast<int64_t>(std::pow(u, 2) * 50000);
    }
    
    Tensor input = ops::from_blob(indices.data(), {32, 512}, DType::Int64)
                       .clone().to(Device::CUDA(0));
    
    // Run many iterations to check for race conditions
    for (int i = 0; i < 1000; ++i) {
        Tensor output = embed.forward(input);
        output.sum().backward();
        
        // Verify gradient is finite
        EXPECT_TRUE(embed.weight.grad().isfinite().all().item<bool>());
        
        embed.weight.grad().zero_();
    }
}

TEST_F(EmbeddingTest, MemoryLeakTest) {
    auto embed = nn::Embedding(10000, 512);
    embed.to(Device::CUDA(0));
    
    size_t initial_memory = get_cuda_memory_usage();
    
    for (int i = 0; i < 1000; ++i) {
        Tensor input = ops::randint(0, 10000, {32, 128}, DType::Int64)
                           .to(Device::CUDA(0));
        Tensor output = embed.forward(input);
        // Output should be freed when it goes out of scope
    }
    
    synchronize();
    size_t final_memory = get_cuda_memory_usage();
    
    // Memory should not grow significantly
    EXPECT_LT(final_memory - initial_memory, 1024 * 1024);  // Less than 1MB growth
}
```

### 5.4 Numerical Stability Tests

```cpp
TEST_F(EmbeddingTest, NumericalGradientCheck) {
    auto embed = nn::Embedding(100, 32);
    embed.weight.set_requires_grad(true);
    
    Tensor input = ops::tensor({{5, 10, 15}}, DType::Int64);
    
    // Compute analytical gradient
    Tensor output = embed.forward(input);
    Tensor loss = output.sum();
    loss.backward();
    Tensor analytical_grad = embed.weight.grad().clone();
    
    // Compute numerical gradient
    float eps = 1e-5f;
    Tensor numerical_grad = ops::zeros_like(embed.weight);
    
    for (int64_t i = 0; i < embed.num_embeddings(); ++i) {
        for (int64_t d = 0; d < embed.embedding_dim(); ++d) {
            // Forward with +eps
            embed.weight[i][d] += eps;
            float loss_plus = embed.forward(input).sum().item<float>();
            
            // Forward with -eps
            embed.weight[i][d] -= 2 * eps;
            float loss_minus = embed.forward(input).sum().item<float>();
            
            // Restore
            embed.weight[i][d] += eps;
            
            // Finite difference
            numerical_grad[i][d] = (loss_plus - loss_minus) / (2 * eps);
        }
    }
    
    EXPECT_TRUE(ops::allclose(analytical_grad, numerical_grad, 1e-3, 1e-3));
}
```

---

## 6. Performance Considerations

### 6.1 Memory Access Patterns

The embedding forward pass has **scattered read** access pattern—indices determine which rows to fetch, and these may be random. This is challenging for GPUs designed for coalesced access.

**Optimization strategies**:
1. **Vectorized loads**: Use `float4` to fetch 4 elements per memory transaction
2. **Warp-level coordination**: Ensure threads in a warp access nearby embedding dimensions
3. **Prefetching**: Hide memory latency with instruction-level parallelism

### 6.2 Atomic Operation Overhead

The backward pass uses `atomicAdd` which can be a bottleneck when:
- The same token appears many times (common with frequent words like "the")
- Multiple warps collide on the same embedding row

**Optimization strategies**:
1. **Warp-level reduction**: Reduce within warp before atomic
2. **Segmented reduce**: Sort indices, then reduce each segment
3. **Two-pass algorithm**: Count frequencies, allocate, then scatter without atomics

### 6.3 Memory Bandwidth

Embedding layers are memory-bound. The arithmetic intensity (FLOPs per byte) is essentially zero—we're just copying data.

**Optimization strategies**:
1. **Fused embeddings**: If you need both token and position embeddings, fuse them into a single kernel
2. **Half precision**: Use FP16 embeddings to halve memory bandwidth requirements
3. **Embedding compression**: Techniques like product quantization can reduce embedding size

---

## 7. Summary

The `Embedding` layer is deceptively simple—a lookup table—but requires careful implementation for efficiency:

| Aspect | Key Points |
|--------|------------|
| **Forward** | Gather operation, handle padding, vectorized GPU kernel |
| **Backward** | Scatter-add with atomics, handle repeated indices |
| **Padding** | Zero output and zero gradient for padding index |
| **Max Norm** | Post-hoc renormalization after each access |
| **Performance** | Memory-bound, optimize with vectorization and fused kernels |

The embedding layer is the entry point for all discrete-to-continuous transformations in deep learning. Master it, and you have the foundation for language models, recommender systems, and any application dealing with categorical data.
