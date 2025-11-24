
# Chapter 29: The `nn.Embedding` Layer

## 1. Introduction

The `Embedding` layer is the bridge between discrete data (like words or token IDs) and continuous vector space. It maps an integer index to a dense vector of fixed size.

## 2. Mathematical Definition

Given a vocabulary of size $V$ and an embedding dimension $D$, the layer holds a weight matrix $W \in \mathbb{R}^{V \times D}$.

For an input index $i$, the output is simply the $i$-th row of $W$:
$$ \text{Output} = W[i, :] $$

For a batch of indices, the operation is applied to each index.

## 3. Implementation Plan

### Forward Pass

The forward pass is a "gather" operation. It does not involve matrix multiplication.

1. Input: `indices` tensor of shape `(Batch, SeqLen)` with dtype `Int64`.
2. Output: `embeddings` tensor of shape `(Batch, SeqLen, EmbedDim)`.
3. Logic: Create a new tensor and copy rows from `weight` based on `indices`.
4. **Padding Index**: If `padding_idx` is specified, the output vector for that index must be all zeros. This is crucial for variable-length sequences.

### Backward Pass (Sparse Gradients)

The backward pass is unique. Since only a few rows of $W$ are used in the forward pass, only those rows receive gradients.

1. Input: `grad_output` of shape `(Batch, SeqLen, EmbedDim)`.
2. Logic:
   - Initialize `grad_weight` as zeros.
   - Accumulate `grad_output` rows into `grad_weight` at the positions specified by `indices`.
   - **Atomic Adds (GPU)**: On GPU, multiple threads might try to update the same row (if an index is repeated in the batch). We must use `atomicAdd` to ensure correctness.
   - Note: If an index appears multiple times, gradients must be summed.

### Constraints

- **Max Norm**: Optionally, if `max_norm` is set, we re-normalize the embedding vectors to have a norm less than or equal to `max_norm` after each update. This prevents the embedding space from exploding.

### Class Structure

```cpp
class Embedding : public Module {
public:
    Embedding(int64_t num_embeddings, int64_t embedding_dim, int64_t padding_idx = -1, float max_norm = -1.0f);
    Tensor forward(const Tensor& input) override;
    
    Tensor weight; // Shape: [num_embeddings, embedding_dim]
    int64_t padding_idx_;
    float max_norm_;
};
```

## 4. Usage Example

```cpp
// Vocab size = 1000, Embedding dim = 64
auto embed = nn::Embedding(1000, 64);

// Batch of 2 sequences, length 4
Tensor input = tensor({
    {1, 5, 9, 2},
    {3, 5, 1, 0}
}, DType::Int64);

Tensor output = embed(input); 
// output shape: [2, 4, 64]
```

## 5. Testing Strategy

1. **Correctness**: Manually verify that `output[b, s]` equals `weight[input[b, s]]`.
2. **Gradient**: Check that gradients are correctly accumulated for repeated indices (e.g., index '5' in the example above).
