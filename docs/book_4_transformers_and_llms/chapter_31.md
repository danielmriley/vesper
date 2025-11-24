# Chapter 31: Batch GEMM and 3D/4D Tensors

## 1. Introduction

Transformers operate on sequences of vectors. This means our tensors are typically 3D `(Batch, SeqLen, HiddenDim)` or 4D `(Batch, Heads, SeqLen, HeadDim)`.
To process these efficiently, we need **Batch GEMM** (General Matrix Multiply).

## 2. The Problem

Standard GEMM computes $C = A \times B$ for 2D matrices.
In Multi-Head Attention, we need to compute attention scores for every head in every batch item simultaneously.
$$ \text{Scores}[b, h] = Q[b, h] \times K[b, h]^T $$

We cannot loop over $b$ and $h$ in C++ because the kernel launch overhead would kill performance. We need a single kernel launch.

## 3. Implementation Plan

### Strided Batch GEMM

We will implement a kernel that treats the batch dimensions as a single "grid" of independent matrix multiplications.

**Kernel Signature:**

```cpp
__global__ void batch_gemm_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    long stride_A, long stride_B, long stride_C,
    int batch_count
);
```

**Logic:**

- `blockIdx.z` maps to the batch index.
- Each block computes a tile of the $(m, n)$ matrix for batch $z$.
- Pointers are offset: `A_ptr = A + blockIdx.z * stride_A`.
- **Memory Coalescing**: Ensure that within each tile, memory accesses are coalesced. This is standard for GEMM but critical to maintain when adding the batch stride.

### `ops::matmul` Update

Update the `matmul` operator to handle broadcasting.

- If inputs are 2D: Call standard GEMM.
- If inputs are >2D:
  1. Check if batch dimensions match (or can be broadcast).
     - **Broadcasting Rule**: If dimension size is 1, stride is 0. This allows a single matrix `(1, M, K)` to be multiplied against a batch `(B, K, N)` without copying data.
  2. Collapse all batch dimensions into a single `batch_count`.
  3. Call `batch_gemm_kernel`.

## 4. Usage Example

```cpp
// Batch=32, Heads=8, Seq=128, Dim=64
Tensor Q = randn({32, 8, 128, 64});
Tensor K = randn({32, 8, 128, 64});

// Transpose K to {32, 8, 64, 128}
Tensor Kt = K.transpose(-2, -1); 

// Batch Matmul: (128x64) @ (64x128) -> (128x128)
// Output shape: {32, 8, 128, 128}
Tensor scores = ops::matmul(Q, Kt);
```

## 5. Testing Strategy

1. **Correctness**: Compare Batch GEMM result against a loop of standard GEMMs on CPU.
2. **Broadcasting**: Test `(Batch, M, K) @ (K, N)` -> `(Batch, M, N)`.
