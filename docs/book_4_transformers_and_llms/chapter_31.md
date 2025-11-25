# Chapter 31: Batch GEMM and 3D/4D Tensors

## 1. Introduction

Transformers process sequences of vectors, resulting in 3D tensors `(Batch, SeqLen, Hidden)`
or 4D tensors `(Batch, Heads, SeqLen, HeadDim)` in multi-head attention. Efficient processing
requires **Batch GEMM**—executing many independent matrix multiplications in a single kernel.

**Why Batch GEMM Matters**:
- Multi-head attention computes $B \times H$ independent attention matrices per forward pass
- Kernel launch overhead (~5-10μs) would dominate if we looped over batches
- A single batched kernel achieves near-peak GPU utilization

---

## 2. Mathematical Foundation

### Standard GEMM

$$ C = A \times B $$

Where $A \in \mathbb{R}^{M \times K}$, $B \in \mathbb{R}^{K \times N}$, $C \in \mathbb{R}^{M \times N}$.

### Strided Batch GEMM

For tensors with leading batch dimensions:

$$ C[b] = A[b] \times B[b], \quad b \in [0, \text{batch\_count}) $$

Each slice $A[b]$, $B[b]$, $C[b]$ is a 2D matrix accessed via strides:

```
A[b] = A_ptr + b * stride_A
B[b] = B_ptr + b * stride_B
C[b] = C_ptr + b * stride_C
```

### Broadcasting Rules

Vesper follows NumPy-style broadcasting for batch dimensions:

| A shape | B shape | Result | stride_A | stride_B |
|---------|---------|--------|----------|----------|
| `(B, M, K)` | `(B, K, N)` | `(B, M, N)` | `M*K` | `K*N` |
| `(1, M, K)` | `(B, K, N)` | `(B, M, N)` | `0` | `K*N` |
| `(B, M, K)` | `(K, N)` | `(B, M, N)` | `M*K` | `0` |
| `(B, H, M, K)` | `(B, H, K, N)` | `(B, H, M, N)` | `H*M*K` | `H*K*N` |

When a dimension has size 1, its stride is set to 0, effectively reusing the same data.

---

## 3. API Design

### Public Interface

```cpp
namespace vesper::ops {
    // Unified matmul handles 2D, 3D, 4D, and broadcasting
    Tensor matmul(const Tensor& a, const Tensor& b);
    
    // Explicit GEMM with transpose flags
    Tensor gemm(const Tensor& a, const Tensor& b, bool transA, bool transB);
}
```

### Backend Dispatch

```cpp
// Batch GEMM dispatch (internal)
void gemm_batch_hip_dispatch(
    const Tensor& a, const Tensor& b, Tensor& c,
    int64_t batch_count,
    int64_t stride_a, int64_t stride_b, int64_t stride_c,
    bool transA, bool transB
);
```

### Dimension Handling Logic

```cpp
Tensor matmul(const Tensor& a, const Tensor& b) {
    // 1. Handle 2D case directly
    if (a.ndim() == 2 && b.ndim() == 2) {
        return gemm(a, b, false, false);
    }
    
    // 2. Extract batch dims and matrix dims
    auto [batch_a, M, K_a] = split_batch_matrix(a);
    auto [batch_b, K_b, N] = split_batch_matrix(b);
    
    // 3. Validate K dimensions match
    VESPER_CHECK(K_a == K_b, "Inner dimensions must match");
    
    // 4. Compute broadcast batch shape
    auto batch_out = broadcast_shapes(batch_a, batch_b);
    int64_t batch_count = product(batch_out);
    
    // 5. Compute strides (0 for broadcast dims)
    auto [stride_a, stride_b, stride_c] = compute_strides(...);
    
    // 6. Dispatch to batched kernel
    return gemm_batch_dispatch(a, b, batch_count, stride_a, stride_b, stride_c);
}
```

---

## 4. Kernel Implementation (HIP/CUDA)

### Grid Configuration

```cpp
// Each block handles a TILE_M x TILE_N tile of output
// blockIdx.z indexes into the batch dimension
dim3 grid(
    (M + TILE_M - 1) / TILE_M,
    (N + TILE_N - 1) / TILE_N,
    batch_count
);
dim3 block(TILE_M, TILE_N);  // Or use thread coarsening
```

### Kernel Pseudocode

```cpp
__global__ void batch_gemm_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    int64_t stride_A, int64_t stride_B, int64_t stride_C,
    int batch_count
) {
    int batch = blockIdx.z;
    
    // Pointer offset for this batch
    const float* A_batch = A + batch * stride_A;
    const float* B_batch = B + batch * stride_B;
    float* C_batch = C + batch * stride_C;
    
    // Standard tiled GEMM within each batch
    // ... shared memory tiling, register blocking, etc.
}
```

### Memory Coalescing

Critical for performance: threads in a warp must access consecutive memory addresses.

```
✓ Good: Thread i reads A[row][i] (consecutive in K dimension)
✗ Bad:  Thread i reads A[i][col] (strided access)
```

---

## 5. Backward Pass

For $C = A \times B$, gradients are:

$$ \frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} \times B^T $$
$$ \frac{\partial L}{\partial B} = A^T \times \frac{\partial L}{\partial C} $$

For batched operations, these apply element-wise across batches:

$$ \frac{\partial L}{\partial A[b]} = \frac{\partial L}{\partial C[b]} \times B[b]^T $$

**Broadcasting Gradient**: When A has stride 0 (broadcast), gradients must be summed:

$$ \frac{\partial L}{\partial A} = \sum_b \frac{\partial L}{\partial C[b]} \times B[b]^T $$

---

## 6. Usage Examples

### 3D Batch Matmul

```cpp
Tensor A = randn({32, 128, 64});   // [Batch, M, K]
Tensor B = randn({32, 64, 256});   // [Batch, K, N]
Tensor C = ops::matmul(A, B);      // [32, 128, 256]
```

### 4D Multi-Head Attention

```cpp
Tensor Q = randn({32, 8, 128, 64});  // [B, H, S, D]
Tensor K = randn({32, 8, 128, 64});  // [B, H, S, D]

// Transpose K: [B, H, S, D] -> [B, H, D, S]
Tensor Kt = K.transpose(-2, -1);

// Compute attention scores: [B, H, S, S]
Tensor scores = ops::matmul(Q, Kt);
```

### Broadcasting

```cpp
// Single weight matrix applied to batch
Tensor W = randn({64, 128});        // [K, N]
Tensor X = randn({32, 10, 64});     // [B, S, K]
Tensor Y = ops::matmul(X, W);       // [32, 10, 128]
```

---

## 7. Comprehensive Testing Strategy

### 7.1 Correctness Tests

| Test Case | Description | Validation |
|-----------|-------------|------------|
| `test_batch_gemm_3d` | `(B,M,K) @ (B,K,N)` | Compare to CPU loop |
| `test_batch_gemm_4d` | `(B,H,M,K) @ (B,H,K,N)` | Compare to CPU loop |
| `test_batch_gemm_large` | B=64, M=N=K=512 | Verify no overflow, correct result |
| `test_batch_gemm_small` | B=1, M=N=K=2 | Edge case, exact values |
| `test_batch_gemm_non_square` | M≠N≠K | Verify dimension handling |

### 7.2 Broadcasting Tests

| Test Case | Description | Validation |
|-----------|-------------|------------|
| `test_broadcast_1_to_B` | `(1,M,K) @ (B,K,N)` | Result shape `(B,M,N)` |
| `test_broadcast_2d_3d` | `(M,K) @ (B,K,N)` | Implicit batch dim |
| `test_broadcast_both` | `(1,M,K) @ (1,K,N)` | Result `(1,M,N)` |
| `test_broadcast_4d` | `(1,H,M,K) @ (B,H,K,N)` | Head broadcasting |

### 7.3 Backward Tests

| Test Case | Description | Validation |
|-----------|-------------|------------|
| `test_backward_shapes` | Verify grad shapes match inputs | Shape equality |
| `test_backward_values` | Finite difference gradient check | `abs(num - ana) < 1e-3` |
| `test_backward_broadcast` | Broadcast with gradient accumulation | Sum reduction correct |
| `test_backward_chain` | Multiple matmuls chained | Gradients propagate |

### 7.4 Consistency Tests

| Test Case | Description | Validation |
|-----------|-------------|------------|
| `test_cpu_vs_hip` | Same inputs, compare outputs | `max_diff < 1e-4` |
| `test_determinism` | Run twice, same result | Bit-identical |
| `test_dtype_float16` | Half precision batch GEMM | Reasonable tolerance |

### 7.5 Performance Tests

| Test Case | Description | Validation |
|-----------|-------------|------------|
| `benchmark_batch_vs_loop` | Compare batched kernel to naive loop | >10x speedup |
| `benchmark_scaling` | Vary B from 1 to 128 | Near-linear scaling |

---

## 8. Common Pitfalls

1. **Stride Calculation Errors**: Forgetting to account for transpose when computing strides
2. **Integer Overflow**: Use `int64_t` for strides with large tensors
3. **Non-Contiguous Tensors**: Must handle or reject strided views
4. **Broadcast Gradient Reduction**: Forgetting to sum when broadcasting backward

---

## 9. References

1. NVIDIA cuBLAS Documentation: `cublasSgemmStridedBatched`
2. ROCm hipBLAS: `hipblasSgemmStridedBatched`
3. Vaswani et al. "Attention Is All You Need" (2017)
