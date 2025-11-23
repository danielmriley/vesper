# Vesper Future Plans - Chapter 38.1: Fused GEMM + Bias + Activation

## 1. Goal

Implement "fused" kernels that combine matrix multiplication (GEMM) with subsequent element-wise operations like Bias Addition and Activation functions (ReLU, GELU, etc.).

## 2. Motivation

In a standard Transformer or MLP block, the operations typically flow as:
`X -> Linear (GEMM) -> Output1 -> Add Bias -> Output2 -> Activation (ReLU/GELU) -> Output3`.

This sequence requires **three separate kernel launches** and **three round-trips to global memory**:
1.  GEMM reads `X` and `W`, writes `Output1`.
2.  Add reads `Output1` and `Bias`, writes `Output2`.
3.  Act reads `Output2`, writes `Output3`.

This is incredibly wasteful. The GEMM operation is compute-bound, but the subsequent element-wise operations are memory-bound. By "fusing" them into the GEMM kernel, we can perform the Bias Add and Activation on the result of the matrix multiplication *while it is still in registers*, before it is ever written to global memory. This effectively makes the Bias and Activation **free**.

## 3. The Fused Kernel

We will extend our `gemm_register_tiled_kernel` to accept optional "epilogues".

### Epilogue Functors
Instead of hardcoding `ReLU` or `Bias`, we will use a template-based **Epilogue Functor** pattern.

```cpp
template <typename EpilogueOp>
__global__ void gemm_fused_kernel(..., EpilogueOp op) {
    // ... (Compute GEMM as usual, result in registers rC) ...

    // 4. Store C (with Epilogue)
    // Instead of just writing rC to global memory:
    
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            float val = rC[i][j];
            
            // Apply Epilogue!
            // e.g., val = ReLU(val + bias[col])
            val = op(val, row, col); 
            
            C[idx] = val;
        }
    }
}
```

### Candidates for Fusion
1.  **Linear + Bias**: `y = xW^T + b`
2.  **Linear + ReLU**: `y = ReLU(xW^T)`
3.  **Linear + Bias + ReLU**: `y = ReLU(xW^T + b)`
4.  **Linear + GELU**: Essential for Transformers (BERT/GPT).

## 4. Implementation Plan

1.  **Define Epilogue Interface**: Create a simple struct/functor interface that device code can use.
2.  **Update GEMM Dispatcher**: Modify `gemm_cuda_dispatch` to accept an optional `Epilogue` object.
3.  **Implement `LinearFused` Module**: A specialized `nn::LinearFused` module that calls this optimized path instead of the standard `matmul` + `add` + `relu` sequence.

## 5. Why It's Next

This is the "low-hanging fruit" of kernel fusion. It requires modifying existing, well-understood kernels and provides immediate, measurable speedups (often 1.5x - 2x) for MLP blocks.
