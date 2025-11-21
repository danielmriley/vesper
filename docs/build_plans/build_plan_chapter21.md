# Vesper Build Plan - Chapter 21: Implementing CUDA Kernels

## 1. Goal

Translate the existing HIP kernels into CUDA kernels to fully enable the NVIDIA GPU backend. Since Vesper uses a "Zero Dependencies" approach with custom kernels, we will port our optimized HIP kernels (like Tiled GEMM) directly to CUDA.

## 2. Strategy

The HIP and CUDA APIs are nearly identical by design. The porting process involves:
1.  Copying the logic from `.hip` files to `.cu` files.
2.  Replacing `hip*` APIs with `cuda*` APIs (e.g., `hipMalloc` -> `cudaMalloc`, `__global__` remains the same, `hipLaunchKernelGGL` -> `<<<...>>>` syntax or `cudaLaunchKernel`).
3.  Ensuring the kernels are correctly dispatched from the C++ layer (already set up in Chapter 20).

## 3. Detailed Steps

### Step 3.1: Port GEMM Kernel (`src/ops/cuda/gemm.cu`)
-   **Source**: `src/ops/hip/gemm.hip`
-   **Task**: Implement the Tiled GEMM kernel in CUDA.
-   **Key Changes**:
    -   Replace `hipLaunchKernelGGL` with standard CUDA kernel launch syntax `kernel<<<blocks, threads>>>(...)`.
    -   Ensure `TILE_WIDTH` matches the HIP implementation (usually 16 or 32).

### Step 3.2: Port Elementwise Kernels (`src/ops/cuda/elementwise.cu`)
-   **Source**: `src/ops/hip/elementwise.hip`
-   **Task**: Implement `add`, `sub`, `mul`, `div` kernels.
-   **Key Changes**:
    -   Standardize on a 1D grid stride loop pattern if used in HIP.

### Step 3.3: Port Reduction Kernels (`src/ops/cuda/reduction.cu`)
-   **Source**: `src/ops/hip/reduction.hip`
-   **Task**: Implement `sum_kernel` (and potentially `mean` if separate).
-   **Key Changes**:
    -   Handle shared memory reduction logic using CUDA intrinsics if necessary (though basic shared mem is same).

### Step 3.4: Port Activation Kernels (`src/ops/cuda/activation.cu`)
-   **Source**: `src/ops/hip/activation.hip`
-   **Task**: Implement `relu`, `sigmoid`, etc.

### Step 3.5: Port Comparison Kernels (`src/ops/cuda/comparison.cu`)
-   **Source**: `src/ops/hip/comparison.hip`
-   **Task**: Implement `eq`, `ne`, `gt`, etc.

### Step 3.6: Port Random Kernels (`src/ops/cuda/random.cu`)
-   **Source**: `src/ops/hip/random.hip`
-   **Task**: Implement random number generation.
-   **Note**: If HIP uses `hiprand`, use `curand` (part of CUDA toolkit). If HIP uses a custom PCG/Philox kernel, port that.

## 4. Verification

Since we cannot run CUDA code on the current environment (likely), verification will be:
1.  **Compilation**: Ensure the project builds with `USE_CUDA=ON`.
2.  **Code Review**: Verify that the CUDA code is syntactically correct and mirrors the HIP logic.

## 5. Future Work
-   Optimize CUDA kernels (e.g., use Tensor Cores for GEMM via `wmma` API).
-   Add cuBLAS support as an optional backend for maximum performance.
