# Vesper Future Plans - Chapter 39: Mixed Precision Training (FP16/BF16)

## 1. Goal

Enable training and inference using 16-bit floating point formats (`Float16` and `BFloat16`). This is essential for reducing memory usage and leveraging Tensor Cores on modern GPUs.

## 2. The Challenge

Vesper currently hardcodes `float` (FP32) in all kernels and dispatchers. Supporting 16-bit types requires:
1.  **Storage & DType:** Updating `DType` to support `Float16` and `BFloat16`.
2.  **Kernel Templating:** Updating all CUDA/HIP kernels to use `template <typename T>` and ensuring `__half` and `__nv_bfloat16` intrinsics are available.
3.  **Master Weights:** Implementing the "Master Weight" pattern for optimizers, where parameters are stored in FP32 for stability but gradients/activations are FP16.
4.  **Loss Scaling:** Implementing dynamic loss scaling to prevent gradient underflow (vanishing gradients) which is common in FP16 (less critical for BF16).

## 3. Implementation Plan

1.  **Type Support:** Add `half` and `bfloat16` classes/typedefs to `vesper::core`.
2.  **Kernel Updates:** Refactor `elementwise`, `gemm`, and `reduction` kernels to be fully templated.
3.  **Cast Operation:** Implement a `cast(dtype)` operation to convert tensors between precisions.
4.  **AMP Context:** Create an `Autocast` context manager (similar to `NoGradGuard`) that automatically casts inputs to `matmul` to FP16 while keeping other ops in FP32 if necessary.

## 4. Why It's Next

FP32 is "legacy" for LLM training. Almost all modern training happens in mixed precision. This update essentially doubles the effective VRAM capacity of the library.
