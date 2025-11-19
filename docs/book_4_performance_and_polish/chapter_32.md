
# Vesper Future Plans - Chapter 32: Implementing the CUDA Backend

## 1. Goal

Achieve true cross-vendor GPU support by fully implementing the CUDA backend. This involves porting all existing HIP kernels to CUDA C++ and filling in the stub functions created in Chapter 21.

## 2. The Porting Process

Thanks to the architectural planning in Chapter 21 and the similarity between HIP and CUDA APIs, this process should be methodical and straightforward, not requiring major architectural changes.

-   **API Mapping:** Most HIP APIs have a direct CUDA equivalent:
    -   `hipMalloc` -> `cudaMalloc`
    -   `hipMemcpy` -> `cudaMemcpy`
    -   `hipLaunchKernelGGL` -> `cudaLaunchKernelGGL` (or the `<<<...>>>` syntax)
    -   `__shared__` -> `__shared__`
    -   `blockIdx`, `threadIdx`, `blockDim` -> `blockIdx`, `threadIdx`, `blockDim`
-   **Implementation:** The task is to go through each `.cu` stub file (`gemm.cu`, `elementwise.cu`, etc.) and replace the `throw std::runtime_error` with a ported version of the corresponding `.hip` kernel. The logic can be translated almost one-to-one.

## 3. Why It's Next

While the project started as "HIP-first," supporting CUDA is essential for making the library usable by the vast majority of the deep learning community, who primarily use NVIDIA GPUs. Completing this chapter makes Vesper a genuinely portable, high-performance C++ deep learning library. It fulfills one of the core promises of the original blueprint.
