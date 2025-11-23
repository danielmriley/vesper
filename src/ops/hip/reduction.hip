#include <vesper/ops/reduction.h>
#include <hip/hip_runtime.h>
#include <vesper/core/tensor.h>
#include <iostream>

namespace vesper::ops {

// ======================================================================================
// Existing Code (reduce_kernel, sum_hip_dispatch)
// ======================================================================================

template <typename T, typename Op>
__global__ void reduce_kernel(const T* in_data, T* out_data, size_t n, Op op, T neutral_element) {
    extern __shared__ T sdata[];
    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * (blockDim.x * 2) + tid;
    
    sdata[tid] = (i < n) ? in_data[i] : neutral_element;
    if (i + blockDim.x < n) {
        sdata[tid] = op(sdata[tid], in_data[i + blockDim.x]);
    }
     __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = op(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_data[blockIdx.x] = sdata[0];
    }
}

void sum_hip_dispatch(const Tensor& input, Tensor& output) {
    if (input.dtype() != DType::Float32) {
        throw std::runtime_error("Sum only supports Float32 for now.");
    }

    const size_t n = input.numel();
    if (n == 0) return;

    const int threads_per_block = 128;
    const int elements_per_block = threads_per_block * 2;
    const int num_blocks = (n + elements_per_block - 1) / elements_per_block;
    const size_t shared_mem_size = threads_per_block * sizeof(float);

    Tensor intermediate = empty({(long long)num_blocks}, input.dtype(), input.device());

    hipLaunchKernelGGL(
        (reduce_kernel<float, std::plus<float>>),
        dim3(num_blocks), dim3(threads_per_block), shared_mem_size, 0,
        input.data_ptr<const float>(), intermediate.data_ptr<float>(), n,
        std::plus<float>(), 0.0f
    );

    if (num_blocks > 1) {
        const int final_threads = (num_blocks + 1) / 2;
        const size_t final_shared_mem = final_threads * sizeof(float);
         hipLaunchKernelGGL(
            (reduce_kernel<float, std::plus<float>>),
            dim3(1), dim3(final_threads), final_shared_mem, 0,
            intermediate.data_ptr<const float>(), output.data_ptr<float>(),
            num_blocks, std::plus<float>(), 0.0f
        );
    } else {
        hipMemcpy(output.data_ptr<void>(), intermediate.data_ptr<void>(), sizeof(float), hipMemcpyDeviceToDevice);
    }
}

// ======================================================================================
// New: Sum Rows (Reduce dim 0) [M, N] -> [N]
// ======================================================================================

__global__ void sum_rows_kernel(const float* input, float* output, int M, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < N) {
        float sum = 0.0f;
        for (int i = 0; i < M; ++i) {
            sum += input[i * N + j];
        }
        output[j] = sum;
    }
}

void sum_rows_hip_dispatch(const Tensor& input, Tensor& output) {
    int64_t M = input.shape()[0];
    int64_t N = input.shape()[1];
    const float* d_input = input.data_ptr<const float>();
    float* d_output = output.data_ptr<float>();
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    hipLaunchKernelGGL(sum_rows_kernel, dim3(blocks), dim3(threads), 0, 0, d_input, d_output, M, N);
}

// ======================================================================================
// New: Sum Cols (Reduce dim 1) [M, N] -> [M, 1]
// ======================================================================================

// Simple kernel: one thread per row. Good for large M, small N.
// If N is large, this is inefficient (thread divergence, lack of parallelism per row).
// But for broadcasting backward of bias vectors (usually N is feature dim, M is batch), M is usually large enough.
__global__ void sum_cols_kernel(const float* input, float* output, int M, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) {
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            sum += input[i * N + j];
        }
        output[i] = sum; // output is [M, 1] or [M], effectively linear index i
    }
}

void sum_cols_hip_dispatch(const Tensor& input, Tensor& output) {
    int64_t M = input.shape()[0];
    int64_t N = input.shape()[1];
    const float* d_input = input.data_ptr<const float>();
    float* d_output = output.data_ptr<float>();
    int threads = 256;
    int blocks = (M + threads - 1) / threads;
    hipLaunchKernelGGL(sum_cols_kernel, dim3(blocks), dim3(threads), 0, 0, d_input, d_output, M, N);
}

} // namespace vesper::ops