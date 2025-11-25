#include <vesper/ops/reduction.h>
#include <cuda_runtime.h>
#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <iostream>

namespace vesper::ops {

// ======================================================================================
// Existing Code (reduce_kernel, sum_cuda_dispatch)
// ======================================================================================

struct SumOp {
    __device__ float operator()(float a, float b) const { return a + b; }
};

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

void sum_cuda_dispatch(const Tensor& input, Tensor& output) {
    if (input.dtype() != DType::Float32) {
        throw std::runtime_error("Sum only supports Float32 for now.");
    }

    const size_t n = input.numel();
    if (n == 0) return;

    const int threads_per_block = 256;
    const int elements_per_block = threads_per_block * 2;
    
    Tensor curr_in = input;
    size_t curr_n = n;

    while (curr_n > 1) {
        int num_blocks = (curr_n + elements_per_block - 1) / elements_per_block;
        Tensor curr_out = empty({(long long)num_blocks}, input.dtype(), input.device());
        
        size_t shared_mem = threads_per_block * sizeof(float);
        
        reduce_kernel<float, SumOp><<<dim3(num_blocks), dim3(threads_per_block), shared_mem, 0>>>(
            curr_in.data_ptr<const float>(), curr_out.data_ptr<float>(), curr_n,
            SumOp(), 0.0f
        );
        
        curr_in = curr_out;
        curr_n = num_blocks;
    }
    
    (void)cudaMemcpy(output.data_ptr<void>(), curr_in.data_ptr<void>(), sizeof(float), cudaMemcpyDeviceToDevice);
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

void sum_rows_cuda_dispatch(const Tensor& input, Tensor& output) {
    int64_t M = input.shape()[0];
    int64_t N = input.shape()[1];
    const float* d_input = input.data_ptr<const float>();
    float* d_output = output.data_ptr<float>();
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    sum_rows_kernel<<<dim3(blocks), dim3(threads), 0, 0>>>( d_input, d_output, M, N);
}

// ======================================================================================
// New: Sum Cols (Reduce dim 1) [M, N] -> [M, 1]
// ======================================================================================

// Block-based reduction: One block per row.
// Threads in the block load the row in a coalesced manner and reduce.
__global__ void sum_cols_kernel(const float* input, float* output, int M, int N) {
    int row = blockIdx.x;
    if (row >= M) return;

    float sum = 0.0f;
    // Grid-stride loop within the row
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        sum += input[row * N + j];
    }

    // Block reduction using shared memory
    __shared__ float sdata[256];
    
    // Initial load into shared memory
    sdata[threadIdx.x] = sum;
    __syncthreads();

    // Tree reduction
    // We assume blockDim.x is 256
    if (threadIdx.x < 128) sdata[threadIdx.x] += sdata[threadIdx.x + 128]; __syncthreads();
    if (threadIdx.x < 64) sdata[threadIdx.x] += sdata[threadIdx.x + 64]; __syncthreads();
    
    // Last warp (32 threads) can use volatile or shuffle, but shared mem is fine for now
    if (threadIdx.x < 32) {
        volatile float* vsmem = sdata;
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 32];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 16];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 8];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 4];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 2];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 1];
    }

    if (threadIdx.x == 0) {
        output[row] = sdata[0];
    }
}

void sum_cols_cuda_dispatch(const Tensor& input, Tensor& output) {
    int64_t M = input.shape()[0];
    int64_t N = input.shape()[1];
    const float* d_input = input.data_ptr<const float>();
    float* d_output = output.data_ptr<float>();
    
    // One block per row
    int threads = 256;
    int blocks = M;
    
    sum_cols_kernel<<<dim3(blocks), dim3(threads), 0, 0>>>( d_input, d_output, M, N);
}

// ======================================================================================
// New: Max Cols (Reduce dim 1) [M, N] -> [M, 1]
// ======================================================================================

__global__ void max_cols_kernel(const float* input, float* output, int M, int N) {
    int row = blockIdx.x;
    if (row >= M) return;

    float max_val = -INFINITY;
    // Grid-stride loop within the row
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        float val = input[row * N + j];
        if (val > max_val) max_val = val;
    }

    // Block reduction using shared memory
    __shared__ float sdata[256];
    
    // Initial load into shared memory
    sdata[threadIdx.x] = max_val;
    __syncthreads();

    // Tree reduction for max
    if (threadIdx.x < 128) sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + 128]); __syncthreads();
    if (threadIdx.x < 64) sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + 64]); __syncthreads();
    
    // Last warp
    if (threadIdx.x < 32) {
        volatile float* vsmem = sdata;
        vsmem[threadIdx.x] = fmaxf(vsmem[threadIdx.x], vsmem[threadIdx.x + 32]);
        vsmem[threadIdx.x] = fmaxf(vsmem[threadIdx.x], vsmem[threadIdx.x + 16]);
        vsmem[threadIdx.x] = fmaxf(vsmem[threadIdx.x], vsmem[threadIdx.x + 8]);
        vsmem[threadIdx.x] = fmaxf(vsmem[threadIdx.x], vsmem[threadIdx.x + 4]);
        vsmem[threadIdx.x] = fmaxf(vsmem[threadIdx.x], vsmem[threadIdx.x + 2]);
        vsmem[threadIdx.x] = fmaxf(vsmem[threadIdx.x], vsmem[threadIdx.x + 1]);
    }

    if (threadIdx.x == 0) {
        output[row] = sdata[0];
    }
}

void max_cols_cuda_dispatch(const Tensor& input, Tensor& output) {
    int64_t M = input.shape()[0];
    int64_t N = input.shape()[1];
    const float* d_input = input.data_ptr<const float>();
    float* d_output = output.data_ptr<float>();
    
    // One block per row
    int threads = 256;
    int blocks = M;
    
    max_cols_kernel<<<dim3(blocks), dim3(threads), 0, 0>>>(d_input, d_output, M, N);
}

} // namespace vesper::ops