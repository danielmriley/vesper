#include <cuda_runtime.h>
#include <vesper/core/tensor.h>
#include <vesper/core/stream.h>

namespace vesper::ops {

template <typename T>
__global__ void max_pool2d_kernel(
    const T* input, T* output, int64_t* indices,
    int B, int C, int H, int W,
    int out_h, int out_w,
    int kh, int kw, int sh, int sw, int ph, int pw)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * C * out_h * out_w;
    
    if (idx < total_elements) {
        int ow = idx % out_w;
        int tmp = idx / out_w;
        int oh = tmp % out_h;
        tmp /= out_h;
        int c = tmp % C;
        int b = tmp / C;
        
        int h_start = oh * sh - ph;
        int w_start = ow * sw - pw;
        int h_end = min(h_start + kh, H);
        int w_end = min(w_start + kw, W);
        h_start = max(h_start, 0);
        w_start = max(w_start, 0);
        
        T max_val = -1e30; 
        int64_t max_idx = -1;
        
        int input_offset_base = (b * C + c) * H * W;
        
        for (int h = h_start; h < h_end; ++h) {
            for (int w = w_start; w < w_end; ++w) {
                int cur_idx = input_offset_base + h * W + w;
                T val = input[cur_idx];
                if (val > max_val) {
                    max_val = val;
                    max_idx = cur_idx;
                }
            }
        }
        
        output[idx] = max_val;
        indices[idx] = max_idx;
    }
}

void max_pool2d_hip_dispatch(const Tensor& input, Tensor& output, Tensor& indices, int kh, int kw, int sh, int sw, int ph, int pw) {
    int B = input.shape()[0];
    int C = input.shape()[1];
    int H = input.shape()[2];
    int W = input.shape()[3];
    
    int out_h = output.shape()[2];
    int out_w = output.shape()[3];
    
    size_t total_elements = output.numel();
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    cudaStream_t stream = static_cast<cudaStream_t>(Stream::current(Device::CUDA).raw_handle());
    
    max_pool2d_kernel<float><<<dim3(blocks), dim3(threads), 0, stream>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), indices.data_ptr<int64_t>(),
        B, C, H, W, out_h, out_w, kh, kw, sh, sw, ph, pw
    );
}

template <typename T>
__global__ void max_pool2d_backward_kernel(
    const T* grad_output, const int64_t* indices, T* grad_input,
    int total_outputs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_outputs) {
        int64_t input_idx = indices[idx];
        if (input_idx != -1) {
            atomicAdd(grad_input + input_idx, grad_output[idx]);
        }
    }
}

void max_pool2d_backward_hip_dispatch(const Tensor& grad_output, const Tensor& indices, Tensor& grad_input) {
    size_t total_outputs = grad_output.numel();
    const int threads = 256;
    const int blocks = (total_outputs + threads - 1) / threads;
    
    cudaStream_t stream = static_cast<cudaStream_t>(Stream::current(Device::CUDA).raw_handle());
    
    max_pool2d_backward_kernel<float><<<dim3(blocks), dim3(threads), 0, stream>>>(
        grad_output.data_ptr<float>(), indices.data_ptr<int64_t>(), grad_input.data_ptr<float>(),
        total_outputs
    );
}

} // namespace vesper::ops
