#include <cuda_runtime.h>
#include <vesper/core/tensor.h>
#include <vesper/core/stream.h>
#include <stdexcept>

namespace vesper::ops {

template <typename T>
__global__ void im2col_kernel(const T* data_im, T* data_col, 
                              int n,
                              int B, int C, int H, int W,
                              int KH, int KW, int SH, int SW, int PH, int PW,
                              int OH, int OW) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        int mat_w = B * OH * OW;
        int w_idx = index % mat_w; 
        int h_idx = index / mat_w;
        
        int ow = w_idx % OW;
        int oh = (w_idx / OW) % OH;
        int b = w_idx / (OW * OH);
        
        int kx = h_idx % KW;
        int ky = (h_idx / KW) % KH;
        int c = h_idx / (KW * KH);
        
        int h_in = oh * SH - PH + ky;
        int w_in = ow * SW - PW + kx;
        
        T val = 0;
        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
            int im_offset = ((b * C + c) * H + h_in) * W + w_in;
            val = data_im[im_offset];
        }
        data_col[index] = val;
    }
}

void im2col_hip_dispatch(const Tensor& input, Tensor& output, int kh, int kw, int sh, int sw, int ph, int pw) {
    int64_t B = input.shape()[0];
    int64_t C = input.shape()[1];
    int64_t H = input.shape()[2];
    int64_t W = input.shape()[3];
    int64_t out_h = (H + 2 * ph - kh) / sh + 1;
    int64_t out_w = (W + 2 * pw - kw) / sw + 1;
    
    size_t num_kernels = output.numel();
    const int threads = 256;
    const int blocks = (num_kernels + threads - 1) / threads;
    
    cudaStream_t stream = static_cast<cudaStream_t>(Stream::current(Device::CUDA).raw_handle());
    
    im2col_kernel<float><<<dim3(blocks), dim3(threads), 0, stream>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        num_kernels, B, C, H, W, kh, kw, sh, sw, ph, pw, out_h, out_w
    );
}

template <typename T>
__global__ void col2im_kernel(const T* data_col, T* data_im,
                              int n,
                              int B, int C, int H, int W,
                              int KH, int KW, int SH, int SW, int PH, int PW,
                              int OH, int OW)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        int mat_w = B * OH * OW;
        int w_idx = index % mat_w; 
        int h_idx = index / mat_w;
        
        int ow = w_idx % OW;
        int oh = (w_idx / OW) % OH;
        int b = w_idx / (OW * OH);
        
        int kx = h_idx % KW;
        int ky = (h_idx / KW) % KH;
        int c = h_idx / (KW * KH);
        
        int h_in = oh * SH - PH + ky;
        int w_in = ow * SW - PW + kx;
        
        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
            int im_offset = ((b * C + c) * H + h_in) * W + w_in;
            atomicAdd(data_im + im_offset, data_col[index]);
        }
    }
}

void col2im_hip_dispatch(const Tensor& grad_col, Tensor& grad_img, int kh, int kw, int sh, int sw, int ph, int pw) {
    int64_t B = grad_img.shape()[0];
    int64_t C = grad_img.shape()[1];
    int64_t H = grad_img.shape()[2];
    int64_t W = grad_img.shape()[3];
    int64_t out_h = (H + 2 * ph - kh) / sh + 1;
    int64_t out_w = (W + 2 * pw - kw) / sw + 1;
    
    size_t num_kernels = grad_col.numel();
    const int threads = 256;
    const int blocks = (num_kernels + threads - 1) / threads;
    
    cudaStream_t stream = static_cast<cudaStream_t>(Stream::current(Device::CUDA).raw_handle());
    
    col2im_kernel<float><<<dim3(blocks), dim3(threads), 0, stream>>>(
        grad_col.data_ptr<float>(), grad_img.data_ptr<float>(),
        num_kernels, B, C, H, W, kh, kw, sh, sw, ph, pw, out_h, out_w
    );
}

} // namespace vesper::ops
