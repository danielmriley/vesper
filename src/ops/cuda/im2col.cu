#include <cuda_runtime.h>
#include <vesper/core/tensor.h>
#include <vesper/core/stream.h>
#include <stdexcept>

namespace vesper::ops {

// CUDA kernel for im2col
// Parallelize over (B, C, OH, OW) -> output columns
// One thread per output element? Or one thread per output column?
// Standard approach: One thread per output element (pixel in `output`).
// Output size: (C * KH * KW) * (B * OH * OW).
// This is large.
// Better: One thread per pixel in the output feature map (B, C, OH, OW). 
// For each such pixel, copy the KH*KW patch to the column.
// BUT our output format is [C*KH*KW, B*OH*OW].
// This means elements from the same patch are in the same COLUMN.
// Elements from different patches are in different COLUMNS.
// So (C*KH*KW) is the "height" of the column matrix.
// 
// We parallelize over the output columns: index `i` from 0 to (B*OH*OW).
// For each `i`, we populate the column vector of size (C*KH*KW).
// This might be too much work per thread.
// 
// Standard "Caffe" im2col kernel parallelizes over (C * OutputH * OutputW).
// Input index `index` maps to (c, oh, ow).
// It then iterates over KH, KW to fill the column.
// Wait, Caffe im2col is for a single image.
// We support batches by folding B into the width: [C*KH*KW, B*OH*OW].
// Effectively, we treat B images as one large image or just iterate B in outer loop.
//
// Let's use the standard approach: One thread per output element `(c_out, col_idx)`? 
// No, that's slow memory access (scattered reads).
// 
// Let's parallelize over the total number of elements in the OUTPUT matrix.
// Total elements = (C * KH * KW) * (B * OH * OW).
// Thread `idx` maps to (row, col) in Output.
// row = c_in * KH * KW + ky * KW + kx
// col = b * OH * OW + oh * OW + ow
// We can reverse map `idx` to (b, c, ky, kx, oh, ow).
// Then calculate input (b, c, h_in, w_in).
// This allows full parallelism.

template <typename T>
__global__ void im2col_kernel(const T* data_im, T* data_col, 
                              int n, // Total elements in output
                              int B, int C, int H, int W,
                              int KH, int KW, int SH, int SW, int PH, int PW,
                              int OH, int OW) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        // output shape [MatH, MatW] where MatH = C*KH*KW, MatW = B*OH*OW
        int mat_w = B * OH * OW;
        
        int w_idx = index % mat_w; // Column index (b, oh, ow)
        int h_idx = index / mat_w; // Row index (c, ky, kx)
        
        // Decode w_idx -> (b, oh, ow)
        int ow = w_idx % OW;
        int oh = (w_idx / OW) % OH;
        int b = w_idx / (OW * OH);
        
        // Decode h_idx -> (c, ky, kx)
        int kx = h_idx % KW;
        int ky = (h_idx / KW) % KH;
        int c = h_idx / (KW * KH);
        
        // Calculate input indices
        int h_in = oh * SH - PH + ky;
        int w_in = ow * SW - PW + kx;
        
        T val = 0;
        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
            // data_im is [B, C, H, W]
            // offset = b*C*H*W + c*H*W + h_in*W + w_in
            int im_offset = ((b * C + c) * H + h_in) * W + w_in;
            val = data_im[im_offset];
        }
        data_col[index] = val;
    }
}

void im2col_cuda_dispatch(const Tensor& input, Tensor& output, int kh, int kw, int sh, int sw, int ph, int pw) {
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
    
    im2col_kernel<float><<<blocks, threads, 0, stream>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        num_kernels,
        B, C, H, W,
        kh, kw, sh, sw, ph, pw,
        out_h, out_w
    );
}

// col2im Kernel
// parallelize over output elements (input image pixels).
// For each pixel (b, c, h, w), sum up contributions from the column matrix.
// But wait, a pixel contributes to multiple columns (sliding windows).
// So we need to sum over all windows that include this pixel.
// This is harder.
// Alternative: Parallelize over the COL matrix (gradients), and use atomicAdd to the image.
// Thread `idx` maps to an element in `grad_col`.
// It computes the corresponding (b, c, h, w) in the image and atomicAdds.
// This is easier to implement but atomicAdd can be slow if many collisions.
// For now, atomicAdd is the standard way for a simple col2im.

template <typename T>
__global__ void col2im_kernel(const T* data_col, T* data_im,
                              int n, // Total elements in col matrix
                              int B, int C, int H, int W,
                              int KH, int KW, int SH, int SW, int PH, int PW,
                              int OH, int OW)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        // map index to col matrix coordinates
        int mat_w = B * OH * OW;
        int w_idx = index % mat_w; // (b, oh, ow)
        int h_idx = index / mat_w; // (c, ky, kx)
        
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

void col2im_cuda_dispatch(const Tensor& grad_col, Tensor& grad_img, int kh, int kw, int sh, int sw, int ph, int pw) {
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
    
    // grad_img must be zeroed before accumulation (factory does this usually, but check caller)
    // caller `col2im` uses `zeros()`, so it's fine.
    
    col2im_kernel<float><<<blocks, threads, 0, stream>>>(
        grad_col.data_ptr<float>(), grad_img.data_ptr<float>(),
        num_kernels,
        B, C, H, W,
        kh, kw, sh, sw, ph, pw,
        out_h, out_w
    );
}

} // namespace vesper::ops
