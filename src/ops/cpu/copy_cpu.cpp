#include <vesper/ops/copy.h>
#include <vesper/core/tensor.h>
#include <cstring>
#include <functional>

namespace vesper::ops {

void copy_strided_cpu_dispatch(const Tensor& src, Tensor& dst) {
    size_t elem_size = GetDTypeSize(src.dtype());
    size_t n = dst.numel();
    
    char* dst_base = static_cast<char*>(dst.data_ptr<void>());
    const char* src_base = static_cast<const char*>(src.data_ptr<void>());
    
    bool src_contig = src.is_contiguous();
    bool dst_contig = dst.is_contiguous();
    
    // Optimization for same shape
    bool same_shape = (src.shape() == dst.shape());
    
    if (same_shape) {
        const auto& shape = dst.shape();
        const auto& src_strides = src.strides();
        const auto& dst_strides = dst.strides();
        int dims = shape.size();
        
        for (size_t i = 0; i < n; ++i) {
            size_t offset_in = 0;
            size_t offset_out = 0;
            size_t remaining = i;
            for (int d = dims - 1; d >= 0; --d) {
                size_t coord = remaining % shape[d];
                remaining /= shape[d];
                offset_in += coord * src_strides[d];
                offset_out += coord * dst_strides[d];
            }
            std::memcpy(dst_base + offset_out * elem_size, src_base + offset_in * elem_size, elem_size);
        }
    } else {
        const auto& dst_shape = dst.shape();
        const auto& dst_strides = dst.strides();
        int dst_dims = dst_shape.size();
        
        const auto& src_shape = src.shape();
        const auto& src_strides = src.strides();
        int src_dims = src_shape.size();

        for (size_t i = 0; i < n; ++i) {
            size_t offset_out = 0;
            if (dst_contig) {
                offset_out = i;
            } else {
                size_t remaining = i;
                for (int d = dst_dims - 1; d >= 0; --d) {
                    size_t coord = remaining % dst_shape[d];
                    remaining /= dst_shape[d];
                    offset_out += coord * dst_strides[d];
                }
            }
            
            size_t offset_in = 0;
            if (src_contig) {
                offset_in = i;
            } else {
                size_t remaining = i;
                for (int d = src_dims - 1; d >= 0; --d) {
                    size_t coord = remaining % src_shape[d];
                    remaining /= src_shape[d];
                    offset_in += coord * src_strides[d];
                }
            }
            
            std::memcpy(dst_base + offset_out * elem_size, src_base + offset_in * elem_size, elem_size);
        }
    }
}

} // namespace vesper::ops
