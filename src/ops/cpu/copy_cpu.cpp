#include <vesper/ops/copy.h>
#include <vesper/core/tensor.h>
#include <cstring>
#include <functional>

namespace vesper::ops {

void copy_strided_cpu_dispatch(const Tensor& src, Tensor& dst) {
    // If src is contiguous and layouts match (not possible here if src is strided view), 
    // but if src is contiguous, just memcpy.
    // Assuming this function is called when src is NOT contiguous relative to dst.
    
    size_t elem_size = GetDTypeSize(src.dtype());
    const char* src_ptr_base = static_cast<const char*>(src.data_ptr<void>()); // data_ptr already adds offset
    // Wait, src.data_ptr() adds offset based on src.offset().
    // My recursive lambda in tensor.cpp added offset.
    // Let's use the tensor data_ptr() which points to start.
    
    // We need to reconstruct the recursive logic but with typed pointers for simplicity or raw bytes?
    // Let's stick to raw bytes to support all types.
    
    // Actually, tensor.cpp logic was:
    // char* dst = ...
    // src_ptr += elem_size
    
    // If dst is contiguous, we can just iterate linearly on dst and compute src offset.
    // This avoids recursion depth limits and is cleaner.
    
    char* dst_ptr = static_cast<char*>(dst.data_ptr<void>());
    const char* src_base = static_cast<const char*>(src.data_ptr<void>()); // wait, data_ptr includes offset?
    // tensor.h: return static_cast<T*>(storage_->data()) + offset_;
    // Yes.
    
    // But wait, get_strided_index logic assumes offset 0 relative to the pointer passed?
    // Yes. The kernel used `in[offset]`. `in` was `src.data_ptr()`.
    // `offset` was computed from `strides`.
    // So `src.data_ptr()` is the base. `strides` are from the tensor.
    // This is correct.
    
    size_t n = dst.numel();
    const auto& shape = dst.shape();
    const auto& strides = src.strides();
    int dims = shape.size();
    
    for (size_t i = 0; i < n; ++i) {
        size_t offset_in = 0;
        size_t remaining = i;
        for (int d = dims - 1; d >= 0; --d) {
            size_t coord = remaining % shape[d];
            remaining /= shape[d];
            offset_in += coord * strides[d];
        }
        
        std::memcpy(dst_ptr + i * elem_size, src_base + offset_in * elem_size, elem_size);
    }
}

} // namespace vesper::ops
