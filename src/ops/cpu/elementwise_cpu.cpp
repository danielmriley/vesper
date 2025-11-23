#include <vesper/ops/elementwise.h>
#include <vesper/core/tensor.h>
#include <vector>

namespace vesper::ops {

// CPU implementation logic with broadcasting support
template <typename Op>
void cpu_broadcast_op(const Tensor& a, const std::vector<int64_t>& strides_a, 
                      const Tensor& b, const std::vector<int64_t>& strides_b, 
                      Tensor& out, Op op) {
    
    const float* a_ptr = a.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    
    size_t n = out.numel();
    int dims = out.shape().size();
    const auto& shape = out.shape();

    // Optimized path for 1D or contiguous identical shapes (simple case)
    // Checking strides is key. If all strides are default contiguous, use flat loop.
    // But here strides_a/b might have 0s for broadcasting.
    
    for (size_t i = 0; i < n; ++i) {
        // Compute offsets
        size_t offset_a = 0;
        size_t offset_b = 0;
        size_t remaining = i;
        
        for (int d = dims - 1; d >= 0; --d) {
            size_t coord = remaining % shape[d];
            remaining /= shape[d];
            offset_a += coord * strides_a[d];
            offset_b += coord * strides_b[d];
        }
        
        out_ptr[i] = op(a_ptr[offset_a], b_ptr[offset_b]);
    }
}

template <typename Op>
void cpu_scalar_op(const Tensor& a, float scalar, Tensor& out, Op op) {
    const float* a_ptr = a.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    
    size_t n = out.numel();
    int dims = out.shape().size();
    const auto& shape = out.shape();
    const auto& strides_a = a.strides();

    for (size_t i = 0; i < n; ++i) {
        size_t offset_a = 0;
        size_t remaining = i;
        
        for (int d = dims - 1; d >= 0; --d) {
            size_t coord = remaining % shape[d];
            remaining /= shape[d];
            offset_a += coord * strides_a[d];
        }
        out_ptr[i] = op(a_ptr[offset_a], scalar);
    }
}

void add_cpu_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out) {
    cpu_broadcast_op(a, strides_a, b, strides_b, out, [](float x, float y) { return x + y; });
}

void add_scalar_cpu_dispatch(const Tensor& a, float b, Tensor& out) {
    cpu_scalar_op(a, b, out, [](float x, float y) { return x + y; });
}

void sub_cpu_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out) {
    cpu_broadcast_op(a, strides_a, b, strides_b, out, [](float x, float y) { return x - y; });
}

void sub_scalar_cpu_dispatch(const Tensor& a, float b, Tensor& out) {
    cpu_scalar_op(a, b, out, [](float x, float y) { return x - y; });
}

void mul_cpu_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out) {
    cpu_broadcast_op(a, strides_a, b, strides_b, out, [](float x, float y) { return x * y; });
}

void mul_scalar_cpu_dispatch(const Tensor& a, float b, Tensor& out) {
    cpu_scalar_op(a, b, out, [](float x, float y) { return x * y; });
}

void div_cpu_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out) {
    cpu_broadcast_op(a, strides_a, b, strides_b, out, [](float x, float y) { return x / y; });
}

void div_scalar_cpu_dispatch(const Tensor& a, float b, Tensor& out) {
    cpu_scalar_op(a, b, out, [](float x, float y) { return x / y; });
}

} // namespace vesper::ops