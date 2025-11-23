#include <vesper/core/tensor.h>
#include <vesper/core/dtype.h>
#include <stdexcept>

namespace vesper {
namespace ops {

template <typename T_IN, typename T_OUT>
void cast_kernel(const Tensor& input, Tensor& output) {
    // Note: input might be strided. output is contiguous.
    
    T_OUT* out_ptr = reinterpret_cast<T_OUT*>(output.data_ptr<void>());
    size_t n = output.numel();
    
    if (input.is_contiguous()) {
        const T_IN* in_ptr = reinterpret_cast<const T_IN*>(input.data_ptr<void>());
        for (size_t i = 0; i < n; ++i) {
            out_ptr[i] = static_cast<T_OUT>(in_ptr[i]);
        }
    } else {
        // Handle strided input
        // We need to access input using its strides.
        // Since output is contiguous, we iterate 0..n-1 and map to input coordinates.
        
        const T_IN* in_base = reinterpret_cast<const T_IN*>(input.data_ptr<void>());
        const auto& shape = input.shape();
        const auto& strides = input.strides();
        int dims = shape.size();
        
        for (size_t i = 0; i < n; ++i) {
            size_t offset_in = 0;
            size_t remaining = i;
            for (int d = dims - 1; d >= 0; --d) {
                size_t coord = remaining % shape[d];
                remaining /= shape[d];
                offset_in += coord * strides[d];
            }
            out_ptr[i] = static_cast<T_OUT>(in_base[offset_in]);
        }
    }
}

void cast_cpu_dispatch(const Tensor& input, Tensor& output) {
    DType in_dt = input.dtype();
    DType out_dt = output.dtype();
    
    // Macro to dispatch
    #define DISPATCH_CAST_OUT(T_IN) \
        switch (out_dt) { \
            case DType::Float32: cast_kernel<T_IN, float>(input, output); break; \
            case DType::Float64: cast_kernel<T_IN, double>(input, output); break; \
            case DType::Int32:   cast_kernel<T_IN, int32_t>(input, output); break; \
            case DType::Int64:   cast_kernel<T_IN, int64_t>(input, output); break; \
            default: throw std::runtime_error("Unsupported output dtype for cast"); \
        }

    switch (in_dt) {
        case DType::Float32: DISPATCH_CAST_OUT(float); break;
        case DType::Float64: DISPATCH_CAST_OUT(double); break;
        case DType::Int32:   DISPATCH_CAST_OUT(int32_t); break;
        case DType::Int64:   DISPATCH_CAST_OUT(int64_t); break;
        default: throw std::runtime_error("Unsupported input dtype for cast");
    }
}

} // namespace ops
} // namespace vesper
