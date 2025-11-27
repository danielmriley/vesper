#include <vesper/core/tensor.h>
#include <vesper/core/dtype.h>
#include <vesper/core/half.h>
#include <stdexcept>

namespace vesper {
namespace ops {

// Helper to access data with proper type based on dtype
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

// Specializations for Float16 conversions (using explicit float conversion)
template <>
void cast_kernel<Float16, float>(const Tensor& input, Tensor& output) {
    float* out_ptr = output.data_ptr<float>();
    size_t n = output.numel();
    
    if (input.is_contiguous()) {
        const Float16* in_ptr = reinterpret_cast<const Float16*>(input.data_ptr<void>());
        for (size_t i = 0; i < n; ++i) {
            out_ptr[i] = static_cast<float>(in_ptr[i]);
        }
    } else {
        const Float16* in_base = reinterpret_cast<const Float16*>(input.data_ptr<void>());
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
            out_ptr[i] = static_cast<float>(in_base[offset_in]);
        }
    }
}

template <>
void cast_kernel<float, Float16>(const Tensor& input, Tensor& output) {
    Float16* out_ptr = reinterpret_cast<Float16*>(output.data_ptr<void>());
    size_t n = output.numel();
    
    if (input.is_contiguous()) {
        const float* in_ptr = input.data_ptr<float>();
        for (size_t i = 0; i < n; ++i) {
            out_ptr[i] = Float16(in_ptr[i]);
        }
    } else {
        const float* in_base = input.data_ptr<float>();
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
            out_ptr[i] = Float16(in_base[offset_in]);
        }
    }
}

// Specializations for BFloat16 conversions
template <>
void cast_kernel<BFloat16, float>(const Tensor& input, Tensor& output) {
    float* out_ptr = output.data_ptr<float>();
    size_t n = output.numel();
    
    if (input.is_contiguous()) {
        const BFloat16* in_ptr = reinterpret_cast<const BFloat16*>(input.data_ptr<void>());
        for (size_t i = 0; i < n; ++i) {
            out_ptr[i] = static_cast<float>(in_ptr[i]);
        }
    } else {
        const BFloat16* in_base = reinterpret_cast<const BFloat16*>(input.data_ptr<void>());
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
            out_ptr[i] = static_cast<float>(in_base[offset_in]);
        }
    }
}

template <>
void cast_kernel<float, BFloat16>(const Tensor& input, Tensor& output) {
    BFloat16* out_ptr = reinterpret_cast<BFloat16*>(output.data_ptr<void>());
    size_t n = output.numel();
    
    if (input.is_contiguous()) {
        const float* in_ptr = input.data_ptr<float>();
        for (size_t i = 0; i < n; ++i) {
            out_ptr[i] = BFloat16(in_ptr[i]);
        }
    } else {
        const float* in_base = input.data_ptr<float>();
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
            out_ptr[i] = BFloat16(in_base[offset_in]);
        }
    }
}

// Float16 <-> BFloat16 conversion (via float)
template <>
void cast_kernel<Float16, BFloat16>(const Tensor& input, Tensor& output) {
    BFloat16* out_ptr = reinterpret_cast<BFloat16*>(output.data_ptr<void>());
    size_t n = output.numel();
    
    if (input.is_contiguous()) {
        const Float16* in_ptr = reinterpret_cast<const Float16*>(input.data_ptr<void>());
        for (size_t i = 0; i < n; ++i) {
            float f = static_cast<float>(in_ptr[i]);
            out_ptr[i] = BFloat16(f);
        }
    } else {
        const Float16* in_base = reinterpret_cast<const Float16*>(input.data_ptr<void>());
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
            float f = static_cast<float>(in_base[offset_in]);
            out_ptr[i] = BFloat16(f);
        }
    }
}

template <>
void cast_kernel<BFloat16, Float16>(const Tensor& input, Tensor& output) {
    Float16* out_ptr = reinterpret_cast<Float16*>(output.data_ptr<void>());
    size_t n = output.numel();
    
    if (input.is_contiguous()) {
        const BFloat16* in_ptr = reinterpret_cast<const BFloat16*>(input.data_ptr<void>());
        for (size_t i = 0; i < n; ++i) {
            float f = static_cast<float>(in_ptr[i]);
            out_ptr[i] = Float16(f);
        }
    } else {
        const BFloat16* in_base = reinterpret_cast<const BFloat16*>(input.data_ptr<void>());
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
            float f = static_cast<float>(in_base[offset_in]);
            out_ptr[i] = Float16(f);
        }
    }
}

void cast_cpu_dispatch(const Tensor& input, Tensor& output) {
    DType in_dt = input.dtype();
    DType out_dt = output.dtype();
    
    // Handle Float16 source
    if (in_dt == DType::Float16) {
        switch (out_dt) {
            case DType::Float32: cast_kernel<Float16, float>(input, output); return;
            case DType::Float16: cast_kernel<Float16, Float16>(input, output); return;
            case DType::BFloat16: cast_kernel<Float16, BFloat16>(input, output); return;
            default: throw std::runtime_error("Unsupported output dtype for cast from Float16");
        }
    }
    
    // Handle BFloat16 source
    if (in_dt == DType::BFloat16) {
        switch (out_dt) {
            case DType::Float32: cast_kernel<BFloat16, float>(input, output); return;
            case DType::Float16: cast_kernel<BFloat16, Float16>(input, output); return;
            case DType::BFloat16: cast_kernel<BFloat16, BFloat16>(input, output); return;
            default: throw std::runtime_error("Unsupported output dtype for cast from BFloat16");
        }
    }
    
    // Handle Float32 source (including to half types)
    if (in_dt == DType::Float32) {
        switch (out_dt) {
            case DType::Float32: cast_kernel<float, float>(input, output); return;
            case DType::Float64: cast_kernel<float, double>(input, output); return;
            case DType::Float16: cast_kernel<float, Float16>(input, output); return;
            case DType::BFloat16: cast_kernel<float, BFloat16>(input, output); return;
            case DType::Int32: cast_kernel<float, int32_t>(input, output); return;
            case DType::Int64: cast_kernel<float, int64_t>(input, output); return;
            default: throw std::runtime_error("Unsupported output dtype for cast from Float32");
        }
    }
    
    // Macro to dispatch for remaining types
    #define DISPATCH_CAST_OUT(T_IN) \
        switch (out_dt) { \
            case DType::Float32: cast_kernel<T_IN, float>(input, output); break; \
            case DType::Float64: cast_kernel<T_IN, double>(input, output); break; \
            case DType::Int32:   cast_kernel<T_IN, int32_t>(input, output); break; \
            case DType::Int64:   cast_kernel<T_IN, int64_t>(input, output); break; \
            default: throw std::runtime_error("Unsupported output dtype for cast"); \
        }

    switch (in_dt) {
        case DType::Float64: DISPATCH_CAST_OUT(double); break;
        case DType::Int32:   DISPATCH_CAST_OUT(int32_t); break;
        case DType::Int64:   DISPATCH_CAST_OUT(int64_t); break;
        default: throw std::runtime_error("Unsupported input dtype for cast");
    }
    
    #undef DISPATCH_CAST_OUT
}

} // namespace ops
} // namespace vesper
