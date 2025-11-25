#include <vesper/ops/normalization.h>
#include <vesper/core/factories.h>
#include <vesper/ops/reduction.h>
#include <vesper/ops/elementwise.h>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <vesper/autograd/node.h>

namespace vesper::ops {

// --- Softmax ---

void softmax_hip_dispatch(const Tensor& input, int64_t dim, Tensor& output);

void softmax_cpu_dispatch(const Tensor& input, int64_t dim, Tensor& output) {
    // 1. Handle negative dim
    int64_t ndim = input.ndim();
    if (dim < 0) dim += ndim;
    
    // 2. Iterate over all other dimensions
    // We can treat this as a set of 1D operations.
    // Stride of the target dimension
    int64_t stride = input.strides()[dim];
    int64_t size = input.shape()[dim];
    
    // Total elements
    int64_t numel = input.numel();
    
    // We need to iterate over "outer" loops. 
    // A simple way for arbitrary dim is to iterate 0..numel, 
    // but that's hard to group by the reduction dim.
    
    // Alternative: Flatten into (outer, dim, inner)
    int64_t outer_stride = 1;
    for (int i = 0; i < dim; ++i) outer_stride *= input.shape()[i];
    int64_t inner_stride = 1;
    for (int i = dim + 1; i < ndim; ++i) inner_stride *= input.shape()[i];
    
    // Actually, strides are already computed.
    // Let's just use a recursive approach or coordinate counter for simplicity in MVP.
    // Or better: if contiguous, we can optimize.
    
    // Let's use the "outer loops" approach.
    // We iterate over all indices EXCEPT dim.
    
    // Calculate number of "rows" (independent softmax operations)
    int64_t num_rows = numel / size;
    
    // This is tricky with arbitrary strides. 
    // Let's use a coordinate system.
    std::vector<int64_t> coords(ndim, 0);
    
    // We will iterate num_rows times.
    // In each iteration, we fix all coords except `dim`.
    // Then we loop `dim` from 0 to size-1.
    
    // To do this efficiently:
    // We can iterate through the tensor, but skip `dim`.
    // This is equivalent to iterating over a tensor with `dim` removed.
    
    // Construct shape without dim
    std::vector<int64_t> outer_shape = input.shape();
    outer_shape.erase(outer_shape.begin() + dim);
    
    int64_t outer_numel = 1;
    for (auto s : outer_shape) outer_numel *= s;
    
    float* out_ptr = output.data_ptr<float>();
    const float* in_ptr = input.data_ptr<float>();
    
    // Iterate over the "outer" space
    std::vector<int64_t> outer_coords(outer_shape.size(), 0);
    
    for (int64_t i = 0; i < outer_numel; ++i) {
        // Reconstruct full coordinates
        std::vector<int64_t> full_coords = outer_coords;
        full_coords.insert(full_coords.begin() + dim, 0); // Placeholder
        
        // Calculate base offset
        int64_t base_offset = 0;
        for (int d = 0; d < ndim; ++d) {
            if (d != dim) base_offset += full_coords[d] * input.strides()[d];
        }
        
        // 1. Find Max
        float max_val = -std::numeric_limits<float>::infinity();
        for (int64_t j = 0; j < size; ++j) {
            int64_t offset = base_offset + j * input.strides()[dim];
            float val = in_ptr[offset];
            if (val > max_val) max_val = val;
        }
        
        // 2. Sum Exps
        float sum_exp = 0.0f;
        for (int64_t j = 0; j < size; ++j) {
            int64_t offset = base_offset + j * input.strides()[dim];
            sum_exp += std::exp(in_ptr[offset] - max_val);
        }
        
        // 3. Compute Softmax
        for (int64_t j = 0; j < size; ++j) {
            int64_t in_offset = base_offset + j * input.strides()[dim];
            int64_t out_offset = base_offset + j * output.strides()[dim]; // Assuming output has same strides/layout
            // Note: output might be contiguous even if input is not. 
            // We should re-calculate out_offset properly if output is new.
            // But here we assume output is created with same shape.
            // Let's calculate out_offset safely.
            
            int64_t safe_out_offset = 0;
            std::vector<int64_t> current_coords = full_coords;
            current_coords[dim] = j;
            for(int d=0; d<ndim; ++d) safe_out_offset += current_coords[d] * output.strides()[d];
            
            out_ptr[safe_out_offset] = std::exp(in_ptr[in_offset] - max_val) / sum_exp;
        }
        
        // Increment outer_coords
        for (int d = outer_shape.size() - 1; d >= 0; --d) {
            outer_coords[d]++;
            if (outer_coords[d] < outer_shape[d]) break;
            outer_coords[d] = 0;
        }
    }
}

Tensor softmax(const Tensor& input, int64_t dim) {
    Tensor output = empty(input.shape(), input.dtype(), input.device());
    
    if (input.device() == Device::CPU) {
        softmax_cpu_dispatch(input, dim, output);
    } else if (input.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        softmax_cuda_dispatch(input, dim, output);
#else
        throw std::runtime_error("CUDA backend not enabled.");
#endif
    } else if (input.device() == Device::HIP) {
#if USE_HIP_BACKEND
        softmax_hip_dispatch(input, dim, output);
#else
        throw std::runtime_error("HIP backend not enabled.");
#endif
    } else {
        throw std::runtime_error("Device not supported for softmax.");
    }
    
    // Backward pass
    if (input.requires_grad()) {
        auto node = std::make_shared<autograd::Node>();
        if (input.grad_node) {
            node->next_edges.push_back({input.grad_node});
        }
        
        // Capture non-const copy of input
        Tensor input_nc = input;
        
        node->backward_fn = [input_nc, weak_out=output.weak(), dim]() mutable {
            auto output = weak_out.lock();
            if (!output) return;

            if (input_nc.requires_grad()) {
                Tensor grad_output = output->grad();
                
                // grad_input = output * (grad_output - sum(output * grad_output, dim, keepdim=True))
                
                // 1. term1 = output * grad_output
                Tensor term1 = ops::mul(*output, grad_output);
                
                // 2. sum_term = sum(term1, dim, true);
                Tensor sum_term = ops::sum(term1, dim, true);
                
                // 3. term2 = grad_output - sum_term
                // Note: ops::sub supports broadcasting, so (..., dim, ...) - (..., 1, ...) works.
                Tensor term2 = ops::sub(grad_output, sum_term);
                
                // 4. grad_input = output * term2
                Tensor grad_input = ops::mul(*output, term2);
                
                input_nc.accumulate_grad(grad_input);
            }
        };
        output.grad_node = node;
    }
    
    return output;
}

// --- Layer Norm ---

void layer_norm_cuda_dispatch(const Tensor& input, const std::vector<int64_t>& normalized_shape, 
                              const Tensor& weight, const Tensor& bias, float eps, Tensor& output);
void layer_norm_hip_dispatch(const Tensor& input, const std::vector<int64_t>& normalized_shape, 
                             const Tensor& weight, const Tensor& bias, float eps, Tensor& output);

void layer_norm_cpu_dispatch(const Tensor& input, const std::vector<int64_t>& normalized_shape, 
                             const Tensor& weight, const Tensor& bias, float eps, Tensor& output) {
    // Assume normalized_shape is the last K dimensions.
    int64_t k = normalized_shape.size();
    int64_t ndim = input.ndim();
    
    // Calculate the size of the normalization slice
    int64_t norm_size = 1;
    for (auto s : normalized_shape) norm_size *= s;
    
    // Calculate number of independent normalizations
    int64_t num_batches = input.numel() / norm_size;
    
    // We assume input is contiguous for simplicity in this MVP implementation of LayerNorm
    // If not, we would need complex stride handling like Softmax.
    // Let's enforce contiguous for now or handle simple cases.
    
    const float* in_ptr = input.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();
    const float* w_ptr = weight.data_ptr<float>();
    const float* b_ptr = bias.data_ptr<float>();
    
    // Iterate over batches
    for (int64_t i = 0; i < num_batches; ++i) {
        int64_t offset = i * norm_size;
        
        // 1. Mean
        float sum = 0.0f;
        for (int64_t j = 0; j < norm_size; ++j) {
            sum += in_ptr[offset + j];
        }
        float mean = sum / norm_size;
        
        // 2. Variance
        float sum_sq_diff = 0.0f;
        for (int64_t j = 0; j < norm_size; ++j) {
            float diff = in_ptr[offset + j] - mean;
            sum_sq_diff += diff * diff;
        }
        float var = sum_sq_diff / norm_size;
        float inv_std = 1.0f / std::sqrt(var + eps);
        
        // 3. Normalize and Affine
        for (int64_t j = 0; j < norm_size; ++j) {
            float val = (in_ptr[offset + j] - mean) * inv_std;
            out_ptr[offset + j] = val * w_ptr[j] + b_ptr[j];
        }
    }
}

Tensor layer_norm(const Tensor& input, const std::vector<int64_t>& normalized_shape, 
                  const Tensor& weight, const Tensor& bias, float eps) {
    Tensor output = empty(input.shape(), input.dtype(), input.device());
    
    if (input.device() == Device::CPU) {
        // CPU implementation assumes contiguous input
        Tensor input_contig = input.contiguous();
        layer_norm_cpu_dispatch(input_contig, normalized_shape, weight, bias, eps, output);
    } else if (input.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        layer_norm_cuda_dispatch(input, normalized_shape, weight, bias, eps, output);
#else
        throw std::runtime_error("CUDA backend not enabled.");
#endif
    } else if (input.device() == Device::HIP) {
#if USE_HIP_BACKEND
        layer_norm_hip_dispatch(input, normalized_shape, weight, bias, eps, output);
#else
        throw std::runtime_error("HIP backend not enabled.");
#endif
    } else {
        throw std::runtime_error("Device not supported for layer_norm.");
    }
    
    return output;
}

// --- RMS Norm ---

void rms_norm_cuda_dispatch(const Tensor& input, const std::vector<int64_t>& normalized_shape, 
                            const Tensor& weight, float eps, Tensor& output);
void rms_norm_hip_dispatch(const Tensor& input, const std::vector<int64_t>& normalized_shape, 
                           const Tensor& weight, float eps, Tensor& output);

void rms_norm_cpu_dispatch(const Tensor& input, const std::vector<int64_t>& normalized_shape, 
                           const Tensor& weight, float eps, Tensor& output) {
    int64_t k = normalized_shape.size();
    int64_t norm_size = 1;
    for (auto s : normalized_shape) norm_size *= s;
    int64_t num_batches = input.numel() / norm_size;
    
    const float* in_ptr = input.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();
    const float* w_ptr = weight.data_ptr<float>();
    
    for (int64_t i = 0; i < num_batches; ++i) {
        int64_t offset = i * norm_size;
        
        // 1. Sum Squares
        float sum_sq = 0.0f;
        for (int64_t j = 0; j < norm_size; ++j) {
            float val = in_ptr[offset + j];
            sum_sq += val * val;
        }
        
        // 2. RMS
        float rms = std::sqrt(sum_sq / norm_size + eps);
        float inv_rms = 1.0f / rms;
        
        // 3. Normalize and Scale
        for (int64_t j = 0; j < norm_size; ++j) {
            out_ptr[offset + j] = in_ptr[offset + j] * inv_rms * w_ptr[j];
        }
    }
}

Tensor rms_norm(const Tensor& input, const std::vector<int64_t>& normalized_shape, 
                const Tensor& weight, float eps) {
    Tensor output = empty(input.shape(), input.dtype(), input.device());
    
    if (input.device() == Device::CPU) {
        // CPU implementation assumes contiguous input
        Tensor input_contig = input.contiguous();
        rms_norm_cpu_dispatch(input_contig, normalized_shape, weight, eps, output);
    } else if (input.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        rms_norm_cuda_dispatch(input, normalized_shape, weight, eps, output);
#else
        throw std::runtime_error("CUDA backend not enabled.");
#endif
    } else if (input.device() == Device::HIP) {
#if USE_HIP_BACKEND
        rms_norm_hip_dispatch(input, normalized_shape, weight, eps, output);
#else
        throw std::runtime_error("HIP backend not enabled.");
#endif
    } else {
        throw std::runtime_error("Device not supported for rms_norm.");
    }
    
    return output;
}

} // namespace vesper::ops
