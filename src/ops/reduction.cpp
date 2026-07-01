#include "vesper/ops/reduction.h"
#include "vesper/core/factories.h"
#include "vesper/autograd/node.h"
#include "vesper/autograd/guard.h"
#include "vesper/ops/elementwise.h"
#include "vesper/ops/comparison.h" // for equal
#include <stdexcept>
#include <limits>

namespace vesper {
namespace ops {

Tensor sum(const Tensor& input) {
    // 1. Check constraints
    if (input.dtype() != DType::Float32) {
        throw std::runtime_error("sum only supports Float32 for now.");
    }
    // sum is order-independent; sum a non-contiguous input via a contiguous copy.
    const Tensor in = input.is_contiguous() ? input : input.contiguous();

    // 2. Create output tensor (scalar)
    // We use empty shape {} for scalar.
    bool requires_grad = input.requires_grad() && autograd::grad_mode_enabled;
    Tensor output = vesper::empty({}, input.dtype(), input.device(), requires_grad);

    // 3. Dispatch
    if (input.device() == Device::CPU) {
        // Simple CPU implementation for testing/fallback
        const float* in_ptr = in.data_ptr<float>();
        float* out_ptr = output.data_ptr<float>();
        float sum_val = 0.0f;
        size_t n = in.numel();
        for (size_t i = 0; i < n; ++i) {
            sum_val += in_ptr[i];
        }
        *out_ptr = sum_val;
    } else if (input.device() == Device::HIP) {
#if USE_HIP_BACKEND
        sum_hip_dispatch(in, output);
#else
        throw std::runtime_error("HIP backend not enabled during build.");
#endif
    } else if (input.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        sum_cuda_dispatch(in, output);
#else
        throw std::runtime_error("CUDA backend not enabled during build.");
#endif
    } else {
        throw std::runtime_error("Unknown device type.");
    }

    if (requires_grad) {
        auto node = std::make_shared<autograd::Node>();
        if (input.requires_grad() && input.grad_node) {
            node->next_edges.push_back({input.grad_node});
        }
        
        node->backward_fn = [input=input, weak_out=output.weak()]() mutable {
            auto output = weak_out.lock();
            if (!output) return;
            if (input.requires_grad()) {
                Tensor& grad_output = output->grad();
                
                // PERFORMANCE FIX: Avoid GPU-CPU synchronization
                // Instead of reading the scalar value to CPU and using full(),
                // we broadcast the scalar gradient directly on the device using add.
                // zeros(shape) + scalar broadcasts the scalar to all elements.
                Tensor zeros_like_input = vesper::zeros(input.shape(), input.dtype(), input.device(), false);
                Tensor grad_input_contrib = ops::add(zeros_like_input, grad_output);
                
                input.accumulate_grad(grad_input_contrib);
            }
        };
        output.grad_node = node;
    }

    return output;
}

Tensor mean(const Tensor& input) {
    Tensor sum_val = sum(input);
    float n = static_cast<float>(input.numel());
    return div(sum_val, n);
}

Tensor sum(const Tensor& input, int64_t dim, bool keepdim) {
    int64_t ndim = input.ndim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("Dimension out of range for sum.");
    }

    // 1. Permute so that dim is the last dimension
    Tensor permuted = input;
    if (dim != ndim - 1) {
        permuted = input.transpose(dim, ndim - 1);
    }

    // 2. Flatten to [M, N] where N is the size of the reduction dimension
    // We need contiguous memory for the reduction kernel
    permuted = permuted.contiguous();
    
    int64_t N = permuted.shape().back();
    int64_t M = permuted.numel() / N;
    
    Tensor reshaped = permuted.reshape({M, N});
    
    // 3. Create output tensor [M, 1]
    bool requires_grad = input.requires_grad() && autograd::grad_mode_enabled;
    Tensor result_flat = vesper::empty({M, 1}, input.dtype(), input.device(), requires_grad);
    
    // 4. Dispatch
    if (input.device() == Device::CPU) {
        const float* in_ptr = reshaped.data_ptr<float>();
        float* out_ptr = result_flat.data_ptr<float>();
        
        for (int64_t i = 0; i < M; ++i) {
            float sum_val = 0.0f;
            for (int64_t j = 0; j < N; ++j) {
                sum_val += in_ptr[i * N + j];
            }
            out_ptr[i] = sum_val;
        }
        
    } else if (input.device() == Device::HIP) {
#if USE_HIP_BACKEND
        sum_cols_hip_dispatch(reshaped, result_flat);
#else
        throw std::runtime_error("HIP backend not enabled.");
#endif
    } else if (input.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        sum_cols_cuda_dispatch(reshaped, result_flat);
#else
        throw std::runtime_error("CUDA backend not enabled.");
#endif
    } else {
        throw std::runtime_error("Unknown device type.");
    }
    
    // 5. Reshape result
    std::vector<int64_t> permuted_out_shape = permuted.shape();
    permuted_out_shape.back() = 1;
    
    Tensor result = result_flat.reshape(permuted_out_shape);
    
    if (dim != ndim - 1) {
        result = result.transpose(dim, ndim - 1);
    }
    
    if (!keepdim) {
        std::vector<int64_t> final_shape = input.shape();
        final_shape.erase(final_shape.begin() + dim);
        result = result.reshape(final_shape);
    }
    
    // 6. Autograd
    if (requires_grad) {
        auto node = std::make_shared<autograd::Node>();
        if (input.requires_grad() && input.grad_node) {
            node->next_edges.push_back({input.grad_node});
        }
        
        node->backward_fn = [input=input, weak_res=result.weak(), dim, keepdim]() mutable {
            auto result = weak_res.lock();
            if (!result) return;
            if (input.requires_grad()) {
                Tensor grad_output = result->grad();
                
                if (!keepdim) {
                    std::vector<int64_t> unsqueezed_shape = input.shape();
                    unsqueezed_shape[dim] = 1;
                    grad_output = grad_output.reshape(unsqueezed_shape);
                }
                
                // Ensure grad_output is broadcasted to input.shape()
                // accumulate_grad might assign grad_output directly if grad is empty,
                // so we must ensure shape matches.
                if (grad_output.shape() != input.shape()) {
                    Tensor z = zeros(input.shape(), input.dtype(), input.device());
                    grad_output = ops::add(z, grad_output);
                }
                
                input.accumulate_grad(grad_output);
            }
        };
        result.grad_node = node;
    }
    
    return result;
}

Tensor max(const Tensor& input) {
    // Full reduction max
    if (input.dtype() != DType::Float32) throw std::runtime_error("max only supports Float32");
    
    bool requires_grad = input.requires_grad() && autograd::grad_mode_enabled;
    Tensor output = vesper::empty({}, input.dtype(), input.device(), requires_grad);
    
    if (input.device() == Device::CPU) {
        const float* in_ptr = input.data_ptr<float>();
        float* out_ptr = output.data_ptr<float>();
        float max_val = -std::numeric_limits<float>::infinity();
        size_t n = input.numel();
        for (size_t i = 0; i < n; ++i) {
            if (in_ptr[i] > max_val) max_val = in_ptr[i];
        }
        *out_ptr = max_val;
    } else {
        throw std::runtime_error("max only supported on CPU for now");
    }
    
    if (requires_grad) {
        auto node = std::make_shared<autograd::Node>();
        if (input.grad_node) node->next_edges.push_back({input.grad_node});
        
        node->backward_fn = [input=input, weak_out=output.weak()]() mutable {
            auto output = weak_out.lock();
            if (!output) return;
            if (input.requires_grad()) {
                // amax semantics: distribute the upstream gradient equally among all
                // elements tied for the maximum (matches torch.amax). The naive
                // mask * grad over-counts when several entries equal the max.
                Tensor mask = ops::equal(input, *output);
                Tensor tie_count = ops::sum(mask);
                Tensor grad_input = ops::mul(mask, ops::div(output->grad(), tie_count));
                input.accumulate_grad(grad_input);
            }
        };
        output.grad_node = node;
    }
    
    return output;
}

Tensor max(const Tensor& input, int64_t dim, bool keepdim) {
    int64_t ndim = input.ndim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("Dimension out of range for max.");
    }

    // 1. Permute so that dim is the last dimension
    Tensor permuted = input;
    if (dim != ndim - 1) {
        permuted = input.transpose(dim, ndim - 1);
    }

    // 2. Flatten to [M, N]
    permuted = permuted.contiguous();
    
    int64_t N = permuted.shape().back();
    int64_t M = permuted.numel() / N;
    
    Tensor reshaped = permuted.reshape({M, N});
    
    // 3. Create output tensor [M, 1]
    bool requires_grad = input.requires_grad() && autograd::grad_mode_enabled;
    Tensor result_flat = vesper::empty({M, 1}, input.dtype(), input.device(), requires_grad);
    
    // 4. Dispatch
    if (input.device() == Device::CPU) {
        const float* in_ptr = reshaped.data_ptr<float>();
        float* out_ptr = result_flat.data_ptr<float>();
        
        for (int64_t i = 0; i < M; ++i) {
            float max_val = -std::numeric_limits<float>::infinity();
            for (int64_t j = 0; j < N; ++j) {
                float val = in_ptr[i * N + j];
                if (val > max_val) max_val = val;
            }
            out_ptr[i] = max_val;
        }
    } else if (input.device() == Device::HIP) {
#if USE_HIP_BACKEND
        max_cols_hip_dispatch(reshaped, result_flat);
#else
        throw std::runtime_error("HIP backend not enabled.");
#endif
    } else if (input.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        max_cols_cuda_dispatch(reshaped, result_flat);
#else
        throw std::runtime_error("CUDA backend not enabled.");
#endif
    } else {
        throw std::runtime_error("Unknown device type.");
    }
    
    // 5. Reshape result
    std::vector<int64_t> permuted_out_shape = permuted.shape();
    permuted_out_shape.back() = 1;
    
    Tensor result = result_flat.reshape(permuted_out_shape);
    
    if (dim != ndim - 1) {
        result = result.transpose(dim, ndim - 1);
    }
    
    if (!keepdim) {
        std::vector<int64_t> final_shape = input.shape();
        final_shape.erase(final_shape.begin() + dim);
        result = result.reshape(final_shape);
    }
    
    // 6. Autograd
    if (requires_grad) {
        auto node = std::make_shared<autograd::Node>();
        if (input.requires_grad() && input.grad_node) {
            node->next_edges.push_back({input.grad_node});
        }
        
        node->backward_fn = [input=input, weak_res=result.weak(), dim, keepdim]() mutable {
            auto result = weak_res.lock();
            if (!result) return;
            if (input.requires_grad()) {
                Tensor grad_output = result->grad();
                
                if (!keepdim) {
                    std::vector<int64_t> unsqueezed_shape = input.shape();
                    unsqueezed_shape[dim] = 1;
                    grad_output = grad_output.reshape(unsqueezed_shape);
                }
                
                // Reshape the result back to a keepdim shape (size 1 along `dim`) so
                // it can be broadcast against the input.
                Tensor result_reshaped = *result;
                if (!keepdim) {
                    std::vector<int64_t> unsqueezed_shape = input.shape();
                    unsqueezed_shape[dim] = 1;
                    result_reshaped = result_reshaped.reshape(unsqueezed_shape);
                }

                // amax semantics: split the gradient equally among the tied maxima of
                // each reduced slice. ops::equal needs matching shapes for non-scalar
                // operands (it only broadcasts a true scalar), so first materialise the
                // per-slice max broadcast to the full input shape, build the tie mask,
                // then divide by the per-slice tie count.
                Tensor zeros_in = zeros(input.shape(), input.dtype(), input.device());
                Tensor result_bc = ops::add(zeros_in, result_reshaped);
                Tensor mask = ops::equal(input, result_bc);
                Tensor tie_count = ops::sum(mask, dim, /*keepdim=*/true);
                Tensor grad_input = ops::mul(grad_output, ops::div(mask, tie_count));

                input.accumulate_grad(grad_input);
            }
        };
        result.grad_node = node;
    }
    
    return result;
}

Tensor min(const Tensor& input) {
    // Full reduction min
    if (input.dtype() != DType::Float32) throw std::runtime_error("min only supports Float32");
    
    bool requires_grad = input.requires_grad() && autograd::grad_mode_enabled;
    Tensor output = vesper::empty({}, input.dtype(), input.device(), requires_grad);
    
    if (input.device() == Device::CPU) {
        const float* in_ptr = input.data_ptr<float>();
        float* out_ptr = output.data_ptr<float>();
        float min_val = std::numeric_limits<float>::infinity();
        size_t n = input.numel();
        for (size_t i = 0; i < n; ++i) {
            if (in_ptr[i] < min_val) min_val = in_ptr[i];
        }
        *out_ptr = min_val;
    } else {
        throw std::runtime_error("min only supported on CPU for now");
    }
    
    if (requires_grad) {
        auto node = std::make_shared<autograd::Node>();
        if (input.grad_node) node->next_edges.push_back({input.grad_node});
        
        node->backward_fn = [input=input, weak_out=output.weak()]() mutable {
            auto output = weak_out.lock();
            if (!output) return;
            if (input.requires_grad()) {
                // amin semantics: distribute the upstream gradient equally among all
                // elements tied for the minimum.
                Tensor mask = ops::equal(input, *output);
                Tensor tie_count = ops::sum(mask);
                Tensor grad_input = ops::mul(mask, ops::div(output->grad(), tie_count));
                input.accumulate_grad(grad_input);
            }
        };
        output.grad_node = node;
    }
    
    return output;
}

} // namespace ops
} // namespace vesper
