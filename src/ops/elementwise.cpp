#include <vesper/ops/elementwise.h>
#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <vesper/core/broadcasting.h>
#include <vesper/core/macros.h>
#include <vesper/autograd/node.h>
#include <vesper/ops/reduction.h>
#include <vesper/autograd/guard.h>
#include <stdexcept>
#include <iostream>
#include <cmath>

namespace vesper::ops {

// --- Reduction Helpers ---
Tensor sum_rows_cpu(const Tensor& input);
Tensor sum_cols_cpu(const Tensor& input);

// Dispatchers
void add_hip_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out);
void sub_hip_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out);
void mul_hip_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out);
void div_hip_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out);

void add_cuda_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out);
void sub_cuda_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out);
void mul_cuda_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out);
void div_cuda_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out);

void add_cpu_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out);
void sub_cpu_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out);
void mul_cpu_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out);
void div_cpu_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, const Tensor& b, const std::vector<int64_t>& strides_b, Tensor& out);

void add_scalar_hip_dispatch(const Tensor& a, float b, Tensor& out);
void sub_scalar_hip_dispatch(const Tensor& a, float b, Tensor& out);
void mul_scalar_hip_dispatch(const Tensor& a, float b, Tensor& out);
void div_scalar_hip_dispatch(const Tensor& a, float b, Tensor& out);

void add_scalar_cuda_dispatch(const Tensor& a, float b, Tensor& out);
void sub_scalar_cuda_dispatch(const Tensor& a, float b, Tensor& out);
void mul_scalar_cuda_dispatch(const Tensor& a, float b, Tensor& out);
void div_scalar_cuda_dispatch(const Tensor& a, float b, Tensor& out);

void add_scalar_cpu_dispatch(const Tensor& a, float b, Tensor& out);
void sub_scalar_cpu_dispatch(const Tensor& a, float b, Tensor& out);
void mul_scalar_cpu_dispatch(const Tensor& a, float b, Tensor& out);
void div_scalar_cpu_dispatch(const Tensor& a, float b, Tensor& out);

// Helper function to reduce a gradient tensor `grad` to `target_shape`
Tensor handle_broadcast_backward(const Tensor& grad, const std::vector<int64_t>& target_shape) {
    if (grad.shape() == target_shape) {
        return grad;
    }

    Tensor result = grad;
    
    // 1. Handle extra dimensions (leading)
    // Example: grad (2, 3, 4), target (3, 4) -> sum dim 0
    while (result.ndim() > static_cast<int64_t>(target_shape.size())) {
        result = ops::sum(result, 0, false);
    }
    
    // 2. Handle broadcasted dimensions (where target is 1 and grad is > 1)
    // Example: grad (3, 4), target (3, 1) -> sum dim 1 with keepdim=true
    for (int64_t i = 0; i < static_cast<int64_t>(target_shape.size()); ++i) {
        if (target_shape[i] == 1 && result.shape()[i] > 1) {
            result = ops::sum(result, i, true);
        }
    }
    
    return result;
}

template<typename DispatchFn>
Tensor binary_op_impl(const Tensor& a, const Tensor& b, DispatchFn dispatch) {
    if (a.device() != b.device()) {
        throw std::runtime_error("Tensor devices do not match for binary operation.");
    }

    std::vector<int64_t> out_shape = broadcast_shapes(a.shape(), b.shape());
    std::vector<int64_t> strides_a = compute_broadcast_strides(a.shape(), a.strides(), out_shape);
    std::vector<int64_t> strides_b = compute_broadcast_strides(b.shape(), b.strides(), out_shape);

    bool requires_grad = (a.requires_grad() || b.requires_grad()) && autograd::grad_mode_enabled;
    Tensor result = empty(out_shape, a.dtype(), a.device(), requires_grad);

    dispatch(a, strides_a, b, strides_b, result);

    return result;
}

template<typename DispatchFn>
Tensor scalar_op_impl(const Tensor& a, float b, DispatchFn dispatch) {
    bool requires_grad = a.requires_grad() && autograd::grad_mode_enabled;
    Tensor result = empty(a.shape(), a.dtype(), a.device(), requires_grad);
    dispatch(a, b, result);
    return result;
}

Tensor add(const Tensor& a, const Tensor& b) {
    auto dispatch = [&](const Tensor& ta, const std::vector<int64_t>& sa, const Tensor& tb, const std::vector<int64_t>& sb, Tensor& out) {
        switch(ta.device()) {
            case Device::HIP:
#if USE_HIP_BACKEND
                add_hip_dispatch(ta, sa, tb, sb, out);
#else
                throw std::runtime_error("HIP backend not enabled.");
#endif
                break;
            case Device::CPU:
                add_cpu_dispatch(ta, sa, tb, sb, out);
                break;
            case Device::CUDA:
#if USE_CUDA_BACKEND
                add_cuda_dispatch(ta, sa, tb, sb, out);
#else
                throw std::runtime_error("CUDA backend not enabled.");
#endif
                break;
            default:
                throw std::runtime_error("Device not supported for add operation.");
        }
    };

    Tensor result = binary_op_impl(a, b, dispatch);

    if (result.requires_grad()) {
        auto node = std::make_shared<autograd::Node>();
        if (a.requires_grad() && a.grad_node) node->next_edges.push_back({a.grad_node});
        if (b.requires_grad() && b.grad_node) node->next_edges.push_back({b.grad_node});

        node->backward_fn = [a=a, b=b, result_weak=result.weak()]() mutable {
            auto result = result_weak.lock();
            if (!result) return;
            Tensor& grad_output = result->grad();
            if (a.requires_grad()) {
                a.accumulate_grad(handle_broadcast_backward(grad_output, a.shape()));
            }
            if (b.requires_grad()) {
                b.accumulate_grad(handle_broadcast_backward(grad_output, b.shape()));
            }
        };
        result.grad_node = node;
    }
    return result;
}

Tensor add(const Tensor& a, float b) {
    auto dispatch = [&](const Tensor& ta, float scalar, Tensor& out) {
        switch(ta.device()) {
            case Device::HIP:
#if USE_HIP_BACKEND
                add_scalar_hip_dispatch(ta, scalar, out);
#else
                throw std::runtime_error("HIP backend not enabled.");
#endif
                break;
            case Device::CPU:
                add_scalar_cpu_dispatch(ta, scalar, out);
                break;
            case Device::CUDA:
#if USE_CUDA_BACKEND
                add_scalar_cuda_dispatch(ta, scalar, out);
#else
                throw std::runtime_error("CUDA backend not enabled.");
#endif
                break;
            default:
                throw std::runtime_error("Device not supported for add scalar.");
        }
    };

    Tensor result = scalar_op_impl(a, b, dispatch);

    if (result.requires_grad()) {
        auto node = std::make_shared<autograd::Node>();
        if (a.requires_grad() && a.grad_node) node->next_edges.push_back({a.grad_node});
        
        node->backward_fn = [a=a, result_weak=result.weak()]() mutable {
            auto result = result_weak.lock();
            if (!result) return;
            if (a.requires_grad()) {
                a.accumulate_grad(result->grad());
            }
        };
        result.grad_node = node;
    }
    return result;
}

Tensor sub(const Tensor& a, const Tensor& b) {
    auto dispatch = [&](const Tensor& ta, const std::vector<int64_t>& sa, const Tensor& tb, const std::vector<int64_t>& sb, Tensor& out) {
        switch(ta.device()) {
            case Device::HIP:
#if USE_HIP_BACKEND
                sub_hip_dispatch(ta, sa, tb, sb, out);
#else
                throw std::runtime_error("HIP backend not enabled.");
#endif
                break;
            case Device::CPU:
                sub_cpu_dispatch(ta, sa, tb, sb, out);
                break;
            case Device::CUDA:
#if USE_CUDA_BACKEND
                sub_cuda_dispatch(ta, sa, tb, sb, out);
#else
                throw std::runtime_error("CUDA backend not enabled.");
#endif
                break;
            default:
                throw std::runtime_error("Device not supported for sub operation.");
        }
    };

    Tensor result = binary_op_impl(a, b, dispatch);

    if (result.requires_grad()) {
        auto node = std::make_shared<autograd::Node>();
        if (a.requires_grad() && a.grad_node) node->next_edges.push_back({a.grad_node});
        if (b.requires_grad() && b.grad_node) node->next_edges.push_back({b.grad_node});

        node->backward_fn = [a=a, b=b, result_weak=result.weak()]() mutable {
            auto result = result_weak.lock();
            if (!result) return;
            Tensor& grad_output = result->grad();
            if (a.requires_grad()) {
                a.accumulate_grad(handle_broadcast_backward(grad_output, a.shape()));
            }
            if (b.requires_grad()) {
                // For c = a - b, dc/db = -1, so grad_b = -grad_output
                Tensor reduced = handle_broadcast_backward(grad_output, b.shape());
                Tensor neg_reduced = mul(reduced, -1.0f);
                b.accumulate_grad(neg_reduced);
            }
        };
        result.grad_node = node;
    }
    return result;
}

Tensor sub(const Tensor& a, float b) {
    auto dispatch = [&](const Tensor& ta, float scalar, Tensor& out) {
        switch(ta.device()) {
            case Device::HIP:
#if USE_HIP_BACKEND
                sub_scalar_hip_dispatch(ta, scalar, out);
#else
                throw std::runtime_error("HIP backend not enabled.");
#endif
                break;
            case Device::CPU:
                sub_scalar_cpu_dispatch(ta, scalar, out);
                break;
            case Device::CUDA:
#if USE_CUDA_BACKEND
                sub_scalar_cuda_dispatch(ta, scalar, out);
#else
                throw std::runtime_error("CUDA backend not enabled.");
#endif
                break;
            default:
                throw std::runtime_error("Device not supported for sub scalar.");
        }
    };

    Tensor result = scalar_op_impl(a, b, dispatch);

    if (result.requires_grad()) {
        auto node = std::make_shared<autograd::Node>();
        if (a.requires_grad() && a.grad_node) node->next_edges.push_back({a.grad_node});
        
        node->backward_fn = [a=a, result_weak=result.weak()]() mutable {
            auto result = result_weak.lock();
            if (!result) return;
            if (a.requires_grad()) {
                a.accumulate_grad(result->grad());
            }
        };
        result.grad_node = node;
    }
    return result;
}

Tensor mul(const Tensor& a, const Tensor& b) {
    auto dispatch = [&](const Tensor& ta, const std::vector<int64_t>& sa, const Tensor& tb, const std::vector<int64_t>& sb, Tensor& out) {
        switch(ta.device()) {
            case Device::HIP:
#if USE_HIP_BACKEND
                mul_hip_dispatch(ta, sa, tb, sb, out);
#else
                throw std::runtime_error("HIP backend not enabled.");
#endif
                break;
            case Device::CPU:
                mul_cpu_dispatch(ta, sa, tb, sb, out);
                break;
            case Device::CUDA:
#if USE_CUDA_BACKEND
                mul_cuda_dispatch(ta, sa, tb, sb, out);
#else
                throw std::runtime_error("CUDA backend not enabled.");
#endif
                break;
            default:
                throw std::runtime_error("Device not supported for mul operation.");
        }
    };

    Tensor result = binary_op_impl(a, b, dispatch);

    if (result.requires_grad()) {
        auto node = std::make_shared<autograd::Node>();
        if (a.requires_grad() && a.grad_node) node->next_edges.push_back({a.grad_node});
        if (b.requires_grad() && b.grad_node) node->next_edges.push_back({b.grad_node});

        node->backward_fn = [a=a, b=b, result_weak=result.weak()]() mutable {
            auto result = result_weak.lock();
            if (!result) return;
            Tensor& grad_output = result->grad();
            if (a.requires_grad()) {
                Tensor grad_a_contrib = mul(grad_output, b);
                a.accumulate_grad(handle_broadcast_backward(grad_a_contrib, a.shape()));
            }
            if (b.requires_grad()) {
                Tensor grad_b_contrib = mul(grad_output, a);
                b.accumulate_grad(handle_broadcast_backward(grad_b_contrib, b.shape()));
            }
        };
        result.grad_node = node;
    }
    return result;
}

Tensor mul(const Tensor& a, float b) {
    auto dispatch = [&](const Tensor& ta, float scalar, Tensor& out) {
        switch(ta.device()) {
            case Device::HIP:
#if USE_HIP_BACKEND
                mul_scalar_hip_dispatch(ta, scalar, out);
#else
                throw std::runtime_error("HIP backend not enabled.");
#endif
                break;
            case Device::CPU:
                mul_scalar_cpu_dispatch(ta, scalar, out);
                break;
            case Device::CUDA:
#if USE_CUDA_BACKEND
                mul_scalar_cuda_dispatch(ta, scalar, out);
#else
                throw std::runtime_error("CUDA backend not enabled.");
#endif
                break;
            default:
                throw std::runtime_error("Device not supported for mul scalar.");
        }
    };

    Tensor result = scalar_op_impl(a, b, dispatch);

    if (result.requires_grad()) {
        auto node = std::make_shared<autograd::Node>();
        if (a.requires_grad() && a.grad_node) node->next_edges.push_back({a.grad_node});
        
        node->backward_fn = [a=a, b, result_weak=result.weak()]() mutable {
            auto result = result_weak.lock();
            if (!result) return;
            if (a.requires_grad()) {
                Tensor grad_contrib = mul(result->grad(), b);
                a.accumulate_grad(grad_contrib);
            }
        };
        result.grad_node = node;
    }
    return result;
}

Tensor div(const Tensor& a, const Tensor& b) {
    auto dispatch = [&](const Tensor& ta, const std::vector<int64_t>& sa, const Tensor& tb, const std::vector<int64_t>& sb, Tensor& out) {
        switch(ta.device()) {
            case Device::HIP:
#if USE_HIP_BACKEND
                div_hip_dispatch(ta, sa, tb, sb, out);
#else
                throw std::runtime_error("HIP backend not enabled.");
#endif
                break;
            case Device::CPU:
                div_cpu_dispatch(ta, sa, tb, sb, out);
                break;
            case Device::CUDA:
#if USE_CUDA_BACKEND
                div_cuda_dispatch(ta, sa, tb, sb, out);
#else
                throw std::runtime_error("CUDA backend not enabled.");
#endif
                break;
            default:
                throw std::runtime_error("Device not supported for div operation.");
        }
    };

    Tensor result = binary_op_impl(a, b, dispatch);

    if (result.requires_grad()) {
        auto node = std::make_shared<autograd::Node>();
        if (a.requires_grad() && a.grad_node) node->next_edges.push_back({a.grad_node});
        if (b.requires_grad() && b.grad_node) node->next_edges.push_back({b.grad_node});

        node->backward_fn = [a=a, b=b, result_weak=result.weak()]() mutable {
            auto result = result_weak.lock();
            if (!result) return;
            Tensor& grad_output = result->grad();
            
            if (a.requires_grad()) {
                Tensor grad_a_contrib = div(grad_output, b);
                a.accumulate_grad(handle_broadcast_backward(grad_a_contrib, a.shape()));
            }
            if (b.requires_grad()) {
                Tensor neg_result = mul(*result, -1.0f);
                Tensor grad_b_contrib = mul(grad_output, div(neg_result, b));
                b.accumulate_grad(handle_broadcast_backward(grad_b_contrib, b.shape()));
            }
        };
        result.grad_node = node;
    }

    return result;
}

Tensor div(const Tensor& a, float b) {
    auto dispatch = [&](const Tensor& ta, float scalar, Tensor& out) {
        switch(ta.device()) {
            case Device::HIP:
#if USE_HIP_BACKEND
                div_scalar_hip_dispatch(ta, scalar, out);
#else
                throw std::runtime_error("HIP backend not enabled.");
#endif
                break;
            case Device::CPU:
                div_scalar_cpu_dispatch(ta, scalar, out);
                break;
            case Device::CUDA:
#if USE_CUDA_BACKEND
                div_scalar_cuda_dispatch(ta, scalar, out);
#else
                throw std::runtime_error("CUDA backend not enabled.");
#endif
                break;
            default:
                throw std::runtime_error("Device not supported for div scalar.");
        }
    };

    Tensor result = scalar_op_impl(a, b, dispatch);

    if (result.requires_grad()) {
        auto node = std::make_shared<autograd::Node>();
        if (a.requires_grad() && a.grad_node) node->next_edges.push_back({a.grad_node});
        
        node->backward_fn = [a=a, b, result_weak=result.weak()]() mutable {
            auto result = result_weak.lock();
            if (!result) return;
            if (a.requires_grad()) {
                float inv_b = 1.0f / b;
                Tensor grad_contrib = mul(result->grad(), inv_b);
                a.accumulate_grad(grad_contrib);
            }
        };
        result.grad_node = node;
    }
    return result;
}

Tensor& add_(Tensor& a, const Tensor& b) {
    if (a.requires_grad() && autograd::grad_mode_enabled) {
        throw std::runtime_error("In-place operations on leaf variables or variables needed for gradient computation are not yet safe.");
    }

    if (a.device() != b.device()) {
        throw std::runtime_error("Tensor devices do not match for in-place add.");
    }
    // Check for broadcasting compatibility where output shape == a.shape
    std::vector<int64_t> out_shape = broadcast_shapes(a.shape(), b.shape());
    if (out_shape != a.shape()) {
        throw std::runtime_error("In-place add: output shape mismatch (broadcasting would resize tensor).");
    }

    std::vector<int64_t> strides_a = a.strides();
    std::vector<int64_t> strides_b = compute_broadcast_strides(b.shape(), b.strides(), out_shape);

    auto dispatch = [&](const Tensor& ta, const std::vector<int64_t>& sa, const Tensor& tb, const std::vector<int64_t>& sb, Tensor& out) {
        switch(ta.device()) {
            case Device::HIP:
#if USE_HIP_BACKEND
                add_hip_dispatch(ta, sa, tb, sb, out);
#else
                throw std::runtime_error("HIP backend not enabled.");
#endif
                break;
            case Device::CPU:
                add_cpu_dispatch(ta, sa, tb, sb, out);
                break;
            case Device::CUDA:
#if USE_CUDA_BACKEND
                add_cuda_dispatch(ta, sa, tb, sb, out);
#else
                throw std::runtime_error("CUDA backend not enabled.");
#endif
                break;
            default:
                throw std::runtime_error("Device not supported for add operation.");
        }
    };

    dispatch(a, strides_a, b, strides_b, a);
    return a;
}

Tensor& sub_(Tensor& a, const Tensor& b) {
    if (a.requires_grad() && autograd::grad_mode_enabled) {
        throw std::runtime_error("In-place operations on leaf variables or variables needed for gradient computation are not yet safe.");
    }

    if (a.device() != b.device()) {
        throw std::runtime_error("Tensor devices do not match for in-place sub.");
    }
    std::vector<int64_t> out_shape = broadcast_shapes(a.shape(), b.shape());
    if (out_shape != a.shape()) {
        throw std::runtime_error("In-place sub: output shape mismatch.");
    }

    std::vector<int64_t> strides_a = a.strides();
    std::vector<int64_t> strides_b = compute_broadcast_strides(b.shape(), b.strides(), out_shape);

    auto dispatch = [&](const Tensor& ta, const std::vector<int64_t>& sa, const Tensor& tb, const std::vector<int64_t>& sb, Tensor& out) {
        switch(ta.device()) {
            case Device::HIP:
#if USE_HIP_BACKEND
                sub_hip_dispatch(ta, sa, tb, sb, out);
#else
                throw std::runtime_error("HIP backend not enabled.");
#endif
                break;
            case Device::CPU:
                sub_cpu_dispatch(ta, sa, tb, sb, out);
                break;
            case Device::CUDA:
#if USE_CUDA_BACKEND
                sub_cuda_dispatch(ta, sa, tb, sb, out);
#else
                throw std::runtime_error("CUDA backend not enabled.");
#endif
                break;
            default:
                throw std::runtime_error("Device not supported for sub operation.");
        }
    };

    dispatch(a, strides_a, b, strides_b, a);
    return a;
}

// --- In-place Scalar Ops ---

Tensor& add_(Tensor& a, float b) {
    if (a.requires_grad() && autograd::grad_mode_enabled) {
        throw std::runtime_error("In-place operations on leaf variables or variables needed for gradient computation are not yet safe.");
    }
    
    auto dispatch = [&](const Tensor& ta, float scalar, Tensor& out) {
        switch(ta.device()) {
            case Device::HIP:
#if USE_HIP_BACKEND
                add_scalar_hip_dispatch(ta, scalar, out);
#else
                throw std::runtime_error("HIP backend not enabled.");
#endif
                break;
            case Device::CPU:
                add_scalar_cpu_dispatch(ta, scalar, out);
                break;
            case Device::CUDA:
#if USE_CUDA_BACKEND
                add_scalar_cuda_dispatch(ta, scalar, out);
#else
                throw std::runtime_error("CUDA backend not enabled.");
#endif
                break;
            default:
                throw std::runtime_error("Device not supported for add scalar.");
        }
    };
    
    dispatch(a, b, a);
    return a;
}

Tensor& sub_(Tensor& a, float b) {
    if (a.requires_grad() && autograd::grad_mode_enabled) {
        throw std::runtime_error("In-place operations on leaf variables or variables needed for gradient computation are not yet safe.");
    }
    
    auto dispatch = [&](const Tensor& ta, float scalar, Tensor& out) {
        switch(ta.device()) {
            case Device::HIP:
#if USE_HIP_BACKEND
                sub_scalar_hip_dispatch(ta, scalar, out);
#else
                throw std::runtime_error("HIP backend not enabled.");
#endif
                break;
            case Device::CPU:
                sub_scalar_cpu_dispatch(ta, scalar, out);
                break;
            case Device::CUDA:
#if USE_CUDA_BACKEND
                sub_scalar_cuda_dispatch(ta, scalar, out);
#else
                throw std::runtime_error("CUDA backend not enabled.");
#endif
                break;
            default:
                throw std::runtime_error("Device not supported for sub scalar.");
        }
    };
    
    dispatch(a, b, a);
    return a;
}

Tensor& mul_(Tensor& a, float b) {
    if (a.requires_grad() && autograd::grad_mode_enabled) {
        throw std::runtime_error("In-place operations on leaf variables or variables needed for gradient computation are not yet safe.");
    }
    
    auto dispatch = [&](const Tensor& ta, float scalar, Tensor& out) {
        switch(ta.device()) {
            case Device::HIP:
#if USE_HIP_BACKEND
                mul_scalar_hip_dispatch(ta, scalar, out);
#else
                throw std::runtime_error("HIP backend not enabled.");
#endif
                break;
            case Device::CPU:
                mul_scalar_cpu_dispatch(ta, scalar, out);
                break;
            case Device::CUDA:
#if USE_CUDA_BACKEND
                mul_scalar_cuda_dispatch(ta, scalar, out);
#else
                throw std::runtime_error("CUDA backend not enabled.");
#endif
                break;
            default:
                throw std::runtime_error("Device not supported for mul scalar.");
        }
    };
    
    dispatch(a, b, a);
    return a;
}

// --- Backend Dispatchers ---


// CPU implementations for unary ops
void sqrt_cpu_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out);
void sign_cpu_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out);
void gelu_cpu_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out);
void exp_cpu_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out);
void log_cpu_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out);
void cos_cpu_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out);
void sin_cpu_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out);

// Note: We explicitly dispatch in each function to avoid linker errors with missing backend symbols
// when passing function pointers to a generic helper.

Tensor sqrt(const Tensor& a) {
    bool requires_grad = a.requires_grad() && autograd::grad_mode_enabled;
    Tensor out = empty(a.shape(), a.dtype(), a.device(), requires_grad);
    
    if (a.device() == Device::CPU) {
        sqrt_cpu_dispatch(a, a.strides(), out);
    } else if (a.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        sqrt_cuda_dispatch(a, a.strides(), out);
#else
        throw std::runtime_error("CUDA backend not enabled.");
#endif
    } else if (a.device() == Device::HIP) {
#if USE_HIP_BACKEND
        sqrt_hip_dispatch(a, a.strides(), out);
#else
        throw std::runtime_error("HIP backend not enabled.");
#endif
    } else {
        throw std::runtime_error("Device not supported for sqrt.");
    }
    
    if (requires_grad) {
        auto node = std::make_shared<autograd::Node>();
        node->next_edges.push_back({a.grad_node});
        node->backward_fn = [a_copy=a, weak_out=out.weak()]() mutable {
            auto out_ptr = weak_out.lock();
            if (!out_ptr) return;
            
            if (a_copy.requires_grad()) {
                // grad_a = grad_out / (2 * out)
                auto grad = out_ptr->grad() / (*out_ptr * 2.0f);
                a_copy.accumulate_grad(grad);
            }
        };
        out.grad_node = node;
    }
    return out;
}

Tensor sign(const Tensor& a) {
    bool requires_grad = a.requires_grad() && autograd::grad_mode_enabled;
    Tensor out = empty(a.shape(), a.dtype(), a.device(), requires_grad);
    
    if (a.device() == Device::CPU) {
        sign_cpu_dispatch(a, a.strides(), out);
    } else if (a.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        sign_cuda_dispatch(a, a.strides(), out);
#else
        throw std::runtime_error("CUDA backend not enabled.");
#endif
    } else if (a.device() == Device::HIP) {
#if USE_HIP_BACKEND
        sign_hip_dispatch(a, a.strides(), out);
#else
        throw std::runtime_error("HIP backend not enabled.");
#endif
    } else {
        throw std::runtime_error("Device not supported for sign.");
    }
    
    if (requires_grad) {
        auto node = std::make_shared<autograd::Node>();
        node->next_edges.push_back({a.grad_node});
        node->backward_fn = [a_copy=a]() mutable {
            if (a_copy.requires_grad()) {
                // grad is zero
                a_copy.accumulate_grad(zeros(a_copy.shape(), a_copy.dtype(), a_copy.device())); 
            }
        };
        out.grad_node = node;
    }
    return out;
}

Tensor gelu(const Tensor& a) {
    bool requires_grad = a.requires_grad() && autograd::grad_mode_enabled;
    Tensor out = empty(a.shape(), a.dtype(), a.device(), requires_grad);
    
    if (a.device() == Device::CPU) {
        gelu_cpu_dispatch(a, a.strides(), out);
    } else if (a.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        gelu_cuda_dispatch(a, a.strides(), out);
#else
        throw std::runtime_error("CUDA backend not enabled.");
#endif
    } else if (a.device() == Device::HIP) {
#if USE_HIP_BACKEND
        gelu_hip_dispatch(a, a.strides(), out);
#else
        throw std::runtime_error("HIP backend not enabled.");
#endif
    } else {
        throw std::runtime_error("Device not supported for gelu.");
    }
    
    if (requires_grad) {
        auto node = std::make_shared<autograd::Node>();
        node->next_edges.push_back({a.grad_node});
        
        // Capture input 'a'
        Tensor a_nc = a;
        
        node->backward_fn = [a_nc, weak_out=out.weak()]() mutable {
            auto out_ptr = weak_out.lock();
            if (!out_ptr) return;
            
            if (a_nc.requires_grad()) {
                Tensor grad = out_ptr->grad();
                Tensor grad_input = empty(a_nc.shape(), a_nc.dtype(), a_nc.device());
                
                if (a_nc.device() == Device::CPU) {
                    gelu_backward_cpu_dispatch(grad, a_nc, grad_input);
                } else if (a_nc.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
                    gelu_backward_cuda_dispatch(grad, a_nc, grad_input);
#else
                    throw std::runtime_error("CUDA backend not enabled.");
#endif
                } else if (a_nc.device() == Device::HIP) {
#if USE_HIP_BACKEND
                    gelu_backward_hip_dispatch(grad, a_nc, grad_input);
#else
                    throw std::runtime_error("HIP backend not enabled.");
#endif
                } else {
                    throw std::runtime_error("Device not supported for gelu backward.");
                }
                
                a_nc.accumulate_grad(grad_input);
            }
        };
        out.grad_node = node;
    }
    return out;
}

Tensor exp(const Tensor& a) {
    bool requires_grad = a.requires_grad() && autograd::grad_mode_enabled;
    Tensor out = empty(a.shape(), a.dtype(), a.device(), requires_grad);
    
    if (a.device() == Device::CPU) {
        exp_cpu_dispatch(a, a.strides(), out);
    } else if (a.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        exp_cuda_dispatch(a, a.strides(), out);
#else
        throw std::runtime_error("CUDA backend not enabled.");
#endif
    } else if (a.device() == Device::HIP) {
#if USE_HIP_BACKEND
        exp_hip_dispatch(a, a.strides(), out);
#else
        throw std::runtime_error("HIP backend not enabled.");
#endif
    } else {
        throw std::runtime_error("Device not supported for exp.");
    }
    
    if (requires_grad) {
        auto node = std::make_shared<autograd::Node>();
        node->next_edges.push_back({a.grad_node});
        node->backward_fn = [a_copy=a, weak_out=out.weak()]() mutable {
            auto out_ptr = weak_out.lock();
            if (!out_ptr) return;
            if (a_copy.requires_grad()) {
                // grad_a = grad_out * out
                auto grad = ops::mul(out_ptr->grad(), *out_ptr);
                a_copy.accumulate_grad(grad);
            }
        };
        out.grad_node = node;
    }
    return out;
}

Tensor log(const Tensor& a) {
    bool requires_grad = a.requires_grad() && autograd::grad_mode_enabled;
    Tensor out = empty(a.shape(), a.dtype(), a.device(), requires_grad);
    
    if (a.device() == Device::CPU) {
        log_cpu_dispatch(a, a.strides(), out);
    } else if (a.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        log_cuda_dispatch(a, a.strides(), out);
#else
        throw std::runtime_error("CUDA backend not enabled.");
#endif
    } else if (a.device() == Device::HIP) {
#if USE_HIP_BACKEND
        log_hip_dispatch(a, a.strides(), out);
#else
        throw std::runtime_error("HIP backend not enabled.");
#endif
    } else {
        throw std::runtime_error("Device not supported for log.");
    }
    
    if (requires_grad) {
        auto node = std::make_shared<autograd::Node>();
        node->next_edges.push_back({a.grad_node});
        node->backward_fn = [a_copy=a, weak_out=out.weak()]() mutable {
            auto out_ptr = weak_out.lock();
            if (!out_ptr) return;
            if (a_copy.requires_grad()) {
                // grad_a = grad_out / a
                auto grad = ops::div(out_ptr->grad(), a_copy);
                a_copy.accumulate_grad(grad);
            }
        };
        out.grad_node = node;
    }
    return out;
}

Tensor cos(const Tensor& a) {
    bool requires_grad = a.requires_grad() && autograd::grad_mode_enabled;
    Tensor out = empty(a.shape(), a.dtype(), a.device(), requires_grad);
    
    if (a.device() == Device::CPU) {
        cos_cpu_dispatch(a, a.strides(), out);
#ifdef USE_CUDA_BACKEND
    } else if (a.device() == Device::CUDA) {
        cos_cuda_dispatch(a, a.strides(), out);
#endif
#ifdef USE_HIP_BACKEND
    } else if (a.device() == Device::HIP) {
        cos_hip_dispatch(a, a.strides(), out);
#endif
    } else {
        throw std::runtime_error("cos: unsupported device");
    }
    
    if (requires_grad) {
        auto node = std::make_shared<autograd::Node>();
        node->next_edges.push_back({a.grad_node});
        node->backward_fn = [a_copy=a, weak_out=out.weak()]() mutable {
            auto out_ptr = weak_out.lock();
            if (!out_ptr) return;
            if (a_copy.requires_grad()) {
                // grad_a = grad_out * -sin(a)
                auto neg_sin = ops::mul(ops::sin(a_copy), -1.0f);
                auto grad = ops::mul(out_ptr->grad(), neg_sin);
                a_copy.accumulate_grad(grad);
            }
        };
        out.grad_node = node;
    }
    return out;
}

Tensor sin(const Tensor& a) {
    bool requires_grad = a.requires_grad() && autograd::grad_mode_enabled;
    Tensor out = empty(a.shape(), a.dtype(), a.device(), requires_grad);
    
    if (a.device() == Device::CPU) {
        sin_cpu_dispatch(a, a.strides(), out);
#ifdef USE_CUDA_BACKEND
    } else if (a.device() == Device::CUDA) {
        sin_cuda_dispatch(a, a.strides(), out);
#endif
#ifdef USE_HIP_BACKEND
    } else if (a.device() == Device::HIP) {
        sin_hip_dispatch(a, a.strides(), out);
#endif
    } else {
        throw std::runtime_error("sin: unsupported device");
    }
    
    if (requires_grad) {
        auto node = std::make_shared<autograd::Node>();
        node->next_edges.push_back({a.grad_node});
        node->backward_fn = [a_copy=a, weak_out=out.weak()]() mutable {
            auto out_ptr = weak_out.lock();
            if (!out_ptr) return;
            if (a_copy.requires_grad()) {
                // grad_a = grad_out * cos(a)
                auto grad = ops::mul(out_ptr->grad(), ops::cos(a_copy));
                a_copy.accumulate_grad(grad);
            }
        };
        out.grad_node = node;
    }
    return out;
}

// --- CPU Implementations for Unary ---

template<typename Op>
void cpu_unary_kernel(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out, Op op) {
    size_t n = out.numel();
    float* out_ptr = out.data_ptr<float>();
    const float* a_ptr = a.data_ptr<float>();
    
    // Handle contiguous case optimization
    if (a.is_contiguous() && out.is_contiguous()) {
        for (size_t i = 0; i < n; ++i) {
            out_ptr[i] = op(a_ptr[i]);
        }
        return;
    }
    
    // Generic strided loop
    // ... (Reuse iterator logic or simplify)
    // For MVP, let's do simple recursive or coordinate calculation
    int dims = a.shape().size();
    std::vector<int64_t> indices(dims, 0);
    
    for (size_t i = 0; i < n; ++i) {
        // Compute offset
        size_t offset_a = 0;
        for (int d = 0; d < dims; ++d) {
            offset_a += indices[d] * strides_a[d];
        }
        
        out_ptr[i] = op(a_ptr[offset_a]); // out is contiguous
        
        // Increment indices
        for (int d = dims - 1; d >= 0; --d) {
            indices[d]++;
            if (indices[d] < a.shape()[d]) break;
            indices[d] = 0;
        }
    }
}

void sqrt_cpu_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out) {
    cpu_unary_kernel(a, strides_a, out, [](float x) { return std::sqrt(x); });
}

void sign_cpu_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out) {
    cpu_unary_kernel(a, strides_a, out, [](float x) { return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f); });
}

void gelu_cpu_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out) {
    // 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3)))
    const float SQRT_2_OVER_PI = 0.7978845608f;
    const float COEFF = 0.044715f;
    
    cpu_unary_kernel(a, strides_a, out, [=](float x) {
        float x3 = x * x * x;
        float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
        return 0.5f * x * (1.0f + std::tanh(inner));
    });
}

void exp_cpu_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out) {
    cpu_unary_kernel(a, strides_a, out, [](float x) { return std::exp(x); });
}

void log_cpu_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out) {
    cpu_unary_kernel(a, strides_a, out, [](float x) { return std::log(x); });
}

void cos_cpu_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out) {
    cpu_unary_kernel(a, strides_a, out, [](float x) { return std::cos(x); });
}

void sin_cpu_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out) {
    cpu_unary_kernel(a, strides_a, out, [](float x) { return std::sin(x); });
}

void gelu_backward_cpu_dispatch(const Tensor& grad, const Tensor& input, Tensor& grad_input) {
    const float SQRT_2_OVER_PI = 0.7978845608f;
    const float COEFF = 0.044715f;
    
    // Ensure inputs are contiguous for linear indexing
    Tensor grad_c = grad.is_contiguous() ? grad : grad.contiguous();
    Tensor input_c = input.is_contiguous() ? input : input.contiguous();
    
    size_t n = input_c.numel();
    const float* g_ptr = grad_c.data_ptr<float>();
    const float* x_ptr = input_c.data_ptr<float>();
    float* out_ptr = grad_input.data_ptr<float>();
    
    for (size_t i = 0; i < n; ++i) {
        float x = x_ptr[i];
        float g = g_ptr[i];
        
        float x3 = x * x * x;
        float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
        float tanh_inner = std::tanh(inner);
        
        float dy_dx = SQRT_2_OVER_PI * (1.0f + 3.0f * COEFF * x * x);
        float sech2 = 1.0f - tanh_inner * tanh_inner;
        
        float d_gelu = 0.5f * (1.0f + tanh_inner) + 0.5f * x * sech2 * dy_dx;
        
        out_ptr[i] = g * d_gelu;
    }
}

void gelu_erf_cpu_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out) {
    cpu_unary_kernel(a, strides_a, out, [](float x) {
        return 0.5f * x * (1.0f + std::erf(x * 0.70710678118f));
    });
}

// gelu_backward_cuda_dispatch and gelu_backward_hip_dispatch are implemented in their respective backend files.

void gelu_erf_backward_cpu_dispatch(const Tensor& grad, const Tensor& input, Tensor& grad_input) {
    // 0.5(1 + erf(x/sqrt(2))) + x/sqrt(2pi) * exp(-x^2/2)
    const float INV_SQRT_2 = 0.70710678118f;
    const float INV_SQRT_2PI = 0.3989422804f;
    
    // Ensure inputs are contiguous for linear indexing
    Tensor grad_c = grad.is_contiguous() ? grad : grad.contiguous();
    Tensor input_c = input.is_contiguous() ? input : input.contiguous();
    
    size_t n = input_c.numel();
    const float* g_ptr = grad_c.data_ptr<float>();
    const float* x_ptr = input_c.data_ptr<float>();
    float* out_ptr = grad_input.data_ptr<float>();
    
    for (size_t i = 0; i < n; ++i) {
        float x = x_ptr[i];
        float g = g_ptr[i];
        
        float cdf = 0.5f * (1.0f + std::erf(x * INV_SQRT_2));
        float pdf = INV_SQRT_2PI * std::exp(-0.5f * x * x);
        
        out_ptr[i] = g * (cdf + x * pdf);
    }
}

Tensor gelu_erf(const Tensor& a) {
    bool requires_grad = a.requires_grad() && autograd::grad_mode_enabled;
    Tensor out = empty(a.shape(), a.dtype(), a.device(), requires_grad);
    
    if (a.device() == Device::CPU) {
        gelu_erf_cpu_dispatch(a, a.strides(), out);
    } else if (a.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        gelu_erf_cuda_dispatch(a, a.strides(), out);
#else
        throw std::runtime_error("CUDA backend not enabled.");
#endif
    } else if (a.device() == Device::HIP) {
#if USE_HIP_BACKEND
        gelu_erf_hip_dispatch(a, a.strides(), out);
#else
        throw std::runtime_error("HIP backend not enabled.");
#endif
    } else {
        throw std::runtime_error("Device not supported for gelu_erf.");
    }
    
    if (requires_grad) {
        auto node = std::make_shared<autograd::Node>();
        node->next_edges.push_back({a.grad_node});
        
        Tensor a_nc = a;
        node->backward_fn = [a_nc, weak_out=out.weak()]() mutable {
            auto out_ptr = weak_out.lock();
            if (!out_ptr) return;
            
            if (a_nc.requires_grad()) {
                Tensor grad = out_ptr->grad();
                Tensor grad_input = empty(a_nc.shape(), a_nc.dtype(), a_nc.device());
                
                if (a_nc.device() == Device::CPU) {
                    gelu_erf_backward_cpu_dispatch(grad, a_nc, grad_input);
                } else if (a_nc.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
                    gelu_erf_backward_cuda_dispatch(grad, a_nc, grad_input);
#else
                    throw std::runtime_error("CUDA backend not enabled.");
#endif
                } else if (a_nc.device() == Device::HIP) {
#if USE_HIP_BACKEND
                    gelu_erf_backward_hip_dispatch(grad, a_nc, grad_input);
#else
                    throw std::runtime_error("HIP backend not enabled.");
#endif
                }
                a_nc.accumulate_grad(grad_input);
            }
        };
        out.grad_node = node;
    }
    return out;
}

// ============================================================================
// SiLU (Swish) Activation: silu(x) = x * sigmoid(x)
// ============================================================================

void silu_cpu_dispatch(const Tensor& a, const std::vector<int64_t>& strides_a, Tensor& out) {
    cpu_unary_kernel(a, strides_a, out, [](float x) {
        float sigmoid_x = 1.0f / (1.0f + std::exp(-x));
        return x * sigmoid_x;
    });
}

void silu_backward_cpu_dispatch(const Tensor& grad, const Tensor& input, Tensor& grad_input) {
    // d/dx[silu(x)] = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    //              = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    
    // Ensure inputs are contiguous for linear indexing
    Tensor grad_c = grad.is_contiguous() ? grad : grad.contiguous();
    Tensor input_c = input.is_contiguous() ? input : input.contiguous();
    
    size_t n = input_c.numel();
    const float* g_ptr = grad_c.data_ptr<float>();
    const float* x_ptr = input_c.data_ptr<float>();
    float* out_ptr = grad_input.data_ptr<float>();
    
    for (size_t i = 0; i < n; ++i) {
        float x = x_ptr[i];
        float g = g_ptr[i];
        
        float sigmoid_x = 1.0f / (1.0f + std::exp(-x));
        float d_silu = sigmoid_x * (1.0f + x * (1.0f - sigmoid_x));
        
        out_ptr[i] = g * d_silu;
    }
}

Tensor silu(const Tensor& a) {
    bool requires_grad = a.requires_grad() && autograd::grad_mode_enabled;
    Tensor out = empty(a.shape(), a.dtype(), a.device(), requires_grad);
    
    if (a.device() == Device::CPU) {
        silu_cpu_dispatch(a, a.strides(), out);
    } else if (a.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        silu_cuda_dispatch(a, a.strides(), out);
#else
        throw std::runtime_error("CUDA backend not enabled.");
#endif
    } else if (a.device() == Device::HIP) {
#if USE_HIP_BACKEND
        silu_hip_dispatch(a, a.strides(), out);
#else
        throw std::runtime_error("HIP backend not enabled.");
#endif
    } else {
        throw std::runtime_error("Device not supported for silu.");
    }
    
    if (requires_grad) {
        auto node = std::make_shared<autograd::Node>();
        node->next_edges.push_back({a.grad_node});
        
        Tensor a_nc = a;
        node->backward_fn = [a_nc, weak_out=out.weak()]() mutable {
            auto out_ptr = weak_out.lock();
            if (!out_ptr) return;
            
            if (a_nc.requires_grad()) {
                Tensor grad = out_ptr->grad();
                Tensor grad_input = empty(a_nc.shape(), a_nc.dtype(), a_nc.device());
                
                if (a_nc.device() == Device::CPU) {
                    silu_backward_cpu_dispatch(grad, a_nc, grad_input);
                } else if (a_nc.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
                    silu_backward_cuda_dispatch(grad, a_nc, grad_input);
#else
                    throw std::runtime_error("CUDA backend not enabled.");
#endif
                } else if (a_nc.device() == Device::HIP) {
#if USE_HIP_BACKEND
                    silu_backward_hip_dispatch(grad, a_nc, grad_input);
#else
                    throw std::runtime_error("HIP backend not enabled.");
#endif
                }
                a_nc.accumulate_grad(grad_input);
            }
        };
        out.grad_node = node;
    }
    return out;
}

void silu_(Tensor& a) {
    if (a.requires_grad() && autograd::grad_mode_enabled) {
        throw std::runtime_error("In-place silu_ not supported for tensors requiring gradients.");
    }
    
    if (a.device() == Device::CPU) {
        // In-place on CPU
        size_t n = a.numel();
        float* ptr = a.data_ptr<float>();
        for (size_t i = 0; i < n; ++i) {
            float x = ptr[i];
            float sigmoid_x = 1.0f / (1.0f + std::exp(-x));
            ptr[i] = x * sigmoid_x;
        }
    } else if (a.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        silu_inplace_cuda_dispatch(a);
#else
        throw std::runtime_error("CUDA backend not enabled.");
#endif
    } else if (a.device() == Device::HIP) {
#if USE_HIP_BACKEND
        silu_inplace_hip_dispatch(a);
#else
        throw std::runtime_error("HIP backend not enabled.");
#endif
    } else {
        throw std::runtime_error("Device not supported for silu_.");
    }
}

// =============================================================================
// Fused SiLU * Multiply: silu_mul(gate, up) = silu(gate) * up
// Key operation in SwiGLU - fuses activation and gating into single kernel
// =============================================================================

Tensor silu_mul(const Tensor& gate, const Tensor& up) {
    VESPER_CHECK(gate.shape() == up.shape(), 
                 "silu_mul: gate and up must have same shape");
    VESPER_CHECK(gate.device() == up.device(),
                 "silu_mul: gate and up must be on same device");
    
    bool requires_grad = (gate.requires_grad() || up.requires_grad()) && autograd::grad_mode_enabled;
    Tensor out = empty(gate.shape(), gate.dtype(), gate.device(), requires_grad);
    
    // Make inputs contiguous for optimal kernel performance
    Tensor gate_c = gate.is_contiguous() ? gate : gate.contiguous();
    Tensor up_c = up.is_contiguous() ? up : up.contiguous();
    
    if (gate.device() == Device::CPU) {
        // CPU fallback: use separate ops
        Tensor silu_gate = silu(gate_c);
        Tensor result = mul(silu_gate, up_c);
        out.copy_(result);
    } else if (gate.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        silu_mul_cuda_dispatch(gate_c, up_c, out);
#else
        throw std::runtime_error("CUDA backend not enabled.");
#endif
    } else if (gate.device() == Device::HIP) {
#if USE_HIP_BACKEND
        silu_mul_hip_dispatch(gate_c, up_c, out);
#else
        throw std::runtime_error("HIP backend not enabled.");
#endif
    } else {
        throw std::runtime_error("Device not supported for silu_mul.");
    }
    
    if (requires_grad) {
        auto node = std::make_shared<autograd::Node>();
        if (gate.requires_grad()) node->next_edges.push_back({gate.grad_node});
        if (up.requires_grad()) node->next_edges.push_back({up.grad_node});
        
        Tensor gate_nc = gate_c;
        Tensor up_nc = up_c;
        bool gate_requires_grad = gate.requires_grad();
        bool up_requires_grad = up.requires_grad();
        
        node->backward_fn = [gate_nc, up_nc, gate_requires_grad, up_requires_grad, weak_out=out.weak()]() mutable {
            auto out_ptr = weak_out.lock();
            if (!out_ptr) return;
            
            Tensor grad_out = out_ptr->grad();
            
            if (gate_nc.device() == Device::HIP) {
#if USE_HIP_BACKEND
                Tensor grad_gate, grad_up;
                if (gate_requires_grad) {
                    grad_gate = empty(gate_nc.shape(), gate_nc.dtype(), gate_nc.device());
                }
                if (up_requires_grad) {
                    grad_up = empty(up_nc.shape(), up_nc.dtype(), up_nc.device());
                }
                
                // Use fused backward kernel
                Tensor grad_gate_tmp = empty(gate_nc.shape(), gate_nc.dtype(), gate_nc.device());
                Tensor grad_up_tmp = empty(up_nc.shape(), up_nc.dtype(), up_nc.device());
                silu_mul_backward_hip_dispatch(grad_out, gate_nc, up_nc, grad_gate_tmp, grad_up_tmp);
                
                if (gate_requires_grad) {
                    const_cast<Tensor&>(gate_nc).accumulate_grad(grad_gate_tmp);
                }
                if (up_requires_grad) {
                    const_cast<Tensor&>(up_nc).accumulate_grad(grad_up_tmp);
                }
#endif
            } else {
                // CPU/CUDA: fallback to decomposed backward
                // d/dup = silu(gate)
                // d/dgate = up * d_silu(gate)
                if (up_requires_grad) {
                    Tensor silu_gate = silu(gate_nc);
                    Tensor grad_up = mul(grad_out, silu_gate);
                    const_cast<Tensor&>(up_nc).accumulate_grad(grad_up);
                }
                if (gate_requires_grad) {
                    // d_silu(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
                    Tensor grad_silu = empty(gate_nc.shape(), gate_nc.dtype(), gate_nc.device());
                    if (gate_nc.device() == Device::CPU) {
                        silu_backward_cpu_dispatch(mul(grad_out, up_nc), gate_nc, grad_silu);
                    }
#if USE_CUDA_BACKEND
                    else if (gate_nc.device() == Device::CUDA) {
                        silu_backward_cuda_dispatch(mul(grad_out, up_nc), gate_nc, grad_silu);
                    }
#endif
                    const_cast<Tensor&>(gate_nc).accumulate_grad(grad_silu);
                }
            }
        };
        out.grad_node = node;
    }
    return out;
}

// =============================================================================
// Fused Operations for Optimizer Efficiency
// =============================================================================

// lerp_: self = self * (1 - weight) + other * weight
// Used in Adam for moment updates: m = m * beta + g * (1 - beta)
// Equivalent to: lerp_(m, g, 1 - beta)
Tensor& lerp_(Tensor& self, const Tensor& other, float weight) {
    if (self.requires_grad() && autograd::grad_mode_enabled) {
        throw std::runtime_error("In-place lerp_ not supported for tensors requiring gradients.");
    }
    
    VESPER_CHECK(self.shape() == other.shape(), 
                 "lerp_ requires tensors of same shape");
    VESPER_CHECK(self.device() == other.device(),
                 "lerp_ requires tensors on same device");
    VESPER_CHECK(self.is_contiguous(),
                 "lerp_: self must be contiguous for in-place operation");
    
    // Make other contiguous if needed (can't modify it in-place anyway)
    Tensor other_c = other.is_contiguous() ? other : other.contiguous();
    
    if (self.device() == Device::CPU) {
        size_t n = self.numel();
        float* self_ptr = self.data_ptr<float>();
        const float* other_ptr = other_c.data_ptr<const float>();
        float one_minus_weight = 1.0f - weight;
        for (size_t i = 0; i < n; ++i) {
            self_ptr[i] = self_ptr[i] * one_minus_weight + other_ptr[i] * weight;
        }
    } else if (self.device() == Device::HIP) {
#if USE_HIP_BACKEND
        lerp_hip_dispatch(self, other_c, weight);
#else
        throw std::runtime_error("HIP backend not enabled.");
#endif
    } else if (self.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        lerp_cuda_dispatch(self, other_c, weight);
#else
        throw std::runtime_error("CUDA backend not enabled.");
#endif
    } else {
        throw std::runtime_error("Device not supported for lerp_.");
    }
    
    return self;
}

// addcmul_: self = self + tensor1 * tensor2 * value
// Used in Adam for: v = v + g^2 * (1 - beta2) after v *= beta2
// More efficient version: self = self * scale1 + tensor1 * tensor2 * scale2
// But for simplicity we just do addcmul_ after mul_
Tensor& addcmul_(Tensor& self, const Tensor& tensor1, const Tensor& tensor2, float value) {
    if (self.requires_grad() && autograd::grad_mode_enabled) {
        throw std::runtime_error("In-place addcmul_ not supported for tensors requiring gradients.");
    }
    
    VESPER_CHECK(self.shape() == tensor1.shape() && self.shape() == tensor2.shape(),
                 "addcmul_ requires tensors of same shape");
    VESPER_CHECK(self.device() == tensor1.device() && self.device() == tensor2.device(),
                 "addcmul_ requires tensors on same device");
    VESPER_CHECK(self.is_contiguous(),
                 "addcmul_: self must be contiguous for in-place operation");
    
    // Make inputs contiguous if needed
    Tensor t1_c = tensor1.is_contiguous() ? tensor1 : tensor1.contiguous();
    Tensor t2_c = tensor2.is_contiguous() ? tensor2 : tensor2.contiguous();
    
    if (self.device() == Device::CPU) {
        size_t n = self.numel();
        float* self_ptr = self.data_ptr<float>();
        const float* t1_ptr = t1_c.data_ptr<const float>();
        const float* t2_ptr = t2_c.data_ptr<const float>();
        for (size_t i = 0; i < n; ++i) {
            self_ptr[i] += t1_ptr[i] * t2_ptr[i] * value;
        }
    } else if (self.device() == Device::HIP) {
#if USE_HIP_BACKEND
        addcmul_hip_dispatch(self, t1_c, t2_c, value);
#else
        throw std::runtime_error("HIP backend not enabled.");
#endif
    } else if (self.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        addcmul_cuda_dispatch(self, t1_c, t2_c, value);
#else
        throw std::runtime_error("CUDA backend not enabled.");
#endif
    } else {
        throw std::runtime_error("Device not supported for addcmul_.");
    }
    
    return self;
}

// adam_update_: Fused Adam parameter update
// param = param - lr * (exp_avg / correction1) / (sqrt(exp_avg_sq / correction2) + eps)
// This avoids creating 6 temporary tensors (m_hat, v_hat, sqrt(v_hat), denom, step_size, scaled_step)
void adam_update_(Tensor& param, const Tensor& exp_avg, const Tensor& exp_avg_sq,
                  float lr, float correction1, float correction2, float eps) {
    VESPER_CHECK(param.shape() == exp_avg.shape() && param.shape() == exp_avg_sq.shape(),
                 "adam_update_ requires tensors of same shape");
    VESPER_CHECK(param.device() == exp_avg.device() && param.device() == exp_avg_sq.device(),
                 "adam_update_ requires tensors on same device");
    VESPER_CHECK(param.is_contiguous(),
                 "adam_update_: param must be contiguous for in-place operation");
    
    // Make inputs contiguous if needed
    Tensor m_c = exp_avg.is_contiguous() ? exp_avg : exp_avg.contiguous();
    Tensor v_c = exp_avg_sq.is_contiguous() ? exp_avg_sq : exp_avg_sq.contiguous();
    
    if (param.device() == Device::CPU) {
        size_t n = param.numel();
        float* param_ptr = param.data_ptr<float>();
        const float* m_ptr = m_c.data_ptr<const float>();
        const float* v_ptr = v_c.data_ptr<const float>();
        
        for (size_t i = 0; i < n; ++i) {
            float m_hat = m_ptr[i] / correction1;
            float v_hat = v_ptr[i] / correction2;
            param_ptr[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
        }
    } else if (param.device() == Device::HIP) {
#if USE_HIP_BACKEND
        adam_update_hip_dispatch(param, m_c, v_c, lr, correction1, correction2, eps);
#else
        throw std::runtime_error("HIP backend not enabled.");
#endif
    } else if (param.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        adam_update_cuda_dispatch(param, m_c, v_c, lr, correction1, correction2, eps);
#else
        throw std::runtime_error("CUDA backend not enabled.");
#endif
    } else {
        throw std::runtime_error("Device not supported for adam_update_.");
    }
}

} // namespace vesper::ops