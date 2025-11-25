#include <vesper/ops/elementwise.h>
#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <vesper/core/broadcasting.h>
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
                a.grad() = add(a.grad(), handle_broadcast_backward(grad_output, a.shape()));
            }
            if (b.requires_grad()) {
                b.grad() = add(b.grad(), handle_broadcast_backward(grad_output, b.shape()));
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
                a.grad() = add(a.grad(), result->grad());
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
                a.grad() = add(a.grad(), handle_broadcast_backward(grad_output, a.shape()));
            }
            if (b.requires_grad()) {
                Tensor reduced = handle_broadcast_backward(grad_output, b.shape());
                b.grad() = sub(b.grad(), reduced);
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
                a.grad() = add(a.grad(), result->grad());
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
                a.grad() = add(a.grad(), handle_broadcast_backward(grad_a_contrib, a.shape()));
            }
            if (b.requires_grad()) {
                Tensor grad_b_contrib = mul(grad_output, a);
                b.grad() = add(b.grad(), handle_broadcast_backward(grad_b_contrib, b.shape()));
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
                a.grad() = add(a.grad(), grad_contrib);
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
                a.grad() = add(a.grad(), handle_broadcast_backward(grad_a_contrib, a.shape()));
            }
            if (b.requires_grad()) {
                Tensor neg_result = mul(*result, -1.0f);
                Tensor grad_b_contrib = mul(grad_output, div(neg_result, b));
                b.grad() = add(b.grad(), handle_broadcast_backward(grad_b_contrib, b.shape()));
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
                a.grad() = add(a.grad(), grad_contrib);
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
    } else {
        throw std::runtime_error("cos only supported on CPU for now");
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
    } else {
        throw std::runtime_error("sin only supported on CPU for now");
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
    
    size_t n = input.numel();
    const float* g_ptr = grad.data_ptr<float>();
    const float* x_ptr = input.data_ptr<float>();
    float* out_ptr = grad_input.data_ptr<float>();
    
    if (grad.is_contiguous() && input.is_contiguous() && grad_input.is_contiguous()) {
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
    } else {
        throw std::runtime_error("GELU backward only supports contiguous tensors for now.");
    }
}

void gelu_backward_cuda_dispatch(const Tensor& grad, const Tensor& input, Tensor& grad_input) {
    throw std::runtime_error("GELU backward CUDA not implemented.");
}

void gelu_backward_hip_dispatch(const Tensor& grad, const Tensor& input, Tensor& grad_input) {
    throw std::runtime_error("GELU backward HIP not implemented.");
}

} // namespace vesper::ops