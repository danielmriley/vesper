#include <vesper/ops/elementwise.h>
#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <vesper/core/broadcasting.h>
#include <vesper/autograd/node.h>
#include <vesper/ops/reduction.h>
#include <stdexcept>
#include <iostream>

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

    // Case: Broadcast scalar to anything -> Reduce to scalar
    if (target_shape.size() == 1 && target_shape[0] == 1) {
        return sum(grad);
    }
    if (target_shape.empty()) {
        return sum(grad);
    }
    
    // Case: 2D reductions
    if (grad.shape().size() == 2) {
        // [M, N] -> [N] (Sum Rows)
        bool r_cond1 = target_shape.size() == 1;
        bool r_cond2 = r_cond1 && target_shape[0] == grad.shape()[1];
        
        if (r_cond1 && r_cond2) {
            if (grad.device() == Device::CPU) return sum_rows_cpu(grad);
            #if USE_HIP_BACKEND
            if (grad.device() == Device::HIP) {
                Tensor out = zeros(target_shape, grad.dtype(), grad.device());
                sum_rows_hip_dispatch(grad, out);
                return out;
            }
            #endif
            #if USE_CUDA_BACKEND
            if (grad.device() == Device::CUDA) {
                Tensor out = zeros(target_shape, grad.dtype(), grad.device());
                sum_rows_cuda_dispatch(grad, out);
                return out;
            }
            #endif
        }
        
        // [M, N] -> [M, 1] (Sum Cols)
        bool c_cond1 = target_shape.size() == 2;
        bool c_cond2 = c_cond1 && target_shape[0] == grad.shape()[0];
        bool c_cond3 = c_cond1 && target_shape[1] == 1;
        
        if (c_cond1 && c_cond2 && c_cond3) {
            if (grad.device() == Device::CPU) return sum_cols_cpu(grad);
            #if USE_HIP_BACKEND
            if (grad.device() == Device::HIP) {
                Tensor out = zeros(target_shape, grad.dtype(), grad.device());
                sum_cols_hip_dispatch(grad, out);
                return out;
            }
            #endif
            #if USE_CUDA_BACKEND
            if (grad.device() == Device::CUDA) {
                Tensor out = zeros(target_shape, grad.dtype(), grad.device());
                sum_cols_cuda_dispatch(grad, out);
                return out;
            }
            #endif
        }
    }

    // Fallback for unsupported cases
    // std::cerr << "handle_broadcast_backward failed..." << std::endl;
    throw std::runtime_error("Backward reduction for general broadcasting not fully implemented yet.");
}

template<typename DispatchFn>
Tensor binary_op_impl(const Tensor& a, const Tensor& b, DispatchFn dispatch) {
    if (a.device() != b.device()) {
        throw std::runtime_error("Tensor devices do not match for binary operation.");
    }

    std::vector<int64_t> out_shape = broadcast_shapes(a.shape(), b.shape());
    std::vector<int64_t> strides_a = compute_broadcast_strides(a.shape(), a.strides(), out_shape);
    std::vector<int64_t> strides_b = compute_broadcast_strides(b.shape(), b.strides(), out_shape);

    bool requires_grad = a.requires_grad() || b.requires_grad();
    Tensor result = empty(out_shape, a.dtype(), a.device(), requires_grad);

    dispatch(a, strides_a, b, strides_b, result);

    return result;
}

template<typename DispatchFn>
Tensor scalar_op_impl(const Tensor& a, float b, DispatchFn dispatch) {
    Tensor result = empty(a.shape(), a.dtype(), a.device(), a.requires_grad());
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

Tensor& mul_(Tensor& a, float b) {
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

} // namespace vesper::ops