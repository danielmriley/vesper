#include <vesper/ops/elementwise.h>
#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <vesper/autograd/node.h>
#include <vesper/ops/reduction.h>
#include <stdexcept>

namespace vesper::ops {

// Helper for CPU reduction of broadcasted gradients
// Reduces [M, N] -> [N] by summing over M
Tensor sum_rows_cpu(const Tensor& input) {
    if (input.device() != Device::CPU) throw std::runtime_error("sum_rows_cpu only supports CPU");
    if (input.shape().size() != 2) throw std::runtime_error("sum_rows_cpu only supports 2D tensors");
    
    int64_t M = input.shape()[0];
    int64_t N = input.shape()[1];
    
    Tensor out = zeros({N}, input.dtype(), input.device());
    const float* in_ptr = input.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            out_ptr[j] += in_ptr[i * N + j];
        }
    }
    return out;
}

Tensor add(const Tensor& a, const Tensor& b) {
    bool exact_match = (a.shape() == b.shape());
    bool broadcast_b = false;

    if (!exact_match) {
        if (a.device() != b.device()) {
            throw std::runtime_error("Tensor devices do not match for add operation.");
        }
        // Allow if numel matches (e.g. [1, N] vs [N]) or simple broadcasting
        if (a.numel() == b.numel()) {
            // Treat as exact match for data processing if numel matches
            // This handles [1, N] + [N] where data layout is identical
            exact_match = true; 
        } else if (b.shape().size() == 1 && a.shape().size() > 0 && b.shape()[0] == a.shape().back()) {
             // Simple broadcasting: b is 1D and matches last dim of a
             // e.g. [B, N] + [N]
             broadcast_b = true;
        } else {
            throw std::runtime_error("Tensor shapes do not match for add operation.");
        }
    }

    bool requires_grad = a.requires_grad() || b.requires_grad();
    Tensor result = empty(a.shape(), a.dtype(), a.device(), requires_grad);

    switch(a.device()) {
        case Device::HIP:
            add_hip_dispatch(a, b, result);
            break;
        case Device::CPU: {
            const float* a_ptr = a.data_ptr<float>();
            const float* b_ptr = b.data_ptr<float>();
            float* res_ptr = result.data_ptr<float>();
            size_t n = a.numel();
            size_t b_numel = b.numel();
            
            for (size_t i = 0; i < n; ++i) {
                float val_b = broadcast_b ? b_ptr[i % b_numel] : b_ptr[i];
                res_ptr[i] = a_ptr[i] + val_b;
            }
            break;
        }
        default:
            throw std::runtime_error("Device not supported for add operation.");
    }

    if (requires_grad) {
        auto node = std::make_shared<autograd::Node>();
        
        if (a.requires_grad() && a.grad_node) {
            node->next_edges.push_back({a.grad_node});
        }
        if (b.requires_grad() && b.grad_node) {
            node->next_edges.push_back({b.grad_node});
        }

        node->backward_fn = [a=a, b=b, result=result, broadcast_b=broadcast_b]() mutable {
            Tensor& grad_output = result.grad();
            
            if (a.requires_grad()) {
                a.grad() = add(a.grad(), grad_output);
            }
            if (b.requires_grad()) {
                if (broadcast_b) {
                    // Handle reduction for broadcasting
                    if (grad_output.shape().size() == 2 && b.shape().size() == 1) {
                        if (grad_output.device() == Device::CPU) {
                            Tensor reduced_grad = sum_rows_cpu(grad_output);
                            b.grad() = add(b.grad(), reduced_grad);
                        } else if (grad_output.device() == Device::HIP) {
                            // HIP reduction
                            Tensor reduced_grad = zeros(b.shape(), b.dtype(), b.device());
                            sum_rows_hip_dispatch(grad_output, reduced_grad);
                            b.grad() = add(b.grad(), reduced_grad);
                        } else {
                            throw std::runtime_error("Unsupported device for broadcasting backward.");
                        }
                    } else {
                        // Fallback or error for unsupported cases (e.g. >2D)
                        throw std::runtime_error("Backward for broadcasting only supported for 2D tensors currently.");
                    }
                } else {
                    b.grad() = add(b.grad(), grad_output);
                }
            }
        };
        
        result.grad_node = node;
    }

    return result;
}

Tensor sub(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape() || a.device() != b.device()) {
        throw std::runtime_error("Tensor shapes or devices do not match for sub operation.");
    }

    bool requires_grad = a.requires_grad() || b.requires_grad();
    Tensor result = empty(a.shape(), a.dtype(), a.device(), requires_grad);

    switch(a.device()) {
        case Device::HIP:
            sub_hip_dispatch(a, b, result);
            break;
        case Device::CPU: {
            const float* a_ptr = a.data_ptr<float>();
            const float* b_ptr = b.data_ptr<float>();
            float* res_ptr = result.data_ptr<float>();
            size_t n = a.numel();
            for (size_t i = 0; i < n; ++i) {
                res_ptr[i] = a_ptr[i] - b_ptr[i];
            }
            break;
        }
        default:
            throw std::runtime_error("Device not supported for sub operation.");
    }

    if (requires_grad) {
        auto node = std::make_shared<autograd::Node>();
        
        if (a.requires_grad() && a.grad_node) {
            node->next_edges.push_back({a.grad_node});
        }
        if (b.requires_grad() && b.grad_node) {
            node->next_edges.push_back({b.grad_node});
        }

        node->backward_fn = [a=a, b=b, result=result]() mutable {
            Tensor& grad_output = result.grad();
            
            if (a.requires_grad()) {
                a.grad() = add(a.grad(), grad_output);
            }
            if (b.requires_grad()) {
                b.grad() = sub(b.grad(), grad_output);
            }
        };
        
        result.grad_node = node;
    }

    return result;
}

Tensor mul(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape() || a.device() != b.device()) {
        throw std::runtime_error("Tensor shapes or devices do not match for mul operation.");
    }

    bool requires_grad = a.requires_grad() || b.requires_grad();
    Tensor result = empty(a.shape(), a.dtype(), a.device(), requires_grad);

    switch(a.device()) {
        case Device::HIP:
            mul_hip_dispatch(a, b, result);
            break;
        case Device::CPU: {
            const float* a_ptr = a.data_ptr<float>();
            const float* b_ptr = b.data_ptr<float>();
            float* res_ptr = result.data_ptr<float>();
            size_t n = a.numel();
            for (size_t i = 0; i < n; ++i) {
                res_ptr[i] = a_ptr[i] * b_ptr[i];
            }
            break;
        }
        default:
            throw std::runtime_error("Device not supported for mul operation.");
    }

    if (requires_grad) {
        auto node = std::make_shared<autograd::Node>();
        
        if (a.requires_grad() && a.grad_node) {
            node->next_edges.push_back({a.grad_node});
        }
        if (b.requires_grad() && b.grad_node) {
            node->next_edges.push_back({b.grad_node});
        }

        node->backward_fn = [a=a, b=b, result=result]() mutable {
            Tensor& grad_output = result.grad();
            
            if (a.requires_grad()) {
                Tensor grad_a_contrib = mul(grad_output, b);
                a.grad() = add(a.grad(), grad_a_contrib);
            }
            if (b.requires_grad()) {
                Tensor grad_b_contrib = mul(grad_output, a);
                b.grad() = add(b.grad(), grad_b_contrib);
            }
        };
        
        result.grad_node = node;
    }

    return result;
}

Tensor mul(const Tensor& a, float b) {
    auto b_tensor = full(a.shape(), a.dtype(), a.device(), b);
    return mul(a, b_tensor);
}

Tensor div(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape() || a.device() != b.device()) {
        throw std::runtime_error("Tensor shapes or devices do not match for div operation.");
    }

    bool requires_grad = a.requires_grad() || b.requires_grad();
    Tensor result = empty(a.shape(), a.dtype(), a.device(), requires_grad);

    switch(a.device()) {
        case Device::HIP:
            div_hip_dispatch(a, b, result);
            break;
        case Device::CPU: {
            const float* a_ptr = a.data_ptr<float>();
            const float* b_ptr = b.data_ptr<float>();
            float* res_ptr = result.data_ptr<float>();
            size_t n = a.numel();
            for (size_t i = 0; i < n; ++i) {
                res_ptr[i] = a_ptr[i] / b_ptr[i];
            }
            break;
        }
        default:
            throw std::runtime_error("Device not supported for div operation.");
    }

    if (requires_grad) {
        auto node = std::make_shared<autograd::Node>();
        
        if (a.requires_grad() && a.grad_node) {
            node->next_edges.push_back({a.grad_node});
        }
        if (b.requires_grad() && b.grad_node) {
            node->next_edges.push_back({b.grad_node});
        }

        node->backward_fn = [a=a, b=b, result=result]() mutable {
            Tensor& grad_output = result.grad();
            
            if (a.requires_grad()) {
                // d(a/b)/da = 1/b
                Tensor grad_a_contrib = div(grad_output, b);
                a.grad() = add(a.grad(), grad_a_contrib);
            }
            if (b.requires_grad()) {
                // d(a/b)/db = -a/b^2 = -result/b
                Tensor neg_result = mul(result, -1.0f);
                Tensor grad_b_local = div(neg_result, b);
                Tensor grad_b_contrib = mul(grad_output, grad_b_local);
                b.grad() = add(b.grad(), grad_b_contrib);
            }
        };
        
        result.grad_node = node;
    }

    return result;
}

Tensor div(const Tensor& a, float b) {
    auto b_tensor = full(a.shape(), a.dtype(), a.device(), b);
    return div(a, b_tensor);
}

} // namespace vesper::ops
