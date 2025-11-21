#include <vesper/ops/elementwise.h>
#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <vesper/autograd/node.h>
#include <stdexcept>

namespace vesper::ops {

Tensor add(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape() || a.device() != b.device()) {
        throw std::runtime_error("Tensor shapes or devices do not match for add operation.");
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
            for (size_t i = 0; i < n; ++i) {
                res_ptr[i] = a_ptr[i] + b_ptr[i];
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

        node->backward_fn = [a=a, b=b, result=result]() mutable {
            Tensor& grad_output = result.grad();
            
            if (a.requires_grad()) {
                a.grad() = add(a.grad(), grad_output);
            }
            if (b.requires_grad()) {
                b.grad() = add(b.grad(), grad_output);
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

} // namespace vesper::ops
