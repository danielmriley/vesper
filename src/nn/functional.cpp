#include <vesper/nn/functional.h>
#include <vesper/core/factories.h>
#include <vesper/ops/elementwise.h>
#include <vesper/ops/comparison.h>
#include <vesper/ops/reduction.h>
#include <vesper/autograd/guard.h>
#include <cmath>

namespace vesper::nn::functional {

void sigmoid_cpu_dispatch(const Tensor& input, Tensor& output) {
    const float* in_ptr = input.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();
    size_t n = input.numel();
    for (size_t i = 0; i < n; ++i) {
        out_ptr[i] = 1.0f / (1.0f + std::exp(-in_ptr[i]));
    }
}

void relu_cpu_dispatch(const Tensor& input, Tensor& output) {
    const float* in_ptr = input.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();
    size_t n = input.numel();
    for (size_t i = 0; i < n; ++i) {
        out_ptr[i] = in_ptr[i] > 0.0f ? in_ptr[i] : 0.0f;
    }
}

Tensor sigmoid(const Tensor& input) {
    bool result_requires_grad = input.requires_grad() && autograd::grad_mode_enabled;
    Tensor result = empty(input.shape(), input.dtype(), input.device(), result_requires_grad);
    
    if (input.device() == Device::CPU) {
        sigmoid_cpu_dispatch(input, result);
    } else if (input.device() == Device::HIP) {
#if USE_HIP_BACKEND
        sigmoid_hip_dispatch(input, result);
#else
        throw std::runtime_error("HIP backend not enabled.");
#endif
    } else if (input.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        sigmoid_cuda_dispatch(input, result);
#else
        throw std::runtime_error("CUDA backend not enabled.");
#endif
    } else {
        throw std::runtime_error("Device not supported for sigmoid.");
    }

    if (result_requires_grad) {
        result.grad_node = std::make_shared<autograd::Node>();
        result.grad_node->next_edges.push_back({input.grad_node});

        // Capture a non-const copy of input to allow accumulate_grad
        Tensor input_nc = input;

        result.grad_node->backward_fn = [input_nc, result]() mutable {
            // Sigmoid gradient: grad_output * (output * (1 - output))
            // We need `ones` factory.
            // Since `ones` is not in factories.h (I checked earlier), I use `full` with 1.0f.
            auto ones = vesper::full(result.shape(), result.dtype(), result.device(), 1.0f);
            auto term2 = ops::sub(ones, result);
            auto local_grad = ops::mul(result, term2);
            auto final_grad = ops::mul(result.grad(), local_grad);
            input_nc.accumulate_grad(final_grad);
        };
    }
    return result;
}

Tensor relu(const Tensor& input) {
    bool result_requires_grad = input.requires_grad() && autograd::grad_mode_enabled;
    Tensor result = empty(input.shape(), input.dtype(), input.device(), result_requires_grad);
    
    if (input.device() == Device::CPU) {
        relu_cpu_dispatch(input, result);
    } else if (input.device() == Device::HIP) {
#if USE_HIP_BACKEND
        relu_hip_dispatch(input, result);
#else
        throw std::runtime_error("HIP backend not enabled.");
#endif
    } else if (input.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        relu_cuda_dispatch(input, result);
#else
        throw std::runtime_error("CUDA backend not enabled.");
#endif
    } else {
        throw std::runtime_error("Device not supported for relu.");
    }

    if (result_requires_grad) {
        result.grad_node = std::make_shared<autograd::Node>();
        result.grad_node->next_edges.push_back({input.grad_node});
        
        // Capture non-const copy
        Tensor input_nc = input;

        result.grad_node->backward_fn = [input_nc, result]() mutable {
            // ReLU gradient: grad_input = grad_output * (input > 0)
            // 1. Create the mask from the original input
            auto mask = ops::greater_than(input_nc, 0.0f);

            // 2. Multiply the upstream gradient by the mask
            auto final_grad = ops::mul(result.grad(), mask);
            input_nc.accumulate_grad(final_grad);
        };
    }
    return result;
}

Tensor mse_loss(const Tensor& y_pred, const Tensor& y_true) {
    // MSE = mean((y_pred - y_true)^2)
    Tensor diff = ops::sub(y_pred, y_true);
    Tensor sq_diff = ops::mul(diff, diff);
    return ops::mean(sq_diff);
}

} // namespace vesper::nn::functional
