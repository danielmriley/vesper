#pragma once
#include <vesper/core/tensor.h>
#include <vesper/ops/elementwise.h>
#include <vesper/ops/reduction.h>
#include <vesper/core/factories.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <functional>

namespace vesper::test {

// Computes numerical gradient using central difference
// f: function that takes input tensor and returns a scalar loss
// input: the tensor to compute gradient for
// epsilon: perturbation size
inline Tensor compute_numerical_gradient(std::function<float(const Tensor&)> loss_fn, Tensor& input, float epsilon = 1e-3f) {
    std::vector<float> numerical_grads(input.numel());
    std::vector<float> original_data(input.numel());
    input.copy_to_host(original_data.data());

    std::vector<float> perturbed_data = original_data;

    for (size_t i = 0; i < input.numel(); ++i) {
        // f(x + eps)
        perturbed_data[i] = original_data[i] + epsilon;
        input.copy_from_host(perturbed_data.data());
        float loss_plus = loss_fn(input);

        // f(x - eps)
        perturbed_data[i] = original_data[i] - epsilon;
        input.copy_from_host(perturbed_data.data());
        float loss_minus = loss_fn(input);

        // Central difference
        numerical_grads[i] = (loss_plus - loss_minus) / (2 * epsilon);

        // Reset
        perturbed_data[i] = original_data[i];
    }

    // Restore original data
    input.copy_from_host(original_data.data());

    auto grad_tensor = empty(input.shape(), input.dtype(), input.device(), false);
    grad_tensor.copy_from_host(numerical_grads.data());
    return grad_tensor;
}

inline bool check_gradients(const Tensor& analytical, const Tensor& numerical, float tolerance = 1e-3f) {
    if (analytical.numel() != numerical.numel()) return false;
    
    std::vector<float> ana_data(analytical.numel());
    std::vector<float> num_data(numerical.numel());
    
    analytical.copy_to_host(ana_data.data());
    numerical.copy_to_host(num_data.data());

    for (size_t i = 0; i < analytical.numel(); ++i) {
        float diff = std::abs(ana_data[i] - num_data[i]);
        // Use relative error for larger values, absolute for small
        float max_val = std::max(std::abs(ana_data[i]), std::abs(num_data[i]));
        if (max_val > 1.0f) {
            if (diff / max_val > tolerance) {
                std::cerr << "Gradient mismatch at index " << i << ": analytical=" << ana_data[i] 
                          << ", numerical=" << num_data[i] << ", rel_diff=" << diff/max_val << std::endl;
                return false;
            }
        } else {
            if (diff > tolerance) {
                std::cerr << "Gradient mismatch at index " << i << ": analytical=" << ana_data[i] 
                          << ", numerical=" << num_data[i] << ", abs_diff=" << diff << std::endl;
                return false;
            }
        }
    }
    return true;
}

} // namespace vesper::test
