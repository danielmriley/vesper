/**
 * @file test_inplace_to.cpp
 * @brief Tests for Tensor::to_() in-place device transfer
 * 
 * Tests that:
 * 1. to_() modifies the tensor in-place (same object identity)
 * 2. to_() correctly transfers data between devices
 * 3. Module::to() using to_() properly updates member variables
 */

#include <vesper/core/tensor.h>
#include <vesper/core/device.h>
#include <vesper/core/factories.h>
#include <vesper/nn/module.h>
#include <vesper/nn/linear.h>
#include <iostream>
#include <cassert>
#include <cmath>

using namespace vesper;

// Helper to check tensor values
bool tensors_equal(const Tensor& a, const Tensor& b, float tol = 1e-5f) {
    if (a.shape() != b.shape()) return false;
    
    Tensor a_cpu = (a.device() != Device::CPU) ? a.to(Device::CPU) : a;
    Tensor b_cpu = (b.device() != Device::CPU) ? b.to(Device::CPU) : b;
    
    const float* a_data = a_cpu.data_ptr<float>();
    const float* b_data = b_cpu.data_ptr<float>();
    
    for (int64_t i = 0; i < a.numel(); ++i) {
        if (std::abs(a_data[i] - b_data[i]) > tol) {
            return false;
        }
    }
    return true;
}

void test_to_inplace_same_device() {
    std::cout << "Testing to_() same device (no-op)..." << std::endl;
    
    Tensor t = randn({2, 3}, DType::Float32, Device::CPU, false);
    float* original_data = t.data_ptr<float>();
    
    // to_() on same device should be a no-op
    Tensor& result = t.to_(Device::CPU);
    
    // Should return reference to same tensor
    assert(&result == &t);
    // Data pointer should be unchanged (same storage)
    assert(t.data_ptr<float>() == original_data);
    assert(t.device() == Device::CPU);
    
    std::cout << "  PASSED: Same device to_() is no-op" << std::endl;
}

void test_to_inplace_preserves_identity() {
    std::cout << "Testing to_() preserves object identity..." << std::endl;
    
    Tensor t = randn({2, 3}, DType::Float32, Device::CPU, false);
    Tensor* original_ptr = &t;
    
    // Store original values for comparison
    Tensor original_values = t.clone();
    
#if USE_HIP_BACKEND
    // Transfer to HIP
    Tensor& result = t.to_(Device::HIP);
    
    // Must be same object (reference to t)
    assert(&result == original_ptr);
    
    // Device should have changed
    assert(t.device() == Device::HIP);
    
    // Data should be preserved
    assert(tensors_equal(t, original_values));
    
    // Transfer back to CPU
    t.to_(Device::CPU);
    assert(t.device() == Device::CPU);
    assert(tensors_equal(t, original_values));
    
    std::cout << "  PASSED: to_() preserves object identity (HIP)" << std::endl;
#else
    std::cout << "  SKIPPED: HIP backend not available" << std::endl;
#endif
}

void test_to_inplace_vs_to_copy() {
    std::cout << "Testing to_() vs to() behavior difference..." << std::endl;
    
    Tensor t1 = randn({3, 4}, DType::Float32, Device::CPU, false);
    Tensor t2 = t1.clone();
    
    // Store original pointers
    Tensor* t1_ptr = &t1;
    
#if USE_HIP_BACKEND
    // to() returns NEW tensor, t1 unchanged
    Tensor t1_new = t1.to(Device::HIP);
    assert(&t1_new != &t1);  // Different object
    assert(t1.device() == Device::CPU);  // Original unchanged
    assert(t1_new.device() == Device::HIP);  // New one on HIP
    
    // to_() modifies IN PLACE
    Tensor& t2_ref = t2.to_(Device::HIP);
    assert(&t2_ref == &t2);  // Same object
    assert(t2.device() == Device::HIP);  // Modified in place
    
    std::cout << "  PASSED: to_() modifies in-place, to() creates copy" << std::endl;
#else
    std::cout << "  SKIPPED: HIP backend not available" << std::endl;
#endif
}

void test_module_to_updates_members() {
    std::cout << "Testing Module::to() updates member variables..." << std::endl;
    
    nn::Linear linear(4, 2);
    
    // Store pointer to weight tensor
    Tensor* weight_ptr = &linear.weight;
    Tensor* bias_ptr = &linear.bias;
    
    // Store original values
    Tensor original_weight = linear.weight.clone();
    Tensor original_bias = linear.bias.clone();
    
    assert(linear.weight.device() == Device::CPU);
    assert(linear.bias.device() == Device::CPU);
    
#if USE_HIP_BACKEND
    // Move module to HIP
    linear.to(Device::HIP);
    
    // CRITICAL: Same memory addresses (pointers unchanged)
    assert(&linear.weight == weight_ptr);
    assert(&linear.bias == bias_ptr);
    
    // Devices updated
    assert(linear.weight.device() == Device::HIP);
    assert(linear.bias.device() == Device::HIP);
    
    // Values preserved
    assert(tensors_equal(linear.weight, original_weight));
    assert(tensors_equal(linear.bias, original_bias));
    
    // Forward should work on HIP
    Tensor x = randn({2, 4}, DType::Float32, Device::HIP, false);
    Tensor y = linear.forward(x);
    assert(y.device() == Device::HIP);
    assert(y.shape()[0] == 2);
    assert(y.shape()[1] == 2);
    
    std::cout << "  PASSED: Module::to() updates member tensors in-place" << std::endl;
#else
    std::cout << "  SKIPPED: HIP backend not available" << std::endl;
#endif
}

void test_to_inplace_chaining() {
    std::cout << "Testing to_() method chaining..." << std::endl;
    
#if USE_HIP_BACKEND
    Tensor t = randn({2, 2}, DType::Float32, Device::CPU, false);
    Tensor original = t.clone();
    
    // Chain multiple operations
    t.to_(Device::HIP).mul_(2.0f).to_(Device::CPU);
    
    assert(t.device() == Device::CPU);
    
    // Value should be doubled
    const float* t_data = t.data_ptr<float>();
    const float* orig_data = original.to(Device::CPU).data_ptr<float>();
    for (int64_t i = 0; i < t.numel(); ++i) {
        assert(std::abs(t_data[i] - 2.0f * orig_data[i]) < 1e-5f);
    }
    
    std::cout << "  PASSED: to_() chains correctly with other ops" << std::endl;
#else
    std::cout << "  SKIPPED: HIP backend not available" << std::endl;
#endif
}

void test_to_inplace_shape_preservation() {
    std::cout << "Testing to_() preserves shape and dtype..." << std::endl;
    
    std::vector<int64_t> shape = {2, 3, 4};
    Tensor t = randn(shape, DType::Float32, Device::CPU, false);
    
#if USE_HIP_BACKEND
    t.to_(Device::HIP);
    
    assert(t.shape() == shape);
    assert(t.dtype() == DType::Float32);
    assert(t.ndim() == 3);
    assert(t.numel() == 24);
    
    std::cout << "  PASSED: to_() preserves shape and dtype" << std::endl;
#else
    std::cout << "  SKIPPED: HIP backend not available" << std::endl;
#endif
}

int main() {
    std::cout << "=== Testing In-Place Device Transfer (to_()) ===" << std::endl;
    
    test_to_inplace_same_device();
    test_to_inplace_preserves_identity();
    test_to_inplace_vs_to_copy();
    test_module_to_updates_members();
    test_to_inplace_chaining();
    test_to_inplace_shape_preservation();
    
    std::cout << "\n=== All In-Place Device Transfer Tests Passed! ===" << std::endl;
    return 0;
}
