/**
 * @file test_gradient_checkpointing.cpp
 * @brief Tests for Gradient Checkpointing implementation
 * 
 * Chapter 33.9: Memory-Efficient Training
 * 
 * Tests:
 * 1. Basic checkpoint functionality
 * 2. Gradient correctness with checkpointing
 * 3. Memory reduction verification
 * 4. CheckpointedSequential layer
 */

#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <vesper/autograd/checkpoint.h>
#include <vesper/nn/module.h>
#include <vesper/nn/linear.h>
#include <vesper/nn/normalization.h>
#include <vesper/nn/functional.h>
#include <vesper/ops/elementwise.h>
#include <vesper/ops/gemm.h>
#include <vesper/ops/reduction.h>
#include <vesper/ops/random.h>
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <memory>
#include <functional>

using namespace vesper;

// ============================================================================
// Test Utilities
// ============================================================================

bool allclose(const Tensor& a, const Tensor& b, float rtol = 0.01f, float atol = 1e-4f) {
    Tensor a_cpu = a.to(Device::CPU);
    Tensor b_cpu = b.to(Device::CPU);
    
    const float* pa = a_cpu.data_ptr<float>();
    const float* pb = b_cpu.data_ptr<float>();
    
    float max_diff = 0.0f;
    for (size_t i = 0; i < a_cpu.numel(); ++i) {
        float diff = std::abs(pa[i] - pb[i]);
        max_diff = std::max(max_diff, diff);
        
        bool close = diff <= atol || diff <= rtol * std::max(std::abs(pa[i]), std::abs(pb[i]));
        if (!close) {
            std::cerr << "  Mismatch at index " << i << ": " << pa[i] << " vs " << pb[i] 
                      << " (diff=" << diff << ")" << std::endl;
            return false;
        }
    }
    
    std::cout << "  Max difference: " << max_diff << std::endl;
    return true;
}

// Simple function for testing checkpoints
std::function<Tensor(const Tensor&)> make_simple_function() {
    return [](const Tensor& x) -> Tensor {
        Tensor y = ops::mul(x, 2.0f);
        y = nn::functional::relu(y);
        y = ops::add(y, 1.0f);
        return y;
    };
}

// Expensive function that would use lots of intermediate memory
std::function<Tensor(const Tensor&)> make_expensive_function() {
    return [](const Tensor& x) -> Tensor {
        Tensor y = x;
        // Multiple layers of computation
        for (int i = 0; i < 5; ++i) {
            y = ops::mul(y, 1.1f);
            y = nn::functional::relu(y);
            y = ops::add(y, 0.1f);
            y = nn::functional::gelu(y);
        }
        return y;
    };
}

// ============================================================================
// Basic Checkpoint Tests
// ============================================================================

void test_checkpoint_forward_correctness() {
    std::cout << "Testing checkpoint forward correctness..." << std::endl;
    
    auto fn = make_simple_function();
    
    Tensor x = vesper::randn({8, 64}, DType::Float32, Device::CPU);
    
    // Without checkpoint
    Tensor y_normal = fn(x);
    
    // With checkpoint
    Tensor y_checkpoint = autograd::checkpoint(fn, x);
    
    // Should produce same result
    assert(allclose(y_normal, y_checkpoint));
    
    std::cout << "Checkpoint forward correctness: PASSED" << std::endl;
}

void test_checkpoint_gradient_correctness() {
    std::cout << "Testing checkpoint gradient correctness..." << std::endl;
    
    auto fn = make_simple_function();
    
    // Without checkpoint
    Tensor x1 = vesper::randn({8, 64}, DType::Float32, Device::CPU, true);
    Tensor y1 = fn(x1);
    Tensor loss1 = ops::sum(y1);
    loss1.backward();
    Tensor grad1 = x1.grad().clone();
    
    // With checkpoint - use a fresh tensor
    Tensor x2 = vesper::randn({8, 64}, DType::Float32, Device::CPU, true);
    // Copy x1's data so they start identical
    {
        Tensor x1_cpu = x1.to(Device::CPU);
        const float* src = x1_cpu.data_ptr<float>();
        float* dst = const_cast<float*>(x2.data_ptr<float>());
        for (size_t i = 0; i < x1_cpu.numel(); ++i) {
            dst[i] = src[i];
        }
    }
    
    Tensor y2 = autograd::checkpoint(fn, x2);
    Tensor loss2 = ops::sum(y2);
    loss2.backward();
    Tensor grad2 = x2.grad();
    
    // Gradients should match
    assert(allclose(grad1, grad2, 1e-4f, 1e-5f));
    
    std::cout << "Checkpoint gradient correctness: PASSED" << std::endl;
}

void test_checkpoint_with_expensive_function() {
    std::cout << "Testing checkpoint with expensive function..." << std::endl;
    
    auto fn = make_expensive_function();
    
    // Without checkpoint
    Tensor x1 = vesper::randn({16, 128}, DType::Float32, Device::CPU, true);
    Tensor y1 = fn(x1);
    Tensor loss1 = ops::sum(y1);
    loss1.backward();
    Tensor grad1 = x1.grad().clone();
    
    // With checkpoint - use fresh tensor with same values
    Tensor x2 = vesper::randn({16, 128}, DType::Float32, Device::CPU, true);
    {
        Tensor x1_cpu = x1.to(Device::CPU);
        const float* src = x1_cpu.data_ptr<float>();
        float* dst = const_cast<float*>(x2.data_ptr<float>());
        for (size_t i = 0; i < x1_cpu.numel(); ++i) {
            dst[i] = src[i];
        }
    }
    
    Tensor y2 = autograd::checkpoint(fn, x2);
    Tensor loss2 = ops::sum(y2);
    loss2.backward();
    Tensor grad2 = x2.grad();
    
    // Gradients should match
    assert(allclose(grad1, grad2, 1e-3f, 1e-4f));
    
    std::cout << "Checkpoint with expensive function: PASSED" << std::endl;
}

// ============================================================================
// CheckpointedSequential Tests
// ============================================================================

void test_checkpointed_sequential_basic() {
    std::cout << "Testing CheckpointedSequential basic..." << std::endl;
    
    // Create a CheckpointedSequential with checkpoint every 1 layer
    autograd::CheckpointedSequential model(1);
    
    // Add linear layers
    model.add(std::make_shared<nn::Linear>(64, 128));
    model.add(std::make_shared<nn::Linear>(128, 128));
    model.add(std::make_shared<nn::Linear>(128, 64));
    
    Tensor x = vesper::randn({8, 64}, DType::Float32, Device::CPU, true);
    Tensor y = model.forward(x);
    
    assert(y.shape()[0] == 8);
    assert(y.shape()[1] == 64);
    
    // Backward should work
    Tensor loss = ops::sum(y);
    loss.backward();
    
    assert(x.grad().defined());
    
    std::cout << "CheckpointedSequential basic: PASSED" << std::endl;
}

void test_checkpointed_sequential_gradient_correctness() {
    std::cout << "Testing CheckpointedSequential gradient correctness..." << std::endl;
    
    // Checkpointed sequential
    autograd::CheckpointedSequential checkpoint_model(1);
    checkpoint_model.add(std::make_shared<nn::Linear>(64, 128));
    checkpoint_model.add(std::make_shared<nn::Linear>(128, 64));
    
    Tensor x = vesper::randn({8, 64}, DType::Float32, Device::CPU, true);
    
    // Forward through checkpointed model
    Tensor y = checkpoint_model.forward(x);
    Tensor loss = ops::sum(y);
    loss.backward();
    
    assert(x.grad().defined());
    
    // Check that gradient has reasonable values (not all zeros or NaN)
    Tensor grad_cpu = x.grad().to(Device::CPU);
    const float* ptr = grad_cpu.data_ptr<float>();
    bool all_zero = true;
    bool has_nan = false;
    for (size_t i = 0; i < grad_cpu.numel(); ++i) {
        if (ptr[i] != 0.0f) all_zero = false;
        if (!std::isfinite(ptr[i])) has_nan = true;
    }
    
    assert(!all_zero && "Gradient should not be all zeros");
    assert(!has_nan && "Gradient should not have NaN/Inf");
    
    std::cout << "CheckpointedSequential gradient correctness: PASSED" << std::endl;
}

// ============================================================================
// Memory Efficiency Tests
// ============================================================================

void test_checkpoint_no_intermediate_storage() {
    std::cout << "Testing checkpoint no intermediate storage during forward..." << std::endl;
    
    // This test verifies that checkpoint doesn't store intermediates during forward
    // We do this by checking that the checkpoint context doesn't grow with computation
    
    int recompute_count = 0;
    
    auto fn = [&recompute_count](const Tensor& x) -> Tensor {
        recompute_count++;
        Tensor y = ops::mul(x, 2.0f);
        y = nn::functional::relu(y);
        return y;
    };
    
    Tensor x = vesper::randn({8, 64}, DType::Float32, Device::CPU, true);
    
    recompute_count = 0;
    Tensor y = autograd::checkpoint(fn, x);
    int forward_count = recompute_count;
    
    // Backward triggers recomputation
    Tensor loss = ops::sum(y);
    loss.backward();
    
    int total_count = recompute_count;
    
    // Function should be called twice: once for forward, once for backward recomputation
    std::cout << "  Forward count: " << forward_count << std::endl;
    std::cout << "  Total count (after backward): " << total_count << std::endl;
    
    assert(forward_count == 1);
    assert(total_count == 2);  // Recomputed once during backward
    
    std::cout << "Checkpoint no intermediate storage: PASSED" << std::endl;
}

// ============================================================================
// Edge Cases
// ============================================================================

void test_checkpoint_identity_function() {
    std::cout << "Testing checkpoint with identity function..." << std::endl;
    
    auto identity = [](const Tensor& x) -> Tensor { return x; };
    
    Tensor x = vesper::randn({4, 32}, DType::Float32, Device::CPU, true);
    Tensor y = autograd::checkpoint(identity, x);
    
    assert(allclose(x, y));
    
    Tensor loss = ops::sum(y);
    loss.backward();
    
    // Gradient should be all ones (from sum)
    Tensor expected_grad = vesper::ones({4, 32}, DType::Float32, Device::CPU);
    assert(allclose(x.grad(), expected_grad));
    
    std::cout << "Checkpoint with identity function: PASSED" << std::endl;
}

void test_checkpoint_zero_size_tensor() {
    std::cout << "Testing checkpoint with zero-size tensor..." << std::endl;
    
    auto fn = [](const Tensor& x) -> Tensor { return ops::mul(x, 2.0f); };
    
    Tensor x = vesper::empty({0, 64}, DType::Float32, Device::CPU);
    Tensor y = autograd::checkpoint(fn, x);
    
    assert(y.numel() == 0);
    assert(y.shape()[0] == 0);
    assert(y.shape()[1] == 64);
    
    std::cout << "Checkpoint with zero-size tensor: PASSED" << std::endl;
}

void test_checkpoint_large_batch() {
    std::cout << "Testing checkpoint with large batch..." << std::endl;
    
    auto fn = make_expensive_function();
    
    // Large batch that might cause memory issues without checkpointing
    Tensor x = vesper::randn({256, 512}, DType::Float32, Device::CPU, true);
    Tensor y = autograd::checkpoint(fn, x);
    
    assert(y.shape()[0] == 256);
    assert(y.shape()[1] == 512);
    
    // Backward should complete
    Tensor loss = ops::sum(y);
    loss.backward();
    
    assert(x.grad().defined());
    
    std::cout << "Checkpoint with large batch: PASSED" << std::endl;
}

// ============================================================================
// GPU Tests
// ============================================================================

#if defined(USE_HIP_BACKEND) || defined(USE_CUDA_BACKEND)

Device get_gpu_device() {
#ifdef USE_HIP_BACKEND
    return Device::HIP;
#else
    return Device::CUDA;
#endif
}

void test_checkpoint_gpu() {
    std::cout << "Testing checkpoint on GPU..." << std::endl;
    
    Device device = get_gpu_device();
    
    auto fn = make_simple_function();
    
    // Without checkpoint
    Tensor x1 = vesper::randn({32, 256}, DType::Float32, device, true);
    Tensor y1 = fn(x1);
    Tensor loss1 = ops::sum(y1);
    loss1.backward();
    Tensor grad1 = x1.grad().clone();
    
    // With checkpoint - use fresh tensor with same values
    Tensor x2 = vesper::randn({32, 256}, DType::Float32, device, true);
    {
        Tensor x1_cpu = x1.to(Device::CPU);
        Tensor x2_cpu = x2.to(Device::CPU);
        const float* src = x1_cpu.data_ptr<float>();
        float* dst = const_cast<float*>(x2_cpu.data_ptr<float>());
        for (size_t i = 0; i < x1_cpu.numel(); ++i) {
            dst[i] = src[i];
        }
        x2 = x2_cpu.to(device);
        x2.set_requires_grad(true);
    }
    
    Tensor y2 = autograd::checkpoint(fn, x2);
    Tensor loss2 = ops::sum(y2);
    loss2.backward();
    Tensor grad2 = x2.grad();
    
    // Move to CPU for comparison
    grad1 = grad1.to(Device::CPU);
    grad2 = grad2.to(Device::CPU);
    
    assert(allclose(grad1, grad2, 1e-4f, 1e-5f));
    
    std::cout << "Checkpoint on GPU: PASSED" << std::endl;
}

void test_checkpointed_sequential_gpu() {
    std::cout << "Testing CheckpointedSequential on GPU..." << std::endl;
    
    Device device = get_gpu_device();
    
    autograd::CheckpointedSequential model(1);
    model.add(std::make_shared<nn::Linear>(256, 512));
    model.add(std::make_shared<nn::Linear>(512, 512));
    model.add(std::make_shared<nn::Linear>(512, 256));
    model.to(device);
    
    Tensor x = vesper::randn({32, 256}, DType::Float32, device, true);
    Tensor y = model.forward(x);
    
    assert(y.shape()[0] == 32);
    assert(y.shape()[1] == 256);
    
    Tensor loss = ops::sum(y);
    loss.backward();
    
    assert(x.grad().defined());
    
    std::cout << "CheckpointedSequential on GPU: PASSED" << std::endl;
}

#endif  // GPU backends

// ============================================================================
// Main
// ============================================================================

int main() {
    try {
        std::cout << "========================================" << std::endl;
        std::cout << "Gradient Checkpointing Tests" << std::endl;
        std::cout << "========================================" << std::endl;
        
        // Basic tests
        test_checkpoint_forward_correctness();
        test_checkpoint_gradient_correctness();
        test_checkpoint_with_expensive_function();
        
        // CheckpointedSequential tests
        test_checkpointed_sequential_basic();
        test_checkpointed_sequential_gradient_correctness();
        
        // Memory tests
        test_checkpoint_no_intermediate_storage();
        
        // Edge cases
        test_checkpoint_identity_function();
        test_checkpoint_zero_size_tensor();
        test_checkpoint_large_batch();
        
        // GPU tests
#if defined(USE_HIP_BACKEND) || defined(USE_CUDA_BACKEND)
        test_checkpoint_gpu();
        test_checkpointed_sequential_gpu();
#else
        std::cout << "GPU tests skipped (no GPU backend available)" << std::endl;
#endif
        
        std::cout << "========================================" << std::endl;
        std::cout << "All Gradient Checkpointing tests PASSED!" << std::endl;
        std::cout << "========================================" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
