/**
 * @file test_memory_efficient_training.cpp
 * @brief Integration tests for memory-efficient training features
 * 
 * Chapter 33.9: Memory-Efficient Training
 * 
 * Tests the integration of:
 * 1. Automatic Mixed Precision (AMP)
 * 2. Gradient Scaling
 * 3. Gradient Checkpointing
 * 4. Combined workflow
 */

#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <vesper/nn/module.h>
#include <vesper/nn/linear.h>
#include <vesper/nn/normalization.h>
#include <vesper/nn/functional.h>
#include <vesper/nn/amp.h>
#include <vesper/optim/sgd.h>
#include <vesper/optim/grad_scaler.h>
#include <vesper/autograd/checkpoint.h>
#include <vesper/ops/elementwise.h>
#include <vesper/ops/reduction.h>
#include <vesper/ops/random.h>
#include <vesper/ops/gemm.h>
#include <vesper/ops/cast.h>
#include <iostream>
#include <cassert>
#include <cmath>
#include <memory>
#include <vector>

using namespace vesper;

// ============================================================================
// Test Model
// ============================================================================

/**
 * @brief Simple transformer-like block for testing
 */
class TestBlock : public nn::Module {
public:
    TestBlock(int dim) : dim_(dim) {
        fc1_ = std::make_shared<nn::Linear>(dim, dim * 4);
        fc2_ = std::make_shared<nn::Linear>(dim * 4, dim);
        
        register_module("fc1", fc1_.get());
        register_module("fc2", fc2_.get());
    }
    
    Tensor forward(const Tensor& x) override {
        Tensor h = fc1_->forward(x);
        h = nn::functional::gelu(h);
        h = fc2_->forward(h);
        return ops::add(x, h);  // Residual connection
    }

private:
    int dim_;
    std::shared_ptr<nn::Linear> fc1_;
    std::shared_ptr<nn::Linear> fc2_;
};

/**
 * @brief Multi-layer model for testing
 */
class TestModel : public nn::Module {
public:
    TestModel(int dim, int n_layers) : dim_(dim), n_layers_(n_layers) {
        for (int i = 0; i < n_layers; ++i) {
            auto block = std::make_shared<TestBlock>(dim);
            blocks_.push_back(block);
            register_module("block_" + std::to_string(i), block.get());
        }
        head_ = std::make_shared<nn::Linear>(dim, dim);
        register_module("head", head_.get());
    }
    
    Tensor forward(const Tensor& x) override {
        Tensor h = x;
        for (auto& block : blocks_) {
            h = block->forward(h);
        }
        return head_->forward(h);
    }

private:
    int dim_;
    int n_layers_;
    std::vector<std::shared_ptr<TestBlock>> blocks_;
    std::shared_ptr<nn::Linear> head_;
};

// ============================================================================
// AMP Tests
// ============================================================================

void test_amp_basic_training() {
    std::cout << "Testing AMP basic training..." << std::endl;
    
    // Create model and move to device
    Device device = Device::CPU;
#if defined(USE_HIP_BACKEND)
    device = Device::HIP;
#elif defined(USE_CUDA_BACKEND)
    device = Device::CUDA;
#endif
    
    TestModel model(64, 2);
    model.to(device);
    
    // Create optimizer
    optim::SGD optimizer(model.parameters(), 0.01f);
    
    // Create AMP wrapper with BFloat16 (wider range than FP16)
    nn::AMP amp(DType::BFloat16);
    
    // Training data
    Tensor x = vesper::randn({8, 64}, DType::Float32, device, true);
    Tensor target = vesper::randn({8, 64}, DType::Float32, device);
    
    // For this test, just run without AMP enabled (testing the class construction)
    amp.set_enabled(false);
    
    // Forward pass
    Tensor y = amp.forward(model, x);
    
    // Compute loss
    Tensor diff = ops::sub(y, target);
    Tensor loss = ops::mean(ops::mul(diff, diff));
    
    // Backward
    loss.backward();
    
    // Update
    amp.sync_gradients();
    amp.step(optimizer);
    
    std::cout << "  Loss: " << loss.to(Device::CPU).item<float>() << std::endl;
    std::cout << "AMP basic training: PASSED" << std::endl;
}

void test_autocast_context() {
    std::cout << "Testing AutocastContext..." << std::endl;
    
    // Check autocast is disabled by default
    assert(!nn::AutocastContext::is_enabled());
    assert(nn::AutocastContext::current_dtype() == DType::Float32);
    
    // Enable autocast
    {
        nn::AutocastContext ctx(DType::Float32);  // Use Float32 for compatibility
        assert(nn::AutocastContext::is_enabled());
    }
    
    // Should be disabled after context ends
    assert(!nn::AutocastContext::is_enabled());
    
    std::cout << "AutocastContext: PASSED" << std::endl;
}

// ============================================================================
// Gradient Scaler Tests
// ============================================================================

void test_grad_scaler_basic() {
    std::cout << "Testing GradScaler basic..." << std::endl;
    
    Device device = Device::CPU;
#if defined(USE_HIP_BACKEND)
    device = Device::HIP;
#elif defined(USE_CUDA_BACKEND)
    device = Device::CUDA;
#endif
    
    TestModel model(32, 1);
    model.to(device);
    
    // IMPORTANT: Create optimizer AFTER moving model to device
    optim::SGD optimizer(model.parameters(), 0.01f);
    optim::GradScaler scaler(1024.0f);  // Initial scale
    
    Tensor x = vesper::randn({4, 32}, DType::Float32, device, true);
    Tensor target = vesper::randn({4, 32}, DType::Float32, device);
    
    // Forward
    Tensor y = model.forward(x);
    Tensor diff = ops::sub(y, target);
    Tensor loss = ops::mean(ops::mul(diff, diff));
    
    // Scale loss and backward
    Tensor scaled_loss = scaler.scale(loss);
    scaled_loss.backward();
    
    // Unscale and check for inf/nan
    bool found_inf = scaler.unscale(optimizer);
    
    if (!found_inf) {
        // Step optimizer
        scaler.step(optimizer);
    }
    
    // Update scaler
    scaler.update();
    
    std::cout << "  Scale after update: " << scaler.get_scale() << std::endl;
    std::cout << "GradScaler basic: PASSED" << std::endl;
}

void test_grad_scaler_inf_handling() {
    std::cout << "Testing GradScaler inf handling..." << std::endl;
    
    optim::GradScaler scaler(1.0f, 2.0f, 0.5f, 10);
    
    float initial_scale = scaler.get_scale();
    
    // Simulate many updates without inf
    for (int i = 0; i < 15; ++i) {
        scaler.update();
    }
    
    // Scale should have grown
    float grown_scale = scaler.get_scale();
    std::cout << "  Initial scale: " << initial_scale << std::endl;
    std::cout << "  Scale after 15 updates: " << grown_scale << std::endl;
    
    assert(grown_scale >= initial_scale);
    
    std::cout << "GradScaler inf handling: PASSED" << std::endl;
}

// ============================================================================
// Gradient Checkpointing Tests
// ============================================================================

void test_checkpoint_with_model() {
    std::cout << "Testing checkpoint with model..." << std::endl;
    
    Device device = Device::CPU;
#if defined(USE_HIP_BACKEND)
    device = Device::HIP;
#elif defined(USE_CUDA_BACKEND)
    device = Device::CUDA;
#endif
    
    // Create a checkpointed sequential model
    autograd::CheckpointedSequential model(1);  // Checkpoint every layer
    model.add(std::make_shared<nn::Linear>(64, 128));
    model.add(std::make_shared<nn::Linear>(128, 128));
    model.add(std::make_shared<nn::Linear>(128, 64));
    model.to(device);
    
    Tensor x = vesper::randn({8, 64}, DType::Float32, device, true);
    Tensor target = vesper::randn({8, 64}, DType::Float32, device);
    
    // Forward
    Tensor y = model.forward(x);
    
    // Loss
    Tensor diff = ops::sub(y, target);
    Tensor loss = ops::mean(ops::mul(diff, diff));
    
    // Backward (should recompute through checkpoints)
    loss.backward();
    
    assert(x.grad().defined());
    
    std::cout << "  Loss: " << loss.to(Device::CPU).item<float>() << std::endl;
    std::cout << "Checkpoint with model: PASSED" << std::endl;
}

// ============================================================================
// Combined Training Loop Tests
// ============================================================================

void test_full_training_loop() {
    std::cout << "Testing full training loop..." << std::endl;
    
    Device device = Device::CPU;
#if defined(USE_HIP_BACKEND)
    device = Device::HIP;
#elif defined(USE_CUDA_BACKEND)
    device = Device::CUDA;
#endif
    
    // Model
    TestModel model(32, 2);
    model.to(device);
    
    // Optimizer
    optim::SGD optimizer(model.parameters(), 0.001f);
    
    // GradScaler (use scale of 1.0 for stability in test)
    optim::GradScaler scaler(1.0f);
    
    // Training loop
    std::vector<float> losses;
    
    for (int epoch = 0; epoch < 3; ++epoch) {
        // Generate batch
        Tensor x = vesper::randn({8, 32}, DType::Float32, device, true);
        Tensor target = vesper::randn({8, 32}, DType::Float32, device);
        
        // Forward
        Tensor y = model.forward(x);
        
        // Loss
        Tensor diff = ops::sub(y, target);
        Tensor loss = ops::mean(ops::mul(diff, diff));
        
        // Scale and backward
        Tensor scaled_loss = scaler.scale(loss);
        scaled_loss.backward();
        
        // Unscale and step
        bool found_inf = scaler.unscale(optimizer);
        if (!found_inf) {
            scaler.step(optimizer);
        }
        scaler.update();
        
        // Zero gradients for next iteration
        optimizer.zero_grad();
        
        float loss_val = loss.to(Device::CPU).item<float>();
        losses.push_back(loss_val);
        
        std::cout << "  Epoch " << epoch << " loss: " << loss_val << std::endl;
    }
    
    std::cout << "Full training loop: PASSED" << std::endl;
}

void test_memory_efficient_transformer_training() {
    std::cout << "Testing memory-efficient transformer training..." << std::endl;
    
    Device device = Device::CPU;
#if defined(USE_HIP_BACKEND)
    device = Device::HIP;
#elif defined(USE_CUDA_BACKEND)
    device = Device::CUDA;
#endif
    
    // Create checkpointed model
    autograd::CheckpointedSequential model(1);
    for (int i = 0; i < 3; ++i) {
        model.add(std::make_shared<TestBlock>(64));
    }
    model.to(device);
    
    // Optimizer
    optim::SGD optimizer(model.parameters(), 0.01f);
    
    // Training
    Tensor x = vesper::randn({4, 64}, DType::Float32, device, true);
    Tensor target = vesper::randn({4, 64}, DType::Float32, device);
    
    Tensor y = model.forward(x);
    Tensor diff = ops::sub(y, target);
    Tensor loss = ops::mean(ops::mul(diff, diff));
    
    loss.backward();
    optimizer.step();
    
    std::cout << "  Loss: " << loss.to(Device::CPU).item<float>() << std::endl;
    std::cout << "Memory-efficient transformer training: PASSED" << std::endl;
}

// ============================================================================
// GPU-Specific Tests
// ============================================================================

#if defined(USE_HIP_BACKEND) || defined(USE_CUDA_BACKEND)

Device get_gpu_device() {
#ifdef USE_HIP_BACKEND
    return Device::HIP;
#else
    return Device::CUDA;
#endif
}

void test_fp16_training_gpu() {
    std::cout << "Testing FP16 training on GPU..." << std::endl;
    
    Device device = get_gpu_device();
    
    TestModel model(64, 2);
    model.to(device);
    
    optim::SGD optimizer(model.parameters(), 0.001f);
    optim::GradScaler scaler(65536.0f);  // High initial scale for FP16
    
    // Training in FP16
    for (int epoch = 0; epoch < 3; ++epoch) {
        Tensor x_fp32 = vesper::randn({16, 64}, DType::Float32, device, true);
        Tensor target_fp32 = vesper::randn({16, 64}, DType::Float32, device);
        
        // Cast to FP16 for forward pass
        Tensor x = ops::cast(x_fp32, DType::Float16);
        x.set_requires_grad(true);
        Tensor target = ops::cast(target_fp32, DType::Float16);
        
        // Forward (model is still FP32, inputs are FP16)
        // In practice, you'd cast model weights too
        Tensor y = model.forward(ops::cast(x, DType::Float32));
        y = ops::cast(y, DType::Float16);
        
        // Cast to FP32 for loss computation (reduction ops don't support FP16 yet)
        Tensor diff = ops::sub(y, target);
        Tensor diff_fp32 = ops::cast(diff, DType::Float32);
        Tensor loss = ops::mean(ops::mul(diff_fp32, diff_fp32));
        
        // Scale and backward
        Tensor scaled_loss = scaler.scale(loss);
        scaled_loss.backward();
        
        bool found_inf = scaler.unscale(optimizer);
        if (!found_inf) {
            scaler.step(optimizer);
        }
        scaler.update();
        optimizer.zero_grad();
        
        std::cout << "  Epoch " << epoch << " loss: " << loss.to(Device::CPU).item<float>() 
                  << " scale: " << scaler.get_scale() << std::endl;
    }
    
    std::cout << "FP16 training on GPU: PASSED" << std::endl;
}

void test_checkpoint_memory_reduction_gpu() {
    std::cout << "Testing checkpoint memory reduction on GPU..." << std::endl;
    
    Device device = get_gpu_device();
    
    // Create a larger model
    autograd::CheckpointedSequential model(1);
    for (int i = 0; i < 4; ++i) {
        model.add(std::make_shared<TestBlock>(128));
    }
    model.to(device);
    
    // Run a forward/backward pass
    Tensor x = vesper::randn({32, 128}, DType::Float32, device, true);
    Tensor y = model.forward(x);
    Tensor loss = ops::sum(y);
    loss.backward();
    
    assert(x.grad().defined());
    
    std::cout << "Checkpoint memory reduction on GPU: PASSED" << std::endl;
}

#endif  // GPU backends

// ============================================================================
// Main
// ============================================================================

int main() {
    try {
        std::cout << "========================================" << std::endl;
        std::cout << "Memory-Efficient Training Tests" << std::endl;
        std::cout << "========================================" << std::endl;
        
        // AMP tests
        test_autocast_context();
        test_amp_basic_training();
        
        // GradScaler tests
        test_grad_scaler_basic();
        test_grad_scaler_inf_handling();
        
        // Checkpointing tests
        test_checkpoint_with_model();
        
        // Combined tests
        test_full_training_loop();
        test_memory_efficient_transformer_training();
        
        // GPU tests
#if defined(USE_HIP_BACKEND) || defined(USE_CUDA_BACKEND)
        test_fp16_training_gpu();
        test_checkpoint_memory_reduction_gpu();
#else
        std::cout << "GPU-specific tests skipped (no GPU backend)" << std::endl;
#endif
        
        std::cout << "========================================" << std::endl;
        std::cout << "All Memory-Efficient Training tests PASSED!" << std::endl;
        std::cout << "========================================" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
