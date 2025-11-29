#include <vesper/nn/embedding.h>
#include <vesper/nn/linear.h>
#include <vesper/nn/utils.h>
#include <vesper/core/factories.h>
#include <vesper/optim/sgd.h>
#include <vesper/ops/reduction.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

#if defined(USE_HIP_BACKEND)
    constexpr vesper::Device TEST_DEVICE = vesper::Device::HIP;
#elif defined(USE_CUDA_BACKEND)
    constexpr vesper::Device TEST_DEVICE = vesper::Device::CUDA;
#elif defined(USE_CPU_BACKEND)
    constexpr vesper::Device TEST_DEVICE = vesper::Device::CPU;
#else
    #error "No backend enabled for testing"
#endif

void test_embedding_forward() {
    std::cout << "Testing Embedding Forward..." << std::endl;
    
    int64_t num_embed = 5;
    int64_t dim = 3;
    auto emb = vesper::nn::Embedding(num_embed, dim, -1, -1.0f, 2.0f, false, false, TEST_DEVICE);
    
    std::vector<float> w_data(num_embed * dim);
    for(int i=0; i<num_embed; ++i) {
        for(int j=0; j<dim; ++j) {
            w_data[i*dim + j] = (float)i;
        }
    }
    emb.weight.copy_from_host(w_data.data());
    
    // Note: full supports Float32. We create float then cast.
    auto input = vesper::full({2}, vesper::DType::Float32, TEST_DEVICE, 0.0f).to(vesper::DType::Int32);
    std::vector<int32_t> idx_data = {1, 3};
    input.copy_from_host(idx_data.data());
    
    auto output = emb(input);
    
    assert(output.shape()[0] == 2);
    assert(output.shape()[1] == 3);
    
    std::vector<float> out_data(6);
    output.copy_to_host(out_data.data());
    
    assert(out_data[0] == 1.0f);
    assert(out_data[3] == 3.0f);
    
    std::cout << "Embedding Forward passed!" << std::endl;
}

void test_embedding_backward() {
    std::cout << "Testing Embedding Backward..." << std::endl;
    
    int64_t num_embed = 4;
    int64_t dim = 2;
    auto emb = vesper::nn::Embedding(num_embed, dim, -1, -1.0f, 2.0f, false, false, TEST_DEVICE);
    
    emb.zero_grad();
    
    std::vector<int32_t> idx_data = {0, 2, 0};
    auto input = vesper::full({3}, vesper::DType::Float32, TEST_DEVICE, 0.0f).to(vesper::DType::Int32);
    input.copy_from_host(idx_data.data());
    
    auto output = emb(input);
    auto loss = vesper::ops::sum(output);
    loss.backward();
    
    std::vector<float> grad_w(num_embed * dim);
    emb.weight.grad().copy_to_host(grad_w.data());
    
    assert(grad_w[0] == 2.0f);
    assert(grad_w[1] == 2.0f);
    assert(grad_w[2] == 0.0f);
    assert(grad_w[4] == 1.0f);
    
    std::cout << "Embedding Backward passed!" << std::endl;
}

void test_padding_idx() {
    std::cout << "Testing Padding Index..." << std::endl;
    
    int64_t num_embed = 5;
    int64_t dim = 2;
    int64_t padding_idx = 4;
    auto emb = vesper::nn::Embedding(num_embed, dim, padding_idx, -1.0f, 2.0f, false, false, TEST_DEVICE);
    
    std::vector<float> w_data(num_embed * dim);
    emb.weight.copy_to_host(w_data.data());
    assert(w_data[8] == 0.0f); 
    assert(w_data[9] == 0.0f); 
    
    std::vector<int32_t> idx_data = {1, 4}; 
    auto input = vesper::full({2}, vesper::DType::Float32, TEST_DEVICE, 0.0f).to(vesper::DType::Int32);
    input.copy_from_host(idx_data.data());
    
    auto output = emb(input);
    auto loss = vesper::ops::sum(output);
    loss.backward();
    
    std::vector<float> grad_w(num_embed * dim);
    emb.weight.grad().copy_to_host(grad_w.data());
    
    assert(grad_w[8] == 0.0f);
    assert(grad_w[9] == 0.0f);
    assert(grad_w[2] == 1.0f);
    
    std::cout << "Padding Index passed!" << std::endl;
}

void test_out_of_bounds() {
    std::cout << "Testing Out of Bounds Indices..." << std::endl;
    
    int64_t num_embed = 5;
    int64_t dim = 2;
    auto emb = vesper::nn::Embedding(num_embed, dim, -1, -1.0f, 2.0f, false, false, TEST_DEVICE);
    
    // Index 10 is out of bounds
    std::vector<int32_t> idx_data = {10};
    auto input = vesper::full({1}, vesper::DType::Float32, TEST_DEVICE, 0.0f).to(vesper::DType::Int32);
    input.copy_from_host(idx_data.data());
    
    if (TEST_DEVICE == vesper::Device::CPU) {
        bool caught = false;
        try {
            auto output = emb(input);
        } catch (const std::runtime_error& e) {
            std::cout << "Caught expected CPU error: " << e.what() << std::endl;
            caught = true;
        }
        assert(caught);
    } else {
        // GPU should not crash, output 0
        auto output = emb(input);
        std::vector<float> out_data(dim);
        output.copy_to_host(out_data.data());
        assert(out_data[0] == 0.0f);
        assert(out_data[1] == 0.0f);
        std::cout << "GPU out-of-bounds handled safely (zero output)." << std::endl;
    }
    std::cout << "Out of Bounds passed!" << std::endl;
}

void test_max_norm() {
    std::cout << "Testing Max Norm..." << std::endl;
    
    // Now supported on all backends!

    int64_t num_embed = 2;
    int64_t dim = 2;
    float max_norm = 1.0f;
    // Pass max_norm to constructor
    auto emb = vesper::nn::Embedding(num_embed, dim, -1, max_norm, 2.0f, false, false, TEST_DEVICE);
    
    // Set weights to have norm > 1
    // Row 0: [2.0, 0.0] -> norm 2.0
    // Row 1: [0.5, 0.5] -> norm sqrt(0.5) ~ 0.707 < 1.0
    std::vector<float> w_data = {2.0f, 0.0f, 0.5f, 0.5f};
    emb.weight.copy_from_host(w_data.data());
    
    std::vector<int32_t> idx_data = {0, 1};
    auto input = vesper::full({2}, vesper::DType::Float32, TEST_DEVICE, 0.0f).to(vesper::DType::Int32);
    input.copy_from_host(idx_data.data());
    
    auto output = emb(input);
    
    // Check output
    std::vector<float> out_data(4);
    output.copy_to_host(out_data.data());
    
    // Row 0 should be renormalized to norm 1.0 -> [1.0, 0.0]
    assert(std::abs(out_data[0] - 1.0f) < 1e-5f);
    assert(std::abs(out_data[1] - 0.0f) < 1e-5f);
    
    // Row 1 should be unchanged
    assert(std::abs(out_data[2] - 0.5f) < 1e-5f);
    assert(std::abs(out_data[3] - 0.5f) < 1e-5f);
    
    // Check that weight was modified in-place!
    std::vector<float> w_final(4);
    emb.weight.copy_to_host(w_final.data());
    assert(std::abs(w_final[0] - 1.0f) < 1e-5f);
    
    std::cout << "Max Norm passed!" << std::endl;
}

void test_batch_shape() {
    std::cout << "Testing Batch Shape (2D Input)..." << std::endl;
    
    int64_t num_embed = 10;
    int64_t dim = 4;
    auto emb = vesper::nn::Embedding(num_embed, dim, -1, -1.0f, 2.0f, false, false, TEST_DEVICE);
    
    // Input shape [2, 3] -> Batch size 2, Sequence length 3
    std::vector<int64_t> shape = {2, 3};
    auto input = vesper::empty(shape, vesper::DType::Int32, TEST_DEVICE);
    
    std::vector<int32_t> idx_data = {
        1, 2, 3,
        4, 5, 6
    };
    input.copy_from_host(idx_data.data());
    
    auto output = emb(input);
    
    // Expected output shape: [2, 3, 4]
    assert(output.shape().size() == 3);
    assert(output.shape()[0] == 2);
    assert(output.shape()[1] == 3);
    assert(output.shape()[2] == dim);
    
    std::cout << "Batch Shape passed!" << std::endl;
}

void test_empty_input() {
    std::cout << "Testing Empty Input..." << std::endl;
    
    int64_t num_embed = 5;
    int64_t dim = 3;
    auto emb = vesper::nn::Embedding(num_embed, dim, -1, -1.0f, 2.0f, false, false, TEST_DEVICE);
    
    // Empty input [0]
    auto input = vesper::empty({0}, vesper::DType::Int32, TEST_DEVICE);
    
    auto output = emb(input);
    
    // Expected output shape: [0, 3]
    assert(output.shape().size() == 2);
    assert(output.shape()[0] == 0);
    assert(output.shape()[1] == dim);
    assert(output.numel() == 0);
    
    std::cout << "Empty Input passed!" << std::endl;
}

void test_scale_grad_by_freq() {
    std::cout << "Testing Scale Grad By Freq..." << std::endl;
    
    int64_t num_embed = 5;
    int64_t dim = 2;
    // scale_grad_by_freq = true
    auto emb = vesper::nn::Embedding(num_embed, dim, -1, -1.0f, 2.0f, true, false, TEST_DEVICE);
    
    // Input: index 1 appears twice, index 2 appears once
    std::vector<int32_t> idx_data = {1, 1, 2};
    auto input = vesper::full({3}, vesper::DType::Float32, TEST_DEVICE, 0.0f).to(vesper::DType::Int32);
    input.copy_from_host(idx_data.data());
    
    auto output = emb(input);
    auto loss = vesper::ops::sum(output);
    loss.backward();
    
    std::vector<float> grad_w(num_embed * dim);
    emb.weight.grad().copy_to_host(grad_w.data());
    
    // Index 1: appears 2 times. Normal grad would be 2. Scaled grad should be 2 / 2 = 1.
    // Index 2: appears 1 time. Normal grad would be 1. Scaled grad should be 1 / 1 = 1.
    
    // Check index 1 (row 1)
    assert(std::abs(grad_w[1*dim + 0] - 1.0f) < 1e-5f);
    assert(std::abs(grad_w[1*dim + 1] - 1.0f) < 1e-5f);
    
    // Check index 2 (row 2)
    assert(std::abs(grad_w[2*dim + 0] - 1.0f) < 1e-5f);
    assert(std::abs(grad_w[2*dim + 1] - 1.0f) < 1e-5f);
    
    std::cout << "Scale Grad By Freq passed!" << std::endl;
}

void test_norm_type() {
    std::cout << "Testing Max Norm with L1 Norm..." << std::endl;
    
    // Now supported on all backends!

    int64_t num_embed = 2;
    int64_t dim = 2;
    float max_norm = 1.0f;
    float norm_type = 1.0f; // L1 norm
    
    auto emb = vesper::nn::Embedding(num_embed, dim, -1, max_norm, norm_type, false, false, TEST_DEVICE);
    
    // Row 0: [0.6, 0.6] -> L1 norm = 1.2 > 1.0
    // Should be renormalized to L1 norm 1.0 -> [0.5, 0.5]
    std::vector<float> w_data = {0.6f, 0.6f, 0.1f, 0.1f};
    emb.weight.copy_from_host(w_data.data());
    
    std::vector<int32_t> idx_data = {0};
    auto input = vesper::full({1}, vesper::DType::Float32, TEST_DEVICE, 0.0f).to(vesper::DType::Int32);
    input.copy_from_host(idx_data.data());
    
    auto output = emb(input);
    
    std::vector<float> out_data(2);
    output.copy_to_host(out_data.data());
    
    assert(std::abs(out_data[0] - 0.5f) < 1e-5f);
    assert(std::abs(out_data[1] - 0.5f) < 1e-5f);
    
    std::cout << "Max Norm with L1 Norm passed!" << std::endl;
}

void test_weight_tying() {
    std::cout << "Testing Weight Tying (Embedding <-> Linear)..." << std::endl;
    
    int64_t vocab_size = 10;
    int64_t embed_dim = 4;
    
    // Create embedding and linear layers
    auto emb = vesper::nn::Embedding(vocab_size, embed_dim, -1, -1.0f, 2.0f, false, false, TEST_DEVICE);
    // Linear(in_features=embed_dim, out_features=vocab_size) has weight [vocab_size, embed_dim]
    auto lm_head = vesper::nn::Linear(embed_dim, vocab_size, false);  // no bias
    lm_head.to(TEST_DEVICE);
    
    // Set embedding to known values
    std::vector<float> emb_data(vocab_size * embed_dim);
    for (size_t i = 0; i < emb_data.size(); ++i) {
        emb_data[i] = static_cast<float>(i);
    }
    emb.weight.copy_from_host(emb_data.data());
    
    // Before tying: linear has different weights
    std::vector<float> lin_before(vocab_size * embed_dim);
    lm_head.weight.copy_to_host(lin_before.data());
    assert(lin_before[0] != emb_data[0]);  // Different before tying
    
    // Tie weights
    vesper::nn::utils::tie_weights(emb, lm_head);
    
    // After tying: linear weight should equal embedding weight
    std::vector<float> lin_after(vocab_size * embed_dim);
    lm_head.weight.copy_to_host(lin_after.data());
    
    for (size_t i = 0; i < emb_data.size(); ++i) {
        assert(std::abs(lin_after[i] - emb_data[i]) < 1e-6f);
    }
    
    // Modify embedding, linear should also change (shared storage)
    std::vector<float> new_emb_data(vocab_size * embed_dim);
    for (size_t i = 0; i < new_emb_data.size(); ++i) {
        new_emb_data[i] = static_cast<float>(i) * 2.0f;
    }
    emb.weight.copy_from_host(new_emb_data.data());
    
    std::vector<float> lin_updated(vocab_size * embed_dim);
    lm_head.weight.copy_to_host(lin_updated.data());
    
    for (size_t i = 0; i < new_emb_data.size(); ++i) {
        assert(std::abs(lin_updated[i] - new_emb_data[i]) < 1e-6f);
    }
    
    std::cout << "Weight Tying passed!" << std::endl;
}

int main() {
    test_embedding_forward();
    test_embedding_backward();
    test_padding_idx();
    test_out_of_bounds();
    test_max_norm();
    test_batch_shape();
    test_empty_input();
    test_scale_grad_by_freq();
    test_norm_type();
    test_weight_tying();
    return 0;
}