#include <vesper/nn/transformer.h>
#include <vesper/optim/sgd.h>
#include <vesper/nn/functional.h>
#include <vesper/core/factories.h>
#include <vesper/ops/random.h>
#include <iostream>
#include <cassert>
#include <vector>
#include <numeric>
#include <cmath>

#include <cstdlib>

using namespace vesper;

Tensor randn(const std::vector<int64_t>& shape) {
    Tensor t = vesper::empty(shape, DType::Float32, Device::CPU, true);
    ops::normal_(t, 0.0f, 1.0f);
    return t;
}

bool all_close(const Tensor& a, const Tensor& b, float tol = 1e-4) {
    if (a.numel() != b.numel()) return false;
    std::vector<float> a_data(a.numel());
    std::vector<float> b_data(b.numel());
    a.copy_to_host(a_data.data());
    b.copy_to_host(b_data.data());
    
    for (size_t i = 0; i < a.numel(); ++i) {
        if (std::abs(a_data[i] - b_data[i]) > tol) {
            return false;
        }
    }
    return true;
}

void test_parameter_count() {
    std::cout << "Testing Parameter Count..." << std::endl;
    int embed_dim = 768;
    int num_heads = 12;
    nn::TransformerBlock block(embed_dim, num_heads);

    // Calculate expected parameters
    // Attention:
    // c_attn: (embed_dim, 3 * embed_dim) weight + bias -> 768 * 2304 + 2304
    // c_proj: (embed_dim, embed_dim) weight + bias -> 768 * 768 + 768
    // MLP:
    // c_fc: (embed_dim, 4 * embed_dim) weight + bias -> 768 * 3072 + 3072
    // c_proj: (4 * embed_dim, embed_dim) weight + bias -> 3072 * 768 + 768
    // LayerNorms:
    // ln1: 2 * embed_dim
    // ln2: 2 * embed_dim

    long long expected_params = 0;
    // Attn
    expected_params += (768LL * (3 * 768)) + (3 * 768); // c_attn
    expected_params += (768LL * 768) + 768; // c_proj
    // MLP
    expected_params += (768LL * (4 * 768)) + (4 * 768); // c_fc
    expected_params += ((4 * 768LL) * 768) + 768; // c_proj
    // LNs
    expected_params += 2 * 768; // ln1
    expected_params += 2 * 768; // ln2

    auto params = block.parameters();
    long long actual_params = 0;
    for (auto& p : params) {
        actual_params += p.numel();
    }

    std::cout << "Expected: " << expected_params << ", Actual: " << actual_params << std::endl;
    assert(actual_params == expected_params);
    std::cout << "Parameter Count Passed!" << std::endl;
}

void test_forward_shape() {
    std::cout << "Testing Forward Shape..." << std::endl;
    int embed_dim = 64;
    int num_heads = 4;
    int batch_size = 2;
    int seq_len = 10;

    nn::TransformerBlock block(embed_dim, num_heads);
    Tensor x = randn({batch_size, seq_len, embed_dim});
    Tensor out = block.forward(x);

    assert(out.shape() == std::vector<int64_t>({batch_size, seq_len, embed_dim}));
    std::cout << "Forward Shape Passed!" << std::endl;
}

void test_overfitting() {
    std::cout << "Testing Overfitting..." << std::endl;
    int embed_dim = 32;
    int num_heads = 4;
    int batch_size = 1;
    int seq_len = 5;

    nn::TransformerBlock block(embed_dim, num_heads);
    optim::SGD optimizer(block.parameters(), 0.01);

    Tensor x = randn({batch_size, seq_len, embed_dim});
    Tensor target = randn({batch_size, seq_len, embed_dim});

    for (int i = 0; i < 100; ++i) {
        optimizer.zero_grad();
        Tensor out = block.forward(x);
        Tensor loss = nn::functional::mse_loss(out, target);
        loss.backward();
        optimizer.step();

        if (i % 20 == 0) {
            std::cout << "Step " << i << ", Loss: " << loss.item<float>() << std::endl;
        }
        if (loss.item<float>() < 0.01) {
            std::cout << "Converged at step " << i << std::endl;
            break;
        }
    }
    
    Tensor out = block.forward(x);
    Tensor loss = nn::functional::mse_loss(out, target);
    std::cout << "Final Loss: " << loss.item<float>() << std::endl;
    assert(loss.item<float>() < 0.1); // Should be small
    std::cout << "Overfitting Passed!" << std::endl;
}

void test_kv_cache_equivalence() {
    std::cout << "Testing KV Cache Equivalence..." << std::endl;
    int embed_dim = 32;
    int num_heads = 4;
    int batch_size = 1;
    int seq_len = 10;

    nn::TransformerBlock block(embed_dim, num_heads);
    block.attn.dropout_ = 0.0f;
    block.mlp.dropout_ = 0.0f;

    Tensor x = randn({batch_size, seq_len, embed_dim});

    // 1. Full forward pass (causal=true)
    Tensor out_full = block.forward(x, true);

    // 2. Cached forward pass
    nn::KVCache cache(batch_size, num_heads, seq_len, embed_dim / num_heads, Device::CPU);
    
    std::vector<Tensor> outputs;
    for (int i = 0; i < seq_len; ++i) {
        Tensor x_t = x.index({Slice(), Slice(i, i+1), Slice()});
        Tensor out_t = block.forward(x_t, &cache, i);
        outputs.push_back(out_t);
    }

    for (int i = 0; i < seq_len; ++i) {
        Tensor out_full_t = out_full.index({Slice(), Slice(i, i+1), Slice()});
        Tensor out_cache_t = outputs[i];
        
        if (!all_close(out_full_t, out_cache_t)) {
            std::cout << "Mismatch at step " << i << std::endl;
            assert(false);
        }
    }
    std::cout << "KV Cache Equivalence Passed!" << std::endl;
}

void test_kv_cache_prefill_decode() {
    std::cout << "Testing KV Cache Prefill + Decode..." << std::endl;
    int embed_dim = 32;
    int num_heads = 4;
    int batch_size = 2; // Test batch > 1
    int total_seq_len = 10;
    int prompt_len = 5;

    nn::TransformerBlock block(embed_dim, num_heads);
    block.attn.dropout_ = 0.0f;
    block.mlp.dropout_ = 0.0f;

    Tensor x = randn({batch_size, total_seq_len, embed_dim});

    // 1. Full forward pass (causal=true)
    Tensor out_full = block.forward(x, true);

    // Verify causality: forward(x_prompt) should match out_full_prompt
    Tensor x_prompt = x.index({Slice(), Slice(0, prompt_len), Slice()});
    Tensor out_prompt_ref = block.forward(x_prompt, true);
    Tensor out_full_prompt = out_full.index({Slice(), Slice(0, prompt_len), Slice()});
    
    if (!all_close(out_full_prompt, out_prompt_ref)) {
        std::cout << "Causality check failed! Full pass prefix != Prompt pass" << std::endl;
        float max_diff = 0.0f;
        std::vector<float> d1(out_full_prompt.numel());
        std::vector<float> d2(out_prompt_ref.numel());
        out_full_prompt.copy_to_host(d1.data());
        out_prompt_ref.copy_to_host(d2.data());
        for(size_t k=0; k<d1.size(); ++k) {
            float diff = std::abs(d1[k] - d2[k]);
            if(diff > max_diff) max_diff = diff;
        }
        std::cout << "Max diff: " << max_diff << std::endl;
        std::abort();
    }

    // 2. Prefill + Decode
    nn::KVCache cache(batch_size, num_heads, total_seq_len, embed_dim / num_heads, Device::CPU);
    
    // Prefill step (0 to prompt_len)
    Tensor out_prompt = block.forward(x_prompt, &cache, 0);

    // Verify prompt output matches full output prefix
    if (!all_close(out_full_prompt, out_prompt)) {
        std::cout << "Mismatch at prefill step" << std::endl;
        // Print max diff
        float max_diff = 0.0f;
        std::vector<float> d1(out_full_prompt.numel());
        std::vector<float> d2(out_prompt.numel());
        out_full_prompt.copy_to_host(d1.data());
        out_prompt.copy_to_host(d2.data());
        
        std::cout << "First 10 elements:" << std::endl;
        for(int k=0; k<10; ++k) {
            std::cout << "Full: " << d1[k] << ", Cache: " << d2[k] << ", Diff: " << d1[k]-d2[k] << std::endl;
        }

        int mismatch_idx = -1;
        for(size_t k=0; k<d1.size(); ++k) {
            float diff = std::abs(d1[k] - d2[k]);
            if(diff > max_diff) max_diff = diff;
            if(diff > 1e-4 && mismatch_idx == -1) mismatch_idx = k;
        }
        std::cout << "Max diff: " << max_diff << std::endl;
        if (mismatch_idx != -1) {
            std::cout << "First mismatch at index " << mismatch_idx << ": Full=" << d1[mismatch_idx] << ", Cache=" << d2[mismatch_idx] << std::endl;
            // Calculate coordinates
            // Shape [B, prompt_len, vocab_size]
            int vocab_size = 50257; // Default
            // Wait, what is the shape?
            // out_prompt is [B, prompt_len, vocab_size]
            // Let's print shape
            auto shape = out_prompt.shape();
            std::cout << "Shape: [";
            for(auto s : shape) std::cout << s << ", ";
            std::cout << "]" << std::endl;
            
            int64_t stride_0 = shape[1] * shape[2];
            int64_t stride_1 = shape[2];
            
            int64_t b = mismatch_idx / stride_0;
            int64_t rem = mismatch_idx % stride_0;
            int64_t t = rem / stride_1;
            int64_t c = rem % stride_1;
            
            std::cout << "Mismatch at [b=" << b << ", t=" << t << ", c=" << c << "]" << std::endl;
        }
        std::abort();
    }

    // Decode steps (prompt_len to total_seq_len)
    for (int i = prompt_len; i < total_seq_len; ++i) {
        Tensor x_t = x.index({Slice(), Slice(i, i+1), Slice()});
        Tensor out_t = block.forward(x_t, &cache, i);
        
        Tensor out_full_t = out_full.index({Slice(), Slice(i, i+1), Slice()});
        if (!all_close(out_full_t, out_t)) {
            std::cout << "Mismatch at decode step " << i << std::endl;
            std::abort();
        }
    }

    std::cout << "KV Cache Prefill + Decode Passed!" << std::endl;
}

int main() {
    test_parameter_count();
    test_forward_shape();
    test_overfitting();
    test_kv_cache_equivalence();
    test_kv_cache_prefill_decode();
    return 0;
}
