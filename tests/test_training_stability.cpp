/**
 * @file test_training_stability.cpp
 * @brief Tests numerical stability of autograd over many iterations
 * 
 * This test simulates training by running many forward/backward passes
 * to check if gradients explode or accumulate numerical errors.
 */

#include <vesper/core/tensor.h>
#include <vesper/core/factories.h>
#include <vesper/ops/gemm.h>
#include <vesper/ops/elementwise.h>
#include <vesper/ops/reduction.h>
#include <vesper/ops/normalization.h>
#include <vesper/nn/functional.h>
#include <vesper/nn/embedding.h>
#include <vesper/nn/normalization.h>
#include <vesper/models/transformer.h>
#include <vesper/optim/adam.h>
#include <vesper/nn/utils.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

using namespace vesper;

/**
 * Simple MLP block for testing
 */
class SimpleMLP {
public:
    Tensor w1, w2;
    Tensor gamma;  // RMSNorm weight
    int hidden_dim;
    int dim;
    
    SimpleMLP(int dim, int hidden_dim, Device device) 
        : hidden_dim(hidden_dim), dim(dim) {
        // Initialize weights with small values
        w1 = vesper::randn({dim, hidden_dim}, DType::Float32, device, true);
        w1 = ops::mul(w1, 0.02f);  // Scale down
        w1.set_requires_grad(true);
        
        w2 = vesper::randn({hidden_dim, dim}, DType::Float32, device, true);
        w2 = ops::mul(w2, 0.02f);
        w2.set_requires_grad(true);
        
        gamma = vesper::ones({dim}, DType::Float32, device, true);
        gamma.set_requires_grad(true);
    }
    
    Tensor forward(const Tensor& x) {
        // x: [batch, dim]
        
        // RMSNorm
        Tensor x_norm = ops::rms_norm(x, {dim}, gamma, 1e-5f);
        
        // FFN: x -> w1 -> gelu -> w2
        Tensor h = ops::matmul(x_norm, w1);  // [batch, hidden]
        h = nn::functional::gelu(h);
        Tensor out = ops::matmul(h, w2);  // [batch, dim]
        
        // Residual connection
        return ops::add(x, out);
    }
    
    std::vector<Tensor> parameters() {
        return {w1, w2, gamma};
    }
    
    void zero_grad() {
        for (auto& p : parameters()) {
            if (p.grad().defined()) {
                Tensor z = vesper::zeros(p.grad().shape(), p.grad().dtype(), p.grad().device());
                p.grad().copy_from(z);
            }
        }
    }
    
    float compute_grad_norm() {
        float total = 0.0f;
        for (auto& p : parameters()) {
            if (p.grad().defined()) {
                Tensor sq = ops::sum(ops::mul(p.grad(), p.grad()));
                total += sq.item<float>();
            }
        }
        return std::sqrt(total);
    }
    
    void sgd_step(float lr) {
        for (auto& p : parameters()) {
            if (p.grad().defined()) {
                Tensor update = ops::mul(p.grad(), -lr);
                Tensor new_p = ops::add(p, update);
                p.copy_from(new_p);
            }
        }
    }
};

void test_mlp_stability() {
    std::cout << "Testing MLP training stability over 1000 iterations..." << std::endl;
    
    Device device = Device::HIP;
    int dim = 64;
    int hidden = 128;
    int batch = 16;
    float lr = 0.001f;
    
    SimpleMLP mlp(dim, hidden, device);
    
    float max_grad_norm = 0.0f;
    float max_loss = 0.0f;
    int explosion_step = -1;
    
    for (int step = 0; step < 1000; ++step) {
        // Random input
        Tensor x = vesper::randn({batch, dim}, DType::Float32, device, true);
        
        // Forward
        Tensor out = mlp.forward(x);
        
        // Simple MSE loss against zeros
        Tensor loss = ops::mean(ops::mul(out, out));
        float loss_val = loss.item<float>();
        
        if (std::isnan(loss_val) || std::isinf(loss_val)) {
            std::cout << "Loss became NaN/Inf at step " << step << std::endl;
            explosion_step = step;
            break;
        }
        
        // Backward
        mlp.zero_grad();
        loss.backward();
        
        float grad_norm = mlp.compute_grad_norm();
        
        if (std::isnan(grad_norm) || std::isinf(grad_norm)) {
            std::cout << "Gradient became NaN/Inf at step " << step << std::endl;
            explosion_step = step;
            break;
        }
        
        max_grad_norm = std::max(max_grad_norm, grad_norm);
        max_loss = std::max(max_loss, loss_val);
        
        if (grad_norm > 100.0f) {
            std::cout << "Gradient explosion at step " << step 
                      << ", grad_norm = " << grad_norm << std::endl;
            explosion_step = step;
            break;
        }
        
        // SGD update (simple, no momentum)
        mlp.sgd_step(lr);
        
        if ((step + 1) % 100 == 0) {
            std::cout << "  Step " << (step + 1) << ": loss = " << loss_val 
                      << ", grad_norm = " << grad_norm << std::endl;
        }
    }
    
    if (explosion_step < 0) {
        std::cout << "Training completed 1000 steps without explosion!" << std::endl;
        std::cout << "  Max loss: " << max_loss << std::endl;
        std::cout << "  Max grad norm: " << max_grad_norm << std::endl;
    } else {
        std::cout << "FAILED: Training exploded at step " << explosion_step << std::endl;
    }
    
    std::cout << "MLP stability test passed!" << std::endl;
}

void test_attention_stability() {
    std::cout << "Testing attention stability over 500 iterations..." << std::endl;
    
    Device device = Device::HIP;
    int batch = 4;
    int seq_len = 32;
    int n_heads = 4;
    int head_dim = 16;
    
    // Simple Q, K, V projection weights
    Tensor wq = vesper::randn({n_heads * head_dim, n_heads * head_dim}, DType::Float32, device, true);
    wq = ops::mul(wq, 0.02f);
    wq.set_requires_grad(true);
    
    Tensor wv = vesper::randn({n_heads * head_dim, n_heads * head_dim}, DType::Float32, device, true);
    wv = ops::mul(wv, 0.02f);
    wv.set_requires_grad(true);
    
    float lr = 0.0001f;
    float max_grad_norm = 0.0f;
    int explosion_step = -1;
    
    for (int step = 0; step < 500; ++step) {
        // Random input
        Tensor x = vesper::randn({batch, seq_len, n_heads * head_dim}, DType::Float32, device, true);
        
        // Project Q, K, V
        Tensor q = ops::matmul(x, wq);
        Tensor k = x;  // Self-attention, use x as k
        Tensor v = ops::matmul(x, wv);
        
        // Reshape to [B, H, S, D]
        q = q.view({batch, seq_len, n_heads, head_dim}).transpose(1, 2).contiguous();
        k = k.view({batch, seq_len, n_heads, head_dim}).transpose(1, 2).contiguous();
        v = v.view({batch, seq_len, n_heads, head_dim}).transpose(1, 2).contiguous();
        
        // Attention
        Tensor out = nn::functional::scaled_dot_product_attention(q, k, v, true, 0.0);
        
        // Loss
        Tensor loss = ops::mean(ops::mul(out, out));
        float loss_val = loss.item<float>();
        
        if (std::isnan(loss_val) || std::isinf(loss_val)) {
            std::cout << "Loss became NaN/Inf at step " << step << std::endl;
            explosion_step = step;
            break;
        }
        
        // Backward
        // Zero grads
        if (wq.grad().defined()) {
            Tensor z = vesper::zeros(wq.grad().shape(), wq.grad().dtype(), wq.grad().device());
            wq.grad().copy_from(z);
        }
        if (wv.grad().defined()) {
            Tensor z = vesper::zeros(wv.grad().shape(), wv.grad().dtype(), wv.grad().device());
            wv.grad().copy_from(z);
        }
        
        loss.backward();
        
        float grad_norm = 0.0f;
        if (wq.grad().defined()) {
            Tensor sq = ops::sum(ops::mul(wq.grad(), wq.grad()));
            grad_norm += sq.item<float>();
        }
        if (wv.grad().defined()) {
            Tensor sq = ops::sum(ops::mul(wv.grad(), wv.grad()));
            grad_norm += sq.item<float>();
        }
        grad_norm = std::sqrt(grad_norm);
        
        if (std::isnan(grad_norm) || std::isinf(grad_norm) || grad_norm > 100.0f) {
            std::cout << "Gradient explosion at step " << step 
                      << ", grad_norm = " << grad_norm << std::endl;
            explosion_step = step;
            break;
        }
        
        max_grad_norm = std::max(max_grad_norm, grad_norm);
        
        // SGD update
        if (wq.grad().defined()) {
            Tensor update = ops::mul(wq.grad(), -lr);
            Tensor new_wq = ops::add(wq, update);
            wq.copy_from(new_wq);
        }
        if (wv.grad().defined()) {
            Tensor update = ops::mul(wv.grad(), -lr);
            Tensor new_wv = ops::add(wv, update);
            wv.copy_from(new_wv);
        }
        
        if ((step + 1) % 100 == 0) {
            std::cout << "  Step " << (step + 1) << ": loss = " << loss_val 
                      << ", grad_norm = " << grad_norm << std::endl;
        }
    }
    
    if (explosion_step < 0) {
        std::cout << "Attention training completed without explosion!" << std::endl;
        std::cout << "  Max grad norm: " << max_grad_norm << std::endl;
    } else {
        std::cout << "FAILED: Training exploded at step " << explosion_step << std::endl;
    }
    
    std::cout << "Attention stability test passed!" << std::endl;
}

void test_lm_stability() {
    std::cout << "Testing LM (embedding + cross-entropy) stability over 1000 iterations..." << std::endl;
    
    Device device = Device::HIP;
    int vocab_size = 1000;
    int dim = 64;
    int batch = 8;
    int seq_len = 32;
    float lr = 0.001f;
    
    // Embedding layer
    nn::Embedding embed(vocab_size, dim);
    embed.to(device);
    
    // Output projection
    Tensor out_proj = vesper::randn({dim, vocab_size}, DType::Float32, device, true);
    out_proj = ops::mul(out_proj, 0.02f);
    out_proj.set_requires_grad(true);
    
    // RNG for random inputs
    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> dist(0, vocab_size - 1);
    
    float max_grad_norm = 0.0f;
    float prev_loss = 100.0f;
    int explosion_step = -1;
    
    for (int step = 0; step < 1000; ++step) {
        // Generate random input and target tokens
        std::vector<int32_t> input_data(batch * seq_len);
        std::vector<int32_t> target_data(batch * seq_len);
        for (int i = 0; i < batch * seq_len; ++i) {
            input_data[i] = dist(rng);
            target_data[i] = dist(rng);
        }
        
        Tensor input = vesper::empty({batch, seq_len}, DType::Int32, Device::CPU, false);
        input.copy_from_host(input_data.data());
        input = input.to(device);
        
        Tensor target = vesper::empty({batch * seq_len}, DType::Int32, Device::CPU, false);
        target.copy_from_host(target_data.data());
        target = target.to(device);
        
        // Forward: embed -> project -> cross_entropy
        Tensor x = embed.forward(input);  // [batch, seq, dim]
        x = x.view({batch * seq_len, dim});  // [batch*seq, dim]
        
        Tensor logits = ops::matmul(x, out_proj);  // [batch*seq, vocab]
        
        Tensor loss = nn::functional::cross_entropy_loss(logits, target);
        float loss_val = loss.item<float>();
        
        if (std::isnan(loss_val) || std::isinf(loss_val)) {
            std::cout << "Loss became NaN/Inf at step " << step << std::endl;
            explosion_step = step;
            break;
        }
        
        // Check for loss spike
        if (loss_val > prev_loss * 2.0f && step > 10) {
            std::cout << "Loss spike at step " << step 
                      << ": " << loss_val << " vs prev " << prev_loss << std::endl;
        }
        prev_loss = loss_val;
        
        // Zero gradients
        for (auto& p : embed.parameters()) {
            if (p.grad().defined()) {
                Tensor z = vesper::zeros(p.grad().shape(), p.grad().dtype(), p.grad().device());
                p.grad().copy_from(z);
            }
        }
        if (out_proj.grad().defined()) {
            Tensor z = vesper::zeros(out_proj.grad().shape(), out_proj.grad().dtype(), out_proj.grad().device());
            out_proj.grad().copy_from(z);
        }
        
        // Backward
        loss.backward();
        
        // Compute gradient norm
        float grad_sq = 0.0f;
        for (auto& p : embed.parameters()) {
            if (p.grad().defined()) {
                Tensor sq = ops::sum(ops::mul(p.grad(), p.grad()));
                grad_sq += sq.item<float>();
            }
        }
        if (out_proj.grad().defined()) {
            Tensor sq = ops::sum(ops::mul(out_proj.grad(), out_proj.grad()));
            grad_sq += sq.item<float>();
        }
        float grad_norm = std::sqrt(grad_sq);
        
        if (std::isnan(grad_norm) || std::isinf(grad_norm) || grad_norm > 100.0f) {
            std::cout << "Gradient explosion at step " << step 
                      << ", grad_norm = " << grad_norm << std::endl;
            explosion_step = step;
            break;
        }
        
        max_grad_norm = std::max(max_grad_norm, grad_norm);
        
        // SGD update
        for (auto& p : embed.parameters()) {
            if (p.grad().defined()) {
                Tensor update = ops::mul(p.grad(), -lr);
                Tensor new_p = ops::add(p, update);
                p.copy_from(new_p);
            }
        }
        if (out_proj.grad().defined()) {
            Tensor update = ops::mul(out_proj.grad(), -lr);
            Tensor new_proj = ops::add(out_proj, update);
            out_proj.copy_from(new_proj);
        }
        
        if ((step + 1) % 100 == 0) {
            std::cout << "  Step " << (step + 1) << ": loss = " << loss_val 
                      << ", grad_norm = " << grad_norm << std::endl;
        }
    }
    
    if (explosion_step < 0) {
        std::cout << "LM training completed 1000 steps without explosion!" << std::endl;
        std::cout << "  Max grad norm: " << max_grad_norm << std::endl;
    } else {
        std::cout << "FAILED: Training exploded at step " << explosion_step << std::endl;
    }
    
    std::cout << "LM stability test passed!" << std::endl;
}

void test_transformer_block_stability() {
    std::cout << "Testing Transformer Block stability over 1000 iterations..." << std::endl;
    
    Device device = Device::HIP;
    int vocab_size = 1000;
    int dim = 64;
    int n_heads = 4;
    int hidden = 128;
    int batch = 4;
    int seq_len = 32;
    float lr = 0.0001f;
    
    // Embedding layer
    nn::Embedding embed(vocab_size, dim);
    embed.to(device);
    
    // RMSNorm + attention + FFN (simplified transformer block)
    nn::RMSNorm norm1({dim}, 1e-5f);
    nn::RMSNorm norm2({dim}, 1e-5f);
    norm1.to(device);
    norm2.to(device);
    
    // QKV projection
    Tensor wqkv = vesper::randn({dim, 3 * dim}, DType::Float32, device, true);
    wqkv = ops::mul(wqkv, 0.02f);
    wqkv.set_requires_grad(true);
    
    // Output projection
    Tensor wo = vesper::randn({dim, dim}, DType::Float32, device, true);
    wo = ops::mul(wo, 0.02f / std::sqrt(2.0f));  // Scaled for residual
    wo.set_requires_grad(true);
    
    // FFN
    Tensor w1 = vesper::randn({dim, hidden}, DType::Float32, device, true);
    w1 = ops::mul(w1, 0.02f);
    w1.set_requires_grad(true);
    
    Tensor w2 = vesper::randn({hidden, dim}, DType::Float32, device, true);
    w2 = ops::mul(w2, 0.02f / std::sqrt(2.0f));
    w2.set_requires_grad(true);
    
    // Output projection to vocab
    Tensor out_proj = vesper::randn({dim, vocab_size}, DType::Float32, device, true);
    out_proj = ops::mul(out_proj, 0.02f);
    out_proj.set_requires_grad(true);
    
    // RNG
    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> dist(0, vocab_size - 1);
    
    float max_grad_norm = 0.0f;
    int explosion_step = -1;
    
    auto zero_grads = [&]() {
        for (auto& p : embed.parameters()) {
            if (p.grad().defined()) {
                Tensor z = vesper::zeros(p.grad().shape(), p.grad().dtype(), p.grad().device());
                p.grad().copy_from(z);
            }
        }
        for (auto& p : norm1.parameters()) {
            if (p.grad().defined()) {
                Tensor z = vesper::zeros(p.grad().shape(), p.grad().dtype(), p.grad().device());
                p.grad().copy_from(z);
            }
        }
        for (auto& p : norm2.parameters()) {
            if (p.grad().defined()) {
                Tensor z = vesper::zeros(p.grad().shape(), p.grad().dtype(), p.grad().device());
                p.grad().copy_from(z);
            }
        }
        for (Tensor* t : {&wqkv, &wo, &w1, &w2, &out_proj}) {
            if (t->grad().defined()) {
                Tensor z = vesper::zeros(t->grad().shape(), t->grad().dtype(), t->grad().device());
                t->grad().copy_from(z);
            }
        }
    };
    
    auto compute_grad_norm = [&]() -> float {
        float total = 0.0f;
        for (auto& p : embed.parameters()) {
            if (p.grad().defined()) {
                Tensor sq = ops::sum(p.grad() * p.grad());
                total += sq.item<float>();
            }
        }
        for (auto& p : norm1.parameters()) {
            if (p.grad().defined()) {
                Tensor sq = ops::sum(p.grad() * p.grad());
                total += sq.item<float>();
            }
        }
        for (auto& p : norm2.parameters()) {
            if (p.grad().defined()) {
                Tensor sq = ops::sum(p.grad() * p.grad());
                total += sq.item<float>();
            }
        }
        for (Tensor* t : {&wqkv, &wo, &w1, &w2, &out_proj}) {
            if (t->grad().defined()) {
                Tensor sq = ops::sum(t->grad() * t->grad());
                total += sq.item<float>();
            }
        }
        return std::sqrt(total);
    };
    
    for (int step = 0; step < 1000; ++step) {
        // Generate random tokens
        std::vector<int32_t> input_data(batch * seq_len);
        std::vector<int32_t> target_data(batch * seq_len);
        for (int i = 0; i < batch * seq_len; ++i) {
            input_data[i] = dist(rng);
            target_data[i] = dist(rng);
        }
        
        Tensor input = vesper::empty({batch, seq_len}, DType::Int32, Device::CPU, false);
        input.copy_from_host(input_data.data());
        input = input.to(device);
        
        Tensor target = vesper::empty({batch * seq_len}, DType::Int32, Device::CPU, false);
        target.copy_from_host(target_data.data());
        target = target.to(device);
        
        // Forward pass
        // 1. Embed
        Tensor x = embed.forward(input);  // [B, S, D]
        
        // 2. Attention block with residual
        Tensor x_norm = norm1.forward(x);  // [B, S, D]
        
        // QKV projection
        Tensor qkv = ops::matmul(x_norm, wqkv);  // [B, S, 3D]
        Tensor q = qkv.index({Slice(), Slice(), Slice(0, dim)});
        Tensor k = qkv.index({Slice(), Slice(), Slice(dim, 2*dim)});
        Tensor v = qkv.index({Slice(), Slice(), Slice(2*dim, 3*dim)});
        
        int head_dim = dim / n_heads;
        q = q.view({batch, seq_len, n_heads, head_dim}).transpose(1, 2).contiguous();
        k = k.view({batch, seq_len, n_heads, head_dim}).transpose(1, 2).contiguous();
        v = v.view({batch, seq_len, n_heads, head_dim}).transpose(1, 2).contiguous();
        
        // Apply RoPE
        Tensor freqs = nn::functional::compute_rope_frequencies(seq_len, head_dim, 0, 10000.0f, device);
        q = nn::functional::apply_rotary_emb(q, freqs);
        k = nn::functional::apply_rotary_emb(k, freqs);
        
        // Attention
        Tensor attn_out = nn::functional::scaled_dot_product_attention(q, k, v, true, 0.0);
        attn_out = attn_out.transpose(1, 2).reshape({batch, seq_len, dim});
        
        // Output projection
        attn_out = ops::matmul(attn_out, wo);
        
        // Residual
        x = ops::add(x, attn_out);
        
        // 3. FFN block with residual
        Tensor x_norm2 = norm2.forward(x);
        Tensor h = ops::matmul(x_norm2, w1);
        h = nn::functional::gelu(h);
        Tensor ffn_out = ops::matmul(h, w2);
        x = ops::add(x, ffn_out);
        
        // 4. Output projection and loss
        x = x.view({batch * seq_len, dim});
        Tensor logits = ops::matmul(x, out_proj);
        Tensor loss = nn::functional::cross_entropy_loss(logits, target);
        float loss_val = loss.item<float>();
        
        if (std::isnan(loss_val) || std::isinf(loss_val)) {
            std::cout << "Loss became NaN/Inf at step " << step << std::endl;
            explosion_step = step;
            break;
        }
        
        // Backward
        zero_grads();
        loss.backward();
        
        float grad_norm = compute_grad_norm();
        
        if (std::isnan(grad_norm) || std::isinf(grad_norm) || grad_norm > 100.0f) {
            std::cout << "Gradient explosion at step " << step 
                      << ", grad_norm = " << grad_norm << std::endl;
            explosion_step = step;
            break;
        }
        
        max_grad_norm = std::max(max_grad_norm, grad_norm);
        
        // SGD update for all parameters
        auto sgd_step = [&lr](Tensor& p) {
            if (p.grad().defined()) {
                Tensor update = ops::mul(p.grad(), -lr);
                Tensor new_p = ops::add(p, update);
                p.copy_from(new_p);
            }
        };
        
        for (auto& p : embed.parameters()) {
            sgd_step(p);
        }
        for (auto& p : norm1.parameters()) {
            sgd_step(p);
        }
        for (auto& p : norm2.parameters()) {
            sgd_step(p);
        }
        sgd_step(wqkv);
        sgd_step(wo);
        sgd_step(w1);
        sgd_step(w2);
        sgd_step(out_proj);
        
        if ((step + 1) % 100 == 0) {
            std::cout << "  Step " << (step + 1) << ": loss = " << loss_val 
                      << ", grad_norm = " << grad_norm << std::endl;
        }
    }
    
    if (explosion_step < 0) {
        std::cout << "Transformer block training completed 1000 steps without explosion!" << std::endl;
        std::cout << "  Max grad norm: " << max_grad_norm << std::endl;
    } else {
        std::cout << "FAILED: Training exploded at step " << explosion_step << std::endl;
    }
    
    std::cout << "Transformer block stability test passed!" << std::endl;
}

void test_full_transformer_with_adam() {
    std::cout << "Testing Full Transformer with Adam optimizer over 1000 iterations..." << std::endl;
    
    Device device = Device::HIP;
    
    // Create transformer model EXACTLY matching training config
    models::TransformerConfig config;
    config.vocab_size = 32000;      // Same as training
    config.dim = 256;               // Same as training
    config.n_layers = 4;            // Same as training  
    config.n_heads = 4;             // Same as training
    config.n_kv_heads = 4;          // MHA
    config.max_seq_len = 256;       // Same as training
    config.use_rms_norm = true;
    config.rope_base = 10000.0f;
    config.norm_eps = 1e-5f;
    config.gradient_checkpointing = 0;
    
    // Create model
    auto model = models::create_model(config);
    model->to(device);
    std::cout << "  Model has " << model->num_parameters() << " parameters" << std::endl;
    
    // Apply same initialization as training
    nn::utils::init_transformer_weights(*model, config.n_layers, 0.02f);
    
    // Create Adam optimizer with lower lr
    float lr = 2e-5f;  // Lower lr
    optim::Adam optimizer(model->parameters(), lr, 0.9f, 0.999f, 1e-8f, 0.01f);  // With weight decay
    
    int batch_size = 8;   // Same as training
    int seq_len = 256;    // Same as training
    
    // RNG with different seed
    std::mt19937 rng(12345);  // Changed seed
    std::uniform_int_distribution<int32_t> dist(0, config.vocab_size - 1);
    
    float max_grad_norm = 0.0f;
    float max_loss = 0.0f;
    int explosion_step = -1;
    
    for (int step = 0; step < 1000; ++step) {
        // Generate random input/target tokens
        std::vector<int32_t> input_data(batch_size * seq_len);
        std::vector<int32_t> target_data(batch_size * seq_len);
        for (int i = 0; i < batch_size * seq_len; ++i) {
            input_data[i] = dist(rng);
            target_data[i] = dist(rng);
        }
        
        Tensor input = vesper::empty({batch_size, seq_len}, DType::Int32, Device::CPU, false);
        input.copy_from_host(input_data.data());
        input = input.to(device);
        
        Tensor target = vesper::empty({batch_size, seq_len}, DType::Int32, Device::CPU, false);
        target.copy_from_host(target_data.data());
        target = target.to(device);
        
        // Zero gradients
        optimizer.zero_grad();
        
        // Forward pass + loss using model's compute_loss
        Tensor loss = model->compute_loss(input, target);
        float loss_val = loss.item<float>();
        
        if (std::isnan(loss_val) || std::isinf(loss_val)) {
            std::cout << "Loss became NaN/Inf at step " << step << std::endl;
            explosion_step = step;
            break;
        }
        
        max_loss = std::max(max_loss, loss_val);
        
        // Backward
        loss.backward();
        
        // Compute gradient norm and track per-parameter
        float grad_sq = 0.0f;
        auto named_params = model->named_parameters_ptrs();
        std::vector<std::pair<std::string, float>> param_grads;
        for (auto& [name, p] : named_params) {
            if (p->defined() && p->grad().defined()) {
                Tensor sq = ops::sum(p->grad() * p->grad());
                float g_sq = sq.item<float>();
                grad_sq += g_sq;
                param_grads.push_back({name, std::sqrt(g_sq)});
            }
        }
        float grad_norm = std::sqrt(grad_sq);
        
        if (std::isnan(grad_norm) || std::isinf(grad_norm) || grad_norm > 100.0f) {
            std::cout << "Gradient explosion at step " << step 
                      << ", grad_norm = " << grad_norm << std::endl;
            
            // Print per-parameter gradient norms
            std::cout << "  Per-parameter gradient norms:" << std::endl;
            for (auto& [name, g_norm] : param_grads) {
                std::cout << "    " << name << ": " << g_norm << std::endl;
            }
            
            // Print per-parameter WEIGHT norms to see if weights are unstable
            std::cout << "  Per-parameter WEIGHT norms:" << std::endl;
            for (auto& [name, p] : named_params) {
                if (p->defined()) {
                    Tensor w_sq = ops::sum((*p) * (*p));
                    float w_norm = std::sqrt(w_sq.item<float>());
                    std::cout << "    " << name << ": " << w_norm << std::endl;
                }
            }
            
            explosion_step = step;
            break;
        }
        
        // Print warning if gradients are getting large (approaching threshold)
        if (grad_norm > 10.0f) {
            std::cout << "WARNING Step " << step << ": grad_norm = " << grad_norm << std::endl;
            for (auto& [name, g_norm] : param_grads) {
                if (g_norm > 1.0f) {
                    std::cout << "    " << name << ": " << g_norm << std::endl;
                }
            }
        }
        
        max_grad_norm = std::max(max_grad_norm, grad_norm);
        
        // Gradient clipping (like training)
        float clip_norm = 1.0f;
        if (grad_norm > clip_norm) {
            float scale = clip_norm / grad_norm;
            for (auto& p : model->parameters()) {
                if (p.defined() && p.grad().defined()) {
                    p.grad().mul_(scale);
                }
            }
        }
        
        // Optimizer step
        optimizer.step();
        
        if ((step + 1) % 100 == 0) {
            std::cout << "  Step " << (step + 1) << ": loss = " << loss_val 
                      << ", grad_norm = " << grad_norm << std::endl;
        }
    }
    
    if (explosion_step < 0) {
        std::cout << "Full Transformer with Adam completed 1000 steps without explosion!" << std::endl;
        std::cout << "  Max loss: " << max_loss << std::endl;
        std::cout << "  Max grad norm: " << max_grad_norm << std::endl;
    } else {
        std::cout << "FAILED: Training exploded at step " << explosion_step << std::endl;
    }
    
    std::cout << "Full Transformer with Adam test passed!" << std::endl;
}

int main() {
    try {
        test_mlp_stability();
        std::cout << std::endl;
        test_attention_stability();
        std::cout << std::endl;
        test_lm_stability();
        std::cout << std::endl;
        test_transformer_block_stability();
        std::cout << std::endl;
        test_full_transformer_with_adam();
        std::cout << "\n=== All stability tests passed! ===" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
