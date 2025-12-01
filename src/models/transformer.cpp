/**
 * @file transformer.cpp
 * @brief Implementation of complete Transformer language model
 * 
 * Chapter 33.6: Building a Complete GPT/Llama Model
 */

#include <vesper/models/transformer.h>
#include <vesper/nn/functional.h>
#include <vesper/nn/init.h>
#include <vesper/ops/elementwise.h>
#include <vesper/ops/gemm.h>
#include <vesper/ops/reduction.h>
#include <vesper/core/factories.h>
#include <vesper/core/slice.h>
#include <vesper/autograd/checkpoint.h>
#include <stdexcept>
#include <sstream>
#include <cmath>
#include <random>

namespace vesper::models {

// =============================================================================
// Helper: Create position indices tensor
// =============================================================================
static Tensor create_positions(int64_t start, int64_t end, Device device) {
    // Create [start, start+1, ..., end-1] as int32 (for embedding)
    int64_t size = end - start;
    auto positions = vesper::empty({size}, DType::Int32, Device::CPU, false);
    auto* data = positions.data_ptr<int32_t>();
    for (int64_t i = 0; i < size; ++i) {
        data[i] = static_cast<int32_t>(start + i);
    }
    if (device != Device::CPU) {
        return positions.to(device);
    }
    return positions;
}

// =============================================================================
// TransformerLM Implementation
// =============================================================================

TransformerLM::TransformerLM(const TransformerConfig& config)
    : config_(config),
      tok_emb_(config.vocab_size, config.dim),
      dropout_(config.dropout)
{
    // =========================================================================
    // Token Embeddings
    // =========================================================================
    register_module("tok_embeddings", &tok_emb_);
    
    // =========================================================================
    // Position Embeddings (GPT-2 only)
    // =========================================================================
    if (!config.uses_rope()) {
        pos_emb_ = std::make_unique<nn::Embedding>(config.max_seq_len, config.dim);
        register_module("position_embeddings", pos_emb_.get());
    }
    
    // =========================================================================
    // Transformer Blocks
    // =========================================================================
    blocks_.reserve(static_cast<size_t>(config.n_layers));
    for (int64_t i = 0; i < config.n_layers; ++i) {
        auto block = std::make_unique<ModelTransformerBlock>(config, i);
        register_module("layers." + std::to_string(i), block.get());
        blocks_.push_back(std::move(block));
    }
    
    // =========================================================================
    // Final Normalization
    // =========================================================================
    if (config.use_rms_norm) {
        auto rms = std::make_unique<nn::RMSNorm>(
            std::vector<int64_t>{config.dim}, config.norm_eps);
        register_module("norm", rms.get());
        final_norm_ = std::move(rms);
    } else {
        auto ln = std::make_unique<nn::LayerNorm>(
            std::vector<int64_t>{config.dim}, config.norm_eps);
        register_module("ln_f", ln.get());
        final_norm_ = std::move(ln);
    }
    
    // =========================================================================
    // Output Projection (LM Head)
    // =========================================================================
    if (!config.tie_word_embeddings) {
        lm_head_ = std::make_unique<nn::Linear>(config.dim, config.vocab_size, false);
        register_module("output", lm_head_.get());
    }
    // If tie_word_embeddings, we'll use tok_emb_.weight.transpose() in forward
}

Tensor TransformerLM::forward(const Tensor& tokens) {
    // tokens: [Batch, SeqLen] of int64
    int64_t B = tokens.shape()[0];
    int64_t S = tokens.shape()[1];
    
    if (S > config_.max_seq_len) {
        throw std::runtime_error(
            "Sequence length " + std::to_string(S) + 
            " exceeds maximum " + std::to_string(config_.max_seq_len));
    }
    
    // 1. Token embeddings: [B, S, D]
    Tensor h = tok_emb_.forward(tokens);
    
    // 2. Position embeddings (GPT-2 only)
    if (pos_emb_) {
        // Create position indices [0, 1, 2, ..., S-1]
        Tensor positions = create_positions(0, S, tokens.device());
        Tensor pos_embed = pos_emb_->forward(positions);  // [S, D]
        h = ops::add(h, pos_embed);  // Broadcast: [B, S, D] + [S, D]
    }
    
    // 3. Dropout after embeddings
    if (dropout_ > 0.0f && training_) {
        h = nn::functional::dropout(h, dropout_, true);
    }
    
    // 4. Transformer blocks (with optional gradient checkpointing)
    for (size_t i = 0; i < blocks_.size(); ++i) {
        if (config_.gradient_checkpointing > 0 && training_ && 
            (i % config_.gradient_checkpointing == 0)) {
            // Checkpoint this layer - recompute during backward
            auto& block = blocks_[i];
            h = autograd::checkpoint([&block](const Tensor& x) {
                return block->forward(x);
            }, h);
        } else {
            h = blocks_[i]->forward(h);
        }
    }
    
    // 5. Final normalization
    h = final_norm_->forward(h);
    
    // 6. Output projection to vocab
    if (lm_head_) {
        return lm_head_->forward(h);  // [B, S, V]
    } else {
        // Tied embeddings: h @ embedding_weight.T
        // tok_emb_.weight: [V, D], need [D, V]
        return ops::matmul(h, tok_emb_.weight.transpose(-2, -1));
    }
}

Tensor TransformerLM::forward_with_cache(const Tensor& tokens, int64_t start_pos) {
    // tokens: [Batch, NewTokens]
    int64_t B = tokens.shape()[0];
    int64_t S = tokens.shape()[1];
    
    if (kv_caches_.empty()) {
        throw std::runtime_error("KV cache not initialized. Call init_cache first.");
    }
    
    // 1. Token embeddings
    Tensor h = tok_emb_.forward(tokens);
    
    // 2. Position embeddings (GPT-2 only)
    if (pos_emb_) {
        // Positions for this chunk: [start_pos, start_pos+1, ..., start_pos+S-1]
        Tensor positions = create_positions(start_pos, start_pos + S, tokens.device());
        Tensor pos_embed = pos_emb_->forward(positions);
        h = ops::add(h, pos_embed);
    }
    
    // No dropout during inference
    
    // 3. Transformer blocks with caching
    for (size_t i = 0; i < blocks_.size(); ++i) {
        h = blocks_[i]->forward(h, kv_caches_[i].get(), start_pos);
    }
    
    // 4. Final normalization
    h = final_norm_->forward(h);
    
    // 5. Output projection
    if (lm_head_) {
        return lm_head_->forward(h);
    } else {
        return ops::matmul(h, tok_emb_.weight.transpose(-2, -1));
    }
}

void TransformerLM::init_cache(int64_t batch_size, Device device) {
    kv_caches_.clear();
    kv_caches_.reserve(static_cast<size_t>(config_.n_layers));
    
    for (int64_t i = 0; i < config_.n_layers; ++i) {
        kv_caches_.push_back(std::make_unique<nn::KVCache>(
            static_cast<int>(batch_size),
            static_cast<int>(config_.kv_heads()),
            static_cast<int>(config_.max_seq_len),
            static_cast<int>(config_.head_dim()),
            device));
    }
}

void TransformerLM::clear_cache() {
    kv_caches_.clear();
}

Tensor TransformerLM::generate(const Tensor& prompt, int64_t max_new_tokens,
                               float temperature, int64_t top_k) {
    // Ensure eval mode for inference
    bool was_training = training_;
    eval();
    
    int64_t B = prompt.shape()[0];
    int64_t prompt_len = prompt.shape()[1];
    Device device = prompt.device();
    
    // Initialize caches
    init_cache(B, device);
    
    // Collect all generated tokens (starting with prompt)
    std::vector<Tensor> all_tokens;
    all_tokens.push_back(prompt);
    
    // Process prompt (prefill)
    Tensor logits = forward_with_cache(prompt, 0);
    
    // Random engine for sampling
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Generate tokens one by one
    for (int64_t i = 0; i < max_new_tokens; ++i) {
        // Get logits for last position: [B, 1, V] -> [B, V]
        Tensor last_logits = logits.index({Slice(), Slice(-1, std::nullopt), Slice()});
        last_logits = last_logits.view({B, config_.vocab_size});
        
        // Apply temperature (with safety check for near-zero values)
        if (temperature != 1.0f) {
            // Clamp temperature to avoid division by zero or numerical overflow
            float safe_temp = std::max(temperature, 1e-6f);
            last_logits = ops::div(last_logits, safe_temp);
        }
        
        // Convert to probabilities
        Tensor probs = nn::functional::softmax(last_logits, -1);
        
        // Sample next token
        Tensor next_tokens = vesper::empty({B, 1}, DType::Int32, Device::CPU, false);
        auto* next_ptr = next_tokens.data_ptr<int32_t>();
        
        // Get probs on CPU for sampling
        Tensor probs_cpu = probs.to(Device::CPU);
        const float* probs_data = probs_cpu.data_ptr<float>();
        
        for (int64_t b = 0; b < B; ++b) {
            // Simple multinomial sampling
            std::vector<float> p(probs_data + b * config_.vocab_size, 
                                 probs_data + (b + 1) * config_.vocab_size);
            
            // Apply top-k filtering if requested
            if (top_k > 0 && top_k < config_.vocab_size) {
                std::vector<std::pair<float, int>> indexed_probs;
                for (size_t j = 0; j < p.size(); ++j) {
                    indexed_probs.emplace_back(p[j], static_cast<int>(j));
                }
                std::partial_sort(indexed_probs.begin(), 
                                  indexed_probs.begin() + top_k,
                                  indexed_probs.end(),
                                  std::greater<>());
                
                // Zero out non-top-k
                std::fill(p.begin(), p.end(), 0.0f);
                float sum = 0.0f;
                for (int64_t k = 0; k < top_k; ++k) {
                    p[indexed_probs[k].second] = indexed_probs[k].first;
                    sum += indexed_probs[k].first;
                }
                // Renormalize
                for (auto& prob : p) prob /= sum;
            }
            
            std::discrete_distribution<> dist(p.begin(), p.end());
            next_ptr[b] = dist(gen);
        }
        
        // Move to device if needed
        if (device != Device::CPU) {
            next_tokens = next_tokens.to(device);
        }
        
        all_tokens.push_back(next_tokens);
        
        // Forward with new token
        logits = forward_with_cache(next_tokens, prompt_len + i);
    }
    
    clear_cache();
    
    // Restore training mode if it was set
    if (was_training) train();
    
    // Concatenate all tokens
    // Note: This is a simplified version - in practice you'd use a proper concat op
    int64_t total_len = prompt_len + max_new_tokens;
    Tensor result = vesper::empty({B, total_len}, DType::Int32, Device::CPU, false);
    auto* result_ptr = result.data_ptr<int32_t>();
    
    int64_t offset = 0;
    for (const auto& t : all_tokens) {
        Tensor t_cpu = t.to(Device::CPU);
        const auto* src = t_cpu.data_ptr<int32_t>();
        int64_t len = t.shape()[1];
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t j = 0; j < len; ++j) {
                result_ptr[b * total_len + offset + j] = src[b * len + j];
            }
        }
        offset += len;
    }
    
    if (device != Device::CPU) {
        result = result.to(device);
    }
    
    return result;
}

Tensor TransformerLM::compute_loss(const Tensor& tokens, const Tensor& targets) {
    // Forward pass to get logits
    Tensor logits = forward(tokens);  // [B, S, V]
    
    // Reshape for cross entropy: [B*S, V] and [B*S]
    int64_t B = logits.shape()[0];
    int64_t S = logits.shape()[1];
    int64_t V = logits.shape()[2];
    
    Tensor logits_flat = logits.view({B * S, V});
    Tensor targets_flat = targets.view({B * S});
    
    // Use the cross_entropy_loss function from functional
    // This handles autograd and stays on GPU
    return nn::functional::cross_entropy_loss(logits_flat, targets_flat);
}

int64_t TransformerLM::num_parameters() const {
    int64_t count = 0;
    // Count parameters from all components directly
    count += tok_emb_.weight.numel();
    
    if (pos_emb_) {
        count += pos_emb_->weight.numel();
    }
    
    for (const auto& block : blocks_) {
        for (const auto& p : block->parameters()) {
            count += p.numel();
        }
    }
    
    for (const auto& p : final_norm_->parameters()) {
        count += p.numel();
    }
    
    if (lm_head_) {
        count += lm_head_->weight.numel();
        // Linear without bias - no bias to count
    }
    
    return count;
}

std::string TransformerLM::describe() const {
    std::ostringstream ss;
    ss << "TransformerLM {\n";
    ss << "  config: " << config_.describe() << "\n";
    ss << "  parameters: " << num_parameters() << "\n";
    ss << "  model_size: " << (model_size_bytes() / 1024 / 1024) << " MB (FP32)\n";
    ss << "}";
    return ss.str();
}

Tensor& TransformerLM::output_weight() {
    if (lm_head_) {
        return lm_head_->weight;
    } else {
        return tok_emb_.weight;
    }
}

// =============================================================================
// Factory Functions
// =============================================================================

std::unique_ptr<TransformerLM> create_model(const TransformerConfig& config) {
    return std::make_unique<TransformerLM>(config);
}

std::unique_ptr<TransformerLM> create_model(const std::string& name) {
    return std::make_unique<TransformerLM>(TransformerConfig::from_name(name));
}

std::unique_ptr<TransformerLM> create_gpt2(const std::string& size) {
    if (size == "small" || size == "124m") {
        return create_model(TransformerConfig::gpt2_small());
    } else if (size == "medium" || size == "355m") {
        return create_model(TransformerConfig::gpt2_medium());
    } else if (size == "large" || size == "774m") {
        return create_model(TransformerConfig::gpt2_large());
    } else if (size == "xl" || size == "1.5b") {
        return create_model(TransformerConfig::gpt2_xl());
    }
    throw std::runtime_error("Unknown GPT-2 size: " + size);
}

std::unique_ptr<TransformerLM> create_llama2(const std::string& size) {
    if (size == "7b") {
        return create_model(TransformerConfig::llama2_7b());
    } else if (size == "13b") {
        return create_model(TransformerConfig::llama2_13b());
    } else if (size == "70b") {
        return create_model(TransformerConfig::llama2_70b());
    }
    throw std::runtime_error("Unknown Llama 2 size: " + size);
}

std::unique_ptr<TransformerLM> create_llama3(const std::string& size) {
    if (size == "8b") {
        return create_model(TransformerConfig::llama3_8b());
    } else if (size == "70b") {
        return create_model(TransformerConfig::llama3_70b());
    }
    throw std::runtime_error("Unknown Llama 3 size: " + size);
}

// =============================================================================
// Weight Initialization
// =============================================================================

void initialize_weights(TransformerLM& model) {
    const auto& config = model.config();
    
    for (auto& [name, param] : model.named_parameters()) {
        // Embedding weights: Normal(0, 0.02)
        if (name.find("embeddings") != std::string::npos ||
            name.find("tok_emb") != std::string::npos) {
            nn::init::normal_(param, 0.0f, 0.02f);
        }
        // Residual projections: scaled initialization
        else if (name.find("c_proj") != std::string::npos ||
                 name.find("wo") != std::string::npos ||
                 name.find("down_proj") != std::string::npos ||
                 name.find("output") != std::string::npos) {
            float std_dev = 0.02f / std::sqrt(2.0f * config.n_layers);
            nn::init::normal_(param, 0.0f, std_dev);
        }
        // Bias terms: zeros
        else if (name.find("bias") != std::string::npos) {
            nn::init::zeros_(param);
        }
        // Other weights: Normal(0, 0.02)
        else if (name.find("weight") != std::string::npos) {
            nn::init::normal_(param, 0.0f, 0.02f);
        }
    }
}

} // namespace vesper::models
