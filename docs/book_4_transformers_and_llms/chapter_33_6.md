```markdown
# Chapter 33.6: Building a Complete GPT/Llama Model

## 1. Introduction

This chapter brings together all the components from previous chapters to build a complete, production-ready Transformer language model. We'll implement both the **GPT-2** architecture (for educational purposes) and the **Llama** architecture (for modern LLM compatibility).

### Architecture Comparison

| Component | GPT-2 | Llama |
|-----------|-------|-------|
| Attention | MHA | GQA |
| Position Encoding | Learned Absolute | RoPE |
| Normalization | LayerNorm | RMSNorm |
| FFN | GELU MLP | SwiGLU |
| Norm Position | Post-Norm | Pre-Norm |
| Bias | Yes | No |

## 2. Model Configuration

### 2.1 Configuration Structure

```cpp
// include/vesper/models/config.h

namespace vesper::models {

struct TransformerConfig {
    // Model architecture
    int64_t vocab_size;
    int64_t dim;              // Hidden dimension (embed_dim)
    int64_t n_layers;         // Number of transformer blocks
    int64_t n_heads;          // Number of attention heads
    int64_t n_kv_heads;       // KV heads for GQA (-1 = same as n_heads)
    int64_t max_seq_len;      // Maximum sequence length
    
    // FFN
    int64_t ffn_dim_multiplier;  // Hidden dim = dim * multiplier (or explicit)
    int64_t ffn_hidden_dim;      // Override: explicit FFN hidden dim
    
    // Normalization
    float norm_eps;
    bool use_rms_norm;        // true = RMSNorm, false = LayerNorm
    
    // RoPE
    float rope_base;
    float rope_scaling;       // For extended context
    
    // Regularization
    float dropout;
    float attention_dropout;
    
    // Other
    bool tie_word_embeddings;
    bool use_bias;
    
    // Factory methods for common architectures
    static TransformerConfig gpt2_small();
    static TransformerConfig gpt2_medium();
    static TransformerConfig gpt2_large();
    static TransformerConfig gpt2_xl();
    
    static TransformerConfig llama2_7b();
    static TransformerConfig llama2_13b();
    static TransformerConfig llama2_70b();
    
    static TransformerConfig llama3_8b();
    static TransformerConfig llama3_70b();
    
    static TransformerConfig mistral_7b();
    
    // Derived values
    int64_t head_dim() const { return dim / n_heads; }
    int64_t kv_heads() const { return n_kv_heads > 0 ? n_kv_heads : n_heads; }
    int64_t hidden_dim() const;
};

} // namespace vesper::models
```

### 2.2 Configuration Implementations

```cpp
// src/models/config.cpp

namespace vesper::models {

int64_t TransformerConfig::hidden_dim() const {
    if (ffn_hidden_dim > 0) return ffn_hidden_dim;
    
    // Llama uses (8/3) * dim, rounded to multiple of 256
    int64_t h = (dim * 8) / 3;
    h = ((h + 255) / 256) * 256;  // Round up to 256
    return h;
}

TransformerConfig TransformerConfig::gpt2_small() {
    return {
        .vocab_size = 50257,
        .dim = 768,
        .n_layers = 12,
        .n_heads = 12,
        .n_kv_heads = 12,  // MHA
        .max_seq_len = 1024,
        .ffn_dim_multiplier = 4,
        .ffn_hidden_dim = 3072,
        .norm_eps = 1e-5f,
        .use_rms_norm = false,
        .rope_base = 0,  // Not used (learned pos)
        .rope_scaling = 1.0f,
        .dropout = 0.1f,
        .attention_dropout = 0.1f,
        .tie_word_embeddings = true,
        .use_bias = true
    };
}

TransformerConfig TransformerConfig::llama2_7b() {
    return {
        .vocab_size = 32000,
        .dim = 4096,
        .n_layers = 32,
        .n_heads = 32,
        .n_kv_heads = 32,  // MHA for 7B
        .max_seq_len = 4096,
        .ffn_dim_multiplier = 0,
        .ffn_hidden_dim = 11008,
        .norm_eps = 1e-5f,
        .use_rms_norm = true,
        .rope_base = 10000.0f,
        .rope_scaling = 1.0f,
        .dropout = 0.0f,
        .attention_dropout = 0.0f,
        .tie_word_embeddings = false,
        .use_bias = false
    };
}

TransformerConfig TransformerConfig::llama2_70b() {
    return {
        .vocab_size = 32000,
        .dim = 8192,
        .n_layers = 80,
        .n_heads = 64,
        .n_kv_heads = 8,  // GQA!
        .max_seq_len = 4096,
        .ffn_dim_multiplier = 0,
        .ffn_hidden_dim = 28672,
        .norm_eps = 1e-5f,
        .use_rms_norm = true,
        .rope_base = 10000.0f,
        .rope_scaling = 1.0f,
        .dropout = 0.0f,
        .attention_dropout = 0.0f,
        .tie_word_embeddings = false,
        .use_bias = false
    };
}

TransformerConfig TransformerConfig::llama3_8b() {
    return {
        .vocab_size = 128256,
        .dim = 4096,
        .n_layers = 32,
        .n_heads = 32,
        .n_kv_heads = 8,  // GQA
        .max_seq_len = 8192,
        .ffn_dim_multiplier = 0,
        .ffn_hidden_dim = 14336,
        .norm_eps = 1e-5f,
        .use_rms_norm = true,
        .rope_base = 500000.0f,  // Higher base for Llama 3
        .rope_scaling = 1.0f,
        .dropout = 0.0f,
        .attention_dropout = 0.0f,
        .tie_word_embeddings = false,
        .use_bias = false
    };
}

TransformerConfig TransformerConfig::mistral_7b() {
    return {
        .vocab_size = 32000,
        .dim = 4096,
        .n_layers = 32,
        .n_heads = 32,
        .n_kv_heads = 8,  // GQA
        .max_seq_len = 8192,  // Sliding window not implemented here
        .ffn_dim_multiplier = 0,
        .ffn_hidden_dim = 14336,
        .norm_eps = 1e-5f,
        .use_rms_norm = true,
        .rope_base = 10000.0f,
        .rope_scaling = 1.0f,
        .dropout = 0.0f,
        .attention_dropout = 0.0f,
        .tie_word_embeddings = false,
        .use_bias = false
    };
}

} // namespace vesper::models
```

## 3. The Transformer Block

### 3.1 Llama-Style Block

```cpp
// include/vesper/models/transformer_block.h

namespace vesper::models {

class TransformerBlock : public nn::Module {
public:
    TransformerBlock(const TransformerConfig& config, int layer_idx = 0);
    
    Tensor forward(const Tensor& x, 
                   KVCache* cache = nullptr,
                   int64_t start_pos = 0);
    
private:
    TransformerConfig config_;
    int layer_idx_;
    
    // Pre-Norm layers
    std::unique_ptr<nn::Module> attn_norm_;  // RMSNorm or LayerNorm
    std::unique_ptr<nn::Module> ffn_norm_;
    
    // Attention (MHA or GQA)
    std::unique_ptr<nn::Module> attn_;
    
    // FFN (SwiGLU or standard MLP)
    std::unique_ptr<nn::Module> ffn_;
    
    float dropout_;
};

} // namespace vesper::models
```

### 3.2 Block Implementation

```cpp
// src/models/transformer_block.cpp

namespace vesper::models {

TransformerBlock::TransformerBlock(const TransformerConfig& config, int layer_idx)
    : config_(config), layer_idx_(layer_idx), dropout_(config.dropout)
{
    // Normalization
    if (config.use_rms_norm) {
        attn_norm_ = register_module("attention_norm", 
            std::make_unique<nn::RMSNorm>(config.dim, config.norm_eps));
        ffn_norm_ = register_module("ffn_norm",
            std::make_unique<nn::RMSNorm>(config.dim, config.norm_eps));
    } else {
        attn_norm_ = register_module("ln_1",
            std::make_unique<nn::LayerNorm>(
                std::vector<int64_t>{config.dim}, config.norm_eps));
        ffn_norm_ = register_module("ln_2",
            std::make_unique<nn::LayerNorm>(
                std::vector<int64_t>{config.dim}, config.norm_eps));
    }
    
    // Attention
    if (config.kv_heads() < config.n_heads) {
        // GQA
        attn_ = register_module("attention",
            std::make_unique<nn::GroupedQueryAttention>(
                config.dim, config.n_heads, config.kv_heads(),
                config.max_seq_len, config.rope_base, config.attention_dropout));
    } else {
        // MHA
        attn_ = register_module("attention",
            std::make_unique<nn::MultiHeadAttention>(
                config.dim, config.n_heads, config.max_seq_len,
                config.rope_base, config.attention_dropout, config.use_bias));
    }
    
    // FFN
    if (config.use_rms_norm) {  // Llama-style = SwiGLU
        ffn_ = register_module("feed_forward",
            std::make_unique<nn::SwiGLUMLP>(
                config.dim, config.hidden_dim(), config.use_bias));
    } else {  // GPT-style = standard MLP
        ffn_ = register_module("mlp",
            std::make_unique<nn::MLP>(
                config.dim, config.ffn_hidden_dim, config.use_bias));
    }
}

Tensor TransformerBlock::forward(const Tensor& x, KVCache* cache, int64_t start_pos) {
    // Pre-Norm architecture (Llama style)
    // x = x + attn(norm(x))
    // x = x + ffn(norm(x))
    
    // Self-attention
    Tensor h = x + static_cast<nn::GroupedQueryAttention*>(attn_.get())->forward(
        attn_norm_->forward(x), cache, start_pos);
    
    if (dropout_ > 0 && is_training()) {
        h = nn::functional::dropout(h, dropout_);
    }
    
    // FFN
    Tensor out = h + ffn_->forward(ffn_norm_->forward(h));
    
    if (dropout_ > 0 && is_training()) {
        out = nn::functional::dropout(out, dropout_);
    }
    
    return out;
}

} // namespace vesper::models
```

## 4. The Complete Transformer Model

### 4.1 Model Definition

```cpp
// include/vesper/models/transformer.h

namespace vesper::models {

class Transformer : public nn::Module {
public:
    explicit Transformer(const TransformerConfig& config);
    
    // Forward for training (full sequence)
    // Input: token_ids [Batch, SeqLen]
    // Output: logits [Batch, SeqLen, VocabSize]
    Tensor forward(const Tensor& tokens);
    
    // Forward for inference with KV cache
    // tokens: [Batch, NewTokens] - just the new tokens
    // start_pos: position in the sequence
    Tensor forward_with_cache(const Tensor& tokens, int64_t start_pos);
    
    // Initialize KV cache for inference
    void init_cache(int64_t batch_size, Device device);
    void clear_cache();
    
    // Accessors
    const TransformerConfig& config() const { return config_; }
    int64_t num_parameters() const;
    
private:
    TransformerConfig config_;
    
    // Token embeddings
    nn::Embedding tok_emb_;
    
    // Position embeddings (GPT-2 only, nullptr for Llama)
    std::unique_ptr<nn::Embedding> pos_emb_;
    
    // Transformer blocks
    std::vector<std::unique_ptr<TransformerBlock>> blocks_;
    
    // Final normalization
    std::unique_ptr<nn::Module> norm_;
    
    // Output projection (lm_head)
    // If tie_word_embeddings: share weights with tok_emb
    std::unique_ptr<nn::Linear> lm_head_;
    
    // KV caches (one per layer)
    std::vector<std::unique_ptr<KVCache>> kv_caches_;
    
    float dropout_;
};

} // namespace vesper::models
```

### 4.2 Model Implementation

```cpp
// src/models/transformer.cpp

namespace vesper::models {

Transformer::Transformer(const TransformerConfig& config)
    : config_(config),
      tok_emb_(config.vocab_size, config.dim),
      dropout_(config.dropout)
{
    // Token embeddings
    register_module("tok_embeddings", tok_emb_);
    
    // Position embeddings (GPT-2 style only)
    if (config.rope_base == 0) {
        pos_emb_ = std::make_unique<nn::Embedding>(config.max_seq_len, config.dim);
        register_module("pos_embeddings", *pos_emb_);
    }
    
    // Transformer blocks
    blocks_.reserve(config.n_layers);
    for (int64_t i = 0; i < config.n_layers; ++i) {
        auto block = std::make_unique<TransformerBlock>(config, i);
        register_module("layers." + std::to_string(i), *block);
        blocks_.push_back(std::move(block));
    }
    
    // Final normalization
    if (config.use_rms_norm) {
        norm_ = std::make_unique<nn::RMSNorm>(config.dim, config.norm_eps);
    } else {
        norm_ = std::make_unique<nn::LayerNorm>(
            std::vector<int64_t>{config.dim}, config.norm_eps);
    }
    register_module("norm", *norm_);
    
    // Output projection
    if (!config.tie_word_embeddings) {
        lm_head_ = std::make_unique<nn::Linear>(config.dim, config.vocab_size, false);
        register_module("output", *lm_head_);
    }
}

Tensor Transformer::forward(const Tensor& tokens) {
    // tokens: [Batch, SeqLen]
    auto [B, S] = tokens.sizes2d();
    
    VESPER_CHECK(S <= config_.max_seq_len, 
        "Sequence length " + std::to_string(S) + 
        " exceeds max " + std::to_string(config_.max_seq_len));
    
    // 1. Token embeddings
    Tensor h = tok_emb_(tokens);  // [B, S, D]
    
    // 2. Position embeddings (GPT-2 only)
    if (pos_emb_) {
        Tensor positions = arange(S, DType::Int64, tokens.device());
        h = h + (*pos_emb_)(positions);
    }
    
    // 3. Dropout
    if (dropout_ > 0 && is_training()) {
        h = nn::functional::dropout(h, dropout_);
    }
    
    // 4. Transformer blocks
    for (auto& block : blocks_) {
        h = block->forward(h, nullptr, 0);
    }
    
    // 5. Final normalization
    h = norm_->forward(h);
    
    // 6. Output projection
    if (lm_head_) {
        return (*lm_head_)(h);  // [B, S, V]
    } else {
        // Tied embeddings: multiply by embedding weight transpose
        return ops::matmul(h, tok_emb_.weight.transpose(-2, -1));
    }
}

Tensor Transformer::forward_with_cache(const Tensor& tokens, int64_t start_pos) {
    // tokens: [Batch, NewTokens] - typically 1 during generation
    auto [B, S] = tokens.sizes2d();
    
    VESPER_CHECK(!kv_caches_.empty(), "KV cache not initialized. Call init_cache first.");
    
    // 1. Token embeddings
    Tensor h = tok_emb_(tokens);  // [B, S, D]
    
    // 2. Position embeddings (GPT-2 only)
    if (pos_emb_) {
        Tensor positions = arange(start_pos, start_pos + S, DType::Int64, tokens.device());
        h = h + (*pos_emb_)(positions);
    }
    
    // 3. Transformer blocks with caching
    for (size_t i = 0; i < blocks_.size(); ++i) {
        h = blocks_[i]->forward(h, kv_caches_[i].get(), start_pos);
    }
    
    // 4. Final normalization
    h = norm_->forward(h);
    
    // 5. Output projection (only last position for generation)
    if (S > 1) {
        h = h.select(1, S - 1).unsqueeze(1);  // [B, 1, D]
    }
    
    if (lm_head_) {
        return (*lm_head_)(h);
    } else {
        return ops::matmul(h, tok_emb_.weight.transpose(-2, -1));
    }
}

void Transformer::init_cache(int64_t batch_size, Device device) {
    kv_caches_.clear();
    kv_caches_.reserve(config_.n_layers);
    
    for (int64_t i = 0; i < config_.n_layers; ++i) {
        kv_caches_.push_back(std::make_unique<KVCache>(
            batch_size, config_.kv_heads(), config_.max_seq_len,
            config_.head_dim(), device));
    }
}

void Transformer::clear_cache() {
    kv_caches_.clear();
}

int64_t Transformer::num_parameters() const {
    int64_t count = 0;
    for (const auto& p : parameters()) {
        count += p.numel();
    }
    return count;
}

} // namespace vesper::models
```

## 5. Model Construction Helpers

### 5.1 Factory Functions

```cpp
// include/vesper/models/factory.h

namespace vesper::models {

// Create model from config
std::unique_ptr<Transformer> create_model(const TransformerConfig& config);

// Create model by name
std::unique_ptr<Transformer> create_model(const std::string& name);

// Named configs
std::unique_ptr<Transformer> create_gpt2(const std::string& size = "small");
std::unique_ptr<Transformer> create_llama2(const std::string& size = "7b");
std::unique_ptr<Transformer> create_llama3(const std::string& size = "8b");

} // namespace vesper::models
```

### 5.2 Implementation

```cpp
// src/models/factory.cpp

namespace vesper::models {

std::unique_ptr<Transformer> create_model(const std::string& name) {
    std::string lower = to_lowercase(name);
    
    if (lower == "gpt2" || lower == "gpt2-small") {
        return std::make_unique<Transformer>(TransformerConfig::gpt2_small());
    } else if (lower == "gpt2-medium") {
        return std::make_unique<Transformer>(TransformerConfig::gpt2_medium());
    } else if (lower == "gpt2-large") {
        return std::make_unique<Transformer>(TransformerConfig::gpt2_large());
    } else if (lower == "gpt2-xl") {
        return std::make_unique<Transformer>(TransformerConfig::gpt2_xl());
    } else if (lower == "llama2-7b" || lower == "llama-2-7b") {
        return std::make_unique<Transformer>(TransformerConfig::llama2_7b());
    } else if (lower == "llama2-13b") {
        return std::make_unique<Transformer>(TransformerConfig::llama2_13b());
    } else if (lower == "llama2-70b") {
        return std::make_unique<Transformer>(TransformerConfig::llama2_70b());
    } else if (lower == "llama3-8b") {
        return std::make_unique<Transformer>(TransformerConfig::llama3_8b());
    } else if (lower == "llama3-70b") {
        return std::make_unique<Transformer>(TransformerConfig::llama3_70b());
    } else if (lower == "mistral-7b") {
        return std::make_unique<Transformer>(TransformerConfig::mistral_7b());
    }
    
    throw std::runtime_error("Unknown model: " + name);
}

} // namespace vesper::models
```

## 6. Weight Initialization

### 6.1 Standard Initialization

```cpp
// src/models/init.cpp

void initialize_weights(Transformer& model, const TransformerConfig& config) {
    for (auto& [name, param] : model.named_parameters()) {
        if (name.find("weight") != std::string::npos) {
            if (name.find("tok_embeddings") != std::string::npos ||
                name.find("output") != std::string::npos) {
                // Embedding weights: normal with small std
                nn::init::normal_(param, 0.0f, 0.02f);
            } else if (name.find("c_proj") != std::string::npos ||
                       name.find("wo") != std::string::npos ||
                       name.find("down_proj") != std::string::npos) {
                // Residual projections: scaled initialization
                float std = 0.02f / std::sqrt(2.0f * config.n_layers);
                nn::init::normal_(param, 0.0f, std);
            } else {
                // Other weights: standard normal
                nn::init::normal_(param, 0.0f, 0.02f);
            }
        } else if (name.find("bias") != std::string::npos) {
            nn::init::zeros_(param);
        }
    }
}
```

## 7. Usage Examples

### 7.1 Creating and Using a Model

```cpp
#include <vesper/models/transformer.h>

int main() {
    // Create Llama 2 7B model
    auto model = vesper::models::create_model("llama2-7b");
    model->to(Device::HIP);
    
    std::cout << "Parameters: " << model->num_parameters() << std::endl;
    // Output: Parameters: 6738415616 (~6.7B)
    
    // Training forward pass
    Tensor tokens = randint(0, 32000, {4, 512}, DType::Int64, Device::HIP);
    Tensor logits = model->forward(tokens);
    std::cout << "Logits shape: " << logits.shape() << std::endl;
    // Output: [4, 512, 32000]
    
    // Inference with KV cache
    model->eval();
    model->init_cache(1, Device::HIP);
    
    Tensor prompt = tensor({{1, 15043, 29889}}, DType::Int64, Device::HIP);  // "Hello."
    Tensor out = model->forward_with_cache(prompt, 0);
    // Next token prediction...
    
    return 0;
}
```

### 7.2 Custom Configuration

```cpp
TransformerConfig my_config;
my_config.vocab_size = 50000;
my_config.dim = 1024;
my_config.n_layers = 16;
my_config.n_heads = 16;
my_config.n_kv_heads = 4;  // GQA with 4:1 ratio
my_config.max_seq_len = 2048;
my_config.ffn_hidden_dim = 2730;  // ~2.67x
my_config.use_rms_norm = true;
my_config.rope_base = 10000.0f;
my_config.use_bias = false;

auto model = std::make_unique<Transformer>(my_config);
```

## 8. Testing Strategy

### 8.1 Unit Tests

```cpp
// tests/models/test_transformer.cpp

TEST(Transformer, GPT2OutputShape) {
    auto model = create_model("gpt2-small");
    
    Tensor tokens = randint(0, 50257, {2, 128}, DType::Int64);
    Tensor logits = model->forward(tokens);
    
    EXPECT_EQ(logits.shape(), std::vector<int64_t>({2, 128, 50257}));
}

TEST(Transformer, Llama2OutputShape) {
    auto config = TransformerConfig::llama2_7b();
    config.n_layers = 2;  // Small for testing
    
    auto model = std::make_unique<Transformer>(config);
    
    Tensor tokens = randint(0, 32000, {1, 64}, DType::Int64);
    Tensor logits = model->forward(tokens);
    
    EXPECT_EQ(logits.shape(), std::vector<int64_t>({1, 64, 32000}));
}

TEST(Transformer, ParameterCount) {
    // GPT-2 Small: ~124M parameters
    auto gpt2 = create_model("gpt2-small");
    int64_t gpt2_params = gpt2->num_parameters();
    EXPECT_NEAR(gpt2_params, 124e6, 1e6);
    
    // Llama 2 7B: ~6.7B parameters
    // (Can only test with reduced layers)
    auto config = TransformerConfig::llama2_7b();
    config.n_layers = 1;
    auto llama = std::make_unique<Transformer>(config);
    
    // Per-layer params + embeddings
    // Embeddings: vocab * dim = 32000 * 4096 = 131M
    // Per layer: ~200M (rough estimate)
    int64_t expected_one_layer = 131e6 + 200e6;
    EXPECT_NEAR(llama->num_parameters(), expected_one_layer, 50e6);
}

TEST(Transformer, GradientFlow) {
    auto config = TransformerConfig::gpt2_small();
    config.n_layers = 2;
    
    auto model = std::make_unique<Transformer>(config);
    
    Tensor tokens = randint(0, 50257, {1, 32}, DType::Int64);
    Tensor logits = model->forward(tokens);
    Tensor loss = logits.mean();
    
    loss.backward();
    
    for (const auto& [name, param] : model->named_parameters()) {
        EXPECT_TRUE(param.grad().defined()) << "No grad for: " << name;
        EXPECT_FALSE(param.grad().isnan().any().item<bool>()) << "NaN grad: " << name;
    }
}

TEST(Transformer, KVCacheEquivalence) {
    auto config = TransformerConfig::llama2_7b();
    config.n_layers = 2;
    config.max_seq_len = 64;
    
    auto model = std::make_unique<Transformer>(config);
    model->eval();
    
    Tensor tokens = randint(0, 32000, {1, 16}, DType::Int64);
    
    // Method 1: Full forward
    Tensor logits_full = model->forward(tokens);
    
    // Method 2: Incremental with cache
    model->init_cache(1, Device::CPU);
    
    Tensor logits_cached;
    for (int64_t i = 0; i < 16; ++i) {
        Tensor single = tokens.select(1, i).unsqueeze(1);  // [1, 1]
        logits_cached = model->forward_with_cache(single, i);
    }
    
    // Last logits should match
    Tensor last_full = logits_full.select(1, 15);
    Tensor last_cached = logits_cached.select(1, 0);
    
    EXPECT_TRUE(allclose(last_full, last_cached, 1e-4, 1e-4));
}

TEST(Transformer, CausalMasking) {
    auto config = TransformerConfig::gpt2_small();
    config.n_layers = 1;
    
    auto model = std::make_unique<Transformer>(config);
    model->eval();
    
    Tensor tokens = randint(0, 50257, {1, 8}, DType::Int64);
    Tensor logits1 = model->forward(tokens);
    
    // Change last token
    Tensor tokens_modified = tokens.clone();
    tokens_modified[{0, 7}] = 100;
    Tensor logits2 = model->forward(tokens_modified);
    
    // First 7 positions should be identical
    for (int i = 0; i < 7; ++i) {
        EXPECT_TRUE(allclose(logits1.select(1, i), logits2.select(1, i)));
    }
}
```

### 8.2 Stress Tests

```cpp
TEST(Transformer, StressTest_LargeBatch) {
    auto config = TransformerConfig::llama2_7b();
    config.n_layers = 4;
    
    auto model = std::make_unique<Transformer>(config);
    model->to(Device::HIP);
    
    // Large batch
    Tensor tokens = randint(0, 32000, {32, 1024}, DType::Int64, Device::HIP);
    
    auto start = std::chrono::high_resolution_clock::now();
    Tensor logits = model->forward(tokens);
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Forward [32, 1024] 4-layer Llama: " << ms << " ms" << std::endl;
    
    EXPECT_FALSE(logits.isnan().any().item<bool>());
}

TEST(Transformer, StressTest_LongContext) {
    auto config = TransformerConfig::llama3_8b();
    config.n_layers = 2;
    
    auto model = std::make_unique<Transformer>(config);
    model->to(Device::HIP);
    model->eval();
    model->init_cache(1, Device::HIP);
    
    // Process 8K context incrementally
    for (int64_t pos = 0; pos < 8000; pos += 100) {
        Tensor chunk = randint(0, 128256, {1, 100}, DType::Int64, Device::HIP);
        Tensor out = model->forward_with_cache(chunk, pos);
        
        EXPECT_FALSE(out.isnan().any().item<bool>());
    }
}

TEST(Transformer, StressTest_MemoryUsage) {
    size_t initial_mem = get_hip_memory_usage();
    
    {
        auto model = create_model("gpt2-small");
        model->to(Device::HIP);
        
        Tensor tokens = randint(0, 50257, {4, 256}, DType::Int64, Device::HIP);
        
        for (int i = 0; i < 10; ++i) {
            Tensor logits = model->forward(tokens);
            Tensor loss = logits.mean();
            loss.backward();
            model->zero_grad();
        }
    }
    
    // Force cleanup
    hipDeviceSynchronize();
    
    size_t final_mem = get_hip_memory_usage();
    
    // Memory should return to near-initial level
    EXPECT_LT(final_mem, initial_mem * 1.1);
}
```

### 8.3 Numerical Tests

```cpp
TEST(Transformer, NumericalStability) {
    auto config = TransformerConfig::gpt2_small();
    config.n_layers = 2;
    
    auto model = std::make_unique<Transformer>(config);
    
    // Test with extreme input IDs
    Tensor tokens_min = zeros({1, 8}, DType::Int64);  // All padding
    Tensor logits_min = model->forward(tokens_min);
    EXPECT_FALSE(logits_min.isnan().any().item<bool>());
    
    Tensor tokens_max = full({1, 8}, 50256, DType::Int64);  // All max
    Tensor logits_max = model->forward(tokens_max);
    EXPECT_FALSE(logits_max.isnan().any().item<bool>());
}
```

## 9. Performance Considerations

### 9.1 Memory Footprint

| Model | Parameters | FP32 Memory | FP16 Memory |
|-------|------------|-------------|-------------|
| GPT-2 Small | 124M | 496 MB | 248 MB |
| GPT-2 XL | 1.5B | 6 GB | 3 GB |
| Llama 2 7B | 6.7B | 27 GB | 13.5 GB |
| Llama 2 70B | 70B | 280 GB | 140 GB |

### 9.2 Inference Optimization Checklist

- [ ] Use FP16/BF16 (Chapter 39)
- [ ] Enable KV cache for generation
- [ ] Use GQA for large models
- [ ] Implement fused kernels (Chapter 38)
- [ ] Consider PagedAttention for serving (Chapter 43)

## 10. Summary

This chapter provided the complete blueprint for building GPT and Llama-style transformer models in Vesper:

1. **Configuration**: Flexible `TransformerConfig` supporting multiple architectures
2. **Components**: Modular blocks using attention, FFN, and normalization from earlier chapters
3. **Caching**: KV cache integration for efficient inference
4. **Factory**: Easy model creation by name

With this foundation, Vesper can now run modern LLM architectures. The next chapters will cover text generation (sampling) and loading pre-trained weights.

```
