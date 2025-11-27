# Building a Llama-2 Style LLM with Vesper and SentencePiece BPE

This document provides a comprehensive plan for building a Llama-2 style language model using the Vesper deep learning library, integrated with the SentencePiece BPE tokenizer.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Summary](#2-architecture-summary)
3. [Phase 1: Project Setup](#3-phase-1-project-setup)
4. [Phase 2: SentencePiece Integration](#4-phase-2-sentencepiece-integration)
5. [Phase 3: Model Architecture](#5-phase-3-model-architecture)
6. [Phase 4: Data Pipeline](#6-phase-4-data-pipeline)
7. [Phase 5: Training Loop](#7-phase-5-training-loop)
8. [Phase 6: Inference and Generation](#8-phase-6-inference-and-generation)
9. [Phase 7: Checkpointing and Serialization](#9-phase-7-checkpointing-and-serialization)
10. [Implementation Checklist](#10-implementation-checklist)
11. [Appendix: Vesper API Reference](#11-appendix-vesper-api-reference)

---

## 1. Project Overview

### Goals

Build a trainable and inferrable Llama-2 style language model that:
- Uses Vesper's native transformer components (`TransformerLM`, `GroupedQueryAttention`, `SwiGLUMLP`, `RoPE`)
- Integrates with SentencePiece for BPE tokenization
- Supports training on custom text data
- Supports efficient autoregressive generation with KV caching

### Llama-2 Architecture Highlights

| Component | Implementation |
|-----------|----------------|
| Position Encoding | RoPE (Rotary Position Embedding) |
| Attention | Multi-Head Attention (7B/13B) or Grouped Query Attention (70B) |
| Normalization | RMSNorm (pre-normalization) |
| Feed-Forward | SwiGLU MLP (3 linear layers with SiLU gating) |
| Bias | No bias in attention and MLP layers |
| Vocabulary | 32,000 tokens (SentencePiece BPE) |

### Vesper Components to Use

Vesper already provides the building blocks for Llama-2:

| Vesper Class/Function | Purpose |
|-----------------------|---------|
| `models::TransformerLM` | Complete transformer model |
| `models::TransformerConfig::llama2_7b()` | Pre-configured Llama-2 settings |
| `nn::GroupedQueryAttention` | GQA for large models |
| `nn::SwiGLUMLP` | SwiGLU feed-forward network |
| `nn::RMSNorm` | RMS normalization |
| `nn::RoPEFrequencies` | Rotary position embeddings |
| `nn::KVCache` / `nn::GQAKVCache` | KV caching for inference |
| `generation::Generator` | High-level generation API |
| `generation::SamplingParams` | Sampling configuration |

---

## 2. Architecture Summary

### Model Dimensions (Llama-2 7B Reference)

```
vocab_size     = 32,000
dim            = 4,096      (hidden dimension)
n_layers       = 32         (transformer blocks)
n_heads        = 32         (attention heads)
n_kv_heads     = 32         (same as n_heads for 7B; use 8 for 70B GQA)
head_dim       = 128        (dim / n_heads)
ffn_hidden_dim = 11,008     (≈ 8/3 × dim, rounded)
max_seq_len    = 4,096
rope_base      = 10,000
```

### Transformer Block Structure

```
Input x [Batch, SeqLen, Dim]
    │
    ├─→ RMSNorm ─→ Self-Attention (with RoPE) ─→ + (residual)
    │                                            │
    └────────────────────────────────────────────┘
    │
    ├─→ RMSNorm ─→ SwiGLU MLP ─→ + (residual)
    │                           │
    └───────────────────────────┘
    │
Output [Batch, SeqLen, Dim]
```

---

## 3. Phase 1: Project Setup

### 3.1 Directory Structure

```
llama2_project/
├── CMakeLists.txt
├── src/
│   ├── main.cpp              # Entry point
│   ├── tokenizer.h           # SentencePiece wrapper
│   ├── tokenizer.cpp
│   ├── data_loader.h         # Text data loading
│   ├── data_loader.cpp
│   ├── training.h            # Training loop
│   ├── training.cpp
│   └── inference.h           # Generation utilities
├── data/
│   └── tokenizer.model       # SentencePiece model file
├── checkpoints/              # Model checkpoints
└── build/
```

### 3.2 CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.15)
project(Llama2Demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Vesper
find_package(vesper 1.0 REQUIRED)

# Find SentencePiece
find_package(PkgConfig REQUIRED)
pkg_check_modules(SENTENCEPIECE REQUIRED sentencepiece)

add_executable(llama2_demo
    src/main.cpp
    src/tokenizer.cpp
    src/data_loader.cpp
    src/training.cpp
)

target_include_directories(llama2_demo PRIVATE 
    ${SENTENCEPIECE_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(llama2_demo
    vesper::vesper
    ${SENTENCEPIECE_LIBRARIES}
)
```

### 3.3 Dependencies

Install SentencePiece:
```bash
# Ubuntu/Debian
sudo apt install libsentencepiece-dev

# From source
git clone https://github.com/google/sentencepiece.git
cd sentencepiece
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc)
sudo make install
```

---

## 4. Phase 2: SentencePiece Integration

### 4.1 Tokenizer Wrapper Design

Create a C++ wrapper around SentencePiece that provides:

```cpp
class Tokenizer {
public:
    // Load a trained SentencePiece model
    bool load(const std::string& model_path);
    
    // Encode text to token IDs
    std::vector<int32_t> encode(const std::string& text);
    
    // Decode token IDs back to text
    std::string decode(const std::vector<int32_t>& ids);
    
    // Batch encode for training
    std::vector<std::vector<int32_t>> encode_batch(
        const std::vector<std::string>& texts);
    
    // Special tokens
    int32_t bos_id() const;  // Beginning of sequence
    int32_t eos_id() const;  // End of sequence
    int32_t pad_id() const;  // Padding token
    int32_t unk_id() const;  // Unknown token
    
    // Vocabulary info
    int32_t vocab_size() const;
};
```

### 4.2 Training a SentencePiece Model

If you need to train your own tokenizer:

```bash
# Using SentencePiece command-line tool
spm_train \
    --input=training_corpus.txt \
    --model_prefix=tokenizer \
    --vocab_size=32000 \
    --model_type=bpe \
    --character_coverage=0.9995 \
    --num_threads=8 \
    --split_digits=true \
    --byte_fallback=true \
    --pad_id=0 \
    --unk_id=1 \
    --bos_id=2 \
    --eos_id=3
```

### 4.3 Using Pre-trained Llama Tokenizer

For compatibility with official Llama-2 weights:
1. Download `tokenizer.model` from Meta's Llama repository
2. This file is directly loadable by SentencePiece

---

## 5. Phase 3: Model Architecture

### 5.1 Using Vesper's Pre-built TransformerLM

Vesper provides a complete `TransformerLM` class. Use the factory method:

```cpp
#include <vesper/vesper.h>
using namespace vesper;

// Option 1: Use predefined Llama-2 7B config
auto config = models::TransformerConfig::llama2_7b();
auto model = models::create_model(config);

// Option 2: Custom configuration
models::TransformerConfig config;
config.vocab_size = 32000;          // From your tokenizer
config.dim = 4096;
config.n_layers = 32;
config.n_heads = 32;
config.n_kv_heads = 32;             // Set < n_heads for GQA
config.max_seq_len = 4096;
config.rope_base = 10000.0f;
config.use_rms_norm = true;         // Llama uses RMSNorm
config.use_bias = false;            // Llama has no bias
config.tie_word_embeddings = false;
config.dropout = 0.0f;              // No dropout for inference

auto model = models::create_model(config);
```

### 5.2 Model Component Details

The `TransformerLM` internally uses:

| Component | Vesper Class | Notes |
|-----------|--------------|-------|
| Token Embedding | `nn::Embedding` | `[vocab_size, dim]` |
| Position Encoding | `nn::RoPEFrequencies` | Applied inside attention |
| Attention | `nn::GroupedQueryAttention` or `nn::MultiHeadAttention` | With RoPE, causal mask |
| Feed-Forward | `nn::SwiGLUMLP` | 3 projections: gate, up, down |
| Normalization | `nn::RMSNorm` | Pre-norm architecture |
| Output Head | `nn::Linear` | `[dim, vocab_size]` |

### 5.3 Custom Model (If Needed)

If you need more control, build from individual components:

```cpp
class MyLlama : public nn::Module {
public:
    nn::Embedding tok_emb{vocab_size, dim};
    std::vector<LlamaBlock> blocks;
    nn::RMSNorm final_norm{dim};
    nn::Linear lm_head{dim, vocab_size, /*bias=*/false};
    
    MyLlama(int vocab_size, int dim, int n_layers, int n_heads, Device device) {
        for (int i = 0; i < n_layers; ++i) {
            blocks.emplace_back(dim, n_heads, max_seq_len);
        }
        register_module("tok_emb", tok_emb);
        // ... register all modules
        to(device);
    }
    
    Tensor forward(const Tensor& tokens) override {
        auto x = tok_emb.forward(tokens);  // [B, T] -> [B, T, D]
        for (auto& block : blocks) {
            x = block.forward(x);
        }
        x = final_norm.forward(x);
        return lm_head.forward(x);  // [B, T, V]
    }
};
```

### 5.4 Moving Model to GPU

```cpp
#if defined(USE_HIP_BACKEND)
    Device device = Device::HIP;
#elif defined(USE_CUDA_BACKEND)
    Device device = Device::CUDA;
#else
    Device device = Device::CPU;
#endif

model->to(device);
```

---

## 6. Phase 4: Data Pipeline

### 6.1 Text Dataset Loader

Design a data loader that:
1. Reads text files (one document per line or separated by delimiters)
2. Tokenizes using SentencePiece
3. Creates fixed-length sequences with proper padding
4. Returns batches as Vesper tensors

```cpp
class TextDataset {
public:
    TextDataset(const std::string& path, Tokenizer& tokenizer, 
                int64_t seq_len, int64_t batch_size);
    
    struct Batch {
        Tensor input_ids;   // [batch_size, seq_len] Int32
        Tensor labels;      // [batch_size, seq_len] Int32 (shifted by 1)
        Tensor attention_mask; // [batch_size, seq_len] (optional)
    };
    
    Batch get_batch(Device device);
    void shuffle();
    bool epoch_complete() const;
    void reset();
    
private:
    std::vector<std::vector<int32_t>> tokenized_docs_;
    int64_t seq_len_;
    int64_t batch_size_;
    size_t current_idx_ = 0;
};
```

### 6.2 Creating Training Batches

For language modeling, create input/target pairs:

```cpp
// Given tokens: [t0, t1, t2, t3, t4]
// Input:  [t0, t1, t2, t3]
// Target: [t1, t2, t3, t4]

Batch TextDataset::get_batch(Device device) {
    std::vector<int32_t> input_data(batch_size_ * seq_len_);
    std::vector<int32_t> label_data(batch_size_ * seq_len_);
    
    for (int64_t b = 0; b < batch_size_; ++b) {
        auto& doc = tokenized_docs_[current_idx_++];
        for (int64_t t = 0; t < seq_len_; ++t) {
            input_data[b * seq_len_ + t] = doc[t];
            label_data[b * seq_len_ + t] = doc[t + 1];  // Next token prediction
        }
    }
    
    Tensor inputs = empty({batch_size_, seq_len_}, DType::Int32, device);
    Tensor labels = empty({batch_size_, seq_len_}, DType::Int32, device);
    inputs.copy_from_host(input_data.data());
    labels.copy_from_host(label_data.data());
    
    return {inputs, labels, {}};
}
```

### 6.3 Sequence Packing (Optional Optimization)

For efficiency, pack multiple short documents into single sequences:

```
[BOS] doc1 tokens [EOS] [BOS] doc2 tokens [EOS] [PAD] [PAD] ...
```

---

## 7. Phase 5: Training Loop

### 7.1 Optimizer Setup

Vesper provides AdamW which is standard for LLM training:

```cpp
// Collect parameters
auto params = model->parameters();

// Create AdamW optimizer (standard for LLMs)
optim::AdamW optimizer(
    params,
    /*lr=*/1e-4f,           // Learning rate
    /*beta1=*/0.9f,
    /*beta2=*/0.95f,        // Llama uses 0.95
    /*eps=*/1e-8f,
    /*weight_decay=*/0.1f   // Weight decay
);
```

### 7.2 Learning Rate Schedule

Implement a cosine schedule with warmup:

```cpp
float get_lr(int64_t step, int64_t warmup_steps, int64_t total_steps, 
             float max_lr, float min_lr) {
    if (step < warmup_steps) {
        // Linear warmup
        return max_lr * static_cast<float>(step) / warmup_steps;
    } else {
        // Cosine decay
        float progress = static_cast<float>(step - warmup_steps) / 
                        (total_steps - warmup_steps);
        return min_lr + 0.5f * (max_lr - min_lr) * (1.0f + std::cos(M_PI * progress));
    }
}

// Update learning rate each step
void update_lr(optim::AdamW& optimizer, float new_lr) {
    optimizer.set_lr(new_lr);
}
```

### 7.3 Training Step

```cpp
void train_step(models::TransformerLM& model, 
                optim::AdamW& optimizer,
                const Tensor& input_ids,    // [B, T]
                const Tensor& labels) {     // [B, T]
    
    // 1. Zero gradients
    optimizer.zero_grad();
    
    // 2. Forward pass
    Tensor logits = model.forward(input_ids);  // [B, T, V]
    
    // 3. Compute loss
    // Reshape for cross entropy: [B*T, V] and [B*T]
    int64_t B = logits.shape()[0];
    int64_t T = logits.shape()[1];
    int64_t V = logits.shape()[2];
    
    Tensor logits_flat = logits.view({B * T, V});
    Tensor labels_flat = labels.view({B * T});
    
    Tensor loss = nn::functional::cross_entropy_loss(logits_flat, labels_flat);
    
    // 4. Backward pass
    loss.backward();
    
    // 5. Gradient clipping (important for stability)
    // Note: Implement gradient clipping manually if not in optimizer
    float grad_norm = compute_grad_norm(model.parameters());
    if (grad_norm > 1.0f) {
        scale_gradients(model.parameters(), 1.0f / grad_norm);
    }
    
    // 6. Update weights
    optimizer.step();
}
```

### 7.4 Full Training Loop

```cpp
void train(models::TransformerLM& model, TextDataset& dataset,
           int64_t num_epochs, Device device) {
    
    auto params = model.parameters();
    optim::AdamW optimizer(params, 1e-4f, 0.9f, 0.95f, 1e-8f, 0.1f);
    
    int64_t total_steps = num_epochs * dataset.steps_per_epoch();
    int64_t warmup_steps = total_steps / 10;  // 10% warmup
    int64_t step = 0;
    
    for (int64_t epoch = 0; epoch < num_epochs; ++epoch) {
        dataset.reset();
        
        while (!dataset.epoch_complete()) {
            // Update learning rate
            float lr = get_lr(step, warmup_steps, total_steps, 1e-4f, 1e-5f);
            optimizer.set_lr(lr);
            
            // Get batch
            auto [input_ids, labels, _] = dataset.get_batch(device);
            
            // Training step
            train_step(model, optimizer, input_ids, labels);
            
            // Logging
            if (step % 100 == 0) {
                std::cout << "Step " << step << " | LR: " << lr << std::endl;
            }
            
            // Checkpointing
            if (step % 1000 == 0) {
                io::save_model(model, "checkpoint_" + std::to_string(step) + ".safetensors");
            }
            
            step++;
        }
    }
}
```

### 7.5 Gradient Accumulation (For Large Effective Batch Sizes)

```cpp
void train_with_accumulation(models::TransformerLM& model,
                             optim::AdamW& optimizer,
                             TextDataset& dataset,
                             int64_t accumulation_steps,
                             Device device) {
    
    optimizer.zero_grad();
    
    for (int64_t acc = 0; acc < accumulation_steps; ++acc) {
        auto [input_ids, labels, _] = dataset.get_batch(device);
        
        Tensor logits = model.forward(input_ids);
        Tensor loss = compute_loss(logits, labels);
        
        // Scale loss for accumulation
        Tensor scaled_loss = loss / static_cast<float>(accumulation_steps);
        scaled_loss.backward();
    }
    
    // Single optimizer step after accumulation
    optimizer.step();
}
```

---

## 8. Phase 6: Inference and Generation

### 8.1 Using Vesper's Generator

Vesper provides a high-level `Generator` class:

```cpp
#include <vesper/generation/generator.h>

// Set model to evaluation mode
model->eval();

// Create generator
generation::Generator generator(model.get());

// Configure sampling
generation::SamplingParams params;
params.temperature = 0.7f;
params.top_k = 40;
params.top_p = 0.9f;
params.max_new_tokens = 256;
params.stop_token_ids = {tokenizer.eos_id()};
params.do_sample = true;

// Tokenize prompt
std::vector<int32_t> prompt_ids = tokenizer.encode("Once upon a time");

// Create tensor
Tensor prompt = empty({1, (int64_t)prompt_ids.size()}, DType::Int32, device);
prompt.copy_from_host(prompt_ids.data());

// Generate
Tensor output_ids = generator.generate(prompt, params);

// Decode back to text
std::vector<int32_t> result(output_ids.numel());
output_ids.copy_to_host(result.data());
std::string text = tokenizer.decode(result);
```

### 8.2 Streaming Generation

For real-time output:

```cpp
Tensor output_ids = generator.generate_streaming(prompt, params, 
    [&tokenizer](int64_t batch_idx, int64_t token_id) -> bool {
        // Decode and print each token as it's generated
        std::string token_str = tokenizer.decode({static_cast<int32_t>(token_id)});
        std::cout << token_str << std::flush;
        return true;  // Continue generation
    }
);
```

### 8.3 KV Cache Management

For efficient autoregressive generation:

```cpp
// Initialize KV caches before generation
model->init_cache(/*batch_size=*/1, device);

// First pass: process full prompt
Tensor logits = model->forward_with_cache(prompt, /*start_pos=*/0);

// Subsequent passes: single token at a time
for (int i = 0; i < max_new_tokens; ++i) {
    // Sample next token
    Tensor next_token = sample_token(logits);
    
    // Forward with cache (only processes new token)
    logits = model->forward_with_cache(next_token, /*start_pos=*/prompt_len + i);
}

// Clear cache when done
model->clear_cache();
```

### 8.4 Batch Generation

Generate multiple sequences in parallel:

```cpp
// Batch of prompts
std::vector<std::string> prompts = {"Hello", "Once upon", "The meaning of"};
std::vector<std::vector<int32_t>> encoded;
int64_t max_len = 0;

for (const auto& p : prompts) {
    encoded.push_back(tokenizer.encode(p));
    max_len = std::max(max_len, (int64_t)encoded.back().size());
}

// Pad to same length
Tensor batch = zeros({(int64_t)prompts.size(), max_len}, DType::Int32, device);
// ... copy and pad each sequence

// Generate
Tensor outputs = generator.generate(batch, params);
```

---

## 9. Phase 7: Checkpointing and Serialization

### 9.1 Saving Model Weights

Using Vesper's SafeTensors support:

```cpp
// Save entire model
io::save_model(*model, "llama2_trained.safetensors");
```

### 9.2 Loading Model Weights

```cpp
// Create model with same architecture
auto config = models::TransformerConfig::llama2_7b();
auto model = models::create_model(config);

// Load weights
io::SafetensorsReader reader("llama2_trained.safetensors");
auto weights = reader.load_all(device);

// Load into model
StateDict state_dict;
for (const auto& [name, tensor] : weights) {
    state_dict[name] = tensor;
}
model->load_state_dict(state_dict);
```

### 9.3 Loading Official Llama-2 Weights

If loading from Meta's official weights:

```cpp
// Load sharded weights (Llama-2 70B comes in multiple files)
auto weights = io::load_sharded_safetensors("./llama-2-7b/", device);

// Map weight names if needed (Vesper names may differ slightly)
StateDict mapped_state;
for (const auto& [name, tensor] : weights) {
    std::string vesper_name = map_llama_to_vesper_name(name);
    mapped_state[vesper_name] = tensor;
}

model->load_state_dict(mapped_state);
```

### 9.4 Checkpointing Training State

For resumable training, save optimizer state too:

```cpp
struct Checkpoint {
    int64_t step;
    int64_t epoch;
    float learning_rate;
    // Model weights saved separately
};

void save_checkpoint(const models::TransformerLM& model,
                     const optim::AdamW& optimizer,
                     int64_t step, int64_t epoch, float lr,
                     const std::string& path) {
    // Save model
    io::save_model(model, path + "/model.safetensors");
    
    // Save training state (simple JSON or binary)
    std::ofstream f(path + "/training_state.json");
    f << "{\"step\":" << step << ",\"epoch\":" << epoch 
      << ",\"lr\":" << lr << "}";
}
```

---

## 10. Implementation Checklist

### Phase 1: Setup
- [ ] Create project directory structure
- [ ] Set up CMakeLists.txt with Vesper and SentencePiece
- [ ] Verify build compiles and links

### Phase 2: Tokenizer
- [ ] Implement SentencePiece wrapper class
- [ ] Test encode/decode roundtrip
- [ ] Handle special tokens (BOS, EOS, PAD)
- [ ] Train or obtain tokenizer model

### Phase 3: Model
- [ ] Instantiate `TransformerConfig::llama2_7b()` (or custom config)
- [ ] Create model with `models::create_model()`
- [ ] Verify model moves to GPU correctly
- [ ] Test forward pass with random input

### Phase 4: Data Pipeline
- [ ] Implement text file loader
- [ ] Implement batch creation with tokenization
- [ ] Test data loader produces correct shapes
- [ ] Implement shuffling

### Phase 5: Training
- [ ] Set up AdamW optimizer
- [ ] Implement learning rate scheduler
- [ ] Implement training step with gradient clipping
- [ ] Add loss logging
- [ ] Add periodic checkpointing

### Phase 6: Inference
- [ ] Test generation with `Generator` class
- [ ] Implement streaming output
- [ ] Verify KV cache speeds up generation
- [ ] Test different sampling strategies

### Phase 7: Serialization
- [ ] Test save/load cycle
- [ ] Verify loaded model produces same outputs
- [ ] (Optional) Test loading official Llama weights

---

## 11. Appendix: Vesper API Reference

### Tensor Creation

```cpp
Tensor empty(shape, dtype, device, requires_grad=false);
Tensor zeros(shape, dtype, device, requires_grad=false);
Tensor ones(shape, dtype, device, requires_grad=false);
Tensor randn(shape, dtype, device, requires_grad=false);
```

### Tensor Operations

```cpp
// Shape manipulation
tensor.view({new_shape});
tensor.reshape({new_shape});
tensor.transpose(dim0, dim1);
tensor.contiguous();

// Device transfer
tensor.to(Device::HIP);
tensor.to_(device);  // in-place

// Data access
tensor.copy_from_host(ptr);
tensor.copy_to_host(ptr);
```

### Neural Network Layers

```cpp
nn::Embedding(num_embeddings, embedding_dim);
nn::Linear(in_features, out_features, bias=true);
nn::RMSNorm(hidden_size, eps);
nn::SwiGLUMLP(d_model, hidden_dim, bias=false);
nn::GroupedQueryAttention(embed_dim, num_heads, num_kv_heads, max_seq_len);
```

### Functional API

```cpp
nn::functional::relu(x);
nn::functional::gelu(x);
nn::functional::silu(x);
nn::functional::softmax(x, dim);
nn::functional::cross_entropy_loss(logits, targets);
```

### Optimizers

```cpp
optim::Adam(params, lr, beta1, beta2, eps, weight_decay);
optim::AdamW(params, lr, beta1, beta2, eps, weight_decay);
optim::SGD(params, lr, momentum, weight_decay);
```

### Autograd

```cpp
tensor.backward();                    // Compute gradients
tensor.grad();                        // Access gradient
autograd::NoGradGuard guard;          // Disable gradient tracking
```

### Serialization

```cpp
io::save_model(module, path);
io::SafetensorsReader(path).load_all(device);
module.state_dict();
module.load_state_dict(dict);
```

---

## Summary

This plan leverages Vesper's comprehensive transformer infrastructure:

1. **Model**: Use `models::TransformerLM` with `TransformerConfig::llama2_7b()` - no need to build from scratch
2. **Components**: All Llama-2 components are available: `RMSNorm`, `SwiGLUMLP`, `GroupedQueryAttention`, `RoPE`
3. **Generation**: Use `generation::Generator` with `SamplingParams` for flexible text generation
4. **Training**: Use `optim::AdamW` with cosine schedule and gradient clipping
5. **Serialization**: Use SafeTensors format via `io::save_model()` and `io::SafetensorsReader`

The main custom work required is:
- SentencePiece integration (tokenizer wrapper)
- Data pipeline (text loading and batching)
- Training loop orchestration
- Learning rate scheduling
