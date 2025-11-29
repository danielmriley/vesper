# TinyShakespeare Training Improvement Plan

**Date**: Current Session  
**Goal**: Match Karpathy's nanoGPT results (~1.48 validation loss on TinyShakespeare)  
**Current Status**: Loss 2.27 after 50 epochs on full dataset

---

(Don't store the "full_training.log" file. It got VERY big last time...)

## 1. Executive Summary

### Current Performance
- **Training Loss**: 2.62 → 2.27 (50 epochs, 1.1M chars)
- **Model Size**: ~2.7M parameters
- **Architecture**: 6 layers, 192 embed_dim, 6 heads, 128 context

### Target Performance (nanoGPT Reference)
- **Validation Loss**: ~1.48
- **Model Size**: ~10M parameters  
- **Architecture**: 6 layers, 384 embed_dim, 6 heads, 256 context
- **Training**: 5000 iterations with cosine LR schedule + warmup

---

## 2. Issues Identified

### 2.1 Critical Performance Issues

#### A. Causal Mask Created Every Forward Pass (HIGH PRIORITY)
**Location**: `src/nn/functional.cpp:376-396`

```cpp
// Current: Creates mask on CPU and transfers to GPU EVERY forward pass
std::vector<float> mask_data(S * S);
// ... populate mask ...
Tensor mask = vesper::empty({S, S}, DType::Float32, Device::CPU);
mask.copy_from_host(mask_data.data());
mask = mask.to(query.device());  // <-- GPU transfer every time!
```

**Impact**: Significant overhead from repeated CPU→GPU transfers, especially with large sequence lengths.

**Solution**: Cache the causal mask or create it directly on GPU:
```cpp
// Option 1: Static cache by sequence length
static std::unordered_map<int64_t, Tensor> causal_mask_cache;

// Option 2: Create directly on GPU using triu kernel
Tensor mask = ops::triu(vesper::full({S, S}, neg_inf, DType::Float32, device), 1);
```

### 2.2 Model Architecture Gaps

#### B. No Weight Tying (MEDIUM PRIORITY)
nanoGPT ties the embedding weights to the output projection (lm_head):
```python
self.transformer.wte.weight = self.lm_head.weight  # share weights
```

This reduces parameters and improves generalization for language modeling.

**Solution**: Add `tie_weights()` method to TransformerLM or do it manually in user code.

#### C. Model Too Small (MEDIUM PRIORITY)
| Parameter | Current | nanoGPT |
|-----------|---------|---------|
| embed_dim | 192 | 384 |
| context_len | 128 | 256 |
| params | ~2.7M | ~10M |

**Solution**: Scale up the model configuration.

### 2.3 Training Configuration Gaps

#### D. No Learning Rate Schedule (HIGH PRIORITY)
nanoGPT uses:
- Linear warmup for first ~100-200 iterations
- Cosine decay to 10% of max LR
- Max LR: 1e-3, Min LR: 1e-4

**Current Vesper**: Has schedulers (`CosineAnnealingLR`, `StepLR`) but may need warmup.

**Solution**: 
1. Check if warmup is supported
2. Add `get_lr()` method to schedulers if missing
3. Use cosine schedule with warmup

#### E. Insufficient Training (HIGH PRIORITY)
- **Current**: 50 epochs × 200 batches = 10,000 batch iterations
- **nanoGPT**: 5,000 iterations but with better LR schedule

The training may need to continue longer OR with better hyperparameters.

---

## 3. Architecture Verification (PASSED ✅)

### Pre-LN Transformer ✅
```cpp
// TransformerBlock::forward() - Correct Pre-LN order
x = ln1(x);           // Normalize FIRST
x = attn.forward(x);  // Then attention
x = residual + x;     // Then residual
```

### Scaled Dot-Product Attention ✅
- Proper QKV computation
- Correct softmax over keys
- Numerical stability (max subtraction in softmax)

### MLP Structure ✅
- 4x expansion (embed_dim → 4×embed_dim → embed_dim)
- GELU activation
- Dropout after projection

### RoPE Implementation ✅
- Now optional via `use_rope` parameter
- Correct frequency computation
- Applied to Q and K only

---

## 4. Implementation Roadmap

### Phase 1: Quick Wins ✅ COMPLETED
1. [x] **Cache causal mask** - Avoid repeated CPU→GPU transfer
2. [x] **Add LR warmup scheduler** - `CosineAnnealingWithWarmup`
3. [x] **Weight tying** - `nn::utils::tie_weights()` utility added
4. [ ] **Scale up model** - Use 384 embed_dim, 256 context
5. [ ] **Train longer** - Run for 5000+ iterations

### Phase 2: Training Improvements (User's Demo Project)
6. [ ] **Cosine LR schedule** - Use `CosineAnnealingWithWarmup`
7. [ ] **Better hyperparameters** - Match nanoGPT config:
   - LR: 1e-3 (peak) → 1e-4 (min)
   - AdamW: β1=0.9, β2=0.95, weight_decay=0.1
   - Gradient clip: 1.0

### Phase 3: Advanced Optimizations (optional)
8. [ ] **Flash Attention** - Use if available for longer sequences
9. [ ] **Mixed precision** - FP16/BF16 for faster training

---

## 5. Recommended Training Configuration

```cpp
// Model configuration (matches ~10M params)
const int vocab_size = 65;       // TinyShakespeare characters
const int n_layers = 6;
const int n_heads = 6;
const int n_embd = 384;          // Scaled up from 192
const int context_len = 256;     // Scaled up from 128
const float dropout = 0.0;       // No dropout for overfitting test

// Training configuration
const int batch_size = 64;
const int total_iterations = 5000;
const float max_lr = 1e-3;
const float min_lr = 1e-4;
const int warmup_iters = 100;

// Model setup with weight tying
nn::Embedding wte(vocab_size, n_embd);
nn::Linear lm_head(n_embd, vocab_size, false);  // no bias
nn::utils::tie_weights(wte, lm_head);  // Share embedding weights

// Optimizer
optim::AdamW optimizer(model.parameters(), max_lr, 0.9, 0.95, 1e-8, 0.1);

// LR Scheduler with warmup
optim::CosineAnnealingWithWarmup scheduler(optimizer, total_iterations, warmup_iters, min_lr);

// Initialize weights
nn::utils::init_transformer_weights(model, n_layers);
```

---

## 6. Success Metrics

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Val Loss (v2/RoPE) | **2.68** | <1.5 | 🟡 |
| Val Loss (v3/Learned) | 2.83 | <1.5 | 🔴 |
| Overfitting (20k chars) | 0.086 | <0.1 | ✅ |
| Text Quality | Recognizable | Fluent | 🟡 |
| GPU Utilization | ~90% | >90% | ✅ |

---

## 7. Experimental Results (Session 2)

### Training Runs Completed

| Version | Position Encoding | Dropout | Batch | β2 | Best Val Loss | Time |
|---------|------------------|---------|-------|-----|---------------|------|
| v2 | RoPE | 0.0 | 32 | 0.95 | **2.68** | 26 min |
| v3 | Learned | 0.2 | 16 | 0.99 | 2.83 | stuck |
| nanoGPT | Learned | 0.2 | 64 | 0.99 | ~1.48 | target |

### Key Findings

1. **RoPE vs Learned Position Embeddings**: RoPE (v2) performed better in our tests
   - Could be due to batch size constraints (32 vs 64)
   - RoPE is more parameter-efficient

2. **Dropout Impact**: 20% dropout with batch_size=16 is too aggressive
   - Model couldn't learn - loss increased over training
   - nanoGPT uses batch_size=64 with dropout=0.2

3. **GPU Memory Limitation**: AMD GPU can't handle batch_size=64
   - Limited to batch_size=32 with RoPE
   - Limited to batch_size=16 with learned embeddings + dropout

### Remaining Gap Analysis (2.68 → 1.48)

The ~1.2 loss gap could be due to:

1. **Mixed Precision (FP16/BF16)**: nanoGPT uses autocast for training
   - Would allow larger batch sizes
   - Need to implement in Vesper

2. **PyTorch Compilation**: nanoGPT uses `torch.compile()`
   - Significant speedup from kernel fusion
   - Not applicable to our C++ implementation

3. **Flash Attention**: More memory efficient
   - Would allow batch_size=64
   - Needs implementation in Vesper

4. **Training Duration**: nanoGPT may train longer in practice

5. **Hyperparameter Sensitivity**: Exact LR schedule matching

---

## 8. Next Steps (Priority Order)

1. **Implement Flash Attention** - Enable batch_size=64
2. **Add FP16/BF16 Support** - Mixed precision training
3. **Continue Training** - Run v2 for 10,000+ iterations
4. **Optimize GEMM Kernels** - Better GPU utilization

---

## Appendix: nanoGPT Reference Config

```python
# nanoGPT train_shakespeare_char.py
n_layer = 6
n_head = 6  
n_embd = 384
block_size = 256
batch_size = 64
learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
warmup_iters = 100
dropout = 0.2
beta2 = 0.99
```

