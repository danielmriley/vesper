# Training Loop Design: TinyStories 250M Model

## 1. Overview
This document outlines the design for training a ~250M parameter Transformer model on the TinyStories dataset using the `vesper` library. The goal is to implement a complete training pipeline including data preprocessing, model configuration, and a C++ training loop.

## 2. Requirements
- **Model Size**: ~250M parameters.
- **Architecture**: Transformer (Llama-style preferred for performance/modernity).
- **Context Length**: 2048 tokens.
- **Layers**: $\ge 16$.
- **Heads**: 12.
- **Tokenizer**: BPE (SentencePiece).
- **Dataset**: `part_0001.txt` from TinyStories.
- **Iterations**: 100,000.

## 3. Model Configuration
To achieve ~250M parameters with the constraints ($H=12, L \ge 16$), we propose the following configuration:

| Hyperparameter | Value | Notes |
| :--- | :--- | :--- |
| **Layers ($L$)** | 32 | Deep network to reach parameter count. |
| **Hidden Dim ($d$)** | 768 | Standard size, divisible by 12. |
| **Heads ($H$)** | 12 | As requested. |
| **Head Dim ($d_k$)** | 64 | $768 / 12 = 64$. Standard. |
| **Context ($T$)** | 2048 | Standard for Llama 2. |
| **Vocab Size ($V$)** | 32,000 | Standard Llama size. |
| **FFN Dim** | 2048 | $\approx 2.66 \times d$ (SwiGLU usually uses $\frac{8}{3}d$). |

**Parameter Count Estimation:**
- **Attention + FFN**: $\approx 12 \times L \times d^2 \approx 12 \times 32 \times 768^2 \approx 226 \text{ M}$.
- **Embeddings**: $V \times d = 32000 \times 768 \approx 24.5 \text{ M}$.
- **Total**: $\approx 250.5 \text{ M}$.

This fits the requirements perfectly.

## 4. Data Preprocessing (Python)
Since `vesper` does not currently have a built-in BPE trainer, we will use a Python script to:
1.  Train a SentencePiece BPE model on `part_0001.txt`.
2.  Tokenize the text file into a binary format (flat array of `uint16_t` tokens).
3.  Save `tokenizer.model` and `train.bin`.

**Script: `scripts/preprocess_tinystories.py`**
- Inputs: Raw text file.
- Outputs: `vocab.model`, `data.bin`.

## 5. C++ Implementation

### 5.1. Dataset Class (`TinyStoriesDataset`)
We will implement a custom `Dataset` in `src/train_tinystories.cpp` (or a separate header) that:
- Loads `train.bin` into memory (memory mapping preferred for larger files, but `std::vector` is fine for `part_0001.txt` which is ~200MB).
- `get_item(index)`: Returns a random slice of length `seq_len + 1` (input + target).
- `size()`: Defined by total tokens / sequence length.

### 5.2. Training Loop (`examples/train_tinystories.cpp`)
The main application will perform the following:

1.  **Setup**:
    - Load `TransformerConfig`.
    - Initialize `TransformerLM` model.
    - Move model to GPU (`Device::kCUDA` or `Device::kHIP`).

2.  **Data**:
    - Initialize `TinyStoriesDataset`.
    - Initialize `DataLoader` with batch size (e.g., 16 or 32 depending on VRAM).

3.  **Optimizer**:
    - Use `AdamW` from `vesper::optim`.
    - Learning rate schedule: Cosine decay with warmup.
    - Parameters: $\beta_1=0.9, \beta_2=0.95, \text{wd}=0.1$.

4.  **Loop (100,000 iterations)**:
    - Fetch batch.
    - `model.forward(input)`.
    - `model.compute_loss(logits, target)`.
    - `loss.backward()`.
    - `optimizer.step()`.
    - `optimizer.zero_grad()`.
    - Logging: Print loss, tokens/sec every N steps.
    - Checkpointing: Save model weights every K steps.

## 6. Action Plan
1.  Create `scripts/preprocess_tinystories.py` to generate data.
2.  Run preprocessing on the user's machine.
3.  Create `examples/train_tinystories.cpp` implementing the design.
4.  Build and run.
