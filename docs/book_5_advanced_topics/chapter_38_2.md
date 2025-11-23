# Vesper Future Plans - Chapter 38.2: Fused Attention (FlashAttention)

## 1. Goal

Implement a simplified version of **FlashAttention** (IO-aware exact attention). This is the single most impactful optimization for modern Large Language Models.

## 2. The Memory Bottleneck

Standard Attention calculates `S = Q @ K^T`, then `P = softmax(S)`, then `O = P @ V`.
The intermediate matrix `S` (scores) and `P` (probabilities) are of size `[Sequence_Length, Sequence_Length]`. For long sequences, this $O(N^2)$ memory requirement is catastrophic. It forces creating massive matrices in HBM (High Bandwidth Memory) just to read them back immediately.

## 3. The FlashAttention Insight

FlashAttention uses **tiling** to compute the attention output in blocks without ever materializing the full `S` or `P` matrices in global memory.

1.  Load blocks of `Q`, `K`, `V` into shared memory (SRAM).
2.  Compute a block of scores `S_ij` in registers.
3.  Update the `softmax` running statistics (max and sum) online.
4.  Accumulate the weighted sum into `O` in registers/shared memory.
5.  Write only the final `O` to global memory.

This reduces memory access from $O(N^2)$ to $O(N)$, making attention linear in memory complexity with respect to sequence length.

## 4. Implementation Strategy (Educational Version)

Implementing full FlashAttention 2/3 is extremely complex (assembly, warp-specialization). Vesper will implement a **"FlashAttention-Lite"** C++ kernel.

*   **Tiling**: We will adapt our `gemm_register_tiled_kernel` logic to handle the 3-matrix multiplication flow.
*   **Online Softmax**: We will implement the specific "safe softmax" update rule required to normalize the partial sums as we iterate through blocks.
*   **Fwd/Bwd**: Start with the Forward pass. The Backward pass is significantly harder (requires re-computing attention) and will be a "stretch goal".

## 5. Why It's Next

Attention is the bottleneck for long contexts. Implementing a custom fused attention kernel distinguishes Vesper as a serious performance library rather than just a toy. It teaches the critical concept of "IO-Awareness" — designing algorithms based on memory hierarchy rather than just FLOPs.
