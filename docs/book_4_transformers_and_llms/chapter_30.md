```markdown

# Vesper Future Plans - Chapter 30: Normalization and Stability

## 1. Goal

Implement normalization layers and stable activation functions required for Transformers. These components are essential for stabilizing the training of deep sequence models.

## 2. Features

-   **LayerNorm:** Implement `nn.LayerNorm`. Unlike BatchNorm (which works across the batch dimension), LayerNorm normalizes across the feature dimension, making it independent of batch size and ideal for sequence tasks.
-   **RMSNorm:** Implement Root Mean Square Normalization (`nn.RMSNorm`). A simplified version of LayerNorm used in Llama and other modern LLMs for better performance.
-   **Stable Softmax:** Implement the Softmax function with the "subtract max" trick to prevent numerical overflow when dealing with large logits.
-   **GELU / SwiGLU:** Implement Gaussian Error Linear Units (GELU) or Swish-Gated Linear Units (SwiGLU), the standard activations for Transformers.

## 3. Why It's Next

The Transformer architecture relies heavily on LayerNorm (or RMSNorm) and Softmax. Without these, the model will not train. This chapter builds the specific building blocks needed before assembling the full Transformer block.

```