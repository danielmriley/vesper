```markdown

# Vesper Future Plans - Chapter 26: Weight Initialization (`nn.init`)

## 1. Goal

Implement a suite of weight initialization functions. Proper initialization is critical for training deep networks; without it, gradients can vanish or explode, preventing convergence.

## 2. Features

-   **Uniform & Normal:** Basic `uniform_` and `normal_` initializers.
-   **Xavier (Glorot):** `xavier_uniform_` and `xavier_normal_`. Designed to keep the scale of gradients roughly the same in all layers. Essential for sigmoid/tanh activations.
-   **Kaiming (He):** `kaiming_uniform_` and `kaiming_normal_`. Designed for ReLU and its variants. This is the standard for modern vision models (ResNets) and Transformers.
-   **Constant & Zeros:** `constant_`, `zeros_`, `ones_` for biases and specific layers.

## 3. Why It's Next

As we move to deeper networks (like the ones in the upcoming chapters), random initialization is no longer sufficient. We need mathematically grounded initialization strategies to ensure our models actually train.

```