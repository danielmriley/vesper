```markdown

# Vesper Future Plans - Chapter 25.2: The Lion Optimizer (`optim.Lion`)

## 1. Goal

Implement the Lion (Evolved Sign Momentum) optimizer. Discovered by Google DeepMind using symbolic program search, Lion is a modern optimizer that often outperforms AdamW while being more memory efficient.

## 2. Features

-   **Memory Efficiency:** Unlike Adam, which tracks two moments (mean and variance) per parameter, Lion only tracks the momentum. This reduces the optimizer state memory footprint by roughly 50%, which is crucial for training large models on limited GPU memory.
-   **Sign-Based Updates:** The core update rule uses the sign of the gradient and momentum, making the magnitude of the update uniform across dimensions (scaled by the learning rate).
    -   Update rule: `p = p - lr * (sign(beta1 * m + (1 - beta1) * g) + weight_decay * p)`
-   **Implementation:**
    -   Create a new `Lion` class inheriting from `Optimizer`.
    -   Implement the `step()` method.
    -   Requires implementing a `sign` element-wise operation kernel if not already present.

## 3. Why It's Next

Lion represents the cutting edge of optimization research (as of 2023/2024). Adding it alongside Adam gives Vesper a "modern" feel and provides a high-performance alternative for training LLMs and Vision Transformers.

```