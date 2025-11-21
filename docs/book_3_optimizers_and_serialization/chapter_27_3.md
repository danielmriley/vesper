```markdown

# Vesper Future Plans - Chapter 27.3: Learning Rate Schedulers

## 1. Goal

Implement learning rate schedulers to adjust the learning rate during training. This is critical for training stability and achieving state-of-the-art performance, especially for Transformers which typically require a warmup phase followed by decay.

## 2. Features

-   **Scheduler Base Class:** A base class that wraps an optimizer and modifies its `param_groups`' learning rates.
-   **Linear Warmup:** Linearly increase the learning rate from 0 to `max_lr` over a fixed number of steps.
-   **Cosine Decay:** Implement `CosineAnnealingLR`, which decreases the learning rate following a cosine curve. This is the standard for training LLMs (often combined with warmup).
-   **Step Decay:** `StepLR` to reduce the learning rate by a factor (gamma) every few epochs.

## 3. Why It's Next

Training a Transformer with a constant learning rate often results in divergence or suboptimal convergence. Implementing schedulers completes the "training loop" toolkit.

```