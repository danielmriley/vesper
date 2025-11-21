```markdown

# Vesper Future Plans - Chapter 33.1: Efficient Inference (KV Cache)

## 1. Goal

Optimize the Transformer model for autoregressive generation (inference) by implementing Key-Value (KV) Caching.

## 2. Features

-   **The Problem:** In naive autoregressive generation, we re-compute the attention for the entire sequence at every step, even though the past tokens haven't changed. This is $O(N^2)$ complexity for generating $N$ tokens.
-   **The Solution (KV Cache):** Cache the Key and Value matrices for previous tokens and only compute the Query, Key, and Value for the *new* token. Concatenate the new K/V with the cached K/V.
-   **Implementation:**
    -   Modify the `Attention` module to accept an optional `past_key_values` argument.
    -   Return the updated `past_key_values` along with the output.
    -   Update the `forward` pass to handle the "step 1" vs "step N" logic.

## 3. Why It's Next

While not strictly necessary for *training*, KV Caching is mandatory for *using* the model. Without it, generating text from even a small LLM is excruciatingly slow.

```