
# Vesper Future Plans - Chapter 28: Advanced Recurrent Layers (`nn.LSTM` / `nn.GRU`)

## 1. Goal

Implement more powerful and popular recurrent layers like LSTM (Long Short-Term Memory) or GRU (Gated Recurrent Unit). These architectures solve the vanishing gradient problem that affects simple RNNs, allowing them to learn long-range dependencies.

## 2. Features

-   **Gated Architectures:** The core of this chapter is implementing the "gating" mechanisms.
    -   **LSTM:** An `LSTMCell` involves implementing three gates (input, forget, output) and a cell state. This requires composing multiple `Linear` layers and `sigmoid` / `tanh` activations within a single forward pass to compute the new hidden and cell states.
    -   **GRU:** A `GRUCell` is a slightly simpler alternative with two gates (update and reset).
-   **Fused Kernels (Optional Optimization):** For performance, the multiple `Linear` layers within a cell are often implemented as a single, larger matrix multiplication, which is then sliced to get the results for each gate. This reduces the number of kernel launches. The initial implementation can use separate `Linear` layers for clarity, with fusion as a potential optimization.
-   **`LSTM` / `GRU` Module:** Similar to the `RNN` module, a full `LSTM` or `GRU` module will be created to loop over a sequence, applying the cell at each time step.

## 3. Why It's Next

For many years, LSTMs and GRUs were the state-of-the-art for most sequence modeling tasks. They are still widely used and are a powerful tool to have in any deep learning library. This builds directly on the work done in the `RNN` chapter.
