
# Vesper Future Plans - Chapter 27: Simple Recurrent Neural Networks (`nn.RNN`)

## 1. Goal

Implement a basic Recurrent Neural Network (RNN) layer for processing sequential data. This introduces the concept of a hidden state and processing data over a time dimension.

## 2. Features

-   **`RNNCell`:** First, a single `RNNCell` module will be implemented. Its `forward` method takes an input `x_t` (at time `t`) and the previous hidden state `h_{t-1}` and produces the next hidden state `h_t`. This is typically composed of one or two `Linear` layers and a `tanh` or `ReLU` activation.
-   **`RNN` Module:** The main `RNN` module will contain an instance of the `RNNCell`. Its `forward` method will take an entire input sequence (e.g., a tensor of shape `[sequence_length, batch_size, input_features]`). It will then loop over the sequence dimension, calling the cell at each time step and passing the hidden state from one step to the next.
-   **Backpropagation Through Time (BPTT):** The primary challenge of this chapter is ensuring the autograd graph is correctly constructed through the unrolled time steps. When `backward()` is called on the final output, the gradients must flow back through the entire sequence, which the existing autograd engine should handle naturally if the loop is implemented correctly.

## 3. Why It's Next

RNNs are the classical architecture for sequence modeling tasks like time series analysis, machine translation, and text generation. While often superseded by LSTMs or Transformers, implementing a simple RNN is the foundational step to understanding and building more complex recurrent models.
