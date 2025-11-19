
# Vesper Future Plans - Chapter 26: The `nn.Embedding` Layer

## 1. Goal

Implement an `Embedding` layer. This layer is the standard way to handle discrete, categorical inputs (like words in a vocabulary) by mapping each category's integer index to a continuous, dense vector representation.

## 2. Features

-   **Lookup Table:** An `Embedding` layer is essentially a large, trainable lookup table. It holds a single weight `Tensor` of shape `[num_embeddings, embedding_dim]`.
-   **Forward Pass:** The forward pass takes a tensor of integer indices (`torch.long`) as input. For each index, it looks up the corresponding vector in the weight matrix. This is a specialized indexing operation, not a matrix multiplication.
-   **Backward Pass:** The backward pass is a sparse update. Gradients are computed for the output embeddings, and these gradients are then accumulated back *only* to the rows of the weight matrix that were looked up in the forward pass. This is much more efficient than a dense gradient update.

## 3. Why It's Next

The `Embedding` layer is the first and most fundamental component for virtually all Natural Language Processing (NLP) tasks, as well as any other task involving categorical inputs (e.g., recommendation systems). It's the gateway to building models that can understand language.
