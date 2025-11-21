```markdown

# Vesper Future Plans - Chapter 31: Batch GEMM and 3D/4D Tensors

## 1. Goal

Extend the General Matrix Multiply (GEMM) operation to support batched inputs. This is the computational engine of the Multi-Head Attention mechanism.

## 2. Features

-   **Batch GEMM Kernel:** Implement a kernel that performs matrix multiplication on batches of matrices.
    -   Input: `(B, M, K)` and `(B, K, N)` -> Output: `(B, M, N)`.
    -   Support broadcasting for the batch dimension (e.g., `(1, M, K)` @ `(B, K, N)`).
-   **4D Tensor Support:** Ensure the `Tensor` class and stride logic correctly handle 3D and 4D tensors (e.g., `(Batch, Heads, Seq, Dim)`).
-   **Matmul Operator:** Update `ops::matmul` to detect tensor rank and dispatch to the appropriate GEMM (2D) or Batch GEMM (3D+) kernel.

## 3. Why It's Next

Multi-Head Attention involves computing attention scores for multiple "heads" in parallel for every sequence in a batch. This is mathematically a Batch GEMM. Without this, we cannot implement efficient Attention.

```