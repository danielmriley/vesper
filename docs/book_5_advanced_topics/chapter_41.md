# Vesper Future Plans - Chapter 41: Distributed Training (DDP)

## 1. Goal

Implement Distributed Data Parallel (DDP) training to scale model training across multiple GPUs and multiple machines.

## 2. Concepts

-   **Single Program Multiple Data (SPMD):** The same script runs on every GPU.
-   **Gradient Synchronization:** After the backward pass, gradients must be averaged across all GPUs so that every model updates its weights identically.
-   **Collectives:** The core primitive is `AllReduce`, which sums data from all processes and distributes the result back to all.

## 3. Implementation Plan

1.  **Communication Backend:** Implement a simple TCP-based ring-allreduce for CPU (educational) and wrap NCCL/RCCL for GPU (performance).
2.  **`ProcessGroup`:** A class to manage connectivity (world size, rank, initialization).
3.  **`DistributedDataParallel` Module:** A wrapper around `nn::Module`.
    -   **Broadcast Buffers:** On init, broadcast rank 0's weights to all others.
    -   **Hook into Backward:** Register a hook on the backward pass. When a parameter's gradient is ready, trigger an async `AllReduce` on it.

## 4. Why It's Next

Deep learning scales with compute. To train anything beyond a toy model (e.g., a decent-sized Transformer), multi-GPU support is mandatory.
