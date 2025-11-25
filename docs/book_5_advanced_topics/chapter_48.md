```markdown
# Chapter 48: Tensor Parallelism

## 1. Introduction

Large models don't fit on a single GPU. A 70B parameter model in FP16 requires:
- Weights: 140 GB
- KV Cache (4k context): ~40 GB
- Activations: ~20 GB
- **Total: ~200 GB**

But even the largest GPUs have ~80 GB. **Tensor Parallelism (TP)** distributes layers across multiple GPUs:

- Split weight matrices across GPUs
- Each GPU computes a portion of each operation
- Synchronize with AllReduce/AllGather

This chapter covers:
1. **Column-parallel Linear**: Split weights along output dimension
2. **Row-parallel Linear**: Split weights along input dimension
3. **Attention parallelism**: Distribute heads across GPUs
4. **Communication primitives**: AllReduce, AllGather

## 2. Parallelism Strategies Overview

| Strategy | What's Split | When to Use |
|----------|--------------|-------------|
| **Tensor Parallelism** | Each layer across GPUs | Largest models, low latency |
| **Pipeline Parallelism** | Different layers on different GPUs | Very deep models |
| **Data Parallelism** | Batch across GPUs | Training, high throughput |

This chapter focuses on **Tensor Parallelism**, the most common for LLM inference.

## 3. Communication Primitives

### 3.1 AllReduce

Combine tensors from all GPUs, broadcast result to all:

```
GPU 0: [a0]     [a0+b0+c0+d0]
GPU 1: [b0]  →  [a0+b0+c0+d0]
GPU 2: [c0]     [a0+b0+c0+d0]
GPU 3: [d0]     [a0+b0+c0+d0]
```

### 3.2 AllGather

Gather tensors from all GPUs, concatenate on each:

```
GPU 0: [a]     [a, b, c, d]
GPU 1: [b]  →  [a, b, c, d]
GPU 2: [c]     [a, b, c, d]
GPU 3: [d]     [a, b, c, d]
```

### 3.3 Implementation with RCCL (ROCm) / NCCL (CUDA)

```cpp
// include/vesper/distributed/comm.h

namespace vesper::distributed {

class Communicator {
public:
    Communicator(int world_size, int rank);
    ~Communicator();
    
    int world_size() const { return world_size_; }
    int rank() const { return rank_; }
    
    // Collective operations
    void all_reduce(Tensor& tensor, ReduceOp op = ReduceOp::Sum);
    void all_gather(Tensor& output, const Tensor& input);
    void reduce_scatter(Tensor& output, const Tensor& input, ReduceOp op = ReduceOp::Sum);
    
    // Point-to-point
    void send(const Tensor& tensor, int dst);
    void recv(Tensor& tensor, int src);
    
    // Synchronization
    void barrier();
    
private:
    int world_size_;
    int rank_;
    
#ifdef VESPER_USE_RCCL
    ncclComm_t nccl_comm_;
    hipStream_t stream_;
#endif
};

// Global communicator access
Communicator& get_comm();
void init_distributed(int world_size, int rank);

} // namespace vesper::distributed
```

```cpp
// src/distributed/rccl_comm.cpp

#include <rccl/rccl.h>

Communicator::Communicator(int world_size, int rank) 
    : world_size_(world_size), rank_(rank) 
{
    // Initialize RCCL
    ncclUniqueId id;
    if (rank == 0) {
        ncclGetUniqueId(&id);
    }
    
    // Broadcast ID to all ranks (via MPI or shared memory)
    broadcast_id(&id);
    
    // Create communicator
    NCCL_CHECK(ncclCommInitRank(&nccl_comm_, world_size, id, rank));
    
    // Create stream
    HIP_CHECK(hipStreamCreate(&stream_));
}

void Communicator::all_reduce(Tensor& tensor, ReduceOp op) {
    ncclDataType_t dtype = to_nccl_dtype(tensor.dtype());
    ncclRedOp_t nccl_op = (op == ReduceOp::Sum) ? ncclSum : ncclProd;
    
    NCCL_CHECK(ncclAllReduce(
        tensor.data_ptr(),
        tensor.data_ptr(),
        tensor.numel(),
        dtype,
        nccl_op,
        nccl_comm_,
        stream_));
    
    HIP_CHECK(hipStreamSynchronize(stream_));
}

void Communicator::all_gather(Tensor& output, const Tensor& input) {
    ncclDataType_t dtype = to_nccl_dtype(input.dtype());
    
    NCCL_CHECK(ncclAllGather(
        input.data_ptr(),
        output.data_ptr(),
        input.numel(),
        dtype,
        nccl_comm_,
        stream_));
    
    HIP_CHECK(hipStreamSynchronize(stream_));
}
```

## 4. Column-Parallel Linear

Split weight matrix along **output dimension** (columns):

```
Full: Y = X @ W^T,  W: [out_features, in_features]

Split W into [W0, W1, W2, W3] along rows (out_features)

GPU 0: Y0 = X @ W0^T   (computes first out_features/4 outputs)
GPU 1: Y1 = X @ W1^T
GPU 2: Y2 = X @ W2^T
GPU 3: Y3 = X @ W3^T

AllGather: Y = [Y0, Y1, Y2, Y3]
```

```cpp
// include/vesper/distributed/parallel_linear.h

namespace vesper::distributed {

class ColumnParallelLinear : public nn::Module {
public:
    ColumnParallelLinear(
        int64_t in_features,
        int64_t out_features,
        bool bias = true,
        bool gather_output = true);
    
    Tensor forward(const Tensor& input) override;
    
    // Access local weight slice
    Tensor& weight() { return weight_; }
    
private:
    int64_t in_features_;
    int64_t out_features_;           // Total output size
    int64_t out_features_per_rank_;  // Local output size
    bool gather_output_;
    
    Tensor weight_;  // [out_features_per_rank, in_features]
    std::optional<Tensor> bias_;
};

} // namespace vesper::distributed
```

```cpp
// src/distributed/column_parallel.cpp

ColumnParallelLinear::ColumnParallelLinear(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool gather_output)
    : in_features_(in_features),
      out_features_(out_features),
      gather_output_(gather_output)
{
    auto& comm = get_comm();
    int world_size = comm.world_size();
    
    VESPER_CHECK(out_features % world_size == 0,
        "out_features must be divisible by world_size");
    
    out_features_per_rank_ = out_features / world_size;
    
    // Each rank holds a slice of the weights
    weight_ = register_parameter("weight", 
        empty({out_features_per_rank_, in_features}));
    
    if (bias) {
        bias_ = register_parameter("bias", 
            zeros({out_features_per_rank_}));
    }
    
    // Initialize weights
    nn::init::kaiming_uniform_(weight_);
}

Tensor ColumnParallelLinear::forward(const Tensor& input) {
    // Local matmul: [batch, seq, in] @ [out_per_rank, in]^T = [batch, seq, out_per_rank]
    Tensor output = matmul(input, weight_.t());
    
    if (bias_) {
        output = output + *bias_;
    }
    
    if (gather_output_) {
        // AllGather to get full output on all GPUs
        auto& comm = get_comm();
        int world_size = comm.world_size();
        
        auto shape = output.shape();
        shape.back() = out_features_;  // Full size
        
        Tensor gathered = empty(shape, output.dtype(), output.device());
        comm.all_gather(gathered, output);
        
        return gathered;
    }
    
    return output;
}
```

## 5. Row-Parallel Linear

Split weight matrix along **input dimension** (rows):

```
Full: Y = X @ W^T,  W: [out_features, in_features]

Split W into [W0; W1; W2; W3] along columns (in_features)
Split X into [X0, X1, X2, X3] along last dimension

GPU 0: Y0 = X0 @ W0^T
GPU 1: Y1 = X1 @ W1^T
GPU 2: Y2 = X2 @ W2^T
GPU 3: Y3 = X3 @ W3^T

AllReduce: Y = Y0 + Y1 + Y2 + Y3
```

```cpp
class RowParallelLinear : public nn::Module {
public:
    RowParallelLinear(
        int64_t in_features,
        int64_t out_features,
        bool bias = true,
        bool input_is_parallel = true);
    
    Tensor forward(const Tensor& input) override;
    
private:
    int64_t in_features_;            // Total input size
    int64_t in_features_per_rank_;   // Local input size
    int64_t out_features_;
    bool input_is_parallel_;
    
    Tensor weight_;  // [out_features, in_features_per_rank]
    std::optional<Tensor> bias_;
};

Tensor RowParallelLinear::forward(const Tensor& input) {
    Tensor local_input;
    
    if (!input_is_parallel_) {
        // Need to scatter input across ranks
        local_input = scatter_last_dim(input);
    } else {
        local_input = input;
    }
    
    // Local matmul
    Tensor output = matmul(local_input, weight_.t());
    
    // AllReduce to sum partial results
    get_comm().all_reduce(output, ReduceOp::Sum);
    
    if (bias_) {
        output = output + *bias_;
    }
    
    return output;
}
```

## 6. Parallel Attention

### 6.1 Strategy

For multi-head attention:
- Split Q, K, V projections column-parallel (each GPU gets subset of heads)
- Compute attention locally (no communication)
- Split output projection row-parallel

```
Heads: [H0, H1, H2, H3, H4, H5, H6, H7]

GPU 0: [H0, H1]  - computes attention for these heads
GPU 1: [H2, H3]
GPU 2: [H4, H5]
GPU 3: [H6, H7]

O projection: row-parallel (AllReduce at the end)
```

```cpp
// include/vesper/distributed/parallel_attention.h

namespace vesper::distributed {

class ParallelMultiHeadAttention : public nn::Module {
public:
    ParallelMultiHeadAttention(
        int64_t dim,
        int64_t n_heads,
        int64_t n_kv_heads = 0);  // For GQA
    
    Tensor forward(
        const Tensor& x,
        int64_t start_pos = 0,
        const std::optional<Tensor>& mask = std::nullopt);
    
private:
    int64_t dim_;
    int64_t n_heads_;
    int64_t n_kv_heads_;
    int64_t n_heads_per_rank_;
    int64_t n_kv_heads_per_rank_;
    int64_t head_dim_;
    
    // Column-parallel projections (split heads)
    std::shared_ptr<ColumnParallelLinear> wq_;
    std::shared_ptr<ColumnParallelLinear> wk_;
    std::shared_ptr<ColumnParallelLinear> wv_;
    
    // Row-parallel output projection
    std::shared_ptr<RowParallelLinear> wo_;
    
    // Local KV cache (only for this rank's heads)
    Tensor k_cache_;
    Tensor v_cache_;
};

} // namespace vesper::distributed
```

```cpp
// src/distributed/parallel_attention.cpp

ParallelMultiHeadAttention::ParallelMultiHeadAttention(
    int64_t dim, int64_t n_heads, int64_t n_kv_heads)
    : dim_(dim), n_heads_(n_heads), 
      n_kv_heads_(n_kv_heads > 0 ? n_kv_heads : n_heads)
{
    auto& comm = get_comm();
    int world_size = comm.world_size();
    
    VESPER_CHECK(n_heads_ % world_size == 0,
        "n_heads must be divisible by world_size");
    VESPER_CHECK(n_kv_heads_ % world_size == 0,
        "n_kv_heads must be divisible by world_size");
    
    n_heads_per_rank_ = n_heads_ / world_size;
    n_kv_heads_per_rank_ = n_kv_heads_ / world_size;
    head_dim_ = dim_ / n_heads_;
    
    // Q: [dim] -> [n_heads_per_rank * head_dim]
    wq_ = register_module("wq", std::make_shared<ColumnParallelLinear>(
        dim_, n_heads_ * head_dim_, /*bias=*/false, /*gather=*/false));
    
    // K, V: [dim] -> [n_kv_heads_per_rank * head_dim]
    wk_ = register_module("wk", std::make_shared<ColumnParallelLinear>(
        dim_, n_kv_heads_ * head_dim_, /*bias=*/false, /*gather=*/false));
    wv_ = register_module("wv", std::make_shared<ColumnParallelLinear>(
        dim_, n_kv_heads_ * head_dim_, /*bias=*/false, /*gather=*/false));
    
    // O: [n_heads_per_rank * head_dim] -> [dim] with AllReduce
    wo_ = register_module("wo", std::make_shared<RowParallelLinear>(
        n_heads_ * head_dim_, dim_, /*bias=*/false, /*input_parallel=*/true));
}

Tensor ParallelMultiHeadAttention::forward(
    const Tensor& x,
    int64_t start_pos,
    const std::optional<Tensor>& mask)
{
    int64_t batch = x.size(0);
    int64_t seq_len = x.size(1);
    
    // Project to Q, K, V (column-parallel, no gather)
    Tensor q = wq_->forward(x);  // [batch, seq, n_heads_per_rank * head_dim]
    Tensor k = wk_->forward(x);
    Tensor v = wv_->forward(x);
    
    // Reshape for attention
    q = q.view({batch, seq_len, n_heads_per_rank_, head_dim_}).transpose(1, 2);
    k = k.view({batch, seq_len, n_kv_heads_per_rank_, head_dim_}).transpose(1, 2);
    v = v.view({batch, seq_len, n_kv_heads_per_rank_, head_dim_}).transpose(1, 2);
    
    // Apply RoPE (position dependent)
    // ...
    
    // Update KV cache
    // ...
    
    // Attention (local, no communication needed)
    Tensor attn_output = scaled_dot_product_attention(q, k, v, mask);
    
    // Reshape back
    attn_output = attn_output.transpose(1, 2).contiguous()
        .view({batch, seq_len, n_heads_per_rank_ * head_dim_});
    
    // Output projection (row-parallel, includes AllReduce)
    return wo_->forward(attn_output);
}
```

## 7. Parallel FFN

For SwiGLU FFN:
```
gate = x @ W_gate  (column-parallel)
up = x @ W_up      (column-parallel)
hidden = silu(gate) * up  (local)
out = hidden @ W_down  (row-parallel, AllReduce)
```

```cpp
class ParallelSwiGLU : public nn::Module {
public:
    ParallelSwiGLU(int64_t dim, int64_t hidden_dim) {
        // Gate and up: column-parallel (no gather)
        w_gate_ = register_module("w_gate", 
            std::make_shared<ColumnParallelLinear>(
                dim, hidden_dim, false, false));
        w_up_ = register_module("w_up",
            std::make_shared<ColumnParallelLinear>(
                dim, hidden_dim, false, false));
        
        // Down: row-parallel (AllReduce)
        w_down_ = register_module("w_down",
            std::make_shared<RowParallelLinear>(
                hidden_dim, dim, false, true));
    }
    
    Tensor forward(const Tensor& x) override {
        Tensor gate = silu(w_gate_->forward(x));
        Tensor up = w_up_->forward(x);
        Tensor hidden = gate * up;
        return w_down_->forward(hidden);
    }
    
private:
    std::shared_ptr<ColumnParallelLinear> w_gate_;
    std::shared_ptr<ColumnParallelLinear> w_up_;
    std::shared_ptr<RowParallelLinear> w_down_;
};
```

## 8. Complete Parallel Transformer

```cpp
// include/vesper/distributed/parallel_transformer.h

namespace vesper::distributed {

class ParallelTransformerBlock : public nn::Module {
public:
    ParallelTransformerBlock(const TransformerConfig& config);
    
    Tensor forward(
        const Tensor& x,
        int64_t start_pos = 0,
        const std::optional<Tensor>& mask = std::nullopt) override;
    
private:
    std::shared_ptr<nn::RMSNorm> attn_norm_;
    std::shared_ptr<ParallelMultiHeadAttention> attention_;
    std::shared_ptr<nn::RMSNorm> ffn_norm_;
    std::shared_ptr<ParallelSwiGLU> ffn_;
};

class ParallelTransformer : public nn::Module {
public:
    ParallelTransformer(const TransformerConfig& config);
    
    Tensor forward(const Tensor& input_ids, int64_t start_pos = 0) override;
    
    // Load weights with proper sharding
    void load_weights(const std::string& path);
    
private:
    TransformerConfig config_;
    
    // Embeddings (replicated on all GPUs)
    std::shared_ptr<nn::Embedding> tok_embeddings_;
    
    // Transformer blocks (tensor-parallel)
    std::vector<std::shared_ptr<ParallelTransformerBlock>> layers_;
    
    // Final norm (replicated)
    std::shared_ptr<nn::RMSNorm> norm_;
    
    // LM head (column-parallel or replicated)
    std::shared_ptr<ColumnParallelLinear> lm_head_;
};

} // namespace vesper::distributed
```

```cpp
// src/distributed/parallel_transformer.cpp

Tensor ParallelTransformer::forward(const Tensor& input_ids, int64_t start_pos) {
    // Embeddings (replicated, same on all GPUs)
    Tensor h = tok_embeddings_->forward(input_ids);
    
    // Transformer blocks (tensor-parallel)
    for (auto& layer : layers_) {
        h = layer->forward(h, start_pos);
    }
    
    // Final norm (replicated)
    h = norm_->forward(h);
    
    // LM head (column-parallel)
    // If using column-parallel, output is split across GPUs
    Tensor logits = lm_head_->forward(h);
    
    // AllGather to get full vocabulary on all GPUs
    // (needed for sampling)
    get_comm().all_gather(logits);
    
    return logits;
}
```

## 9. Weight Loading for Tensor Parallelism

### 9.1 Sharding Strategy

When loading from a single checkpoint:

```cpp
void ParallelTransformer::load_weights(const std::string& path) {
    auto& comm = get_comm();
    int rank = comm.rank();
    int world_size = comm.world_size();
    
    // Load full weights (or memory-map)
    auto full_weights = io::load_sharded_safetensors(path);
    
    for (auto& [name, param] : named_parameters()) {
        Tensor full_tensor = full_weights.at(map_weight_name(name));
        
        if (is_column_parallel(name)) {
            // Shard along first dimension (output features)
            int64_t chunk_size = full_tensor.size(0) / world_size;
            param.copy_(full_tensor.narrow(0, rank * chunk_size, chunk_size));
        } else if (is_row_parallel(name)) {
            // Shard along last dimension (input features)
            int64_t chunk_size = full_tensor.size(-1) / world_size;
            param.copy_(full_tensor.narrow(-1, rank * chunk_size, chunk_size));
        } else {
            // Replicated (embeddings, norms)
            param.copy_(full_tensor);
        }
    }
}
```

### 9.2 Pre-Sharded Checkpoints

For efficiency, save already-sharded checkpoints:

```cpp
void save_sharded_checkpoint(
    const ParallelTransformer& model,
    const std::string& path) 
{
    auto& comm = get_comm();
    int rank = comm.rank();
    
    std::string rank_path = path + "/rank_" + std::to_string(rank) + ".safetensors";
    
    io::SafetensorsWriter writer(rank_path);
    
    for (const auto& [name, param] : model.named_parameters()) {
        writer.add_tensor(name, param);
    }
    
    writer.write();
}
```

## 10. Testing Strategy

### 10.1 Unit Tests (Single-GPU Simulation)

```cpp
// tests/distributed/test_parallel_linear.cpp

TEST(ColumnParallel, ShapesCorrect) {
    // Simulate 4-GPU setup on single GPU
    init_distributed(/*world_size=*/4, /*rank=*/0);
    
    ColumnParallelLinear layer(512, 1024, /*bias=*/true, /*gather=*/false);
    layer.to(Device::HIP);
    
    Tensor input = randn({2, 32, 512}, DType::Float32, Device::HIP);
    Tensor output = layer.forward(input);
    
    // Output should be 1024/4 = 256 per rank
    EXPECT_EQ(output.shape(), std::vector<int64_t>({2, 32, 256}));
}

TEST(RowParallel, ShapesCorrect) {
    init_distributed(4, 0);
    
    RowParallelLinear layer(1024, 512, /*bias=*/true, /*input_parallel=*/true);
    layer.to(Device::HIP);
    
    // Input is already partitioned (1024/4 = 256)
    Tensor input = randn({2, 32, 256}, DType::Float32, Device::HIP);
    Tensor output = layer.forward(input);
    
    // Output is full size (after AllReduce)
    EXPECT_EQ(output.shape(), std::vector<int64_t>({2, 32, 512}));
}
```

### 10.2 Multi-GPU Integration Tests

```cpp
// Run with: mpirun -n 4 ./test_tensor_parallel

TEST(TensorParallel, EndToEnd) {
    // Initialize from MPI
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    init_distributed(world_size, rank);
    
    // Each rank uses different GPU
    set_device(rank);
    
    // Create parallel model
    TransformerConfig config = TransformerConfig::llama2_7b();
    ParallelTransformer model(config);
    model.to(Device::HIP);
    
    // All ranks process same input
    Tensor input = tensor({{1, 2, 3, 4, 5}}, DType::Int64).to(Device::HIP);
    
    // Forward
    Tensor output = model.forward(input);
    
    // Output should be same on all ranks (AllGather at end)
    // Verify by comparing across ranks
    Tensor all_outputs = empty({world_size, output.numel()});
    get_comm().all_gather(all_outputs.view({world_size, -1}), 
                          output.view({1, -1}));
    
    // All should be identical
    for (int r = 1; r < world_size; ++r) {
        EXPECT_TRUE(allclose(all_outputs[0], all_outputs[r]));
    }
}

TEST(TensorParallel, WeightLoading) {
    int rank = get_rank();
    int world_size = get_world_size();
    
    // Create model
    auto config = TransformerConfig::llama2_7b();
    ParallelTransformer model(config);
    
    // Load weights
    model.load_weights("test_models/llama-7b");
    
    // Verify each rank got different weight slices
    // (by checking weight values are different)
    // ... verification code ...
}
```

### 10.3 Stress Tests

```cpp
TEST(TensorParallel, StressTest_LargeModel) {
    init_from_mpi();
    
    // 70B model requires 8 GPUs
    GTEST_SKIP_IF(get_world_size() < 8);
    
    auto config = TransformerConfig::llama2_70b();
    ParallelTransformer model(config);
    model.to(Device::HIP);
    
    // Generate
    Tensor input = randint(0, config.vocab_size, {1, 128}, DType::Int64);
    
    auto start = std::chrono::high_resolution_clock::now();
    Tensor output = model.forward(input);
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Rank " << get_rank() << ": forward time = " << ms << " ms" << std::endl;
}

TEST(TensorParallel, StressTest_CommunicationOverhead) {
    init_from_mpi();
    
    // Measure AllReduce latency
    Tensor tensor = randn({4096, 4096}, DType::Float16, Device::HIP);
    
    // Warmup
    get_comm().all_reduce(tensor);
    hipDeviceSynchronize();
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        get_comm().all_reduce(tensor);
    }
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    double per_allreduce = ms / 100;
    
    std::cout << "AllReduce 64MB: " << per_allreduce << " ms" << std::endl;
    
    // Should be fast with NVLink/InfiniBand
    EXPECT_LT(per_allreduce, 10.0);  // < 10ms for 64MB
}

TEST(TensorParallel, StressTest_ScalingEfficiency) {
    init_from_mpi();
    
    auto config = TransformerConfig::llama2_7b();
    ParallelTransformer model(config);
    model.to(Device::HIP);
    
    Tensor input = randint(0, config.vocab_size, {1, 512}, DType::Int64);
    
    // Warmup
    for (int i = 0; i < 5; ++i) {
        model.forward(input);
    }
    hipDeviceSynchronize();
    get_comm().barrier();
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 50; ++i) {
        model.forward(input);
    }
    hipDeviceSynchronize();
    get_comm().barrier();
    auto end = std::chrono::high_resolution_clock::now();
    
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    double per_forward = ms / 50;
    
    // Calculate efficiency
    // Single GPU baseline would be ~4x slower on 4 GPUs
    // Good scaling: 3.5x+ speedup on 4 GPUs
    
    if (get_rank() == 0) {
        std::cout << "Forward time per iteration: " << per_forward << " ms" << std::endl;
        std::cout << "World size: " << get_world_size() << std::endl;
    }
}
```

## 11. Communication Optimization

### 11.1 Overlapping Communication with Computation

```cpp
// Pipeline AllReduce with next layer's computation
Tensor ParallelTransformerBlock::forward_pipelined(
    const Tensor& x,
    const Tensor& pending_allreduce) 
{
    // Start AllReduce for previous layer's output
    auto& comm = get_comm();
    auto allreduce_handle = comm.all_reduce_async(pending_allreduce);
    
    // Compute current layer (overlapped with AllReduce)
    Tensor h = attn_norm_->forward(x);
    h = attention_->forward_no_final_reduce(h);
    
    // Wait for previous AllReduce
    allreduce_handle.wait();
    
    // Continue...
}
```

### 11.2 Fused AllReduce

```cpp
// Fuse multiple small AllReduces into one
void fused_all_reduce(std::vector<Tensor>& tensors) {
    // Flatten all tensors into one buffer
    size_t total_size = 0;
    for (const auto& t : tensors) total_size += t.numel();
    
    Tensor buffer = empty({total_size}, tensors[0].dtype(), tensors[0].device());
    
    // Copy to buffer
    size_t offset = 0;
    for (const auto& t : tensors) {
        buffer.narrow(0, offset, t.numel()).copy_(t.view({-1}));
        offset += t.numel();
    }
    
    // Single AllReduce
    get_comm().all_reduce(buffer);
    
    // Copy back
    offset = 0;
    for (auto& t : tensors) {
        t.view({-1}).copy_(buffer.narrow(0, offset, t.numel()));
        offset += t.numel();
    }
}
```

## 12. Summary

This chapter covered:

1. **Tensor parallelism concept**: Split layers across GPUs
2. **Column-parallel Linear**: Split output features, AllGather
3. **Row-parallel Linear**: Split input features, AllReduce
4. **Parallel attention**: Distribute heads, local attention computation
5. **Communication primitives**: NCCL/RCCL AllReduce, AllGather
6. **Weight loading**: Sharding strategies for distributed models

Key metrics:
- **Scaling efficiency**: Actual speedup / theoretical speedup
- **Communication overhead**: Time spent in collective operations
- **Memory per GPU**: Should decrease linearly with more GPUs

With tensor parallelism, Vesper can run 70B+ parameter models across multiple GPUs.
```
