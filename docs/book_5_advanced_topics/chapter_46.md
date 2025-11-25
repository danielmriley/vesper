```markdown
# Chapter 46: Continuous Batching for LLM Serving

## 1. Introduction

Static batching wastes compute. When serving LLMs:
- Requests arrive at different times
- Requests have different lengths
- Requests finish at different times

**Continuous batching** (also called **dynamic batching** or **iteration-level scheduling**) maximizes GPU utilization by:
- Adding new requests mid-generation
- Removing finished requests immediately
- Processing a variable number of tokens per step

This chapter covers:
1. **Request lifecycle**: From arrival to completion
2. **Scheduler design**: When to add/remove requests
3. **Memory management**: Dynamic KV cache allocation
4. **Implementation**: A production-quality inference server

## 2. The Problem with Static Batching

### 2.1 Static Batching

```
Request A: [prompt: 10 tokens, generate: 50 tokens]
Request B: [prompt: 20 tokens, generate: 10 tokens]
Request C: [prompt: 5 tokens, generate: 100 tokens]

Static batch: Wait for all, pad to max length
Time: max(60, 30, 105) = 105 decode steps
Wasted compute: Request B sits idle for 75 steps
```

### 2.2 Continuous Batching

```
Step 0:  [A_prefill, B_prefill, C_prefill]
Step 10: [A_decode, B_decode, C_decode]
Step 20: [A_decode, B_done!, C_decode, D_new!]  <- B removed, D added
Step 30: [A_decode, C_decode, D_decode]
...
```

Benefits:
- **Higher throughput**: No waiting for slowest request
- **Lower latency**: Requests start immediately
- **Better GPU utilization**: Always processing useful tokens

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Inference Server                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐     ┌──────────────┐     ┌─────────────────┐  │
│  │   Request   │────▶│  Scheduler   │────▶│  Model Engine   │  │
│  │    Queue    │     │              │     │                 │  │
│  └─────────────┘     │  ┌────────┐  │     │  ┌───────────┐  │  │
│                      │  │ Active │  │     │  │ GPU Model │  │  │
│  ┌─────────────┐     │  │ Batch  │  │     │  └───────────┘  │  │
│  │   Output    │◀────│  └────────┘  │     │  ┌───────────┐  │  │
│  │   Stream    │     │              │     │  │ KV Cache  │  │  │
│  └─────────────┘     └──────────────┘     │  │  Manager  │  │  │
│                                           │  └───────────┘  │  │
│                                           └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 4. Request and Sequence Data Structures

```cpp
// include/vesper/serve/request.h

namespace vesper::serve {

enum class RequestState {
    Pending,      // Waiting in queue
    Prefilling,   // Processing prompt
    Decoding,     // Generating tokens
    Completed,    // Finished (EOS or max length)
    Cancelled,    // User cancelled
    Error         // Error during processing
};

struct SamplingParams {
    float temperature = 1.0f;
    float top_p = 1.0f;
    int top_k = 0;
    float repetition_penalty = 1.0f;
    int max_tokens = 256;
    std::vector<int64_t> stop_token_ids = {};
};

class Sequence {
public:
    Sequence(int64_t seq_id, std::vector<int64_t> prompt_tokens,
             SamplingParams params);
    
    int64_t id() const { return seq_id_; }
    RequestState state() const { return state_; }
    
    // Token management
    const std::vector<int64_t>& tokens() const { return tokens_; }
    void append_token(int64_t token);
    int64_t num_prompt_tokens() const { return num_prompt_tokens_; }
    int64_t num_generated_tokens() const { return tokens_.size() - num_prompt_tokens_; }
    
    // Check completion conditions
    bool is_finished() const;
    
    // KV cache slot assignment
    int kv_cache_slot() const { return kv_cache_slot_; }
    void set_kv_cache_slot(int slot) { kv_cache_slot_ = slot; }
    
    // Sampling params
    const SamplingParams& sampling_params() const { return params_; }
    
private:
    int64_t seq_id_;
    std::vector<int64_t> tokens_;
    int64_t num_prompt_tokens_;
    RequestState state_;
    SamplingParams params_;
    int kv_cache_slot_ = -1;
};

class Request {
public:
    Request(int64_t request_id, std::string prompt,
            SamplingParams params = {});
    
    int64_t id() const { return request_id_; }
    const std::string& prompt() const { return prompt_; }
    const SamplingParams& sampling_params() const { return params_; }
    
    // Tokenized prompt (set after tokenization)
    void set_prompt_tokens(std::vector<int64_t> tokens);
    const std::vector<int64_t>& prompt_tokens() const { return prompt_tokens_; }
    
    // Output streaming callback
    using StreamCallback = std::function<void(const std::string& token)>;
    void set_stream_callback(StreamCallback cb) { stream_callback_ = cb; }
    void stream_token(const std::string& token);
    
    // Result
    std::string output() const { return output_; }
    void append_output(const std::string& text);
    
    // Timing
    std::chrono::time_point<std::chrono::steady_clock> arrival_time() const;
    std::chrono::time_point<std::chrono::steady_clock> first_token_time() const;
    
private:
    int64_t request_id_;
    std::string prompt_;
    std::vector<int64_t> prompt_tokens_;
    SamplingParams params_;
    StreamCallback stream_callback_;
    std::string output_;
    std::chrono::time_point<std::chrono::steady_clock> arrival_time_;
    std::chrono::time_point<std::chrono::steady_clock> first_token_time_;
};

} // namespace vesper::serve
```

## 5. KV Cache Manager

Efficiently allocate and reclaim KV cache memory:

```cpp
// include/vesper/serve/kv_cache_manager.h

namespace vesper::serve {

struct KVCacheConfig {
    int64_t num_layers;
    int64_t num_heads;
    int64_t head_dim;
    int64_t max_seq_len;
    int64_t max_num_sequences;  // Max concurrent sequences
    DType dtype = DType::Float16;
    Device device = Device::HIP;
};

class KVCacheManager {
public:
    explicit KVCacheManager(const KVCacheConfig& config);
    
    // Allocate a slot for a new sequence
    // Returns slot ID, or -1 if no slots available
    int allocate_slot();
    
    // Free a slot when sequence completes
    void free_slot(int slot);
    
    // Get cache tensors for a slot
    // Returns (key_cache, value_cache) for all layers
    std::pair<Tensor, Tensor> get_cache(int slot, int layer);
    
    // Get all caches for batch processing
    // Shape: [max_num_sequences, num_layers, max_seq_len, num_heads, head_dim]
    const Tensor& key_cache() const { return key_cache_; }
    const Tensor& value_cache() const { return value_cache_; }
    
    // Stats
    int num_free_slots() const;
    int num_used_slots() const;
    size_t memory_bytes() const;
    
private:
    KVCacheConfig config_;
    
    // Preallocated cache
    // Shape: [max_num_sequences, num_layers, max_seq_len, num_heads, head_dim]
    Tensor key_cache_;
    Tensor value_cache_;
    
    // Slot management
    std::vector<bool> slot_used_;
    std::mutex mutex_;
};

} // namespace vesper::serve
```

```cpp
// src/serve/kv_cache_manager.cpp

KVCacheManager::KVCacheManager(const KVCacheConfig& config) : config_(config) {
    // Preallocate all cache memory
    std::vector<int64_t> shape = {
        config.max_num_sequences,
        config.num_layers,
        config.max_seq_len,
        config.num_heads,
        config.head_dim
    };
    
    key_cache_ = zeros(shape, config.dtype, config.device);
    value_cache_ = zeros(shape, config.dtype, config.device);
    
    slot_used_.resize(config.max_num_sequences, false);
    
    size_t bytes = 2 * key_cache_.numel() * dtype_size(config.dtype);
    std::cout << "KV Cache allocated: " << (bytes / 1e9) << " GB" << std::endl;
}

int KVCacheManager::allocate_slot() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (size_t i = 0; i < slot_used_.size(); ++i) {
        if (!slot_used_[i]) {
            slot_used_[i] = true;
            return static_cast<int>(i);
        }
    }
    
    return -1;  // No free slots
}

void KVCacheManager::free_slot(int slot) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (slot >= 0 && slot < static_cast<int>(slot_used_.size())) {
        slot_used_[slot] = false;
        
        // Optionally: zero out the cache for this slot
        // key_cache_[slot].zero_();
        // value_cache_[slot].zero_();
    }
}

std::pair<Tensor, Tensor> KVCacheManager::get_cache(int slot, int layer) {
    return {
        key_cache_.select(0, slot).select(0, layer),
        value_cache_.select(0, slot).select(0, layer)
    };
}
```

## 6. Scheduler

The scheduler decides which sequences to process each step:

```cpp
// include/vesper/serve/scheduler.h

namespace vesper::serve {

enum class SchedulerPolicy {
    FCFS,           // First-come-first-served
    ShortestFirst,  // Prioritize shorter prompts
    FairnessAware   // Balance between requests
};

struct SchedulerConfig {
    int max_batch_size = 64;
    int max_tokens_per_step = 4096;  // Total tokens across batch
    SchedulerPolicy policy = SchedulerPolicy::FCFS;
};

struct SchedulerOutput {
    std::vector<Sequence*> prefill_sequences;  // Need prefill
    std::vector<Sequence*> decode_sequences;   // Continue decoding
    
    // Combined for batched execution
    std::vector<int64_t> all_token_ids;
    std::vector<int> slot_ids;
    std::vector<int> seq_positions;  // Current position in each sequence
    
    bool has_work() const { 
        return !prefill_sequences.empty() || !decode_sequences.empty(); 
    }
};

class Scheduler {
public:
    Scheduler(SchedulerConfig config, KVCacheManager* cache_manager);
    
    // Add request to queue
    void add_request(std::shared_ptr<Request> request);
    
    // Schedule next batch of work
    SchedulerOutput schedule();
    
    // Handle completed tokens
    void process_outputs(const std::vector<int64_t>& next_tokens,
                        SchedulerOutput& batch);
    
    // Check for finished sequences
    std::vector<std::shared_ptr<Request>> get_finished_requests();
    
    // Stats
    size_t pending_requests() const;
    size_t active_sequences() const;
    
private:
    SchedulerConfig config_;
    KVCacheManager* cache_manager_;
    
    // Request queues
    std::deque<std::shared_ptr<Request>> pending_queue_;
    std::vector<std::unique_ptr<Sequence>> active_sequences_;
    std::vector<std::shared_ptr<Request>> finished_requests_;
    
    std::mutex mutex_;
    
    // Select sequences for next batch
    void select_prefill_candidates(SchedulerOutput& output, int& remaining_tokens);
    void select_decode_candidates(SchedulerOutput& output, int& remaining_tokens);
    
    // Build batched inputs
    void build_batch(SchedulerOutput& output);
};

} // namespace vesper::serve
```

```cpp
// src/serve/scheduler.cpp

SchedulerOutput Scheduler::schedule() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    SchedulerOutput output;
    int remaining_tokens = config_.max_tokens_per_step;
    
    // Priority 1: Continue active decode sequences (cheap, one token each)
    select_decode_candidates(output, remaining_tokens);
    
    // Priority 2: Start new prefills if capacity allows
    select_prefill_candidates(output, remaining_tokens);
    
    // Build batched inputs
    if (output.has_work()) {
        build_batch(output);
    }
    
    return output;
}

void Scheduler::select_decode_candidates(SchedulerOutput& output, int& remaining_tokens) {
    for (auto& seq : active_sequences_) {
        if (seq->state() != RequestState::Decoding) continue;
        if (remaining_tokens <= 0) break;
        
        output.decode_sequences.push_back(seq.get());
        remaining_tokens -= 1;  // Decode is 1 token per sequence
    }
}

void Scheduler::select_prefill_candidates(SchedulerOutput& output, int& remaining_tokens) {
    while (!pending_queue_.empty() && remaining_tokens > 0) {
        auto& request = pending_queue_.front();
        int prompt_len = request->prompt_tokens().size();
        
        if (prompt_len > remaining_tokens) {
            break;  // Can't fit this prompt
        }
        
        // Allocate KV cache slot
        int slot = cache_manager_->allocate_slot();
        if (slot < 0) {
            break;  // No cache slots available
        }
        
        // Create sequence
        auto seq = std::make_unique<Sequence>(
            request->id(),
            request->prompt_tokens(),
            request->sampling_params()
        );
        seq->set_kv_cache_slot(slot);
        
        output.prefill_sequences.push_back(seq.get());
        active_sequences_.push_back(std::move(seq));
        pending_queue_.pop_front();
        
        remaining_tokens -= prompt_len;
    }
}

void Scheduler::build_batch(SchedulerOutput& output) {
    output.all_token_ids.clear();
    output.slot_ids.clear();
    output.seq_positions.clear();
    
    // Add prefill tokens (full prompt for each)
    for (auto* seq : output.prefill_sequences) {
        for (int64_t token : seq->tokens()) {
            output.all_token_ids.push_back(token);
            output.slot_ids.push_back(seq->kv_cache_slot());
        }
        output.seq_positions.push_back(seq->tokens().size() - 1);
    }
    
    // Add decode tokens (just the last token for each)
    for (auto* seq : output.decode_sequences) {
        output.all_token_ids.push_back(seq->tokens().back());
        output.slot_ids.push_back(seq->kv_cache_slot());
        output.seq_positions.push_back(seq->tokens().size() - 1);
    }
}

void Scheduler::process_outputs(
    const std::vector<int64_t>& next_tokens,
    SchedulerOutput& batch) 
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t idx = 0;
    
    // Process prefill sequences
    for (auto* seq : batch.prefill_sequences) {
        seq->append_token(next_tokens[idx++]);
        // Transition from prefilling to decoding
        // (state management handled internally)
    }
    
    // Process decode sequences
    for (auto* seq : batch.decode_sequences) {
        seq->append_token(next_tokens[idx++]);
    }
    
    // Check for finished sequences
    for (auto it = active_sequences_.begin(); it != active_sequences_.end(); ) {
        if ((*it)->is_finished()) {
            cache_manager_->free_slot((*it)->kv_cache_slot());
            // Move request to finished list
            // ... handle output collection ...
            it = active_sequences_.erase(it);
        } else {
            ++it;
        }
    }
}
```

## 7. Model Engine

Runs batched forward passes with continuous batching:

```cpp
// include/vesper/serve/model_engine.h

namespace vesper::serve {

class ModelEngine {
public:
    ModelEngine(
        std::unique_ptr<models::Transformer> model,
        std::unique_ptr<Tokenizer> tokenizer,
        KVCacheManager* cache_manager);
    
    // Run one step of the batch
    // Returns next token IDs for each sequence in the batch
    std::vector<int64_t> step(const SchedulerOutput& batch);
    
    // Tokenize a prompt
    std::vector<int64_t> tokenize(const std::string& text);
    
    // Detokenize
    std::string detokenize(int64_t token_id);
    std::string detokenize(const std::vector<int64_t>& token_ids);
    
private:
    std::unique_ptr<models::Transformer> model_;
    std::unique_ptr<Tokenizer> tokenizer_;
    KVCacheManager* cache_manager_;
    
    // Batched attention with variable sequence lengths
    Tensor forward_with_cache(
        const Tensor& token_ids,        // [total_tokens]
        const Tensor& positions,        // [total_tokens]
        const Tensor& slot_ids,         // [total_tokens]
        const Tensor& seq_lens,         // [num_seqs]
        bool is_prefill);
    
    // Sample next tokens
    std::vector<int64_t> sample(
        const Tensor& logits,           // [num_seqs, vocab_size]
        const std::vector<Sequence*>& sequences);
};

} // namespace vesper::serve
```

```cpp
// src/serve/model_engine.cpp

std::vector<int64_t> ModelEngine::step(const SchedulerOutput& batch) {
    if (!batch.has_work()) {
        return {};
    }
    
    // Build input tensors
    Tensor token_ids = tensor(batch.all_token_ids, DType::Int64).to(Device::HIP);
    Tensor slot_ids = tensor(batch.slot_ids, DType::Int32).to(Device::HIP);
    
    // Build position tensor
    std::vector<int64_t> positions;
    size_t token_idx = 0;
    
    for (auto* seq : batch.prefill_sequences) {
        for (int64_t pos = 0; pos < seq->tokens().size(); ++pos) {
            positions.push_back(pos);
            token_idx++;
        }
    }
    for (auto* seq : batch.decode_sequences) {
        positions.push_back(seq->tokens().size() - 1);
        token_idx++;
    }
    
    Tensor pos_tensor = tensor(positions, DType::Int64).to(Device::HIP);
    
    // Determine sequence lengths for attention masking
    std::vector<int64_t> seq_lens;
    for (auto* seq : batch.prefill_sequences) {
        seq_lens.push_back(seq->tokens().size());
    }
    for (auto* seq : batch.decode_sequences) {
        seq_lens.push_back(seq->tokens().size());
    }
    Tensor seq_lens_tensor = tensor(seq_lens, DType::Int64).to(Device::HIP);
    
    // Forward pass
    bool is_prefill = !batch.prefill_sequences.empty();
    Tensor logits = forward_with_cache(
        token_ids, pos_tensor, slot_ids, seq_lens_tensor, is_prefill);
    
    // Sample next tokens
    std::vector<Sequence*> all_seqs;
    all_seqs.insert(all_seqs.end(), 
                    batch.prefill_sequences.begin(), 
                    batch.prefill_sequences.end());
    all_seqs.insert(all_seqs.end(), 
                    batch.decode_sequences.begin(), 
                    batch.decode_sequences.end());
    
    return sample(logits, all_seqs);
}

std::vector<int64_t> ModelEngine::sample(
    const Tensor& logits,
    const std::vector<Sequence*>& sequences) 
{
    std::vector<int64_t> next_tokens;
    
    for (size_t i = 0; i < sequences.size(); ++i) {
        Tensor seq_logits = logits[i];  // [vocab_size]
        const auto& params = sequences[i]->sampling_params();
        
        // Apply temperature
        if (params.temperature != 1.0f) {
            seq_logits = seq_logits / params.temperature;
        }
        
        // Apply top-k
        if (params.top_k > 0) {
            seq_logits = apply_top_k(seq_logits, params.top_k);
        }
        
        // Apply top-p
        if (params.top_p < 1.0f) {
            seq_logits = apply_top_p(seq_logits, params.top_p);
        }
        
        // Sample
        Tensor probs = softmax(seq_logits, -1);
        int64_t next_token = multinomial(probs, 1).item<int64_t>();
        
        next_tokens.push_back(next_token);
    }
    
    return next_tokens;
}
```

## 8. Complete Inference Server

```cpp
// include/vesper/serve/inference_server.h

namespace vesper::serve {

struct ServerConfig {
    std::string model_path;
    int max_batch_size = 64;
    int max_seq_len = 2048;
    int max_concurrent_requests = 256;
    int port = 8080;
    Device device = Device::HIP;
};

class InferenceServer {
public:
    InferenceServer(const ServerConfig& config);
    
    // Start the server (blocking)
    void run();
    
    // Stop the server
    void stop();
    
    // Add a request (returns immediately)
    std::future<std::string> generate(
        const std::string& prompt,
        SamplingParams params = {});
    
    // Streaming generation
    void generate_stream(
        const std::string& prompt,
        SamplingParams params,
        std::function<void(const std::string&)> callback);
    
private:
    ServerConfig config_;
    
    std::unique_ptr<ModelEngine> engine_;
    std::unique_ptr<KVCacheManager> cache_manager_;
    std::unique_ptr<Scheduler> scheduler_;
    
    std::atomic<bool> running_{false};
    std::thread inference_thread_;
    
    // Main inference loop
    void inference_loop();
};

} // namespace vesper::serve
```

```cpp
// src/serve/inference_server.cpp

InferenceServer::InferenceServer(const ServerConfig& config) : config_(config) {
    // Load model
    auto model_config = io::ModelLoader::load_config(config.model_path);
    auto model = std::make_unique<models::Transformer>(model_config);
    model->to(config.device);
    
    io::LoadConfig load_config;
    load_config.model_path = config.model_path;
    load_config.device = config.device;
    io::ModelLoader::load(*model, load_config);
    
    // Initialize tokenizer
    auto tokenizer = std::make_unique<Tokenizer>(config.model_path + "/tokenizer.json");
    
    // Initialize KV cache
    KVCacheConfig cache_config;
    cache_config.num_layers = model_config.n_layers;
    cache_config.num_heads = model_config.n_kv_heads;
    cache_config.head_dim = model_config.dim / model_config.n_heads;
    cache_config.max_seq_len = config.max_seq_len;
    cache_config.max_num_sequences = config.max_concurrent_requests;
    cache_config.device = config.device;
    
    cache_manager_ = std::make_unique<KVCacheManager>(cache_config);
    
    // Initialize scheduler
    SchedulerConfig sched_config;
    sched_config.max_batch_size = config.max_batch_size;
    
    scheduler_ = std::make_unique<Scheduler>(sched_config, cache_manager_.get());
    
    // Initialize engine
    engine_ = std::make_unique<ModelEngine>(
        std::move(model), std::move(tokenizer), cache_manager_.get());
}

void InferenceServer::run() {
    running_ = true;
    inference_thread_ = std::thread(&InferenceServer::inference_loop, this);
    
    // HTTP server setup would go here
    // ...
    
    inference_thread_.join();
}

void InferenceServer::inference_loop() {
    while (running_) {
        // Schedule next batch
        auto batch = scheduler_->schedule();
        
        if (!batch.has_work()) {
            // No work, sleep briefly
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            continue;
        }
        
        // Run model
        auto next_tokens = engine_->step(batch);
        
        // Process outputs
        scheduler_->process_outputs(next_tokens, batch);
        
        // Stream tokens to clients
        size_t idx = 0;
        for (auto* seq : batch.prefill_sequences) {
            // First token for this sequence
            // Stream callback handled in Request
            idx++;
        }
        for (auto* seq : batch.decode_sequences) {
            // Continuation token
            idx++;
        }
        
        // Handle finished requests
        for (auto& req : scheduler_->get_finished_requests()) {
            // Complete the future/callback
        }
    }
}

std::future<std::string> InferenceServer::generate(
    const std::string& prompt,
    SamplingParams params) 
{
    auto request = std::make_shared<Request>(next_id_++, prompt, params);
    
    // Tokenize
    request->set_prompt_tokens(engine_->tokenize(prompt));
    
    // Create promise/future
    auto promise = std::make_shared<std::promise<std::string>>();
    auto future = promise->get_future();
    
    request->set_completion_callback([promise](const std::string& output) {
        promise->set_value(output);
    });
    
    // Add to scheduler
    scheduler_->add_request(request);
    
    return future;
}
```

## 9. Paged Attention (Optional Advanced Topic)

For even better memory efficiency, use **paged attention** (vLLM-style):

```cpp
// Block-based KV cache allocation
struct KVBlock {
    static constexpr int BLOCK_SIZE = 16;  // Tokens per block
    Tensor key_block;   // [num_heads, BLOCK_SIZE, head_dim]
    Tensor value_block; // [num_heads, BLOCK_SIZE, head_dim]
};

class PagedKVCacheManager {
public:
    // Allocate blocks on demand
    std::vector<int> allocate_blocks(int num_tokens);
    
    // Free blocks when sequence completes
    void free_blocks(const std::vector<int>& block_ids);
    
    // Copy-on-write for beam search
    std::vector<int> fork_blocks(const std::vector<int>& block_ids);
    
private:
    std::vector<KVBlock> block_pool_;
    std::queue<int> free_block_ids_;
};
```

## 10. Testing Strategy

### 10.1 Unit Tests

```cpp
// tests/serve/test_scheduler.cpp

TEST(Scheduler, BasicScheduling) {
    KVCacheConfig cache_config;
    cache_config.num_layers = 2;
    cache_config.num_heads = 8;
    cache_config.head_dim = 64;
    cache_config.max_seq_len = 128;
    cache_config.max_num_sequences = 16;
    cache_config.device = Device::CPU;
    
    KVCacheManager cache(cache_config);
    
    SchedulerConfig sched_config;
    sched_config.max_batch_size = 8;
    sched_config.max_tokens_per_step = 256;
    
    Scheduler scheduler(sched_config, &cache);
    
    // Add requests
    for (int i = 0; i < 5; ++i) {
        auto req = std::make_shared<Request>(i, "test prompt");
        req->set_prompt_tokens({1, 2, 3, 4, 5});  // 5 tokens
        scheduler.add_request(req);
    }
    
    // Schedule first batch
    auto batch = scheduler.schedule();
    
    EXPECT_EQ(batch.prefill_sequences.size(), 5);
    EXPECT_EQ(batch.decode_sequences.size(), 0);
    EXPECT_EQ(batch.all_token_ids.size(), 25);  // 5 seqs * 5 tokens
}

TEST(Scheduler, ContinuousBatching) {
    // ... setup ...
    
    // Add initial request
    auto req1 = std::make_shared<Request>(1, "first");
    req1->set_prompt_tokens({1, 2, 3});
    scheduler.add_request(req1);
    
    // First step: prefill
    auto batch1 = scheduler.schedule();
    EXPECT_EQ(batch1.prefill_sequences.size(), 1);
    
    // Simulate token generation
    scheduler.process_outputs({100}, batch1);
    
    // Add second request mid-generation
    auto req2 = std::make_shared<Request>(2, "second");
    req2->set_prompt_tokens({4, 5, 6, 7});
    scheduler.add_request(req2);
    
    // Second step: decode first, prefill second
    auto batch2 = scheduler.schedule();
    EXPECT_EQ(batch2.decode_sequences.size(), 1);  // First still generating
    EXPECT_EQ(batch2.prefill_sequences.size(), 1);  // Second starts
}

TEST(KVCache, SlotAllocation) {
    KVCacheConfig config;
    config.max_num_sequences = 4;
    // ... other config ...
    
    KVCacheManager cache(config);
    
    // Allocate all slots
    std::vector<int> slots;
    for (int i = 0; i < 4; ++i) {
        int slot = cache.allocate_slot();
        EXPECT_GE(slot, 0);
        slots.push_back(slot);
    }
    
    // Should be full
    EXPECT_EQ(cache.allocate_slot(), -1);
    
    // Free one
    cache.free_slot(slots[1]);
    
    // Should be able to allocate again
    int new_slot = cache.allocate_slot();
    EXPECT_EQ(new_slot, slots[1]);
}
```

### 10.2 Integration Tests

```cpp
TEST(InferenceServer, EndToEnd) {
    ServerConfig config;
    config.model_path = "test_models/tiny-llama";
    config.max_batch_size = 4;
    config.device = Device::HIP;
    
    InferenceServer server(config);
    
    // Run in background thread
    std::thread server_thread([&]() { server.run(); });
    
    // Wait for startup
    std::this_thread::sleep_for(std::chrono::seconds(1));
    
    // Submit requests
    auto future1 = server.generate("Hello, world!");
    auto future2 = server.generate("What is 2+2?");
    
    // Wait for results
    std::string result1 = future1.get();
    std::string result2 = future2.get();
    
    EXPECT_FALSE(result1.empty());
    EXPECT_FALSE(result2.empty());
    
    server.stop();
    server_thread.join();
}
```

### 10.3 Stress Tests

```cpp
TEST(Server, StressTest_HighConcurrency) {
    ServerConfig config;
    config.model_path = "test_models/llama-7b";
    config.max_batch_size = 64;
    config.max_concurrent_requests = 256;
    
    InferenceServer server(config);
    std::thread server_thread([&]() { server.run(); });
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // Submit many concurrent requests
    std::vector<std::future<std::string>> futures;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100; ++i) {
        SamplingParams params;
        params.max_tokens = 64;
        futures.push_back(server.generate("Write a haiku about " + std::to_string(i), params));
    }
    
    // Wait for all
    int completed = 0;
    for (auto& f : futures) {
        f.wait();
        if (!f.get().empty()) completed++;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();
    
    std::cout << "Completed: " << completed << "/100" << std::endl;
    std::cout << "Time: " << seconds << "s" << std::endl;
    std::cout << "Throughput: " << (100 / seconds) << " req/s" << std::endl;
    
    EXPECT_EQ(completed, 100);
    
    server.stop();
    server_thread.join();
}

TEST(Server, StressTest_MemoryStability) {
    ServerConfig config;
    config.max_concurrent_requests = 32;
    
    InferenceServer server(config);
    std::thread server_thread([&]() { server.run(); });
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // Record baseline memory
    size_t baseline_memory = get_gpu_memory_used();
    
    // Run many requests
    for (int round = 0; round < 10; ++round) {
        std::vector<std::future<std::string>> futures;
        
        for (int i = 0; i < 32; ++i) {
            futures.push_back(server.generate("Test " + std::to_string(i)));
        }
        
        for (auto& f : futures) {
            f.get();
        }
        
        // Check memory
        size_t current_memory = get_gpu_memory_used();
        std::cout << "Round " << round << " memory: " 
                  << (current_memory / 1e9) << " GB" << std::endl;
        
        // Should not grow significantly
        EXPECT_LT(current_memory, baseline_memory * 1.2);
    }
    
    server.stop();
    server_thread.join();
}

TEST(Server, StressTest_Latency) {
    // Measure time-to-first-token (TTFT) and inter-token latency (ITL)
    
    ServerConfig config;
    InferenceServer server(config);
    std::thread server_thread([&]() { server.run(); });
    
    std::vector<double> ttft_ms;
    std::vector<double> itl_ms;
    
    for (int i = 0; i < 20; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        std::chrono::time_point<std::chrono::high_resolution_clock> first_token_time;
        int token_count = 0;
        
        SamplingParams params;
        params.max_tokens = 32;
        
        server.generate_stream("Hello", params, [&](const std::string& token) {
            if (token_count == 0) {
                first_token_time = std::chrono::high_resolution_clock::now();
            }
            token_count++;
        });
        
        // Record TTFT
        double ttft = std::chrono::duration<double, std::milli>(
            first_token_time - start).count();
        ttft_ms.push_back(ttft);
    }
    
    // Calculate statistics
    double avg_ttft = std::accumulate(ttft_ms.begin(), ttft_ms.end(), 0.0) / ttft_ms.size();
    
    std::cout << "Average TTFT: " << avg_ttft << " ms" << std::endl;
    
    // TTFT should be reasonable (< 500ms for small models)
    EXPECT_LT(avg_ttft, 500.0);
    
    server.stop();
    server_thread.join();
}
```

## 11. Summary

This chapter covered:

1. **Continuous batching concept**: Dynamic request management for higher throughput
2. **Request/Sequence lifecycle**: From arrival to completion
3. **KV Cache management**: Pre-allocated slots with dynamic allocation
4. **Scheduler design**: Balancing prefill and decode workloads
5. **Model engine**: Batched forward passes with variable lengths
6. **Complete inference server**: Production-ready architecture

Key metrics to optimize:
- **Throughput**: Requests per second
- **Time-to-first-token (TTFT)**: Latency until first output
- **Inter-token latency (ITL)**: Time between tokens
- **GPU utilization**: Keep the GPU busy

With continuous batching, Vesper can efficiently serve LLMs at scale.
```
