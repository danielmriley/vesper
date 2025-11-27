#include <vesper/generation/generator.h>
#include <vesper/nn/functional.h>
#include <vesper/ops/elementwise.h>
#include <vesper/ops/cat.h>
#include <vesper/ops/sampling_ops.h>
#include <vesper/core/factories.h>
#include <vesper/core/slice.h>
#include <vesper/core/macros.h>

#include <algorithm>
#include <chrono>

namespace vesper::generation {

Generator::Generator(models::TransformerLM* model) : model_(model) {
    VESPER_CHECK(model != nullptr, "Generator requires a non-null model");
}

bool Generator::should_stop(int64_t token_id, const SamplingParams& params,
                            int64_t generated_count) const {
    // Check minimum tokens constraint
    if (generated_count < params.min_new_tokens) {
        return false;
    }
    
    // Check stop tokens
    for (int64_t stop_id : params.stop_token_ids) {
        if (token_id == stop_id) {
            return true;
        }
    }
    
    return false;
}

Tensor Generator::generate(const Tensor& prompt_ids, const SamplingParams& params) {
    VESPER_CHECK(prompt_ids.ndim() == 2, "prompt_ids must be 2D [Batch, SeqLen]");
    
    int64_t batch_size = prompt_ids.shape()[0];
    int64_t prompt_len = prompt_ids.shape()[1];
    Device device = prompt_ids.device();
    
    // Validate sequence length
    VESPER_CHECK(prompt_len + params.max_new_tokens <= model_->config().max_seq_len,
                 "Total sequence length would exceed model's max_seq_len");
    
    // Set model to eval mode
    bool was_training = model_->is_training();
    model_->eval();
    
    // Initialize KV cache
    model_->init_cache(batch_size, device);
    
    // Track all generated tokens - keep on GPU
    std::vector<Tensor> all_tokens;
    all_tokens.push_back(prompt_ids);
    
    // Track which sequences are done - use GPU mask (0 = not done, 1 = done)
    Tensor done_mask = vesper::zeros({batch_size}, DType::Int32, device);
    
    // Prefill: process entire prompt
    Tensor logits = model_->forward_with_cache(prompt_ids, 0);
    
    // Get logits for last position: [Batch, SeqLen, Vocab] -> [Batch, Vocab]
    logits = logits.index({Slice(), Slice(-1, std::nullopt), Slice()});
    logits = logits.view({batch_size, model_->config().vocab_size});
    
    int64_t generated = 0;
    int64_t pos = prompt_len;
    
    // Build history tensor for repetition penalty (just the prompt initially)
    // Keep on GPU for efficient concatenation
    Tensor history = prompt_ids.clone();
    
    while (generated < params.max_new_tokens) {
        // Sample next token
        SamplingParams step_params = params;
        if (seed_ >= 0) {
            step_params.seed = seed_ + generated;  // Vary seed per step for determinism
        }
        
        Tensor next_token;
        if (params.repetition_penalty != 1.0f) {
            next_token = sample_next_token(logits, step_params, &history);
        } else {
            next_token = sample_next_token(logits, step_params, nullptr);
        }
        
        // Check stopping conditions on GPU
        // This returns a mask of which sequences should stop
        Tensor stop_mask = ops::check_stop_tokens(
            next_token, 
            params.stop_token_ids,
            params.min_new_tokens,
            generated + 1
        );
        
        // Update done_mask: done = done OR stop_mask
        // For GPU: transfer just the stop mask (small - batch_size bytes)
        Tensor stop_cpu = stop_mask.to(Device::CPU);
        Tensor done_cpu = done_mask.to(Device::CPU);
        const int8_t* stop_data = stop_cpu.data_ptr<const int8_t>();
        int8_t* done_data = done_cpu.data_ptr<int8_t>();
        for (int64_t b = 0; b < batch_size; ++b) {
            if (stop_data[b]) done_data[b] = 1;
        }
        done_mask.copy_from(done_cpu);
        
        // Add to token list [Batch] -> [Batch, 1]
        Tensor next_2d = next_token.view({batch_size, 1});
        all_tokens.push_back(next_2d);
        
        // Check if all done using GPU reduction
        bool all_done = ops::all_true(done_mask);
        if (all_done) {
            break;
        }
        
        // Update history for repetition penalty using GPU cat
        if (params.repetition_penalty != 1.0f) {
            // Concatenate on GPU: history = cat([history, next_2d], dim=1)
            history = ops::cat({history, next_2d}, 1);
        }
        
        // Forward with new token
        logits = model_->forward_with_cache(next_2d, pos);
        logits = logits.view({batch_size, model_->config().vocab_size});
        
        ++generated;
        ++pos;
    }
    
    model_->clear_cache();
    
    // Restore training mode if it was set
    if (was_training) model_->train();
    
    // Concatenate all tokens using GPU cat operation
    Tensor result = ops::cat(all_tokens, 1);
    
    return result;
}

Tensor Generator::generate_streaming(const Tensor& prompt_ids,
                                      const SamplingParams& params,
                                      TokenCallback on_token) {
    VESPER_CHECK(prompt_ids.ndim() == 2, "prompt_ids must be 2D [Batch, SeqLen]");
    
    int64_t batch_size = prompt_ids.shape()[0];
    int64_t prompt_len = prompt_ids.shape()[1];
    Device device = prompt_ids.device();
    
    VESPER_CHECK(prompt_len + params.max_new_tokens <= model_->config().max_seq_len,
                 "Total sequence length would exceed model's max_seq_len");
    
    bool was_training = model_->is_training();
    model_->eval();
    
    model_->init_cache(batch_size, device);
    
    std::vector<Tensor> all_tokens;
    all_tokens.push_back(prompt_ids);
    
    // For streaming, we need CPU access for callbacks
    // But we can still keep tensors on GPU and only transfer what's needed
    std::vector<bool> done(batch_size, false);
    
    // Prefill
    Tensor logits = model_->forward_with_cache(prompt_ids, 0);
    logits = logits.index({Slice(), Slice(-1, std::nullopt), Slice()});
    logits = logits.view({batch_size, model_->config().vocab_size});
    
    int64_t generated = 0;
    int64_t pos = prompt_len;
    
    Tensor history = prompt_ids.clone();
    
    while (generated < params.max_new_tokens) {
        SamplingParams step_params = params;
        if (seed_ >= 0) {
            step_params.seed = seed_ + generated;
        }
        
        Tensor next_token;
        if (params.repetition_penalty != 1.0f) {
            next_token = sample_next_token(logits, step_params, &history);
        } else {
            next_token = sample_next_token(logits, step_params, nullptr);
        }
        
        // For streaming, we MUST transfer to CPU for the callback
        // This is inherent to streaming - the user needs the token values
        Tensor next_cpu = next_token.to(Device::CPU);
        const int32_t* next_data = next_cpu.data_ptr<int32_t>();
        
        bool all_done = true;
        for (int64_t b = 0; b < batch_size; ++b) {
            if (!done[b]) {
                int64_t token = next_data[b];
                
                // Call the callback - if it returns false, stop this sequence
                bool continue_gen = on_token(b, token);
                
                if (!continue_gen || should_stop(token, params, generated + 1)) {
                    done[b] = true;
                } else {
                    all_done = false;
                }
            }
        }
        
        Tensor next_2d = next_token.view({batch_size, 1});
        all_tokens.push_back(next_2d);
        
        if (all_done) {
            break;
        }
        
        // Update history for repetition penalty using GPU cat
        if (params.repetition_penalty != 1.0f) {
            history = ops::cat({history, next_2d}, 1);
        }
        
        logits = model_->forward_with_cache(next_2d, pos);
        logits = logits.view({batch_size, model_->config().vocab_size});
        
        ++generated;
        ++pos;
    }
    
    model_->clear_cache();
    
    if (was_training) model_->train();
    
    // Build result tensor using GPU cat
    Tensor result = ops::cat(all_tokens, 1);
    
    return result;
}

} // namespace vesper::generation
