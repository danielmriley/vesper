#include <vesper/generation/beam_search.h>
#include <vesper/nn/functional.h>
#include <vesper/core/factories.h>
#include <vesper/core/slice.h>
#include <vesper/core/macros.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>

namespace vesper::generation {

// =============================================================================
// GPU UPGRADE PATH NOTE (See docs/book_5_advanced_topics/chapter_50.md)
// =============================================================================
// This beam search implementation uses CPU-based data structures (std::vector,
// std::priority_queue) for candidate selection. While functional, this approach
// requires CPU-GPU synchronization each step.
//
// PyTorch/HuggingFace Transformers achieves full GPU beam search by:
// 1. Using fixed-size preallocated tensors: [batch, num_beams, max_length]
// 2. Using torch.topk() for GPU-native top-k selection
// 3. Using score masking (-1e9) instead of dynamic data structures
// 4. Recovering beam/token indices via integer division: beam = idx // vocab_size
//
// To upgrade this implementation to full GPU:
// - Replace beam_tokens with preallocated Tensor [B, beams, max_len]
// - Replace beam_scores with Tensor [B, beams]
// - Replace priority_queue with ops::topk() GPU kernel
// - Replace finished tracking with boolean mask tensor
// - Use score masking for EOS/finished handling
//
// This is a significant architectural change documented in Chapter 50.
// =============================================================================

BeamSearcher::BeamSearcher(models::TransformerLM* model) : model_(model) {
    VESPER_CHECK(model != nullptr, "BeamSearcher requires a non-null model");
}

bool BeamSearcher::is_eos(int64_t token_id, const BeamSearchParams& params) const {
    for (int64_t eos : params.eos_token_ids) {
        if (token_id == eos) return true;
    }
    return false;
}

float BeamSearcher::apply_length_penalty(float score, int64_t length, float penalty) const {
    if (penalty == 1.0f) return score;
    
    // Length penalty formula from Wu et al. (Google NMT)
    // score / ((5 + length) / 6) ^ penalty
    float lp = std::pow((5.0f + length) / 6.0f, penalty);
    return score / lp;
}

Tensor BeamSearcher::search(const Tensor& prompt_ids, const BeamSearchParams& params) {
    VESPER_CHECK(prompt_ids.ndim() == 2, "prompt_ids must be 2D [Batch, SeqLen]");
    VESPER_CHECK(params.num_beams >= 1, "num_beams must be at least 1");
    
    int64_t batch_size = prompt_ids.shape()[0];
    int64_t prompt_len = prompt_ids.shape()[1];
    int64_t vocab_size = model_->config().vocab_size;
    int64_t num_beams = params.num_beams;
    Device device = prompt_ids.device();
    
    VESPER_CHECK(prompt_len + params.max_new_tokens <= model_->config().max_seq_len,
                 "Total sequence length would exceed model's max_seq_len");
    
    bool was_training = model_->is_training();
    model_->eval();
    
    // Clear finished sequences
    finished_sequences_.clear();
    finished_sequences_.resize(batch_size);
    
    // For beam search, we expand the batch dimension by num_beams
    // Initial: only first beam per batch is active
    
    // Beam scores: [Batch, NumBeams]
    std::vector<std::vector<float>> beam_scores(batch_size, std::vector<float>(num_beams, -std::numeric_limits<float>::infinity()));
    for (int64_t b = 0; b < batch_size; ++b) {
        beam_scores[b][0] = 0.0f;  // First beam starts with score 0
    }
    
    // Token sequences for each beam: [Batch][NumBeams][Tokens...]
    std::vector<std::vector<std::vector<int32_t>>> beam_tokens(batch_size);
    for (int64_t b = 0; b < batch_size; ++b) {
        beam_tokens[b].resize(num_beams);
        
        // Copy prompt to first beam
        Tensor prompt_cpu = prompt_ids.to(Device::CPU);
        const int32_t* prompt_data = prompt_cpu.data_ptr<int32_t>();
        for (int64_t s = 0; s < prompt_len; ++s) {
            beam_tokens[b][0].push_back(prompt_data[b * prompt_len + s]);
        }
        
        // Other beams start empty (will be populated on first step)
        for (int64_t beam = 1; beam < num_beams; ++beam) {
            beam_tokens[b][beam] = beam_tokens[b][0];  // Copy prompt
        }
    }
    
    // Track which beams are done
    std::vector<std::vector<bool>> beam_done(batch_size, std::vector<bool>(num_beams, false));
    
    // Expand prompt for all beams: [B, S] -> [B * NumBeams, S]
    Tensor expanded_prompt = vesper::empty({batch_size * num_beams, prompt_len}, 
                                           DType::Int32, Device::CPU);
    {
        Tensor prompt_cpu = prompt_ids.to(Device::CPU);
        const int32_t* src = prompt_cpu.data_ptr<int32_t>();
        int32_t* dst = expanded_prompt.data_ptr<int32_t>();
        
        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t beam = 0; beam < num_beams; ++beam) {
                for (int64_t s = 0; s < prompt_len; ++s) {
                    dst[(b * num_beams + beam) * prompt_len + s] = src[b * prompt_len + s];
                }
            }
        }
    }
    
    if (device != Device::CPU) {
        expanded_prompt = expanded_prompt.to(device);
    }
    
    // Initialize KV cache for expanded batch
    model_->init_cache(batch_size * num_beams, device);
    
    // Prefill: process entire prompt for all beams
    Tensor logits = model_->forward_with_cache(expanded_prompt, 0);
    
    // Get logits for last position: [B * Beams, SeqLen, Vocab] -> [B * Beams, Vocab]
    logits = logits.index({Slice(), Slice(-1, std::nullopt), Slice()});
    logits = logits.view({batch_size * num_beams, vocab_size});
    
    // Convert to log probabilities
    Tensor log_probs_tensor = nn::functional::log_softmax(logits, -1);
    Tensor log_probs_cpu = log_probs_tensor.to(Device::CPU).contiguous();
    const float* log_probs = log_probs_cpu.data_ptr<float>();
    
    int64_t pos = prompt_len;
    
    for (int64_t gen = 0; gen < params.max_new_tokens; ++gen) {
        // Process each batch element independently
        for (int64_t b = 0; b < batch_size; ++b) {
            // Check if all beams for this batch are done
            bool all_done = true;
            for (int64_t beam = 0; beam < num_beams; ++beam) {
                if (!beam_done[b][beam]) {
                    all_done = false;
                    break;
                }
            }
            if (all_done) continue;
            
            // Collect all candidates: beam * vocab possibilities
            // We use a priority queue to find top-k
            struct Candidate {
                float score;
                int64_t beam;
                int32_t token;
                
                bool operator<(const Candidate& other) const {
                    return score < other.score;  // Max heap
                }
            };
            
            std::priority_queue<Candidate> candidates;
            
            for (int64_t beam = 0; beam < num_beams; ++beam) {
                if (beam_done[b][beam]) continue;
                
                float current_score = beam_scores[b][beam];
                const float* beam_log_probs = log_probs + (b * num_beams + beam) * vocab_size;
                
                // Add candidates for this beam
                for (int64_t v = 0; v < vocab_size; ++v) {
                    float new_score = current_score + beam_log_probs[v];
                    
                    // Pruning: skip if too far from best
                    if (new_score < params.beam_score_threshold) continue;
                    
                    candidates.push({new_score, beam, static_cast<int32_t>(v)});
                }
            }
            
            // Select top num_beams candidates
            std::vector<std::vector<int32_t>> new_tokens(num_beams);
            std::vector<float> new_scores(num_beams, -std::numeric_limits<float>::infinity());
            std::vector<bool> new_done(num_beams, true);
            
            int64_t selected = 0;
            while (!candidates.empty() && selected < num_beams) {
                Candidate cand = candidates.top();
                candidates.pop();
                
                // Copy tokens from source beam
                new_tokens[selected] = beam_tokens[b][cand.beam];
                new_tokens[selected].push_back(cand.token);
                new_scores[selected] = cand.score;
                
                // Check if this is EOS
                if (is_eos(cand.token, params)) {
                    // Add to finished sequences with length penalty
                    float final_score = apply_length_penalty(
                        cand.score, 
                        static_cast<int64_t>(new_tokens[selected].size()) - prompt_len,
                        params.length_penalty
                    );
                    
                    std::vector<int64_t> seq(new_tokens[selected].begin(), new_tokens[selected].end());
                    finished_sequences_[b].emplace_back(seq, final_score);
                    
                    new_done[selected] = true;
                } else {
                    new_done[selected] = false;
                }
                
                ++selected;
            }
            
            // Fill remaining beams with invalid if not enough candidates
            while (selected < num_beams) {
                new_tokens[selected] = beam_tokens[b][0];  // Copy any valid sequence
                new_scores[selected] = -std::numeric_limits<float>::infinity();
                new_done[selected] = true;
                ++selected;
            }
            
            beam_tokens[b] = new_tokens;
            beam_scores[b] = new_scores;
            beam_done[b] = new_done;
        }
        
        // Check if we should stop early
        if (params.early_stopping) {
            bool should_stop = true;
            for (int64_t b = 0; b < batch_size; ++b) {
                if (finished_sequences_[b].size() < static_cast<size_t>(num_beams)) {
                    // Check if any beam is still active
                    for (int64_t beam = 0; beam < num_beams; ++beam) {
                        if (!beam_done[b][beam]) {
                            should_stop = false;
                            break;
                        }
                    }
                }
            }
            if (should_stop) break;
        }
        
        // Check if all beams across all batches are done
        bool all_done = true;
        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t beam = 0; beam < num_beams; ++beam) {
                if (!beam_done[b][beam]) {
                    all_done = false;
                    break;
                }
            }
            if (!all_done) break;
        }
        if (all_done) break;
        
        // Prepare next token tensor for forward pass
        Tensor next_tokens = vesper::empty({batch_size * num_beams, 1}, DType::Int32, Device::CPU);
        int32_t* next_data = next_tokens.data_ptr<int32_t>();
        
        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t beam = 0; beam < num_beams; ++beam) {
                // Get last token from beam
                const auto& tokens = beam_tokens[b][beam];
                next_data[b * num_beams + beam] = tokens.empty() ? 0 : tokens.back();
            }
        }
        
        if (device != Device::CPU) {
            next_tokens = next_tokens.to(device);
        }
        
        // Forward with new tokens
        // Note: We need to handle the KV cache reordering for beam search
        // For simplicity, we're treating each beam independently
        logits = model_->forward_with_cache(next_tokens, pos);
        logits = logits.view({batch_size * num_beams, vocab_size});
        
        log_probs_tensor = nn::functional::log_softmax(logits, -1);
        log_probs_cpu = log_probs_tensor.to(Device::CPU).contiguous();
        log_probs = log_probs_cpu.data_ptr<float>();
        
        ++pos;
    }
    
    model_->clear_cache();
    
    if (was_training) model_->train();
    
    // Add any unfinished beams to finished sequences
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t beam = 0; beam < num_beams; ++beam) {
            if (!beam_done[b][beam]) {
                float final_score = apply_length_penalty(
                    beam_scores[b][beam],
                    static_cast<int64_t>(beam_tokens[b][beam].size()) - prompt_len,
                    params.length_penalty
                );
                
                std::vector<int64_t> seq(beam_tokens[b][beam].begin(), beam_tokens[b][beam].end());
                finished_sequences_[b].emplace_back(seq, final_score);
            }
        }
    }
    
    // Sort finished sequences by score and select best
    for (auto& batch_seqs : finished_sequences_) {
        std::sort(batch_seqs.begin(), batch_seqs.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
    }
    
    // Build result tensor
    // Find max sequence length
    int64_t max_len = 0;
    for (int64_t b = 0; b < batch_size; ++b) {
        if (!finished_sequences_[b].empty()) {
            max_len = std::max(max_len, static_cast<int64_t>(finished_sequences_[b][0].first.size()));
        }
    }
    
    // If no sequences finished, return prompt
    if (max_len == 0) {
        return prompt_ids;
    }
    
    int64_t result_batch = params.return_all_beams ? batch_size * num_beams : batch_size;
    Tensor result = vesper::zeros({result_batch, max_len}, DType::Int32, Device::CPU);
    int32_t* result_data = result.data_ptr<int32_t>();
    
    for (int64_t b = 0; b < batch_size; ++b) {
        int64_t num_to_return = params.return_all_beams ? 
            std::min(num_beams, static_cast<int64_t>(finished_sequences_[b].size())) : 1;
        
        for (int64_t beam = 0; beam < num_to_return; ++beam) {
            if (beam < static_cast<int64_t>(finished_sequences_[b].size())) {
                const auto& seq = finished_sequences_[b][beam].first;
                int64_t row = params.return_all_beams ? b * num_beams + beam : b;
                
                for (size_t s = 0; s < seq.size(); ++s) {
                    result_data[row * max_len + s] = static_cast<int32_t>(seq[s]);
                }
            }
        }
    }
    
    if (device != Device::CPU) {
        result = result.to(device);
    }
    
    return result;
}

} // namespace vesper::generation
