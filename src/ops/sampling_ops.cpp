#include <vesper/ops/sampling_ops.h>
#include <vesper/core/factories.h>
#include <vesper/core/macros.h>

#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <limits>

namespace vesper::ops {

// =============================================================================
// CPU Implementations
// =============================================================================

void argmax_cpu_dispatch(const Tensor& input, Tensor& output) {
    int64_t batch_size = input.shape()[0];
    int64_t vocab_size = input.shape()[1];
    
    const float* in_data = input.data_ptr<const float>();
    int32_t* out_data = output.data_ptr<int32_t>();
    
    for (int64_t b = 0; b < batch_size; ++b) {
        const float* batch_input = in_data + b * vocab_size;
        float max_val = batch_input[0];
        int32_t max_idx = 0;
        
        for (int64_t i = 1; i < vocab_size; ++i) {
            if (batch_input[i] > max_val) {
                max_val = batch_input[i];
                max_idx = static_cast<int32_t>(i);
            }
        }
        out_data[b] = max_idx;
    }
}

void top_k_filter_cpu_dispatch(const Tensor& logits, Tensor& output, int64_t k) {
    int64_t batch_size = logits.shape()[0];
    int64_t vocab_size = logits.shape()[1];
    
    const float* in_data = logits.data_ptr<const float>();
    float* out_data = output.data_ptr<float>();
    
    for (int64_t b = 0; b < batch_size; ++b) {
        const float* batch_input = in_data + b * vocab_size;
        float* batch_output = out_data + b * vocab_size;
        
        // Initialize output to -inf
        for (int64_t i = 0; i < vocab_size; ++i) {
            batch_output[i] = -std::numeric_limits<float>::infinity();
        }
        
        // Create indices and partial sort
        std::vector<int64_t> indices(vocab_size);
        std::iota(indices.begin(), indices.end(), 0);
        
        std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
            [batch_input](int64_t a, int64_t b) {
                return batch_input[a] > batch_input[b];
            });
        
        // Keep top-k
        for (int64_t i = 0; i < k; ++i) {
            int64_t idx = indices[i];
            batch_output[idx] = batch_input[idx];
        }
    }
}

void top_p_filter_cpu_dispatch(const Tensor& logits, Tensor& output, float p) {
    int64_t batch_size = logits.shape()[0];
    int64_t vocab_size = logits.shape()[1];
    
    const float* in_data = logits.data_ptr<const float>();
    float* out_data = output.data_ptr<float>();
    
    for (int64_t b = 0; b < batch_size; ++b) {
        const float* batch_input = in_data + b * vocab_size;
        float* batch_output = out_data + b * vocab_size;
        
        // Compute softmax probabilities
        float max_logit = *std::max_element(batch_input, batch_input + vocab_size);
        
        std::vector<float> probs(vocab_size);
        float sum = 0.0f;
        for (int64_t i = 0; i < vocab_size; ++i) {
            probs[i] = std::exp(batch_input[i] - max_logit);
            sum += probs[i];
        }
        for (auto& prob : probs) prob /= sum;
        
        // Sort indices by probability
        std::vector<int64_t> indices(vocab_size);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
            [&probs](int64_t a, int64_t b) { return probs[a] > probs[b]; });
        
        // Find cutoff
        float cumsum = 0.0f;
        size_t cutoff = vocab_size;
        for (size_t i = 0; i < static_cast<size_t>(vocab_size); ++i) {
            cumsum += probs[indices[i]];
            if (cumsum > p) {
                cutoff = i + 1;
                break;
            }
        }
        
        // Initialize output to -inf
        for (int64_t i = 0; i < vocab_size; ++i) {
            batch_output[i] = -std::numeric_limits<float>::infinity();
        }
        
        // Keep tokens within nucleus
        for (size_t i = 0; i < cutoff; ++i) {
            int64_t idx = indices[i];
            batch_output[idx] = batch_input[idx];
        }
    }
}

void repetition_penalty_cpu_dispatch(const Tensor& logits, Tensor& output,
                                      const Tensor& input_ids, float penalty) {
    int64_t batch_size = logits.shape()[0];
    int64_t vocab_size = logits.shape()[1];
    int64_t seq_len = input_ids.shape()[1];
    
    const float* in_data = logits.data_ptr<const float>();
    float* out_data = output.data_ptr<float>();
    const int32_t* ids_data = input_ids.data_ptr<const int32_t>();
    
    // Copy logits to output
    std::copy(in_data, in_data + batch_size * vocab_size, out_data);
    
    for (int64_t b = 0; b < batch_size; ++b) {
        float* batch_output = out_data + b * vocab_size;
        const int32_t* batch_ids = ids_data + b * seq_len;
        
        std::vector<bool> penalized(vocab_size, false);
        
        for (int64_t s = 0; s < seq_len; ++s) {
            int32_t token_id = batch_ids[s];
            if (token_id >= 0 && token_id < vocab_size && !penalized[token_id]) {
                float& logit = batch_output[token_id];
                if (logit > 0) {
                    logit /= penalty;
                } else {
                    logit *= penalty;
                }
                penalized[token_id] = true;
            }
        }
    }
}

void multinomial_cpu_dispatch(const Tensor& probs, Tensor& output,
                               int64_t num_samples, uint64_t seed) {
    int64_t batch_size = probs.shape()[0];
    int64_t vocab_size = probs.shape()[1];
    
    const float* in_data = probs.data_ptr<const float>();
    int32_t* out_data = output.data_ptr<int32_t>();
    
    std::mt19937 gen(seed != static_cast<uint64_t>(-1) ? 
                     static_cast<unsigned>(seed) : std::random_device{}());
    
    for (int64_t b = 0; b < batch_size; ++b) {
        const float* batch_probs = in_data + b * vocab_size;
        int32_t* batch_output = out_data + b * num_samples;
        
        std::vector<float> p(batch_probs, batch_probs + vocab_size);
        
        // Handle numerical issues
        float sum = 0.0f;
        for (auto& v : p) {
            if (v < 0) v = 0;
            sum += v;
        }
        
        if (sum < 1e-8f) {
            // Uniform sampling
            std::uniform_int_distribution<int32_t> dist(0, static_cast<int32_t>(vocab_size - 1));
            for (int64_t s = 0; s < num_samples; ++s) {
                batch_output[s] = dist(gen);
            }
        } else {
            for (auto& v : p) v /= sum;
            std::discrete_distribution<int32_t> dist(p.begin(), p.end());
            for (int64_t s = 0; s < num_samples; ++s) {
                batch_output[s] = dist(gen);
            }
        }
    }
}

void topk_cpu_dispatch(const Tensor& input, int64_t k, Tensor& values, Tensor& indices) {
    int64_t batch_size = input.shape()[0];
    int64_t n = input.shape()[1];
    
    const float* in_data = input.data_ptr<const float>();
    float* val_data = values.data_ptr<float>();
    int32_t* idx_data = indices.data_ptr<int32_t>();
    
    for (int64_t b = 0; b < batch_size; ++b) {
        const float* batch_input = in_data + b * n;
        float* batch_values = val_data + b * k;
        int32_t* batch_indices = idx_data + b * k;
        
        std::vector<int64_t> all_indices(n);
        std::iota(all_indices.begin(), all_indices.end(), 0);
        
        std::partial_sort(all_indices.begin(), all_indices.begin() + k, all_indices.end(),
            [batch_input](int64_t a, int64_t b) {
                return batch_input[a] > batch_input[b];
            });
        
        for (int64_t i = 0; i < k; ++i) {
            int64_t idx = all_indices[i];
            batch_values[i] = batch_input[idx];
            batch_indices[i] = static_cast<int32_t>(idx);
        }
    }
}

void softmax_2d_cpu_dispatch(const Tensor& input, Tensor& output) {
    int64_t batch_size = input.shape()[0];
    int64_t n = input.shape()[1];
    
    const float* in_data = input.data_ptr<const float>();
    float* out_data = output.data_ptr<float>();
    
    for (int64_t b = 0; b < batch_size; ++b) {
        const float* batch_input = in_data + b * n;
        float* batch_output = out_data + b * n;
        
        float max_val = *std::max_element(batch_input, batch_input + n);
        
        float sum = 0.0f;
        for (int64_t i = 0; i < n; ++i) {
            batch_output[i] = std::exp(batch_input[i] - max_val);
            sum += batch_output[i];
        }
        
        for (int64_t i = 0; i < n; ++i) {
            batch_output[i] /= sum;
        }
    }
}

void check_stop_tokens_cpu_dispatch(const Tensor& tokens, Tensor& output,
                                     const int64_t* stop_ids, int64_t num_stop_ids,
                                     int64_t min_generated, int64_t generated_count) {
    int64_t batch_size = tokens.numel();
    const int32_t* token_data = tokens.data_ptr<const int32_t>();
    int32_t* out_data = output.data_ptr<int32_t>();
    
    for (int64_t b = 0; b < batch_size; ++b) {
        if (generated_count < min_generated) {
            out_data[b] = 0;
            continue;
        }
        
        int32_t token = token_data[b];
        int32_t should_stop = 0;
        
        for (int64_t i = 0; i < num_stop_ids; ++i) {
            if (token == static_cast<int32_t>(stop_ids[i])) {
                should_stop = 1;
                break;
            }
        }
        
        out_data[b] = should_stop;
    }
}

bool all_true_cpu_dispatch(const Tensor& mask) {
    int64_t n = mask.numel();
    const int32_t* data = mask.data_ptr<const int32_t>();
    
    for (int64_t i = 0; i < n; ++i) {
        if (data[i] == 0) return false;
    }
    return true;
}

// =============================================================================
// Public API with Device Dispatch
// =============================================================================

Tensor argmax(const Tensor& input, int64_t dim) {
    VESPER_CHECK(input.ndim() == 2, "argmax currently only supports 2D tensors");
    VESPER_CHECK(dim == -1 || dim == 1, "argmax currently only supports dim=-1 or dim=1");
    
    int64_t batch_size = input.shape()[0];
    Tensor output = empty({batch_size}, DType::Int32, input.device());
    
    if (input.device() == Device::CPU) {
        Tensor input_contig = input.contiguous();
        argmax_cpu_dispatch(input_contig, output);
    } else if (input.device() == Device::HIP) {
#if USE_HIP_BACKEND
        Tensor input_contig = input.contiguous();
        argmax_hip_dispatch(input_contig, output);
#else
        throw std::runtime_error("HIP backend not enabled");
#endif
    } else if (input.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        Tensor input_contig = input.contiguous();
        argmax_cuda_dispatch(input_contig, output);
#else
        throw std::runtime_error("CUDA backend not enabled");
#endif
    } else {
        throw std::runtime_error("Unsupported device for argmax");
    }
    
    return output;
}

Tensor top_k_filter(const Tensor& logits, int64_t k) {
    VESPER_CHECK(logits.ndim() == 2, "top_k_filter expects 2D input [Batch, Vocab]");
    
    int64_t vocab_size = logits.shape().back();
    if (k <= 0 || k >= vocab_size) {
        return logits.clone();
    }
    
    Tensor output = empty(logits.shape(), logits.dtype(), logits.device());
    
    if (logits.device() == Device::CPU) {
        Tensor input_contig = logits.contiguous();
        top_k_filter_cpu_dispatch(input_contig, output, k);
    } else if (logits.device() == Device::HIP) {
#if USE_HIP_BACKEND
        Tensor input_contig = logits.contiguous();
        top_k_filter_hip_dispatch(input_contig, output, k);
#else
        throw std::runtime_error("HIP backend not enabled");
#endif
    } else if (logits.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        Tensor input_contig = logits.contiguous();
        top_k_filter_cuda_dispatch(input_contig, output, k);
#else
        throw std::runtime_error("CUDA backend not enabled");
#endif
    } else {
        throw std::runtime_error("Unsupported device for top_k_filter");
    }
    
    return output;
}

Tensor top_p_filter(const Tensor& logits, float p) {
    VESPER_CHECK(logits.ndim() == 2, "top_p_filter expects 2D input [Batch, Vocab]");
    
    if (p >= 1.0f) {
        return logits.clone();
    }
    
    Tensor output = empty(logits.shape(), logits.dtype(), logits.device());
    
    if (logits.device() == Device::CPU) {
        Tensor input_contig = logits.contiguous();
        top_p_filter_cpu_dispatch(input_contig, output, p);
    } else if (logits.device() == Device::HIP) {
#if USE_HIP_BACKEND
        Tensor input_contig = logits.contiguous();
        top_p_filter_hip_dispatch(input_contig, output, p);
#else
        throw std::runtime_error("HIP backend not enabled");
#endif
    } else if (logits.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        Tensor input_contig = logits.contiguous();
        top_p_filter_cuda_dispatch(input_contig, output, p);
#else
        throw std::runtime_error("CUDA backend not enabled");
#endif
    } else {
        throw std::runtime_error("Unsupported device for top_p_filter");
    }
    
    return output;
}

Tensor apply_repetition_penalty(const Tensor& logits, const Tensor& input_ids, float penalty) {
    VESPER_CHECK(logits.ndim() == 2, "apply_repetition_penalty expects 2D logits");
    VESPER_CHECK(input_ids.ndim() == 2, "apply_repetition_penalty expects 2D input_ids");
    VESPER_CHECK(logits.shape()[0] == input_ids.shape()[0], "Batch size mismatch");
    
    if (penalty == 1.0f) {
        return logits.clone();
    }
    
    Tensor output = empty(logits.shape(), logits.dtype(), logits.device());
    
    if (logits.device() == Device::CPU) {
        Tensor logits_contig = logits.contiguous();
        Tensor ids_contig = input_ids.contiguous();
        repetition_penalty_cpu_dispatch(logits_contig, output, ids_contig, penalty);
    } else if (logits.device() == Device::HIP) {
#if USE_HIP_BACKEND
        Tensor logits_contig = logits.contiguous();
        Tensor ids_contig = input_ids.to(logits.device()).contiguous();
        repetition_penalty_hip_dispatch(logits_contig, output, ids_contig, penalty);
#else
        throw std::runtime_error("HIP backend not enabled");
#endif
    } else if (logits.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        Tensor logits_contig = logits.contiguous();
        Tensor ids_contig = input_ids.to(logits.device()).contiguous();
        repetition_penalty_cuda_dispatch(logits_contig, output, ids_contig, penalty);
#else
        throw std::runtime_error("CUDA backend not enabled");
#endif
    } else {
        throw std::runtime_error("Unsupported device for apply_repetition_penalty");
    }
    
    return output;
}

Tensor multinomial(const Tensor& probs, int64_t num_samples, int64_t seed) {
    VESPER_CHECK(probs.ndim() == 2, "multinomial expects 2D input [Batch, Vocab]");
    VESPER_CHECK(num_samples >= 1, "num_samples must be at least 1");
    
    int64_t batch_size = probs.shape()[0];
    Tensor output = empty({batch_size, num_samples}, DType::Int32, probs.device());
    
    uint64_t actual_seed = (seed < 0) ? 
        static_cast<uint64_t>(std::random_device{}()) : 
        static_cast<uint64_t>(seed);
    
    if (probs.device() == Device::CPU) {
        Tensor probs_contig = probs.contiguous();
        multinomial_cpu_dispatch(probs_contig, output, num_samples, actual_seed);
    } else if (probs.device() == Device::HIP) {
#if USE_HIP_BACKEND
        Tensor probs_contig = probs.contiguous();
        multinomial_hip_dispatch(probs_contig, output, num_samples, actual_seed);
#else
        throw std::runtime_error("HIP backend not enabled");
#endif
    } else if (probs.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        Tensor probs_contig = probs.contiguous();
        multinomial_cuda_dispatch(probs_contig, output, num_samples, actual_seed);
#else
        throw std::runtime_error("CUDA backend not enabled");
#endif
    } else {
        throw std::runtime_error("Unsupported device for multinomial");
    }
    
    return output;
}

std::pair<Tensor, Tensor> topk(const Tensor& input, int64_t k, int64_t dim) {
    VESPER_CHECK(input.ndim() == 2, "topk currently only supports 2D tensors");
    VESPER_CHECK(dim == -1 || dim == 1, "topk currently only supports dim=-1 or dim=1");
    
    int64_t batch_size = input.shape()[0];
    int64_t n = input.shape()[1];
    k = std::min(k, n);
    
    Tensor values = empty({batch_size, k}, input.dtype(), input.device());
    Tensor indices = empty({batch_size, k}, DType::Int32, input.device());
    
    if (input.device() == Device::CPU) {
        Tensor input_contig = input.contiguous();
        topk_cpu_dispatch(input_contig, k, values, indices);
    } else if (input.device() == Device::HIP) {
#if USE_HIP_BACKEND
        Tensor input_contig = input.contiguous();
        topk_hip_dispatch(input_contig, k, values, indices);
#else
        throw std::runtime_error("HIP backend not enabled");
#endif
    } else if (input.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        Tensor input_contig = input.contiguous();
        topk_cuda_dispatch(input_contig, k, values, indices);
#else
        throw std::runtime_error("CUDA backend not enabled");
#endif
    } else {
        throw std::runtime_error("Unsupported device for topk");
    }
    
    return {values, indices};
}

Tensor check_stop_tokens(const Tensor& tokens, const std::vector<int64_t>& stop_token_ids,
                          int64_t min_generated, int64_t generated_count) {
    int64_t batch_size = tokens.numel();
    
    // Output is Int32 tensor (0 = continue, 1 = stop)
    Tensor output = vesper::zeros({batch_size}, DType::Int32, tokens.device());
    
    if (stop_token_ids.empty()) {
        // No stop tokens, all false (continue generating) - already zeros
        return output;
    }
    
    if (tokens.device() == Device::CPU) {
        Tensor tokens_contig = tokens.contiguous();
        check_stop_tokens_cpu_dispatch(tokens_contig, output, 
                                        stop_token_ids.data(), stop_token_ids.size(),
                                        min_generated, generated_count);
    } else if (tokens.device() == Device::HIP) {
#if USE_HIP_BACKEND
        Tensor tokens_contig = tokens.contiguous();
        check_stop_tokens_hip_dispatch(tokens_contig, output,
                                        stop_token_ids.data(), stop_token_ids.size(),
                                        min_generated, generated_count);
#else
        throw std::runtime_error("HIP backend not enabled");
#endif
    } else if (tokens.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        Tensor tokens_contig = tokens.contiguous();
        check_stop_tokens_cuda_dispatch(tokens_contig, output,
                                         stop_token_ids.data(), stop_token_ids.size(),
                                         min_generated, generated_count);
#else
        throw std::runtime_error("CUDA backend not enabled");
#endif
    } else {
        throw std::runtime_error("Unsupported device for check_stop_tokens");
    }
    
    return output;
}

bool all_true(const Tensor& mask) {
    VESPER_CHECK(mask.dtype() == DType::Int32, "all_true expects Int32 (boolean) tensor");
    
    if (mask.numel() == 0) return true;
    
    if (mask.device() == Device::CPU) {
        Tensor mask_contig = mask.contiguous();
        return all_true_cpu_dispatch(mask_contig);
    } else if (mask.device() == Device::HIP) {
#if USE_HIP_BACKEND
        Tensor mask_contig = mask.contiguous();
        return all_true_hip_dispatch(mask_contig);
#else
        throw std::runtime_error("HIP backend not enabled");
#endif
    } else if (mask.device() == Device::CUDA) {
#if USE_CUDA_BACKEND
        Tensor mask_contig = mask.contiguous();
        return all_true_cuda_dispatch(mask_contig);
#else
        throw std::runtime_error("CUDA backend not enabled");
#endif
    } else {
        throw std::runtime_error("Unsupported device for all_true");
    }
}

} // namespace vesper::ops
