#include <vesper/generation/sampling.h>
#include <vesper/ops/sampling_ops.h>
#include <vesper/core/factories.h>
#include <vesper/nn/functional.h>
#include <vesper/ops/elementwise.h>
#include <vesper/core/macros.h>

#include <algorithm>
#include <random>
#include <cmath>
#include <limits>

namespace vesper::generation {

Tensor apply_temperature(const Tensor& logits, float temperature) {
    if (temperature == 1.0f) {
        return logits;
    }
    
    VESPER_CHECK(temperature > 0.0f, "Temperature must be positive, got " + std::to_string(temperature));
    
    // GPU-accelerated division
    return ops::div(logits, temperature);
}

Tensor apply_top_k(const Tensor& logits, int64_t k) {
    // Delegate to GPU-accelerated implementation
    return ops::top_k_filter(logits, k);
}

Tensor apply_top_p(const Tensor& logits, float p) {
    // Delegate to GPU-accelerated implementation
    return ops::top_p_filter(logits, p);
}

Tensor apply_repetition_penalty(const Tensor& logits, 
                                const Tensor& input_ids,
                                float penalty) {
    // Delegate to GPU-accelerated implementation
    return ops::apply_repetition_penalty(logits, input_ids, penalty);
}

Tensor greedy_sample(const Tensor& logits) {
    VESPER_CHECK(logits.ndim() == 2, "greedy_sample expects 2D input [Batch, Vocab]");
    
    // GPU-accelerated argmax
    return ops::argmax(logits, -1);
}

Tensor multinomial_sample(const Tensor& probs, int64_t num_samples, int64_t seed) {
    // GPU-accelerated multinomial sampling
    return ops::multinomial(probs, num_samples, seed);
}

Tensor sample_next_token(const Tensor& logits, 
                         const SamplingParams& params,
                         const Tensor* input_ids) {
    VESPER_CHECK(logits.ndim() == 2, "sample_next_token expects 2D input [Batch, Vocab]");
    
    Tensor processed = logits;
    
    // 1. Apply repetition penalty if input_ids provided (GPU-accelerated)
    if (input_ids != nullptr && params.repetition_penalty != 1.0f) {
        processed = apply_repetition_penalty(processed, *input_ids, params.repetition_penalty);
    }
    
    // 2. Apply temperature scaling (GPU-accelerated)
    processed = apply_temperature(processed, params.temperature);
    
    // 3. Apply Top-K filtering (GPU-accelerated)
    if (params.top_k > 0) {
        processed = apply_top_k(processed, params.top_k);
    }
    
    // 4. Apply Top-P filtering (GPU-accelerated)
    if (params.top_p < 1.0f) {
        processed = apply_top_p(processed, params.top_p);
    }
    
    // 5. Greedy or sample
    if (!params.do_sample) {
        return greedy_sample(processed);
    }
    
    // 6. Convert to probabilities and sample (GPU-accelerated)
    Tensor probs = nn::functional::softmax(processed, -1);
    
    // Sample single token per batch element
    Tensor sampled = multinomial_sample(probs, 1, params.seed);
    
    // Squeeze the sample dimension: [Batch, 1] -> [Batch]
    return sampled.view({sampled.shape()[0]});
}

} // namespace vesper::generation
