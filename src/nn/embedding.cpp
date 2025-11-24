#include <vesper/nn/embedding.h>
#include <vesper/core/factories.h>
#include <vesper/ops/embedding.h>
#include <vesper/nn/init.h>

namespace vesper::nn {

Embedding::Embedding(int64_t num_embeddings, int64_t embedding_dim, int64_t padding_idx, float max_norm, Device device)
    : num_embeddings_(num_embeddings), embedding_dim_(embedding_dim), padding_idx_(padding_idx), max_norm_(max_norm)
{
    // Create weight tensor [num_embeddings, embedding_dim]
    weight = empty({num_embeddings, embedding_dim}, DType::Float32, device, true);
    
    // Init: Standard Normal(0, 1)
    init::normal_(weight, 0.0f, 1.0f);
    
    // If padding_idx is used, init that row to 0
    if (padding_idx_ != -1) {
        // Slice returns a temporary view. Bind to variable to pass as lvalue reference.
        Tensor pad_row = weight.slice(padding_idx);
        init::zeros_(pad_row);
    }
    
    register_parameter("weight", weight);
}

Tensor Embedding::forward(const Tensor& input) {
    return ops::embedding(input, weight, padding_idx_, false, false, max_norm_);
}

} // namespace vesper::nn
