#include <vesper/nn/embedding.h>
#include <vesper/core/factories.h>
#include <vesper/ops/embedding.h>
#include <vesper/nn/init.h>

namespace vesper::nn {

Embedding::Embedding(int64_t num_embeddings, 
                     int64_t embedding_dim, 
                     int64_t padding_idx, 
                     float max_norm, 
                     float norm_type,
                     bool scale_grad_by_freq,
                     bool sparse,
                     Device device)
    : num_embeddings_(num_embeddings), 
      embedding_dim_(embedding_dim), 
      padding_idx_(padding_idx), 
      max_norm_(max_norm),
      norm_type_(norm_type),
      scale_grad_by_freq_(scale_grad_by_freq),
      sparse_(sparse)
{
    // Create weight tensor [num_embeddings, embedding_dim]
    weight = empty({num_embeddings, embedding_dim}, DType::Float32, device, true);
    
    // Init: Normal(0, 0.02) - standard transformer initialization
    // Using 1.0 causes saturated softmax and vanishing gradients
    init::normal_(weight, 0.0f, 0.02f);
    
    reset_padding_idx();
    
    register_parameter("weight", &weight);  // Pass pointer to member variable!
}

void Embedding::reset_padding_idx() {
    // If padding_idx is used, init that row to 0
    if (padding_idx_ != -1) {
        // Slice returns a temporary view. Bind to variable to pass as lvalue reference.
        Tensor pad_row = weight.slice(padding_idx_);
        init::zeros_(pad_row);
    }
}

void Embedding::from_pretrained(const Tensor& embeddings, bool freeze) {
    if (embeddings.ndim() != 2) {
        throw std::runtime_error("Embedding matrix must be 2D");
    }
    if (embeddings.shape()[0] != num_embeddings_) {
        throw std::runtime_error("Embedding matrix num_embeddings mismatch");
    }
    if (embeddings.shape()[1] != embedding_dim_) {
        throw std::runtime_error("Embedding matrix embedding_dim mismatch");
    }

    weight.copy_from(embeddings);
    
    if (freeze) {
        weight.set_requires_grad(false);
    }
    
    reset_padding_idx();
}

Tensor Embedding::forward(const Tensor& input) {
    // Note: max_norm is handled inside ops::embedding currently for CPU, 
    // but we should probably move it out or ensure ops handles it correctly for all backends.
    // The book implementation does it in the module.
    // However, to keep consistent with existing ops structure, we will pass it down.
    // But wait, the book says:
    // "The max_norm parameter clips vectors after each update" - usually this means during forward?
    // PyTorch docs: "If given, each embedding vector with norm larger than max_norm is renormalized to have norm max_norm."
    // This happens in-place on the weights.
    
    // For now, we pass it to ops::embedding which seems to handle it.
    // We need to update ops::embedding to take norm_type.
    
    return ops::embedding(input, weight, padding_idx_, scale_grad_by_freq_, sparse_, max_norm_, norm_type_);
}

} // namespace vesper::nn
