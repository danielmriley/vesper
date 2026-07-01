#pragma once

#include <vesper/core/tensor.h>
#include <vesper/nn/module.h>
#include <functional>
#include <memory>
#include <vector>
#include <stdexcept>

namespace vesper::autograd {

/**
 * @brief Checkpoint a function to trade compute for memory
 * 
 * During training, activations are saved for the backward pass. For deep
 * networks, this can consume enormous amounts of memory. Checkpointing
 * saves only the inputs and recomputes the forward pass during backward.
 * 
 * Trade-off: ~33% more compute for ~60-70% less activation memory.
 * 
 * Usage:
 * ```cpp
 * // Instead of:
 * Tensor y = expensive_function(x);
 * 
 * // Use:
 * Tensor y = checkpoint(expensive_function, x);
 * ```
 * 
 * @tparam Fn Callable type (function, lambda, etc.)
 * @param fn The function to checkpoint
 * @param input Input tensor
 * @return Output tensor (gradients will recompute fn during backward)
 */
template<typename Fn>
Tensor checkpoint(Fn&& fn, const Tensor& input);

/**
 * @brief Checkpoint with multiple inputs
 */
template<typename Fn>
Tensor checkpoint(Fn&& fn, const std::vector<Tensor>& inputs);

/**
 * @brief A module wrapper that checkpoints its forward pass
 * 
 * Usage:
 * ```cpp
 * auto layer = std::make_shared<TransformerBlock>(...);
 * CheckpointedModule ckpt_layer(layer);
 * Tensor y = ckpt_layer.forward(x);  // Forward will be recomputed during backward
 * ```
 */
class CheckpointedModule : public nn::Module {
public:
    explicit CheckpointedModule(std::shared_ptr<nn::Module> module);
    
    Tensor forward(const Tensor& input) override;
    
    /// Get the underlying module
    std::shared_ptr<nn::Module> module() const { return module_; }
    
private:
    std::shared_ptr<nn::Module> module_;
};

/**
 * @brief Sequential container with automatic checkpointing
 * 
 * Checkpoints every N layers during forward pass to reduce memory usage.
 * 
 * Usage:
 * @code
 * CheckpointedSequential layers(2);  // checkpoint_every=2
 * layers.add(std::make_shared<TransformerBlock>(...));
 * layers.add(std::make_shared<TransformerBlock>(...));
 * layers.add(std::make_shared<TransformerBlock>(...));
 * layers.add(std::make_shared<TransformerBlock>(...));
 * // Layers 0, 2 will be checkpointed; layers 1, 3 run normally
 * Tensor y = layers.forward(x);
 * @endcode
 */
class CheckpointedSequential : public nn::Module {
public:
    /**
     * @brief Construct checkpointed sequential
     * @param checkpoint_every Checkpoint every N layers (1 = every layer)
     */
    explicit CheckpointedSequential(int checkpoint_every = 1);
    
    /**
     * @brief Add a module to the sequence
     */
    void add(std::shared_ptr<nn::Module> module);
    
    /**
     * @brief Forward pass with checkpointing
     */
    Tensor forward(const Tensor& x) override;
    
    /// Get number of modules
    size_t size() const { return modules_.size(); }
    
    /// Get module at index
    std::shared_ptr<nn::Module> get(size_t idx) const { return modules_[idx]; }
    
private:
    std::vector<std::shared_ptr<nn::Module>> modules_;
    int checkpoint_every_;
};

// Implementation details
namespace detail {

/**
 * @brief Custom autograd function for checkpointing
 * 
 * Stores the function and inputs, then recomputes forward during backward.
 */
class CheckpointFunction {
public:
    // Type-erased function storage
    using ForwardFn = std::function<Tensor(const Tensor&)>;
    
    /**
     * @brief Apply checkpointing to a function
     */
    static Tensor apply(ForwardFn fn, const Tensor& input);
};

} // namespace detail

// Template implementation
template<typename Fn>
Tensor checkpoint(Fn&& fn, const Tensor& input) {
    // Wrap the callable in a std::function
    detail::CheckpointFunction::ForwardFn wrapped = 
        [fn = std::forward<Fn>(fn)](const Tensor& x) -> Tensor {
            return fn(x);
        };
    return detail::CheckpointFunction::apply(std::move(wrapped), input);
}

template<typename Fn>
Tensor checkpoint(Fn&& fn, const std::vector<Tensor>& inputs) {
    // For multiple inputs, we need a different approach
    // For now, just support single input
    if (inputs.size() != 1) {
        throw std::runtime_error("checkpoint with multiple inputs not yet supported");
    }
    return checkpoint(std::forward<Fn>(fn), inputs[0]);
}

} // namespace vesper::autograd
