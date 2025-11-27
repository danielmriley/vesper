#include <vesper/autograd/checkpoint.h>
#include <vesper/autograd/node.h>
#include <vesper/autograd/guard.h>
#include <vesper/ops/elementwise.h>
#include <vesper/core/factories.h>

namespace vesper::autograd {

namespace detail {

Tensor CheckpointFunction::apply(ForwardFn fn, const Tensor& input) {
    // If gradient tracking is disabled, just run forward normally
    if (!grad_mode_enabled) {
        return fn(input);
    }
    
    // Run forward WITHOUT gradient tracking
    Tensor output;
    {
        NoGradGuard no_grad;
        output = fn(input);
    }
    
    // If input doesn't require grad, no need for backward setup
    if (!input.requires_grad()) {
        return output;
    }
    
    // Set output to require grad
    output.set_requires_grad(true);
    
    // Create autograd node that will recompute forward during backward
    auto node = std::make_shared<Node>();
    
    // Track input dependency
    if (input.grad_node) {
        node->next_edges.push_back({input.grad_node});
    }
    
    // The backward function: recompute forward and backprop
    node->backward_fn = [fn=std::move(fn), input=input, result_weak=output.weak()]() mutable {
        auto result = result_weak.lock();
        if (!result) return;
        
        Tensor grad_output = result->grad();
        
        // Recompute forward WITH gradient tracking enabled
        Tensor recomputed_input = input.clone();
        recomputed_input.set_requires_grad(true);
        
        Tensor recomputed_output;
        {
            EnableGradGuard enable_grad;
            recomputed_output = fn(recomputed_input);
        }
        
        // Backpropagate through the recomputed graph
        recomputed_output.backward(grad_output);
        
        // Accumulate gradient to original input
        if (input.requires_grad() && recomputed_input.grad().defined()) {
            input.grad() = ops::add(input.grad(), recomputed_input.grad());
        }
    };
    
    output.grad_node = node;
    
    return output;
}

} // namespace detail

// CheckpointedModule implementation
CheckpointedModule::CheckpointedModule(std::shared_ptr<nn::Module> module)
    : module_(std::move(module)) {}

Tensor CheckpointedModule::forward(const Tensor& input) {
    if (!is_training() || !input.requires_grad()) {
        // During eval or when not training, don't checkpoint
        return module_->forward(input);
    }
    
    // Use checkpointing during training
    return checkpoint([this](const Tensor& x) {
        return module_->forward(x);
    }, input);
}

// CheckpointedSequential implementation
CheckpointedSequential::CheckpointedSequential(int checkpoint_every)
    : checkpoint_every_(checkpoint_every) {
    if (checkpoint_every < 1) {
        throw std::runtime_error("checkpoint_every must be >= 1");
    }
}

void CheckpointedSequential::add(std::shared_ptr<nn::Module> module) {
    // Register the module so to() works correctly
    register_module("layer_" + std::to_string(modules_.size()), module.get());
    modules_.push_back(std::move(module));
}

Tensor CheckpointedSequential::forward(const Tensor& x) {
    Tensor out = x;
    
    for (size_t i = 0; i < modules_.size(); ++i) {
        bool should_checkpoint = is_training() && 
                                 out.requires_grad() &&
                                 (i % checkpoint_every_ == 0);
        
        if (should_checkpoint) {
            // Checkpoint this layer
            auto& mod = modules_[i];
            out = checkpoint([&mod](const Tensor& input) {
                return mod->forward(input);
            }, out);
        } else {
            // Normal forward
            out = modules_[i]->forward(out);
        }
    }
    
    return out;
}

} // namespace vesper::autograd
