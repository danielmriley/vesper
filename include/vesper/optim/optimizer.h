#pragma once

#include <vesper/core/tensor.h>
#include <vesper/core/state_dict.h>
#include <vector>
#include <unordered_map>
#include <string>

namespace vesper::optim {

class Optimizer {
public:
    // The constructor takes a vector of Tensors (shared_ptr wrappers)
    explicit Optimizer(std::vector<Tensor> params);
    
    virtual ~Optimizer() = default;

    // `step` must be implemented by concrete optimizers (e.g., SGD, Adam)
    virtual void step() = 0;

    // Get/Set Learning Rate (for Schedulers)
    virtual void set_lr(float lr) = 0;
    virtual float get_lr() const = 0;

    // A convenience method to zero the gradients of all managed parameters
    void zero_grad();

    // Serialization
    virtual StateDict state_dict() const;
    virtual void load_state_dict(const StateDict& state_dict);

protected:
    std::vector<Tensor> params_;
    // State for each parameter (e.g., momentum buffers). 
    // state_[i] corresponds to params_[i].
    // Maps state name (e.g., "momentum", "exp_avg") to a Tensor.
    std::vector<std::unordered_map<std::string, Tensor>> state_;
};

} // namespace vesper::optim
