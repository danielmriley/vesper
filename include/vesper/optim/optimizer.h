#pragma once

#include <vesper/core/tensor.h>
#include <vector>

namespace vesper::optim {

class Optimizer {
public:
    // The constructor takes a vector of non-owning pointers to the parameters
    explicit Optimizer(std::vector<Tensor*> params);
    
    virtual ~Optimizer() = default;

    // `step` must be implemented by concrete optimizers (e.g., SGD, Adam)
    virtual void step() = 0;

    // A convenience method to zero the gradients of all managed parameters
    void zero_grad();

protected:
    std::vector<Tensor*> params_;
};

} // namespace vesper::optim
