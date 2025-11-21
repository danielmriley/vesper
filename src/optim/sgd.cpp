#include <vesper/optim/sgd.h>
#include <vesper/core/factories.h>
#include <vesper/ops/elementwise.h>
#include <vesper/autograd/guard.h>

namespace vesper::optim {

SGD::SGD(std::vector<Tensor*> params, float lr)
    : Optimizer(std::move(params)), lr_(lr) {}

void SGD::step() {
    vesper::autograd::NoGradGuard guard; // Disable gradient tracking
    
    for (auto* param : params_) {
        // In a real implementation, we would check if grad is actually present/computed.
        // Here, param->grad() will lazily initialize it to zeros if missing,
        // which results in a no-op update (param = param - lr * 0).
        
        // Compute update: grad * lr
        // Use the overload that takes a float to handle broadcasting/shape matching
        auto update = vesper::ops::mul(param->grad(), lr_);
        
        // Compute new param: param - update
        auto new_param = vesper::ops::sub(*param, update);
        
        // Update the parameter in place (by assigning to the Tensor object)
        // This updates the storage pointer of the Tensor object held by the Optimizer
        // and the Module (since they share the same Tensor* pointer).
        *param = new_param;
    }
}

} // namespace vesper::optim
