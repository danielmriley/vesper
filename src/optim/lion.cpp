#include <vesper/optim/lion.h>
#include <vesper/core/factories.h>
#include <vesper/ops/elementwise.h>
#include <vesper/nn/init.h>
#include <vesper/autograd/guard.h>
#include <cmath>

namespace vesper::optim {

Lion::Lion(std::vector<Tensor> params, float lr, float beta1, float beta2, float weight_decay)
    : Optimizer(std::move(params)), lr_(lr), beta1_(beta1), beta2_(beta2), weight_decay_(weight_decay) {}

void Lion::step() {
    autograd::NoGradGuard guard;
    for (size_t i = 0; i < params_.size(); ++i) {
        auto& p = params_[i];
        if (!p.requires_grad()) continue;

        Tensor grad = p.grad();

        // Initialize State (Momentum)
        auto& state = state_[i];
        if (state.find("exp_avg") == state.end()) {
            state["exp_avg"] = vesper::zeros(p.shape(), p.dtype(), p.device());
        }
        Tensor& exp_avg = state["exp_avg"];

        // Lion Update
        // c = beta1 * m + (1 - beta1) * g
        // update = sign(c) + weight_decay * p
        // p = p - lr * update
        // m = beta2 * m + (1 - beta2) * g

        // 1. Compute c (interp between m and g using beta1)
        Tensor c = exp_avg * beta1_ + grad * (1.0f - beta1_);
        
        // 2. Update parameter
        Tensor update = ops::sign(c);
        if (weight_decay_ > 0.0f) {
            update.add_(p * weight_decay_);
        }
        p.sub_(update * lr_);

        // 3. Update momentum (interp using beta2)
        // m.mul_(beta2).add_(grad, 1 - beta2)
        exp_avg.mul_(beta2_);
        exp_avg.add_(grad * (1.0f - beta2_));
    }
}

} // namespace vesper::optim
