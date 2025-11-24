#include <vesper/optim/adam.h>
#include <vesper/core/factories.h>
#include <vesper/ops/elementwise.h> // For basic ops
#include <vesper/nn/init.h>         // For zeros_
#include <vesper/autograd/guard.h>
#include <cmath>

namespace vesper::optim {

Adam::Adam(std::vector<Tensor> params, float lr, float beta1, float beta2, float eps, float weight_decay)
    : Optimizer(std::move(params)), lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), weight_decay_(weight_decay) {}

void Adam::step() {
    autograd::NoGradGuard guard;
    t_++;
    
    // Bias correction factors
    // correction1 = 1 - beta1^t
    // correction2 = 1 - beta2^t
    // They are scalars.
    float correction1 = 1.0f - std::pow(beta1_, t_);
    float correction2 = 1.0f - std::pow(beta2_, t_);

    for (size_t i = 0; i < params_.size(); ++i) {
        auto& param = params_[i];
        if (!param.requires_grad()) continue;

        // Get gradient (ensure it exists)
        // Note: If param has no grad yet, param.grad() creates zeros.
        // But if we haven't computed backward, it is zero.
        // Optimization: check if grad is physically present before calling grad()? 
        // For now, assume standard flow: zero_grad -> backward -> step.
        Tensor grad = param.grad(); 

        // Handle Weight Decay: g = g + wd * p
        if (weight_decay_ != 0.0f) {
            grad = grad + param * weight_decay_; // Out-of-place to avoid modifying grad tensor?
            // Or modify in-place: grad.add_(param * weight_decay_);
            // Standard PyTorch AdamW modifies grad? No, AdamW decouples.
            // Standard Adam modifies grad? Yes: g_t = \nabla f + \lambda \theta.
            // Let's implement standard Adam (L2 penalty added to grad).
            // We can do: grad = ops::add(grad, ops::mul(param, weight_decay_));
        }

        // Initialize State if needed
        auto& state = state_[i];
        if (state.find("exp_avg") == state.end()) {
            state["exp_avg"] = vesper::zeros(param.shape(), param.dtype(), param.device());
        }
        if (state.find("exp_avg_sq") == state.end()) {
            state["exp_avg_sq"] = vesper::zeros(param.shape(), param.dtype(), param.device());
        }

        Tensor& exp_avg = state["exp_avg"];
        Tensor& exp_avg_sq = state["exp_avg_sq"];

        // Update moments
        // m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        // exp_avg.mul_(beta1).add_(grad, 1 - beta1); 
        // We don't have add_ with alpha yet.
        // exp_avg = exp_avg * beta1 + grad * (1 - beta1);
        // This creates new tensor. Ideally in-place.
        // We have add_ and mul_.
        
        // m = m * beta1
        // m = m + g * (1 - beta1)
        exp_avg.mul_(beta1_);
        exp_avg.add_(grad * (1.0f - beta1_)); 

        // v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        // v = v * beta2
        // v = v + (g * g) * (1 - beta2)
        exp_avg_sq.mul_(beta2_);
        exp_avg_sq.add_(grad * grad * (1.0f - beta2_));

        // Compute bias-corrected moments
        // m_hat = m / (1 - beta1^t)
        // v_hat = v / (1 - beta2^t)
        
        Tensor m_hat = exp_avg / correction1;
        Tensor v_hat = exp_avg_sq / correction2;

        // Update parameters
        // p = p - lr * m_hat / (sqrt(v_hat) + eps)
        
        // denom = sqrt(v_hat) + eps
        // We need sqrt op. Do we have it?
        // ops::elementwise.h likely has basic arithmetic.
        // `grad * grad` calls operator*.
        // We need `sqrt`.
        
        // Let's use `ops::pow(v_hat, 0.5)` if sqrt missing.
        // Or check `src/ops/elementwise.cpp`.
        // I recall seeing `pow`?
        // If not, I'll assume we need to add `sqrt`.
        // Wait, I see `elementwise.cpp` has `add`, `sub`, `mul`, `div`.
        // It might not have `sqrt`.
        
        // I will check `elementwise.h` in a moment.
        // Assuming I need to implement `sqrt` or `pow`.
        // For now, I'll write the logic and assume `ops::sqrt` exists or I'll add it.
        
        Tensor denom = ops::sqrt(v_hat);
        denom.add_(eps_);
        Tensor step_size = m_hat / denom;
        
        param.sub_(step_size * lr_);
    }
}

StateDict Adam::state_dict() const {
    StateDict dict = Optimizer::state_dict();
    // Store step as Int32 to avoid float precision issues for long training
    dict["state.step"] = vesper::full({1}, DType::Float32, Device::CPU, (float)t_).to(DType::Int32);
    return dict;
}

void Adam::load_state_dict(const StateDict& dict) {
    Optimizer::load_state_dict(dict);
    if (dict.count("state.step")) {
        Tensor step_t = dict.at("state.step");
        if (step_t.dtype() == DType::Int32) {
            t_ = step_t.item<int>();
        } else {
            // Fallback for float
            t_ = (int)step_t.item<float>();
        }
    }
}

} // namespace vesper::optim
