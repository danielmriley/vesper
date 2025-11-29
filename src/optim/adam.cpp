#include <vesper/optim/adam.h>
#include <vesper/core/factories.h>
#include <vesper/ops/elementwise.h> // For basic ops and fused ops
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
    float correction1 = 1.0f - std::pow(beta1_, t_);
    float correction2 = 1.0f - std::pow(beta2_, t_);

    for (size_t i = 0; i < params_.size(); ++i) {
        auto& param = params_[i];
        if (!param.requires_grad()) continue;

        Tensor grad = param.grad(); 

        // AdamW: Decoupled weight decay (applied directly to params, not gradients)
        // NOTE: This modifies param in-place. Users with tied weights should be aware.
        if (weight_decay_ != 0.0f) {
            param.sub_(param * (lr_ * weight_decay_));
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

        // Update moments using fused operations to avoid temporary allocations
        // m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        // Using lerp_: m = m * (1 - weight) + g * weight, where weight = (1 - beta1)
        ops::lerp_(exp_avg, grad, 1.0f - beta1_);

        // v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        // Two-step: v *= beta2, then v += g*g*(1-beta2)
        exp_avg_sq.mul_(beta2_);
        ops::addcmul_(exp_avg_sq, grad, grad, 1.0f - beta2_);

        // Compute bias-corrected moments
        Tensor m_hat = exp_avg / correction1;
        Tensor v_hat = exp_avg_sq / correction2;

        // Update parameters: p = p - lr * m_hat / (sqrt(v_hat) + eps)
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
