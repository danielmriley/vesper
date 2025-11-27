#include <vesper/optim/grad_scaler.h>
#include <vesper/ops/elementwise.h>
#include <vesper/ops/reduction.h>
#include <vesper/core/factories.h>
#include <cmath>
#include <limits>

namespace vesper::optim {

GradScaler::GradScaler(float init_scale, float growth_factor, 
                       float backoff_factor, int growth_interval)
    : scale_(init_scale)
    , growth_factor_(growth_factor)
    , backoff_factor_(backoff_factor)
    , growth_interval_(growth_interval)
{}

Tensor GradScaler::scale(const Tensor& loss) {
    // Multiply loss by scale factor
    return ops::mul(loss, scale_);
}

bool GradScaler::unscale(Optimizer& optimizer) {
    found_inf_ = false;
    gradients_unscaled_ = true;
    
    float inv_scale = 1.0f / scale_;
    
    // Iterate over all parameters and unscale their gradients
    for (Tensor& param : optimizer.params()) {
        if (param.grad().defined()) {
            Tensor grad = param.grad();
            
            // Make contiguous for cross-device transfer
            if (!grad.is_contiguous()) {
                grad = grad.contiguous();
            }
            
            // Check for inf/nan before unscaling
            // We need to do this on CPU for now
            Tensor grad_cpu = grad.to(Device::CPU);
            const float* data = grad_cpu.data_ptr<float>();
            size_t n = grad_cpu.numel();
            
            for (size_t i = 0; i < n; ++i) {
                if (!std::isfinite(data[i])) {
                    found_inf_ = true;
                    break;
                }
            }
            
            if (found_inf_) {
                break;
            }
            
            // Unscale: grad = grad / scale
            // Use the original grad (may be on GPU) for multiplication
            Tensor unscaled_grad = ops::mul(param.grad(), inv_scale);
            
            // Ensure same device - ops::mul might return CPU tensor for scalar
            if (unscaled_grad.device() != param.grad().device()) {
                unscaled_grad = unscaled_grad.to(param.grad().device());
            }
            
            // Make contiguous if needed
            if (!unscaled_grad.is_contiguous()) {
                unscaled_grad = unscaled_grad.contiguous();
            }
            
            // Simply replace the gradient with the unscaled version
            param.grad() = unscaled_grad;
        }
        
        if (found_inf_) {
            break;
        }
    }
    
    if (found_inf_) {
        inf_count_++;
    }
    
    return found_inf_;
}

bool GradScaler::step(Optimizer& optimizer) {
    // Ensure gradients have been unscaled
    if (!gradients_unscaled_) {
        unscale(optimizer);
    }
    
    // Skip step if inf/nan was found
    if (found_inf_) {
        // Zero gradients since step was skipped
        optimizer.zero_grad();
        return false;
    }
    
    // Perform optimizer step
    optimizer.step();
    return true;
}

void GradScaler::update() {
    // Reset unscale flag for next iteration
    gradients_unscaled_ = false;
    
    if (found_inf_) {
        // Decrease scale on inf/nan
        scale_ *= backoff_factor_;
        steps_since_growth_ = 0;
    } else {
        // Potentially increase scale
        steps_since_growth_++;
        if (steps_since_growth_ >= growth_interval_) {
            scale_ *= growth_factor_;
            steps_since_growth_ = 0;
            
            // Cap at max float16 range to avoid immediate overflow
            constexpr float max_scale = 65504.0f;
            if (scale_ > max_scale) {
                scale_ = max_scale;
            }
        }
    }
    
    // Reset found_inf for next iteration
    found_inf_ = false;
}

} // namespace vesper::optim
