#include <vesper/nn/amp.h>
#include <vesper/ops/cast.h>
#include <vesper/core/factories.h>
#include <vesper/autograd/guard.h>
#include <stdexcept>

namespace vesper::nn {

// Thread-local storage for autocast context
thread_local DType AutocastContext::current_dtype_ = DType::Float32;
thread_local bool AutocastContext::enabled_ = false;

AMP::AMP(DType compute_dtype) : compute_dtype_(compute_dtype) {
    if (compute_dtype != DType::Float16 && compute_dtype != DType::BFloat16) {
        throw std::runtime_error("AMP compute_dtype must be Float16 or BFloat16");
    }
}

void AMP::init_master_weights(Module& model) {
    master_weights_.clear();
    
    auto params = model.parameters_ptrs();
    for (Tensor* param : params) {
        if (param->dtype() == DType::Float32) {
            // Create FP32 master weight copy
            MasterWeight mw;
            mw.master = param->clone();  // Deep copy
            mw.half_param = param;
            
            // Cast the model parameter to half precision
            *param = ops::cast(*param, compute_dtype_);
            
            master_weights_.push_back(std::move(mw));
        }
    }
}

Tensor AMP::forward(Module& model, const Tensor& input) {
    if (!enabled_) {
        return model.forward(input);
    }
    
    // Cast input if needed
    Tensor x = input;
    if (x.dtype() != compute_dtype_) {
        x = ops::cast(x, compute_dtype_);
    }
    
    return model.forward(x);
}

void AMP::sync_gradients() {
    // Copy gradients from half-precision parameters to master weights
    for (auto& mw : master_weights_) {
        if (mw.half_param->grad().defined()) {
            // Cast gradient to FP32 and copy to master
            Tensor grad_fp32 = ops::cast(mw.half_param->grad(), DType::Float32);
            
            if (!mw.master.grad().defined()) {
                mw.master.grad() = grad_fp32;
            } else {
                mw.master.grad().copy_(grad_fp32);
            }
        }
    }
}

void AMP::step(optim::Optimizer& optimizer) {
    if (!enabled_) {
        optimizer.step();
        return;
    }
    
    // Sync gradients from half to master
    sync_gradients();
    
    // The optimizer should be configured to work on master weights
    // For now, we manually update master weights and copy back
    
    // Update master weights (optimizer step)
    optimizer.step();
    
    // Copy master weights back to half-precision parameters
    {
        autograd::NoGradGuard guard;
        for (auto& mw : master_weights_) {
            Tensor half_weight = ops::cast(mw.master, compute_dtype_);
            mw.half_param->copy_(half_weight);
        }
    }
}

// AutocastContext implementation
AutocastContext::AutocastContext(DType dtype) {
    prev_dtype_ = current_dtype_;
    prev_enabled_ = enabled_;
    current_dtype_ = dtype;
    enabled_ = true;
}

AutocastContext::~AutocastContext() {
    current_dtype_ = prev_dtype_;
    enabled_ = prev_enabled_;
}

DType AutocastContext::current_dtype() {
    return enabled_ ? current_dtype_ : DType::Float32;
}

bool AutocastContext::is_enabled() {
    return enabled_;
}

} // namespace vesper::nn
