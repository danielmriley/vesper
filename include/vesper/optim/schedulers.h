#pragma once

#include <vesper/optim/optimizer.h>
#include <vector>

namespace vesper::optim {

class LRScheduler {
public:
    explicit LRScheduler(Optimizer& optimizer, int last_epoch = -1);
    virtual ~LRScheduler() = default;

    void step();
    virtual float get_lr() = 0; // Returns the NEW learning rate for the next step

protected:
    Optimizer& optimizer_;
    int last_epoch_;
    float base_lr_; // Initial LR (assumed same for all params for simplicity)
    // Note: PyTorch schedulers often handle param_groups. Vesper currently has global LR in Optimizer?
    // Optimizer class doesn't expose `lr` or `param_groups`. 
    // Subclasses like SGD/Adam have `lr_`.
    // We need a way to set LR on Optimizer.
    // I will add `set_lr(float)` to Optimizer.
};

class LinearLR : public LRScheduler {
public:
    LinearLR(Optimizer& optimizer, float start_factor = 1.0f/3.0f, float end_factor = 1.0f, int total_iters = 5, int last_epoch = -1);
    float get_lr() override;
private:
    float start_factor_;
    float end_factor_;
    int total_iters_;
};

class StepLR : public LRScheduler {
public:
    StepLR(Optimizer& optimizer, int step_size, float gamma = 0.1f, int last_epoch = -1);
    float get_lr() override;
private:
    int step_size_;
    float gamma_;
};

class CosineAnnealingLR : public LRScheduler {
public:
    CosineAnnealingLR(Optimizer& optimizer, int T_max, float eta_min = 0.0f, int last_epoch = -1);
    float get_lr() override;
private:
    int T_max_;
    float eta_min_;
};

/// Cosine annealing with linear warmup (commonly used for transformer training)
/// 
/// Learning rate schedule:
/// - Steps 0 to warmup_iters: Linear warmup from 0 to base_lr
/// - Steps warmup_iters to T_max: Cosine decay from base_lr to eta_min
/// 
/// Example for nanoGPT-style training:
/// ```cpp
/// CosineAnnealingWithWarmup scheduler(optimizer, 
///     /*T_max=*/5000,      // Total training iterations
///     /*warmup_iters=*/100, // Warmup steps  
///     /*eta_min=*/1e-4);   // Minimum LR (10% of max)
/// ```
class CosineAnnealingWithWarmup : public LRScheduler {
public:
    CosineAnnealingWithWarmup(Optimizer& optimizer, int T_max, int warmup_iters = 0, 
                              float eta_min = 0.0f, int last_epoch = -1);
    float get_lr() override;
private:
    int T_max_;
    int warmup_iters_;
    float eta_min_;
};

} // namespace vesper::optim
