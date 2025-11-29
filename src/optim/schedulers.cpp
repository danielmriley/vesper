#include <vesper/optim/schedulers.h>
#include <cmath>
#include <iostream>

namespace vesper::optim {

LRScheduler::LRScheduler(Optimizer& optimizer, int last_epoch)
    : optimizer_(optimizer), last_epoch_(last_epoch) {
    base_lr_ = optimizer_.get_lr();
}

void LRScheduler::step() {
    last_epoch_++;
    float new_lr = get_lr();
    optimizer_.set_lr(new_lr);
}

// --- LinearLR ---
LinearLR::LinearLR(Optimizer& optimizer, float start_factor, float end_factor, int total_iters, int last_epoch)
    : LRScheduler(optimizer, last_epoch), start_factor_(start_factor), end_factor_(end_factor), total_iters_(total_iters) {}

float LinearLR::get_lr() {
    if (last_epoch_ >= total_iters_) {
        return base_lr_ * end_factor_;
    }
    
    // Linearly interpolate
    float factor = start_factor_ + (end_factor_ - start_factor_) * (float(last_epoch_) / total_iters_);
    return base_lr_ * factor;
}

// --- StepLR ---
StepLR::StepLR(Optimizer& optimizer, int step_size, float gamma, int last_epoch)
    : LRScheduler(optimizer, last_epoch), step_size_(step_size), gamma_(gamma) {}

float StepLR::get_lr() {
    // Decay every step_size epochs
    // lr = base_lr * gamma ^ (epoch / step_size)
    // Note: This implementation calculates absolute LR based on base_lr.
    // PyTorch often implements recursive decay (lr = lr * gamma).
    // But absolute is stateless (easier).
    int steps = last_epoch_ / step_size_;
    return base_lr_ * std::pow(gamma_, steps);
}

// --- CosineAnnealingLR ---
CosineAnnealingLR::CosineAnnealingLR(Optimizer& optimizer, int T_max, float eta_min, int last_epoch)
    : LRScheduler(optimizer, last_epoch), T_max_(T_max), eta_min_(eta_min) {}

float CosineAnnealingLR::get_lr() {
    if (last_epoch_ >= T_max_) {
        return eta_min_;
    }
    // 0.5 * (base_lr - eta_min) * (1 + cos(pi * epoch / T_max)) + eta_min
    float cos_val = std::cos(M_PI * float(last_epoch_) / T_max_);
    return eta_min_ + 0.5f * (base_lr_ - eta_min_) * (1.0f + cos_val);
}

// --- CosineAnnealingWithWarmup ---
CosineAnnealingWithWarmup::CosineAnnealingWithWarmup(Optimizer& optimizer, int T_max, 
                                                     int warmup_iters, float eta_min, int last_epoch)
    : LRScheduler(optimizer, last_epoch), T_max_(T_max), warmup_iters_(warmup_iters), eta_min_(eta_min) {}

float CosineAnnealingWithWarmup::get_lr() {
    if (last_epoch_ < warmup_iters_) {
        // Linear warmup: 0 -> base_lr over warmup_iters steps
        return base_lr_ * float(last_epoch_ + 1) / float(warmup_iters_);
    }
    
    if (last_epoch_ >= T_max_) {
        return eta_min_;
    }
    
    // Cosine annealing after warmup
    // Adjust epoch to start cosine from 0 after warmup
    int decay_steps = T_max_ - warmup_iters_;
    int step_in_decay = last_epoch_ - warmup_iters_;
    
    float cos_val = std::cos(M_PI * float(step_in_decay) / float(decay_steps));
    return eta_min_ + 0.5f * (base_lr_ - eta_min_) * (1.0f + cos_val);
}

} // namespace vesper::optim
