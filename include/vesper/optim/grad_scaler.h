#pragma once

#include <vesper/core/tensor.h>
#include <vesper/optim/optimizer.h>
#include <vector>

namespace vesper::optim {

/**
 * @brief Gradient scaler for FP16 mixed precision training
 * 
 * FP16 has limited range (±65504) which can cause gradient underflow.
 * GradScaler multiplies the loss by a scale factor before backward,
 * then unscales gradients before optimizer step.
 * 
 * The scale is dynamically adjusted:
 * - Decreased when inf/nan gradients are detected
 * - Increased periodically when training is stable
 * 
 * Note: BFloat16 typically doesn't need loss scaling due to its
 * wider dynamic range (same as FP32).
 * 
 * Usage:
 * ```cpp
 * GradScaler scaler;
 * 
 * // Training loop
 * Tensor loss = compute_loss(...);
 * Tensor scaled_loss = scaler.scale(loss);
 * scaled_loss.backward();
 * 
 * scaler.unscale(optimizer);
 * if (scaler.step(optimizer)) {
 *     // Step was successful
 * }
 * scaler.update();
 * ```
 */
class GradScaler {
public:
    /**
     * @brief Construct a GradScaler
     * @param init_scale Initial scale factor (default: 65536.0 = 2^16)
     * @param growth_factor Factor to multiply scale by on successful steps
     * @param backoff_factor Factor to multiply scale by on inf/nan detection
     * @param growth_interval Steps between scale increases
     */
    GradScaler(float init_scale = 65536.0f,
               float growth_factor = 2.0f,
               float backoff_factor = 0.5f,
               int growth_interval = 2000);
    
    /**
     * @brief Scale the loss before backward pass
     * @param loss The loss tensor
     * @return Scaled loss tensor
     */
    Tensor scale(const Tensor& loss);
    
    /**
     * @brief Unscale gradients after backward pass
     * 
     * Divides all gradients by the current scale factor.
     * Also checks for inf/nan and sets internal flag.
     * 
     * @param optimizer The optimizer containing parameters to unscale
     * @return true if inf/nan was found, false otherwise
     */
    bool unscale(Optimizer& optimizer);
    
    /**
     * @brief Perform optimizer step if gradients are valid
     * 
     * Skips the step if inf/nan gradients were detected.
     * 
     * @param optimizer The optimizer to step
     * @return true if step was performed, false if skipped
     */
    bool step(Optimizer& optimizer);
    
    /**
     * @brief Update the scale factor based on gradient health
     * 
     * Call this after step() each iteration.
     */
    void update();
    
    /**
     * @brief Get current scale factor
     */
    float get_scale() const { return scale_; }
    
    /**
     * @brief Check if gradients are finite (after unscale)
     */
    bool gradients_finite() const { return !found_inf_; }
    
    /**
     * @brief Get number of inf/nan detections since last reset
     */
    int inf_count() const { return inf_count_; }
    
private:
    float scale_;
    float growth_factor_;
    float backoff_factor_;
    int growth_interval_;
    int steps_since_growth_ = 0;
    bool found_inf_ = false;
    int inf_count_ = 0;
    bool gradients_unscaled_ = false;
};

} // namespace vesper::optim
