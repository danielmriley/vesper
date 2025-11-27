#pragma once

#include <vesper/core/tensor.h>
#include <vesper/nn/module.h>
#include <vesper/optim/optimizer.h>
#include <unordered_map>
#include <memory>

namespace vesper::nn {

/**
 * @brief Automatic Mixed Precision (AMP) helper for training
 * 
 * AMP enables training in half precision (FP16/BF16) while maintaining
 * numerical stability by:
 * 1. Keeping master weights in FP32
 * 2. Running forward/backward in half precision
 * 3. Updating master weights in FP32
 * 
 * Usage:
 * ```cpp
 * AMP amp(DType::BFloat16);
 * amp.init_master_weights(model);
 * 
 * // Training loop
 * Tensor logits = amp.forward(model, input);
 * Tensor loss = compute_loss(logits, labels);
 * loss.backward();
 * amp.step(optimizer);
 * ```
 */
class AMP {
public:
    /**
     * @brief Construct AMP wrapper
     * @param compute_dtype Target dtype for forward/backward (Float16 or BFloat16)
     */
    explicit AMP(DType compute_dtype = DType::BFloat16);
    
    /**
     * @brief Get the compute dtype
     */
    DType compute_dtype() const { return compute_dtype_; }
    
    /**
     * @brief Initialize master weights from model parameters
     * 
     * Creates FP32 copies of all parameters and casts the model
     * parameters to half precision.
     */
    void init_master_weights(Module& model);
    
    /**
     * @brief Run forward pass with automatic casting
     * 
     * Casts input to compute_dtype and runs forward.
     */
    Tensor forward(Module& model, const Tensor& input);
    
    /**
     * @brief Synchronize gradients from half to FP32 master weights
     */
    void sync_gradients();
    
    /**
     * @brief Update master weights and copy back to half
     * @param optimizer The optimizer to use for the update
     */
    void step(optim::Optimizer& optimizer);
    
    /**
     * @brief Check if AMP is enabled
     */
    bool enabled() const { return enabled_; }
    
    /**
     * @brief Enable or disable AMP
     */
    void set_enabled(bool enabled) { enabled_ = enabled; }
    
private:
    DType compute_dtype_;
    bool enabled_ = true;
    
    // Map from half-precision parameter pointer to FP32 master weight
    struct MasterWeight {
        Tensor master;      // FP32 master weight
        Tensor* half_param; // Pointer to half-precision parameter in model
    };
    std::vector<MasterWeight> master_weights_;
};

/**
 * @brief RAII context for automatic tensor casting
 * 
 * When enabled, tensor operations within this context will automatically
 * cast inputs to the specified dtype.
 * 
 * Usage:
 * ```cpp
 * {
 *     AutocastContext ctx(DType::BFloat16);
 *     Tensor y = model.forward(x);  // x automatically cast to BF16
 * }
 * ```
 */
class AutocastContext {
public:
    explicit AutocastContext(DType dtype);
    ~AutocastContext();
    
    /// Get the current autocast dtype (or Float32 if disabled)
    static DType current_dtype();
    
    /// Check if autocast is currently enabled
    static bool is_enabled();
    
private:
    DType prev_dtype_;
    bool prev_enabled_;
    
    static thread_local DType current_dtype_;
    static thread_local bool enabled_;
};

} // namespace vesper::nn
