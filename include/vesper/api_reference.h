/**
 * @file api_reference.h
 * @brief Vesper API Quick Reference
 * 
 * This file provides a comprehensive overview of the Vesper Deep Learning Library API.
 * It serves as both documentation and a quick reference for common operations.
 * 
 * Include this file for documentation only - use vesper.h for actual code.
 * 
 * ============================================================================
 * TABLE OF CONTENTS
 * ============================================================================
 * 
 * 1. TENSOR CREATION (factories.h)
 * 2. TENSOR OPERATIONS (tensor.h, ops/*.h)
 * 3. NEURAL NETWORK LAYERS (nn/*.h)
 * 4. FUNCTIONAL API (nn/functional.h)
 * 5. LOSS FUNCTIONS (nn/functional.h, nn/loss.h)
 * 6. OPTIMIZERS (optim/*.h)
 * 7. AUTOGRAD (autograd/*.h)
 * 8. DEVICE MANAGEMENT (core/device.h)
 * 9. DATA TYPES (core/dtype.h)
 * 10. SERIALIZATION (io/io.h)
 * 
 * ============================================================================
 * 1. TENSOR CREATION
 * ============================================================================
 * 
 * All factory functions are in namespace vesper:
 * 
 *   // Uninitialized tensor (fastest, use when you'll fill it immediately)
 *   Tensor empty(shape, dtype, device, requires_grad=false);
 *   
 *   // Initialized tensors
 *   Tensor zeros(shape, dtype, device, requires_grad=false);
 *   Tensor ones(shape, dtype, device, requires_grad=false);
 *   Tensor full(shape, dtype, device, value, requires_grad=false);
 *   Tensor randn(shape, dtype, device, requires_grad=false);  // Normal(0,1)
 * 
 * Example:
 *   Tensor x = zeros({32, 128}, DType::Float32, Device::HIP);
 *   Tensor w = randn({128, 64}, DType::Float32, Device::HIP, true);  // requires grad
 * 
 * ============================================================================
 * 2. TENSOR OPERATIONS
 * ============================================================================
 * 
 * --- Shape Information ---
 *   tensor.shape()           -> std::vector<int64_t>
 *   tensor.strides()         -> std::vector<int64_t>
 *   tensor.ndim()            -> int64_t
 *   tensor.numel()           -> size_t (total elements)
 *   tensor.dtype()           -> DType
 *   tensor.device()          -> Device
 *   tensor.is_contiguous()   -> bool
 *   tensor.defined()         -> bool (has valid storage)
 * 
 * --- Reshape Operations (return views when possible) ---
 *   tensor.view({new_shape})      // Must be contiguous, same numel
 *   tensor.reshape({new_shape})   // Like view but copies if needed
 *   tensor.transpose(dim0, dim1)  // Swap two dimensions
 *   tensor.permute({dims})        // Arbitrary dimension reordering
 *   tensor.contiguous()           // Make contiguous copy if needed
 * 
 * --- Slicing ---
 *   tensor.slice(dim, start, end)     // Slice along dimension [start:end]
 *   tensor.slice(index)               // Legacy: slice first dim at index
 *   tensor.index({selectors...})      // Advanced indexing
 * 
 *   NOTE: slice() returns a view (may not be contiguous).
 *         Call .contiguous() if needed for operations requiring contiguous data.
 * 
 * --- Device Transfer ---
 *   tensor.to(Device::HIP)       // Returns copy on new device
 *   tensor.to(Device::CPU)       // Returns copy on CPU
 *   tensor.to_(device)           // In-place device transfer
 * 
 * --- Type Casting ---
 *   tensor.to(DType::Float16)    // Returns copy with new dtype
 * 
 * --- Data Access ---
 *   tensor.data_ptr<float>()     // Raw pointer (use carefully!)
 *   tensor.copy_from_host(ptr)   // Copy from CPU buffer
 *   tensor.copy_to_host(ptr)     // Copy to CPU buffer
 *   tensor.item<float>()         // Get scalar value (numel must be 1)
 *   tensor.clone()               // Deep copy
 * 
 * --- In-Place Operations ---
 *   tensor.add_(other)           // tensor += other
 *   tensor.add_(scalar)          // tensor += scalar
 *   tensor.sub_(other)           // tensor -= other
 *   tensor.mul_(scalar)          // tensor *= scalar
 *   tensor.copy_(src)            // Copy data from src
 *   tensor.zero_()               // Fill with zeros
 * 
 * --- Arithmetic Operators ---
 *   a + b, a - b, a * b, a / b   // Element-wise (with broadcasting)
 *   a + 1.0f, 2.0f * a           // Scalar operations
 * 
 * --- Matrix Operations (ops namespace) ---
 *   ops::matmul(a, b)            // Matrix multiplication
 *   ops::bmm(a, b)               // Batched matrix multiply
 * 
 * --- Reduction Operations (ops namespace) ---
 *   ops::sum(tensor)             // Sum all elements -> scalar
 *   ops::sum(tensor, dim)        // Sum along dimension
 *   ops::mean(tensor)            // Mean of all elements
 *   ops::mean(tensor, dim)       // Mean along dimension
 *   ops::max(tensor, dim)        // Max along dimension
 *   ops::min(tensor, dim)        // Min along dimension
 * 
 * --- Other Operations (ops namespace) ---
 *   ops::exp(tensor)             // Element-wise exp
 *   ops::log(tensor)             // Element-wise log
 *   ops::sqrt(tensor)            // Element-wise sqrt
 *   ops::pow(tensor, exp)        // Element-wise power
 *   ops::cat({t1, t2}, dim)      // Concatenate tensors
 *   ops::stack({t1, t2}, dim)    // Stack tensors (new dim)
 * 
 * ============================================================================
 * 3. NEURAL NETWORK LAYERS
 * ============================================================================
 * 
 * All layers are in namespace vesper::nn:
 * 
 * --- Basic Layers ---
 *   Linear(in_features, out_features, bias=true)
 *     - Input:  [*, in_features]
 *     - Output: [*, out_features]
 * 
 *   Embedding(num_embeddings, embedding_dim)
 *     - Input:  [*] int32/int64 indices
 *     - Output: [*, embedding_dim]
 * 
 *   Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
 *     - Input:  [N, C_in, H, W]
 *     - Output: [N, C_out, H_out, W_out]
 * 
 * --- Normalization ---
 *   LayerNorm(normalized_shape, eps=1e-5)
 *     - normalized_shape: usually {hidden_dim}
 *     - Input:  [*, hidden_dim]
 *     - Output: [*, hidden_dim]
 * 
 *   RMSNorm(hidden_size, eps=1e-5)
 *     - Input:  [*, hidden_size]
 *     - Output: [*, hidden_size]
 * 
 * --- Pooling ---
 *   MaxPool2d(kernel_size, stride=kernel_size, padding=0)
 *   AvgPool2d(kernel_size, stride=kernel_size, padding=0)
 * 
 * --- Activations (as modules) ---
 *   ReLU(), GELU(), SiLU(), Sigmoid(), Tanh()
 * 
 * --- Dropout ---
 *   Dropout(p=0.5)  // Set module.training = false for inference
 * 
 * --- Transformer Components ---
 *   MultiHeadAttention(embed_dim, num_heads, dropout=0.0)
 *     - Input:  query [B, T, D], key [B, S, D], value [B, S, D]
 *     - Output: [B, T, D]
 *   
 *   GQAttention(hidden_size, num_heads, num_kv_heads, max_seq_len)
 *     - Grouped-Query Attention with RoPE
 *     - Input:  x [B, T, D], start_pos
 *     - Output: [B, T, D]
 *   
 *   RoPE(dim, max_seq_len, theta=10000.0)
 *     - Input:  x [B, H, T, head_dim], start_pos
 *     - Output: [B, H, T, head_dim]
 *   
 *   SwiGLU(hidden_size, intermediate_size)
 *     - Input:  [B, T, hidden_size]
 *     - Output: [B, T, hidden_size]
 * 
 * --- Container Modules ---
 *   ModuleList  // Container for list of modules
 *     - modules.push_back(module_ptr)
 *     - modules[i]->forward(x)
 * 
 * --- Module Usage Pattern ---
 *   class MyModel : public nn::Module {
 *       nn::Linear fc1{128, 64};
 *       nn::Linear fc2{64, 10};
 *   public:
 *       MyModel() { 
 *           register_module("fc1", fc1);
 *           register_module("fc2", fc2);
 *       }
 *       Tensor forward(const Tensor& x) {
 *           auto h = nn::functional::relu(fc1.forward(x));
 *           return fc2.forward(h);
 *       }
 *   };
 * 
 *   // Get all parameters for optimizer
 *   auto params = model.parameters();
 * 
 * ============================================================================
 * 4. FUNCTIONAL API
 * ============================================================================
 * 
 * All functions are in namespace vesper::nn::functional:
 * 
 * --- Activations ---
 *   relu(x)           // max(0, x)
 *   sigmoid(x)        // 1 / (1 + exp(-x))
 *   gelu(x)           // Gaussian Error Linear Unit
 *   silu(x)           // x * sigmoid(x), aka Swish
 *   softmax(x, dim)   // exp(x) / sum(exp(x)) along dim
 *   log_softmax(x, dim)
 * 
 * --- Normalization ---
 *   layer_norm(x, normalized_shape, weight, bias, eps)
 *   rms_norm(x, normalized_shape, weight, eps)
 * 
 * --- Dropout ---
 *   dropout(x, p=0.5, training=true)
 * 
 * --- Attention ---
 *   scaled_dot_product_attention(Q, K, V, is_causal=false, dropout_p=0.0)
 *     - Q: [B, H, T, D] or [B, T, D]
 *     - K: [B, H, S, D] or [B, S, D]
 *     - V: [B, H, S, D] or [B, S, D]
 *     - Returns: [B, H, T, D] or [B, T, D]
 *     - is_causal: applies causal mask (for autoregressive models)
 * 
 * ============================================================================
 * 5. LOSS FUNCTIONS
 * ============================================================================
 * 
 * In namespace vesper::nn::functional:
 * 
 *   mse_loss(predictions, targets)
 *     - Both tensors same shape
 *     - Returns: scalar mean squared error
 * 
 *   cross_entropy_loss(logits, targets)
 *     - logits:  [N, C] raw scores (before softmax)
 *     - targets: [N] integer class labels (0 to C-1), Int32 or Int64
 *     - Returns: scalar cross-entropy loss
 *     - NOTE: Applies log_softmax internally, do NOT apply softmax first!
 * 
 * In namespace vesper::nn:
 * 
 *   MSELoss (module version)
 *   CrossEntropyLoss (module version)
 * 
 * ============================================================================
 * 6. OPTIMIZERS
 * ============================================================================
 * 
 * All optimizers are in namespace vesper::optim:
 * 
 *   SGD(parameters, lr, momentum=0, weight_decay=0, nesterov=false)
 *   Adam(parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0)
 *   AdamW(parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01)
 * 
 * Usage:
 *   auto optimizer = optim::Adam(model.parameters(), 0.001);
 *   
 *   for (epoch...) {
 *       optimizer.zero_grad();              // Clear gradients
 *       Tensor loss = compute_loss(...);
 *       loss.backward();                    // Compute gradients
 *       optimizer.step();                   // Update parameters
 *   }
 * 
 * --- Learning Rate Schedulers ---
 *   StepLR(optimizer, step_size, gamma=0.1)
 *   ExponentialLR(optimizer, gamma)
 *   CosineAnnealingLR(optimizer, T_max, eta_min=0)
 * 
 * ============================================================================
 * 7. AUTOGRAD
 * ============================================================================
 * 
 * --- Enabling Gradients ---
 *   Tensor x = randn({...}, DType::Float32, device, true);  // requires_grad=true
 *   // OR
 *   x.set_requires_grad(true);
 * 
 * --- Backward Pass ---
 *   loss.backward();           // Compute gradients (loss must be scalar)
 *   loss.backward(grad);       // With custom gradient
 * 
 * --- Accessing Gradients ---
 *   Tensor& g = param.grad();  // Get gradient tensor
 * 
 * --- Disabling Gradients (for inference) ---
 *   {
 *       autograd::NoGradGuard guard;
 *       // Operations here won't track gradients
 *       Tensor out = model.forward(x);
 *   }
 * 
 * --- Which operations support autograd? ---
 *   Most operations have backward implementations:
 *   - Arithmetic: +, -, *, /
 *   - Matrix: matmul, bmm
 *   - Activations: relu, sigmoid, gelu, silu, softmax
 *   - Reductions: sum, mean
 *   - Normalization: layer_norm, rms_norm
 *   - All nn::Module layers
 * 
 * ============================================================================
 * 8. DEVICE MANAGEMENT
 * ============================================================================
 * 
 *   Device::CPU    // CPU
 *   Device::HIP    // AMD GPU (ROCm)
 *   Device::CUDA   // NVIDIA GPU
 * 
 *   // Check available backend
 *   #ifdef VESPER_HIP
 *       Device device = Device::HIP;
 *   #elif defined(VESPER_CUDA)
 *       Device device = Device::CUDA;
 *   #else
 *       Device device = Device::CPU;
 *   #endif
 * 
 * ============================================================================
 * 9. DATA TYPES
 * ============================================================================
 * 
 *   DType::Float32   // 32-bit float (default for most operations)
 *   DType::Float16   // 16-bit float (for memory efficiency)
 *   DType::Int32     // 32-bit integer
 *   DType::Int64     // 64-bit integer
 *   DType::Bool      // Boolean
 * 
 *   // Type casting
 *   Tensor x_fp16 = x.to(DType::Float16);
 * 
 * ============================================================================
 * 10. SERIALIZATION
 * ============================================================================
 * 
 * In namespace vesper::io:
 * 
 *   // Save/load single tensor
 *   save_tensor(tensor, "weights.bin");
 *   Tensor t = load_tensor("weights.bin");
 * 
 *   // Save/load model state
 *   auto state = model.state_dict();
 *   save_state_dict(state, "model.bin");
 *   
 *   auto loaded = load_state_dict("model.bin");
 *   model.load_state_dict(loaded);
 * 
 * ============================================================================
 * COMMON PATTERNS
 * ============================================================================
 * 
 * --- Basic Training Loop ---
 * 
 *   auto optimizer = optim::Adam(model.parameters(), 1e-3);
 *   
 *   for (int epoch = 0; epoch < num_epochs; ++epoch) {
 *       for (auto& [inputs, targets] : dataloader) {
 *           optimizer.zero_grad();
 *           
 *           Tensor outputs = model.forward(inputs);
 *           Tensor loss = nn::functional::cross_entropy_loss(outputs, targets);
 *           
 *           loss.backward();
 *           optimizer.step();
 *       }
 *   }
 * 
 * --- Inference (no gradients) ---
 * 
 *   model.eval();  // Set training=false for dropout, etc.
 *   {
 *       autograd::NoGradGuard guard;
 *       Tensor output = model.forward(input);
 *   }
 * 
 * --- LLM Generation Pattern ---
 * 
 *   // For autoregressive generation, extract last token's logits:
 *   Tensor logits = model.forward(tokens);      // [1, seq_len, vocab_size]
 *   int64_t V = logits.shape()[2];
 *   
 *   // Flatten and slice (workaround for slice on higher dims)
 *   Tensor flat = logits.view({seq_len, V});
 *   Tensor last = flat.slice(0, seq_len-1, seq_len).contiguous();  // [1, V]
 *   
 *   Tensor probs = nn::functional::softmax(last, -1);
 * 
 * ============================================================================
 */

#pragma once

// This is a documentation-only header.
// For actual usage, include <vesper/vesper.h>

#include <vesper/vesper.h>
