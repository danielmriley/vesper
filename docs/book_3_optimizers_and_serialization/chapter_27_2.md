# Chapter 27.2: The Lion Optimizer (`optim.Lion`)

## 1. Introduction

**Lion (Evolved Sign Momentum)** is a modern optimizer discovered by Google DeepMind in 2023 using symbolic program search. It is simpler and more memory-efficient than Adam, often matching or exceeding its performance on large transformers.

## 2. Comparison: Lion vs Adam

| Feature | Adam | Lion |
| :--- | :--- | :--- |
| **State Memory** | 2 buffers ($m, v$) | 1 buffer ($m$) |
| **Update Magnitude** | Adaptive (scaled by variance) | Uniform (sign-based) |
| **Hyperparameters** | $\beta_1=0.9, \beta_2=0.999$ | $\beta_1=0.9, \beta_2=0.99$ |
| **Batch Size Sensitivity** | Lower | Higher (needs larger batch) |

## 3. Algorithm

Lion tracks only the momentum $m_t$.

### Update Rules
At timestep $t$:
1.  **Get Gradient**: $g_t = \nabla_\theta J(\theta_{t-1})$
2.  **Compute Update Direction**:
    $$ c_t = \text{sign}(\beta_1 m_{t-1} + (1 - \beta_1) g_t) $$
3.  **Update Parameter**:
    $$ \theta_t = \theta_{t-1} - \eta (c_t + \lambda \theta_{t-1}) $$
    *(Note: $\lambda$ is decoupled weight decay)*
4.  **Update Momentum**:
    $$ m_t = \beta_2 m_{t-1} + (1 - \beta_2) g_t $$

## 4. Implementation Plan

We will create `vesper::optim::Lion`.

### Required Operations
We need a `sign()` element-wise operation.
$$ \text{sign}(x) = \begin{cases} 1 & \text{if } x \ge 0 \\ -1 & \text{if } x < 0 \end{cases} $$

### Class Structure
```cpp
class Lion : public Optimizer {
public:
    Lion(std::vector<Tensor*> params, float lr=1e-4, float beta1=0.9, float beta2=0.99, float weight_decay=0.0);
    
    void step() override;

private:
    // Only one state buffer per parameter!
    std::vector<Tensor> m_; 
};
```

## 5. Usage Example

```cpp
// Lion typically uses a lower learning rate than Adam (e.g., 3x-10x smaller)
auto optimizer = vesper::optim::Lion(model.parameters(), 1e-4, 0.9, 0.99, 0.01);
```

## 6. Testing Strategy

1.  **Memory Usage**: Verify that a Lion optimizer uses ~50% less auxiliary memory than Adam for the same model.
2.  **Sign Logic**: Ensure the update step moves parameters by a fixed magnitude (ignoring weight decay) determined by `lr`.
