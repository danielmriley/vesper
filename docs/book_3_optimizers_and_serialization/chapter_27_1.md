
# Vesper Future Plans - Chapter 25: The Adam Optimizer (`optim.Adam`)

## 1. Introduction

Stochastic Gradient Descent (SGD) is simple and effective, but often slow to converge. **Adam (Adaptive Moment Estimation)** is a stateful optimizer that adapts the learning rate for each parameter individually. It combines the advantages of AdaGrad (adaptive learning rates) and RMSProp (momentum).

## 2. Algorithm

Adam maintains two state tensors for each model parameter $\theta$:
1.  $m_t$: Exponential moving average of gradients (1st moment).
2.  $v_t$: Exponential moving average of squared gradients (2nd moment).

### Update Rules
At timestep $t$:
1.  **Get Gradient**: $g_t = \nabla_\theta J(\theta_{t-1})$
2.  **Update Moments**:
    $$ m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t $$
    $$ v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 $$
3.  **Bias Correction**:
    $$ \hat{m}_t = \frac{m_t}{1 - \beta_1^t} $$
    $$ \hat{v}_t = \frac{v_t}{1 - \beta_2^t} $$
4.  **Update Parameter**:
    $$ \theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} $$

Where:
- $\eta$: Learning rate
- $\beta_1, \beta_2$: Decay rates (typically 0.9 and 0.999)
- $\epsilon$: Small constant for stability (typically $10^{-8}$)

## 3. Implementation Plan

We will create `vesper::optim::Adam` inheriting from `Optimizer`.

### Class Structure
```cpp
class Adam : public Optimizer {
public:
    Adam(std::vector<Tensor*> params, float lr=1e-3, float beta1=0.9, float beta2=0.999, float eps=1e-8, float weight_decay=0.0);
    
    void step() override;

private:
    float lr_, beta1_, beta2_, eps_, weight_decay_;
    int t_ = 0; // Timestep
    
    // State buffers
    std::vector<Tensor> m_;
    std::vector<Tensor> v_;
};
```

### Key Implementation Details
- **Initialization**: On the first `step()`, initialize `m_` and `v_` as zeros with the same shape/device as parameters.
- **Weight Decay**: Can be implemented as L2 regularization (add to gradient) or AdamW style (decay weights directly). We will implement standard L2 regularization first: $g_t = g_t + \lambda \theta_{t-1}$.
- **Efficiency**: Use in-place operations (`add_`, `mul_`) to minimize memory allocation overhead.

## 4. Usage Example

```cpp
// Create model and optimizer
auto model = MyModel();
auto optimizer = vesper::optim::Adam(model.parameters(), 0.001);

// Training loop
for (int epoch = 0; epoch < 10; ++epoch) {
    // ... forward pass ...
    loss.backward();
    
    optimizer.step();
    optimizer.zero_grad();
}
```

## 5. Testing Strategy

1.  **Convergence Test**: Train a simple quadratic function $f(x) = x^2$ and verify $x$ converges to 0 faster than SGD.
2.  **Reference Check**: Compare step-by-step updates against a PyTorch reference implementation for a fixed sequence of gradients.
