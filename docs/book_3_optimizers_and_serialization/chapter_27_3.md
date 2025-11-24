# Chapter 27.3: Learning Rate Schedulers

## 1. Introduction

Training deep networks often requires changing the learning rate (LR) during training.
- **Warmup**: Starting with a small LR and increasing it helps stabilize early training (crucial for Transformers).
- **Decay**: Lowering the LR later in training helps the model settle into a sharper minimum.

## 2. Scheduler Types

We will implement a base `LRScheduler` and several concrete strategies.

### 1. StepLR
Decays the learning rate by a factor `gamma` every `step_size` epochs.
$$ \eta_{t} = \eta_{0} \cdot \gamma^{\lfloor t / \text{step\_size} \rfloor} $$

### 2. Linear Warmup
Increases LR linearly from 0 to `target_lr` over `warmup_steps`.
$$ \eta_t = \eta_{\text{target}} \cdot \frac{t}{\text{warmup\_steps}} $$

### 3. Cosine Annealing
Decreases LR following a cosine curve.
$$ \eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T_{\max}}\pi\right)\right) $$

## 3. Implementation Plan

### Base Class
The scheduler needs access to the optimizer to modify its learning rate.

```cpp
class LRScheduler {
public:
    explicit LRScheduler(Optimizer& optimizer);
    virtual void step() = 0; // Update the optimizer's LR
    virtual float get_lr() const = 0;
protected:
    Optimizer& optimizer_;
    int last_epoch_ = -1;
};
```

### Integration with Optimizer
We need to modify the `Optimizer` class to support setting the learning rate dynamically.
```cpp
class Optimizer {
    // ...
    void set_lr(float lr);
    float get_lr() const;
};
```

## 4. Usage Example

```cpp
auto optimizer = optim::Adam(model.parameters(), 1e-3);

// Warmup for 100 steps, then cosine decay
auto scheduler = optim::CosineAnnealingLR(optimizer, T_max=1000);

for (int step = 0; step < 1000; ++step) {
    train_step();
    optimizer.step();
    scheduler.step(); // Update LR for next step
}
```

## 5. Testing Strategy

1.  **LR Curve Verification**: Run the scheduler for N steps without training, record the LR at each step, and plot/verify it matches the expected curve.
2.  **Optimizer Interaction**: Ensure that calling `scheduler.step()` actually changes the `lr` inside the optimizer instance.
