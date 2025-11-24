#pragma once

#include <vesper/optim/optimizer.h>

namespace vesper::optim {

class Adam : public Optimizer {
public:
    Adam(std::vector<Tensor> params, float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f, float weight_decay = 0.0f);

    void step() override;

    void set_lr(float lr) override { lr_ = lr; }
    float get_lr() const override { return lr_; }

    StateDict state_dict() const override;
    void load_state_dict(const StateDict& state_dict) override;

private:
    float lr_;
    float beta1_;
    float beta2_;
    float eps_;
    float weight_decay_;

protected:
    int t_ = 0; // Timestep (Protected for testing access)
};

} // namespace vesper::optim
