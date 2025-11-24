#pragma once

#include <vesper/optim/optimizer.h>

namespace vesper::optim {

class Lion : public Optimizer {
public:
    Lion(std::vector<Tensor> params, float lr = 1e-4f, float beta1 = 0.9f, float beta2 = 0.99f, float weight_decay = 0.0f);

    void step() override;

    void set_lr(float lr) override { lr_ = lr; }
    float get_lr() const override { return lr_; }

private:
    float lr_;
    float beta1_;
    float beta2_;
    float weight_decay_;
};

} // namespace vesper::optim
