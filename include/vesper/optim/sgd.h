#pragma once

#include <vesper/optim/optimizer.h>

namespace vesper::optim {

class SGD : public Optimizer {
public:
    SGD(std::vector<Tensor*> params, float lr);

    void step() override;

private:
    float lr_;
};

} // namespace vesper::optim
