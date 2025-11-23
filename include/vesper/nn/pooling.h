#pragma once
#include <vesper/nn/module.h>
#include <vesper/core/tensor.h>

namespace vesper::nn {

class MaxPool2d : public Module {
public:
    MaxPool2d(int64_t kernel_size, int64_t stride = -1, int64_t padding = 0);
    MaxPool2d(std::pair<int64_t, int64_t> kernel_size, std::pair<int64_t, int64_t> stride = {-1, -1}, std::pair<int64_t, int64_t> padding = {0, 0});

    Tensor forward(const Tensor& input) override;

private:
    std::pair<int64_t, int64_t> kernel_size_;
    std::pair<int64_t, int64_t> stride_;
    std::pair<int64_t, int64_t> padding_;
};

} // namespace vesper::nn
