#pragma once
#include <vesper/nn/module.h>
#include <vesper/core/tensor.h>

namespace vesper::nn {

class Conv2d : public Module {
public:
    Conv2d(int64_t in_channels, int64_t out_channels, int64_t kernel_size, int64_t stride = 1, int64_t padding = 0, bool use_bias = true, Device device = Device::CPU);
    Conv2d(int64_t in_channels, int64_t out_channels, std::pair<int64_t, int64_t> kernel_size, std::pair<int64_t, int64_t> stride = {1, 1}, std::pair<int64_t, int64_t> padding = {0, 0}, bool use_bias = true, Device device = Device::CPU);

    Tensor forward(const Tensor& input) override;

    Tensor weight;
    Tensor bias;

private:
    int64_t in_channels_;
    int64_t out_channels_;
    std::pair<int64_t, int64_t> kernel_size_;
    std::pair<int64_t, int64_t> stride_;
    std::pair<int64_t, int64_t> padding_;
    bool use_bias_;
};

} // namespace vesper::nn
