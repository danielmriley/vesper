#include <vesper/nn/pooling.h>
#include <vesper/ops/pooling.h>

namespace vesper::nn {

MaxPool2d::MaxPool2d(int64_t kernel_size, int64_t stride, int64_t padding)
    : MaxPool2d({kernel_size, kernel_size}, {stride, stride}, {padding, padding}) {}

MaxPool2d::MaxPool2d(std::pair<int64_t, int64_t> kernel_size, std::pair<int64_t, int64_t> stride, std::pair<int64_t, int64_t> padding)
    : kernel_size_(kernel_size), stride_(stride), padding_(padding) 
{
    if (stride_.first == -1) stride_.first = kernel_size_.first;
    if (stride_.second == -1) stride_.second = kernel_size_.second;
}

Tensor MaxPool2d::forward(const Tensor& input) {
    // Input: [B, C, H, W]
    auto result = ops::max_pool2d(input, 
        kernel_size_.first, kernel_size_.second,
        stride_.first, stride_.second,
        padding_.first, padding_.second);
    
    return result.first; // Return only the output, indices are hidden (handled by ops::max_pool2d via closure)
}

} // namespace vesper::nn
