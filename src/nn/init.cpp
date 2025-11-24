#include <vesper/nn/init.h>
#include <vesper/ops/random.h>
#include <vesper/core/factories.h>
#include <cmath>
#include <stdexcept>

namespace vesper::nn::init {

void uniform_(Tensor& tensor, float min, float max) {
    ops::uniform_(tensor, min, max);
}

void normal_(Tensor& tensor, float mean, float std) {
    ops::normal_(tensor, mean, std);
}

void constant_(Tensor& tensor, float value) {
    // Using `full` creates a new tensor. We need in-place fill.
    // We don't have a dedicated `fill_` op exposed in `ops/elementwise`.
    // But we can use `zeros_` then `add_`? Or just implement `fill_`.
    // `ops::uniform_` is implemented manually.
    // Let's reuse `uniform_` with min=max=value.
    ops::uniform_(tensor, value, value);
}

void ones_(Tensor& tensor) {
    constant_(tensor, 1.0f);
}

void zeros_(Tensor& tensor) {
    // We have a `zeros` factory, but not in-place.
    // `constant_(tensor, 0.0f)` works.
    constant_(tensor, 0.0f);
}

// Helper to calculate fan_in and fan_out
std::pair<int64_t, int64_t> calculate_fan_in_and_fan_out(const Tensor& tensor) {
    // Supports 2D (Linear) and 4D (Conv2d) weights
    const auto& shape = tensor.shape();
    if (shape.size() == 2) {
        // Linear: [Out, In]
        return {shape[1], shape[0]};
    } else if (shape.size() == 4) {
        // Conv2d: [Out, In, KH, KW]
        int64_t receptive_field_size = shape[2] * shape[3];
        return {shape[1] * receptive_field_size, shape[0] * receptive_field_size};
    } else {
        // Fallback or throw?
        // For 1D, it's [Size]. fan_in = size, fan_out = size?
        if (shape.size() == 1) return {shape[0], shape[0]};
        return {1, 1}; // Unknown
    }
}

float calculate_gain(const std::string& nonlinearity, float a) {
    if (nonlinearity == "linear" || nonlinearity == "sigmoid" || nonlinearity == "conv2d") return 1.0f;
    if (nonlinearity == "tanh") return 5.0f / 3.0f;
    if (nonlinearity == "relu") return std::sqrt(2.0f);
    if (nonlinearity == "leaky_relu") return std::sqrt(2.0f / (1 + a * a));
    return 1.0f;
}

void kaiming_uniform_(Tensor& tensor, float a, const std::string& mode, const std::string& nonlinearity) {
    auto fans = calculate_fan_in_and_fan_out(tensor);
    int64_t fan = (mode == "fan_in") ? fans.first : fans.second;
    float gain = calculate_gain(nonlinearity, a);
    float std = gain / std::sqrt(static_cast<float>(fan));
    float bound = std::sqrt(3.0f) * std; // Uniform bound
    
    uniform_(tensor, -bound, bound);
}

void kaiming_normal_(Tensor& tensor, float a, const std::string& mode, const std::string& nonlinearity) {
    auto fans = calculate_fan_in_and_fan_out(tensor);
    int64_t fan = (mode == "fan_in") ? fans.first : fans.second;
    float gain = calculate_gain(nonlinearity, a);
    float std = gain / std::sqrt(static_cast<float>(fan));
    
    normal_(tensor, 0.0f, std);
}

void xavier_uniform_(Tensor& tensor, float gain) {
    auto fans = calculate_fan_in_and_fan_out(tensor);
    int64_t fan_in = fans.first;
    int64_t fan_out = fans.second;
    float std = gain * std::sqrt(2.0f / static_cast<float>(fan_in + fan_out));
    float bound = std::sqrt(3.0f) * std;
    
    uniform_(tensor, -bound, bound);
}

void xavier_normal_(Tensor& tensor, float gain) {
    auto fans = calculate_fan_in_and_fan_out(tensor);
    int64_t fan_in = fans.first;
    int64_t fan_out = fans.second;
    float std = gain * std::sqrt(2.0f / static_cast<float>(fan_in + fan_out));
    
    normal_(tensor, 0.0f, std);
}

} // namespace vesper::nn::init
