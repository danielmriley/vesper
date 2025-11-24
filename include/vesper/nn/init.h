#pragma once
#include <vesper/core/tensor.h>

namespace vesper::nn::init {

void uniform_(Tensor& tensor, float min = 0.0f, float max = 1.0f);
void normal_(Tensor& tensor, float mean = 0.0f, float std = 1.0f);
void constant_(Tensor& tensor, float value);
void ones_(Tensor& tensor);
void zeros_(Tensor& tensor);

void kaiming_uniform_(Tensor& tensor, float a = 0.0f, const std::string& mode = "fan_in", const std::string& nonlinearity = "leaky_relu");
void kaiming_normal_(Tensor& tensor, float a = 0.0f, const std::string& mode = "fan_in", const std::string& nonlinearity = "leaky_relu");
void xavier_uniform_(Tensor& tensor, float gain = 1.0f);
void xavier_normal_(Tensor& tensor, float gain = 1.0f);

} // namespace vesper::nn::init
