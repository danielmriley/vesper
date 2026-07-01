#pragma once
#include <vesper/core/tensor.h>
#include <utility>
#include <memory>
#include <stdexcept>

namespace vesper::data {

// A sample is typically a pair of (data, target)
using Sample = std::pair<Tensor, Tensor>;

// Abstract base class for all datasets
class Dataset : public std::enable_shared_from_this<Dataset> {
public:
    virtual ~Dataset() = default;
    virtual Sample get_item(size_t index) = 0;
    virtual size_t size() const = 0;
};

// A concrete dataset for tensors held in memory.
// The first dimension of the tensors is assumed to be the batch dimension.
class TensorDataset : public Dataset {
public:
    TensorDataset(Tensor x, Tensor y)
        : x_data(std::move(x)), y_data(std::move(y)) {
        if (x_data.shape()[0] != y_data.shape()[0]) {
            throw std::runtime_error("Tensors must have the same size in dimension 0.");
        }
    }

    Sample get_item(size_t index) override {
        return {x_data.slice(index), y_data.slice(index)};
    }

    size_t size() const override {
        return x_data.shape()[0];
    }
private:
    Tensor x_data;
    Tensor y_data;
};

} // namespace vesper::data
