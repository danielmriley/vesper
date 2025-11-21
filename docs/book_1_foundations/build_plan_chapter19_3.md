
# Vesper Build Plan - Chapter 19.3: The `DataLoader` Iterator

## 1. Goal

Implement the `DataLoader` and its C++ iterator. This class will use a `Dataset` and the `ops::stack` function to provide an easy-to-use, range-based `for` loop interface for iterating over a dataset in shuffled mini-batches.

## 2. Prerequisites

-   Chapter 19.1: The `Dataset` abstraction.
-   Chapter 19.2: The `ops::stack` function for collating batches.

## 3. Detailed Steps

### Step 3.1: Implement the `DataLoader`

We will now implement the full `DataLoader` and `DataLoaderIterator` logic, using `ops::stack` to correctly form batches.

Create/update `include/vesper/data/dataloader.h` and `src/data/dataloader.cpp`.
```cpp
// include/vesper/data/dataloader.h
#pragma once
#include <vesper/data/dataset.h>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>

namespace vesper::data {

using Batch = std::pair<Tensor, Tensor>;

class DataLoader; // Forward declaration

class DataLoaderIterator {
public:
    // ... C++ iterator boilerplate (typedefs) ...
    DataLoaderIterator(DataLoader* loader, size_t index);
    Batch operator*();
    DataLoaderIterator& operator++();
    bool operator!=(const DataLoaderIterator& other) const;
private:
    DataLoader* loader_;
    size_t current_index_;
};

class DataLoader {
public:
    DataLoader(std::shared_ptr<Dataset> dataset, size_t batch_size, bool shuffle = false);
    DataLoaderIterator begin();
    DataLoaderIterator end();
    friend class DataLoaderIterator; // Grant access to private members
private:
    std::shared_ptr<Dataset> dataset_;
    size_t batch_size_;
    bool shuffle_;
    std::vector<size_t> indices_;
};

} // namespace vesper::data
```
Implement the logic in `src/data/dataloader.cpp`:
```cpp
// src/data/dataloader.cpp
#include <vesper/data/dataloader.h>
#include <vesper/ops/stack.h>

namespace vesper::data {

DataLoader::DataLoader(std::shared_ptr<Dataset> dataset, size_t batch_size, bool shuffle)
    : dataset_(std::move(dataset)), batch_size_(batch_size), shuffle_(shuffle) {
    indices_.resize(dataset_->size());
    std::iota(indices_.begin(), indices_.end(), 0);
}

DataLoaderIterator DataLoader::begin() {
    if (shuffle_) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices_.begin(), indices_.end(), g);
    }
    return DataLoaderIterator(this, 0);
}

DataLoaderIterator DataLoader::end() {
    return DataLoaderIterator(this, indices_.size());
}

// --- Iterator Implementation ---
DataLoaderIterator::DataLoaderIterator(DataLoader* loader, size_t index)
    : loader_(loader), current_index_(index) {}

Batch DataLoaderIterator::operator*() {
    size_t start = current_index_;
    size_t end = std::min(start + loader_->batch_size_, loader_->indices_.size());

    std::vector<Tensor> x_tensors, y_tensors;
    x_tensors.reserve(end - start);
    y_tensors.reserve(end - start);

    for (size_t i = start; i < end; ++i) {
        Sample sample = loader_->dataset_->get_item(loader_->indices_[i]);
        x_tensors.push_back(sample.first);
        y_tensors.push_back(sample.second);
    }

    // Use the stack operation to collate the batch
    return {ops::stack(x_tensors, 0), ops::stack(y_tensors, 0)};
}

DataLoaderIterator& DataLoaderIterator::operator++() {
    current_index_ = std::min(current_index_ + loader_->batch_size_, loader_->indices_.size());
    return *this;
}

bool DataLoaderIterator::operator!=(const DataLoaderIterator& other) const {
    return this->current_index_ != other.current_index_;
}

} // namespace vesper::data
```

## 4. Verification

Update `tests/test_data.cpp` to verify that the `DataLoader` yields batches with the correct shapes and sizes.

```cpp
// tests/test_data.cpp
void test_dataloader_batching() {
    std::cout << "Testing DataLoader batching..." << std::endl;
    
    auto x = vesper::zeros({10, 5}, vesper::DType::Float32, vesper::Device::CPU);
    auto y = vesper::zeros({10, 1}, vesper::DType::Float32, vesper::Device::CPU);
    auto dataset = std::make_shared<vesper::data::TensorDataset>(x, y);

    const size_t batch_size = 4;
    auto loader = vesper::data::DataLoader(dataset, batch_size, false);

    size_t num_batches = 0;
    size_t total_samples = 0;
    for (const auto& batch : loader) {
        num_batches++;
        total_samples += batch.first.shape()[0];

        if (num_batches == 1) { // First batch
            assert(batch.first.shape() == std::vector<int64_t>({4, 5}));
            assert(batch.second.shape() == std::vector<int64_t>({4, 1}));
        } else if (num_batches == 3) { // Last batch
            assert(batch.first.shape() == std::vector<int64_t>({2, 5}));
            assert(batch.second.shape() == std::vector<int64_t>({2, 1}));
        }
    }

    assert(num_batches == 3);
    assert(total_samples == 10);

    std::cout << "DataLoader batching test passed!" << std::endl;
}
```
A passing test confirms the full data loading pipeline is now functional and ready to be used in a training loop.
