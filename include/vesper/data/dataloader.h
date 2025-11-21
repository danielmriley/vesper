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
    // Iterator traits
    using iterator_category = std::forward_iterator_tag;
    using value_type = Batch;
    using difference_type = std::ptrdiff_t;
    using pointer = Batch*;
    using reference = Batch&;

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
