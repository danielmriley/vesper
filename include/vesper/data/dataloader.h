#pragma once
#include <vesper/data/dataset.h>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <cstdint>

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
    // seed == 0 keeps the previous behaviour (reseed from std::random_device);
    // a fixed nonzero seed makes the shuffle order reproducible.
    DataLoader(std::shared_ptr<Dataset> dataset, size_t batch_size,
               bool shuffle = false, uint64_t seed = 0);
    DataLoaderIterator begin();
    DataLoaderIterator end();
    friend class DataLoaderIterator; // Grant access to private members
private:
    std::shared_ptr<Dataset> dataset_;
    size_t batch_size_;
    bool shuffle_;
    std::mt19937 rng_;  // Shuffle RNG; seeded in constructor (see seed param)
    std::vector<size_t> indices_;
};

} // namespace vesper::data
