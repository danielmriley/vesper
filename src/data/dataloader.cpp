#include <vesper/data/dataloader.h>
#include <vesper/ops/stack.h>
#include <numeric>
#include <algorithm>
#include <random>

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
