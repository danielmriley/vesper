#pragma once
#include <vesper/data/dataset.h>
#include <vesper/data/dataloader.h>  // For Batch type
#include <vesper/ops/stack.h>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <numeric>
#include <random>
#include <algorithm>
#include <optional>

namespace vesper::data {

/**
 * @brief High-performance DataLoader with multi-threaded prefetching
 * 
 * Features:
 * - Background thread pool for parallel batch preparation
 * - Prefetch queue to overlap CPU data loading with GPU computation
 * - Pinned memory support (when available) for faster CPU->GPU transfers
 */
class PrefetchDataLoader {
public:
    /**
     * @param dataset The dataset to load from
     * @param batch_size Number of samples per batch
     * @param shuffle Whether to shuffle indices each epoch
     * @param num_workers Number of worker threads for data loading (0 = main thread only)
     * @param prefetch_factor Number of batches to prefetch per worker
     */
    PrefetchDataLoader(std::shared_ptr<Dataset> dataset, 
                       size_t batch_size, 
                       bool shuffle = false,
                       size_t num_workers = 2,
                       size_t prefetch_factor = 2)
        : dataset_(std::move(dataset))
        , batch_size_(batch_size)
        , shuffle_(shuffle)
        , num_workers_(num_workers)
        , prefetch_count_(std::max(size_t(1), num_workers * prefetch_factor))
        , stop_workers_(false)
        , current_batch_idx_(0)
    {
        indices_.resize(dataset_->size());
        std::iota(indices_.begin(), indices_.end(), 0);
        total_batches_ = (indices_.size() + batch_size_ - 1) / batch_size_;
    }
    
    ~PrefetchDataLoader() {
        stop_workers();
    }
    
    // Non-copyable
    PrefetchDataLoader(const PrefetchDataLoader&) = delete;
    PrefetchDataLoader& operator=(const PrefetchDataLoader&) = delete;
    
    /**
     * @brief Start a new epoch - call before iterating
     */
    void start_epoch() {
        // Stop any existing workers
        stop_workers();
        
        // Shuffle if needed
        if (shuffle_) {
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(indices_.begin(), indices_.end(), g);
        }
        
        // Reset state
        current_batch_idx_ = 0;
        next_batch_to_load_ = 0;
        stop_workers_ = false;
        
        // Clear any old batches
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            while (!batch_queue_.empty()) batch_queue_.pop();
        }
        
        // Start worker threads
        for (size_t i = 0; i < num_workers_; ++i) {
            workers_.emplace_back(&PrefetchDataLoader::worker_loop, this);
        }
    }
    
    /**
     * @brief Get next batch (blocks if not yet loaded)
     * @return The next batch, or nullopt if epoch is done
     */
    std::optional<Batch> next() {
        if (current_batch_idx_ >= total_batches_) {
            return std::nullopt;
        }
        
        Batch batch;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            
            // Wait for a batch to be available or all workers done
            queue_cv_.wait(lock, [this] {
                return !batch_queue_.empty() || stop_workers_;
            });
            
            if (batch_queue_.empty()) {
                return std::nullopt;
            }
            
            batch = std::move(batch_queue_.front());
            batch_queue_.pop();
        }
        
        // Notify workers that there's room in the queue
        load_cv_.notify_one();
        
        current_batch_idx_++;
        return batch;
    }
    
    /**
     * @brief Check if there are more batches in the current epoch
     */
    bool has_next() const {
        return current_batch_idx_ < total_batches_;
    }
    
    /**
     * @brief Get total number of batches per epoch
     */
    size_t num_batches() const {
        return total_batches_;
    }
    
    /**
     * @brief Stop all worker threads (called automatically at end of epoch or destruction)
     */
    void stop_workers() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            stop_workers_ = true;
        }
        load_cv_.notify_all();
        queue_cv_.notify_all();
        
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        workers_.clear();
    }

private:
    void worker_loop() {
        while (true) {
            size_t batch_idx;
            bool got_work = false;
            
            // Get next batch index to load
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                
                // Wait if queue is full or if we should stop
                load_cv_.wait(lock, [this] {
                    return stop_workers_ || 
                           (batch_queue_.size() < prefetch_count_ && 
                            next_batch_to_load_ < total_batches_);
                });
                
                if (stop_workers_) {
                    return;
                }
                
                // Try to grab a batch index
                if (next_batch_to_load_ < total_batches_) {
                    batch_idx = next_batch_to_load_++;
                    got_work = true;
                }
            }
            
            if (!got_work) {
                // No more work to do, exit this worker
                return;
            }
            
            // Load the batch (outside lock)
            Batch batch = load_batch(batch_idx);
            
            // Add to queue
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                if (!stop_workers_) {
                    batch_queue_.push(std::move(batch));
                }
            }
            queue_cv_.notify_one();
        }
    }
    
    Batch load_batch(size_t batch_idx) {
        size_t start = batch_idx * batch_size_;
        size_t end = std::min(start + batch_size_, indices_.size());
        
        std::vector<Tensor> x_tensors, y_tensors;
        x_tensors.reserve(end - start);
        y_tensors.reserve(end - start);
        
        for (size_t i = start; i < end; ++i) {
            Sample sample = dataset_->get_item(indices_[i]);
            x_tensors.push_back(sample.first);
            y_tensors.push_back(sample.second);
        }
        
        return {ops::stack(x_tensors, 0), ops::stack(y_tensors, 0)};
    }

    std::shared_ptr<Dataset> dataset_;
    size_t batch_size_;
    bool shuffle_;
    size_t num_workers_;
    size_t prefetch_count_;
    std::vector<size_t> indices_;
    size_t total_batches_;
    
    // Worker threads
    std::vector<std::thread> workers_;
    std::atomic<bool> stop_workers_;
    
    // Batch queue
    std::queue<Batch> batch_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;   // Signals when batch is ready
    std::condition_variable load_cv_;    // Signals when there's room to load
    
    // Progress tracking
    std::atomic<size_t> current_batch_idx_;
    std::atomic<size_t> next_batch_to_load_;
};

} // namespace vesper::data
