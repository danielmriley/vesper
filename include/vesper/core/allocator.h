#pragma once

#include <vesper/core/device.h>
#include <cstddef>
#include <memory>
#include <unordered_map>
#include <vector>
#include <mutex>

namespace vesper {

// Abstract Allocator Interface
class Allocator {
public:
    virtual ~Allocator() = default;
    virtual void* allocate(size_t size) = 0;
    virtual void free(void* ptr) = 0;
};

// Global access point for backend-specific allocators
Allocator* get_allocator(Device device);

// Simple Caching Allocator
// Strategy: Keep a pool of free blocks sorted by size.
// When allocating, find a block >= size. If not found, allocate from driver.
// When freeing, return to pool.
// For simplicity in MVP: Exact size matching or Best Fit buckets.
// We'll use a map: size -> list of pointers.
class CachingAllocator : public Allocator {
public:
    CachingAllocator(Device device);
    ~CachingAllocator();

    void* allocate(size_t size) override;
    void free(void* ptr) override;

    // Clears the cache and releases all free memory to the driver
    void empty_cache();

private:
    // Helper for raw allocation
    void* malloc_driver(size_t size);
    void free_driver(void* ptr);

    Device device_;
    
    struct Block {
        void* ptr;
        size_t size;
        bool allocated; // true if currently in use
    };

    // We map size -> list of free blocks
    std::unordered_map<size_t, std::vector<void*>> free_blocks_;
    
    // Map ptr -> size to handle free(ptr)
    std::unordered_map<void*, size_t> allocated_blocks_;

    std::mutex mutex_;
    size_t total_allocated_ = 0;
    size_t total_cached_ = 0;
};

} // namespace vesper
