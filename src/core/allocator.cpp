#include <vesper/core/allocator.h>
#include <iostream>
#include <stdexcept>

#if USE_HIP_BACKEND
#include <hip/hip_runtime.h>
#endif

#if USE_CUDA_BACKEND
#include <cuda_runtime.h>
#endif

namespace vesper {

// --- Singleton Management ---

// We need a static instance for each device type.
// For simplicity, we'll lazily initialize them.

static std::unique_ptr<CachingAllocator> cpu_allocator;
#if USE_HIP_BACKEND
static std::unique_ptr<CachingAllocator> hip_allocator;
#endif
#if USE_CUDA_BACKEND
static std::unique_ptr<CachingAllocator> cuda_allocator;
#endif

Allocator* get_allocator(Device device) {
    switch (device) {
        case Device::CPU:
            if (!cpu_allocator) cpu_allocator = std::make_unique<CachingAllocator>(Device::CPU);
            return cpu_allocator.get();
        case Device::HIP:
#if USE_HIP_BACKEND
            if (!hip_allocator) hip_allocator = std::make_unique<CachingAllocator>(Device::HIP);
            return hip_allocator.get();
#else
            throw std::runtime_error("HIP backend not enabled.");
#endif
        case Device::CUDA:
#if USE_CUDA_BACKEND
            if (!cuda_allocator) cuda_allocator = std::make_unique<CachingAllocator>(Device::CUDA);
            return cuda_allocator.get();
#else
            throw std::runtime_error("CUDA backend not enabled.");
#endif
        default:
            throw std::runtime_error("Unknown device allocator requested.");
    }
}

void release_all_memory() {
    // Mark all allocators as shutting down and empty their caches
    // This prevents the "blocks still in use" warning during static destruction
    if (cpu_allocator) {
        cpu_allocator->mark_shutdown();
        cpu_allocator->empty_cache();
    }
#if USE_HIP_BACKEND
    if (hip_allocator) {
        hip_allocator->mark_shutdown();
        hip_allocator->empty_cache();
    }
#endif
#if USE_CUDA_BACKEND
    if (cuda_allocator) {
        cuda_allocator->mark_shutdown();
        cuda_allocator->empty_cache();
    }
#endif
}

// --- CachingAllocator Implementation ---

CachingAllocator::CachingAllocator(Device device) : device_(device) {}

CachingAllocator::~CachingAllocator() {
    empty_cache();
    // Note: If allocated_blocks_ is not empty, it means the user is holding onto pointers
    // after the allocator is destroyed. This is a leak/bug in user code (or static destruction order).
    // If shutdown_ is true, we've been asked to shut down cleanly so suppress the warning.
    if (!allocated_blocks_.empty() && !shutdown_) {
        std::cerr << "Warning: CachingAllocator destroyed with " << allocated_blocks_.size() 
                  << " blocks still in use!" << std::endl;
    }
}

void* CachingAllocator::malloc_driver(size_t size) {
    void* ptr = nullptr;
    if (device_ == Device::CPU) {
        ptr = new char[size];
    } else if (device_ == Device::HIP) {
#if USE_HIP_BACKEND
        if (hipMalloc(&ptr, size) != hipSuccess) return nullptr;
#endif
    } else if (device_ == Device::CUDA) {
#if USE_CUDA_BACKEND
        if (cudaMalloc(&ptr, size) != cudaSuccess) return nullptr;
#endif
    }
    return ptr;
}

void CachingAllocator::free_driver(void* ptr) {
    if (device_ == Device::CPU) {
        delete[] static_cast<char*>(ptr);
    } else if (device_ == Device::HIP) {
#if USE_HIP_BACKEND
        hipFree(ptr);
#endif
    } else if (device_ == Device::CUDA) {
#if USE_CUDA_BACKEND
        cudaFree(ptr);
#endif
    }
}

void* CachingAllocator::allocate(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);

    // 1. Try to find a free block in the cache
    auto it = free_blocks_.find(size);
    if (it != free_blocks_.end() && !it->second.empty()) {
        void* ptr = it->second.back();
        it->second.pop_back();
        
        allocated_blocks_[ptr] = size;
        return ptr;
    }

    // 2. Allocate new block from driver
    void* ptr = malloc_driver(size);
    if (!ptr) {
        // Optional: Try to free cache and retry?
        // For simple MVP, we just throw or try to clear cache.
        empty_cache(); // Free up memory and try again
        ptr = malloc_driver(size);
        if (!ptr) {
            throw std::runtime_error("Out of memory!");
        }
    }

    allocated_blocks_[ptr] = size;
    total_allocated_ += size;
    return ptr;
}

void CachingAllocator::free(void* ptr) {
    if (!ptr) return;
    
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = allocated_blocks_.find(ptr);
    if (it == allocated_blocks_.end()) {
        throw std::runtime_error("Attempt to free pointer not managed by CachingAllocator");
    }

    size_t size = it->second;
    allocated_blocks_.erase(it);

    // Return to cache
    free_blocks_[size].push_back(ptr);
    total_cached_ += size;
}

void CachingAllocator::empty_cache() {
    // std::lock_guard<std::mutex> lock(mutex_); // Assuming caller might hold lock or called from dtor
    // Actually, empty_cache might be called by user.
    
    // Iterate over all free blocks and release them
    for (auto& pair : free_blocks_) {
        for (void* ptr : pair.second) {
            free_driver(ptr);
            total_cached_ -= pair.first;
            total_allocated_ -= pair.first;
        }
        pair.second.clear();
    }
    free_blocks_.clear();
}

} // namespace vesper
