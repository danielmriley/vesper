#include <vesper/core/allocator.h>
#include <iostream>
#include <stdexcept>
#include <algorithm>

#if USE_HIP_BACKEND
#include <hip/hip_runtime.h>
#endif

#if USE_CUDA_BACKEND
#include <cuda_runtime.h>
#endif

namespace vesper {

// --- Size Class Binning Utilities ---

// Minimum allocation size (avoid tiny allocations)
constexpr size_t MIN_BLOCK_SIZE = 512;

// Round up size to nearest power of 2 for efficient caching
// This allows blocks of similar sizes to be reused, reducing fragmentation
static size_t round_to_bin_size(size_t size) {
    if (size <= MIN_BLOCK_SIZE) {
        return MIN_BLOCK_SIZE;
    }
    
    // Find next power of 2
    size_t power = MIN_BLOCK_SIZE;
    while (power < size) {
        power *= 2;
    }
    return power;
}

// Get bin index for a given size (for O(1) cache lookup)
static size_t get_bin_index(size_t size) {
    if (size <= MIN_BLOCK_SIZE) {
        return 0;
    }
    // Calculate log2 of size relative to MIN_BLOCK_SIZE
    size_t bin = 0;
    size_t s = MIN_BLOCK_SIZE;
    while (s < size) {
        s *= 2;
        ++bin;
    }
    return bin;
}

// --- Singleton Management ---

// We need a static instance for each device type.
// For simplicity, we'll lazily initialize them.

// Leaky singletons: one allocator per device, intentionally never destroyed.
// A process-wide allocator must outlive every Storage; destroying it at static
// teardown (the old std::unique_ptr) raced with late Storage destructors and
// crashed on exit. These objects are reclaimed by the OS at process exit.
static CachingAllocator* cpu_allocator = nullptr;
#if USE_HIP_BACKEND
static CachingAllocator* hip_allocator = nullptr;
#endif
#if USE_CUDA_BACKEND
static CachingAllocator* cuda_allocator = nullptr;
#endif

Allocator* get_allocator(Device device) {
    switch (device) {
        case Device::CPU:
            if (!cpu_allocator) cpu_allocator = new CachingAllocator(Device::CPU);
            return cpu_allocator;
        case Device::HIP:
#if USE_HIP_BACKEND
            if (!hip_allocator) hip_allocator = new CachingAllocator(Device::HIP);
            return hip_allocator;
#else
            throw std::runtime_error("HIP backend not enabled.");
#endif
        case Device::CUDA:
#if USE_CUDA_BACKEND
            if (!cuda_allocator) cuda_allocator = new CachingAllocator(Device::CUDA);
            return cuda_allocator;
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
    
    // Debug: detect suspiciously large allocations (> 1GB)
    if (size > 1024ULL * 1024ULL * 1024ULL) {
        std::cerr << "WARNING: Attempting to allocate " << (size / (1024.0 * 1024.0 * 1024.0)) 
                  << " GB (" << size << " bytes) - this is likely a bug!" << std::endl;
        // Print stack trace hint
        std::cerr << "  This often indicates: shape overflow, negative dimensions, or uninitialized size" << std::endl;
        // Abort to get stack trace
        // std::abort();
    }
    
    if (device_ == Device::CPU) {
        ptr = new char[size];
    } else if (device_ == Device::HIP) {
#if USE_HIP_BACKEND
        hipError_t err = hipMalloc(&ptr, size);
        if (err != hipSuccess) {
            std::cerr << "hipMalloc failed for size " << size << " bytes (" 
                      << (size / (1024.0 * 1024.0)) << " MB): " 
                      << hipGetErrorString(err) << std::endl;
            return nullptr;
        }
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

    // Round size up to bin size for efficient cache reuse
    size_t bin_size = round_to_bin_size(size);

    // 1. Try to find a free block in the cache (using binned size)
    auto it = free_blocks_.find(bin_size);
    if (it != free_blocks_.end() && !it->second.empty()) {
        void* ptr = it->second.back();
        it->second.pop_back();
        
        // Track actual bin size for proper freeing
        allocated_blocks_[ptr] = bin_size;
        return ptr;
    }

    // 2. Allocate new block from driver (allocate bin_size, not original size)
    void* ptr = malloc_driver(bin_size);
    if (!ptr) {
        // Optional: Try to free cache and retry?
        // For simple MVP, we just throw or try to clear cache.
        empty_cache(); // Free up memory and try again
        ptr = malloc_driver(bin_size);
        if (!ptr) {
            throw std::runtime_error("Out of memory!");
        }
    }

    allocated_blocks_[ptr] = bin_size;
    total_allocated_ += bin_size;
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
