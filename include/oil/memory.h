#pragma once

#include <cstdint>
#include <cstddef>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <new>
#include <algorithm>
#include <vector>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace oil {

struct AlignedAllocator {
    static void* allocate(size_t bytes, size_t alignment = 64) {
#ifdef _WIN32
        void* ptr = _aligned_malloc(bytes, alignment);
#else
        void* ptr = nullptr;
        if (posix_memalign(&ptr, alignment, bytes) != 0)
            ptr = nullptr;
#endif
        if (!ptr && bytes)
            throw std::bad_alloc();
        return ptr;
    }

    static void deallocate(void* ptr) {
#ifdef _WIN32
        _aligned_free(ptr);
#else
        std::free(ptr);
#endif
    }
};

class Buffer {
public:
    Buffer() = default;

    explicit Buffer(size_t size_bytes, size_t alignment = 64)
        : block_{allocate_block(size_bytes, alignment)} {}

    Buffer(const Buffer& other)
        : block_{other.block_} {
        if (block_.refcount)
            block_.refcount->fetch_add(1, std::memory_order_acq_rel);
    }

    Buffer& operator=(const Buffer& other) {
        if (this == &other)
            return *this;
        release();
        block_ = other.block_;
        if (block_.refcount)
            block_.refcount->fetch_add(1, std::memory_order_acq_rel);
        return *this;
    }

    ~Buffer() {
        release();
    }

    Buffer(Buffer&& other) noexcept
        : block_{other.block_} {
        other.block_ = {nullptr, nullptr, nullptr, 0};
    }

    Buffer& operator=(Buffer&& other) noexcept {
        if (this == &other) return *this;
        release();
        block_ = other.block_;
        other.block_ = {nullptr, nullptr, nullptr, 0};
        return *this;
    }

    void* data() const { return block_.ptr; }

    size_t size() const { return block_.size; }

    bool empty() const { return block_.size == 0; }

    void resize(size_t new_size) {
        if (new_size == block_.size) return;
        Buffer new_buf(new_size, 64);
        if (block_.ptr && new_size > 0) {
            size_t copy_size = (std::min)(block_.size, new_size);
            std::memcpy(new_buf.data(), block_.ptr, copy_size);
        }
        *this = std::move(new_buf);
    }

    Buffer slice(size_t offset, size_t count) const {
        if (offset + count > block_.size)
            return Buffer();
        Buffer sub;
        sub.block_.ptr = static_cast<char*>(block_.ptr) + offset;
        sub.block_.base_ptr = block_.base_ptr ? block_.base_ptr : block_.ptr;
        sub.block_.refcount = block_.refcount;
        sub.block_.size = count;
        if (sub.block_.refcount)
            sub.block_.refcount->fetch_add(1, std::memory_order_acq_rel);
        return sub;
    }

private:
    struct Block {
        void* ptr;
        void* base_ptr;
        std::atomic<int>* refcount;
        size_t size;
    };

    Block block_{nullptr, nullptr, nullptr, 0};

    static Block allocate_block(size_t bytes, size_t alignment) {
        if (bytes == 0) return {nullptr, nullptr, nullptr, 0};
        void* ptr = AlignedAllocator::allocate(bytes, alignment);
        auto* rc = new std::atomic<int>(1);
        return {ptr, ptr, rc, bytes};
    }

    void release() {
        if (block_.refcount &&
            block_.refcount->fetch_sub(1, std::memory_order_acq_rel) == 1) {
            AlignedAllocator::deallocate(block_.base_ptr ? block_.base_ptr : block_.ptr);
            delete block_.refcount;
        }
        block_ = {nullptr, nullptr, nullptr, 0};
    }
};

class MemoryPool {
public:
    explicit MemoryPool(size_t total_size)
        : capacity_(total_size), offset_(0), peak_(0) {
        data_ = static_cast<char*>(AlignedAllocator::allocate(total_size));
    }

    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    MemoryPool(MemoryPool&& other) noexcept
        : data_(other.data_), capacity_(other.capacity_),
          offset_(other.offset_.load()), peak_(other.peak_.load()) {
        other.data_ = nullptr;
        other.capacity_ = 0;
        other.offset_.store(0);
        other.peak_.store(0);
    }

    MemoryPool& operator=(MemoryPool&& other) noexcept {
        if (this == &other) return *this;
        if (data_) AlignedAllocator::deallocate(data_);
        data_ = other.data_;
        capacity_ = other.capacity_;
        offset_.store(other.offset_.load());
        peak_.store(other.peak_.load());
        other.data_ = nullptr;
        other.capacity_ = 0;
        other.offset_.store(0);
        other.peak_.store(0);
        return *this;
    }

    ~MemoryPool() {
        if (data_) AlignedAllocator::deallocate(data_);
    }

    void* allocate(size_t bytes, size_t alignment = 64) {
        size_t aligned = (bytes + alignment - 1) & ~(alignment - 1);
        size_t old_offset = offset_.fetch_add(aligned, std::memory_order_relaxed);
        if (old_offset + aligned > capacity_) {
            offset_.fetch_sub(aligned, std::memory_order_relaxed);
            return nullptr;
        }
        size_t prev_peak = peak_.load(std::memory_order_relaxed);
        while (old_offset + aligned > prev_peak) {
            if (peak_.compare_exchange_weak(prev_peak, old_offset + aligned,
                                            std::memory_order_relaxed))
                break;
        }
        return data_ + old_offset;
    }

    void deallocate(void* ptr) {
        (void)ptr;
    }

    void reset() {
        offset_.store(0);
        peak_.store(0);
    }

    size_t bytes_used() const { return offset_.load(std::memory_order_relaxed); }
    size_t bytes_wasted() const { return capacity_ - offset_.load(std::memory_order_relaxed); }
    size_t peak_usage() const { return peak_.load(std::memory_order_relaxed); }
    size_t capacity() const { return capacity_; }

private:
    char* data_;
    size_t capacity_;
    std::atomic<size_t> offset_;
    std::atomic<size_t> peak_;
};

class StackAllocator {
public:
    struct Marker {
        size_t offset;
    };

    explicit StackAllocator(size_t total_size)
        : capacity_(total_size), offset_(0), peak_(0) {
        data_ = static_cast<char*>(AlignedAllocator::allocate(total_size));
    }

    StackAllocator(const StackAllocator&) = delete;
    StackAllocator& operator=(const StackAllocator&) = delete;

    StackAllocator(StackAllocator&& other) noexcept
        : data_(other.data_), capacity_(other.capacity_),
          offset_(other.offset_.load()), peak_(other.peak_.load()) {
        other.data_ = nullptr;
        other.capacity_ = 0;
        other.offset_.store(0);
        other.peak_.store(0);
    }

    StackAllocator& operator=(StackAllocator&& other) noexcept {
        if (this == &other) return *this;
        if (data_) AlignedAllocator::deallocate(data_);
        data_ = other.data_;
        capacity_ = other.capacity_;
        offset_.store(other.offset_.load());
        peak_.store(other.peak_.load());
        other.data_ = nullptr;
        other.capacity_ = 0;
        other.offset_.store(0);
        other.peak_.store(0);
        return *this;
    }

    ~StackAllocator() {
        if (data_) AlignedAllocator::deallocate(data_);
    }

    Marker mark() const {
        return Marker{offset_.load(std::memory_order_relaxed)};
    }

    void restore(Marker m) {
        offset_.store(m.offset, std::memory_order_relaxed);
    }

    void* allocate(size_t bytes, size_t alignment = 64) {
        size_t aligned = (bytes + alignment - 1) & ~(alignment - 1);
        size_t old_offset = offset_.fetch_add(aligned, std::memory_order_relaxed);
        if (old_offset + aligned > capacity_) {
            offset_.fetch_sub(aligned, std::memory_order_relaxed);
            return nullptr;
        }
        size_t prev_peak = peak_.load(std::memory_order_relaxed);
        while (old_offset + aligned > prev_peak) {
            if (peak_.compare_exchange_weak(prev_peak, old_offset + aligned,
                                            std::memory_order_relaxed))
                break;
        }
        return data_ + old_offset;
    }

    void reset() {
        offset_.store(0);
        peak_.store(0);
    }

    size_t bytes_used() const { return offset_.load(std::memory_order_relaxed); }
    size_t peak_usage() const { return peak_.load(std::memory_order_relaxed); }
    size_t capacity() const { return capacity_; }

private:
    char* data_;
    size_t capacity_;
    std::atomic<size_t> offset_;
    std::atomic<size_t> peak_;
};

class ThreadLocalPoolRegistry {
public:
    static ThreadLocalPoolRegistry& instance() {
        static ThreadLocalPoolRegistry inst;
        return inst;
    }

    MemoryPool* get_or_create_pool(size_t pool_size) {
        std::thread::id tid = std::this_thread::get_id();
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = pools_.find(tid);
        if (it != pools_.end() && it->second.generation == active_generation_[tid])
            return it->second.pool;
        auto* pool = new MemoryPool(pool_size);
        pools_[tid] = {pool, active_generation_[tid]};
        owners_[tid] = pool;
        return pool;
    }

    MemoryPool* get_pool() {
        std::thread::id tid = std::this_thread::get_id();
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = pools_.find(tid);
        if (it != pools_.end() && it->second.generation == active_generation_[tid])
            return it->second.pool;
        return nullptr;
    }

    void release_pool() {
        std::thread::id tid = std::this_thread::get_id();
        std::lock_guard<std::mutex> lock(mutex_);
        auto own_it = owners_.find(tid);
        if (own_it != owners_.end()) {
            delete own_it->second;
            owners_.erase(own_it);
        }
        pools_.erase(tid);
        active_generation_[tid]++;
    }

private:
    ThreadLocalPoolRegistry() = default;

    struct PoolEntry {
        MemoryPool* pool;
        uint64_t generation;
    };

    std::mutex mutex_;
    std::unordered_map<std::thread::id, PoolEntry> pools_;
    std::unordered_map<std::thread::id, MemoryPool*> owners_;
    std::unordered_map<std::thread::id, uint64_t> active_generation_;
};

inline MemoryPool* get_thread_local_pool(size_t pool_size = 64 * 1024 * 1024) {
    return ThreadLocalPoolRegistry::instance().get_or_create_pool(pool_size);
}

} // namespace oil
