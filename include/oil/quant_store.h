#pragma once

#include "oil/tensor.h"
#include "oil/memory.h"
#include "oil/types.h"

#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <string>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <list>

namespace oil {

struct QuantizedTensorInfo {
    Format format = Format::OIL8;
    int64_t num_elements = 0;
    size_t storage_bytes = 0;
    int64_t ref_count = 1;
    Shape original_shape;
    DType original_dtype = DType::F32;
};

class QuantizedStore {
public:
    QuantizedStore(size_t lru_capacity_mb = 1024, size_t lru_block_count = 64);

    ~QuantizedStore();

    // Store a quantized tensor. Returns a handle ID.
    int64_t store(const void* quantized_data, size_t bytes, const QuantizedTensorInfo& info);

    // Dequantize on-the-fly. If already in LRU cache, returns cached copy.
    // Thread-safe.
    Tensor dequantize(int64_t handle);

    // Increment reference count (for shared MoE expert weights)
    void retain(int64_t handle);

    // Decrement reference count. When it reaches 0, storage is freed.
    void release(int64_t handle);

    // Get reference count
    int ref_count(int64_t handle) const;

    // Check if a handle is valid and still stored
    bool contains(int64_t handle) const;

    // Remove from cache (but keep storage if refcount > 0)
    void evict_from_cache(int64_t handle);

    // Clear everything
    void clear();

    // Memory-mapped backing file support
    void set_backing_file(const std::string& filepath);

    // Flush LRU cache to backing file
    void flush_cache_to_disk();

    // Statistics
    size_t num_stored() const;
    size_t cache_size_bytes() const;
    size_t lru_capacity_bytes() const;
    size_t total_storage_bytes() const;

    // Dequantize a buffer in a given format to float
    static void dequantize_buffer(const void* src, float* dst,
                                   int64_t num_elements, Format fmt);

private:
    struct StorageEntry {
        QuantizedTensorInfo info;
        std::vector<uint8_t> data;
        std::string backing_file;
        bool on_disk = false;
    };

    struct CacheEntry {
        Tensor tensor;
        int64_t handle;
        size_t size_bytes;
    };

    mutable std::mutex mutex_;
    std::unordered_map<int64_t, StorageEntry> storage_;
    int64_t next_handle_ = 1;

    // LRU cache for dequantized tensors
    size_t lru_capacity_bytes_;
    size_t current_cache_bytes_ = 0;
    std::list<int64_t> lru_order_;
    std::unordered_map<int64_t, CacheEntry> lru_cache_;

    // Backing file
    std::string backing_file_;

    void evict_lru();
    void validate_handle(int64_t handle) const;

    // Dequantize helpers per format
    static void dequantize_oil8(const uint8_t* src, float* dst, int64_t n);
    static void dequantize_oil4(const uint8_t* src, float* dst, int64_t n);
    static void dequantize_i2s(const uint8_t* src, float* dst, int64_t n);
    static void dequantize_spark_q0(const uint8_t* src, float* dst, int64_t n);
    static void dequantize_oil1(const uint8_t* src, float* dst, int64_t n,
                                     const float* block_means = nullptr, int64_t num_blocks = 0);
};

} // namespace oil
