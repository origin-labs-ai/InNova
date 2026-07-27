#include "oil/quant_store.h"
#include <cstring>
#include <cmath>
#include <fstream>
#include <sstream>
#include <cstdio>

namespace oil {

// ===========================================================================
// Dequantization kernels per format
// ===========================================================================

void QuantizedStore::dequantize_oil8(const uint8_t* src, float* dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        dst[i] = ((float)src[i] - 128.0f) / 128.0f;
    }
}

void QuantizedStore::dequantize_oil4(const uint8_t* src, float* dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        int idx = static_cast<int>(i / 2);
        int shift = (i % 2) ? 0 : 4;
        uint8_t nibble = (src[idx] >> shift) & 0x0F;
        dst[i] = ((float)nibble - 8.0f) / 8.0f;
    }
}

void QuantizedStore::dequantize_i2s(const uint8_t* src, float* dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        int idx = static_cast<int>(i / 4);
        int shift = (i % 4) * 2;
        int8_t val = (int8_t)((src[idx] >> shift) & 0x03);
        if (val >= 2) val -= 4;
        dst[i] = (float)val;
    }
}

void QuantizedStore::dequantize_ternary(const uint8_t* src, float* dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        int idx = static_cast<int>(i / 4);
        int shift = (i % 4) * 2;
        int8_t val = (int8_t)((src[idx] >> shift) & 0x03);
        if (val == 0) dst[i] = -1.0f;
        else if (val == 1) dst[i] = 0.0f;
        else dst[i] = 1.0f;
    }
}

void QuantizedStore::dequantize_binary(const uint8_t* src, float* dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        int idx = static_cast<int>(i / 8);
        int shift = (i % 8);
        dst[i] = ((src[idx] >> shift) & 1) ? 1.0f : -1.0f;
    }
}

void QuantizedStore::dequantize_buffer(const void* src, float* dst,
                                        int64_t num_elements, Format fmt) {
    const uint8_t* s = static_cast<const uint8_t*>(src);
    switch (fmt) {
        case Format::OIL8:   dequantize_oil8(s, dst, num_elements); break;
        case Format::OIL4:   dequantize_oil4(s, dst, num_elements); break;
        case Format::SPARK_Q0: dequantize_ternary(s, dst, num_elements); break;
        case Format::OIL1:  dequantize_binary(s, dst, num_elements); break;
        case Format::OIL16: {
            // Each pair of bytes is an FP16 value - naive cast to float
            for (int64_t i = 0; i < num_elements; i++) {
                uint16_t half = ((uint16_t)s[i * 2 + 1] << 8) | s[i * 2];
                // FP16 -> FP32 conversion
                uint32_t sign = (uint32_t)(half >> 15) << 31;
                int32_t exp = (half >> 10) & 0x1F;
                uint32_t mant = half & 0x03FF;
                if (exp == 0) {
                    // Subnormal / zero
                    if (mant == 0) {
                        dst[i] = 0.0f;
                    } else {
                        float m = (float)mant / 1024.0f;
                        dst[i] = (sign ? -1.0f : 1.0f) * m * 1.5258789e-5f;
                    }
                } else if (exp == 31) {
                    dst[i] = (mant == 0) ? (sign ? -INFINITY : INFINITY) : NAN;
                } else {
                    uint32_t f32 = sign | ((uint32_t)(exp + 112) << 23) | (mant << 13);
                    std::memcpy(&dst[i], &f32, sizeof(float));
                }
            }
            break;
        }
        default:
            std::memcpy(dst, src, (size_t)num_elements * sizeof(float));
            break;
    }
}

// ===========================================================================
// QuantizedStore implementation
// ===========================================================================

QuantizedStore::QuantizedStore(size_t lru_capacity_mb, size_t lru_block_count)
    : lru_capacity_bytes_(lru_capacity_mb * 1024 * 1024) {
    (void)lru_block_count;
}

QuantizedStore::~QuantizedStore() {
    clear();
}

int64_t QuantizedStore::store(const void* quantized_data, size_t bytes,
                               const QuantizedTensorInfo& info) {
    std::lock_guard<std::mutex> lock(mutex_);
    int64_t handle = next_handle_++;
    StorageEntry entry;
    entry.info = info;
    entry.data.resize(bytes);
    if (quantized_data && bytes > 0) {
        std::memcpy(entry.data.data(), quantized_data, bytes);
    }
    entry.info.storage_bytes = bytes;
    entry.info.ref_count = 1;
    storage_[handle] = std::move(entry);
    return handle;
}

Tensor QuantizedStore::dequantize(int64_t handle) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto cache_it = lru_cache_.find(handle);
        if (cache_it != lru_cache_.end()) {
            lru_order_.remove(handle);
            lru_order_.push_front(handle);
            return cache_it->second.tensor;
        }
    }

    QuantizedTensorInfo info;
    std::vector<uint8_t> data_copy;
    bool found = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = storage_.find(handle);
        if (it == storage_.end())
            return Tensor();
        auto& entry = it->second;
        info = entry.info;
        if (entry.on_disk) {
            std::ifstream ifs(entry.backing_file, std::ios::binary);
            if (ifs) {
                data_copy.resize(entry.data.size());
                ifs.read(reinterpret_cast<char*>(data_copy.data()),
                         (std::streamsize)entry.data.size());
            }
        } else {
            data_copy = entry.data;
        }
        found = true;
    }

    if (!found || data_copy.empty())
        return Tensor();

    Tensor result(info.original_shape, DType::F32);
    dequantize_buffer(data_copy.data(), result.data<float>(),
                      info.num_elements, info.format);

    size_t tensor_bytes = (size_t)info.num_elements * sizeof(float);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        while (current_cache_bytes_ + tensor_bytes > lru_capacity_bytes_ &&
               !lru_order_.empty()) {
            evict_lru();
        }
        lru_cache_[handle] = CacheEntry{result, handle, tensor_bytes};
        lru_order_.push_front(handle);
        current_cache_bytes_ += tensor_bytes;
    }

    return result;
}

void QuantizedStore::retain(int64_t handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = storage_.find(handle);
    if (it != storage_.end()) {
        it->second.info.ref_count++;
    }
}

void QuantizedStore::release(int64_t handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = storage_.find(handle);
    if (it == storage_.end()) return;
    it->second.info.ref_count--;
    if (it->second.info.ref_count <= 0) {
        if (it->second.on_disk && !it->second.backing_file.empty()) {
            std::remove(it->second.backing_file.c_str());
        }
        storage_.erase(it);
        lru_cache_.erase(handle);
        lru_order_.remove(handle);
    }
}

int QuantizedStore::ref_count(int64_t handle) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = storage_.find(handle);
    return it != storage_.end() ? (int)it->second.info.ref_count : 0;
}

bool QuantizedStore::contains(int64_t handle) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return storage_.find(handle) != storage_.end();
}

void QuantizedStore::evict_from_cache(int64_t handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto cache_it = lru_cache_.find(handle);
    if (cache_it != lru_cache_.end()) {
        current_cache_bytes_ -= cache_it->second.size_bytes;
        lru_cache_.erase(cache_it);
        lru_order_.remove(handle);
    }
}

void QuantizedStore::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [id, entry] : storage_) {
        if (entry.on_disk && !entry.backing_file.empty()) {
            std::remove(entry.backing_file.c_str());
        }
    }
    storage_.clear();
    lru_cache_.clear();
    lru_order_.clear();
    current_cache_bytes_ = 0;
    next_handle_ = 1;
}

void QuantizedStore::set_backing_file(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(mutex_);
    backing_file_ = filepath;
}

void QuantizedStore::flush_cache_to_disk() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (backing_file_.empty()) return;

    std::ofstream ofs(backing_file_, std::ios::binary);
    if (!ofs) return;

    int64_t num_entries = (int64_t)storage_.size();
    ofs.write(reinterpret_cast<const char*>(&num_entries), sizeof(num_entries));
    for (auto& [id, entry] : storage_) {
        ofs.write(reinterpret_cast<const char*>(&id), sizeof(id));
        int8_t fmt = (int8_t)entry.info.format;
        ofs.write(reinterpret_cast<const char*>(&fmt), sizeof(fmt));
        ofs.write(reinterpret_cast<const char*>(&entry.info.num_elements), sizeof(entry.info.num_elements));
        int64_t bytes = (int64_t)entry.data.size();
        ofs.write(reinterpret_cast<const char*>(&bytes), sizeof(bytes));
        if (bytes > 0) {
            ofs.write(reinterpret_cast<const char*>(entry.data.data()), (std::streamsize)bytes);
        }
    }
    ofs.close();
}

size_t QuantizedStore::num_stored() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return storage_.size();
}

size_t QuantizedStore::cache_size_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_cache_bytes_;
}

size_t QuantizedStore::lru_capacity_bytes() const {
    return lru_capacity_bytes_;
}

size_t QuantizedStore::total_storage_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t total = 0;
    for (auto& [id, entry] : storage_) {
        total += entry.data.size();
    }
    return total;
}

void QuantizedStore::evict_lru() {
    if (lru_order_.empty()) return;
    int64_t oldest = lru_order_.back();
    lru_order_.pop_back();
    auto cit = lru_cache_.find(oldest);
    if (cit != lru_cache_.end()) {
        current_cache_bytes_ -= cit->second.size_bytes;
        // Optionally offload to backing file
        auto sit = storage_.find(oldest);
        if (sit != storage_.end() && !backing_file_.empty()) {
            // Already in storage, no need to write back since cache is derived
        }
        lru_cache_.erase(cit);
    }
}

void QuantizedStore::validate_handle(int64_t handle) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (storage_.find(handle) == storage_.end()) {
        throw std::runtime_error("QuantizedStore: invalid handle " + std::to_string(handle));
    }
}

} // namespace oil
