#include "quant/expert_prefetch.h"
#include <iostream>
#include <algorithm>
#include <chrono>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace quant {

ExpertPrefetcher::ExpertPrefetcher(int num_experts, int num_layers, size_t expert_size,
                                   int prefetch_ahead, int max_resident)
    : num_experts_(num_experts),
      num_layers_(num_layers),
      expert_size_(expert_size),
      prefetch_ahead_(prefetch_ahead),
      max_resident_(max_resident)
{
    pages_.resize(num_layers_);
    for (int l = 0; l < num_layers_; ++l) {
        pages_[l].resize(num_experts_);
        for (int e = 0; e < num_experts_; ++e) {
            pages_[l][e].expert_id = e;
            pages_[l][e].layer_id = l;
            pages_[l][e].size_bytes = expert_size_;
        }
    }
}

ExpertPrefetcher::~ExpertPrefetcher() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_flag_ = true;
    }
    cv_.notify_one();
    if (prefetch_thread_.joinable()) {
        prefetch_thread_.join();
    }
    
    // Unpin memory
    for (int l = 0; l < num_layers_; ++l) {
        for (int e = 0; e < num_experts_; ++e) {
            if (pages_[l][e].is_pinned && pages_[l][e].host_ptr) {
#if defined(_WIN32)
                VirtualUnlock(pages_[l][e].host_ptr, pages_[l][e].size_bytes);
#else
                munlock(pages_[l][e].host_ptr, pages_[l][e].size_bytes);
#endif
                pages_[l][e].is_pinned = false;
            }
        }
    }
}

void ExpertPrefetcher::initialize() {
    prefetch_thread_ = std::thread(&ExpertPrefetcher::prefetch_thread_func, this);
}

void ExpertPrefetcher::schedule_prefetch(int layer_id, const std::vector<int>& expert_ids) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (int e : expert_ids) {
        if (!pages_[layer_id][e].is_on_device && !pages_[layer_id][e].transfer_in_progress) {
            prefetch_queue_.push_back({layer_id, e});
        }
    }
    cv_.notify_one();
}

const float* ExpertPrefetcher::get_expert_weights(int layer_id, int expert_id) {
    auto start_time = std::chrono::steady_clock::now();
    ExpertPage& page = pages_[layer_id][expert_id];
    
    bool waited = false;
    while (page.transfer_in_progress.load(std::memory_order_acquire)) {
        std::this_thread::yield();
        waited = true;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    if (!page.is_on_device) {
        // Miss! We have to do it synchronously if it wasn't prefetch queued or finished.
        stats_.prefetch_misses++;
        // Actually load/pin it synchronously here if needed.
        // For CPU only, pinning means page-locked host memory.
        if (!page.is_pinned && page.host_ptr) {
#if defined(_WIN32)
            VirtualLock(page.host_ptr, page.size_bytes);
#else
            mlock(page.host_ptr, page.size_bytes);
#endif
            page.is_pinned = true;
        }
        page.is_on_device = true;
        
        // LRU update
        evict_lru();
        lru_list_.push_back({layer_id, expert_id});
    } else {
        if (waited) {
            stats_.prefetch_misses++; // If we had to wait, count as a miss for timing purposes.
        } else {
            stats_.prefetch_hits++;
        }
        
        // Update LRU: move to back
        auto it = std::find_if(lru_list_.begin(), lru_list_.end(), [&](const std::pair<int, int>& p) {
            return p.first == layer_id && p.second == expert_id;
        });
        if (it != lru_list_.end()) {
            std::pair<int, int> val = *it;
            lru_list_.erase(it);
            lru_list_.push_back(val);
        }
    }
    
    if (waited) {
        auto end_time = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        double total_wait = stats_.avg_wait_ms * (stats_.prefetch_hits + stats_.prefetch_misses - 1) + ms;
        stats_.avg_wait_ms = total_wait / (stats_.prefetch_hits + stats_.prefetch_misses);
    }
    
    return reinterpret_cast<const float*>(page.host_ptr); // or device_ptr if using GPU
}

void ExpertPrefetcher::release_expert(int layer_id, int expert_id) {
    // We don't immediately unpin, we rely on LRU eviction when limit is reached.
    // This allows re-use if the expert is requested again soon.
}

void ExpertPrefetcher::evict_lru() {
    while ((int)lru_list_.size() >= max_resident_) {
        auto [l, e] = lru_list_.front();
        lru_list_.erase(lru_list_.begin());
        
        ExpertPage& evict_page = pages_[l][e];
        if (evict_page.is_pinned && evict_page.host_ptr) {
#if defined(_WIN32)
            VirtualUnlock(evict_page.host_ptr, evict_page.size_bytes);
#else
            munlock(evict_page.host_ptr, evict_page.size_bytes);
#endif
            evict_page.is_pinned = false;
        }
        evict_page.is_on_device = false;
        stats_.evictions++;
    }
}

void ExpertPrefetcher::prefetch_thread_func() {
    while (true) {
        PrefetchRequest req;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this]() { return stop_flag_ || !prefetch_queue_.empty(); });
            
            if (stop_flag_ && prefetch_queue_.empty()) {
                break;
            }
            
            req = prefetch_queue_.front();
            prefetch_queue_.erase(prefetch_queue_.begin());
            
            pages_[req.layer_id][req.expert_id].transfer_in_progress.store(true, std::memory_order_release);
        }
        
        ExpertPage& page = pages_[req.layer_id][req.expert_id];
        
        // Perform the pinning
        if (!page.is_pinned && page.host_ptr) {
#if defined(_WIN32)
            VirtualLock(page.host_ptr, page.size_bytes);
#else
            mlock(page.host_ptr, page.size_bytes);
#endif
            page.is_pinned = true;
        }
        
        {
            std::lock_guard<std::mutex> lock(mutex_);
            page.is_on_device = true;
            page.transfer_in_progress.store(false, std::memory_order_release);
            
            // LRU logic
            auto it = std::find_if(lru_list_.begin(), lru_list_.end(), [&](const std::pair<int, int>& p) {
                return p.first == req.layer_id && p.second == req.expert_id;
            });
            if (it == lru_list_.end()) {
                evict_lru();
                lru_list_.push_back({req.layer_id, req.expert_id});
            } else {
                std::pair<int, int> val = *it;
                lru_list_.erase(it);
                lru_list_.push_back(val);
            }
        }
    }
}

ExpertPrefetcher::Stats ExpertPrefetcher::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

} // namespace quant
