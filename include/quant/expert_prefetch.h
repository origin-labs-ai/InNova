#pragma once
#include <quant/types.h>
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cstring>
#include <functional>

namespace quant {

// Expert weight page - represents one expert's weights in memory
struct SourceSegment {
    const void* ptr = nullptr;
    size_t bytes = 0;
};

struct ExpertPage {
    void* host_ptr = nullptr;       // Host memory (possibly page-locked)
    void* device_ptr = nullptr;     // GPU device memory (if GPU backend active)
    const void* source_ptr = nullptr; // Single contiguous source (when expert weights are packed)
    std::vector<SourceSegment> source_segments; // Multi-segment source (gate/up/down separate tensors)
    size_t size_bytes = 0;
    int expert_id = -1;
    int layer_id = -1;
    bool is_pinned = false;         // Is host memory page-locked?
    bool is_on_device = false;      // Is data currently on GPU?
    std::atomic<bool> transfer_in_progress{false};

    ExpertPage() = default;
    ExpertPage(const ExpertPage& other)
        : host_ptr(other.host_ptr), device_ptr(other.device_ptr),
          source_ptr(other.source_ptr), source_segments(other.source_segments),
          size_bytes(other.size_bytes), expert_id(other.expert_id),
          layer_id(other.layer_id), is_pinned(other.is_pinned),
          is_on_device(other.is_on_device),
          transfer_in_progress(other.transfer_in_progress.load()) {}

    ExpertPage(ExpertPage&& other) noexcept
        : host_ptr(other.host_ptr), device_ptr(other.device_ptr),
          source_ptr(other.source_ptr), source_segments(std::move(other.source_segments)),
          size_bytes(other.size_bytes), expert_id(other.expert_id),
          layer_id(other.layer_id), is_pinned(other.is_pinned),
          is_on_device(other.is_on_device),
          transfer_in_progress(other.transfer_in_progress.load()) {}

    ExpertPage& operator=(const ExpertPage& other) {
        if (this != &other) {
            host_ptr = other.host_ptr;
            device_ptr = other.device_ptr;
            source_ptr = other.source_ptr;
            source_segments = other.source_segments;
            size_bytes = other.size_bytes;
            expert_id = other.expert_id;
            layer_id = other.layer_id;
            is_pinned = other.is_pinned;
            is_on_device = other.is_on_device;
            transfer_in_progress.store(other.transfer_in_progress.load());
        }
        return *this;
    }

    ExpertPage& operator=(ExpertPage&& other) noexcept {
        if (this != &other) {
            host_ptr = other.host_ptr;
            device_ptr = other.device_ptr;
            source_ptr = other.source_ptr;
            source_segments = std::move(other.source_segments);
            size_bytes = other.size_bytes;
            expert_id = other.expert_id;
            layer_id = other.layer_id;
            is_pinned = other.is_pinned;
            is_on_device = other.is_on_device;
            transfer_in_progress.store(other.transfer_in_progress.load());
        }
        return *this;
    }
};

class ExpertPrefetcher {
public:
    ExpertPrefetcher(int num_experts, int num_layers, size_t expert_size,
                     int prefetch_ahead = 1, int max_resident = 8);
    ~ExpertPrefetcher();
    
    // Initialize: allocate pages, pin memory
    void initialize();
    
    // Register the source data pointer(s) for an expert (from model's in-memory weights).
    // Use the contiguous form when the expert blob is packed, or the segmented form
    // when gate/up/down live in separate tensors.
    void set_expert_source(int layer_id, int expert_id, const void* src_ptr);
    void set_expert_segments(int layer_id, int expert_id,
                             const std::vector<SourceSegment>& segments);

    // Called by router: which experts will be needed for next layer?
    void schedule_prefetch(int layer_id, const std::vector<int>& expert_ids);
    
    // Get expert weights (blocks until available)
    const float* get_expert_weights(int layer_id, int expert_id);
    
    // Release expert (mark as evictable)
    void release_expert(int layer_id, int expert_id);
    
    // Stats
    struct Stats {
        uint64_t prefetch_hits = 0;
        uint64_t prefetch_misses = 0;
        uint64_t evictions = 0;
        double avg_wait_ms = 0.0;
    };
    Stats get_stats() const;
    
private:
    void prefetch_thread_func();
    void evict_lru();
    void load_from_disk(ExpertPage& page);
    void upload_to_device(ExpertPage& page);
    void save_to_disk(ExpertPage& page);

    int num_experts_;
    int num_layers_;
    size_t expert_size_;
    int prefetch_ahead_;
    int max_resident_;
    std::string cache_dir_;
    
    std::vector<std::vector<ExpertPage>> pages_;
    std::vector<std::pair<int, int>> lru_list_;
    
    std::thread prefetch_thread_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_flag_ = false;
    
    struct PrefetchRequest {
        int layer_id;
        int expert_id;
    };
    std::vector<PrefetchRequest> prefetch_queue_;
    
    // Double buffers for prefetching
    std::vector<void*> prefetch_buffers_;
    int current_buffer_idx_ = 0;
    
    // GPU copy stream for async transfers
    void* copy_stream_ = nullptr;
    
    Stats stats_;
};

} // namespace quant
