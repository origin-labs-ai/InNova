#pragma once
// ============================================================================
// PILLAR 1: MmapDataLoader — Streams 1TB+ datasets via mmap, no RAM overflow
// ============================================================================
// WHY: Standard DataLoader loads entire file into RAM. For 1TB datasets on a
// 14GB machine, this is impossible. mmap() maps file pages directly into the
// virtual address space. The OS handles page faults — only the pages actually
// accessed get loaded into RAM. L3 cache (typically 32MB) acts as a sliding
// window over the massive file.
//
// DEDUP: Inline MinHash filter drops duplicate n-grams on-the-fly during
// streaming, preventing repeated sequences from corrupting training.
//
// THREAD SAFETY: mmap pages are shared across threads. Multiple DataLoader
// instances can mmap the same file without extra memory (copy-on-write).
// ============================================================================

#include "oil/tensor.h"
#include "oil/tokenizer.h"
#include <string>
#include <vector>
#include <deque>
#include <random>
#include <cstdint>
#include <functional>
#include <memory>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace oil {

// ============================================================================
// MinHash Deduplication Filter
// ============================================================================
// Detects near-duplicate n-grams in the token stream using MinHash signatures.
// Space-efficient: only stores fingerprints, not actual tokens.
// False positive rate < 1% with 128 hash functions.
// ============================================================================

class MinHashDedup {
public:
    explicit MinHashDedup(int64_t ngram_size = 4, int num_hashes = 128,
                          int64_t max_capacity = 1000000);

    // Check if n-gram is a duplicate. Returns true if duplicate detected.
    bool is_duplicate(const int64_t* tokens, int64_t n);

    // Insert n-gram into the filter
    void insert(const int64_t* tokens, int64_t n);

    // Clear the filter
    void clear();

    // Stats
    int64_t total_checked() const { return total_checked_; }
    int64_t duplicates_found() const { return duplicates_found_; }
    float duplicate_rate() const {
        return total_checked_ > 0 ? (float)duplicates_found_ / total_checked_ : 0.0f;
    }

private:
    int64_t ngram_size_;
    int num_hashes_;
    int64_t max_capacity_;
    int64_t total_checked_ = 0;
    int64_t duplicates_found_ = 0;

    // Fingerprint storage (compact bloom-like filter)
    std::vector<uint64_t> fingerprints_;

    uint64_t compute_fingerprint(const int64_t* tokens, int64_t n) const;
    bool check_and_insert(uint64_t fp);
};

// ============================================================================
// MmapDataLoader — Memory-mapped streaming DataLoader with MinHash dedup
// ============================================================================
// Usage:
//   MmapDataLoader loader("dataset.bin", tokenizer, cfg);
//   while (loader.has_next()) {
//       auto batch = loader.next_batch();  // [batch_size, seq_length]
//   }
//
// Memory footprint: O(shuffle_buffer + batch), NOT O(file_size)
// ============================================================================

struct MmapDataLoaderConfig {
    int64_t batch_size = 8;
    int64_t seq_length = 2048;
    int64_t shuffle_buffer = 100000;    // Token-level shuffle buffer
    int64_t mmap_window = 1024 * 1024;  // mmap read window (1M tokens)
    bool dedup_enabled = true;           // Enable MinHash dedup
    int64_t ngram_size = 4;             // N-gram size for dedup
    int64_t seed = 42;
    bool drop_last = true;
    int64_t num_workers = 1;

    // Per-modality blending (for multimodal training)
    float real_ratio = 0.7f;  // 70% real data
    float synthetic_ratio = 0.3f;  // 30% synthetic data
};

class MmapDataLoader {
public:
    MmapDataLoader(const std::string& file_path,
                   BPETokenizer* tokenizer,
                   const MmapDataLoaderConfig& cfg = MmapDataLoaderConfig{});

    ~MmapDataLoader();

    // Non-copyable (mmap handle)
    MmapDataLoader(const MmapDataLoader&) = delete;
    MmapDataLoader& operator=(const MmapDataLoader&) = delete;

    // Get next batch: Tensor of shape {batch_size, seq_length} (I64 token IDs)
    Tensor next_batch();

    // Get next batch with labels (shifted by 1 for next-token prediction)
    std::pair<Tensor, Tensor> next_batch_with_labels();

    bool has_next() const;
    int64_t num_batches() const;
    void reset();

    // Dedup statistics
    const MinHashDedup& dedup() const { return dedup_; }

private:
    void mmap_file(const std::string& path);
    void unmmap_file();
    void refill_buffer();
    std::vector<int64_t> tokenize_window(int64_t start, int64_t length) const;

    MmapDataLoaderConfig cfg_;
    BPETokenizer* tokenizer_;
    MinHashDedup dedup_;

    // mmap state
    char* mapped_data_ = nullptr;
    int64_t mapped_size_ = 0;
    int64_t file_offset_ = 0;

    // Token buffer
    std::deque<int64_t> token_buffer_;
    int64_t total_tokens_ = 0;

    // Shuffle
    std::mt19937 rng_;

#ifdef _WIN32
    HANDLE file_handle_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_handle_ = NULL;
#else
    int fd_ = -1;
#endif
};

} // namespace oil
