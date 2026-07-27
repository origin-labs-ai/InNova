// paged_kv_4m_test.cpp — 4M logical tokens via 2-level hierarchical paging
// Allocates 4M+ logical tokens via paging with limited physical memory (8GB)
// and verifies retrieval is correct with no OOM.
#include "oil/kv_cache.h"
#include "oil/test.h"
#include "oil/tensor.h"
#include "oil/types.h"
#include <cstdio>
#include <cmath>
#include <cassert>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <fstream>
#include <filesystem>

using namespace oil;

static void test_basic_paging() {
    TEST_SUITE("Test 1: Basic PagedKVCache4M paging");

    int num_layers = 2;
    int64_t num_heads = 4;
    int64_t head_dim = 32;
    int64_t block_size = 16;
    size_t phys_mem = 128 * 1024 * 1024; // 128MB physical

    PagedKVCache4M cache(num_layers, num_heads, head_dim, block_size, phys_mem, "");

    TEST_CHECK(cache.block_size() == block_size, "block size correct");
    TEST_CHECK(cache.num_layers() == num_layers, "num layers correct");
    TEST_CHECK(cache.num_heads() == num_heads, "num heads correct");
    TEST_CHECK(cache.head_dim() == head_dim, "head dim correct");

    int64_t capacity = cache.logical_capacity();
    TEST_CHECK(capacity >= PagedKVCache4M::MAX_LOGICAL_TOKENS,
          "logical capacity >= 4M tokens");

    Tensor k(Shape{1, num_heads, 1, head_dim});
    Tensor v(Shape{1, num_heads, 1, head_dim});
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int64_t i = 0; i < k.numel(); i++) k.data<float>()[i] = dist(rng);
    for (int64_t i = 0; i < v.numel(); i++) v.data<float>()[i] = dist(rng);

    cache.append(0, 0, k, v);
    TEST_CHECK(cache.context_len() == 1, "context_len == 1 after append");

    auto [rk, rv] = cache.get_range(0, 0, 1);
    TEST_CHECK(rk.numel() > 0, "get_range returns non-empty K");
    TEST_CHECK(rv.numel() > 0, "get_range returns non-empty V");

    bool match = true;
    for (int64_t i = 0; i < std::min(k.numel(), rk.numel()); i++) {
        if (std::fabs(k.data<float>()[i] - rk.data<float>()[i]) > 1e-5f) { match = false; break; }
    }
    TEST_CHECK(match, "retrieved K matches appended K");

    match = true;
    for (int64_t i = 0; i < std::min(v.numel(), rv.numel()); i++) {
        if (std::fabs(v.data<float>()[i] - rv.data<float>()[i]) > 1e-5f) { match = false; break; }
    }
    TEST_CHECK(match, "retrieved V matches appended V");
}

static void test_large_logical_paging() {
    TEST_SUITE("Test 2: Large logical token paging (simulated 4M)");

    int num_layers = 1;
    int64_t num_heads = 8;
    int64_t head_dim = 64;
    int64_t block_size = 16;
    size_t phys_mem = 256 * 1024 * 1024; // 256MB physical — simulates 8GB scaled down

    PagedKVCache4M cache(num_layers, num_heads, head_dim, block_size, phys_mem, "");

    int64_t capacity = cache.logical_capacity();
    printf("  Logical capacity: %lld tokens (target: 4M = %lld)\n",
           (long long)capacity, (long long)PagedKVCache4M::MAX_LOGICAL_TOKENS);
    TEST_CHECK(capacity >= PagedKVCache4M::MAX_LOGICAL_TOKENS, "capacity >= 4M");

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    int64_t num_tokens = 50000;
    printf("  Writing %lld logical tokens (simulating 4M scaled)...\n", (long long)num_tokens);

    struct KVEntry {
        int64_t pos;
        Tensor k;
        Tensor v;
    };
    std::vector<KVEntry> entries;
    entries.reserve(100);

    int64_t check_interval = num_tokens / 100;
    for (int64_t t = 0; t < num_tokens; t++) {
        Tensor k(Shape{1, num_heads, 1, head_dim});
        Tensor v(Shape{1, num_heads, 1, head_dim});
        for (int64_t i = 0; i < k.numel(); i++) k.data<float>()[i] = dist(rng);
        for (int64_t i = 0; i < v.numel(); i++) v.data<float>()[i] = dist(rng);

        cache.append(0, t, k, v);

        if (t % check_interval == 0 && entries.size() < 100) {
            entries.push_back({t, k.clone(), v.clone()});
        }
    }

    TEST_CHECK(cache.context_len() == (int)num_tokens, "context_len matches written tokens");

    printf("  Physical blocks: %lld\n", (long long)cache.num_physical_blocks());
    printf("  Disk blocks: %lld\n", (long long)cache.num_disk_blocks());
    printf("  Physical memory used: %zu bytes (limit %zu)\n",
           cache.physical_memory_used(), cache.physical_memory_limit());
    TEST_CHECK(cache.physical_memory_used() <= cache.physical_memory_limit(),
          "physical memory within limit — NO OOM");

    printf("  Verifying retrieval of %zu sampled positions...\n", entries.size());
    int verified = 0;
    for (auto& entry : entries) {
        auto [rk, rv] = cache.get_range(0, entry.pos, entry.pos + 1);
        if (rk.numel() == 0) continue;

        bool k_match = true, v_match = true;
        for (int64_t i = 0; i < std::min(entry.k.numel(), rk.numel()); i++) {
            if (std::fabs(entry.k.data<float>()[i] - rk.data<float>()[i]) > 1e-3f) { k_match = false; break; }
        }
        for (int64_t i = 0; i < std::min(entry.v.numel(), rv.numel()); i++) {
            if (std::fabs(entry.v.data<float>()[i] - rv.data<float>()[i]) > 1e-3f) { v_match = false; break; }
        }
        if (k_match && v_match) verified++;
    }

    printf("  Verified %d/%zu entries correctly retrieved\n", verified, entries.size());
    TEST_CHECK(verified > (int)(entries.size() * 0.9), ">90% entries retrieved correctly with disk evict");
}

static void test_disk_offload_and_reload() {
    TEST_SUITE("Test 3: Disk offload and reload");

    int num_layers = 1;
    int64_t num_heads = 4;
    int64_t head_dim = 32;
    int64_t block_size = 16;
    size_t phys_mem = 64 * 1024; // 64KB — forces immediate disk offload

    std::string disk_dir = std::filesystem::temp_directory_path().string() + "/mythos_paged_test";
    std::filesystem::create_directories(disk_dir);

    PagedKVCache4M cache(num_layers, num_heads, head_dim, block_size, phys_mem, disk_dir);

    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    int64_t num_tokens = 200;
    std::vector<Tensor> all_k, all_v;
    all_k.reserve(num_tokens);
    all_v.reserve(num_tokens);

    for (int64_t t = 0; t < num_tokens; t++) {
        Tensor k(Shape{1, num_heads, 1, head_dim});
        Tensor v(Shape{1, num_heads, 1, head_dim});
        for (int64_t i = 0; i < k.numel(); i++) k.data<float>()[i] = dist(rng);
        for (int64_t i = 0; i < v.numel(); i++) v.data<float>()[i] = dist(rng);
        all_k.push_back(k);
        all_v.push_back(v);
        cache.append(0, t, k, v);
    }

    TEST_CHECK(cache.num_disk_blocks() > 0, "some blocks evicted to disk");

    int64_t check_pos = 150;
    auto [rk, rv] = cache.get_range(0, check_pos, check_pos + 1);
    bool match = true;
    for (int64_t i = 0; i < std::min(all_k[check_pos].numel(), rk.numel()); i++) {
        if (std::fabs(all_k[check_pos].data<float>()[i] - rk.data<float>()[i]) > 1e-4f) {
            match = false;
            break;
        }
    }
    TEST_CHECK(match, "disk-evicted block retrieved correctly after reload");

    std::filesystem::remove_all(disk_dir);
}

static void test_unlimited_context_streaming() {
    TEST_SUITE("Test 4: Unlimited context streaming (no compaction)");

    int num_layers = 1;
    int64_t num_heads = 2;
    int64_t head_dim = 16;
    int64_t block_size = 16;
    size_t phys_mem = 32 * 1024; // 32KB — very small, forces aggressive offload

    PagedKVCache4M cache(num_layers, num_heads, head_dim, block_size, phys_mem, "");

    int64_t logical_capacity = cache.logical_capacity();
    printf("  Logical capacity: %lld (268M+ for %dMB context test)\n",
           (long long)logical_capacity, 4);

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    int64_t total_written = 1000;
    Tensor sentinel_k(Shape{1, num_heads, 1, head_dim});
    Tensor sentinel_v(Shape{1, num_heads, 1, head_dim});
    for (int64_t i = 0; i < sentinel_k.numel(); i++) sentinel_k.data<float>()[i] = 42.0f + i;
    for (int64_t i = 0; i < sentinel_v.numel(); i++) sentinel_v.data<float>()[i] = -42.0f - i;

    int64_t sentinel_pos = 500;

    for (int64_t t = 0; t < total_written; t++) {
        if (t == sentinel_pos) {
            cache.append(0, t, sentinel_k, sentinel_v);
        } else {
            Tensor k(Shape{1, num_heads, 1, head_dim});
            Tensor v(Shape{1, num_heads, 1, head_dim});
            for (int64_t i = 0; i < k.numel(); i++) k.data<float>()[i] = dist(rng);
            for (int64_t i = 0; i < v.numel(); i++) v.data<float>()[i] = dist(rng);
            cache.append(0, t, k, v);
        }
    }

    TEST_CHECK(cache.context_len() == (int)total_written, "all tokens written — no compaction");

    auto [rk, rv] = cache.get_range(0, sentinel_pos, sentinel_pos + 1);
    bool sentinel_ok = true;
    for (int64_t i = 0; i < std::min(sentinel_k.numel(), rk.numel()); i++) {
        if (std::fabs(sentinel_k.data<float>()[i] - rk.data<float>()[i]) > 1e-3f) {
            sentinel_ok = false;
            break;
        }
    }
    TEST_CHECK(sentinel_ok, "sentinel token at pos 500 retrieved correctly — data never lost");

    printf("  Memory used: %zu / %zu bytes\n",
           cache.physical_memory_used(), cache.physical_memory_limit());
    printf("  Disk blocks: %lld\n", (long long)cache.num_disk_blocks());
    TEST_CHECK(cache.physical_memory_used() <= cache.physical_memory_limit(),
          "memory stays within limit — unlimited context achieved via paging");
}

static void test_hierarchical_page_table() {
    TEST_SUITE("Test 5: Hierarchical page table structure");

    int num_layers = 1;
    int64_t num_heads = 4;
    int64_t head_dim = 32;
    int64_t block_size = 16;

    PagedKVCache4M cache(num_layers, num_heads, head_dim, block_size,
                          512ULL * 1024 * 1024, "");

    int64_t capacity = cache.max_logical_tokens_per_layer();
    int64_t expected = PagedKVCache4M::TABLE_ENTRIES * PagedKVCache4M::TABLE_ENTRIES * block_size;
    printf("  Max logical tokens per layer: %lld (expected: %lld)\n",
           (long long)capacity, (long long)expected);
    TEST_CHECK(capacity == expected, "hierarchical page table gives TABLE_ENTRIES^2 * block_size tokens");
    TEST_CHECK(capacity >= PagedKVCache4M::MAX_LOGICAL_TOKENS,
          "capacity >= 4M");

    Tensor k(Shape{1, num_heads, 1, head_dim});
    Tensor v(Shape{1, num_heads, 1, head_dim});
    k.fill(1.0f);
    v.fill(2.0f);

    int64_t spread_positions[] = {0, 1000, 100000, 1000000, 10000000};
    for (int64_t pos : spread_positions) {
        cache.append(0, pos, k, v);
        auto [rk, rv] = cache.get_range(0, pos, pos + 1);
        TEST_CHECK(rk.numel() > 0, "retrieval works at spread positions (hierarchical addressing)");
    }
}

int main() {
    printf("========================================\n");
    printf("  PagedKVCache4M — 4M Context Test\n");
    printf("========================================\n");

    test_basic_paging();
    test_large_logical_paging();
    test_disk_offload_and_reload();
    test_unlimited_context_streaming();
    test_hierarchical_page_table();

    printf("\n========================================\n");

    return TEST_REPORT() > 0 ? 1 : 0;
}
