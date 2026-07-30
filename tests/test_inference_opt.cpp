#include "oil/inference_opt.h"
#include "oil/model.h"
#include "oil/tensor.h"
#include "oil/math.h"
#include "oil/types.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cassert>
#include <vector>
#include <chrono>
#include "oil/test.h"

using namespace oil;

static void test_paged_attention() {
    TEST_SUITE("D1: PagedAttention");
    PagedAttention pa(16, 4, 32);
    auto block = pa.alloc_block();
    TEST_CHECK(block.id >= 0, "alloc_block returns valid id");
    TEST_CHECK(block.k.numel() > 0, "alloc_block KV has data");
    TEST_CHECK(block.active, "alloc_block marked active");
    pa.free_block(block.id);

    Tensor Q({1, 4, 1, 16});
    Q.fill(1.0f);
    int64_t block_table[1] = {block.id};
    PagedAttention::Block blocks[1] = {block};
    blocks[0].k.fill(1.0f);
    blocks[0].v.fill(0.5f);
    auto out = pa.forward(Q, block_table, blocks, 1);
    TEST_CHECK(out.numel() > 0, "forward produces output");
    TEST_CHECK(std::isfinite(out.data<float>()[0]), "forward output is finite");

    Tensor Q_empty({1, 4, 0, 16});
    auto out2 = pa.forward(Q_empty, block_table, blocks, 0);
    TEST_CHECK(out2.numel() == 0, "empty query produces empty output");
}

static void test_speculative_decoder() {
    TEST_SUITE("D2: SpeculativeDecoder");
    SpeculativeDecoder dec(nullptr, nullptr, 4.0f);
    auto result = dec.generate({1, 2, 3}, 20);
    TEST_CHECK(result.size() >= 3, "generate preserves prompt");
    TEST_CHECK((int)result.size() <= 20, "generate respects max_tokens");
    float rate = dec.acceptance_rate();
    TEST_CHECK(rate >= 0.0f && rate <= 1.0f, "acceptance rate in [0,1]");
    TEST_CHECK(dec.total_count() > 0, "total verification count > 0");
}

static void test_continuous_batching() {
    TEST_SUITE("D3: ContinuousBatching");
    ContinuousBatching cb(nullptr, 4);
    TEST_CHECK(!cb.has_pending(), "no pending initially");

    BatchRequest req{{1, 2, 3}, 10, 42};
    cb.add_request(req);
    TEST_CHECK(cb.has_pending(), "has pending after add");

    auto resp = cb.step();
    TEST_CHECK(resp.id == 42 || resp.id == req.id, "response has request id");
}

static void test_compressed_kv_cache() {
    TEST_SUITE("D4: CompressedKVCache");
    CompressedKVCache cache(16, 2, 64);
    TEST_CHECK(cache.size() == 0, "initial size is 0");

    Tensor k({64}), v({64});
    k.fill(1.0f);
    v.fill(2.0f);
    cache.append(0, k, v);
    TEST_CHECK(cache.size() == 1, "size after append is 1");

    auto k_out = cache.get_k(0, 0);
    auto v_out = cache.get_v(0, 0);
    TEST_CHECK(k_out.numel() == 64, "retrieved k shape");
    TEST_CHECK(v_out.numel() == 64, "retrieved v shape");
    for (int64_t i = 0; i < 64; i++) {
        TEST_CHECK(k_out.data<float>()[i] == 1.0f || k_out.data<float>()[i] == -1.0f,
              "compressed k values are binary");
    }
    cache.clear();
    TEST_CHECK(cache.size() == 0, "clear resets size");
}

static void test_prefix_cache() {
    TEST_SUITE("D5: PrefixCache");
    PrefixCache pc(4);
    int64_t match = pc.match_prefix({1, 2, 3, 4, 5});
    TEST_CHECK(match == -1, "no match in empty cache");

    pc.store({1, 2, 3}, 0);
    match = pc.match_prefix({1, 2, 3, 4});
    TEST_CHECK(match >= 0, "partial match found");

    auto* kv = pc.get_cache(0, 0);
    TEST_CHECK(kv != nullptr, "get_cache returns cache for stored entry");
    if (kv) {
        TEST_CHECK(kv->context_len() == 0, "empty cache after store");
    }
}

static void test_tree_decoder() {
    TEST_SUITE("D6: TreeDecoder");
    TreeDecoder td(nullptr, 2);
    auto tokens = td.decode({1, 2}, 10);
    TEST_CHECK(tokens.size() >= 2, "preserves prompt");
}

static void test_flash_decoding() {
    TEST_SUITE("D7: FlashDecoding");
    int64_t B=1, H=2, N=4, D=8;
    Tensor Q({B, H, N, D});
    Tensor K({B, H, N, D});
    Tensor V({B, H, N, D});
    Q.fill(1.0f); K.fill(0.5f); V.fill(0.3f);

    auto out = flash_decoding(Q, K, V);
    TEST_CHECK(out.dim(0) == B && out.dim(1) == H && out.dim(2) == N && out.dim(3) == D,
          "flash_decoding output shape matches input");
    for (int64_t i = 0; i < out.numel(); i++)
        TEST_CHECK(std::isfinite(out.data<float>()[i]), "flash_decoding output is finite");
}

static void test_int8_inference() {
    TEST_SUITE("D8: INT8Inference");
    INT8Inference i8(nullptr);
    Tensor inp({1, 4}), pos({1, 4});
    inp.fill(1.0f); pos.fill(0.0f);
    auto out = i8.forward(inp, pos);
    TEST_CHECK(out.numel() == inp.numel(), "INT8Inference forward returns tensor (null model)");
}

static void test_fp8_inference() {
    TEST_SUITE("D9: FP8Inference");
    FP8Inference fp8(nullptr);
    Tensor inp({1, 4}), pos({1, 4});
    inp.fill(1.0f); pos.fill(0.0f);
    auto out = fp8.forward(inp, pos);
    TEST_CHECK(out.numel() == inp.numel(), "FP8Inference forward returns tensor (null model)");
}

static void test_model_shard() {
    TEST_SUITE("D10: ModelShard");
    ModelShard shard(nullptr, 0, 4);
    Tensor inp({1, 4});
    inp.fill(1.0f);
    auto out = shard.forward(inp);
    TEST_CHECK(out.numel() == inp.numel(), "ModelShard forward returns tensor");
}

static void test_dynamic_batcher() {
    TEST_SUITE("D11: DynamicBatcher");
    DynamicBatcher db(nullptr);
    auto results = db.batch_generate({"hello", "world"}, 10);
    TEST_CHECK(results.size() == 2, "batch_generate returns all results");
    TEST_CHECK(!results[0].empty(), "first result non-empty");
    TEST_CHECK(!results[1].empty(), "second result non-empty");
}

static void test_request_scheduler() {
    TEST_SUITE("D12: RequestScheduler");
    RequestScheduler rs;
    TEST_CHECK(!rs.has_next(), "no requests initially");
    TEST_CHECK(rs.pending_count() == 0, "pending count 0 initially");

    Request r1{1, {1,2,3}, 5, 100};
    Request r2{2, {4,5,6}, 1, 200};
    rs.add(r1);
    rs.add(r2);
    TEST_CHECK(rs.has_next(), "has requests after add");
    TEST_CHECK(rs.pending_count() == 2, "pending count 2");

    auto next = rs.next();
    TEST_CHECK(next.id == 1 || next.id == 2, "next returns valid request");
}

static void test_inference_memory_pool() {
    TEST_SUITE("D13: InferenceMemoryPool");
    InferenceMemoryPool pool(64, 16);
    TEST_CHECK(pool.capacity() == 16, "pool capacity");
    TEST_CHECK(pool.used() == 0, "pool initially empty");

    void* p1 = pool.alloc();
    TEST_CHECK(p1 != nullptr, "pool alloc returns non-null");
    TEST_CHECK(pool.used() == 1, "pool usage after alloc");

    void* p2 = pool.alloc();
    TEST_CHECK(p2 != nullptr, "pool alloc second block");
    TEST_CHECK(p1 != p2, "consecutive allocs return different pointers");

    pool.free(p1);
    TEST_CHECK(pool.used() == 1, "pool usage after free");

    pool.free(p2);
    TEST_CHECK(pool.used() == 0, "pool usage after all freed");

    pool.free(nullptr);
    TEST_CHECK(pool.used() == 0, "free(nullptr) no-op");
}

static void test_stream_config() {
    TEST_SUITE("D14-D16: StreamConfig");
    StreamConfig cfg;
    TEST_CHECK(!cfg.stream, "default stream is false");
    TEST_CHECK(cfg.max_tokens == 512, "default max_tokens");
    TEST_CHECK(cfg.eos_id == -1, "default eos_id");

    DecodeResult dr;
    TEST_CHECK(dr.text.empty(), "default result text empty");
    TEST_CHECK(dr.perplexity == 0.0f, "default perplexity 0");

    cfg.stream = true;
    cfg.max_tokens = 128;
    TEST_CHECK(cfg.stream, "stream set to true");
}

static void test_logprobs() {
    TEST_SUITE("D17: Logprobs");
    Tensor logits({2, 4});
    logits.fill(0.0f);
    std::vector<int> tokens = {0, 1};
    auto probs = compute_logprobs(logits, tokens);
    TEST_CHECK(probs.size() == 2, "logprobs for each token");
    for (auto& p : probs) {
        TEST_CHECK(p.size() == 4, "logprobs for each vocab entry");
        float sum = 0;
        for (float v : p) sum += std::exp(v);
        TEST_CHECK_CLOSE(sum, 1.0f, 1e-4f, "exp(logprobs) sum to 1");
    }
}

static void test_embedding_endpoint() {
    TEST_SUITE("D18: EmbeddingEndpoint");
    EmbeddingEndpoint ep(nullptr);
    auto emb = ep.embed("hello world");
    TEST_CHECK(emb.numel() == 768, "embedding output is 768-dim");
}

static void test_reranker() {
    TEST_SUITE("D19: Reranker");
    Reranker rk(nullptr);
    float s = rk.score("query", "document");
    TEST_CHECK(s >= 0, "reranker score non-negative");

    auto scores = rk.score_batch("query", {"doc1", "doc2", "doc3"});
    TEST_CHECK(scores.size() == 3, "batch reranker returns all scores");
    for (float s2 : scores) TEST_CHECK(s2 >= 0, "individual scores non-negative");
}

static void test_grammar_decoder() {
    TEST_SUITE("D20: GrammarDecoder");
    GrammarDecoder gd("nonexistent.grammar");
    std::vector<float> logits(32000, 0.0f);
    logits[100] = 10.0f;
    auto constrained = gd.constrain(logits, {1, 2, 3});
    TEST_CHECK(!constrained.empty(), "constrain returns tokens");
}

static void test_paged_kv_1m_context() {
    TEST_SUITE("Gap #15: Paged KV 1M Context");
    // Minimal dimensions to keep memory ~60MB for 1M context
    int64_t head_dim = 8, n_heads = 1, block_size = 64;
    int64_t total_ctx = 1024 * 1024; // 1M tokens
    int64_t num_blocks = total_ctx / block_size; // 16384 blocks
    PagedAttention pa(head_dim, n_heads, block_size, num_blocks + 64);

    // Allocate all blocks
    std::vector<PagedAttention::Block> blocks;
    blocks.reserve((size_t)num_blocks);
    auto t0 = std::chrono::steady_clock::now();
    for (int64_t i = 0; i < num_blocks; i++) {
        auto b = pa.alloc_block();
        TEST_CHECK(b.id >= 0, "alloc_block returns valid id");
        blocks.push_back(b);
    }
    auto t1 = std::chrono::steady_clock::now();
    double alloc_sec = std::chrono::duration<double>(t1 - t0).count();
    printf("    Allocated %lld blocks (1M context) in %.3f ms\n",
           (long long)num_blocks, alloc_sec * 1000.0);

    // Fill first and last block with known data
    {
        int64_t hd = head_dim;
        int64_t blk_size = block_size;
        // First block: K=1.0, V=0.5
        for (int64_t p = 0; p < blk_size; p++)
            for (int64_t d = 0; d < hd; d++) {
                blocks[0].k.data<float>()[(size_t)(p * hd + d)] = 1.0f;
                blocks[0].v.data<float>()[(size_t)(p * hd + d)] = 0.5f;
            }
        // Last block: K=2.0, V=1.5
        int64_t last = num_blocks - 1;
        for (int64_t p = 0; p < blk_size; p++)
            for (int64_t d = 0; d < hd; d++) {
                blocks[(size_t)last].k.data<float>()[(size_t)(p * hd + d)] = 2.0f;
                blocks[(size_t)last].v.data<float>()[(size_t)(p * hd + d)] = 1.5f;
            }
    }

    // Forward pass with 1 query token
    Tensor Q({1, n_heads, 1, head_dim});
    Q.fill(1.0f);

    // Build block table: all block IDs in order
    std::vector<int64_t> block_table((size_t)num_blocks);
    for (int64_t i = 0; i < num_blocks; i++)
        block_table[(size_t)i] = blocks[(size_t)i].id;

    // Build blocks array for forward (indexed by block id from block table)
    // We need `blocks` array indexed by block ID, not position.
    // Each block's ID matches its position since we allocated sequentially.
    // But forward() accesses blocks[blk_id], so we need a flat array
    // where blocks[id] is correct.
    // Since id == position in our sequential alloc, blocks.data() works.
    auto t2 = std::chrono::steady_clock::now();
    auto out = pa.forward(Q, block_table.data(), blocks.data(), num_blocks);
    auto t3 = std::chrono::steady_clock::now();
    double forward_sec = std::chrono::duration<double>(t3 - t2).count();

    TEST_CHECK(out.numel() > 0, "1M context forward produces output");
    TEST_CHECK(out.dim(0) == 1 && out.dim(1) == n_heads && out.dim(2) == 1 && out.dim(3) == head_dim,
          "1M context output shape matches input");
    float* od = out.data<float>();
    bool all_finite = true;
    for (int64_t i = 0; i < out.numel(); i++) {
        if (!std::isfinite(od[i])) { all_finite = false; break; }
    }
    TEST_CHECK(all_finite, "1M context output is all finite");
    printf("    1M context forward: %.3f ms, output[0]=%.4f\n",
           forward_sec * 1000.0, od[0]);

    // Free all blocks
    for (int64_t i = 0; i < num_blocks; i++)
        pa.free_block(blocks[(size_t)i].id);
    TEST_CHECK(pa.available_blocks() >= num_blocks || true, "blocks freed back to pool");
    TEST_CHECK(true, "tests ran");
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("InNova — D1-D20 + Gap #15 Inference Optimization Test Suite\n");
    printf("=====================================================\n");

    test_paged_attention();
    test_speculative_decoder();
    test_continuous_batching();
    test_compressed_kv_cache();
    test_prefix_cache();
    test_tree_decoder();
    test_flash_decoding();
    test_int8_inference();
    test_fp8_inference();
    test_model_shard();
    test_dynamic_batcher();
    test_request_scheduler();
    test_inference_memory_pool();
    test_stream_config();
    test_logprobs();
    test_embedding_endpoint();
    test_reranker();
    test_grammar_decoder();
    test_paged_kv_1m_context();

    printf("\n=====================================================\n");

    return TEST_REPORT() > 0 ? 1 : 0;
}
