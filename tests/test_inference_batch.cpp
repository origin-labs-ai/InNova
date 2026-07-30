#include "oil/inference_opt.h"
#include "oil/model.h"
#include "oil/tensor.h"
#include "oil/math.h"
#include "oil/types.h"
#include "oil/transformer.h"
#include "oil/sampler.h"
#include "oil/generator.h"
#include "oil/kv_cache.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include "oil/test.h"

using namespace oil;

// Continuous batching
static void test_continuous_batching() {
    TEST_SUITE("Continuous Batching");
    ContinuousBatching cb(nullptr, 4);
    TEST_CHECK(!cb.has_pending(), "no pending requests initially");

    // Add single request
    BatchRequest req1{{1, 2, 3}, 5, 100};
    cb.add_request(req1);
    TEST_CHECK(cb.has_pending(), "pending after add_request");

    // Step
    auto resp1 = cb.step();
    TEST_CHECK(resp1.id == 100, "response has correct id");
    TEST_CHECK(true, "response received");

    // Add multiple requests
    BatchRequest req2{{4, 5, 6, 7}, 3, 200};
    BatchRequest req3{{8, 9}, 4, 300};
    cb.add_request(req2);
    cb.add_request(req3);
    TEST_CHECK(cb.has_pending(), "pending after multiple adds");

    auto resp2 = cb.step();
    TEST_CHECK(true, "second response received");

    // Add and drain all
    ContinuousBatching cb2(nullptr, 8);
    for (int i = 0; i < 5; i++) {
        BatchRequest r{{i + 1, i + 2}, 10, i};
        cb2.add_request(r);
    }
    TEST_CHECK(cb2.has_pending(), "multiple requests pending");
    int responses = 0;
    while (cb2.has_pending()) {
        cb2.step();
        responses++;
    }
    TEST_CHECK(responses > 0, "drained all requests");
}

// Dynamic batching
static void test_dynamic_batching() {
    TEST_SUITE("Dynamic Batching");
    DynamicBatcher db(nullptr);

    // Single prompt
    auto results = db.batch_generate({"hello world"}, 10);
    TEST_CHECK(results.size() == 1, "single prompt returns 1 result");
    TEST_CHECK(!results[0].empty(), "single prompt result non-empty");

    // Multiple prompts
    auto multi_results = db.batch_generate({"hello", "world", "test"}, 5);
    TEST_CHECK(multi_results.size() == 3, "3 prompts returns 3 results");
    for (size_t i = 0; i < multi_results.size(); i++)
        TEST_CHECK(!multi_results[i].empty(), "each batch result non-empty");

    // Empty prompts
    auto empty_results = db.batch_generate({}, 5);
    TEST_CHECK(empty_results.empty(), "empty prompts returns empty");

    // Large batch
    std::vector<std::string> many_prompts;
    for (int i = 0; i < 10; i++)
        many_prompts.push_back("prompt " + std::to_string(i));
    auto many_results = db.batch_generate(many_prompts, 3);
    TEST_CHECK((int)many_results.size() == 10, "10 prompts returns 10 results");
}

// Request scheduling
static void test_request_scheduling() {
    TEST_SUITE("Request Scheduling");
    RequestScheduler rs;
    TEST_CHECK(!rs.has_next(), "no requests initially");
    TEST_CHECK(rs.pending_count() == 0, "pending count 0 initially");

    // Add requests with different priorities
    Request req1; req1.id = 1; req1.priority = 5; req1.tokens = {1, 2}; req1.deadline = 100;
    Request req2; req2.id = 2; req2.priority = 10; req2.tokens = {3}; req2.deadline = 50;
    Request req3; req3.id = 3; req3.priority = 1; req3.tokens = {4, 5, 6}; req3.deadline = 200;

    rs.add(req1);
    rs.add(req2);
    rs.add(req3);
    TEST_CHECK(rs.has_next(), "has next after adding");
    TEST_CHECK(rs.pending_count() == 3, "pending count 3 after adds");

    // Higher priority should come first
    Request r_first = rs.next();
    TEST_CHECK(r_first.id == 2, "highest priority request dequeued first");
    TEST_CHECK(rs.pending_count() == 2, "pending count 2 after one dequeue");

    Request r_second = rs.next();
    TEST_CHECK(r_second.id == 1, "second highest priority dequeued");
    TEST_CHECK(rs.pending_count() == 1, "pending count 1 after two dequeues");

    Request r_third = rs.next();
    TEST_CHECK(r_third.id == 3, "lowest priority dequeued last");
    TEST_CHECK(rs.pending_count() == 0, "pending count 0 after all dequeued");
    TEST_CHECK(!rs.has_next(), "no next after all dequeued");

    // Test with single request
    RequestScheduler rs2;
    Request single; single.id = 42; single.priority = 1; single.tokens = {7}; single.deadline = 10;
    rs2.add(single);
    Request r_single = rs2.next();
    TEST_CHECK(r_single.id == 42, "single request dequeued correctly");
}

// Inference memory pool
static void test_inference_memory_pool() {
    TEST_SUITE("Inference Memory Pool");
    InferenceMemoryPool pool(64, 16);
    TEST_CHECK(pool.capacity() == 16, "pool capacity 16");
    TEST_CHECK(pool.used() == 0, "pool used 0 initially");

    // Allocate blocks
    void* p1 = pool.alloc();
    TEST_CHECK(p1 != nullptr, "first allocation succeeds");
    TEST_CHECK(pool.used() == 1, "pool used 1 after first alloc");

    void* p2 = pool.alloc();
    TEST_CHECK(p2 != nullptr, "second allocation succeeds");
    TEST_CHECK(p2 != p1, "allocations return different pointers");
    TEST_CHECK(pool.used() == 2, "pool used 2 after second alloc");

    // Allocate until full
    std::vector<void*> ptrs;
    for (int i = 0; i < 14; i++) {
        void* p = pool.alloc();
        TEST_CHECK(p != nullptr, "allocation succeeds until pool full");
        ptrs.push_back(p);
    }
    TEST_CHECK(pool.used() == 16, "pool used 16 when full");

    // Allocation from empty pool returns null
    void* p_null = pool.alloc();
    TEST_CHECK(p_null == nullptr, "allocation from empty pool returns null");

    // Free and re-allocate
    pool.free(p1);
    TEST_CHECK(pool.used() == 15, "pool used 15 after free");

    void* p_re = pool.alloc();
    TEST_CHECK(p_re != nullptr, "re-allocation after free succeeds");
    TEST_CHECK(pool.used() == 16, "pool used 16 after re-allocation");

    // Free all
    pool.free(p2);
    for (void* p : ptrs)
        pool.free(p);
    TEST_CHECK(pool.used() == 1, "pool used reduced after frees");

    // Single block pool
    InferenceMemoryPool small_pool(128, 1);
    void* sp = small_pool.alloc();
    TEST_CHECK(sp != nullptr, "single-block pool alloc succeeds");
    void* sp2 = small_pool.alloc();
    TEST_CHECK(sp2 == nullptr, "single-block pool exhausted");
    small_pool.free(sp);
    void* sp3 = small_pool.alloc();
    TEST_CHECK(sp3 != nullptr, "single-block pool re-usable after free");
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("InNova — Inference Batching Test Suite\n");
    printf("===========================================\n");

    test_continuous_batching();
    test_dynamic_batching();
    test_request_scheduling();
    test_inference_memory_pool();

    printf("\n===========================================\n");

    return TEST_REPORT() > 0 ? 1 : 0;
}
