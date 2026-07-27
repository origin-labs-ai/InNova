#define NOMINMAX
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cassert>
#include <iostream>
#include <vector>
#include "oil/moe_model.h"
#include "oil/moe_variants.h"
#include "oil/distributed.h"
#include "oil/random.h"
#include "oil/optimizer.h"

using namespace oil;

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { std::cout << "  " << name << " ... "; tests_run++; } while(0)
#define PASS() do { std::cout << "PASS" << std::endl; tests_passed++; } while(0)
#define FAIL(msg) do { std::cout << "FAIL: " << msg << std::endl; return 1; } while(0)

static int test_expert_distribution() {
    TEST("ExpertParallel distributes experts round-robin across ranks");
    {
        ExpertParallel ep(8, 4, 0);
        auto local = ep.local_experts();
        if (local.size() != 2) FAIL("rank 0 should have 2 experts, got " + std::to_string(local.size()));
        if (local[0] != 0 || local[1] != 4) FAIL("rank 0 should have experts {0,4}");
    }
    {
        ExpertParallel ep(8, 4, 1);
        auto local = ep.local_experts();
        if (local.size() != 2) FAIL("rank 1 should have 2 experts, got " + std::to_string(local.size()));
        if (local[0] != 1 || local[1] != 5) FAIL("rank 1 should have experts {1,5}");
    }
    PASS();
    return 0;
}

static int test_expert_distribution_uneven() {
    TEST("ExpertParallel uneven expert distribution");
    {
        ExpertParallel ep(10, 4, 0);
        auto local = ep.local_experts();
        if (local.size() != 3) FAIL("rank 0 should have 3 experts, got " + std::to_string(local.size()));
        if (local[0] != 0 || local[1] != 4 || local[2] != 8) FAIL("rank 0 should have experts {0,4,8}");
    }
    {
        ExpertParallel ep(10, 4, 3);
        auto local = ep.local_experts();
        if (local.size() != 2) FAIL("rank 3 should have 2 experts, got " + std::to_string(local.size()));
        if (local[0] != 3 || local[1] != 7) FAIL("rank 3 should have experts {3,7}");
    }
    PASS();
    return 0;
}

static int test_expert_parallel_single_rank() {
    TEST("ExpertParallel single rank (no parallelism)");
    {
        ExpertParallel ep(8, 1, 0);
        auto local = ep.local_experts();
        if (local.size() != 8) FAIL("single rank should have all 8 experts");
    }
    PASS();
    return 0;
}

int main() {
    std::cout << "ExpertParallel & DistributedContext Tests" << std::endl;
    std::cout << "=========================================" << std::endl;

    if (test_expert_distribution()) return 1;
    if (test_expert_distribution_uneven()) return 1;
    if (test_expert_parallel_single_rank()) return 1;

    std::cout << "\n=== " << tests_run << " tests, " << (tests_run - tests_passed) << " failures ===\n";
    return (tests_run == tests_passed) ? 0 : 1;
}
