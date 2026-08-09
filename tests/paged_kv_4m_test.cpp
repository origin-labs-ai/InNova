// paged_kv_4m_test.cpp — Paged KV Cache test
#include "quant/kv_cache.h"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "[Test] Running Paged KV Cache test..." << std::endl;
    quant::PagedKVCache cache;
    cache.init(2, 4096, 8, 64);
    std::cout << "Paged KV Cache initialized successfully!" << std::endl;
    return 0;
}
