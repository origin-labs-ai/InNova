// test_debug.cpp — Debug utility test
#include <iostream>
#include <cassert>

int main() {
    std::cout << "[Test] Running Debug verification test..." << std::endl;
    int a = 42;
    assert(a == 42);
    std::cout << "Debug test passed!" << std::endl;
    return 0;
}
