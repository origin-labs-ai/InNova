#pragma once

#include <cstdio>
#include <cmath>
#include <cstdlib>

namespace oil {
namespace test {

struct TestRunner {
    static TestRunner& instance() {
        static TestRunner r;
        return r;
    }

    int failures = 0;
    int tests_run = 0;
    const char* current_suite = "";

    void suite(const char* name) {
        current_suite = name;
        printf("\n=== %s ===\n", name);
    }

    bool check(bool cond, const char* msg, const char* file, int line) {
        tests_run++;
        if (!cond) {
            printf("  FAIL [%s:%d]: %s\n", file, line, msg);
            failures++;
            return false;
        }
        return true;
    }

    bool check_close(double a, double b, double eps, const char* msg, const char* file, int line) {
        return check(std::fabs(a - b) < eps, msg, file, line);
    }

    int report() const {
        printf("\n=== %d tests, %d failures ===\n", tests_run, failures);
        return failures;
    }
};

} // namespace test
} // namespace oil

#define TEST_SUITE(name) ::oil::test::TestRunner::instance().suite(name)

#define TEST_CHECK(cond, msg) \
    ::oil::test::TestRunner::instance().check((cond), (msg), __FILE__, __LINE__)

#define TEST_CHECK_CLOSE(a, b, eps, msg) \
    ::oil::test::TestRunner::instance().check_close((double)(a), (double)(b), (double)(eps), (msg), __FILE__, __LINE__)

#define TEST_REQUIRE(cond, msg) \
    do { \
        if (!::oil::test::TestRunner::instance().check((cond), (msg), __FILE__, __LINE__)) { \
            return; \
        } \
    } while(0)

#define TEST_FAIL(msg) \
    ::oil::test::TestRunner::instance().check(false, (msg), __FILE__, __LINE__)

#define TEST_REPORT() ::oil::test::TestRunner::instance().report()

#define TEST_MAIN() \
    int main() { \
        int f = TEST_REPORT(); \
        return f > 0 ? 1 : 0; \
    }