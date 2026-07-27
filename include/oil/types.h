#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <functional>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cassert>

namespace oil {

enum class Activation : uint8_t { None, ReLU, GELU, SiLU };

enum class Format : uint8_t {
    OIL1            = 0,   // 1.00 BPW
    SPARK_Q0        = 1,   // 1.50 BPW (sign + shared FP16 scale, lossy)
    OIL2            = 2,   // 2.00 BPW, 4 centroids Lloyd-Max (lossy)
    OIL4            = 3,   // 4.00 BPW, 16 centroids Lloyd-Max (lossy)
    OIL8            = 4,   // 8.00 BPW, 256 centroids Lloyd-Max (lossy)
    OIL16           = 5,   // 16.00 BPW, FP16 storage (lossy)
    OIL32           = 6,   // 32.00 BPW, FP32 identity (lossless)
    OIL1_GRP        = 7,   // 1.00 BPW, lossless grouped
    SPARK_Q0_GRP    = 8,   // 1.50 BPW, lossless, sign + per-group scale
    OIL2_GRP        = 9,   // 2.00 BPW, lossless grouped
    OIL4_GRP        = 10,  // 4.00 BPW, lossless grouped
    OIL8_GRP        = 11,  // 8.00 BPW, lossless grouped
    OIL16_GRP       = 12,  // 16.00 BPW, lossless grouped
    SPARK_SPARSE     = 13,  // ~2.00 BPW, lossy, sparse (index,value) pairs
    SPARK_SPARSE_GRP = 14,  // ~2.00 BPW, lossless, sparse grouped scale
};

inline const char* format_name(Format f) {
    switch (f) {
        case Format::OIL1:          return "OIL1";
        case Format::OIL2:          return "OIL2";
        case Format::OIL4:          return "OIL4";
        case Format::OIL8:          return "OIL8";
        case Format::OIL16:         return "OIL16";
        case Format::OIL32:         return "OIL32";
        case Format::OIL1_GRP:      return "OIL1_GRP";
        case Format::OIL2_GRP:      return "OIL2_GRP";
        case Format::OIL4_GRP:      return "OIL4_GRP";
        case Format::OIL8_GRP:      return "OIL8_GRP";
        case Format::OIL16_GRP:     return "OIL16_GRP";
        case Format::SPARK_SPARSE:      return "SPARK_SPARSE";
        case Format::SPARK_SPARSE_GRP:  return "SPARK_SPARSE_GRP";
        case Format::SPARK_Q0:          return "SPARK_Q0";
        case Format::SPARK_Q0_GRP:      return "SPARK_Q0_GRP";
        default: return "unknown";
    }
}

inline float format_bpw(Format f) {
    switch (f) {
        case Format::OIL1:      return 1.0f;
        case Format::OIL2:      return 2.0f;
        case Format::OIL4:      return 4.0f;
        case Format::OIL8:      return 8.0f;
        case Format::OIL16:     return 16.0f;
        case Format::OIL32:     return 32.0f;
        case Format::OIL1_GRP:  return 1.0f;
        case Format::OIL2_GRP:  return 2.0f;
        case Format::OIL4_GRP:  return 4.0f;
        case Format::OIL8_GRP:  return 8.0f;
        case Format::OIL16_GRP: return 16.0f;
        case Format::SPARK_SPARSE:     return 1.5f;
        case Format::SPARK_SPARSE_GRP:  return 2.0f;
        case Format::SPARK_Q0:          return 1.5f;
        case Format::SPARK_Q0_GRP:      return 1.5f;
        default: return 0;
    }
}

enum class DType : uint8_t {
    I64,     // int64_t
    U8,      // uint8_t
    U4,      // 4-bit packed (2 per byte)
    F16,     // half precision
    F32,     // single precision
};

inline size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::I64: return 8;
        case DType::U8: return 1;
        case DType::U4: return 1;
        case DType::F16: return 2;
        case DType::F32: return 4;
        default: return 0;
    }
}

inline DType format_to_dtype(Format f) {
    switch (f) {
        case Format::OIL1:
        case Format::OIL2:    return DType::U8;
        case Format::OIL4:    return DType::U4;
        case Format::OIL8:    return DType::U8;
        case Format::OIL16:   return DType::F16;
        case Format::OIL32:   return DType::F32;
        default: return DType::F32;
    }
}

struct Shape {
    int64_t dims[8];
    int rank;

    Shape() : rank(0) { dims[0]=dims[1]=dims[2]=dims[3]=dims[4]=dims[5]=dims[6]=dims[7]=0; }
    explicit Shape(int64_t d0) : rank(1) { dims[0]=d0; dims[1]=dims[2]=dims[3]=dims[4]=dims[5]=dims[6]=dims[7]=0; }
    Shape(int64_t d0, int64_t d1) : rank(2) { dims[0]=d0; dims[1]=d1; dims[2]=dims[3]=dims[4]=dims[5]=dims[6]=dims[7]=0; }
    Shape(int64_t d0, int64_t d1, int64_t d2) : rank(3) { dims[0]=d0; dims[1]=d1; dims[2]=d2; dims[3]=dims[4]=dims[5]=dims[6]=dims[7]=0; }
    Shape(std::initializer_list<int64_t> l) : rank((int)l.size()) {
        dims[0]=dims[1]=dims[2]=dims[3]=dims[4]=dims[5]=dims[6]=dims[7]=0;
        if (rank > 8) throw std::runtime_error("Shape: rank exceeds maximum of 8");
        int i=0; for (auto x: l) dims[i++] = x;
    }

    int64_t& operator[](int i) { if (i < 0 || i >= rank) throw std::out_of_range("Shape index out of range"); return dims[i]; }
    const int64_t& operator[](int i) const { if (i < 0 || i >= rank) throw std::out_of_range("Shape index out of range"); return dims[i]; }

    int64_t numel() const {
        int64_t n = 1;
        for (int i=0; i<rank; i++) n *= dims[i];
        return n;
    }

    bool operator==(const Shape& o) const {
        if (rank != o.rank) return false;
        for (int i=0; i<rank; i++) if (dims[i] != o.dims[i]) return false;
        return true;
    }

    bool operator!=(const Shape& o) const { return !(*this == o); }

    std::string to_string() const {
        std::string s = "[";
        for (int i=0; i<rank; i++) {
            if (i) s += ",";
            s += std::to_string(dims[i]);
        }
        s += "]";
        return s;
    }
};

struct Status {
    bool ok;
    std::string msg;
    Status() : ok(true) {}
    Status(const std::string& e) : ok(false), msg(e) {}
    static Status error(const std::string& m) { return Status(m); }
    static Status success() { return Status(); }
    explicit operator bool() const { return ok; }
};

struct Config {
    int num_threads = 1;
    uint64_t seed = 42;
    size_t pool_size = 64 * 1024 * 1024; // 64MB temp pool
    bool verbose = false;
};

class Error : public std::runtime_error {
public:
    explicit Error(const std::string& msg) : std::runtime_error(msg) {}
};

#define OIL_CHECK(cond, msg) \
    do { if (!(cond)) throw oil::Error(msg); } while(0)

} // namespace oil
