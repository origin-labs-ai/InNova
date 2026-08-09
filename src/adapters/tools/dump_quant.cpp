// dump_quant.cpp — dump tensor stats/first values from a .quant file
#include "quant/quant_format.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: dump_quant <model.quant> <tensor_name> [count]\n");
        return 1;
    }
    quant::QUANTReader reader(argv[1]);
    if (!reader.valid()) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    std::string name = argv[2];
    int count = argc >= 4 ? std::atoi(argv[3]) : 8;

    auto t = reader.read_tensor(name);
    if (t.data() == nullptr || t.numel() == 0) {
        std::fprintf(stderr, "tensor %s not found / empty\n", name.c_str());
        return 1;
    }
    const float* d = (const float*)t.data();
    int64_t n = t.numel();
    double sum = 0, sumsq = 0, mn = 1e30, mx = -1e30;
    for (int64_t i = 0; i < n; i++) {
        float v = d[i];
        sum += v; sumsq += (double)v * v;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    double mean = sum / n;
    double var = sumsq / n - mean * mean;
    std::printf("%s n=%lld mean=%.6f std=%.6f min=%.6f max=%.6f\n",
                name.c_str(), (long long)n, mean, std::sqrt(var > 0 ? var : 0), mn, mx);
    std::printf("first %d: ", count);
    for (int i = 0; i < count && i < n; i++) std::printf("%.4f ", d[i]);
    std::printf("\n");

    if (n % 4096 == 0) {
        int64_t rows = n / 4096;
        std::printf("row means[%lld]: ", (long long)rows);
        for (int64_t r = 0; r < rows && r < 32; r++) {
            double rs = 0;
            for (int64_t c = 0; c < 4096; c++) rs += d[r * 4096 + c];
            std::printf("%.4f ", rs / 4096.0);
        }
        std::printf("\nrow stds[%lld]: ", (long long)rows);
        for (int64_t r = 0; r < rows && r < 32; r++) {
            double rs = 0, rs2 = 0;
            for (int64_t c = 0; c < 4096; c++) { double v = d[r * 4096 + c]; rs += v; rs2 += v * v; }
            double m = rs / 4096.0, vv = rs2 / 4096.0 - m * m;
            std::printf("%.4f ", std::sqrt(vv > 0 ? vv : 0));
        }
        std::printf("\n");
    }
    return 0;
}
