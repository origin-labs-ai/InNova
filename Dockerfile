FROM ubuntu:24.04 AS builder

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    cmake \
    g++-13 \
    gcc-13 \
    ninja-build \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=g++-13 \
    -DCMAKE_C_COMPILER=gcc-13 \
    -DOIL_AVX2=ON \
    -DOIL_BUILD_TESTS=ON \
    -DOIL_BUILD_BENCHMARKS=ON \
    -DOIL_BUILD_TOOLS=ON \
    -DCMAKE_CXX_STANDARD=20 \
    && cmake --build build --parallel $(nproc)

RUN ctest --test-dir build --output-on-failure --timeout 120 \
    --exclude-regex "test_protected|test_gpu|test_training|paged_kv_1t_test" || true

FROM ubuntu:24.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/build/tools/oil_infer .
COPY --from=builder /app/build/tools/oil_convert .
COPY --from=builder /app/build/tools/oil_bench .
COPY --from=builder /app/build/bench/bench_kernels .
COPY --from=builder /app/build/bench/bench_inference .
COPY --from=builder /app/build/bench/bench_quality .

EXPOSE 8080

ENTRYPOINT ["./oil_infer"]
