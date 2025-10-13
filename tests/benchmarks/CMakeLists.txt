# Benchmarks

cmake_minimum_required(VERSION 3.16)

# NPIE Benchmark
add_executable(benchmark_npie
    benchmark_npie.c
)

target_link_libraries(benchmark_npie
    PRIVATE npie
    PRIVATE Threads::Threads
    PRIVATE m
)

# Note: Benchmarks are not added as tests, run manually

