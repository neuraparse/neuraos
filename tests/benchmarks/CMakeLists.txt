# Benchmarks

cmake_minimum_required(VERSION 3.16)

# NPIE Benchmark
add_executable(benchmark_npie
    benchmark_npie.c
)

target_link_libraries(benchmark_npie
    Threads::Threads
)

# Note: Benchmarks are not added as tests, run manually

