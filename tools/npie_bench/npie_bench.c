/**
 * @file npie_bench.c
 * @brief NPIE Multi-Backend Benchmark Tool
 * @version 1.0.0
 *
 * Compares inference performance across available NPIE backends.
 * Outputs results in both human-readable and JSON formats.
 *
 * Usage:
 *   npie-bench                           # Run all benchmarks
 *   npie-bench --backend emlearn         # Specific backend
 *   npie-bench --json                    # JSON output
 *   npie-bench --iterations 500          # Custom iterations
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include "npie.h"

#define DEFAULT_ITERATIONS 100
#define WARMUP_ITERATIONS 10
#define MAX_BACKENDS 6

typedef struct {
    double mean_us;
    double median_us;
    double min_us;
    double max_us;
    double stddev_us;
    double p95_us;
    double p99_us;
    double throughput;  /* inferences per second */
} bench_stats_t;

typedef struct {
    const char* name;
    npie_backend_t backend;
    int available;
    bench_stats_t init_stats;
    bench_stats_t tensor_alloc_stats;
    bench_stats_t inference_stats;
    bench_stats_t memory_stats;
} backend_result_t;

static uint64_t get_time_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000 + tv.tv_usec;
}

static int cmp_u64(const void* a, const void* b) {
    uint64_t va = *(const uint64_t*)a;
    uint64_t vb = *(const uint64_t*)b;
    return (va > vb) - (va < vb);
}

static void calc_stats(uint64_t* times, int count, bench_stats_t* stats) {
    qsort(times, count, sizeof(uint64_t), cmp_u64);

    double sum = 0;
    for (int i = 0; i < count; i++) sum += times[i];

    stats->mean_us = sum / count;
    stats->median_us = (count % 2 == 0)
        ? (times[count/2 - 1] + times[count/2]) / 2.0
        : times[count/2];
    stats->min_us = times[0];
    stats->max_us = times[count - 1];
    stats->p95_us = times[(int)(count * 0.95)];
    stats->p99_us = times[(int)(count * 0.99)];

    double var = 0;
    for (int i = 0; i < count; i++) {
        double d = times[i] - stats->mean_us;
        var += d * d;
    }
    stats->stddev_us = sqrt(var / count);
    stats->throughput = (stats->mean_us > 0) ? 1000000.0 / stats->mean_us : 0;
}

static void print_stats(const char* label, const bench_stats_t* s) {
    printf("  %-28s %10.1f us  (med: %.1f, p95: %.1f, p99: %.1f)  %10.0f ops/s\n",
           label, s->mean_us, s->median_us, s->p95_us, s->p99_us, s->throughput);
}

static void bench_context_init(npie_backend_t backend, int iterations,
                                bench_stats_t* stats) {
    uint64_t* times = calloc(iterations, sizeof(uint64_t));

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        npie_context_t ctx;
        npie_options_t opts = {
            .backend = backend,
            .accelerator = NPIE_ACCELERATOR_NONE,
            .num_threads = 1,
            .timeout_ms = 1000,
        };
        npie_init(&ctx, &opts);
        npie_shutdown(ctx);
    }

    /* Benchmark */
    for (int i = 0; i < iterations; i++) {
        npie_context_t ctx;
        npie_options_t opts = {
            .backend = backend,
            .accelerator = NPIE_ACCELERATOR_NONE,
            .num_threads = 1,
            .timeout_ms = 1000,
        };
        uint64_t t0 = get_time_us();
        npie_init(&ctx, &opts);
        npie_shutdown(ctx);
        times[i] = get_time_us() - t0;
    }

    calc_stats(times, iterations, stats);
    free(times);
}

static void bench_tensor_ops(npie_backend_t backend, int iterations,
                              bench_stats_t* stats) {
    uint64_t* times = calloc(iterations, sizeof(uint64_t));

    npie_context_t ctx;
    npie_options_t opts = {
        .backend = backend,
        .accelerator = NPIE_ACCELERATOR_NONE,
        .num_threads = 1,
        .timeout_ms = 1000,
    };
    npie_init(&ctx, &opts);

    /* Benchmark tensor alloc/free for 224x224x3 image tensor */
    for (int i = 0; i < iterations; i++) {
        npie_tensor_t tensor = {
            .dtype = NPIE_DTYPE_FLOAT32,
            .shape = { .rank = 4, .dims = {1, 224, 224, 3} },
        };
        uint64_t t0 = get_time_us();
        npie_tensor_alloc(&tensor);
        npie_tensor_free(&tensor);
        times[i] = get_time_us() - t0;
    }

    calc_stats(times, iterations, stats);
    npie_shutdown(ctx);
    free(times);
}

static void bench_memory_pool(int iterations, bench_stats_t* stats) {
    uint64_t* times = calloc(iterations, sizeof(uint64_t));

    npie_context_t ctx;
    npie_options_t opts = {
        .backend = NPIE_BACKEND_AUTO,
        .accelerator = NPIE_ACCELERATOR_NONE,
        .num_threads = 1,
        .timeout_ms = 1000,
    };
    npie_init(&ctx, &opts);

    npie_memory_init(64 * 1024 * 1024, false);

    for (int i = 0; i < iterations; i++) {
        uint64_t t0 = get_time_us();
        void* ptr = npie_memory_alloc(1024 * 1024);
        npie_memory_free(ptr);
        times[i] = get_time_us() - t0;
    }

    calc_stats(times, iterations, stats);
    npie_memory_shutdown();
    npie_shutdown(ctx);
    free(times);
}

static void print_json(backend_result_t* results, int count, int iterations) {
    printf("{\n  \"tool\": \"npie-bench\",\n");
    printf("  \"npie_version\": \"%s\",\n", npie_version());
    printf("  \"iterations\": %d,\n", iterations);
    printf("  \"backends\": [\n");

    for (int i = 0; i < count; i++) {
        backend_result_t* r = &results[i];
        printf("    {\n");
        printf("      \"name\": \"%s\",\n", r->name);
        printf("      \"available\": %s,\n", r->available ? "true" : "false");
        if (r->available) {
            printf("      \"context_init\": {\"mean_us\": %.1f, \"throughput\": %.0f},\n",
                   r->init_stats.mean_us, r->init_stats.throughput);
            printf("      \"tensor_alloc\": {\"mean_us\": %.1f, \"throughput\": %.0f},\n",
                   r->tensor_alloc_stats.mean_us, r->tensor_alloc_stats.throughput);
        }
        printf("    }%s\n", i < count - 1 ? "," : "");
    }

    printf("  ]\n}\n");
}

static void print_report(backend_result_t* results, int count, int iterations) {
    printf("\n");
    printf("========================================================================\n");
    printf("  NPIE Multi-Backend Benchmark Report\n");
    printf("  Version: %s | Iterations: %d\n", npie_version(), iterations);
    printf("========================================================================\n\n");

    /* System info */
    FILE* f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "model name", 10) == 0) {
                printf("  CPU: %s", strchr(line, ':') + 2);
                break;
            }
        }
        fclose(f);
    }

    f = fopen("/proc/meminfo", "r");
    if (f) {
        char line[256];
        if (fgets(line, sizeof(line), f)) {
            unsigned long mem_kb = 0;
            sscanf(line, "MemTotal: %lu kB", &mem_kb);
            printf("  Memory: %lu MB\n", mem_kb / 1024);
        }
        fclose(f);
    }
    printf("\n");

    /* Backend comparison table */
    printf("  %-15s  %-10s  %12s  %12s  %12s\n",
           "Backend", "Status", "Init (us)", "Tensor (us)", "Throughput");
    printf("  %-15s  %-10s  %12s  %12s  %12s\n",
           "---------------", "----------", "------------", "------------", "------------");

    for (int i = 0; i < count; i++) {
        backend_result_t* r = &results[i];
        if (r->available) {
            printf("  %-15s  %-10s  %12.1f  %12.1f  %10.0f/s\n",
                   r->name, "OK",
                   r->init_stats.mean_us,
                   r->tensor_alloc_stats.mean_us,
                   r->init_stats.throughput);
        } else {
            printf("  %-15s  %-10s  %12s  %12s  %12s\n",
                   r->name, "N/A", "-", "-", "-");
        }
    }

    /* Detailed per-backend stats */
    printf("\n--- Detailed Statistics ---\n");
    for (int i = 0; i < count; i++) {
        backend_result_t* r = &results[i];
        if (!r->available) continue;

        printf("\n  [%s]\n", r->name);
        print_stats("Context Init/Shutdown:", &r->init_stats);
        print_stats("Tensor Alloc/Free (224x224x3):", &r->tensor_alloc_stats);
    }

    /* Memory pool (backend-independent) */
    printf("\n  [Memory Pool]\n");
    bench_stats_t mem_stats;
    bench_memory_pool(iterations, &mem_stats);
    print_stats("Pool Alloc/Free (1MB):", &mem_stats);

    printf("\n========================================================================\n");
}

int main(int argc, char** argv) {
    int iterations = DEFAULT_ITERATIONS;
    int json_output = 0;
    npie_backend_t filter_backend = -1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--json") == 0) {
            json_output = 1;
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
            if (iterations < 1) iterations = DEFAULT_ITERATIONS;
        } else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "litert") == 0) filter_backend = NPIE_BACKEND_LITERT;
            else if (strcmp(argv[i], "onnx") == 0) filter_backend = NPIE_BACKEND_ONNXRUNTIME;
            else if (strcmp(argv[i], "emlearn") == 0) filter_backend = NPIE_BACKEND_EMLEARN;
            else if (strcmp(argv[i], "wasm") == 0) filter_backend = NPIE_BACKEND_WASMEDGE;
            else if (strcmp(argv[i], "auto") == 0) filter_backend = NPIE_BACKEND_AUTO;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: npie-bench [options]\n");
            printf("Options:\n");
            printf("  --iterations N    Number of benchmark iterations (default: 100)\n");
            printf("  --backend NAME    Test specific backend (auto|litert|onnx|emlearn|wasm)\n");
            printf("  --json            Output results in JSON format\n");
            printf("  --help            Show this help\n");
            return 0;
        }
    }

    backend_result_t backends[5];
    memset(backends, 0, sizeof(backends));
    backends[0].name = "Auto";      backends[0].backend = NPIE_BACKEND_AUTO;
    backends[1].name = "LiteRT";    backends[1].backend = NPIE_BACKEND_LITERT;
    backends[2].name = "ONNX";      backends[2].backend = NPIE_BACKEND_ONNXRUNTIME;
    backends[3].name = "emlearn";   backends[3].backend = NPIE_BACKEND_EMLEARN;
    backends[4].name = "WasmEdge";  backends[4].backend = NPIE_BACKEND_WASMEDGE;
    int num_backends = 5;

    if (!json_output) {
        printf("NPIE Multi-Backend Benchmark v%s\n", npie_version());
        printf("Running %d iterations per test...\n\n", iterations);
    }

    for (int i = 0; i < num_backends; i++) {
        if (filter_backend != (npie_backend_t)-1 &&
            backends[i].backend != filter_backend) {
            continue;
        }

        /* Try to initialize with this backend */
        npie_context_t ctx;
        npie_options_t opts = {
            .backend = backends[i].backend,
            .accelerator = NPIE_ACCELERATOR_NONE,
            .num_threads = 1,
            .timeout_ms = 1000,
        };

        if (npie_init(&ctx, &opts) == NPIE_SUCCESS) {
            npie_shutdown(ctx);
            backends[i].available = 1;

            if (!json_output) {
                printf("Benchmarking: %s...\n", backends[i].name);
            }

            bench_context_init(backends[i].backend, iterations,
                               &backends[i].init_stats);
            bench_tensor_ops(backends[i].backend, iterations,
                             &backends[i].tensor_alloc_stats);
        } else {
            if (!json_output) {
                printf("Skipping: %s (not available)\n", backends[i].name);
            }
        }
    }

    if (json_output) {
        print_json(backends, num_backends, iterations);
    } else {
        print_report(backends, num_backends, iterations);
    }

    return 0;
}
