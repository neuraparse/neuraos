/**
 * @file benchmark_npie.c
 * @brief NPIE Performance Benchmark Suite
 * @version 1.0.0-alpha
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include "npie.h"

#define WARMUP_ITERATIONS 10
#define BENCHMARK_ITERATIONS 100

/**
 * @brief Get current time in microseconds
 */
static uint64_t get_time_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000 + tv.tv_usec;
}

/**
 * @brief Calculate statistics
 */
typedef struct {
    double mean;
    double median;
    double min;
    double max;
    double stddev;
    double p95;
    double p99;
} stats_t;

static void calculate_stats(uint64_t* times, int count, stats_t* stats) {
    /* Sort times */
    for (int i = 0; i < count - 1; i++) {
        for (int j = i + 1; j < count; j++) {
            if (times[i] > times[j]) {
                uint64_t temp = times[i];
                times[i] = times[j];
                times[j] = temp;
            }
        }
    }
    
    /* Calculate mean */
    uint64_t sum = 0;
    for (int i = 0; i < count; i++) {
        sum += times[i];
    }
    stats->mean = (double)sum / count;
    
    /* Median */
    stats->median = (double)times[count / 2];
    
    /* Min/Max */
    stats->min = (double)times[0];
    stats->max = (double)times[count - 1];
    
    /* Standard deviation */
    double variance = 0;
    for (int i = 0; i < count; i++) {
        double diff = times[i] - stats->mean;
        variance += diff * diff;
    }
    stats->stddev = sqrt(variance / count);
    
    /* Percentiles */
    stats->p95 = (double)times[(int)(count * 0.95)];
    stats->p99 = (double)times[(int)(count * 0.99)];
}

/**
 * @brief Print statistics
 */
static void print_stats(const char* name, stats_t* stats) {
    printf("\n%s:\n", name);
    printf("  Mean:   %8.2f μs (%6.2f ms)\n", stats->mean, stats->mean / 1000.0);
    printf("  Median: %8.2f μs (%6.2f ms)\n", stats->median, stats->median / 1000.0);
    printf("  Min:    %8.2f μs (%6.2f ms)\n", stats->min, stats->min / 1000.0);
    printf("  Max:    %8.2f μs (%6.2f ms)\n", stats->max, stats->max / 1000.0);
    printf("  StdDev: %8.2f μs\n", stats->stddev);
    printf("  P95:    %8.2f μs (%6.2f ms)\n", stats->p95, stats->p95 / 1000.0);
    printf("  P99:    %8.2f μs (%6.2f ms)\n", stats->p99, stats->p99 / 1000.0);
    printf("  Throughput: %.1f inferences/sec\n", 1000000.0 / stats->mean);
}

/**
 * @brief Benchmark context initialization
 */
static void benchmark_context_init(void) {
    printf("\n╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║         Benchmark: Context Initialization                     ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    
    uint64_t times[BENCHMARK_ITERATIONS];
    
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
        uint64_t start = get_time_us();
        
        npie_context_t ctx;
        npie_init(&ctx, NULL);
        npie_shutdown(ctx);
        
        uint64_t end = get_time_us();
        times[i] = end - start;
    }
    
    stats_t stats;
    calculate_stats(times, BENCHMARK_ITERATIONS, &stats);
    print_stats("Context Init/Shutdown", &stats);
}

/**
 * @brief Benchmark tensor allocation
 */
static void benchmark_tensor_alloc(void) {
    printf("\n╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║         Benchmark: Tensor Allocation                          ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    
    uint64_t times[BENCHMARK_ITERATIONS];
    
    /* Test different tensor sizes */
    size_t sizes[] = {
        224 * 224 * 3,      /* MobileNet input */
        1000,               /* Classification output */
        512 * 512 * 3,      /* Larger image */
        1024 * 1024         /* 1M elements */
    };
    
    const char* names[] = {
        "Small (224x224x3)",
        "Tiny (1000)",
        "Medium (512x512x3)",
        "Large (1M)"
    };
    
    for (int s = 0; s < 4; s++) {
        npie_tensor_t tensor = {
            .name = "benchmark",
            .dtype = NPIE_DTYPE_FLOAT32,
            .shape = {.rank = 1, .dims = {sizes[s]}},
            .data = NULL,
            .size = 0
        };
        
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            uint64_t start = get_time_us();
            
            npie_tensor_alloc(&tensor);
            npie_tensor_free(&tensor);
            
            uint64_t end = get_time_us();
            times[i] = end - start;
        }
        
        stats_t stats;
        calculate_stats(times, BENCHMARK_ITERATIONS, &stats);
        print_stats(names[s], &stats);
    }
}

/**
 * @brief Benchmark memory operations
 */
static void benchmark_memory_ops(void) {
    printf("\n╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║         Benchmark: Memory Operations                          ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    
    /* Initialize memory pool */
    npie_memory_init(64 * 1024 * 1024, false);
    
    uint64_t times[BENCHMARK_ITERATIONS];
    
    /* Benchmark allocation */
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
        uint64_t start = get_time_us();
        
        void* ptr = npie_memory_alloc(1024 * 1024); /* 1 MB */
        npie_memory_free(ptr);
        
        uint64_t end = get_time_us();
        times[i] = end - start;
    }
    
    stats_t stats;
    calculate_stats(times, BENCHMARK_ITERATIONS, &stats);
    print_stats("Memory Pool Alloc/Free (1MB)", &stats);
    
    npie_memory_shutdown();
}

/**
 * @brief Benchmark hardware detection
 */
static void benchmark_hardware_detection(void) {
    printf("\n╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║         Benchmark: Hardware Detection                         ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    
    npie_context_t ctx;
    npie_init(&ctx, NULL);
    
    uint64_t times[BENCHMARK_ITERATIONS];
    
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
        npie_accelerator_t accelerators[16];
        uint32_t count;
        
        uint64_t start = get_time_us();
        npie_detect_accelerators(ctx, accelerators, 16, &count);
        uint64_t end = get_time_us();
        
        times[i] = end - start;
    }
    
    stats_t stats;
    calculate_stats(times, BENCHMARK_ITERATIONS, &stats);
    print_stats("Hardware Detection", &stats);
    
    npie_shutdown(ctx);
}

/**
 * @brief System information
 */
static void print_system_info(void) {
    printf("\n╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║                    System Information                         ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    
    printf("\nNPIE Version: %s\n", npie_version());
    
    /* CPU info */
    FILE* fp = fopen("/proc/cpuinfo", "r");
    if (fp) {
        char line[256];
        while (fgets(line, sizeof(line), fp)) {
            if (strstr(line, "model name")) {
                printf("CPU: %s", strchr(line, ':') + 2);
                break;
            }
        }
        fclose(fp);
    }
    
    /* Memory info */
    fp = fopen("/proc/meminfo", "r");
    if (fp) {
        char line[256];
        while (fgets(line, sizeof(line), fp)) {
            if (strstr(line, "MemTotal")) {
                printf("Memory: %s", strchr(line, ':') + 2);
                break;
            }
        }
        fclose(fp);
    }
    
    /* Accelerators */
    npie_context_t ctx;
    npie_init(&ctx, NULL);
    
    npie_accelerator_t accelerators[16];
    uint32_t count;
    npie_detect_accelerators(ctx, accelerators, 16, &count);
    
    printf("Accelerators: %u detected\n", count);
    const char* accel_names[] = {"None", "Auto", "GPU", "NPU", "TPU", "DSP"};
    for (uint32_t i = 0; i < count; i++) {
        printf("  - %s\n", accel_names[accelerators[i]]);
    }
    
    npie_shutdown(ctx);
}

/**
 * @brief Main benchmark runner
 */
int main(int argc, char** argv) {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║              NPIE Performance Benchmark Suite                 ║\n");
    printf("║                    Version 1.0.0-alpha                        ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    
    print_system_info();
    
    printf("\n\nRunning benchmarks...\n");
    printf("Warmup iterations: %d\n", WARMUP_ITERATIONS);
    printf("Benchmark iterations: %d\n", BENCHMARK_ITERATIONS);
    
    benchmark_context_init();
    benchmark_tensor_alloc();
    benchmark_memory_ops();
    benchmark_hardware_detection();
    
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║                  Benchmark Complete!                          ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    
    return 0;
}

