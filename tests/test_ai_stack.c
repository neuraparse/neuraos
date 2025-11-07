/**
 * @file test_ai_stack.c
 * @brief NeuralOS AI Stack Test Program
 * 
 * Tests all AI components to verify they're working:
 * - emlearn (classical ML)
 * - OpenCV (image processing)
 * - NPIE (inference engine)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

// Test results
typedef struct {
    const char *name;
    bool passed;
    const char *message;
} test_result_t;

#define MAX_TESTS 10
test_result_t results[MAX_TESTS];
int test_count = 0;

void add_test_result(const char *name, bool passed, const char *message) {
    results[test_count].name = name;
    results[test_count].passed = passed;
    results[test_count].message = message;
    test_count++;
}

// Test 1: emlearn header availability
bool test_emlearn_headers() {
    printf("Testing emlearn headers...\n");
    
    // Try to include emlearn headers
    #ifdef __has_include
    #if __has_include(<emlearn/eml_trees.h>)
        printf("  ✅ emlearn/eml_trees.h found\n");
        return true;
    #else
        printf("  ❌ emlearn headers not found\n");
        return false;
    #endif
    #else
        // Fallback: assume headers exist if emlearn package is built
        FILE *f = fopen("/usr/include/emlearn/eml_trees.h", "r");
        if (f) {
            fclose(f);
            printf("  ✅ emlearn headers found\n");
            return true;
        } else {
            printf("  ❌ emlearn headers not found\n");
            return false;
        }
    #endif
}

// Test 2: OpenCV library availability
bool test_opencv_library() {
    printf("Testing OpenCV library...\n");
    
    // Check if OpenCV shared library exists
    FILE *f = fopen("/usr/lib/libopencv_core.so", "r");
    if (!f) {
        f = fopen("/usr/lib/libopencv_core.so.4.10", "r");
    }
    
    if (f) {
        fclose(f);
        printf("  ✅ OpenCV library found\n");
        return true;
    } else {
        printf("  ⚠️  OpenCV library not found (expected if not built)\n");
        return false;
    }
}

// Test 3: NPIE library availability
bool test_npie_library() {
    printf("Testing NPIE library...\n");
    
    // Check if NPIE shared library exists
    FILE *f = fopen("/usr/lib/libnpie.so", "r");
    if (!f) {
        f = fopen("/usr/lib/libnpie.so.1", "r");
    }
    
    if (f) {
        fclose(f);
        printf("  ✅ NPIE library found\n");
        return true;
    } else {
        printf("  ⚠️  NPIE library not found (expected if not built)\n");
        return false;
    }
}

// Test 4: NPIE headers availability
bool test_npie_headers() {
    printf("Testing NPIE headers...\n");
    
    FILE *f = fopen("/usr/include/npie/npie.h", "r");
    if (!f) {
        f = fopen("/usr/include/npie.h", "r");
    }
    
    if (f) {
        fclose(f);
        printf("  ✅ NPIE headers found\n");
        return true;
    } else {
        printf("  ⚠️  NPIE headers not found (expected if not built)\n");
        return false;
    }
}

// Test 5: Check for AI accelerator support in kernel
bool test_kernel_accelerator_support() {
    printf("Testing kernel accelerator support...\n");
    
    // Check for GPU/NPU device nodes
    bool has_gpu = false;
    bool has_npu = false;
    
    FILE *f = fopen("/dev/dri/card0", "r");
    if (f) {
        fclose(f);
        has_gpu = true;
        printf("  ✅ GPU device found (/dev/dri/card0)\n");
    }
    
    f = fopen("/dev/apex_0", "r");  // Google Coral Edge TPU
    if (f) {
        fclose(f);
        has_npu = true;
        printf("  ✅ NPU device found (/dev/apex_0)\n");
    }
    
    if (!has_gpu && !has_npu) {
        printf("  ℹ️  No hardware accelerators found (CPU-only mode)\n");
    }
    
    return true;  // Not having accelerators is OK
}

// Test 6: Check system memory for AI workloads
bool test_system_memory() {
    printf("Testing system memory...\n");
    
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) {
        printf("  ❌ Cannot read /proc/meminfo\n");
        return false;
    }
    
    char line[256];
    long total_mem_kb = 0;
    long free_mem_kb = 0;
    
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "MemTotal:", 9) == 0) {
            sscanf(line + 9, "%ld", &total_mem_kb);
        } else if (strncmp(line, "MemAvailable:", 13) == 0) {
            sscanf(line + 13, "%ld", &free_mem_kb);
        }
    }
    fclose(f);
    
    printf("  Total memory: %ld MB\n", total_mem_kb / 1024);
    printf("  Available memory: %ld MB\n", free_mem_kb / 1024);
    
    if (total_mem_kb < 256 * 1024) {
        printf("  ⚠️  Low memory (< 256 MB) - AI workloads may be limited\n");
        return false;
    } else {
        printf("  ✅ Sufficient memory for AI workloads\n");
        return true;
    }
}

// Test 7: Check CPU features (NEON for ARM)
bool test_cpu_features() {
    printf("Testing CPU features...\n");
    
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f) {
        printf("  ❌ Cannot read /proc/cpuinfo\n");
        return false;
    }
    
    char line[256];
    bool has_neon = false;
    bool has_fp = false;
    
    while (fgets(line, sizeof(line), f)) {
        if (strstr(line, "neon") || strstr(line, "NEON")) {
            has_neon = true;
        }
        if (strstr(line, "fp") || strstr(line, "vfp")) {
            has_fp = true;
        }
    }
    fclose(f);
    
    if (has_neon) {
        printf("  ✅ NEON SIMD support detected\n");
    }
    if (has_fp) {
        printf("  ✅ Floating-point support detected\n");
    }
    
    return has_neon || has_fp;
}

// Test 8: Check for Python (needed for model conversion)
bool test_python_availability() {
    printf("Testing Python availability...\n");
    
    int ret = system("python3 --version > /dev/null 2>&1");
    if (ret == 0) {
        printf("  ✅ Python3 available\n");
        return true;
    } else {
        printf("  ⚠️  Python3 not available (optional)\n");
        return false;
    }
}

void print_summary() {
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║                                                                ║\n");
    printf("║              NEURAOS AI STACK TEST SUMMARY                     ║\n");
    printf("║                                                                ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    
    int passed = 0;
    int failed = 0;
    
    for (int i = 0; i < test_count; i++) {
        const char *status = results[i].passed ? "✅ PASS" : "❌ FAIL";
        printf("  %s  %s\n", status, results[i].name);
        if (results[i].message && strlen(results[i].message) > 0) {
            printf("         %s\n", results[i].message);
        }
        
        if (results[i].passed) {
            passed++;
        } else {
            failed++;
        }
    }
    
    printf("\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("  Total: %d tests, %d passed, %d failed\n", test_count, passed, failed);
    printf("  Success rate: %.1f%%\n", (float)passed / test_count * 100);
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("\n");
}

int main(int argc, char *argv[]) {
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║                                                                ║\n");
    printf("║              NEURAOS AI STACK TEST PROGRAM                     ║\n");
    printf("║                      Version 1.0.0-alpha                       ║\n");
    printf("║                                                                ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    
    // Run all tests
    add_test_result("emlearn headers", test_emlearn_headers(), "");
    add_test_result("OpenCV library", test_opencv_library(), "");
    add_test_result("NPIE library", test_npie_library(), "");
    add_test_result("NPIE headers", test_npie_headers(), "");
    add_test_result("Kernel accelerator support", test_kernel_accelerator_support(), "");
    add_test_result("System memory", test_system_memory(), "");
    add_test_result("CPU features", test_cpu_features(), "");
    add_test_result("Python availability", test_python_availability(), "");
    
    // Print summary
    print_summary();
    
    // Return 0 if all critical tests passed
    int critical_failures = 0;
    for (int i = 0; i < test_count; i++) {
        // emlearn, CPU features, and memory are critical
        if (!results[i].passed && 
            (strstr(results[i].name, "emlearn") || 
             strstr(results[i].name, "CPU") ||
             strstr(results[i].name, "memory"))) {
            critical_failures++;
        }
    }
    
    if (critical_failures > 0) {
        printf("⚠️  %d critical test(s) failed!\n\n", critical_failures);
        return 1;
    } else {
        printf("✅ All critical tests passed!\n\n");
        return 0;
    }
}

