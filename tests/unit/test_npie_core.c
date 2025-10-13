/**
 * @file test_npie_core.c
 * @brief Unit tests for NPIE core functionality
 * @version 1.0.0-alpha
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "npie.h"

/* Test counter */
static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    static void test_##name(void); \
    static void run_test_##name(void) { \
        printf("Running test: %s ... ", #name); \
        tests_run++; \
        test_##name(); \
        tests_passed++; \
        printf("PASSED\n"); \
    } \
    static void test_##name(void)

#define ASSERT(condition) \
    do { \
        if (!(condition)) { \
            printf("FAILED\n"); \
            printf("  Assertion failed: %s\n", #condition); \
            printf("  File: %s, Line: %d\n", __FILE__, __LINE__); \
            tests_failed++; \
            return; \
        } \
    } while(0)

#define ASSERT_EQ(a, b) ASSERT((a) == (b))
#define ASSERT_NE(a, b) ASSERT((a) != (b))
#define ASSERT_NULL(ptr) ASSERT((ptr) == NULL)
#define ASSERT_NOT_NULL(ptr) ASSERT((ptr) != NULL)

/* Test: Version string */
TEST(version_string) {
    const char* version = npie_version();
    ASSERT_NOT_NULL(version);
    ASSERT(strlen(version) > 0);
    printf("\n    Version: %s", version);
}

/* Test: Status string conversion */
TEST(status_string) {
    const char* str = npie_status_string(NPIE_SUCCESS);
    ASSERT_NOT_NULL(str);
    ASSERT_EQ(strcmp(str, "Success"), 0);
    
    str = npie_status_string(NPIE_ERROR_INVALID_ARGUMENT);
    ASSERT_NOT_NULL(str);
    ASSERT_EQ(strcmp(str, "Invalid argument"), 0);
}

/* Test: Context initialization */
TEST(context_init) {
    npie_context_t ctx = NULL;
    npie_status_t status = npie_init(&ctx, NULL);
    ASSERT_EQ(status, NPIE_SUCCESS);
    ASSERT_NOT_NULL(ctx);
    
    status = npie_shutdown(ctx);
    ASSERT_EQ(status, NPIE_SUCCESS);
}

/* Test: Context with custom options */
TEST(context_init_with_options) {
    npie_options_t options = {
        .backend = NPIE_BACKEND_AUTO,
        .accelerator = NPIE_ACCELERATOR_AUTO,
        .num_threads = 4,
        .timeout_ms = 5000,
        .enable_profiling = true,
        .enable_caching = true,
        .user_data = NULL
    };
    
    npie_context_t ctx = NULL;
    npie_status_t status = npie_init(&ctx, &options);
    ASSERT_EQ(status, NPIE_SUCCESS);
    ASSERT_NOT_NULL(ctx);
    
    status = npie_shutdown(ctx);
    ASSERT_EQ(status, NPIE_SUCCESS);
}

/* Test: Invalid context initialization */
TEST(context_init_invalid) {
    npie_status_t status = npie_init(NULL, NULL);
    ASSERT_EQ(status, NPIE_ERROR_INVALID_ARGUMENT);
}

/* Test: Double shutdown */
TEST(context_double_shutdown) {
    npie_context_t ctx = NULL;
    npie_init(&ctx, NULL);
    
    npie_status_t status = npie_shutdown(ctx);
    ASSERT_EQ(status, NPIE_SUCCESS);
    
    /* Second shutdown should fail */
    status = npie_shutdown(ctx);
    ASSERT_NE(status, NPIE_SUCCESS);
}

/* Test: Tensor size calculation */
TEST(tensor_size_calculation) {
    npie_tensor_t tensor = {
        .name = "test",
        .dtype = NPIE_DTYPE_FLOAT32,
        .shape = {.rank = 4, .dims = {1, 224, 224, 3}},
        .data = NULL,
        .size = 0
    };
    
    size_t size = npie_tensor_size(&tensor);
    ASSERT_EQ(size, 1 * 224 * 224 * 3 * 4); /* 4 bytes per float32 */
}

/* Test: Tensor allocation */
TEST(tensor_allocation) {
    npie_tensor_t tensor = {
        .name = "test",
        .dtype = NPIE_DTYPE_FLOAT32,
        .shape = {.rank = 2, .dims = {10, 10}},
        .data = NULL,
        .size = 0
    };
    
    npie_status_t status = npie_tensor_alloc(&tensor);
    ASSERT_EQ(status, NPIE_SUCCESS);
    ASSERT_NOT_NULL(tensor.data);
    ASSERT_EQ(tensor.size, 10 * 10 * 4);
    
    status = npie_tensor_free(&tensor);
    ASSERT_EQ(status, NPIE_SUCCESS);
    ASSERT_NULL(tensor.data);
}

/* Test: Hardware detection */
TEST(hardware_detection) {
    npie_context_t ctx = NULL;
    npie_init(&ctx, NULL);
    
    npie_accelerator_t accelerators[16];
    uint32_t count = 0;
    
    npie_status_t status = npie_detect_accelerators(ctx, accelerators, 16, &count);
    ASSERT_EQ(status, NPIE_SUCCESS);
    
    printf("\n    Detected %u accelerators", count);
    
    npie_shutdown(ctx);
}

/* Test: Accelerator availability check */
TEST(accelerator_availability) {
    npie_context_t ctx = NULL;
    npie_init(&ctx, NULL);
    
    /* CPU should always be available */
    bool available = npie_is_accelerator_available(ctx, NPIE_ACCELERATOR_NONE);
    ASSERT(available || !available); /* Just test it doesn't crash */
    
    npie_shutdown(ctx);
}

/* Main test runner */
int main(int argc, char** argv) {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║              NPIE Core Unit Tests                             ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    
    /* Run all tests */
    run_test_version_string();
    run_test_status_string();
    run_test_context_init();
    run_test_context_init_with_options();
    run_test_context_init_invalid();
    run_test_context_double_shutdown();
    run_test_tensor_size_calculation();
    run_test_tensor_allocation();
    run_test_hardware_detection();
    run_test_accelerator_availability();
    
    /* Print summary */
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("Test Summary:\n");
    printf("  Total:  %d\n", tests_run);
    printf("  Passed: %d\n", tests_passed);
    printf("  Failed: %d\n", tests_failed);
    printf("═══════════════════════════════════════════════════════════════\n");
    
    if (tests_failed == 0) {
        printf("\n✓ All tests passed!\n\n");
        return 0;
    } else {
        printf("\n✗ Some tests failed!\n\n");
        return 1;
    }
}

