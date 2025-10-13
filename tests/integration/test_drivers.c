/**
 * @file test_drivers.c
 * @brief Hardware Driver Integration Tests
 * @version 1.0.0-alpha
 * @date October 2025
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../src/drivers/npu/npu_driver.h"
#include "../../src/drivers/accelerators/gpu_accel.h"

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL: %s at %s:%d\n", msg, __FILE__, __LINE__); \
            return -1; \
        } \
    } while(0)

#define TEST_PASS(name) \
    printf("PASS: %s\n", name)

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

/**
 * @brief Test NPU driver initialization
 */
static int test_npu_init(void) {
    printf("Running: test_npu_init\n");
    
    int ret = npu_driver_init();
    TEST_ASSERT(ret == 0, "NPU driver initialization failed");
    
    npu_driver_cleanup();
    
    TEST_PASS("test_npu_init");
    return 0;
}

/**
 * @brief Test NPU device detection
 */
static int test_npu_detection(void) {
    printf("Running: test_npu_detection\n");
    
    npu_driver_init();
    
    npu_device_t devices[8];
    int num_devices = npu_detect_devices(devices, 8);
    
    printf("  Detected %d NPU device(s)\n", num_devices);
    
    for (int i = 0; i < num_devices; i++) {
        npu_capabilities_t caps;
        if (npu_get_capabilities(devices[i], &caps) == 0) {
            printf("  NPU %d: %s\n", i, caps.name);
            printf("    Cores: %d\n", caps.num_cores);
            printf("    Frequency: %d MHz\n", caps.max_frequency_mhz);
            printf("    Memory: %llu MB\n", caps.memory_size / (1024*1024));
        }
    }
    
    npu_driver_cleanup();
    
    TEST_PASS("test_npu_detection");
    return 0;
}

/**
 * @brief Test NPU buffer operations
 */
static int test_npu_buffers(void) {
    printf("Running: test_npu_buffers\n");
    
    npu_driver_init();
    
    npu_device_t devices[8];
    int num_devices = npu_detect_devices(devices, 8);
    
    if (num_devices > 0) {
        npu_device_t dev = npu_open(0);
        TEST_ASSERT(dev != NULL, "Failed to open NPU device");
        
        /* Allocate buffer */
        npu_buffer_t* buffer = npu_alloc_buffer(dev, 1024);
        TEST_ASSERT(buffer != NULL, "Failed to allocate NPU buffer");
        
        /* Copy data to buffer */
        uint8_t test_data[1024];
        memset(test_data, 0xAA, sizeof(test_data));
        
        int ret = npu_copy_to_buffer(dev, buffer, test_data, sizeof(test_data));
        TEST_ASSERT(ret == 0, "Failed to copy to NPU buffer");
        
        /* Copy data from buffer */
        uint8_t read_data[1024];
        ret = npu_copy_from_buffer(dev, buffer, read_data, sizeof(read_data));
        TEST_ASSERT(ret == 0, "Failed to copy from NPU buffer");
        
        /* Verify data */
        TEST_ASSERT(memcmp(test_data, read_data, sizeof(test_data)) == 0,
                   "Buffer data mismatch");
        
        /* Free buffer */
        npu_free_buffer(dev, buffer);
        
        npu_close(dev);
    } else {
        printf("  No NPU devices available, skipping buffer test\n");
    }
    
    npu_driver_cleanup();
    
    TEST_PASS("test_npu_buffers");
    return 0;
}

/**
 * @brief Test GPU driver initialization
 */
static int test_gpu_init(void) {
    printf("Running: test_gpu_init\n");
    
    int ret = gpu_accel_init();
    TEST_ASSERT(ret == 0, "GPU acceleration initialization failed");
    
    gpu_accel_cleanup();
    
    TEST_PASS("test_gpu_init");
    return 0;
}

/**
 * @brief Test GPU device detection
 */
static int test_gpu_detection(void) {
    printf("Running: test_gpu_detection\n");
    
    gpu_accel_init();
    
    gpu_device_t devices[8];
    int num_devices = gpu_detect_devices(devices, 8);
    
    printf("  Detected %d GPU device(s)\n", num_devices);
    
    for (int i = 0; i < num_devices; i++) {
        gpu_capabilities_t caps;
        if (gpu_get_capabilities(devices[i], &caps) == 0) {
            printf("  GPU %d: %s (%s)\n", i, caps.name, caps.vendor);
            printf("    Compute Units: %d\n", caps.compute_units);
            printf("    Frequency: %d MHz\n", caps.max_frequency_mhz);
            printf("    Memory: %llu MB\n", caps.memory_size / (1024*1024));
            printf("    Supported APIs: ");
            for (int j = 0; j < caps.num_apis; j++) {
                const char* api_names[] = {"OpenCL", "OpenGL ES", "Vulkan", "CUDA"};
                printf("%s ", api_names[caps.supported_apis[j]]);
            }
            printf("\n");
        }
    }
    
    gpu_accel_cleanup();
    
    TEST_PASS("test_gpu_detection");
    return 0;
}

/**
 * @brief Test GPU API availability
 */
static int test_gpu_api_availability(void) {
    printf("Running: test_gpu_api_availability\n");
    
    gpu_accel_init();
    
    printf("  OpenCL:    %s\n", gpu_is_api_available(GPU_API_OPENCL) ? "Available" : "Not available");
    printf("  Vulkan:    %s\n", gpu_is_api_available(GPU_API_VULKAN) ? "Available" : "Not available");
    printf("  CUDA:      %s\n", gpu_is_api_available(GPU_API_CUDA) ? "Available" : "Not available");
    printf("  OpenGL ES: %s\n", gpu_is_api_available(GPU_API_OPENGL_ES) ? "Available" : "Not available");
    
    gpu_accel_cleanup();
    
    TEST_PASS("test_gpu_api_availability");
    return 0;
}

/**
 * @brief Run all tests
 */
static void run_all_tests(void) {
    typedef int (*test_func_t)(void);
    
    test_func_t tests[] = {
        test_npu_init,
        test_npu_detection,
        test_npu_buffers,
        test_gpu_init,
        test_gpu_detection,
        test_gpu_api_availability,
    };
    
    int num_tests = sizeof(tests) / sizeof(tests[0]);
    
    printf("\n");
    printf("========================================\n");
    printf("Hardware Driver Integration Tests\n");
    printf("========================================\n\n");
    
    for (int i = 0; i < num_tests; i++) {
        tests_run++;
        
        if (tests[i]() == 0) {
            tests_passed++;
        } else {
            tests_failed++;
        }
        
        printf("\n");
    }
    
    printf("========================================\n");
    printf("Test Results\n");
    printf("========================================\n");
    printf("Total:  %d\n", tests_run);
    printf("Passed: %d\n", tests_passed);
    printf("Failed: %d\n", tests_failed);
    printf("========================================\n");
}

/**
 * @brief Main entry point
 */
int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    run_all_tests();
    
    return (tests_failed == 0) ? 0 : 1;
}

