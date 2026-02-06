/**
 * @file test_drivers.c
 * @brief Hardware Driver Integration Tests
 * @version 1.0.0-alpha
 * @date October 2025
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
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
            printf("    Memory: %" PRIu64 " MB\n", (uint64_t)(caps.memory_size / (1024*1024)));
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
 * @brief Test simulated NPU full inference pipeline
 */
static int test_npu_simulated_inference(void) {
    printf("Running: test_npu_simulated_inference\n");

    npu_driver_init();

    npu_device_t devices[8];
    int num_devices = npu_detect_devices(devices, 8);
    TEST_ASSERT(num_devices > 0, "No NPU devices detected (simulated should always exist)");

    /* Find simulated NPU */
    npu_device_t sim_dev = NULL;
    for (int i = 0; i < num_devices; i++) {
        npu_capabilities_t caps;
        if (npu_get_capabilities(devices[i], &caps) == 0 &&
            caps.type == NPU_TYPE_SIMULATED) {
            sim_dev = devices[i];
            printf("  Found simulated NPU: %s (v%s)\n", caps.name, caps.version);
            printf("    Cores: %d, Freq: %d MHz, Mem: %" PRIu64 " MB\n",
                   caps.num_cores, caps.max_frequency_mhz,
                   (uint64_t)(caps.memory_size / (1024 * 1024)));
            printf("    INT8: %d, INT16: %d, FP16: %d, FP32: %d\n",
                   caps.supports_int8, caps.supports_int16,
                   caps.supports_float16, caps.supports_float32);
            break;
        }
    }
    TEST_ASSERT(sim_dev != NULL, "Simulated NPU not found");

    /* Load a fake model (bytes with values above 128 to produce positive weights) */
    uint8_t model_data[64];
    for (int i = 0; i < 64; i++) model_data[i] = (uint8_t)(128 + (i % 64));
    void* model = npu_load_model(sim_dev, model_data, sizeof(model_data));
    TEST_ASSERT(model != NULL, "Failed to load model on simulated NPU");

    /* Use 1024-element buffers for measurable computation time */
    const int num_elems = 1024;
    npu_buffer_t* input = npu_alloc_buffer(sim_dev, num_elems * sizeof(float));
    TEST_ASSERT(input != NULL, "Failed to allocate input buffer");
    float* in_data = (float*)malloc(num_elems * sizeof(float));
    for (int i = 0; i < num_elems; i++) in_data[i] = 1.0f;
    npu_copy_to_buffer(sim_dev, input, in_data, num_elems * sizeof(float));

    npu_buffer_t* output = npu_alloc_buffer(sim_dev, num_elems * sizeof(float));
    TEST_ASSERT(output != NULL, "Failed to allocate output buffer");

    /* Run inference */
    npu_buffer_t* inputs_arr[1] = {input};
    npu_buffer_t* outputs_arr[1] = {output};
    int ret = npu_execute(sim_dev, model, inputs_arr, 1, outputs_arr, 1);
    TEST_ASSERT(ret == 0, "Inference execution failed");

    /* Read output and verify ReLU: all values >= 0 */
    float* out_data = (float*)malloc(num_elems * sizeof(float));
    npu_copy_from_buffer(sim_dev, output, out_data, num_elems * sizeof(float));
    printf("  Inference output[0..3]: [%.4f, %.4f, %.4f, %.4f]\n",
           out_data[0], out_data[1], out_data[2], out_data[3]);

    int has_positive = 0;
    for (int i = 0; i < num_elems; i++) {
        TEST_ASSERT(out_data[i] >= 0.0f, "ReLU violation: negative output");
        if (out_data[i] > 0.0f) has_positive = 1;
    }
    TEST_ASSERT(has_positive, "Expected at least some positive output values");

    /* Check statistics after 1 inference */
    uint64_t inferences, total_time_us, power_mw;
    ret = npu_get_stats(sim_dev, &inferences, &total_time_us, &power_mw);
    TEST_ASSERT(ret == 0, "Failed to get NPU stats");
    TEST_ASSERT(inferences == 1, "Inference count should be 1");
    TEST_ASSERT(total_time_us > 0, "Total inference time should be > 0");
    TEST_ASSERT(power_mw > 0, "Simulated power should be > 0");
    printf("  Stats: inferences=%" PRIu64 ", time=%" PRIu64 " us, power=%" PRIu64 " mW\n",
           inferences, total_time_us, power_mw);

    /* Test power management */
    ret = npu_set_power_state(sim_dev, false);
    TEST_ASSERT(ret == 0, "Failed to set power state");
    npu_get_stats(sim_dev, NULL, NULL, &power_mw);
    TEST_ASSERT(power_mw == 0, "Power should be 0 when disabled");

    ret = npu_set_power_state(sim_dev, true);
    TEST_ASSERT(ret == 0, "Failed to re-enable power");

    /* Test frequency scaling */
    ret = npu_set_frequency(sim_dev, 500);
    TEST_ASSERT(ret == 0, "Failed to set frequency to 500 MHz");

    ret = npu_set_frequency(sim_dev, 2000);
    TEST_ASSERT(ret != 0, "Setting frequency above max should fail");

    /* Run a second inference and check stats increment */
    ret = npu_execute(sim_dev, model, inputs_arr, 1, outputs_arr, 1);
    TEST_ASSERT(ret == 0, "Second inference failed");
    npu_get_stats(sim_dev, &inferences, NULL, NULL);
    TEST_ASSERT(inferences == 2, "Inference count should be 2 after second run");

    /* Cleanup */
    free(in_data);
    free(out_data);
    npu_free_buffer(sim_dev, input);
    npu_free_buffer(sim_dev, output);
    npu_unload_model(sim_dev, model);
    npu_driver_cleanup();

    TEST_PASS("test_npu_simulated_inference");
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
            printf("    Memory: %" PRIu64 " MB\n", (uint64_t)(caps.memory_size / (1024*1024)));
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
        test_npu_simulated_inference,
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

