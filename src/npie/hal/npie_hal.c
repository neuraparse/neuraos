/**
 * @file npie_hal.c
 * @brief Hardware Abstraction Layer for NPIE
 * @version 1.0.0-alpha
 * @date October 2025
 *
 * Provides unified interface for hardware accelerators (GPU, NPU, TPU)
 */

#include "npie.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>

#include <unistd.h>

/**
 * @brief Detect GPU accelerators
 */
static int detect_gpu(npie_accelerator_t* accelerators, int max_count, int* count) {
    int found = 0;

    /* Check for DRM devices (generic GPU) */
    if (access("/dev/dri/card0", F_OK) == 0) {
        if (found < max_count) {
            accelerators[found] = NPIE_ACCELERATOR_GPU;
            found++;
        }
    }

    /* Check for Mali GPU */
    if (access("/dev/mali0", F_OK) == 0) {
        if (found < max_count) {
            accelerators[found] = NPIE_ACCELERATOR_GPU;
            found++;
        }
    }

    *count = found;
    return found > 0 ? 0 : -1;
}

/**
 * @brief Detect NPU accelerators
 */
static int detect_npu(npie_accelerator_t* accelerators, int max_count, int* count) {
    int found = 0;

    /* Check for generic NPU device */
    if (access("/dev/npu", F_OK) == 0) {
        if (found < max_count) {
            accelerators[found] = NPIE_ACCELERATOR_NPU;
            found++;
        }
    }

    /* Check for Qualcomm Hexagon */
    if (access("/dev/qcom_npu", F_OK) == 0) {
        if (found < max_count) {
            accelerators[found] = NPIE_ACCELERATOR_NPU;
            found++;
        }
    }

    /* Check for ARM Ethos-U */
    if (access("/dev/ethosu0", F_OK) == 0) {
        if (found < max_count) {
            accelerators[found] = NPIE_ACCELERATOR_NPU;
            found++;
        }
    }

    *count = found;
    return found > 0 ? 0 : -1;
}

/**
 * @brief Detect TPU accelerators
 */
static int detect_tpu(npie_accelerator_t* accelerators, int max_count, int* count) {
    int found = 0;

    /* Check for Edge TPU via USB */
    FILE* fp = popen("lsusb | grep -i 'Google.*Edge TPU'", "r");
    if (fp) {
        char line[256];
        if (fgets(line, sizeof(line), fp)) {
            if (found < max_count) {
                accelerators[found] = NPIE_ACCELERATOR_TPU;
                found++;
            }
        }
        pclose(fp);
    }

    /* Check for Edge TPU via PCIe */
    if (access("/dev/apex_0", F_OK) == 0) {
        if (found < max_count) {
            accelerators[found] = NPIE_ACCELERATOR_TPU;
            found++;
        }
    }

    *count = found;
    return found > 0 ? 0 : -1;
}

/**
 * @brief Detect DSP accelerators
 */
static int detect_dsp(npie_accelerator_t* accelerators, int max_count, int* count) {
    int found = 0;

    /* Check for Hexagon DSP */
    if (access("/dev/adsprpc-smd", F_OK) == 0) {
        if (found < max_count) {
            accelerators[found] = NPIE_ACCELERATOR_DSP;
            found++;
        }
    }

    *count = found;
    return found > 0 ? 0 : -1;
}

/**
 * @brief Detect all available accelerators
 */
npie_status_t npie_detect_accelerators(npie_context_t ctx,
                                       npie_accelerator_t* accelerators,
                                       uint32_t max_count,
                                       uint32_t* count) {
    if (!ctx || !accelerators || !count || max_count == 0) {
        return NPIE_ERROR_INVALID_ARGUMENT;
    }

    uint32_t total = 0;
    int temp_count;

    /* Detect GPUs */
    if (detect_gpu(&accelerators[total], max_count - total, &temp_count) == 0) {
        total += temp_count;
    }

    /* Detect NPUs */
    if (total < max_count &&
        detect_npu(&accelerators[total], max_count - total, &temp_count) == 0) {
        total += temp_count;
    }

    /* Detect TPUs */
    if (total < max_count &&
        detect_tpu(&accelerators[total], max_count - total, &temp_count) == 0) {
        total += temp_count;
    }

    /* Detect DSPs */
    if (total < max_count &&
        detect_dsp(&accelerators[total], max_count - total, &temp_count) == 0) {
        total += temp_count;
    }

    *count = total;

    /* Caching skipped: context internals are opaque here */

    return NPIE_SUCCESS;
}

/**
 * @brief Check if specific accelerator is available
 */
bool npie_is_accelerator_available(npie_context_t ctx, npie_accelerator_t accelerator) {
    if (!ctx) {
        return false;
    }

    npie_accelerator_t accel_list[16];
    uint32_t count = 0;
    if (npie_detect_accelerators(ctx, accel_list, 16, &count) != NPIE_SUCCESS) {
        return false;
    }

    for (uint32_t i = 0; i < count; i++) {
        if (accel_list[i] == accelerator) {
            return true;
        }
    }
    return false;
}

/**
 * @brief Initialize accelerator for use
 */
npie_status_t npie_hal_init_accelerator(npie_accelerator_t accelerator, void** handle) {
    if (!handle) {
        return NPIE_ERROR_INVALID_ARGUMENT;
    }

    switch (accelerator) {
        case NPIE_ACCELERATOR_GPU:
            /* Initialize GPU (OpenCL, Vulkan, etc.) */
            #ifdef NEURAOS_ENABLE_GPU
            return npie_hal_init_gpu(handle);
            #else
            return NPIE_ERROR_UNSUPPORTED_OPERATION;
            #endif

        case NPIE_ACCELERATOR_NPU:
            /* Initialize NPU */
            #ifdef NEURAOS_ENABLE_NPU
            return npie_hal_init_npu(handle);
            #else
            return NPIE_ERROR_UNSUPPORTED_OPERATION;
            #endif

        case NPIE_ACCELERATOR_TPU:
            /* Initialize TPU (Edge TPU) */
            #ifdef NEURAOS_ENABLE_TPU
            return npie_hal_init_tpu(handle);
            #else
            return NPIE_ERROR_UNSUPPORTED_OPERATION;
            #endif

        case NPIE_ACCELERATOR_DSP:
            /* Initialize DSP */
            #ifdef NEURAOS_ENABLE_DSP
            return npie_hal_init_dsp(handle);
            #else
            return NPIE_ERROR_UNSUPPORTED_OPERATION;
            #endif

        default:
            return NPIE_ERROR_UNSUPPORTED_OPERATION;
    }
}

/**
 * @brief Shutdown accelerator
 */
npie_status_t npie_hal_shutdown_accelerator(npie_accelerator_t accelerator, void* handle) {
    if (!handle) {
        return NPIE_ERROR_INVALID_ARGUMENT;
    }

    switch (accelerator) {
        case NPIE_ACCELERATOR_GPU:
            #ifdef NEURAOS_ENABLE_GPU
            return npie_hal_shutdown_gpu(handle);
            #endif
            break;

        case NPIE_ACCELERATOR_NPU:
            #ifdef NEURAOS_ENABLE_NPU
            return npie_hal_shutdown_npu(handle);
            #endif
            break;

        case NPIE_ACCELERATOR_TPU:
            #ifdef NEURAOS_ENABLE_TPU
            return npie_hal_shutdown_tpu(handle);
            #endif
            break;

        case NPIE_ACCELERATOR_DSP:
            #ifdef NEURAOS_ENABLE_DSP
            return npie_hal_shutdown_dsp(handle);
            #endif
            break;

        default:
            break;
    }

    return NPIE_SUCCESS;
}

