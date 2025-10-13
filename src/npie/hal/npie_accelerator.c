/**
 * @file npie_accelerator.c
 * @brief NPIE Hardware Accelerator Management
 * @version 1.0.0-alpha
 * @date October 2025
 */

#include "npie.h"
#include "npie_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef ENABLE_NPU_SUPPORT
#include "../../drivers/npu/npu_driver.h"
#endif

#ifdef ENABLE_GPU_ACCELERATION
#include "../../drivers/accelerators/gpu_accel.h"
#endif

/**
 * @brief Initialize accelerator
 */
npie_status_t npie_accelerator_init(npie_accelerator_desc_t* accel) {
    if (!accel) {
        return NPIE_ERROR_INVALID_ARGUMENT;
    }
    
    switch (accel->type) {
#ifdef ENABLE_NPU_SUPPORT
        case NPIE_ACCELERATOR_NPU: {
            int ret = npu_driver_init();
            if (ret != 0) {
                NPIE_LOG_ERROR("Failed to initialize NPU driver");
                return NPIE_ERROR_HARDWARE_NOT_AVAILABLE;
            }
            break;
        }
#endif

#ifdef ENABLE_GPU_ACCELERATION
        case NPIE_ACCELERATOR_GPU: {
            int ret = gpu_accel_init();
            if (ret != 0) {
                NPIE_LOG_ERROR("Failed to initialize GPU accelerator");
                return NPIE_ERROR_HARDWARE_NOT_AVAILABLE;
            }
            break;
        }
#endif

        case NPIE_ACCELERATOR_NONE:
            /* CPU always available */
            break;
            
        default:
            NPIE_LOG_WARN("Unknown accelerator type: %d", accel->type);
            return NPIE_ERROR_UNSUPPORTED_OPERATION;
    }
    
    return NPIE_SUCCESS;
}

/**
 * @brief Cleanup accelerator
 */
npie_status_t npie_accelerator_cleanup(npie_accelerator_desc_t* accel) {
    if (!accel) {
        return NPIE_ERROR_INVALID_ARGUMENT;
    }
    
    switch (accel->type) {
#ifdef ENABLE_NPU_SUPPORT
        case NPIE_ACCELERATOR_NPU:
            npu_driver_cleanup();
            break;
#endif

#ifdef ENABLE_GPU_ACCELERATION
        case NPIE_ACCELERATOR_GPU:
            gpu_accel_cleanup();
            break;
#endif

        default:
            break;
    }
    
    return NPIE_SUCCESS;
}

/**
 * @brief Get accelerator capabilities
 */
npie_status_t npie_accelerator_get_capabilities(npie_accelerator_desc_t* accel,
                                                npie_accelerator_caps_t* caps) {
    if (!accel || !caps) {
        return NPIE_ERROR_INVALID_ARGUMENT;
    }
    
    memset(caps, 0, sizeof(npie_accelerator_caps_t));
    
    switch (accel->type) {
        case NPIE_ACCELERATOR_NONE:
            strncpy(caps->name, "CPU", sizeof(caps->name) - 1);
            caps->supports_fp32 = true;
            caps->supports_fp16 = false;
            caps->supports_int8 = true;
            caps->max_batch_size = 1;
            caps->memory_size = 0; /* System RAM */
            break;
            
#ifdef ENABLE_NPU_SUPPORT
        case NPIE_ACCELERATOR_NPU: {
            npu_device_t dev = npu_open(accel->device_id);
            if (dev) {
                npu_capabilities_t npu_caps;
                if (npu_get_capabilities(dev, &npu_caps) == 0) {
                    strncpy(caps->name, npu_caps.name, sizeof(caps->name) - 1);
                    caps->supports_fp32 = npu_caps.supports_float32;
                    caps->supports_fp16 = npu_caps.supports_float16;
                    caps->supports_int8 = npu_caps.supports_int8;
                    caps->max_batch_size = npu_caps.max_batch_size;
                    caps->memory_size = npu_caps.memory_size;
                }
                npu_close(dev);
            }
            break;
        }
#endif

#ifdef ENABLE_GPU_ACCELERATION
        case NPIE_ACCELERATOR_GPU: {
            gpu_device_t dev = gpu_open(accel->device_id);
            if (dev) {
                gpu_capabilities_t gpu_caps;
                if (gpu_get_capabilities(dev, &gpu_caps) == 0) {
                    strncpy(caps->name, gpu_caps.name, sizeof(caps->name) - 1);
                    caps->supports_fp32 = gpu_caps.supports_fp32;
                    caps->supports_fp16 = gpu_caps.supports_fp16;
                    caps->supports_int8 = gpu_caps.supports_int8;
                    caps->max_batch_size = 8; /* Typical GPU batch size */
                    caps->memory_size = gpu_caps.memory_size;
                }
                gpu_close(dev);
            }
            break;
        }
#endif

        default:
            return NPIE_ERROR_NOT_SUPPORTED;
    }
    
    return NPIE_SUCCESS;
}

/**
 * @brief Select best accelerator for model
 */
npie_status_t npie_accelerator_select_best(npie_context_t ctx,
                                           const npie_model_info_t* model_info,
                                           npie_accelerator_desc_t* best_accel) {
    if (!ctx || !model_info || !best_accel) {
        return NPIE_ERROR_INVALID_ARGUMENT;
    }
    
    /* Simple heuristic: prefer NPU > GPU > CPU */
    npie_accelerator_t accelerators[NPIE_MAX_ACCELERATORS];
    uint32_t count = 0;
    
    npie_status_t status = npie_detect_accelerators(ctx, accelerators, 
                                                    NPIE_MAX_ACCELERATORS, &count);
    if (status != NPIE_SUCCESS || count == 0) {
        /* Fallback to CPU */
        best_accel->type = NPIE_ACCELERATOR_NONE;
        best_accel->device_id = 0;
        strncpy(best_accel->name, "CPU", sizeof(best_accel->name) - 1);
        return NPIE_SUCCESS;
    }
    
    /* Prefer NPU for quantized models */
    if (model_info->quantized) {
        for (uint32_t i = 0; i < count; i++) {
            if (accelerators[i].type == NPIE_ACCELERATOR_NPU) {
                memcpy(best_accel, &accelerators[i], sizeof(npie_accelerator_desc_t));
                return NPIE_OK;
            }
        }
    }
    
    /* Prefer GPU for large models */
    if (model_info->size > 10 * 1024 * 1024) { /* > 10 MB */
        for (uint32_t i = 0; i < count; i++) {
            if (accelerators[i].type == NPIE_ACCELERATOR_GPU) {
                memcpy(best_accel, &accelerators[i], sizeof(npie_accelerator_desc_t));
                return NPIE_OK;
            }
        }
    }
    
    /* Use first available accelerator */
    memcpy(best_accel, &accelerators[0], sizeof(npie_accelerator_desc_t));

    return NPIE_SUCCESS;
}

/**
 * @brief Allocate memory on accelerator
 */
npie_status_t npie_accelerator_alloc(npie_accelerator_desc_t* accel,
                                     size_t size, void** ptr) {
    if (!accel || !ptr || size == 0) {
        return NPIE_ERROR_INVALID_ARGUMENT;
    }
    
    switch (accel->type) {
        case NPIE_ACCELERATOR_NONE:
            *ptr = malloc(size);
            if (!*ptr) {
                return NPIE_ERROR_OUT_OF_MEMORY;
            }
            break;
            
#ifdef ENABLE_NPU_SUPPORT
        case NPIE_ACCELERATOR_NPU: {
            npu_device_t dev = npu_open(accel->device_id);
            if (!dev) {
                return NPIE_ERROR_HARDWARE_NOT_AVAILABLE;
            }
            
            npu_buffer_t* buffer = npu_alloc_buffer(dev, size);
            if (!buffer) {
                npu_close(dev);
                return NPIE_ERROR_OUT_OF_MEMORY;
            }
            
            *ptr = buffer;
            npu_close(dev);
            break;
        }
#endif

        default:
            /* Fallback to system memory */
            *ptr = malloc(size);
            if (!*ptr) {
                return NPIE_ERROR_OUT_OF_MEMORY;
            }
            break;
    }
    
    return NPIE_SUCCESS;
}

/**
 * @brief Free memory on accelerator
 */
npie_status_t npie_accelerator_free(npie_accelerator_desc_t* accel, void* ptr) {
    if (!accel || !ptr) {
        return NPIE_ERROR_INVALID_ARGUMENT;
    }
    
    switch (accel->type) {
        case NPIE_ACCELERATOR_NONE:
            free(ptr);
            break;
            
#ifdef ENABLE_NPU_SUPPORT
        case NPIE_ACCELERATOR_NPU: {
            npu_device_t dev = npu_open(accel->device_id);
            if (dev) {
                npu_free_buffer(dev, (npu_buffer_t*)ptr);
                npu_close(dev);
            }
            break;
        }
#endif

        default:
            free(ptr);
            break;
    }
    
    return NPIE_SUCCESS;
}

