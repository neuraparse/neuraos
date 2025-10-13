/**
 * @file gpu_accel.h
 * @brief GPU Acceleration Interface for AI Inference
 * @version 1.0.0-alpha
 * @date October 2025
 */

#ifndef GPU_ACCEL_H
#define GPU_ACCEL_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* GPU types */
typedef enum {
    GPU_TYPE_UNKNOWN = 0,
    GPU_TYPE_MALI,          /* ARM Mali GPU */
    GPU_TYPE_ADRENO,        /* Qualcomm Adreno */
    GPU_TYPE_VIDEOCORE,     /* Broadcom VideoCore */
    GPU_TYPE_INTEL_UHD,     /* Intel UHD Graphics */
    GPU_TYPE_NVIDIA,        /* NVIDIA GPU */
    GPU_TYPE_AMD,           /* AMD GPU */
} gpu_type_t;

/* GPU API types */
typedef enum {
    GPU_API_OPENCL = 0,
    GPU_API_OPENGL_ES,
    GPU_API_VULKAN,
    GPU_API_CUDA,
} gpu_api_t;

/* GPU capabilities */
typedef struct {
    gpu_type_t type;
    char name[64];
    char vendor[64];
    char version[32];
    uint32_t compute_units;
    uint32_t max_frequency_mhz;
    uint64_t memory_size;
    bool supports_fp16;
    bool supports_fp32;
    bool supports_int8;
    gpu_api_t supported_apis[4];
    int num_apis;
} gpu_capabilities_t;

/* GPU handle */
typedef struct gpu_device* gpu_device_t;

/* GPU context */
typedef struct gpu_context* gpu_context_t;

/**
 * @brief Initialize GPU acceleration
 */
int gpu_accel_init(void);

/**
 * @brief Cleanup GPU acceleration
 */
void gpu_accel_cleanup(void);

/**
 * @brief Detect available GPUs
 */
int gpu_detect_devices(gpu_device_t* devices, int max_devices);

/**
 * @brief Open GPU device
 */
gpu_device_t gpu_open(int device_id);

/**
 * @brief Close GPU device
 */
void gpu_close(gpu_device_t device);

/**
 * @brief Get GPU capabilities
 */
int gpu_get_capabilities(gpu_device_t device, gpu_capabilities_t* caps);

/**
 * @brief Create GPU context
 */
gpu_context_t gpu_create_context(gpu_device_t device, gpu_api_t api);

/**
 * @brief Destroy GPU context
 */
void gpu_destroy_context(gpu_context_t context);

/**
 * @brief Check if GPU API is available
 */
bool gpu_is_api_available(gpu_api_t api);

/**
 * @brief Get OpenCL platform/device info
 */
int gpu_get_opencl_info(gpu_device_t device, void** platform, void** cl_device);

/**
 * @brief Get Vulkan device info
 */
int gpu_get_vulkan_info(gpu_device_t device, void** vk_device);

#ifdef __cplusplus
}
#endif

#endif /* GPU_ACCEL_H */

