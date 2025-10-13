/**
 * @file gpu_accel.c
 * @brief GPU Acceleration Implementation
 * @version 1.0.0-alpha
 * @date October 2025
 */

#include "gpu_accel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>

/* Internal device structure */
struct gpu_device {
    int device_id;
    gpu_type_t type;
    gpu_capabilities_t caps;
    bool initialized;
    void* internal_handle;
};

/* Internal context structure */
struct gpu_context {
    gpu_device_t device;
    gpu_api_t api;
    void* api_context;
};

/* Global state */
static bool g_gpu_initialized = false;
static struct gpu_device g_devices[8];
static int g_num_devices = 0;

/**
 * @brief Initialize GPU acceleration
 */
int gpu_accel_init(void) {
    if (g_gpu_initialized) {
        return 0;
    }
    
    memset(g_devices, 0, sizeof(g_devices));
    g_num_devices = 0;
    g_gpu_initialized = true;
    
    return 0;
}

/**
 * @brief Cleanup GPU acceleration
 */
void gpu_accel_cleanup(void) {
    g_num_devices = 0;
    g_gpu_initialized = false;
}

/**
 * @brief Detect Mali GPU
 */
static int detect_mali_gpu(struct gpu_device* dev) {
    /* Check for Mali GPU */
    if (access("/dev/mali0", F_OK) != 0) {
        return -1;
    }
    
    dev->type = GPU_TYPE_MALI;
    strcpy(dev->caps.name, "ARM Mali GPU");
    strcpy(dev->caps.vendor, "ARM");
    strcpy(dev->caps.version, "1.0");
    dev->caps.compute_units = 4;
    dev->caps.max_frequency_mhz = 800;
    dev->caps.memory_size = 512 * 1024 * 1024; /* 512 MB shared */
    dev->caps.supports_fp16 = true;
    dev->caps.supports_fp32 = true;
    dev->caps.supports_int8 = true;
    dev->caps.supported_apis[0] = GPU_API_OPENCL;
    dev->caps.supported_apis[1] = GPU_API_OPENGL_ES;
    dev->caps.num_apis = 2;
    
    return 0;
}

/**
 * @brief Detect VideoCore GPU (Raspberry Pi)
 */
static int detect_videocore_gpu(struct gpu_device* dev) {
    /* Check for VideoCore */
    if (access("/dev/vchiq", F_OK) != 0) {
        return -1;
    }
    
    dev->type = GPU_TYPE_VIDEOCORE;
    strcpy(dev->caps.name, "Broadcom VideoCore VI");
    strcpy(dev->caps.vendor, "Broadcom");
    strcpy(dev->caps.version, "6.0");
    dev->caps.compute_units = 1;
    dev->caps.max_frequency_mhz = 500;
    dev->caps.memory_size = 256 * 1024 * 1024; /* 256 MB shared */
    dev->caps.supports_fp16 = true;
    dev->caps.supports_fp32 = true;
    dev->caps.supports_int8 = false;
    dev->caps.supported_apis[0] = GPU_API_OPENGL_ES;
    dev->caps.num_apis = 1;
    
    return 0;
}

/**
 * @brief Detect Intel GPU
 */
static int detect_intel_gpu(struct gpu_device* dev) {
    /* Check for Intel GPU via DRM */
    DIR* dir = opendir("/sys/class/drm");
    if (!dir) {
        return -1;
    }
    
    struct dirent* entry;
    bool found = false;
    
    while ((entry = readdir(dir)) != NULL) {
        if (strstr(entry->d_name, "card") != NULL) {
            char path[512];
            snprintf(path, sizeof(path), "/sys/class/drm/%s/device/vendor", entry->d_name);

            FILE* fp = fopen(path, "r");
            if (fp) {
                char vendor[16];
                if (fgets(vendor, sizeof(vendor), fp)) {
                    if (strstr(vendor, "0x8086")) { /* Intel vendor ID */
                        found = true;
                    }
                }
                fclose(fp);
            }
            
            if (found) break;
        }
    }
    
    closedir(dir);
    
    if (!found) {
        return -1;
    }
    
    dev->type = GPU_TYPE_INTEL_UHD;
    strcpy(dev->caps.name, "Intel UHD Graphics");
    strcpy(dev->caps.vendor, "Intel");
    strcpy(dev->caps.version, "1.0");
    dev->caps.compute_units = 24;
    dev->caps.max_frequency_mhz = 1200;
    dev->caps.memory_size = 1024 * 1024 * 1024; /* 1 GB shared */
    dev->caps.supports_fp16 = true;
    dev->caps.supports_fp32 = true;
    dev->caps.supports_int8 = true;
    dev->caps.supported_apis[0] = GPU_API_OPENCL;
    dev->caps.supported_apis[1] = GPU_API_VULKAN;
    dev->caps.num_apis = 2;
    
    return 0;
}

/**
 * @brief Detect NVIDIA GPU
 */
static int detect_nvidia_gpu(struct gpu_device* dev) {
    /* Check for NVIDIA GPU */
    if (access("/dev/nvidia0", F_OK) != 0) {
        return -1;
    }
    
    dev->type = GPU_TYPE_NVIDIA;
    strcpy(dev->caps.name, "NVIDIA GPU");
    strcpy(dev->caps.vendor, "NVIDIA");
    strcpy(dev->caps.version, "1.0");
    dev->caps.compute_units = 128;
    dev->caps.max_frequency_mhz = 1500;
    dev->caps.memory_size = 2ULL * 1024 * 1024 * 1024; /* 2 GB */
    dev->caps.supports_fp16 = true;
    dev->caps.supports_fp32 = true;
    dev->caps.supports_int8 = true;
    dev->caps.supported_apis[0] = GPU_API_CUDA;
    dev->caps.supported_apis[1] = GPU_API_OPENCL;
    dev->caps.supported_apis[2] = GPU_API_VULKAN;
    dev->caps.num_apis = 3;
    
    return 0;
}

/**
 * @brief Detect available GPUs
 */
int gpu_detect_devices(gpu_device_t* devices, int max_devices) {
    if (!g_gpu_initialized) {
        gpu_accel_init();
    }
    
    g_num_devices = 0;
    
    /* Try to detect different GPU types */
    if (g_num_devices < max_devices && detect_mali_gpu(&g_devices[g_num_devices]) == 0) {
        g_devices[g_num_devices].device_id = g_num_devices;
        g_devices[g_num_devices].initialized = true;
        if (devices) devices[g_num_devices] = &g_devices[g_num_devices];
        g_num_devices++;
    }
    
    if (g_num_devices < max_devices && detect_videocore_gpu(&g_devices[g_num_devices]) == 0) {
        g_devices[g_num_devices].device_id = g_num_devices;
        g_devices[g_num_devices].initialized = true;
        if (devices) devices[g_num_devices] = &g_devices[g_num_devices];
        g_num_devices++;
    }
    
    if (g_num_devices < max_devices && detect_intel_gpu(&g_devices[g_num_devices]) == 0) {
        g_devices[g_num_devices].device_id = g_num_devices;
        g_devices[g_num_devices].initialized = true;
        if (devices) devices[g_num_devices] = &g_devices[g_num_devices];
        g_num_devices++;
    }
    
    if (g_num_devices < max_devices && detect_nvidia_gpu(&g_devices[g_num_devices]) == 0) {
        g_devices[g_num_devices].device_id = g_num_devices;
        g_devices[g_num_devices].initialized = true;
        if (devices) devices[g_num_devices] = &g_devices[g_num_devices];
        g_num_devices++;
    }
    
    return g_num_devices;
}

/**
 * @brief Open GPU device
 */
gpu_device_t gpu_open(int device_id) {
    if (device_id < 0 || device_id >= g_num_devices) {
        return NULL;
    }
    
    return &g_devices[device_id];
}

/**
 * @brief Close GPU device
 */
void gpu_close(gpu_device_t device) {
    (void)device;
    /* Device remains open for reuse */
}

/**
 * @brief Get GPU capabilities
 */
int gpu_get_capabilities(gpu_device_t device, gpu_capabilities_t* caps) {
    if (!device || !caps) {
        return -1;
    }
    
    memcpy(caps, &device->caps, sizeof(gpu_capabilities_t));
    return 0;
}

/**
 * @brief Create GPU context
 */
gpu_context_t gpu_create_context(gpu_device_t device, gpu_api_t api) {
    if (!device) {
        return NULL;
    }
    
    /* Check if API is supported */
    bool supported = false;
    for (int i = 0; i < device->caps.num_apis; i++) {
        if (device->caps.supported_apis[i] == api) {
            supported = true;
            break;
        }
    }
    
    if (!supported) {
        return NULL;
    }
    
    gpu_context_t ctx = (gpu_context_t)malloc(sizeof(struct gpu_context));
    if (!ctx) {
        return NULL;
    }
    
    ctx->device = device;
    ctx->api = api;
    ctx->api_context = NULL;
    
    return ctx;
}

/**
 * @brief Destroy GPU context
 */
void gpu_destroy_context(gpu_context_t context) {
    if (context) {
        free(context);
    }
}

/**
 * @brief Check if GPU API is available
 */
bool gpu_is_api_available(gpu_api_t api) {
    /* Check system-wide API availability */
    switch (api) {
        case GPU_API_OPENCL:
            return access("/usr/lib/libOpenCL.so", F_OK) == 0;
        case GPU_API_VULKAN:
            return access("/usr/lib/libvulkan.so", F_OK) == 0;
        case GPU_API_CUDA:
            return access("/usr/lib/libcuda.so", F_OK) == 0;
        case GPU_API_OPENGL_ES:
            return access("/usr/lib/libGLESv2.so", F_OK) == 0;
        default:
            return false;
    }
}

/**
 * @brief Get OpenCL platform/device info
 */
int gpu_get_opencl_info(gpu_device_t device, void** platform, void** cl_device) {
    if (!device) {
        return -1;
    }
    
    /* This would return actual OpenCL platform/device handles */
    /* For now, return NULL as placeholder */
    if (platform) *platform = NULL;
    if (cl_device) *cl_device = NULL;
    
    return 0;
}

/**
 * @brief Get Vulkan device info
 */
int gpu_get_vulkan_info(gpu_device_t device, void** vk_device) {
    if (!device) {
        return -1;
    }
    
    /* This would return actual Vulkan device handle */
    if (vk_device) *vk_device = NULL;
    
    return 0;
}

