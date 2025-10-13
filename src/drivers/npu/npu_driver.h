/**
 * @file npu_driver.h
 * @brief Generic NPU (Neural Processing Unit) Driver Interface
 * @version 1.0.0-alpha
 * @date October 2025
 */

#ifndef NPU_DRIVER_H
#define NPU_DRIVER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* NPU types */
typedef enum {
    NPU_TYPE_UNKNOWN = 0,
    NPU_TYPE_EDGE_TPU,      /* Google Edge TPU */
    NPU_TYPE_ETHOS_U,       /* ARM Ethos-U NPU */
    NPU_TYPE_HAILO,         /* Hailo AI Processor */
    NPU_TYPE_ROCKCHIP_NPU,  /* Rockchip NPU */
    NPU_TYPE_AMLOGIC_NPU,   /* Amlogic NPU */
    NPU_TYPE_CUSTOM         /* Custom NPU */
} npu_type_t;

/* NPU capabilities */
typedef struct {
    npu_type_t type;
    char name[64];
    char version[32];
    uint32_t max_frequency_mhz;
    uint32_t num_cores;
    uint64_t memory_size;
    bool supports_int8;
    bool supports_int16;
    bool supports_float16;
    bool supports_float32;
    uint32_t max_batch_size;
} npu_capabilities_t;

/* NPU handle */
typedef struct npu_device* npu_device_t;

/* NPU buffer */
typedef struct {
    void* data;
    size_t size;
    uint64_t physical_addr;
    void* internal;
} npu_buffer_t;

/**
 * @brief Initialize NPU driver
 */
int npu_driver_init(void);

/**
 * @brief Cleanup NPU driver
 */
void npu_driver_cleanup(void);

/**
 * @brief Detect available NPUs
 * @param devices Array to store detected devices
 * @param max_devices Maximum number of devices to detect
 * @return Number of devices detected
 */
int npu_detect_devices(npu_device_t* devices, int max_devices);

/**
 * @brief Open NPU device
 * @param device_id Device ID (0-based)
 * @return NPU device handle or NULL on error
 */
npu_device_t npu_open(int device_id);

/**
 * @brief Close NPU device
 */
void npu_close(npu_device_t device);

/**
 * @brief Get NPU capabilities
 */
int npu_get_capabilities(npu_device_t device, npu_capabilities_t* caps);

/**
 * @brief Allocate NPU buffer
 */
npu_buffer_t* npu_alloc_buffer(npu_device_t device, size_t size);

/**
 * @brief Free NPU buffer
 */
void npu_free_buffer(npu_device_t device, npu_buffer_t* buffer);

/**
 * @brief Copy data to NPU buffer
 */
int npu_copy_to_buffer(npu_device_t device, npu_buffer_t* buffer, 
                       const void* data, size_t size);

/**
 * @brief Copy data from NPU buffer
 */
int npu_copy_from_buffer(npu_device_t device, const npu_buffer_t* buffer,
                         void* data, size_t size);

/**
 * @brief Load model to NPU
 * @param device NPU device
 * @param model_data Model binary data
 * @param model_size Size of model data
 * @return Model handle or NULL on error
 */
void* npu_load_model(npu_device_t device, const void* model_data, size_t model_size);

/**
 * @brief Unload model from NPU
 */
void npu_unload_model(npu_device_t device, void* model_handle);

/**
 * @brief Execute inference on NPU
 * @param device NPU device
 * @param model_handle Model handle
 * @param inputs Array of input buffers
 * @param num_inputs Number of input buffers
 * @param outputs Array of output buffers
 * @param num_outputs Number of output buffers
 * @return 0 on success, negative on error
 */
int npu_execute(npu_device_t device, void* model_handle,
                npu_buffer_t** inputs, int num_inputs,
                npu_buffer_t** outputs, int num_outputs);

/**
 * @brief Set NPU power state
 */
int npu_set_power_state(npu_device_t device, bool enabled);

/**
 * @brief Set NPU frequency
 */
int npu_set_frequency(npu_device_t device, uint32_t frequency_mhz);

/**
 * @brief Get NPU statistics
 */
int npu_get_stats(npu_device_t device, uint64_t* inferences, 
                  uint64_t* total_time_us, uint64_t* power_mw);

#ifdef __cplusplus
}
#endif

#endif /* NPU_DRIVER_H */

