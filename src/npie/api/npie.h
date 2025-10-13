/**
 * @file npie.h
 * @brief NeuraParse Inference Engine (NPIE) Public API
 * @version 1.0.0-alpha
 * @date October 2025
 *
 * This is the main public API for the NeuraParse Inference Engine.
 * It provides a unified interface for AI model loading, inference,
 * and hardware acceleration across multiple backends (LiteRT, ONNX Runtime, etc.)
 */

#ifndef NPIE_H
#define NPIE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/**
 * @brief NPIE version information
 */
#define NPIE_VERSION_MAJOR 1
#define NPIE_VERSION_MINOR 0
#define NPIE_VERSION_PATCH 0
#define NPIE_VERSION_STRING "1.0.0-alpha"

/**
 * @brief Return codes
 */
typedef enum {
    NPIE_SUCCESS = 0,
    NPIE_ERROR_INVALID_ARGUMENT = -1,
    NPIE_ERROR_OUT_OF_MEMORY = -2,
    NPIE_ERROR_MODEL_LOAD_FAILED = -3,
    NPIE_ERROR_INFERENCE_FAILED = -4,
    NPIE_ERROR_UNSUPPORTED_OPERATION = -5,
    NPIE_ERROR_HARDWARE_NOT_AVAILABLE = -6,
    NPIE_ERROR_TIMEOUT = -7,
    NPIE_ERROR_NOT_INITIALIZED = -8,
    NPIE_ERROR_ALREADY_INITIALIZED = -9,
    NPIE_ERROR_IO = -10,
    NPIE_ERROR_UNKNOWN = -99
} npie_status_t;

/**
 * @brief AI backend types
 */
typedef enum {
    NPIE_BACKEND_AUTO = 0,          ///< Automatically select best backend
    NPIE_BACKEND_LITERT,            ///< LiteRT (TensorFlow Lite)
    NPIE_BACKEND_ONNXRUNTIME,       ///< ONNX Runtime
    NPIE_BACKEND_EMLEARN,           ///< emlearn (classical ML)
    NPIE_BACKEND_WASMEDGE,          ///< WebAssembly Edge
    NPIE_BACKEND_CUSTOM             ///< Custom backend
} npie_backend_t;

/**
 * @brief Hardware accelerator types
 */
typedef enum {
    NPIE_ACCELERATOR_NONE = 0,      ///< CPU only
    NPIE_ACCELERATOR_AUTO,          ///< Auto-detect and use best available
    NPIE_ACCELERATOR_GPU,           ///< GPU (OpenCL, Vulkan, CUDA)
    NPIE_ACCELERATOR_NPU,           ///< Neural Processing Unit
    NPIE_ACCELERATOR_TPU,           ///< Tensor Processing Unit (e.g., Edge TPU)
    NPIE_ACCELERATOR_DSP,           ///< Digital Signal Processor
    NPIE_ACCELERATOR_CUSTOM         ///< Custom accelerator
} npie_accelerator_t;

/**
 * @brief Data types for tensors
 */
typedef enum {
    NPIE_DTYPE_FLOAT32 = 0,
    NPIE_DTYPE_FLOAT16,
    NPIE_DTYPE_INT8,
    NPIE_DTYPE_UINT8,
    NPIE_DTYPE_INT16,
    NPIE_DTYPE_INT32,
    NPIE_DTYPE_INT64,
    NPIE_DTYPE_BOOL,
    NPIE_DTYPE_STRING
} npie_dtype_t;

/**
 * @brief Tensor shape information
 */
typedef struct {
    uint32_t rank;                  ///< Number of dimensions
    uint32_t dims[8];               ///< Dimension sizes (max 8 dimensions)
} npie_shape_t;

/**
 * @brief Tensor descriptor
 */
typedef struct {
    const char* name;               ///< Tensor name
    npie_dtype_t dtype;             ///< Data type
    npie_shape_t shape;             ///< Shape information
    void* data;                     ///< Pointer to data
    size_t size;                    ///< Size in bytes
} npie_tensor_t;

/**
 * @brief Model metadata
 */
typedef struct {
    char name[256];                 ///< Model name
    char path[512];                 ///< Model file path (if loaded from file)
    const char* version;            ///< Model version (optional)
    const char* description;        ///< Model description (optional)
    const char* author;             ///< Model author (optional)
    uint32_t input_count;           ///< Number of inputs
    uint32_t output_count;          ///< Number of outputs
    size_t model_size;              ///< Model size in bytes
    bool quantized;                 ///< Whether model is quantized (if known)
    npie_backend_t backend;         ///< Backend type
    npie_accelerator_t accelerator; ///< Accelerator type
} npie_model_info_t;

/**
 * @brief Inference options
 */
typedef struct {
    npie_backend_t backend;         ///< Preferred backend
    npie_accelerator_t accelerator; ///< Preferred accelerator
    uint32_t num_threads;           ///< Number of CPU threads (0 = auto)
    uint32_t timeout_ms;            ///< Timeout in milliseconds (0 = no timeout)
    bool enable_profiling;          ///< Enable performance profiling
    bool enable_caching;            ///< Enable model caching
    void* user_data;                ///< User-defined data
} npie_options_t;

/**
 * @brief Performance metrics
 */
typedef struct {
    uint64_t inference_time_us;     ///< Inference time in microseconds
    uint64_t preprocessing_time_us; ///< Preprocessing time in microseconds
    uint64_t postprocessing_time_us;///< Postprocessing time in microseconds
    uint64_t total_time_us;         ///< Total time in microseconds
    size_t memory_used_bytes;       ///< Memory used in bytes
    float cpu_usage_percent;        ///< CPU usage percentage
    float accelerator_usage_percent;///< Accelerator usage percentage
} npie_metrics_t;

/**
 * @brief Opaque handle to NPIE context
 */
typedef struct npie_context* npie_context_t;

/**
 * @brief Opaque handle to loaded model
 */
typedef struct npie_model* npie_model_t;

/**
 * @brief Callback function for logging
 */
typedef void (*npie_log_callback_t)(int level, const char* message, void* user_data);

/**
 * @brief Generic completion callback used by async inference and scheduler
 */
typedef void (*npie_callback_t)(npie_status_t status, const npie_metrics_t* metrics, void* user_data);

/**
 * @brief Scheduler statistics structure
 */
typedef struct {
    uint64_t tasks_submitted;
    uint64_t tasks_completed;
    uint64_t tasks_failed;
    uint64_t tasks_pending;
    uint32_t num_workers;
    uint64_t avg_inference_time_us;
} npie_scheduler_stats_t;

/**
 * @brief Memory manager statistics
 */
typedef struct {
    size_t total_size;
    size_t used_size;
    size_t free_size;
    bool use_hugepages;
    uint32_t num_blocks;
    uint32_t num_free_blocks;
} npie_memory_stats_t;


/**

 * @brief Callback function for progress updates
 */
typedef void (*npie_progress_callback_t)(float progress, void* user_data);

/*
 * ============================================================================
 * Core API Functions
 * ============================================================================
 */

/**
 * @brief Get NPIE version string
 * @return Version string
 */
const char* npie_version(void);

/**
 * @brief Get error message for status code
 * @param status Status code
 * @return Error message string
 */
const char* npie_status_string(npie_status_t status);

/**
 * @brief Initialize NPIE context
 * @param ctx Pointer to context handle
 * @param options Initialization options (NULL for defaults)
 * @return Status code
 */
npie_status_t npie_init(npie_context_t* ctx, const npie_options_t* options);

/**
 * @brief Shutdown NPIE context and free resources
 * @param ctx Context handle
 * @return Status code
 */
npie_status_t npie_shutdown(npie_context_t ctx);

/**
 * @brief Set logging callback
 * @param ctx Context handle
 * @param callback Logging callback function
 * @param user_data User data passed to callback
 * @return Status code
 */
npie_status_t npie_set_log_callback(npie_context_t ctx,
                                    npie_log_callback_t callback,
                                    void* user_data);

/*
 * ============================================================================
 * Model Management API
 * ============================================================================
 */

/**
 * @brief Load model from file
 * @param ctx Context handle
 * @param model Pointer to model handle
 * @param path Path to model file
 * @param options Load options (NULL for defaults)
 * @return Status code
 */
npie_status_t npie_model_load(npie_context_t ctx,
                              npie_model_t* model,
                              const char* path,
                              const npie_options_t* options);

/**
 * @brief Load model from memory buffer
 * @param ctx Context handle
 * @param model Pointer to model handle
 * @param buffer Model data buffer
 * @param size Buffer size in bytes
 * @param options Load options (NULL for defaults)
 * @return Status code
 */
npie_status_t npie_model_load_from_buffer(npie_context_t ctx,
                                          npie_model_t* model,
                                          const void* buffer,
                                          size_t size,
                                          const npie_options_t* options);

/**
 * @brief Unload model and free resources
 * @param model Model handle
 * @return Status code
 */
npie_status_t npie_model_unload(npie_model_t model);

/**
 * @brief Get model information
 * @param model Model handle
 * @param info Pointer to model info structure
 * @return Status code
 */
npie_status_t npie_model_get_info(npie_model_t model, npie_model_info_t* info);

/**
 * @brief Get input tensor information
 * @param model Model handle
 * @param index Input tensor index
 * @param tensor Pointer to tensor descriptor
 * @return Status code
 */
npie_status_t npie_model_get_input(npie_model_t model,
                                   uint32_t index,
                                   npie_tensor_t* tensor);

/**
 * @brief Get output tensor information
 * @param model Model handle
 * @param index Output tensor index
 * @param tensor Pointer to tensor descriptor
 * @return Status code
 */
npie_status_t npie_model_get_output(npie_model_t model,
                                    uint32_t index,
                                    npie_tensor_t* tensor);

/*
 * ============================================================================
 * Inference API
 * ============================================================================
 */

/**
 * @brief Run inference on model
 * @param model Model handle
 * @param inputs Array of input tensors
 * @param num_inputs Number of input tensors
 * @param outputs Array of output tensors (allocated by caller)
 * @param num_outputs Number of output tensors
 * @param metrics Performance metrics (NULL if not needed)
 * @return Status code
 */
npie_status_t npie_inference_run(npie_model_t model,
                                 const npie_tensor_t* inputs,
                                 uint32_t num_inputs,
                                 npie_tensor_t* outputs,
                                 uint32_t num_outputs,
                                 npie_metrics_t* metrics);

/**
 * @brief Run inference asynchronously
 * @param model Model handle
 * @param inputs Array of input tensors
 * @param num_inputs Number of input tensors
 * @param callback Completion callback
 * @param user_data User data passed to callback
 * @return Status code
 */
npie_status_t npie_inference_run_async(npie_model_t model,
                                       const npie_tensor_t* inputs,
                                       uint32_t num_inputs,
                                       npie_tensor_t* outputs,
                                       uint32_t num_outputs,
                                       npie_callback_t callback,
                                       void* user_data);

/*
 * ============================================================================
 * Hardware Detection API
 * ============================================================================
 */

/**
 * @brief Detect available hardware accelerators
 * @param ctx Context handle
 * @param accelerators Array to store detected accelerators
 * @param max_count Maximum number of accelerators to detect
 * @param count Pointer to store actual count
 * @return Status code
 */
npie_status_t npie_detect_accelerators(npie_context_t ctx,
                                       npie_accelerator_t* accelerators,
                                       uint32_t max_count,
                                       uint32_t* count);

/**
 * @brief Check if accelerator is available
 * @param ctx Context handle
 * @param accelerator Accelerator type to check
 * @return true if available, false otherwise
 */
bool npie_is_accelerator_available(npie_context_t ctx, npie_accelerator_t accelerator);

/*
 * ============================================================================
 * Utility Functions
 * ============================================================================
 */

/**
 * @brief Calculate tensor size in bytes
 * @param tensor Tensor descriptor
 * @return Size in bytes
 */
size_t npie_tensor_size(const npie_tensor_t* tensor);

/**
 * @brief Allocate tensor data
 * @param tensor Tensor descriptor
 * @return Status code
 */
npie_status_t npie_tensor_alloc(npie_tensor_t* tensor);

/**
 * @brief Free tensor data
 * @param tensor Tensor descriptor
 * @return Status code
 */
npie_status_t npie_tensor_free(npie_tensor_t* tensor);

#ifdef __cplusplus
}
#endif

#endif /* NPIE_H */

