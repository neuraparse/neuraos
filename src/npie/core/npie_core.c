/**
 * @file npie_core.c
 * @brief NeuraParse Inference Engine Core Implementation
 * @version 1.0.0-alpha
 * @date October 2025
 */

#include "npie.h"
#include "npie_internal.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>
#include <time.h>

/**
 * @brief Internal context structure
 */
struct npie_context {
    npie_options_t options;
    npie_log_callback_t log_callback;
    void* log_user_data;
    pthread_mutex_t mutex;
    bool initialized;
    uint32_t model_count;
    npie_model_t* models;
    
    /* Hardware detection cache */
    bool accelerators_detected;
    npie_accelerator_t available_accelerators[16];
    uint32_t accelerator_count;
};

/**
 * @brief Internal model structure
 */
struct npie_model {
    npie_context_t ctx;
    npie_backend_t backend;
    npie_accelerator_t accelerator;
    npie_model_info_t info;
    void* backend_handle;
    npie_tensor_t* inputs;
    npie_tensor_t* outputs;
    pthread_mutex_t mutex;
    bool loaded;
};

/**
 * @brief Default options
 */
static const npie_options_t default_options = {
    .backend = NPIE_BACKEND_AUTO,
    .accelerator = NPIE_ACCELERATOR_AUTO,
    .num_threads = 0,
    .timeout_ms = 0,
    .enable_profiling = false,
    .enable_caching = true,
    .user_data = NULL
};

/**
 * @brief Internal logging function
 */
static void npie_log(npie_context_t ctx, int level, const char* format, ...) {
    if (!ctx || !ctx->log_callback) {
        return;
    }
    
    char buffer[1024];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    
    ctx->log_callback(level, buffer, ctx->log_user_data);
}

/*
 * ============================================================================
 * Core API Implementation
 * ============================================================================
 */

const char* npie_version(void) {
    return NPIE_VERSION_STRING;
}

const char* npie_status_string(npie_status_t status) {
    switch (status) {
        case NPIE_SUCCESS:
            return "Success";
        case NPIE_ERROR_INVALID_ARGUMENT:
            return "Invalid argument";
        case NPIE_ERROR_OUT_OF_MEMORY:
            return "Out of memory";
        case NPIE_ERROR_MODEL_LOAD_FAILED:
            return "Model load failed";
        case NPIE_ERROR_INFERENCE_FAILED:
            return "Inference failed";
        case NPIE_ERROR_UNSUPPORTED_OPERATION:
            return "Unsupported operation";
        case NPIE_ERROR_HARDWARE_NOT_AVAILABLE:
            return "Hardware not available";
        case NPIE_ERROR_TIMEOUT:
            return "Timeout";
        case NPIE_ERROR_NOT_INITIALIZED:
            return "Not initialized";
        case NPIE_ERROR_ALREADY_INITIALIZED:
            return "Already initialized";
        case NPIE_ERROR_IO:
            return "I/O error";
        default:
            return "Unknown error";
    }
}

npie_status_t npie_init(npie_context_t* ctx, const npie_options_t* options) {
    if (!ctx) {
        return NPIE_ERROR_INVALID_ARGUMENT;
    }
    
    /* Allocate context */
    npie_context_t new_ctx = (npie_context_t)calloc(1, sizeof(struct npie_context));
    if (!new_ctx) {
        return NPIE_ERROR_OUT_OF_MEMORY;
    }
    
    /* Initialize options */
    if (options) {
        memcpy(&new_ctx->options, options, sizeof(npie_options_t));
    } else {
        memcpy(&new_ctx->options, &default_options, sizeof(npie_options_t));
    }
    
    /* Auto-detect number of threads if not specified */
    if (new_ctx->options.num_threads == 0) {
        new_ctx->options.num_threads = sysconf(_SC_NPROCESSORS_ONLN);
        if (new_ctx->options.num_threads == 0) {
            new_ctx->options.num_threads = 1;
        }
    }
    
    /* Initialize mutex */
    if (pthread_mutex_init(&new_ctx->mutex, NULL) != 0) {
        free(new_ctx);
        return NPIE_ERROR_UNKNOWN;
    }
    
    new_ctx->initialized = true;
    new_ctx->model_count = 0;
    new_ctx->models = NULL;
    new_ctx->accelerators_detected = false;
    new_ctx->accelerator_count = 0;
    
    *ctx = new_ctx;
    
    return NPIE_SUCCESS;
}

npie_status_t npie_shutdown(npie_context_t ctx) {
    if (!ctx) {
        return NPIE_ERROR_INVALID_ARGUMENT;
    }
    
    if (!ctx->initialized) {
        return NPIE_ERROR_NOT_INITIALIZED;
    }
    
    pthread_mutex_lock(&ctx->mutex);
    
    /* Unload all models */
    for (uint32_t i = 0; i < ctx->model_count; i++) {
        if (ctx->models[i]) {
            npie_model_unload(ctx->models[i]);
        }
    }
    
    if (ctx->models) {
        free(ctx->models);
    }
    
    ctx->initialized = false;
    
    pthread_mutex_unlock(&ctx->mutex);
    pthread_mutex_destroy(&ctx->mutex);
    
    free(ctx);
    
    return NPIE_SUCCESS;
}

npie_status_t npie_set_log_callback(npie_context_t ctx,
                                    npie_log_callback_t callback,
                                    void* user_data) {
    if (!ctx) {
        return NPIE_ERROR_INVALID_ARGUMENT;
    }
    
    pthread_mutex_lock(&ctx->mutex);
    ctx->log_callback = callback;
    ctx->log_user_data = user_data;
    pthread_mutex_unlock(&ctx->mutex);
    
    return NPIE_SUCCESS;
}

/*
 * ============================================================================
 * Model Management Implementation
 * ============================================================================
 */

npie_status_t npie_model_load(npie_context_t ctx,
                              npie_model_t* model,
                              const char* path,
                              const npie_options_t* options) {
    if (!ctx || !model || !path) {
        return NPIE_ERROR_INVALID_ARGUMENT;
    }
    
    if (!ctx->initialized) {
        return NPIE_ERROR_NOT_INITIALIZED;
    }
    
    /* Read model file into buffer */
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        npie_log(ctx, 3, "Failed to open model file: %s", path);
        return NPIE_ERROR_IO;
    }
    
    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    void* buffer = malloc(size);
    if (!buffer) {
        fclose(fp);
        return NPIE_ERROR_OUT_OF_MEMORY;
    }
    
    size_t read_size = fread(buffer, 1, size, fp);
    fclose(fp);
    
    if (read_size != size) {
        free(buffer);
        return NPIE_ERROR_IO;
    }
    
    /* Load from buffer */
    npie_status_t status = npie_model_load_from_buffer(ctx, model, buffer, size, options);
    
    free(buffer);
    
    if (status == NPIE_SUCCESS) {
        npie_log(ctx, 1, "Model loaded successfully: %s", path);
    }
    
    return status;
}

npie_status_t npie_model_load_from_buffer(npie_context_t ctx,
                                          npie_model_t* model,
                                          const void* buffer,
                                          size_t size,
                                          const npie_options_t* options) {
    if (!ctx || !model || !buffer || size == 0) {
        return NPIE_ERROR_INVALID_ARGUMENT;
    }
    
    if (!ctx->initialized) {
        return NPIE_ERROR_NOT_INITIALIZED;
    }
    
    /* Allocate model structure */
    npie_model_t new_model = (npie_model_t)calloc(1, sizeof(struct npie_model));
    if (!new_model) {
        return NPIE_ERROR_OUT_OF_MEMORY;
    }
    
    new_model->ctx = ctx;
    
    /* Use provided options or context defaults */
    if (options) {
        new_model->backend = options->backend;
        new_model->accelerator = options->accelerator;
    } else {
        new_model->backend = ctx->options.backend;
        new_model->accelerator = ctx->options.accelerator;
    }
    
    /* Auto-detect backend if needed */
    if (new_model->backend == NPIE_BACKEND_AUTO) {
        /* Try to detect model format */
        /* This is a simplified version - real implementation would check magic numbers */
        #ifdef NEURAOS_ENABLE_LITERT
        new_model->backend = NPIE_BACKEND_LITERT;
        #elif defined(NEURAOS_ENABLE_ONNXRUNTIME)
        new_model->backend = NPIE_BACKEND_ONNXRUNTIME;
        #else
        free(new_model);
        return NPIE_ERROR_UNSUPPORTED_OPERATION;
        #endif
    }
    
    /* Initialize mutex */
    if (pthread_mutex_init(&new_model->mutex, NULL) != 0) {
        free(new_model);
        return NPIE_ERROR_UNKNOWN;
    }
    
    /* Load model using appropriate backend */
    npie_status_t status = NPIE_ERROR_UNSUPPORTED_OPERATION;
    
    switch (new_model->backend) {
        case NPIE_BACKEND_LITERT:
            #ifdef NEURAOS_ENABLE_LITERT
            status = npie_backend_litert_load(new_model, buffer, size);
            #endif
            break;
            
        case NPIE_BACKEND_ONNXRUNTIME:
            #ifdef NEURAOS_ENABLE_ONNXRUNTIME
            status = npie_backend_onnx_load(new_model, buffer, size);
            #endif
            break;
            
        case NPIE_BACKEND_EMLEARN:
            #ifdef NEURAOS_ENABLE_EMLEARN
            status = npie_backend_emlearn_load(new_model, buffer, size);
            #endif
            break;
            
        default:
            status = NPIE_ERROR_UNSUPPORTED_OPERATION;
            break;
    }
    
    if (status != NPIE_SUCCESS) {
        pthread_mutex_destroy(&new_model->mutex);
        free(new_model);
        return status;
    }
    
    new_model->loaded = true;
    
    /* Add to context's model list */
    pthread_mutex_lock(&ctx->mutex);
    ctx->model_count++;
    ctx->models = (npie_model_t*)realloc(ctx->models, 
                                         ctx->model_count * sizeof(npie_model_t));
    ctx->models[ctx->model_count - 1] = new_model;
    pthread_mutex_unlock(&ctx->mutex);
    
    *model = new_model;
    
    return NPIE_SUCCESS;
}

npie_status_t npie_model_unload(npie_model_t model) {
    if (!model) {
        return NPIE_ERROR_INVALID_ARGUMENT;
    }
    
    if (!model->loaded) {
        return NPIE_ERROR_NOT_INITIALIZED;
    }
    
    pthread_mutex_lock(&model->mutex);
    
    /* Unload using appropriate backend */
    switch (model->backend) {
        case NPIE_BACKEND_LITERT:
            #ifdef NEURAOS_ENABLE_LITERT
            npie_backend_litert_unload(model);
            #endif
            break;
            
        case NPIE_BACKEND_ONNXRUNTIME:
            #ifdef NEURAOS_ENABLE_ONNXRUNTIME
            npie_backend_onnx_unload(model);
            #endif
            break;
            
        case NPIE_BACKEND_EMLEARN:
            #ifdef NEURAOS_ENABLE_EMLEARN
            npie_backend_emlearn_unload(model);
            #endif
            break;
            
        default:
            break;
    }
    
    /* Free tensors */
    if (model->inputs) {
        for (uint32_t i = 0; i < model->info.input_count; i++) {
            npie_tensor_free(&model->inputs[i]);
        }
        free(model->inputs);
    }
    
    if (model->outputs) {
        for (uint32_t i = 0; i < model->info.output_count; i++) {
            npie_tensor_free(&model->outputs[i]);
        }
        free(model->outputs);
    }
    
    model->loaded = false;
    
    pthread_mutex_unlock(&model->mutex);
    pthread_mutex_destroy(&model->mutex);
    
    free(model);
    
    return NPIE_SUCCESS;
}

npie_status_t npie_model_get_info(npie_model_t model, npie_model_info_t* info) {
    if (!model || !info) {
        return NPIE_ERROR_INVALID_ARGUMENT;
    }
    
    if (!model->loaded) {
        return NPIE_ERROR_NOT_INITIALIZED;
    }
    
    pthread_mutex_lock(&model->mutex);
    memcpy(info, &model->info, sizeof(npie_model_info_t));
    pthread_mutex_unlock(&model->mutex);
    
    return NPIE_SUCCESS;
}

/*
 * ============================================================================
 * Utility Functions Implementation
 * ============================================================================
 */

size_t npie_tensor_size(const npie_tensor_t* tensor) {
    if (!tensor) {
        return 0;
    }
    
    size_t element_size = 0;
    switch (tensor->dtype) {
        case NPIE_DTYPE_FLOAT32:
        case NPIE_DTYPE_INT32:
            element_size = 4;
            break;
        case NPIE_DTYPE_FLOAT16:
        case NPIE_DTYPE_INT16:
            element_size = 2;
            break;
        case NPIE_DTYPE_INT8:
        case NPIE_DTYPE_UINT8:
        case NPIE_DTYPE_BOOL:
            element_size = 1;
            break;
        case NPIE_DTYPE_INT64:
            element_size = 8;
            break;
        default:
            return 0;
    }
    
    size_t total_elements = 1;
    for (uint32_t i = 0; i < tensor->shape.rank; i++) {
        total_elements *= tensor->shape.dims[i];
    }
    
    return total_elements * element_size;
}

npie_status_t npie_tensor_alloc(npie_tensor_t* tensor) {
    if (!tensor) {
        return NPIE_ERROR_INVALID_ARGUMENT;
    }
    
    size_t size = npie_tensor_size(tensor);
    if (size == 0) {
        return NPIE_ERROR_INVALID_ARGUMENT;
    }
    
    tensor->data = malloc(size);
    if (!tensor->data) {
        return NPIE_ERROR_OUT_OF_MEMORY;
    }
    
    tensor->size = size;
    memset(tensor->data, 0, size);
    
    return NPIE_SUCCESS;
}

npie_status_t npie_tensor_free(npie_tensor_t* tensor) {
    if (!tensor) {
        return NPIE_ERROR_INVALID_ARGUMENT;
    }
    
    if (tensor->data) {
        free(tensor->data);
        tensor->data = NULL;
        tensor->size = 0;
    }
    
    return NPIE_SUCCESS;
}

