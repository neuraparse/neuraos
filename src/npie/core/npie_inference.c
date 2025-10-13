/**
 * @file npie_inference.c
 * @brief NPIE Inference Engine Implementation
 * @version 1.0.0-alpha
 */

#include "npie.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#include <pthread.h>

/**
 * @brief Get current time in microseconds
 */
static uint64_t get_time_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000 + tv.tv_usec;
}

/**
 * @brief Validate input tensors
 */
static npie_status_t validate_inputs(npie_model_t model,
                                     const npie_tensor_t* inputs,
                                     uint32_t num_inputs) {
    if (!model || !inputs) {
        return NPIE_ERROR_INVALID_ARGUMENT;
    }

    npie_model_info_t info;
    npie_model_get_info(model, &info);

    if (num_inputs != info.input_count) {
        return NPIE_ERROR_INVALID_ARGUMENT;
    }

    /* Validate each input tensor */
    for (uint32_t i = 0; i < num_inputs; i++) {
        if (!inputs[i].data) {
            return NPIE_ERROR_INVALID_ARGUMENT;
        }

        /* Check tensor size */
        size_t expected_size = npie_tensor_size(&inputs[i]);
        if (inputs[i].size != expected_size) {
            return NPIE_ERROR_INVALID_ARGUMENT;
        }
    }

    return NPIE_SUCCESS;
}

/**
 * @brief Run inference on model
 */
npie_status_t npie_inference_run(npie_model_t model,
                                 const npie_tensor_t* inputs,
                                 uint32_t num_inputs,
                                 npie_tensor_t* outputs,
                                 uint32_t num_outputs,
                                 npie_metrics_t* metrics) {
    if (!model || !inputs || !outputs) {
        return NPIE_ERROR_INVALID_ARGUMENT;
    }

    uint64_t start_time = get_time_us();
    uint64_t preprocess_start = start_time;

    /* Validate inputs */
    npie_status_t status = validate_inputs(model, inputs, num_inputs);
    if (status != NPIE_SUCCESS) {
        return status;
    }

    uint64_t preprocess_end = get_time_us();
    uint64_t inference_start = preprocess_end;

    /* Get model info */
    npie_model_info_t info;
    npie_model_get_info(model, &info);

    /* Run backend-specific inference */
    switch (info.backend) {
#ifdef NEURAOS_ENABLE_LITERT
        case NPIE_BACKEND_LITERT:
            status = npie_backend_litert_inference(model, inputs, num_inputs,
                                                   outputs, num_outputs);
            break;
#endif

#ifdef NEURAOS_ENABLE_ONNXRUNTIME
        case NPIE_BACKEND_ONNXRUNTIME:
            status = npie_backend_onnx_inference(model, inputs, num_inputs,
                                                 outputs, num_outputs);
            break;
#endif

#ifdef NEURAOS_ENABLE_EMLEARN
        case NPIE_BACKEND_EMLEARN:
            status = npie_backend_emlearn_inference(model, inputs, num_inputs,
                                                    outputs, num_outputs);
            break;
#endif

        default:
            status = NPIE_ERROR_UNSUPPORTED_OPERATION;
            break;
    }

    uint64_t inference_end = get_time_us();
    uint64_t postprocess_start = inference_end;

    /* Post-processing (if needed) */
    /* ... */

    uint64_t postprocess_end = get_time_us();
    uint64_t end_time = postprocess_end;

    /* Fill metrics if requested */
    if (metrics) {
        metrics->preprocessing_time_us = preprocess_end - preprocess_start;
        metrics->inference_time_us = inference_end - inference_start;
        metrics->postprocessing_time_us = postprocess_end - postprocess_start;
        metrics->total_time_us = end_time - start_time;

        /* Get memory usage (simplified) */
        metrics->memory_used_bytes = 0;
        for (uint32_t i = 0; i < num_inputs; i++) {
            metrics->memory_used_bytes += inputs[i].size;
        }
        for (uint32_t i = 0; i < num_outputs; i++) {
            metrics->memory_used_bytes += outputs[i].size;
        }

        /* CPU/accelerator usage (would need platform-specific code) */
        metrics->cpu_usage_percent = 0.0f;
        metrics->accelerator_usage_percent = 0.0f;
    }

    return status;
}

/**
 * @brief Run inference asynchronously
 */
npie_status_t npie_inference_run_async(npie_model_t model,
                                       const npie_tensor_t* inputs,
                                       uint32_t num_inputs,
                                       npie_tensor_t* outputs,
                                       uint32_t num_outputs,
                                       npie_callback_t callback,
                                       void* user_data) {
    if (!model || !inputs || !outputs || !callback) {
        return NPIE_ERROR_INVALID_ARGUMENT;
    }

    /* Create async context */
    struct async_context {
        npie_model_t model;
        const npie_tensor_t* inputs;
        uint32_t num_inputs;
        npie_tensor_t* outputs;
        uint32_t num_outputs;
        npie_callback_t callback;
        void* user_data;
    };

    struct async_context* ctx = malloc(sizeof(struct async_context));
    if (!ctx) {
        return NPIE_ERROR_OUT_OF_MEMORY;
    }

    ctx->model = model;
    ctx->inputs = inputs;
    ctx->num_inputs = num_inputs;
    ctx->outputs = outputs;
    ctx->num_outputs = num_outputs;
    ctx->callback = callback;
    ctx->user_data = user_data;

    /* Create thread for async execution */
    pthread_t thread;
    pthread_create(&thread, NULL, async_inference_thread, ctx);
    pthread_detach(thread);

    return NPIE_SUCCESS;
}

/**
 * @brief Async inference thread function
 */
static void* async_inference_thread(void* arg) {
    struct async_context* ctx = (struct async_context*)arg;

    npie_metrics_t metrics;
    npie_status_t status = npie_inference_run(
        ctx->model,
        ctx->inputs,
        ctx->num_inputs,
        ctx->outputs,
        ctx->num_outputs,
        &metrics
    );

    /* Call user callback */
    ctx->callback(status, &metrics, ctx->user_data);

    free(ctx);
    return NULL;
}

/**
 * @brief Batch inference
 */
npie_status_t npie_inference_run_batch(npie_model_t model,
                                       const npie_tensor_t* inputs,
                                       uint32_t batch_size,
                                       npie_tensor_t* outputs,
                                       npie_metrics_t* metrics) {
    if (!model || !inputs || !outputs || batch_size == 0) {
        return NPIE_ERROR_INVALID_ARGUMENT;
    }

    uint64_t start_time = get_time_us();

    /* Get model info */
    npie_model_info_t info;
    npie_model_get_info(model, &info);

    /* Process each item in batch */
    for (uint32_t i = 0; i < batch_size; i++) {
        npie_status_t status = npie_inference_run(
            model,
            &inputs[i * info.input_count],
            info.input_count,
            &outputs[i * info.output_count],
            info.output_count,
            NULL
        );

        if (status != NPIE_SUCCESS) {
            return status;
        }
    }

    uint64_t end_time = get_time_us();

    /* Fill metrics */
    if (metrics) {
        metrics->total_time_us = end_time - start_time;
        metrics->inference_time_us = metrics->total_time_us / batch_size;
        metrics->preprocessing_time_us = 0;
        metrics->postprocessing_time_us = 0;
        metrics->memory_used_bytes = 0;
        metrics->cpu_usage_percent = 0.0f;
        metrics->accelerator_usage_percent = 0.0f;
    }

    return NPIE_SUCCESS;
}

/**
 * @brief Calculate tensor size in bytes
 */
size_t npie_tensor_size(const npie_tensor_t* tensor) {
    if (!tensor) {
        return 0;
    }

    /* Calculate total elements */
    size_t elements = 1;
    for (uint32_t i = 0; i < tensor->shape.rank; i++) {
        elements *= tensor->shape.dims[i];
    }

    /* Get element size based on dtype */
    size_t element_size = 0;
    switch (tensor->dtype) {
        case NPIE_DTYPE_FLOAT32:
            element_size = 4;
            break;
        case NPIE_DTYPE_FLOAT16:
            element_size = 2;
            break;
        case NPIE_DTYPE_INT32:
            element_size = 4;
            break;
        case NPIE_DTYPE_INT16:
            element_size = 2;
            break;
        case NPIE_DTYPE_INT8:
        case NPIE_DTYPE_UINT8:
            element_size = 1;
            break;
        case NPIE_DTYPE_INT64:
            element_size = 8;
            break;
        case NPIE_DTYPE_BOOL:
            element_size = 1;
            break;
        default:
            element_size = 0;
            break;
    }

    return elements * element_size;
}

/**
 * @brief Allocate tensor data
 */
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

/**
 * @brief Free tensor data
 */
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

