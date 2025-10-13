/**
 * @file npie_litert.cpp
 * @brief LiteRT (TensorFlow Lite) Backend Implementation
 * @version 1.0.0-alpha
 */

#include "npie.h"
#include "npie_internal.h"


#ifdef NEURAOS_ENABLE_LITERT

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/optional_debug_tools.h>

extern "C" {

/**
 * @brief LiteRT backend context
 */
struct litert_context {
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
};

/**
 * @brief Load model with LiteRT backend
 */
npie_status_t npie_backend_litert_load(npie_model_t model) {
    if (!model) {
        return NPIE_ERROR_INVALID_ARGUMENT;
    }

    struct npie_model* m = (struct npie_model*)model;

    /* Create LiteRT context */
    litert_context* ctx = new litert_context();
    if (!ctx) {
        return NPIE_ERROR_OUT_OF_MEMORY;
    }

    /* Load model from buffer */
    ctx->model = tflite::FlatBufferModel::BuildFromBuffer(
        (const char*)m->model_data,
        m->model_size
    );

    if (!ctx->model) {
        delete ctx;
        return NPIE_ERROR_MODEL_LOAD_FAILED;
    }

    /* Build interpreter */
    tflite::InterpreterBuilder builder(*ctx->model, ctx->resolver);
    builder(&ctx->interpreter);

    if (!ctx->interpreter) {
        delete ctx;
        return NPIE_ERROR_MODEL_LOAD_FAILED;
    }

    /* Set number of threads */
    if (m->context && m->context->options.num_threads > 0) {
        ctx->interpreter->SetNumThreads(m->context->options.num_threads);
    }

    /* Allocate tensors */
    if (ctx->interpreter->AllocateTensors() != kTfLiteOk) {
        delete ctx;
        return NPIE_ERROR_OUT_OF_MEMORY;
    }

    /* Get input/output info */
    m->input_count = ctx->interpreter->inputs().size();
    m->output_count = ctx->interpreter->outputs().size();

    /* Allocate input tensor descriptors */
    m->inputs = (npie_tensor_t*)calloc(m->input_count, sizeof(npie_tensor_t));
    if (!m->inputs) {
        delete ctx;
        return NPIE_ERROR_OUT_OF_MEMORY;
    }

    /* Fill input tensor info */
    for (uint32_t i = 0; i < m->input_count; i++) {
        int input_idx = ctx->interpreter->inputs()[i];
        TfLiteTensor* tensor = ctx->interpreter->tensor(input_idx);

        m->inputs[i].name = tensor->name;

        /* Convert dtype */
        switch (tensor->type) {
            case kTfLiteFloat32:
                m->inputs[i].dtype = NPIE_DTYPE_FLOAT32;
                break;
            case kTfLiteInt8:
                m->inputs[i].dtype = NPIE_DTYPE_INT8;
                break;
            case kTfLiteUInt8:
                m->inputs[i].dtype = NPIE_DTYPE_UINT8;
                break;
            default:
                m->inputs[i].dtype = NPIE_DTYPE_FLOAT32;
                break;
        }

        /* Copy shape */
        m->inputs[i].shape.rank = tensor->dims->size;
        for (int j = 0; j < tensor->dims->size && j < 8; j++) {
            m->inputs[i].shape.dims[j] = tensor->dims->data[j];
        }

        m->inputs[i].data = nullptr;
        m->inputs[i].size = tensor->bytes;
    }

    /* Allocate output tensor descriptors */
    m->outputs = (npie_tensor_t*)calloc(m->output_count, sizeof(npie_tensor_t));
    if (!m->outputs) {
        free(m->inputs);
        delete ctx;
        return NPIE_ERROR_OUT_OF_MEMORY;
    }

    /* Fill output tensor info */
    for (uint32_t i = 0; i < m->output_count; i++) {
        int output_idx = ctx->interpreter->outputs()[i];
        TfLiteTensor* tensor = ctx->interpreter->tensor(output_idx);

        m->outputs[i].name = tensor->name;

        /* Convert dtype */
        switch (tensor->type) {
            case kTfLiteFloat32:
                m->outputs[i].dtype = NPIE_DTYPE_FLOAT32;
                break;
            case kTfLiteInt8:
                m->outputs[i].dtype = NPIE_DTYPE_INT8;
                break;
            case kTfLiteUInt8:
                m->outputs[i].dtype = NPIE_DTYPE_UINT8;
                break;
            default:
                m->outputs[i].dtype = NPIE_DTYPE_FLOAT32;
                break;
        }

        /* Copy shape */
        m->outputs[i].shape.rank = tensor->dims->size;
        for (int j = 0; j < tensor->dims->size && j < 8; j++) {
            m->outputs[i].shape.dims[j] = tensor->dims->data[j];
        }

        m->outputs[i].data = nullptr;
        m->outputs[i].size = tensor->bytes;
    }

    m->backend_handle = ctx;

    return NPIE_SUCCESS;
}

/**
 * @brief Unload LiteRT model
 */
npie_status_t npie_backend_litert_unload(npie_model_t model) {
    if (!model) {
        return NPIE_ERROR_INVALID_ARGUMENT;
    }

    struct npie_model* m = (struct npie_model*)model;

    if (m->backend_handle) {
        litert_context* ctx = (litert_context*)m->backend_handle;
        delete ctx;
        m->backend_handle = nullptr;
    }

    return NPIE_SUCCESS;
}

/**
 * @brief Run inference with LiteRT
 */
npie_status_t npie_backend_litert_inference(npie_model_t model,
                                            const npie_tensor_t* inputs,
                                            uint32_t num_inputs,
                                            npie_tensor_t* outputs,
                                            uint32_t num_outputs) {
    if (!model || !inputs || !outputs) {
        return NPIE_ERROR_INVALID_ARGUMENT;
    }

    struct npie_model* m = (struct npie_model*)model;
    litert_context* ctx = (litert_context*)m->backend_handle;

    if (!ctx || !ctx->interpreter) {
        return NPIE_ERROR_NOT_INITIALIZED;
    }

    /* Copy input data */
    for (uint32_t i = 0; i < num_inputs; i++) {
        int input_idx = ctx->interpreter->inputs()[i];
        TfLiteTensor* tensor = ctx->interpreter->tensor(input_idx);

        if (inputs[i].size != tensor->bytes) {
            return NPIE_ERROR_INVALID_ARGUMENT;
        }

        memcpy(tensor->data.raw, inputs[i].data, inputs[i].size);
    }

    /* Run inference */
    if (ctx->interpreter->Invoke() != kTfLiteOk) {
        return NPIE_ERROR_INFERENCE_FAILED;
    }

    /* Copy output data */
    for (uint32_t i = 0; i < num_outputs; i++) {
        int output_idx = ctx->interpreter->outputs()[i];
        TfLiteTensor* tensor = ctx->interpreter->tensor(output_idx);

        if (!outputs[i].data) {
            outputs[i].data = malloc(tensor->bytes);
            if (!outputs[i].data) {
                return NPIE_ERROR_OUT_OF_MEMORY;
            }
            outputs[i].size = tensor->bytes;
        }

        memcpy(outputs[i].data, tensor->data.raw, tensor->bytes);
    }

    return NPIE_SUCCESS;
}

} /* extern "C" */

#endif /* NEURAOS_ENABLE_LITERT */

