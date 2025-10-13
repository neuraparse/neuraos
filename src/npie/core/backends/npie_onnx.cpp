/**
 * @file npie_onnx.cpp
 * @brief ONNX Runtime Backend Implementation
 * @version 1.0.0-alpha
 */

#include "npie.h"
#include "npie_internal.h"


#ifdef NEURAOS_ENABLE_ONNXRUNTIME

#include <onnxruntime_cxx_api.h>
#include <memory>
#include <vector>

extern "C" {

/**
 * @brief ONNX Runtime backend context
 */
struct onnx_context {
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::SessionOptions> session_options;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<std::vector<int64_t>> output_shapes;
};

/**
 * @brief Convert ONNX data type to NPIE data type
 */
static npie_dtype_t onnx_to_npie_dtype(ONNXTensorElementDataType onnx_type) {
    switch (onnx_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return NPIE_DTYPE_FLOAT32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            return NPIE_DTYPE_FLOAT16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            return NPIE_DTYPE_INT32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            return NPIE_DTYPE_INT8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            return NPIE_DTYPE_UINT8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return NPIE_DTYPE_INT64;
        default:
            return NPIE_DTYPE_FLOAT32;
    }
}

/**
 * @brief Load model with ONNX Runtime backend
 */
npie_status_t npie_backend_onnx_load(npie_model_t model) {
    if (!model) {
        return NPIE_ERROR_INVALID_ARGUMENT;
    }

    struct npie_model* m = (struct npie_model*)model;

    try {
        /* Create ONNX context */
        onnx_context* ctx = new onnx_context();

        /* Initialize ONNX Runtime environment */
        ctx->env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "NeuralOS");

        /* Create session options */
        ctx->session_options = std::make_unique<Ort::SessionOptions>();

        /* Set number of threads */
        if (m->context && m->context->options.num_threads > 0) {
            ctx->session_options->SetIntraOpNumThreads(m->context->options.num_threads);
        }

        /* Enable optimizations */
        ctx->session_options->SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL
        );

        /* Enable execution providers based on accelerator */
        if (m->accelerator == NPIE_ACCELERATOR_GPU) {
            // Try CUDA provider
            // OrtCUDAProviderOptions cuda_options;
            // ctx->session_options->AppendExecutionProvider_CUDA(cuda_options);
        }

        /* Create session from memory buffer */
        ctx->session = std::make_unique<Ort::Session>(
            *ctx->env,
            m->model_data,
            m->model_size,
            *ctx->session_options
        );

        /* Get input info */
        size_t num_inputs = ctx->session->GetInputCount();
        m->input_count = num_inputs;

        m->inputs = (npie_tensor_t*)calloc(num_inputs, sizeof(npie_tensor_t));
        if (!m->inputs) {
            delete ctx;
            return NPIE_ERROR_OUT_OF_MEMORY;
        }

        for (size_t i = 0; i < num_inputs; i++) {
            /* Get input name */
            char* name = ctx->session->GetInputName(i, ctx->allocator);
            ctx->input_names.push_back(name);
            m->inputs[i].name = name;

            /* Get input type info */
            Ort::TypeInfo type_info = ctx->session->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

            /* Get data type */
            m->inputs[i].dtype = onnx_to_npie_dtype(tensor_info.GetElementType());

            /* Get shape */
            std::vector<int64_t> shape = tensor_info.GetShape();
            ctx->input_shapes.push_back(shape);

            m->inputs[i].shape.rank = shape.size();
            for (size_t j = 0; j < shape.size() && j < 8; j++) {
                m->inputs[i].shape.dims[j] = shape[j] > 0 ? shape[j] : 1;
            }

            m->inputs[i].data = nullptr;
            m->inputs[i].size = tensor_info.GetElementCount() *
                               (m->inputs[i].dtype == NPIE_DTYPE_FLOAT32 ? 4 : 1);
        }

        /* Get output info */
        size_t num_outputs = ctx->session->GetOutputCount();
        m->output_count = num_outputs;

        m->outputs = (npie_tensor_t*)calloc(num_outputs, sizeof(npie_tensor_t));
        if (!m->outputs) {
            free(m->inputs);
            delete ctx;
            return NPIE_ERROR_OUT_OF_MEMORY;
        }

        for (size_t i = 0; i < num_outputs; i++) {
            /* Get output name */
            char* name = ctx->session->GetOutputName(i, ctx->allocator);
            ctx->output_names.push_back(name);
            m->outputs[i].name = name;

            /* Get output type info */
            Ort::TypeInfo type_info = ctx->session->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

            /* Get data type */
            m->outputs[i].dtype = onnx_to_npie_dtype(tensor_info.GetElementType());

            /* Get shape */
            std::vector<int64_t> shape = tensor_info.GetShape();
            ctx->output_shapes.push_back(shape);

            m->outputs[i].shape.rank = shape.size();
            for (size_t j = 0; j < shape.size() && j < 8; j++) {
                m->outputs[i].shape.dims[j] = shape[j] > 0 ? shape[j] : 1;
            }

            m->outputs[i].data = nullptr;
            m->outputs[i].size = tensor_info.GetElementCount() *
                                (m->outputs[i].dtype == NPIE_DTYPE_FLOAT32 ? 4 : 1);
        }

        m->backend_handle = ctx;

        return NPIE_SUCCESS;

    } catch (const Ort::Exception& e) {
        return NPIE_ERROR_MODEL_LOAD_FAILED;
    } catch (...) {
        return NPIE_ERROR_UNKNOWN;
    }
}

/**
 * @brief Unload ONNX model
 */
npie_status_t npie_backend_onnx_unload(npie_model_t model) {
    if (!model) {
        return NPIE_ERROR_INVALID_ARGUMENT;
    }

    struct npie_model* m = (struct npie_model*)model;

    if (m->backend_handle) {
        onnx_context* ctx = (onnx_context*)m->backend_handle;
        delete ctx;
        m->backend_handle = nullptr;
    }

    return NPIE_SUCCESS;
}

/**
 * @brief Run inference with ONNX Runtime
 */
npie_status_t npie_backend_onnx_inference(npie_model_t model,
                                          const npie_tensor_t* inputs,
                                          uint32_t num_inputs,
                                          npie_tensor_t* outputs,
                                          uint32_t num_outputs) {
    if (!model || !inputs || !outputs) {
        return NPIE_ERROR_INVALID_ARGUMENT;
    }

    struct npie_model* m = (struct npie_model*)model;
    onnx_context* ctx = (onnx_context*)m->backend_handle;

    if (!ctx || !ctx->session) {
        return NPIE_ERROR_NOT_INITIALIZED;
    }

    try {
        /* Create input tensors */
        std::vector<Ort::Value> input_tensors;

        for (uint32_t i = 0; i < num_inputs; i++) {
            auto memory_info = Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator, OrtMemTypeDefault
            );

            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                memory_info,
                (float*)inputs[i].data,
                inputs[i].size / sizeof(float),
                ctx->input_shapes[i].data(),
                ctx->input_shapes[i].size()
            ));
        }

        /* Run inference */
        auto output_tensors = ctx->session->Run(
            Ort::RunOptions{nullptr},
            ctx->input_names.data(),
            input_tensors.data(),
            num_inputs,
            ctx->output_names.data(),
            num_outputs
        );

        /* Copy output data */
        for (uint32_t i = 0; i < num_outputs; i++) {
            float* output_data = output_tensors[i].GetTensorMutableData<float>();
            size_t output_size = outputs[i].size;

            if (!outputs[i].data) {
                outputs[i].data = malloc(output_size);
                if (!outputs[i].data) {
                    return NPIE_ERROR_OUT_OF_MEMORY;
                }
            }

            memcpy(outputs[i].data, output_data, output_size);
        }

        return NPIE_SUCCESS;

    } catch (const Ort::Exception& e) {
        return NPIE_ERROR_INFERENCE_FAILED;
    } catch (...) {
        return NPIE_ERROR_UNKNOWN;
    }
}

} /* extern "C" */

#endif /* NEURAOS_ENABLE_ONNXRUNTIME */

