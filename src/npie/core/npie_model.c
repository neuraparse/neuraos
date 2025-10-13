/**
 * @file npie_model.c
 * @brief NPIE Model Management Implementation
 * @version 1.0.0-alpha
 */

#include "npie.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <pthread.h>

/**
 * @brief Internal model structure
 */
struct npie_model {
    char name[256];
    char path[512];
    npie_backend_t backend;
    npie_accelerator_t accelerator;
    void* backend_handle;
    npie_context_t context;

    /* Model metadata */
    uint32_t input_count;
    uint32_t output_count;
    npie_tensor_t* inputs;
    npie_tensor_t* outputs;

    /* Model data */
    void* model_data;
    size_t model_size;

    bool loaded;
    pthread_mutex_t mutex;
};

/**
 * @brief Detect model format from file extension
 */
static npie_backend_t detect_model_format(const char* path) {
    const char* ext = strrchr(path, '.');
    if (!ext) {
        return NPIE_BACKEND_AUTO;
    }

    if (strcmp(ext, ".tflite") == 0) {
        return NPIE_BACKEND_LITERT;
    } else if (strcmp(ext, ".onnx") == 0) {
        return NPIE_BACKEND_ONNXRUNTIME;
    } else if (strcmp(ext, ".wasm") == 0) {
        return NPIE_BACKEND_WASMEDGE;
    } else if (strcmp(ext, ".pkl") == 0 || strcmp(ext, ".json") == 0) {
        return NPIE_BACKEND_EMLEARN;
    }

    return NPIE_BACKEND_AUTO;
}

/**
 * @brief Load model file into memory
 */
static npie_status_t load_model_file(const char* path, void** data, size_t* size) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        return NPIE_ERROR_IO;
    }

    /* Get file size */
    fseek(fp, 0, SEEK_END);
    *size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    /* Allocate buffer */
    *data = malloc(*size);
    if (!*data) {
        fclose(fp);
        return NPIE_ERROR_OUT_OF_MEMORY;
    }

    /* Read file */
    size_t read = fread(*data, 1, *size, fp);
    fclose(fp);

    if (read != *size) {
        free(*data);
        *data = NULL;
        return NPIE_ERROR_IO;
    }

    return NPIE_SUCCESS;
}

/**
 * @brief Load model from file
 */
npie_status_t npie_model_load(npie_context_t ctx,
                              npie_model_t* model,
                              const char* path,
                              const npie_options_t* options) {
    if (!ctx || !model || !path) {
        return NPIE_ERROR_INVALID_ARGUMENT;
    }

    /* Check if file exists */
    if (access(path, R_OK) != 0) {
        return NPIE_ERROR_IO;
    }

    /* Allocate model structure */
    struct npie_model* m = (struct npie_model*)calloc(1, sizeof(struct npie_model));
    if (!m) {
        return NPIE_ERROR_OUT_OF_MEMORY;
    }

    /* Initialize model */
    strncpy(m->path, path, sizeof(m->path) - 1);

    /* Extract model name from path */
    const char* filename = strrchr(path, '/');
    if (filename) {
        filename++;
    } else {
        filename = path;
    }
    strncpy(m->name, filename, sizeof(m->name) - 1);

    /* Detect backend */
    m->backend = detect_model_format(path);
    if (options && options->backend != NPIE_BACKEND_AUTO) {
        m->backend = options->backend;
    }

    /* Set accelerator */
    m->accelerator = NPIE_ACCELERATOR_NONE;
    if (options && options->accelerator != NPIE_ACCELERATOR_AUTO) {
        m->accelerator = options->accelerator;
    }

    /* Load model file */
    npie_status_t status = load_model_file(path, &m->model_data, &m->model_size);
    if (status != NPIE_SUCCESS) {
        free(m);
        return status;
    }

    /* Initialize backend */
    switch (m->backend) {
#ifdef NEURAOS_ENABLE_LITERT
        case NPIE_BACKEND_LITERT:
            status = npie_backend_litert_load(m);
            break;
#endif
#ifdef NEURAOS_ENABLE_ONNXRUNTIME
        case NPIE_BACKEND_ONNXRUNTIME:
            status = npie_backend_onnx_load(m);
            break;
#endif
#ifdef NEURAOS_ENABLE_EMLEARN
        case NPIE_BACKEND_EMLEARN:
            status = npie_backend_emlearn_load(m);
            break;
#endif
        default:
            status = NPIE_ERROR_UNSUPPORTED_OPERATION;
            break;
    }

    if (status != NPIE_SUCCESS) {
        free(m->model_data);
        free(m);
        return status;
    }

    /* Initialize mutex */
    pthread_mutex_init(&m->mutex, NULL);

    m->context = ctx;
    m->loaded = true;
    *model = m;

    return NPIE_SUCCESS;
}

/**
 * @brief Load model from memory buffer
 */
npie_status_t npie_model_load_from_buffer(npie_context_t ctx,
                                          npie_model_t* model,
                                          const void* buffer,
                                          size_t size,
                                          const npie_options_t* options) {
    if (!ctx || !model || !buffer || size == 0) {
        return NPIE_ERROR_INVALID_ARGUMENT;
    }

    /* Allocate model structure */
    struct npie_model* m = (struct npie_model*)calloc(1, sizeof(struct npie_model));
    if (!m) {
        return NPIE_ERROR_OUT_OF_MEMORY;
    }

    /* Copy model data */
    m->model_data = malloc(size);
    if (!m->model_data) {
        free(m);
        return NPIE_ERROR_OUT_OF_MEMORY;
    }
    memcpy(m->model_data, buffer, size);
    m->model_size = size;

    strcpy(m->name, "memory_model");

    /* Set backend and accelerator from options */
    if (options) {
        m->backend = options->backend;
        m->accelerator = options->accelerator;
    }

    /* Initialize backend (similar to npie_model_load) */
    npie_status_t status = NPIE_SUCCESS;
    /* ... backend initialization ... */

    if (status != NPIE_SUCCESS) {
        free(m->model_data);
        free(m);
        return status;
    }

    pthread_mutex_init(&m->mutex, NULL);
    m->context = ctx;
    m->loaded = true;
    *model = m;

    return NPIE_SUCCESS;
}

/**
 * @brief Unload model and free resources
 */
npie_status_t npie_model_unload(npie_model_t model) {
    if (!model) {
        return NPIE_ERROR_INVALID_ARGUMENT;
    }

    struct npie_model* m = (struct npie_model*)model;

    pthread_mutex_lock(&m->mutex);

    if (!m->loaded) {
        pthread_mutex_unlock(&m->mutex);
        return NPIE_ERROR_NOT_INITIALIZED;
    }

    /* Unload backend */
    if (m->backend_handle) {
        /* Backend-specific cleanup */
        switch (m->backend) {
#ifdef NEURAOS_ENABLE_LITERT
            case NPIE_BACKEND_LITERT:
                npie_backend_litert_unload(m);
                break;
#endif
#ifdef NEURAOS_ENABLE_ONNXRUNTIME
            case NPIE_BACKEND_ONNXRUNTIME:
                npie_backend_onnx_unload(m);
                break;
#endif
            default:
                break;
        }
    }

    /* Free tensors */
    if (m->inputs) {
        for (uint32_t i = 0; i < m->input_count; i++) {
            npie_tensor_free(&m->inputs[i]);
        }
        free(m->inputs);
    }

    if (m->outputs) {
        for (uint32_t i = 0; i < m->output_count; i++) {
            npie_tensor_free(&m->outputs[i]);
        }
        free(m->outputs);
    }

    /* Free model data */
    if (m->model_data) {
        free(m->model_data);
    }

    m->loaded = false;

    pthread_mutex_unlock(&m->mutex);
    pthread_mutex_destroy(&m->mutex);

    free(m);

    return NPIE_SUCCESS;
}

/**
 * @brief Get model information
 */
npie_status_t npie_model_get_info(npie_model_t model, npie_model_info_t* info) {
    if (!model || !info) {
        return NPIE_ERROR_INVALID_ARGUMENT;
    }

    struct npie_model* m = (struct npie_model*)model;

    pthread_mutex_lock(&m->mutex);

    strncpy(info->name, m->name, sizeof(info->name) - 1);
    strncpy(info->path, m->path, sizeof(info->path) - 1);
    info->backend = m->backend;
    info->accelerator = m->accelerator;
    info->input_count = m->input_count;
    info->output_count = m->output_count;
    info->model_size = m->model_size;

    pthread_mutex_unlock(&m->mutex);

    return NPIE_SUCCESS;
}

/**
 * @brief Get input tensor descriptor
 */
npie_status_t npie_model_get_input(npie_model_t model, uint32_t index, npie_tensor_t* tensor) {
    if (!model || !tensor) {
        return NPIE_ERROR_INVALID_ARGUMENT;
    }

    struct npie_model* m = (struct npie_model*)model;

    if (index >= m->input_count) {
        return NPIE_ERROR_INVALID_ARGUMENT;
    }

    pthread_mutex_lock(&m->mutex);
    memcpy(tensor, &m->inputs[index], sizeof(npie_tensor_t));
    pthread_mutex_unlock(&m->mutex);

    return NPIE_SUCCESS;
}

/**
 * @brief Get output tensor descriptor
 */
npie_status_t npie_model_get_output(npie_model_t model, uint32_t index, npie_tensor_t* tensor) {
    if (!model || !tensor) {
        return NPIE_ERROR_INVALID_ARGUMENT;
    }

    struct npie_model* m = (struct npie_model*)model;

    if (index >= m->output_count) {
        return NPIE_ERROR_INVALID_ARGUMENT;
    }

    pthread_mutex_lock(&m->mutex);
    memcpy(tensor, &m->outputs[index], sizeof(npie_tensor_t));
    pthread_mutex_unlock(&m->mutex);

    return NPIE_SUCCESS;
}

