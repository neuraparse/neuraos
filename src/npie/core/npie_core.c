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

#include <stdarg.h>
#include <unistd.h>


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
static __attribute__((unused)) void npie_log(npie_context_t ctx, int level, const char* format, ...) {
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






