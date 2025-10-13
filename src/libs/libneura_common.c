/**
 * @file libneura_common.c
 * @brief Common utilities implementation for NeuralOS
 * @version 1.0.0-alpha
 * @date October 2025
 */

#include "libneura_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>

/* Global state */
static neura_log_callback_t g_log_callback = NULL;
static void* g_log_user_data = NULL;
static neura_log_level_t g_log_level = NEURA_LOG_INFO;

/**
 * @brief Initialize common library
 */
int neura_common_init(void) {
    return NEURA_OK;
}

/**
 * @brief Cleanup common library
 */
void neura_common_cleanup(void) {
    g_log_callback = NULL;
    g_log_user_data = NULL;
}

/**
 * @brief Get version string
 */
const char* neura_get_version(void) {
    return NEURAOS_VERSION_STRING;
}

/**
 * @brief Set log callback
 */
void neura_set_log_callback(neura_log_callback_t callback, void* user_data) {
    g_log_callback = callback;
    g_log_user_data = user_data;
}

/**
 * @brief Log a message
 */
void neura_log(neura_log_level_t level, const char* format, ...) {
    if (level < g_log_level) {
        return;
    }
    
    char buffer[1024];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    
    if (g_log_callback) {
        g_log_callback(level, buffer, g_log_user_data);
    } else {
        const char* level_str[] = {
            "TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL"
        };
        
        struct timeval tv;
        gettimeofday(&tv, NULL);
        
        fprintf(stderr, "[%ld.%06d] [%s] %s\n",
                (long)tv.tv_sec, tv.tv_usec, level_str[level], buffer);
    }
}

/**
 * @brief Get error string
 */
const char* neura_error_string(neura_error_t error) {
    switch (error) {
        case NEURA_OK: return "Success";
        case NEURA_ERROR_INVALID_PARAM: return "Invalid parameter";
        case NEURA_ERROR_OUT_OF_MEMORY: return "Out of memory";
        case NEURA_ERROR_NOT_FOUND: return "Not found";
        case NEURA_ERROR_IO: return "I/O error";
        case NEURA_ERROR_TIMEOUT: return "Timeout";
        case NEURA_ERROR_NOT_SUPPORTED: return "Not supported";
        case NEURA_ERROR_BUSY: return "Resource busy";
        case NEURA_ERROR_PERMISSION: return "Permission denied";
        default: return "Unknown error";
    }
}

/**
 * @brief Memory allocation with alignment
 */
void* neura_malloc_aligned(size_t size, size_t alignment) {
    void* ptr = NULL;
    
#ifdef _WIN32
    ptr = _aligned_malloc(size, alignment);
#else
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
#endif
    
    return ptr;
}

/**
 * @brief Free aligned memory
 */
void neura_free_aligned(void* ptr) {
    if (!ptr) return;
    
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

/**
 * @brief Get current timestamp in microseconds
 */
uint64_t neura_get_timestamp_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000ULL + (uint64_t)tv.tv_usec;
}

/**
 * @brief Get current timestamp in milliseconds
 */
uint64_t neura_get_timestamp_ms(void) {
    return neura_get_timestamp_us() / 1000ULL;
}

/**
 * @brief Sleep for microseconds
 */
void neura_sleep_us(uint64_t us) {
    usleep((useconds_t)us);
}

/**
 * @brief Sleep for milliseconds
 */
void neura_sleep_ms(uint64_t ms) {
    neura_sleep_us(ms * 1000ULL);
}

