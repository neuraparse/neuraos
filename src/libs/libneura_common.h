/**
 * @file libneura_common.h
 * @brief Common utilities and definitions for NeuralOS
 * @version 1.0.0-alpha
 * @date October 2025
 */

#ifndef LIBNEURA_COMMON_H
#define LIBNEURA_COMMON_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Version information */
#define NEURAOS_VERSION_MAJOR 1
#define NEURAOS_VERSION_MINOR 0
#define NEURAOS_VERSION_PATCH 0
#define NEURAOS_VERSION_STRING "1.0.0-alpha"

/* Error codes */
typedef enum {
    NEURA_OK = 0,
    NEURA_ERROR_INVALID_PARAM = -1,
    NEURA_ERROR_OUT_OF_MEMORY = -2,
    NEURA_ERROR_NOT_FOUND = -3,
    NEURA_ERROR_IO = -4,
    NEURA_ERROR_TIMEOUT = -5,
    NEURA_ERROR_NOT_SUPPORTED = -6,
    NEURA_ERROR_BUSY = -7,
    NEURA_ERROR_PERMISSION = -8,
    NEURA_ERROR_UNKNOWN = -99
} neura_error_t;

/* Log levels */
typedef enum {
    NEURA_LOG_TRACE = 0,
    NEURA_LOG_DEBUG = 1,
    NEURA_LOG_INFO = 2,
    NEURA_LOG_WARN = 3,
    NEURA_LOG_ERROR = 4,
    NEURA_LOG_FATAL = 5
} neura_log_level_t;

/* Log callback */
typedef void (*neura_log_callback_t)(neura_log_level_t level, const char* message, void* user_data);

/**
 * @brief Initialize common library
 */
int neura_common_init(void);

/**
 * @brief Cleanup common library
 */
void neura_common_cleanup(void);

/**
 * @brief Get version string
 */
const char* neura_get_version(void);

/**
 * @brief Set log callback
 */
void neura_set_log_callback(neura_log_callback_t callback, void* user_data);

/**
 * @brief Log a message
 */
void neura_log(neura_log_level_t level, const char* format, ...);

/**
 * @brief Get error string
 */
const char* neura_error_string(neura_error_t error);

/**
 * @brief Memory allocation with alignment
 */
void* neura_malloc_aligned(size_t size, size_t alignment);

/**
 * @brief Free aligned memory
 */
void neura_free_aligned(void* ptr);

/**
 * @brief Get current timestamp in microseconds
 */
uint64_t neura_get_timestamp_us(void);

/**
 * @brief Get current timestamp in milliseconds
 */
uint64_t neura_get_timestamp_ms(void);

/**
 * @brief Sleep for microseconds
 */
void neura_sleep_us(uint64_t us);

/**
 * @brief Sleep for milliseconds
 */
void neura_sleep_ms(uint64_t ms);

#ifdef __cplusplus
}
#endif

#endif /* LIBNEURA_COMMON_H */

