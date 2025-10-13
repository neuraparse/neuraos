/**
 * @file libneura_ota.h
 * @brief Over-The-Air (OTA) Update System for NeuralOS
 * @version 1.0.0-alpha
 * @date October 2025
 */

#ifndef LIBNEURA_OTA_H
#define LIBNEURA_OTA_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* OTA update states */
typedef enum {
    OTA_STATE_IDLE = 0,
    OTA_STATE_CHECKING,
    OTA_STATE_DOWNLOADING,
    OTA_STATE_VERIFYING,
    OTA_STATE_INSTALLING,
    OTA_STATE_COMPLETE,
    OTA_STATE_ERROR
} ota_state_t;

/* OTA error codes */
typedef enum {
    OTA_OK = 0,
    OTA_ERROR_NETWORK = -1,
    OTA_ERROR_INVALID_PACKAGE = -2,
    OTA_ERROR_VERIFICATION_FAILED = -3,
    OTA_ERROR_INSUFFICIENT_SPACE = -4,
    OTA_ERROR_INSTALL_FAILED = -5,
    OTA_ERROR_ROLLBACK_FAILED = -6
} ota_error_t;

/* Update package information */
typedef struct {
    char version[32];
    char description[256];
    uint64_t size;
    char checksum[65];  /* SHA-256 */
    char signature[256];
    char url[512];
    bool critical;
} ota_package_info_t;

/* OTA configuration */
typedef struct {
    char server_url[512];
    char device_id[64];
    char current_version[32];
    bool auto_check;
    uint32_t check_interval_seconds;
    bool verify_signature;
    char public_key_path[256];
} ota_config_t;

/* Progress callback */
typedef void (*ota_progress_callback_t)(ota_state_t state, int progress_percent, void* user_data);

/**
 * @brief Initialize OTA system
 */
int ota_init(const ota_config_t* config);

/**
 * @brief Cleanup OTA system
 */
void ota_cleanup(void);

/**
 * @brief Check for updates
 * @param info Output parameter for update information
 * @return 1 if update available, 0 if no update, negative on error
 */
int ota_check_update(ota_package_info_t* info);

/**
 * @brief Download update package
 */
int ota_download_update(const ota_package_info_t* info, 
                        ota_progress_callback_t callback, void* user_data);

/**
 * @brief Verify update package
 */
int ota_verify_update(const char* package_path);

/**
 * @brief Install update
 */
int ota_install_update(const char* package_path,
                       ota_progress_callback_t callback, void* user_data);

/**
 * @brief Rollback to previous version
 */
int ota_rollback(void);

/**
 * @brief Get current OTA state
 */
ota_state_t ota_get_state(void);

/**
 * @brief Get last error
 */
ota_error_t ota_get_last_error(void);

/**
 * @brief Set progress callback
 */
void ota_set_progress_callback(ota_progress_callback_t callback, void* user_data);

#ifdef __cplusplus
}
#endif

#endif /* LIBNEURA_OTA_H */

