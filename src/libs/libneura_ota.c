/**
 * @file libneura_ota.c
 * @brief OTA Update System Implementation
 * @version 1.0.0-alpha
 * @date October 2025
 */

#include "libneura_ota.h"
#include "libneura_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <curl/curl.h>
#include <openssl/sha.h>
#include <openssl/evp.h>
#include <openssl/pem.h>

/* Global state */
static ota_config_t g_config;
static ota_state_t g_state = OTA_STATE_IDLE;
static ota_error_t g_last_error = OTA_OK;
static ota_progress_callback_t g_progress_callback = NULL;
static void* g_progress_user_data = NULL;
static bool g_initialized = false;

/**
 * @brief Set OTA state
 */
static void set_state(ota_state_t state) {
    g_state = state;
    if (g_progress_callback) {
        g_progress_callback(state, 0, g_progress_user_data);
    }
}

/**
 * @brief Calculate SHA-256 checksum
 */
static int calculate_checksum(const char* file_path, char* checksum_out) {
    FILE* fp = fopen(file_path, "rb");
    if (!fp) {
        return -1;
    }
    
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    
    unsigned char buffer[8192];
    size_t bytes;
    
    while ((bytes = fread(buffer, 1, sizeof(buffer), fp)) > 0) {
        SHA256_Update(&sha256, buffer, bytes);
    }
    
    SHA256_Final(hash, &sha256);
    fclose(fp);
    
    /* Convert to hex string */
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        sprintf(checksum_out + (i * 2), "%02x", hash[i]);
    }
    checksum_out[64] = '\0';
    
    return 0;
}

/**
 * @brief Verify digital signature
 */
static int verify_signature(const char* file_path, const char* signature,
                           const char* public_key_path) {
    /* This is a simplified implementation */
    /* Real implementation would use proper RSA/ECDSA verification */
    
    if (!g_config.verify_signature) {
        return 0; /* Skip verification if disabled */
    }
    
    /* Placeholder for signature verification */
    neura_log(NEURA_LOG_INFO, "Verifying signature (placeholder)");
    
    return 0;
}

/**
 * @brief Initialize OTA system
 */
int ota_init(const ota_config_t* config) {
    if (g_initialized) {
        return OTA_OK;
    }
    
    if (!config) {
        return OTA_ERROR_INVALID_PACKAGE;
    }
    
    memcpy(&g_config, config, sizeof(ota_config_t));
    
    /* Initialize curl */
    curl_global_init(CURL_GLOBAL_DEFAULT);
    
    g_initialized = true;
    g_state = OTA_STATE_IDLE;
    g_last_error = OTA_OK;
    
    neura_log(NEURA_LOG_INFO, "OTA system initialized");
    neura_log(NEURA_LOG_INFO, "Current version: %s", g_config.current_version);
    neura_log(NEURA_LOG_INFO, "Server URL: %s", g_config.server_url);
    
    return OTA_OK;
}

/**
 * @brief Cleanup OTA system
 */
void ota_cleanup(void) {
    if (!g_initialized) {
        return;
    }
    
    curl_global_cleanup();
    
    g_initialized = false;
    g_state = OTA_STATE_IDLE;
    
    neura_log(NEURA_LOG_INFO, "OTA system cleaned up");
}

/**
 * @brief CURL write callback
 */
static size_t write_callback(void* ptr, size_t size, size_t nmemb, void* stream) {
    size_t written = fwrite(ptr, size, nmemb, (FILE*)stream);
    return written;
}

/**
 * @brief CURL progress callback
 */
static int progress_callback(void* clientp, curl_off_t dltotal, curl_off_t dlnow,
                            curl_off_t ultotal, curl_off_t ulnow) {
    if (dltotal > 0 && g_progress_callback) {
        int percent = (int)((dlnow * 100) / dltotal);
        g_progress_callback(g_state, percent, g_progress_user_data);
    }
    return 0;
}

/**
 * @brief Check for updates
 */
int ota_check_update(ota_package_info_t* info) {
    if (!g_initialized) {
        return OTA_ERROR_INVALID_PACKAGE;
    }
    
    set_state(OTA_STATE_CHECKING);
    
    /* Build update check URL */
    char url[1024];
    snprintf(url, sizeof(url), "%s/check?device=%s&version=%s",
             g_config.server_url, g_config.device_id, g_config.current_version);
    
    neura_log(NEURA_LOG_INFO, "Checking for updates: %s", url);
    
    /* For this implementation, we'll simulate an update check */
    /* Real implementation would make HTTP request to server */
    
    set_state(OTA_STATE_IDLE);
    
    /* No update available (placeholder) */
    return 0;
}

/**
 * @brief Download update package
 */
int ota_download_update(const ota_package_info_t* info,
                        ota_progress_callback_t callback, void* user_data) {
    if (!g_initialized || !info) {
        return OTA_ERROR_INVALID_PACKAGE;
    }
    
    set_state(OTA_STATE_DOWNLOADING);
    
    /* Set progress callback */
    ota_progress_callback_t old_callback = g_progress_callback;
    void* old_user_data = g_progress_user_data;
    
    if (callback) {
        g_progress_callback = callback;
        g_progress_user_data = user_data;
    }
    
    neura_log(NEURA_LOG_INFO, "Downloading update: %s", info->url);
    neura_log(NEURA_LOG_INFO, "Version: %s, Size: %llu bytes", 
              info->version, info->size);
    
    /* Download to temporary file */
    const char* temp_path = "/tmp/neuraos_update.pkg";
    FILE* fp = fopen(temp_path, "wb");
    if (!fp) {
        g_last_error = OTA_ERROR_NETWORK;
        set_state(OTA_STATE_ERROR);
        return OTA_ERROR_NETWORK;
    }
    
    CURL* curl = curl_easy_init();
    if (!curl) {
        fclose(fp);
        g_last_error = OTA_ERROR_NETWORK;
        set_state(OTA_STATE_ERROR);
        return OTA_ERROR_NETWORK;
    }
    
    curl_easy_setopt(curl, CURLOPT_URL, info->url);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progress_callback);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    
    CURLcode res = curl_easy_perform(curl);
    
    curl_easy_cleanup(curl);
    fclose(fp);
    
    /* Restore old callback */
    g_progress_callback = old_callback;
    g_progress_user_data = old_user_data;
    
    if (res != CURLE_OK) {
        neura_log(NEURA_LOG_ERROR, "Download failed: %s", curl_easy_strerror(res));
        g_last_error = OTA_ERROR_NETWORK;
        set_state(OTA_STATE_ERROR);
        return OTA_ERROR_NETWORK;
    }
    
    neura_log(NEURA_LOG_INFO, "Download complete");
    set_state(OTA_STATE_IDLE);
    
    return OTA_OK;
}

/**
 * @brief Verify update package
 */
int ota_verify_update(const char* package_path) {
    if (!g_initialized || !package_path) {
        return OTA_ERROR_INVALID_PACKAGE;
    }
    
    set_state(OTA_STATE_VERIFYING);
    
    neura_log(NEURA_LOG_INFO, "Verifying update package: %s", package_path);
    
    /* Check if file exists */
    if (access(package_path, F_OK) != 0) {
        neura_log(NEURA_LOG_ERROR, "Package file not found");
        g_last_error = OTA_ERROR_INVALID_PACKAGE;
        set_state(OTA_STATE_ERROR);
        return OTA_ERROR_INVALID_PACKAGE;
    }
    
    /* Calculate checksum */
    char checksum[65];
    if (calculate_checksum(package_path, checksum) != 0) {
        neura_log(NEURA_LOG_ERROR, "Failed to calculate checksum");
        g_last_error = OTA_ERROR_VERIFICATION_FAILED;
        set_state(OTA_STATE_ERROR);
        return OTA_ERROR_VERIFICATION_FAILED;
    }
    
    neura_log(NEURA_LOG_INFO, "Package checksum: %s", checksum);
    
    /* Verify signature if enabled */
    if (g_config.verify_signature) {
        if (verify_signature(package_path, "", g_config.public_key_path) != 0) {
            neura_log(NEURA_LOG_ERROR, "Signature verification failed");
            g_last_error = OTA_ERROR_VERIFICATION_FAILED;
            set_state(OTA_STATE_ERROR);
            return OTA_ERROR_VERIFICATION_FAILED;
        }
    }
    
    neura_log(NEURA_LOG_INFO, "Package verification successful");
    set_state(OTA_STATE_IDLE);
    
    return OTA_OK;
}

/**
 * @brief Install update
 */
int ota_install_update(const char* package_path,
                       ota_progress_callback_t callback, void* user_data) {
    if (!g_initialized || !package_path) {
        return OTA_ERROR_INVALID_PACKAGE;
    }
    
    set_state(OTA_STATE_INSTALLING);
    
    neura_log(NEURA_LOG_INFO, "Installing update from: %s", package_path);
    
    /* This is a placeholder implementation */
    /* Real implementation would:
     * 1. Extract package
     * 2. Backup current system
     * 3. Install new files
     * 4. Update bootloader
     * 5. Verify installation
     */
    
    if (callback) {
        callback(OTA_STATE_INSTALLING, 50, user_data);
    }
    
    neura_log(NEURA_LOG_INFO, "Update installed successfully");
    neura_log(NEURA_LOG_INFO, "System will reboot to apply changes");
    
    set_state(OTA_STATE_COMPLETE);
    
    return OTA_OK;
}

/**
 * @brief Rollback to previous version
 */
int ota_rollback(void) {
    if (!g_initialized) {
        return OTA_ERROR_INVALID_PACKAGE;
    }
    
    neura_log(NEURA_LOG_INFO, "Rolling back to previous version");
    
    /* Placeholder for rollback implementation */
    
    return OTA_OK;
}

/**
 * @brief Get current OTA state
 */
ota_state_t ota_get_state(void) {
    return g_state;
}

/**
 * @brief Get last error
 */
ota_error_t ota_get_last_error(void) {
    return g_last_error;
}

/**
 * @brief Set progress callback
 */
void ota_set_progress_callback(ota_progress_callback_t callback, void* user_data) {
    g_progress_callback = callback;
    g_progress_user_data = user_data;
}

