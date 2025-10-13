/**
 * @file image_classifier.c
 * @brief Simple image classification example using NPIE
 * @version 1.0.0-alpha
 * @date October 2025
 * 
 * This example demonstrates how to use the NeuraParse Inference Engine
 * to perform image classification using MobileNetV2.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <npie.h>

#ifdef NEURAOS_ENABLE_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv;
#endif

/* ImageNet class labels (top 5 for demo) */
static const char* imagenet_labels[] = {
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead",
    /* ... 995 more labels ... */
};

/**
 * @brief Load and preprocess image
 */
int load_image(const char* path, npie_tensor_t* tensor) {
#ifdef NEURAOS_ENABLE_OPENCV
    /* Load image using OpenCV */
    Mat img = imread(path, IMREAD_COLOR);
    if (img.empty()) {
        fprintf(stderr, "Error: Failed to load image: %s\n", path);
        return -1;
    }
    
    /* Resize to 224x224 (MobileNetV2 input size) */
    Mat resized;
    resize(img, resized, Size(224, 224));
    
    /* Convert BGR to RGB */
    Mat rgb;
    cvtColor(resized, rgb, COLOR_BGR2RGB);
    
    /* Normalize to [-1, 1] */
    Mat normalized;
    rgb.convertTo(normalized, CV_32F, 1.0/127.5, -1.0);
    
    /* Copy to tensor */
    memcpy(tensor->data, normalized.data, tensor->size);
    
    return 0;
#else
    fprintf(stderr, "Error: OpenCV support not enabled\n");
    return -1;
#endif
}

/**
 * @brief Find top-k predictions
 */
void get_top_k(const float* predictions, int num_classes, int k, 
               int* indices, float* scores) {
    /* Simple selection sort for top-k */
    for (int i = 0; i < k; i++) {
        int max_idx = 0;
        float max_val = -1.0f;
        
        for (int j = 0; j < num_classes; j++) {
            /* Skip already selected indices */
            int skip = 0;
            for (int m = 0; m < i; m++) {
                if (indices[m] == j) {
                    skip = 1;
                    break;
                }
            }
            if (skip) continue;
            
            if (predictions[j] > max_val) {
                max_val = predictions[j];
                max_idx = j;
            }
        }
        
        indices[i] = max_idx;
        scores[i] = max_val;
    }
}

/**
 * @brief Logging callback
 */
void log_callback(int level, const char* message, void* user_data) {
    const char* level_str[] = {"DEBUG", "INFO", "WARN", "ERROR"};
    printf("[%s] %s\n", level_str[level], message);
}

/**
 * @brief Main function
 */
int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <model_path> <image_path>\n", argv[0]);
        fprintf(stderr, "Example: %s mobilenet_v2.tflite cat.jpg\n", argv[0]);
        return 1;
    }
    
    const char* model_path = argv[1];
    const char* image_path = argv[2];
    
    printf("NeuralOS Image Classification Example\n");
    printf("======================================\n\n");
    
    /* Initialize NPIE */
    npie_context_t ctx;
    npie_options_t options = {
        .backend = NPIE_BACKEND_AUTO,
        .accelerator = NPIE_ACCELERATOR_AUTO,
        .num_threads = 4,
        .timeout_ms = 5000,
        .enable_profiling = 1,
        .enable_caching = 1,
        .user_data = NULL
    };
    
    npie_status_t status = npie_init(&ctx, &options);
    if (status != NPIE_SUCCESS) {
        fprintf(stderr, "Error: Failed to initialize NPIE: %s\n", 
                npie_status_string(status));
        return 1;
    }
    
    /* Set logging callback */
    npie_set_log_callback(ctx, log_callback, NULL);
    
    printf("NPIE Version: %s\n", npie_version());
    
    /* Detect available accelerators */
    npie_accelerator_t accelerators[16];
    uint32_t accel_count;
    npie_detect_accelerators(ctx, accelerators, 16, &accel_count);
    
    printf("Available accelerators: ");
    for (uint32_t i = 0; i < accel_count; i++) {
        const char* accel_name[] = {
            "None", "Auto", "GPU", "NPU", "TPU", "DSP", "Custom"
        };
        printf("%s ", accel_name[accelerators[i]]);
    }
    printf("\n\n");
    
    /* Load model */
    printf("Loading model: %s\n", model_path);
    npie_model_t model;
    status = npie_model_load(ctx, &model, model_path, NULL);
    if (status != NPIE_SUCCESS) {
        fprintf(stderr, "Error: Failed to load model: %s\n", 
                npie_status_string(status));
        npie_shutdown(ctx);
        return 1;
    }
    
    /* Get model info */
    npie_model_info_t info;
    npie_model_get_info(model, &info);
    
    printf("Model Information:\n");
    printf("  Name: %s\n", info.name ? info.name : "Unknown");
    printf("  Backend: %d\n", info.backend);
    printf("  Accelerator: %d\n", info.accelerator);
    printf("  Inputs: %u\n", info.input_count);
    printf("  Outputs: %u\n", info.output_count);
    printf("\n");
    
    /* Get input tensor info */
    npie_tensor_t input_tensor;
    npie_model_get_input(model, 0, &input_tensor);
    
    printf("Input Tensor:\n");
    printf("  Name: %s\n", input_tensor.name ? input_tensor.name : "input");
    printf("  Shape: [");
    for (uint32_t i = 0; i < input_tensor.shape.rank; i++) {
        printf("%u%s", input_tensor.shape.dims[i], 
               i < input_tensor.shape.rank - 1 ? ", " : "");
    }
    printf("]\n");
    printf("  Size: %zu bytes\n", input_tensor.size);
    printf("\n");
    
    /* Allocate input tensor */
    npie_tensor_alloc(&input_tensor);
    
    /* Load and preprocess image */
    printf("Loading image: %s\n", image_path);
    if (load_image(image_path, &input_tensor) != 0) {
        npie_tensor_free(&input_tensor);
        npie_model_unload(model);
        npie_shutdown(ctx);
        return 1;
    }
    
    /* Get output tensor info */
    npie_tensor_t output_tensor;
    npie_model_get_output(model, 0, &output_tensor);
    npie_tensor_alloc(&output_tensor);
    
    /* Run inference */
    printf("Running inference...\n");
    npie_metrics_t metrics;
    status = npie_inference_run(model, &input_tensor, 1, 
                                &output_tensor, 1, &metrics);
    if (status != NPIE_SUCCESS) {
        fprintf(stderr, "Error: Inference failed: %s\n", 
                npie_status_string(status));
        npie_tensor_free(&input_tensor);
        npie_tensor_free(&output_tensor);
        npie_model_unload(model);
        npie_shutdown(ctx);
        return 1;
    }
    
    /* Print performance metrics */
    printf("\nPerformance Metrics:\n");
    printf("  Inference time: %.2f ms\n", metrics.inference_time_us / 1000.0);
    printf("  Preprocessing: %.2f ms\n", metrics.preprocessing_time_us / 1000.0);
    printf("  Postprocessing: %.2f ms\n", metrics.postprocessing_time_us / 1000.0);
    printf("  Total time: %.2f ms\n", metrics.total_time_us / 1000.0);
    printf("  Memory used: %.2f MB\n", metrics.memory_used_bytes / (1024.0 * 1024.0));
    printf("  CPU usage: %.1f%%\n", metrics.cpu_usage_percent);
    if (metrics.accelerator_usage_percent > 0) {
        printf("  Accelerator usage: %.1f%%\n", metrics.accelerator_usage_percent);
    }
    printf("\n");
    
    /* Get top-5 predictions */
    float* predictions = (float*)output_tensor.data;
    int num_classes = output_tensor.shape.dims[output_tensor.shape.rank - 1];
    
    int top_indices[5];
    float top_scores[5];
    get_top_k(predictions, num_classes, 5, top_indices, top_scores);
    
    printf("Top-5 Predictions:\n");
    for (int i = 0; i < 5; i++) {
        const char* label = top_indices[i] < 1000 ? 
                           imagenet_labels[top_indices[i]] : "Unknown";
        printf("  %d. %s (%.2f%%)\n", i + 1, label, top_scores[i] * 100.0);
    }
    printf("\n");
    
    /* Cleanup */
    npie_tensor_free(&input_tensor);
    npie_tensor_free(&output_tensor);
    npie_model_unload(model);
    npie_shutdown(ctx);
    
    printf("Done!\n");
    
    return 0;
}

