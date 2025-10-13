# NPIE API Reference

## Overview

The NeuraParse Inference Engine (NPIE) provides a unified C API for running AI models on embedded devices. This document describes the complete API.

---

## Table of Contents

1. [Core API](#core-api)
2. [Model Management](#model-management)
3. [Inference API](#inference-api)
4. [Hardware Detection](#hardware-detection)
5. [Utility Functions](#utility-functions)
6. [Error Handling](#error-handling)
7. [Examples](#examples)

---

## Core API

### npie_version()

Get NPIE version string.

**Signature:**
```c
const char* npie_version(void);
```

**Returns:** Version string (e.g., "1.0.0-alpha")

**Example:**
```c
printf("NPIE Version: %s\n", npie_version());
```

---

### npie_init()

Initialize NPIE context.

**Signature:**
```c
npie_status_t npie_init(npie_context_t* ctx, const npie_options_t* options);
```

**Parameters:**
- `ctx` - Pointer to context handle (output)
- `options` - Initialization options (NULL for defaults)

**Returns:** `NPIE_SUCCESS` or error code

**Example:**
```c
npie_context_t ctx;
npie_options_t options = {
    .backend = NPIE_BACKEND_AUTO,
    .accelerator = NPIE_ACCELERATOR_AUTO,
    .num_threads = 4,
    .enable_profiling = true
};

if (npie_init(&ctx, &options) != NPIE_SUCCESS) {
    fprintf(stderr, "Failed to initialize NPIE\n");
    return 1;
}
```

---

### npie_shutdown()

Shutdown NPIE context and free resources.

**Signature:**
```c
npie_status_t npie_shutdown(npie_context_t ctx);
```

**Parameters:**
- `ctx` - Context handle

**Returns:** `NPIE_SUCCESS` or error code

**Example:**
```c
npie_shutdown(ctx);
```

---

## Model Management

### npie_model_load()

Load model from file.

**Signature:**
```c
npie_status_t npie_model_load(npie_context_t ctx,
                              npie_model_t* model,
                              const char* path,
                              const npie_options_t* options);
```

**Parameters:**
- `ctx` - Context handle
- `model` - Pointer to model handle (output)
- `path` - Path to model file
- `options` - Load options (NULL for defaults)

**Returns:** `NPIE_SUCCESS` or error code

**Example:**
```c
npie_model_t model;
if (npie_model_load(ctx, &model, "/opt/neuraparse/models/mobilenet_v2.tflite", NULL) != NPIE_SUCCESS) {
    fprintf(stderr, "Failed to load model\n");
    return 1;
}
```

---

### npie_model_get_info()

Get model information.

**Signature:**
```c
npie_status_t npie_model_get_info(npie_model_t model, npie_model_info_t* info);
```

**Parameters:**
- `model` - Model handle
- `info` - Pointer to model info structure (output)

**Returns:** `NPIE_SUCCESS` or error code

**Example:**
```c
npie_model_info_t info;
npie_model_get_info(model, &info);
printf("Model: %s\n", info.name);
printf("Inputs: %u, Outputs: %u\n", info.input_count, info.output_count);
```

---

## Inference API

### npie_inference_run()

Run inference on model.

**Signature:**
```c
npie_status_t npie_inference_run(npie_model_t model,
                                 const npie_tensor_t* inputs,
                                 uint32_t num_inputs,
                                 npie_tensor_t* outputs,
                                 uint32_t num_outputs,
                                 npie_metrics_t* metrics);
```

**Parameters:**
- `model` - Model handle
- `inputs` - Array of input tensors
- `num_inputs` - Number of input tensors
- `outputs` - Array of output tensors (allocated by caller)
- `num_outputs` - Number of output tensors
- `metrics` - Performance metrics (NULL if not needed)

**Returns:** `NPIE_SUCCESS` or error code

**Example:**
```c
// Prepare input
npie_tensor_t input;
npie_model_get_input(model, 0, &input);
npie_tensor_alloc(&input);
// ... fill input.data with image data ...

// Prepare output
npie_tensor_t output;
npie_model_get_output(model, 0, &output);
npie_tensor_alloc(&output);

// Run inference
npie_metrics_t metrics;
if (npie_inference_run(model, &input, 1, &output, 1, &metrics) == NPIE_SUCCESS) {
    printf("Inference time: %.2f ms\n", metrics.inference_time_us / 1000.0);
    // Process output.data
}

// Cleanup
npie_tensor_free(&input);
npie_tensor_free(&output);
```

---

## Hardware Detection

### npie_detect_accelerators()

Detect available hardware accelerators.

**Signature:**
```c
npie_status_t npie_detect_accelerators(npie_context_t ctx,
                                       npie_accelerator_t* accelerators,
                                       uint32_t max_count,
                                       uint32_t* count);
```

**Parameters:**
- `ctx` - Context handle
- `accelerators` - Array to store detected accelerators
- `max_count` - Maximum number of accelerators to detect
- `count` - Pointer to store actual count (output)

**Returns:** `NPIE_SUCCESS` or error code

**Example:**
```c
npie_accelerator_t accelerators[16];
uint32_t count;

npie_detect_accelerators(ctx, accelerators, 16, &count);
printf("Found %u accelerators:\n", count);
for (uint32_t i = 0; i < count; i++) {
    const char* names[] = {"None", "Auto", "GPU", "NPU", "TPU", "DSP"};
    printf("  - %s\n", names[accelerators[i]]);
}
```

---

## Data Types

### npie_status_t

Return status codes.

```c
typedef enum {
    NPIE_SUCCESS = 0,
    NPIE_ERROR_INVALID_ARGUMENT = -1,
    NPIE_ERROR_OUT_OF_MEMORY = -2,
    NPIE_ERROR_MODEL_LOAD_FAILED = -3,
    NPIE_ERROR_INFERENCE_FAILED = -4,
    NPIE_ERROR_UNSUPPORTED_OPERATION = -5,
    NPIE_ERROR_HARDWARE_NOT_AVAILABLE = -6,
    NPIE_ERROR_TIMEOUT = -7,
    NPIE_ERROR_NOT_INITIALIZED = -8,
    NPIE_ERROR_ALREADY_INITIALIZED = -9,
    NPIE_ERROR_IO = -10,
    NPIE_ERROR_UNKNOWN = -99
} npie_status_t;
```

### npie_tensor_t

Tensor descriptor.

```c
typedef struct {
    const char* name;               // Tensor name
    npie_dtype_t dtype;             // Data type
    npie_shape_t shape;             // Shape information
    void* data;                     // Pointer to data
    size_t size;                    // Size in bytes
} npie_tensor_t;
```

### npie_metrics_t

Performance metrics.

```c
typedef struct {
    uint64_t inference_time_us;     // Inference time in microseconds
    uint64_t preprocessing_time_us; // Preprocessing time
    uint64_t postprocessing_time_us;// Postprocessing time
    uint64_t total_time_us;         // Total time
    size_t memory_used_bytes;       // Memory used
    float cpu_usage_percent;        // CPU usage
    float accelerator_usage_percent;// Accelerator usage
} npie_metrics_t;
```

---

## Complete Example

```c
#include <npie.h>
#include <stdio.h>

int main() {
    // Initialize NPIE
    npie_context_t ctx;
    npie_options_t options = {
        .backend = NPIE_BACKEND_AUTO,
        .accelerator = NPIE_ACCELERATOR_AUTO,
        .num_threads = 4
    };
    
    if (npie_init(&ctx, &options) != NPIE_SUCCESS) {
        return 1;
    }
    
    // Load model
    npie_model_t model;
    if (npie_model_load(ctx, &model, "model.tflite", NULL) != NPIE_SUCCESS) {
        npie_shutdown(ctx);
        return 1;
    }
    
    // Get input/output info
    npie_tensor_t input, output;
    npie_model_get_input(model, 0, &input);
    npie_model_get_output(model, 0, &output);
    
    // Allocate tensors
    npie_tensor_alloc(&input);
    npie_tensor_alloc(&output);
    
    // Fill input with data
    // ... (load image, preprocess, etc.) ...
    
    // Run inference
    npie_metrics_t metrics;
    if (npie_inference_run(model, &input, 1, &output, 1, &metrics) == NPIE_SUCCESS) {
        printf("Inference completed in %.2f ms\n", metrics.inference_time_us / 1000.0);
        
        // Process output
        float* predictions = (float*)output.data;
        // ... (find top predictions, etc.) ...
    }
    
    // Cleanup
    npie_tensor_free(&input);
    npie_tensor_free(&output);
    npie_model_unload(model);
    npie_shutdown(ctx);
    
    return 0;
}
```

**Compile:**
```bash
gcc -o app app.c -lnpie -lpthread -lm
```

---

## Thread Safety

- `npie_context_t` is thread-safe for read operations
- `npie_model_t` can be used from multiple threads simultaneously
- Inference operations are thread-safe
- Use separate contexts for complete isolation

---

## Performance Tips

1. **Reuse tensors** - Allocate once, reuse for multiple inferences
2. **Batch processing** - Process multiple inputs together when possible
3. **Hardware acceleration** - Always enable when available
4. **Quantization** - Use INT8 models for 4x speedup
5. **Threading** - Set `num_threads` to number of CPU cores
6. **Caching** - Enable model caching for faster startup

---

## See Also

- [Getting Started Guide](getting_started.md)
- [Architecture Documentation](architecture_2025.md)
- [Hardware Support](hardware_support.md)
- [Examples](../examples/)

