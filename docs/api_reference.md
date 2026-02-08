# NPIE v2.0.0 & Drone/Robotics API Reference

## Overview

The NeuraParse Inference Engine (NPIE) v2.0.0 provides a unified C API for running AI models on embedded devices. All 12 inference backends have full C/C++ implementation files with load/inference/unload support, including LLM text generation, speech-to-text, image generation, and quantum circuit simulation.

### Backend Implementation Files (12/12)

| Backend | Source File | Library Dependency |
|---------|-----------|-------------------|
| LiteRT | `npie_litert.cpp` | tensorflow-lite |
| ONNX Runtime | `npie_onnx.cpp` | onnxruntime |
| emlearn | `npie_emlearn.c` | emlearn (header-only) |
| WasmEdge | `npie_wasm.cpp` | wasmedge |
| NCNN | `npie_ncnn.cpp` | ncnn (+ Vulkan optional) |
| ExecuTorch | `npie_executorch.cpp` | executorch |
| OpenVINO | `npie_openvino.cpp` | openvino::runtime |
| llama.cpp | `npie_llama.cpp` | llama, common |
| whisper.cpp | `npie_whisper.cpp` | whisper |
| stable-diffusion.cpp | `npie_stable_diffusion.cpp` | stable-diffusion |
| MLC LLM | `npie_mlc_llm.cpp` | tvm_runtime |
| QuEST | `npie_quest.cpp` | QuEST |

---

## Table of Contents

1. [Core API](#core-api)
2. [Model Management](#model-management)
3. [Inference API](#inference-api)
4. [LLM API](#llm-api) (NEW in v2.0.0)
5. [Speech API](#speech-api) (NEW in v2.0.0)
6. [Quantum API](#quantum-api) (NEW in v2.0.0)
7. [Hardware Detection](#hardware-detection)
8. [Data Types](#data-types)
9. [Examples](#examples)
10. [MAVLink API](#mavlink-api) (NEW in v5.0.0)
11. [Swarm API](#swarm-api) (NEW in v5.0.0)
12. [Sensor Fusion API](#sensor-fusion-api) (NEW in v5.0.0)
13. [Network API](#network-api) (NEW in v5.0.0)
14. [Security API](#security-api) (NEW in v5.0.0)

---

## Core API

### npie_version()

Get NPIE version string.

```c
const char* npie_version(void);
// Returns: "2.0.0"
```

### npie_init()

Initialize NPIE context.

```c
npie_status_t npie_init(npie_context_t* ctx, const npie_options_t* options);
```

**Parameters:**
- `ctx` - Pointer to context handle (output)
- `options` - Initialization options (NULL for defaults)

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

### npie_shutdown()

Shutdown NPIE context and free resources.

```c
npie_status_t npie_shutdown(npie_context_t ctx);
```

---

## Model Management

### npie_model_load()

Load model from file.

```c
npie_status_t npie_model_load(npie_context_t ctx,
                              npie_model_t* model,
                              const char* path,
                              const npie_options_t* options);
```

### npie_model_get_info()

Get model information.

```c
npie_status_t npie_model_get_info(npie_model_t model, npie_model_info_t* info);
```

---

## Inference API

### npie_inference_run()

Run inference on model.

```c
npie_status_t npie_inference_run(npie_model_t model,
                                 const npie_tensor_t* inputs,
                                 uint32_t num_inputs,
                                 npie_tensor_t* outputs,
                                 uint32_t num_outputs,
                                 npie_metrics_t* metrics);
```

---

## LLM API

*New in NPIE v2.0.0* - Large Language Model inference using llama.cpp backend.

### npie_llm_load()

Load a GGUF model for LLM inference.

```c
npie_status_t npie_llm_load(npie_context_t ctx,
                             npie_llm_t* llm,
                             const char* model_path,
                             const npie_llm_params_t* params);
```

**Parameters:**
- `ctx` - NPIE context
- `llm` - Pointer to LLM handle (output)
- `model_path` - Path to GGUF model file
- `params` - LLM configuration parameters

**LLM Parameters:**
```c
typedef struct {
    uint32_t max_tokens;          // Maximum tokens to generate (default: 256)
    float    temperature;          // Sampling temperature (default: 0.7)
    float    top_p;               // Nucleus sampling (default: 0.9)
    uint32_t top_k;               // Top-K sampling (default: 40)
    float    repeat_penalty;       // Repetition penalty (default: 1.1)
    uint32_t context_size;         // Context window size (default: 2048)
    uint32_t num_threads;          // CPU threads for inference
    npie_accelerator_t accelerator;// GPU/NPU acceleration
    npie_quantization_t quantization; // Model quantization mode
    bool     use_mmap;            // Memory-mapped model loading
    bool     use_mlock;           // Lock model in memory
} npie_llm_params_t;
```

**Quantization Modes:**
```c
typedef enum {
    NPIE_QUANT_NONE = 0,
    NPIE_QUANT_Q4_K_M,    // 4-bit K-quant medium (recommended)
    NPIE_QUANT_Q4_K_S,    // 4-bit K-quant small
    NPIE_QUANT_Q5_K_M,    // 5-bit K-quant medium (higher quality)
    NPIE_QUANT_Q8_0,      // 8-bit quantization
    NPIE_QUANT_IQ2_XXS,   // 2-bit importance quantization (smallest)
    NPIE_QUANT_IQ3_S,     // 3-bit importance quantization
    NPIE_QUANT_FP16_NF4,  // FP16 with NormalFloat4
} npie_quantization_t;
```

**Example:**
```c
npie_llm_t llm;
npie_llm_params_t params = {
    .max_tokens = 512,
    .temperature = 0.7,
    .top_p = 0.9,
    .top_k = 40,
    .context_size = 4096,
    .num_threads = 4,
    .accelerator = NPIE_ACCELERATOR_GPU_VULKAN,
    .quantization = NPIE_QUANT_Q4_K_M,
    .use_mmap = true
};

npie_llm_load(ctx, &llm, "/opt/neuraparse/models/llama-3.2-3b-q4_k_m.gguf", &params);
```

### npie_llm_generate()

Generate text from a prompt with optional streaming callback.

```c
npie_status_t npie_llm_generate(npie_llm_t llm,
                                 const char* prompt,
                                 char* output,
                                 size_t output_size,
                                 npie_llm_token_callback_t callback,
                                 void* user_data);
```

**Token Callback:**
```c
// Called for each generated token (return false to stop)
typedef bool (*npie_llm_token_callback_t)(const char* token, void* user_data);
```

**Example:**
```c
// Streaming callback
bool on_token(const char* token, void* user_data) {
    printf("%s", token);
    fflush(stdout);
    return true; // continue generating
}

char output[4096];
npie_llm_generate(llm, "Explain quantum computing in one paragraph:",
                  output, sizeof(output), on_token, NULL);
```

### npie_llm_unload()

Unload LLM model and free resources.

```c
npie_status_t npie_llm_unload(npie_llm_t llm);
```

---

## Speech API

*New in NPIE v2.0.0* - Speech-to-text using whisper.cpp backend.

### npie_speech_load()

Load a Whisper model for speech recognition.

```c
npie_status_t npie_speech_load(npie_context_t ctx,
                                npie_speech_t* speech,
                                const char* model_path,
                                const npie_speech_params_t* params);
```

**Speech Parameters:**
```c
typedef struct {
    const char* language;    // Language code (e.g., "en", "auto")
    bool        translate;   // Translate to English
    uint32_t    num_threads; // CPU threads
    bool        use_gpu;     // GPU acceleration
} npie_speech_params_t;
```

### npie_speech_transcribe()

Transcribe audio data to text.

```c
npie_status_t npie_speech_transcribe(npie_speech_t speech,
                                      const float* audio_data,
                                      uint32_t num_samples,
                                      uint32_t sample_rate,
                                      char* output,
                                      size_t output_size);
```

**Example:**
```c
npie_speech_t speech;
npie_speech_params_t params = {
    .language = "en",
    .translate = false,
    .num_threads = 4,
    .use_gpu = true
};

npie_speech_load(ctx, &speech, "/opt/neuraparse/models/whisper-base.bin", &params);

// Transcribe 16kHz PCM audio
char transcript[4096];
npie_speech_transcribe(speech, audio_pcm, num_samples, 16000,
                       transcript, sizeof(transcript));
printf("Transcript: %s\n", transcript);

npie_speech_unload(speech);
```

### npie_speech_unload()

```c
npie_status_t npie_speech_unload(npie_speech_t speech);
```

---

## Quantum API

*New in NPIE v2.0.0* - Quantum circuit simulation using QuEST/Qulacs/Stim backends.

### Gate Types

```c
typedef enum {
    NPIE_GATE_H,        // Hadamard (superposition)
    NPIE_GATE_X,        // Pauli-X (NOT)
    NPIE_GATE_Y,        // Pauli-Y
    NPIE_GATE_Z,        // Pauli-Z (phase flip)
    NPIE_GATE_T,        // T gate (pi/8 phase)
    NPIE_GATE_S,        // S gate (pi/4 phase)
    NPIE_GATE_RX,       // X-axis rotation
    NPIE_GATE_RY,       // Y-axis rotation
    NPIE_GATE_RZ,       // Z-axis rotation
    NPIE_GATE_CNOT,     // Controlled-NOT (2-qubit)
    NPIE_GATE_CZ,       // Controlled-Z (2-qubit)
    NPIE_GATE_SWAP,     // SWAP qubits (2-qubit)
    NPIE_GATE_TOFFOLI,  // Toffoli CCNOT (3-qubit)
    NPIE_GATE_MEASURE,  // Measurement
} npie_gate_t;
```

### npie_quantum_create()

Create a quantum register.

```c
npie_status_t npie_quantum_create(npie_context_t ctx,
                                   npie_qureg_t* qureg,
                                   const npie_quantum_params_t* params);
```

**Quantum Parameters:**
```c
typedef struct {
    uint32_t num_qubits;    // Number of qubits (1-30)
    uint32_t num_shots;     // Measurement shots (default: 1024)
    bool     use_density;   // Use density matrix (for noise modeling)
    bool     use_gpu;       // GPU acceleration
} npie_quantum_params_t;
```

### npie_quantum_gate()

Apply a quantum gate to the register.

```c
npie_status_t npie_quantum_gate(npie_qureg_t qureg,
                                 npie_gate_t gate,
                                 int32_t target,
                                 int32_t control,
                                 double angle);
```

**Parameters:**
- `qureg` - Quantum register handle
- `gate` - Gate type
- `target` - Target qubit index
- `control` - Control qubit index (-1 for single-qubit gates)
- `angle` - Rotation angle in radians (for Rx, Ry, Rz gates)

**Example - Bell State:**
```c
npie_qureg_t qureg;
npie_quantum_params_t params = { .num_qubits = 2, .num_shots = 1024 };
npie_quantum_create(ctx, &qureg, &params);

// Create Bell state: |00⟩ → (|00⟩ + |11⟩)/√2
npie_quantum_gate(qureg, NPIE_GATE_H, 0, -1, 0.0);     // H on qubit 0
npie_quantum_gate(qureg, NPIE_GATE_CNOT, 1, 0, 0.0);   // CNOT(0→1)
```

### npie_quantum_measure()

Perform measurement on the quantum register.

```c
npie_status_t npie_quantum_measure(npie_qureg_t qureg,
                                    npie_measurement_t* results,
                                    uint32_t max_results,
                                    uint32_t* num_results);
```

**Measurement Result:**
```c
typedef struct {
    uint32_t state;          // Measured state as integer
    uint32_t count;          // Number of times measured
    double   probability;    // Measured probability
} npie_measurement_t;
```

### npie_quantum_get_statevector()

Get the full statevector (real and imaginary amplitudes).

```c
npie_status_t npie_quantum_get_statevector(npie_qureg_t qureg,
                                            double* real,
                                            double* imag);
```

### npie_quantum_destroy()

Destroy quantum register and free resources.

```c
npie_status_t npie_quantum_destroy(npie_qureg_t qureg);
```

**Complete Quantum Example:**
```c
#include <npie.h>
#include <stdio.h>

int main() {
    npie_context_t ctx;
    npie_init(&ctx, NULL);

    // Create 3-qubit register
    npie_qureg_t qureg;
    npie_quantum_params_t qparams = { .num_qubits = 3, .num_shots = 4096 };
    npie_quantum_create(ctx, &qureg, &qparams);

    // Build GHZ state: (|000⟩ + |111⟩)/√2
    npie_quantum_gate(qureg, NPIE_GATE_H, 0, -1, 0.0);
    npie_quantum_gate(qureg, NPIE_GATE_CNOT, 1, 0, 0.0);
    npie_quantum_gate(qureg, NPIE_GATE_CNOT, 2, 0, 0.0);

    // Measure
    npie_measurement_t results[8];
    uint32_t num_results;
    npie_quantum_measure(qureg, results, 8, &num_results);

    for (uint32_t i = 0; i < num_results; i++) {
        printf("|%03b⟩: %d counts (%.1f%%)\n",
               results[i].state, results[i].count,
               results[i].probability * 100.0);
    }
    // Expected: |000⟩ ~50%, |111⟩ ~50%

    npie_quantum_destroy(qureg);
    npie_shutdown(ctx);
    return 0;
}
```

---

## Hardware Detection

### npie_detect_accelerators()

Detect available hardware accelerators.

```c
npie_status_t npie_detect_accelerators(npie_context_t ctx,
                                       npie_accelerator_t* accelerators,
                                       uint32_t max_count,
                                       uint32_t* count);
```

---

## Data Types

### Backends (v2.0.0)

```c
typedef enum {
    NPIE_BACKEND_AUTO = 0,
    NPIE_BACKEND_LITERT,           // TensorFlow Lite
    NPIE_BACKEND_ONNXRUNTIME,      // ONNX Runtime
    NPIE_BACKEND_EMLEARN,          // Tiny ML
    NPIE_BACKEND_WASM,             // WebAssembly
    NPIE_BACKEND_OPENCV,           // OpenCV DNN
    NPIE_BACKEND_NCNN,             // NCNN mobile inference
    NPIE_BACKEND_EXECUTORCH,       // ExecuTorch (PyTorch)
    NPIE_BACKEND_OPENVINO,         // OpenVINO (Intel)
    NPIE_BACKEND_LLAMA_CPP,        // llama.cpp LLM
    NPIE_BACKEND_WHISPER_CPP,      // whisper.cpp Speech
    NPIE_BACKEND_STABLE_DIFFUSION, // stable-diffusion.cpp
    NPIE_BACKEND_MLC_LLM,         // MLC LLM
} npie_backend_t;
```

### Accelerators (v2.0.0)

```c
typedef enum {
    NPIE_ACCELERATOR_NONE = 0,
    NPIE_ACCELERATOR_AUTO,
    NPIE_ACCELERATOR_GPU,          // Generic GPU (OpenCL)
    NPIE_ACCELERATOR_NPU,          // Generic NPU
    NPIE_ACCELERATOR_TPU,          // Google Edge TPU
    NPIE_ACCELERATOR_DSP,          // Qualcomm DSP
    NPIE_ACCELERATOR_GPU_VULKAN,   // Vulkan GPU compute
    NPIE_ACCELERATOR_GPU_CUDA,     // NVIDIA CUDA
    NPIE_ACCELERATOR_GPU_METAL,    // Apple Metal
    NPIE_ACCELERATOR_NPU_HEXAGON,  // Qualcomm Hexagon
    NPIE_ACCELERATOR_NPU_ETHOS,    // ARM Ethos-U
    NPIE_ACCELERATOR_NPU_INTEL,    // Intel NPU (Meteor Lake+)
} npie_accelerator_t;
```

### Data Types (v2.0.0)

```c
typedef enum {
    NPIE_DTYPE_FLOAT32 = 0,
    NPIE_DTYPE_FLOAT16,
    NPIE_DTYPE_INT8,
    NPIE_DTYPE_UINT8,
    NPIE_DTYPE_INT32,
    NPIE_DTYPE_INT64,
    NPIE_DTYPE_BFLOAT16,          // NEW: BFloat16
    NPIE_DTYPE_FLOAT8_E4M3,       // NEW: FP8
    NPIE_DTYPE_INT4,              // NEW: 4-bit integer
} npie_dtype_t;
```

### Status Codes

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

### Metrics

```c
typedef struct {
    uint64_t inference_time_us;
    uint64_t preprocessing_time_us;
    uint64_t postprocessing_time_us;
    uint64_t total_time_us;
    size_t   memory_used_bytes;
    float    cpu_usage_percent;
    float    accelerator_usage_percent;
} npie_metrics_t;
```

---

## Thread Safety

- `npie_context_t` is thread-safe for read operations
- `npie_model_t` can be used from multiple threads simultaneously
- `npie_llm_t` should be accessed from a single thread
- `npie_qureg_t` is NOT thread-safe; use separate registers per thread
- Inference operations are thread-safe
- Use separate contexts for complete isolation

---

## Performance Tips

1. **Use Vulkan acceleration** for llama.cpp on GPUs without CUDA
2. **Use Q4_K_M quantization** for best quality/size tradeoff on LLMs
3. **Use IQ2_XXS quantization** for minimum model size (2-bit)
4. **Enable mmap** for faster model loading
5. **Set context_size** appropriately (smaller = faster, less memory)
6. **Reuse tensors** - allocate once, reuse for multiple inferences
7. **Batch processing** for traditional ML inference
8. **Set num_threads** to number of CPU performance cores

---

## Compile & Link

```bash
# Core inference
gcc -o app app.c -lnpie -lpthread -lm

# With LLM support
gcc -o llm_app llm_app.c -lnpie -lllama -lpthread -lm

# With quantum support
gcc -o quantum_app quantum_app.c -lnpie -lQuEST -lpthread -lm
```

---

## See Also

- [Getting Started Guide](getting_started.md)
- [Architecture Documentation](architecture_2025.md)
- [GUI Build Guide](GUI_BUILD.md)
- [Security Best Practices](security.md)

---

## MAVLink API

### mavlink_connection_create()
Create MAVLink connection over UART or UDP.
```c
#include "neuraos_mavlink.h"
mavlink_connection_t *mavlink_connection_create(const mavlink_config_t *cfg);
```

### mavlink_send_heartbeat()
```c
int mavlink_send_heartbeat(mavlink_connection_t *conn, uint8_t type, uint8_t autopilot, uint8_t mode);
```

### mavlink_request_data_stream()
```c
int mavlink_request_data_stream(mavlink_connection_t *conn, uint8_t stream_id, uint16_t rate_hz);
```

---

## Swarm API

### neura_swarm_create()
```c
#include "neuraos_swarm.h"
neura_swarm_ctx_t *neura_swarm_create(const neura_swarm_config_t *cfg);
```

### neura_swarm_set_formation()
```c
int neura_swarm_set_formation(neura_swarm_ctx_t *ctx, neura_formation_type_t type, float spacing_m, float heading_deg);
```
Formation types: NEURA_FORMATION_V, NEURA_FORMATION_LINE, NEURA_FORMATION_CIRCLE, NEURA_FORMATION_GRID

### neura_swarm_start_mission()
```c
int neura_swarm_start_mission(neura_swarm_ctx_t *ctx, const neura_swarm_mission_t *mission);
```

---

## Sensor Fusion API

### neura_sensor_fusion_create()
```c
#include "neuraos_sensors.h"
neura_sensor_fusion_ctx_t *neura_sensor_fusion_create(const neura_fusion_config_t *cfg);
```

### neura_sensor_fusion_update()
```c
int neura_sensor_fusion_update(neura_sensor_fusion_ctx_t *ctx, const neura_sensor_data_t *data);
```

### neura_sensor_fusion_get_state()
```c
int neura_sensor_fusion_get_state(neura_sensor_fusion_ctx_t *ctx, neura_fusion_state_t *state);
// state contains: position (lat/lon/alt), velocity (NED), attitude (quaternion), gyro/accel bias
```

---

## Network API

### neura_net_create()
```c
#include "neuraos_network.h"
neura_net_ctx_t *neura_net_create(const neura_net_config_t *cfg);
```

### neura_mesh_create() / neura_wg_create()
WiFi Mesh and WireGuard VPN tunnel APIs.

---

## Security API

### neura_secure_boot_verify()
```c
#include "neuraos_security.h"
int neura_secure_boot_verify(const char *image_path, const char *keyring_path);
```

### neura_model_encrypt() / neura_model_decrypt()
```c
int neura_model_encrypt(const char *input, const char *output, const uint8_t *key, size_t key_len);
int neura_model_decrypt(const char *input, const char *output, const uint8_t *key, size_t key_len);
```
