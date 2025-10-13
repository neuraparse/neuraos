# NeuralOS Hardware Drivers

cmake_minimum_required(VERSION 3.16)

# NPU Driver
add_library(npu_driver SHARED
    npu/npu_driver.c
)

target_include_directories(npu_driver PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/npu
)

set_target_properties(npu_driver PROPERTIES
    VERSION 1.0.0
    SOVERSION 1
)

install(TARGETS npu_driver
    LIBRARY DESTINATION lib
)

install(FILES npu/npu_driver.h
    DESTINATION include/neuraos/drivers
)

# GPU Accelerator
add_library(gpu_accel SHARED
    accelerators/gpu_accel.c
)

target_include_directories(gpu_accel PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/accelerators
)

set_target_properties(gpu_accel PROPERTIES
    VERSION 1.0.0
    SOVERSION 1
)

install(TARGETS gpu_accel
    LIBRARY DESTINATION lib
)

install(FILES accelerators/gpu_accel.h
    DESTINATION include/neuraos/drivers
)

