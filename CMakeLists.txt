#
# NeuralOS Root CMake Configuration
# Version: 1.0.0-alpha
# Updated: October 2025
#

cmake_minimum_required(VERSION 3.20)

project(NeuralOS
    VERSION 1.0.0
    DESCRIPTION "AI-Native Embedded Operating System"
    LANGUAGES C CXX ASM
)

# Set C/C++ standards
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# Options
option(BUILD_NPIE "Build NeuraParse Inference Engine" ON)
option(BUILD_NPI "Build NeuralOS Init System" ON)
option(BUILD_TOOLS "Build development tools" ON)
option(BUILD_EXAMPLES "Build example applications" ON)
option(BUILD_TESTS "Build test suite" ON)
option(ENABLE_LITERT "Enable LiteRT (TensorFlow Lite) support" ON)
option(ENABLE_ONNXRUNTIME "Enable ONNX Runtime support" ON)
option(ENABLE_EMLEARN "Enable emlearn support" ON)
option(ENABLE_OPENCV "Enable OpenCV support" ON)
option(ENABLE_WASMEDGE "Enable WebAssembly support" ON)
option(ENABLE_GPU_ACCELERATION "Enable GPU acceleration" ON)
option(ENABLE_NPU_SUPPORT "Enable NPU support" ON)
option(ENABLE_PROFILING "Enable profiling support" OFF)
option(ENABLE_SANITIZERS "Enable sanitizers (ASan, UBSan)" OFF)

# Platform detection
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(NEURAOS_ARCH "arm64")
    set(NEURAOS_ARM_NEON ON)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7")
    set(NEURAOS_ARCH "arm32")
    set(NEURAOS_ARM_NEON ON)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
    set(NEURAOS_ARCH "x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64")
    set(NEURAOS_ARCH "riscv64")
else()
    message(WARNING "Unknown architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    set(NEURAOS_ARCH "unknown")
endif()

message(STATUS "NeuralOS Architecture: ${NEURAOS_ARCH}")

# Compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")

# Optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -DNDEBUG")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
    
    # ARM NEON optimizations
    if(NEURAOS_ARM_NEON)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfpu=neon -mfloat-abi=hard")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon -mfloat-abi=hard")
    endif()
    
    # x86_64 optimizations
    if(NEURAOS_ARCH STREQUAL "x86_64")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -mtune=native")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native")
    endif()
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -g3 -ggdb")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g3 -ggdb")
endif()

# Sanitizers
if(ENABLE_SANITIZERS)
    set(SANITIZER_FLAGS "-fsanitize=address,undefined -fno-omit-frame-pointer")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SANITIZER_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SANITIZER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${SANITIZER_FLAGS}")
endif()

# Profiling
if(ENABLE_PROFILING)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pg")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/npie/api
    ${CMAKE_BINARY_DIR}/include
)

# Find required packages
find_package(Threads REQUIRED)

# Find optional packages
if(ENABLE_LITERT)
    find_package(TensorFlowLite)
    if(TensorFlowLite_FOUND)
        message(STATUS "LiteRT (TensorFlow Lite) found")
        add_definitions(-DNEURAOS_ENABLE_LITERT)
    else()
        message(WARNING "LiteRT not found, disabling support")
        set(ENABLE_LITERT OFF)
    endif()
endif()

if(ENABLE_ONNXRUNTIME)
    find_package(ONNXRuntime)
    if(ONNXRuntime_FOUND)
        message(STATUS "ONNX Runtime found")
        add_definitions(-DNEURAOS_ENABLE_ONNXRUNTIME)
    else()
        message(WARNING "ONNX Runtime not found, disabling support")
        set(ENABLE_ONNXRUNTIME OFF)
    endif()
endif()

if(ENABLE_OPENCV)
    find_package(OpenCV COMPONENTS core imgproc imgcodecs videoio)
    if(OpenCV_FOUND)
        message(STATUS "OpenCV found: ${OpenCV_VERSION}")
        add_definitions(-DNEURAOS_ENABLE_OPENCV)
    else()
        message(WARNING "OpenCV not found, disabling support")
        set(ENABLE_OPENCV OFF)
    endif()
endif()

if(ENABLE_WASMEDGE)
    find_package(WasmEdge)
    if(WasmEdge_FOUND)
        message(STATUS "WasmEdge found")
        add_definitions(-DNEURAOS_ENABLE_WASMEDGE)
    else()
        message(WARNING "WasmEdge not found, disabling support")
        set(ENABLE_WASMEDGE OFF)
    endif()
endif()

# Configuration header
configure_file(
    ${CMAKE_SOURCE_DIR}/src/neuraos_config.h.in
    ${CMAKE_BINARY_DIR}/include/neuraos_config.h
)

# Subdirectories
# Base system libraries
add_subdirectory(src/libs)

# Hardware drivers
add_subdirectory(src/drivers)

if(BUILD_NPIE)
    add_subdirectory(src/npie)
endif()

if(BUILD_NPI)
    add_subdirectory(src/npi)
endif()

if(BUILD_TOOLS)
    add_subdirectory(tools)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Install targets
install(DIRECTORY ${CMAKE_BINARY_DIR}/include/
    DESTINATION include/neuraos
    FILES_MATCHING PATTERN "*.h"
)

# Install documentation (guard LICENSE if missing)
if(EXISTS "${CMAKE_SOURCE_DIR}/LICENSE")
    install(FILES
        ${CMAKE_SOURCE_DIR}/README.md
        ${CMAKE_SOURCE_DIR}/LICENSE
        DESTINATION share/doc/neuraos
    )
else()
    install(FILES
        ${CMAKE_SOURCE_DIR}/README.md
        DESTINATION share/doc/neuraos
    )
endif()

# Package configuration (optional; only if template exists)
if(EXISTS "${CMAKE_SOURCE_DIR}/cmake/NeuralOSConfig.cmake.in")
    include(CMakePackageConfigHelpers)

    configure_package_config_file(
        ${CMAKE_SOURCE_DIR}/cmake/NeuralOSConfig.cmake.in
        ${CMAKE_BINARY_DIR}/NeuralOSConfig.cmake
        INSTALL_DESTINATION lib/cmake/NeuralOS
    )

    write_basic_package_version_file(
        ${CMAKE_BINARY_DIR}/NeuralOSConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )

    install(FILES
        ${CMAKE_BINARY_DIR}/NeuralOSConfig.cmake
        ${CMAKE_BINARY_DIR}/NeuralOSConfigVersion.cmake
        DESTINATION lib/cmake/NeuralOS
    )
endif()

# Uninstall target (optional; only if template exists)
if(EXISTS "${CMAKE_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in")
    if(NOT TARGET uninstall)
        configure_file(
            ${CMAKE_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in
            ${CMAKE_BINARY_DIR}/cmake_uninstall.cmake
            IMMEDIATE @ONLY
        )

        add_custom_target(uninstall
            COMMAND ${CMAKE_COMMAND} -P ${CMAKE_BINARY_DIR}/cmake_uninstall.cmake
        )
    endif()
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "NeuralOS Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Architecture: ${NEURAOS_ARCH}")
message(STATUS "  C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "  C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
message(STATUS "Components:")
message(STATUS "  NPIE: ${BUILD_NPIE}")
message(STATUS "  NPI Init: ${BUILD_NPI}")
message(STATUS "  Tools: ${BUILD_TOOLS}")
message(STATUS "  Examples: ${BUILD_EXAMPLES}")
message(STATUS "  Tests: ${BUILD_TESTS}")
message(STATUS "")
message(STATUS "AI Frameworks:")
message(STATUS "  LiteRT: ${ENABLE_LITERT}")
message(STATUS "  ONNX Runtime: ${ENABLE_ONNXRUNTIME}")
message(STATUS "  emlearn: ${ENABLE_EMLEARN}")
message(STATUS "  OpenCV: ${ENABLE_OPENCV}")
message(STATUS "  WasmEdge: ${ENABLE_WASMEDGE}")
message(STATUS "")
message(STATUS "Hardware Acceleration:")
message(STATUS "  GPU: ${ENABLE_GPU_ACCELERATION}")
message(STATUS "  NPU: ${ENABLE_NPU_SUPPORT}")
message(STATUS "")
message(STATUS "Development:")
message(STATUS "  Profiling: ${ENABLE_PROFILING}")
message(STATUS "  Sanitizers: ${ENABLE_SANITIZERS}")
message(STATUS "")

