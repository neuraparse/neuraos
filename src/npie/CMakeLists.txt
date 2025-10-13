#
# NeuraParse Inference Engine (NPIE) CMake Configuration
#

cmake_minimum_required(VERSION 3.20)

project(NPIE
    VERSION 1.0.0
    DESCRIPTION "NeuraParse Inference Engine"
    LANGUAGES C CXX
)

# Options
option(ENABLE_LITERT "Enable LiteRT backend" ON)
option(ENABLE_ONNXRUNTIME "Enable ONNX Runtime backend" ON)
option(ENABLE_EMLEARN "Enable emlearn backend" ON)
option(ENABLE_OPENCV "Enable OpenCV support" ON)
option(ENABLE_WASMEDGE "Enable WebAssembly support" ON)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

# C/C++ standards
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/api
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/hal
    ${CMAKE_CURRENT_SOURCE_DIR}/scheduler
    ${CMAKE_CURRENT_SOURCE_DIR}/memory
)

# Source files
set(NPIE_CORE_SOURCES
    core/npie_core.c
    core/npie_inference.c
    core/npie_model.c
    hal/npie_hal.c
    scheduler/npie_scheduler.c
    memory/npie_memory.c
)

# Create internal header
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/core/npie_internal.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/npie_internal.h
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Backend sources
if(ENABLE_LITERT)
    list(APPEND NPIE_CORE_SOURCES core/backends/npie_litert.cpp)
    add_definitions(-DNEURAOS_ENABLE_LITERT)
endif()

if(ENABLE_ONNXRUNTIME)
    list(APPEND NPIE_CORE_SOURCES core/backends/npie_onnx.cpp)
    add_definitions(-DNEURAOS_ENABLE_ONNXRUNTIME)
endif()

if(ENABLE_EMLEARN)
    list(APPEND NPIE_CORE_SOURCES core/backends/npie_emlearn.c)
    add_definitions(-DNEURAOS_ENABLE_EMLEARN)
endif()

if(ENABLE_WASMEDGE)
    list(APPEND NPIE_CORE_SOURCES core/backends/npie_wasm.cpp)
    add_definitions(-DNEURAOS_ENABLE_WASMEDGE)
endif()

# Create NPIE library
add_library(npie ${NPIE_CORE_SOURCES})

# Link dependencies
target_link_libraries(npie
    pthread
    m
    dl
)

# Find and link backends
if(ENABLE_LITERT)
    find_package(TensorFlowLite)
    if(TensorFlowLite_FOUND)
        target_link_libraries(npie ${TensorFlowLite_LIBRARIES})
        target_include_directories(npie PRIVATE ${TensorFlowLite_INCLUDE_DIRS})
    endif()
endif()

if(ENABLE_ONNXRUNTIME)
    find_package(ONNXRuntime)
    if(ONNXRuntime_FOUND)
        target_link_libraries(npie ${ONNXRuntime_LIBRARIES})
        target_include_directories(npie PRIVATE ${ONNXRuntime_INCLUDE_DIRS})
    endif()
endif()

if(ENABLE_OPENCV)
    find_package(OpenCV COMPONENTS core imgproc imgcodecs)
    if(OpenCV_FOUND)
        target_link_libraries(npie ${OpenCV_LIBRARIES})
        target_include_directories(npie PRIVATE ${OpenCV_INCLUDE_DIRS})
        add_definitions(-DNEURAOS_ENABLE_OPENCV)
    endif()
endif()

if(ENABLE_WASMEDGE)
    find_package(WasmEdge)
    if(WasmEdge_FOUND)
        target_link_libraries(npie ${WasmEdge_LIBRARIES})
        target_include_directories(npie PRIVATE ${WasmEdge_INCLUDE_DIRS})
    endif()
endif()

# Set library properties
set_target_properties(npie PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER api/npie.h
)

# Install library
install(TARGETS npie
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)

# Install additional headers
install(FILES
    api/npie.h
    DESTINATION include/npie
)

# Create pkg-config file
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/npie.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/npie.pc
    @ONLY
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/npie.pc
    DESTINATION lib/pkgconfig
)

