# NeuralOS Example Applications

cmake_minimum_required(VERSION 3.16)

# Image Classification Example
add_subdirectory(image_classification)

# Object Detection Example
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/object_detection/CMakeLists.txt)
    add_subdirectory(object_detection)
endif()

# Voice Recognition Example
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/voice_recognition/CMakeLists.txt)
    add_subdirectory(voice_recognition)
endif()

# WebAssembly Inference Example
if(ENABLE_WASMEDGE AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/wasm_inference/CMakeLists.txt)
    add_subdirectory(wasm_inference)
endif()

