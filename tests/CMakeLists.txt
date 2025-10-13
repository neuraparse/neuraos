# NeuralOS Test Suite

cmake_minimum_required(VERSION 3.16)

# Unit tests
add_subdirectory(unit)

# Integration tests
add_subdirectory(integration)

# Benchmarks
add_subdirectory(benchmarks)

# Add test target
add_custom_target(check
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS
        test_npie_core
        test_npie_integration
        test_drivers
    COMMENT "Running all tests"
)

