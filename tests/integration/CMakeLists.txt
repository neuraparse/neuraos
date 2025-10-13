# Integration Tests

cmake_minimum_required(VERSION 3.16)

# NPIE Integration Tests
add_executable(test_npie_integration
    test_npie_integration.c
)

target_link_libraries(test_npie_integration
    Threads::Threads
)

add_test(NAME npie_integration_tests
    COMMAND test_npie_integration
)

# Driver Integration Tests
add_executable(test_drivers
    test_drivers.c
)

target_link_libraries(test_drivers
    npu_driver
    gpu_accel
    Threads::Threads
)

add_test(NAME driver_tests
    COMMAND test_drivers
)

