# Unit Tests

cmake_minimum_required(VERSION 3.16)

# NPIE Core Unit Tests
add_executable(test_npie_core
    test_npie_core.c
)

target_link_libraries(test_npie_core
    Threads::Threads
)

add_test(NAME npie_core_tests
    COMMAND test_npie_core
)

