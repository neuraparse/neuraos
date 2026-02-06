# NeuralOS Development Tools

cmake_minimum_required(VERSION 3.16)

# Python tools - just install them
install(PROGRAMS
    npconvert
    npprofiler
    npsim
    npie-cli
    DESTINATION bin
)

# Make tools executable
install(CODE "
    execute_process(
        COMMAND chmod +x
            ${CMAKE_INSTALL_PREFIX}/bin/npconvert
            ${CMAKE_INSTALL_PREFIX}/bin/npprofiler
            ${CMAKE_INSTALL_PREFIX}/bin/npsim
            ${CMAKE_INSTALL_PREFIX}/bin/npie-cli
    )
")

# npie-bench: Multi-backend benchmark tool (C)
add_executable(npie-bench npie_bench/npie_bench.c)
target_link_libraries(npie-bench PRIVATE npie m)
target_include_directories(npie-bench PRIVATE ${PROJECT_SOURCE_DIR}/src/npie/api)
install(TARGETS npie-bench DESTINATION bin)

