# NeuralOS Init System (NPI)

cmake_minimum_required(VERSION 3.16)

# NPI Init executable
add_executable(npi
    npi_init.c
)

target_link_libraries(npi
    neura_common
)

# Install as /sbin/init
install(TARGETS npi
    RUNTIME DESTINATION sbin
)

# Create symlink for init
install(CODE "
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_INSTALL_PREFIX}/sbin/npi
        ${CMAKE_INSTALL_PREFIX}/sbin/init
    )
")

