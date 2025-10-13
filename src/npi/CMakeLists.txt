# NeuralOS Init System (NPI)

cmake_minimum_required(VERSION 3.16)

# NPI Init executable
add_executable(npi
    npi_init.c
)

# Optionally install as init (symlink). Default OFF to avoid conflicts with BusyBox init.
option(NPI_INSTALL_AS_INIT "Install npi as /sbin/init symlink" OFF)

# Install binary under /usr/sbin (CMAKE_INSTALL_PREFIX is usually /usr in Buildroot)
install(TARGETS npi
    RUNTIME DESTINATION sbin
)

# Create symlink for init only if explicitly enabled
if(NPI_INSTALL_AS_INIT)
    install(CODE "
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E create_symlink
            ${CMAKE_INSTALL_PREFIX}/sbin/npi
            ${CMAKE_INSTALL_PREFIX}/sbin/init
        )
    ")
endif()

