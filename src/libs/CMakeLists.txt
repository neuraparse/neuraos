# NeuralOS Base System Libraries

cmake_minimum_required(VERSION 3.16)

# Find required packages
find_package(CURL)
find_package(OpenSSL)

# libneura_common - Common utilities
add_library(neura_common SHARED
    libneura_common.c
)

target_include_directories(neura_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(neura_common PROPERTIES
    VERSION 1.0.0
    SOVERSION 1
    PUBLIC_HEADER libneura_common.h
)

install(TARGETS neura_common
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include/neuraos
)

# libneura_ota - OTA Update System
if(CURL_FOUND AND OPENSSL_FOUND)
    add_library(neura_ota SHARED
        libneura_ota.c
    )

    target_include_directories(neura_ota PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CURL_INCLUDE_DIRS}
        ${OPENSSL_INCLUDE_DIR}
    )

    target_link_libraries(neura_ota
        neura_common
        ${CURL_LIBRARIES}
        ${OPENSSL_LIBRARIES}
    )

    set_target_properties(neura_ota PROPERTIES
        VERSION 1.0.0
        SOVERSION 1
        PUBLIC_HEADER libneura_ota.h
    )

    install(TARGETS neura_ota
        LIBRARY DESTINATION lib
        PUBLIC_HEADER DESTINATION include/neuraos
    )
else()
    message(WARNING "CURL or OpenSSL not found, OTA library will not be built")
endif()

