# Image Classification Example

cmake_minimum_required(VERSION 3.16)

add_executable(image_classifier
    image_classifier.c
)

target_link_libraries(image_classifier
    PRIVATE npie
    PRIVATE Threads::Threads
)

# Install example
install(TARGETS image_classifier
    RUNTIME DESTINATION share/neuraos/examples
)

# Install README
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/README.md)
    install(FILES README.md
        DESTINATION share/neuraos/examples/image_classification
    )
endif()

