cmake_minimum_required(VERSION 3.20)

project(neuraos-dashboard VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Find Qt5
find_package(Qt5 COMPONENTS Core Widgets Charts QUIET)

if(Qt5_FOUND)
    message(STATUS "Qt5 found - building AI Dashboard")
    
    # Dashboard executable
    add_executable(neuraos-dashboard
        main.cpp
    )
    
    target_link_libraries(neuraos-dashboard
        Qt5::Core
        Qt5::Widgets
        Qt5::Charts
        npie
    )
    
    target_include_directories(neuraos-dashboard PRIVATE
        ${CMAKE_SOURCE_DIR}/src/npie/include
    )
    
    # Install
    install(TARGETS neuraos-dashboard
        RUNTIME DESTINATION bin
    )
    
    # Desktop entry for Weston
    install(FILES neuraos-dashboard.desktop
        DESTINATION share/applications
    )
else()
    message(STATUS "Qt5 not found - skipping AI Dashboard build")
endif()

