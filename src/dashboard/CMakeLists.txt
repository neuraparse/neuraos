cmake_minimum_required(VERSION 3.20)

project(neuraos-dashboard VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

# Find Qt5 QML/Quick components
find_package(Qt5 COMPONENTS Core Quick Qml QuickControls2 Svg QUIET)

if(Qt5_FOUND AND Qt5Quick_FOUND)
    message(STATUS "Qt5 QML found - building NeuralOS Desktop Shell")

    # Backend sources
    set(BACKEND_SOURCES
        backend/systeminfo.cpp
        backend/npiebridge.cpp
        backend/npumonitor.cpp
        backend/processmanager.cpp
        backend/networkmanager.cpp
        backend/settingsmanager.cpp
    )

    set(BACKEND_HEADERS
        backend/systeminfo.h
        backend/npiebridge.h
        backend/npumonitor.h
        backend/processmanager.h
        backend/networkmanager.h
        backend/settingsmanager.h
    )

    # Dashboard executable
    add_executable(neuraos-dashboard
        main.cpp
        ${BACKEND_SOURCES}
        ${BACKEND_HEADERS}
        resources.qrc
    )

    target_link_libraries(neuraos-dashboard
        Qt5::Core
        Qt5::Quick
        Qt5::Qml
        Qt5::QuickControls2
        Qt5::Svg
        npie
        npu_driver
    )

    target_include_directories(neuraos-dashboard PRIVATE
        ${CMAKE_SOURCE_DIR}/src/npie/api
        ${CMAKE_SOURCE_DIR}/src/drivers/npu
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    # Install
    install(TARGETS neuraos-dashboard RUNTIME DESTINATION bin)
    install(FILES neuraos-dashboard.desktop DESTINATION share/applications)
else()
    message(STATUS "Qt5 QML not found - skipping NeuralOS Desktop Shell build")
endif()
