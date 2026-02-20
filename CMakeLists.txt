cmake_minimum_required(VERSION 3.16)
project(object-detection)

include(GNUInstallDirs)

set(CONFIG_DIR
    "${CMAKE_INSTALL_PREFIX}/config/")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_C_COMPILER=clang)
set(CMAKE_CXX_COMPILER=clang++)

# find requried packages
find_package(Threads)
find_package(HailoRT REQUIRED)
find_package(OpenCV REQUIRED)
find_package(tomlplusplus CONFIG REQUIRED)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED 
    gstreamer-1.0
    gstreamer-app-1.0
    gstreamer-rtsp-server-1.0)

# add main executable
add_executable(${PROJECT_NAME} 
    src/main.cpp
    src/object_detection.cpp)
target_include_directories(${PROJECT_NAME} 
    PUBLIC 
    ./include
    PRIVATE
    ${GST_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} 
    PRIVATE
    HailoRT::libhailort 
    Threads::Threads
    tomlplusplus::tomlplusplus
    ${OpenCV_LIBS}
    object_detection_config
    logger
    ${GST_LIBRARIES})
target_compile_definitions(${PROJECT_NAME} PRIVATE
    CONFIG_DIR="${CONFIG_DIR}")

# Add Object detection lib
add_library(object_detection_config)
target_sources(object_detection_config
    PRIVATE
    src/object_detection_config.cpp)
target_include_directories(object_detection_config PUBLIC ./include)
target_link_libraries(object_detection_config 
    PRIVATE
    tomlplusplus::tomlplusplus
    logger)

# Add logger library
add_library(logger INTERFACE)
target_include_directories(logger INTERFACE ./include)
target_link_libraries(logger
    INTERFACE
    fmt)

# Install bins
install(
    TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Install configs
install(
    DIRECTORY config/
    DESTINATION config
)
