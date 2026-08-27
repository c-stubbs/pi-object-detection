cmake_minimum_required(VERSION 3.31)
project(object-detection)

include(GNUInstallDirs)

set(CONFIG_DIR
    "${CMAKE_INSTALL_PREFIX}/config/")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_C_COMPILER=clang)
set(CMAKE_CXX_COMPILER=clang++)

# find requried packages
find_package(OpenCV REQUIRED)
find_package(tomlplusplus CONFIG REQUIRED)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED 
    gstreamer-1.0
    gstreamer-app-1.0
    gstreamer-rtsp-server-1.0)

# Add main executable
add_executable(${PROJECT_NAME} 
    src/main.cpp)

target_link_libraries(${PROJECT_NAME} 
    PRIVATE
    tomlplusplus::tomlplusplus
    object_detection
    object_detection_config
    logger
    api)

target_compile_definitions(${PROJECT_NAME} 
    PRIVATE
    CONFIG_DIR="${CONFIG_DIR}")

# Add Object detection config lib
# TODO: need to move away from centralized CMake file. Things like the 
#       object_detection_config library would be more natural if the CMakeList
#       is in a subdir
add_library(object_detection_config)

target_sources(object_detection_config
    PRIVATE
    src/object_detection/object_detection_config.cpp)

target_include_directories(object_detection_config 
    PUBLIC 
    ./include/object_detection)

target_link_libraries(object_detection_config 
    PRIVATE
    tomlplusplus::tomlplusplus
    logger
    async_queue
)

# Add Object detection lib
add_library(object_detection)

target_sources(object_detection
    PRIVATE
    src/object_detection/object_detection.cpp)

target_include_directories(object_detection 
    PUBLIC 
    include/object_detection
    ${GST_INCLUDE_DIRS}
    )

target_link_libraries(object_detection
    PUBLIC
    ${OpenCV_LIBS}
    object_detection_config
    logger
    ${GST_LIBRARIES}
    hailo-apps-cpp-common)

# Add logger library
add_library(logger INTERFACE)

target_include_directories(logger 
    INTERFACE 
    ./include/logger)

target_link_libraries(logger
    INTERFACE
    fmt)

# Add async queue library
add_library(async_queue INTERFACE)

target_include_directories(async_queue 
    INTERFACE 
    ./include/utils)

# Add API library
add_library(api)

target_sources(api
    PRIVATE
    src/api/api.cpp)

target_include_directories(api 
    PUBLIC 
    ./include/api)

target_link_libraries(api
    PUBLIC
    cpp-httplib)

add_subdirectory(external)

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

# Install hailo-apps configs
install(
    DIRECTORY external/hailo-apps/hailo_apps/config/
    DESTINATION config
)
