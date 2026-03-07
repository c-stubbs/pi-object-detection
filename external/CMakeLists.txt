include(FetchContent)

message("Fetching Nlohmann JSON...")
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz)
FetchContent_MakeAvailable(json)

find_package(Threads)
find_package(HailoRT REQUIRED)

add_library(hailo-apps-cpp-common)
target_compile_definitions(hailo-apps-cpp-common PRIVATE HEADLESS_OPENCV)
target_compile_options(hailo-apps-cpp-common PRIVATE -include ${CMAKE_CURRENT_LIST_DIR}/overrides/headless_opencv.h)
target_sources(hailo-apps-cpp-common
    PRIVATE
    hailo-apps/hailo_apps/cpp/common/hailo_infer.cpp
    hailo-apps/hailo_apps/cpp/common/toolbox.cpp
    hailo-apps/hailo_apps/cpp/object_detection/utils/utils.cpp)
target_include_directories(hailo-apps-cpp-common
    PUBLIC
    hailo-apps/
    hailo-apps/hailo_apps/cpp/common/
)
target_link_libraries(hailo-apps-cpp-common
    PRIVATE
    HailoRT::libhailort
    Threads::Threads
    headless_opencv
    ${OpenCV_LIBS}
    nlohmann_json::nlohmann_json)

add_library(headless_opencv INTERFACE)
target_include_directories(headless_opencv
    INTERFACE
    overrides/
    ${OpenCV_INCLUDE_DIRS}
)
