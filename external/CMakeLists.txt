include(FetchContent)

message("Fetching Nlohmann JSON...")
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz)
FetchContent_MakeAvailable(json)

find_package(Threads)
find_package(HailoRT REQUIRED)

add_library(hailo-apps-cpp-common)
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
    ${OpenCV_LIBS}
    nlohmann_json::nlohmann_json)
