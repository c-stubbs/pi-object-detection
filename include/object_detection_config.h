#pragma once

#include <toml++/toml.h>

class ObjectDetectionConfig {
    public:
        ObjectDetectionConfig(toml::v3::ex::parse_result config);
        std::string src_url_;
        double src_width_;
        double src_height_;
        std::string mtx_url_;
        std::string model_name_;
        std::vector<std::string> valid_objects_;
        size_t batch_size_;
        double target_fps_;
        bool restream_grayscale_;
        std::string log_level_;

    private:
        template <typename T>
        std::vector<T> readToVector(toml::array* arr)
        {
            std::vector<T> vector;
            vector.reserve(arr->size());
            for (const auto& v : *arr)
                vector.push_back(*v.value<std::string>());
            return vector;
        };
};
