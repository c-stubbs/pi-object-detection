#pragma once

#include <toml++/toml.h>
#include "logger.h"

class ObjectDetectionConfig {
    public:
        ObjectDetectionConfig(toml::v3::ex::parse_result config);
        std::string src_url_;
        std::string mtx_url_;
        std::string model_name_;
        size_t batch_size_;
        double target_fps_;
        std::string log_level_;

};
