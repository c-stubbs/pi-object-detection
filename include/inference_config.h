#pragma once

#include <toml++/toml.h>
#include "logger.h"

class InferenceConfig {
    public:
        InferenceConfig(toml::v3::ex::parse_result config);
        std::string model_name_;
        int batch_size_;
        int target_fps_;
        std::string log_level_;

};
