#include "inference_config.h"

InferenceConfig::InferenceConfig(toml::v3::ex::parse_result config)
{
    model_name_ = config["inference"]["model_name"].value_or(std::string("yolov8n"));
    batch_size_ = config["inference"]["batch_size"].value_or(1);
    target_fps_ = config["inference"]["target_fps"].value_or(30);
    log_level_ = config["inference"]["log_level"].value_or(std::string("error"));
    
    Logger logger("InferenceConfig", "info");

    logger.info("Loading Config:");
    logger.info(" -- Model Name: {}", model_name_);
    logger.info(" -- Batch Size: {}", batch_size_);
    logger.info(" -- Target FPS: {}", target_fps_);
    logger.info(" -- Log Level: {}", log_level_);
}
