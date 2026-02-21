#include "object_detection_config.h"

ObjectDetectionConfig::ObjectDetectionConfig(toml::v3::ex::parse_result config)
{
    src_url_ = config["object_detection"]["src_url"].value_or(std::string("127.0.0.1"));
    mtx_url_ = config["object_detection"]["mtx_url"].value_or(std::string("127.0.0.1"));
    log_level_ = config["object_detection"]["log_level"].value_or(std::string("error"));
    
    Logger logger("ObjectDetectionConfig", "info");

    logger.info("Loading Config:");
    logger.info(" -- Source URL: {}", src_url_);
    logger.info(" -- MediaMTX URL: {}", mtx_url_);
    logger.info(" -- Log Level: {}", log_level_);
}
