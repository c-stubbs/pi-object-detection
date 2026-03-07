#include "object_detection_config.h"
#include "logger.h"

ObjectDetectionConfig::ObjectDetectionConfig(toml::v3::ex::parse_result config)
{
    src_url_ = config["object_detection"]["src_url"].value_or(std::string("127.0.0.1"));
    src_width_ = config["object_detection"]["src_width"].value_or(720.0);
    src_height_ = config["object_detection"]["src_height"].value_or(480.0);
    mtx_url_ = config["object_detection"]["mtx_url"].value_or(std::string("127.0.0.1"));
    model_name_ = config["object_detection"]["model_name"].value_or(std::string("yolov8n"));
    batch_size_ = config["object_detection"]["batch_size"].value_or(1);
    target_fps_ = config["object_detection"]["target_fps"].value_or(30.0);
    restream_grayscale_ = config["object_detection"]["restream_grayscale"].value_or(true);
    log_level_ = config["object_detection"]["log_level"].value_or(std::string("error"));
    
    Logger logger("ObjectDetectionConfig", "info");

    logger.info("Loading Config:");
    logger.info(" -- Source URL: {}", src_url_);
    logger.info(" -- Source Width: {}", src_width_);
    logger.info(" -- Source Height: {}", src_height_);
    logger.info(" -- MediaMTX URL: {}", mtx_url_);
    logger.info(" -- Model Name: {}", model_name_);
    logger.info(" -- Batch Size: {}", batch_size_);
    logger.info(" -- Target FPS: {}", target_fps_);
    logger.info(" -- Restream in Grayscale: {}", restream_grayscale_);
    logger.info(" -- Log Level: {}", log_level_);
}
