#include <iostream>
#include <toml++/toml.h>
#include "object_detection_config.h"
#include "object_detection.h"
#include "logger.h"

int main()
{
    Logger logger("Main", "info");
    logger.info("Starting...");

    std::string config_dir = CONFIG_DIR;
    std::string config_path = config_dir + std::string("config.toml");
    logger.info("Loading config file: {}", config_path);
    
    auto config = toml::parse_file(config_path);
    ObjectDetectionConfig object_detection_config(config);

    ObjectDetection object_detection(object_detection_config);
    object_detection.run();

}