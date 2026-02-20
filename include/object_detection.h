#pragma once

#include "object_detection_config.h"
#include "logger.h"

class ObjectDetection{
    public:
        ObjectDetection(ObjectDetectionConfig config);
        void run();
    private:
        std::string src_url_;
        Logger logger_;
};