#pragma once

#include "object_detection_config.h"
#include "logger.h"

class ObjectDetection{
    public:
        ObjectDetection(ObjectDetectionConfig config);
        void run();
    
    private:
        // Config
        std::string src_url_;
        std::string mtx_url_;
        int batch_size_;
        std::string model_name_;
        int target_fps_;    

        Logger logger_;
};
