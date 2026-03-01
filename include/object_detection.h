#pragma once

#include <opencv2/opencv.hpp>

#include "object_detection_config.h"
#include "logger.h"
#include "async_queue.h"

class ObjectDetection{
    public:
        ObjectDetection(ObjectDetectionConfig config);
        void run();
        using BatchQueue = AsyncQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>;
    
    private:
        // Config
        std::string src_url_;
        std::string mtx_url_;
        int batch_size_;
        std::string model_name_;
        int target_fps_;    

        Logger logger_;
        int model_input_width_;
        int model_input_height_;

        void preprocessAsync(cv::VideoCapture& capture, std::shared_ptr<BatchQueue> preprocessed_batch_queue);
        void reformatFrames(const std::vector<cv::Mat>& org_frames, std::vector<cv::Mat>& preprocessed_frames);
        void inferAsync();
        void postprocessAsync();
};
