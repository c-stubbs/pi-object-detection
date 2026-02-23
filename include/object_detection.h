#pragma once

#include <opencv2/opencv.hpp>

#include "object_detection_config.h"
#include "logger.h"
#include "hailo/hailort.hpp"
#include "async_queue.h"

//using namespace hailort;

class ObjectDetection{
    public:
        ObjectDetection(ObjectDetectionConfig config);
        void run();
        typedef AsyncQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>> BatchQueue;
    
    private:
        // Config
        std::string src_url_;
        std::string mtx_url_;
        int batch_size_;
        std::string model_name_;
        int target_fps_;    

        Logger logger_;

        // Inference
        std::unique_ptr<hailort::VDevice> vdevice_;
        std::shared_ptr<hailort::InferModel> infer_model_;
        hailort::ConfiguredInferModel configured_infer_model_;
        std::vector<hailort::ConfiguredInferModel::Bindings> multiple_bindings_;
        hailort::AsyncInferJob last_infer_job_;
        std::map<std::string, hailo_vstream_info_t> output_vstream_info_by_name_;
        int model_input_width_;
        int model_input_height_;

        void initializeHailo();
        void preprocessAsync(cv::VideoCapture& capture, std::shared_ptr<BatchQueue> preprocessed_batch_queue);
        void reformatFrames(const std::vector<cv::Mat>& org_frames, std::vector<cv::Mat>& preprocessed_frames);
        void inferAsync();
        void postprocessAsync();
};
