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
        std::unique_ptr<hailort::VDevice> vdevice;
        std::shared_ptr<hailort::InferModel> infer_model;
        hailort::ConfiguredInferModel configured_infer_model;
        std::vector<hailort::ConfiguredInferModel::Bindings> multiple_bindings;
        hailort::AsyncInferJob last_infer_job;
        std::map<std::string, hailo_vstream_info_t> output_vstream_info_by_name;
        size_t batch_size;

        void initializeHailo();
        void preprocessAsync(cv::VideoCapture& capture, std::shared_ptr<BatchQueue> preprocessed_batch_queue);
        void inferAsync();
        void postprocessAsync();
};
