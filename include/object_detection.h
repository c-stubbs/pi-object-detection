#pragma once

#include <hailo/hailort.h>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#include "hailo_apps/cpp/common/toolbox.hpp"
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

        // Misc
        cv::VideoCapture cap_;
        std::chrono::duration<double> inference_time_; 
        int model_input_width_;
        int model_input_height_;
        bool save_stream_output_ = false;
        std::string output_dir_ = "./";
        std::string output_resolution_ = "720x480";
        double org_height_;
        double org_width_;
        size_t frame_count_;
        std::unique_ptr<HailoInfer> model;
        std::string input_path; // remove need for this
        hailo_utils::InputType input_type_;
        double framerate_;
        size_t batch_size_st_;
        hailo_utils::VisualizationParams vis_params;

        static constexpr size_t MAX_QUEUE_SIZE = 60;

        std::shared_ptr<hailo_utils::BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>> preprocessed_batch_queue =
            std::make_shared<hailo_utils::BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>>(MAX_QUEUE_SIZE);

        std::shared_ptr<hailo_utils::BoundedTSQueue<hailo_utils::InferenceResult>> results_queue =
            std::make_shared<hailo_utils::BoundedTSQueue<hailo_utils::InferenceResult>>(MAX_QUEUE_SIZE);


        void preprocessCallback(const std::vector<cv::Mat>& org_frames, std::vector<cv::Mat>& preprocessed_frames, uint32_t target_width, uint32_t target_height);
        void postprocessCallback(cv::Mat& frame_to_draw, const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>>& output_data_and_infos, const hailo_utils::VisualizationParams& vis);
        hailo_status initializeHailo();
};
