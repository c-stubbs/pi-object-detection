#pragma once

#include <hailo/hailort.h>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#include "hailo_apps/cpp/common/toolbox.hpp"
#include "hailo_apps/cpp/common/hailo_infer.hpp"

#include "object_detection_config.h"
#include "logger.h"

class ObjectDetection{
    public:
        ObjectDetection(ObjectDetectionConfig config);
        ~ObjectDetection();
        void run();
    
    private:
        bool setupStreamHandlers();
        bool setupPreprocessing();
        bool setupInference();
        bool setupPostprocessing();
        void keepRunning();
        std::vector<size_t> processValidObjects(std::vector<std::string> valid_strings);

        void preprocessCallback(const std::vector<cv::Mat>& org_frames, std::vector<cv::Mat>& preprocessed_frames, uint32_t target_width, uint32_t target_height);
        void postprocessCallback(cv::Mat& frame_to_draw, const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>>& output_data_and_infos, const hailo_utils::VisualizationParams& vis);
        
        // Config
        std::string src_url_;
        std::string mtx_url_;
        size_t batch_size_;
        std::string model_name_;
        double target_fps_;
        bool restream_grayscale_;
        std::vector<size_t> valid_objects_;

        Logger logger_;

        // Hailo utils
        HailoInfer model_;
        hailo_utils::ModelInputQueuesMap input_queues_;
        cv::VideoCapture cap_;
        cv::VideoWriter writer_;
        std::chrono::duration<double> inference_time_; 
        bool save_stream_output_;
        std::string save_stream_output_dir_;
        std::string save_stream_output_res_;
        double org_height_;
        double org_width_;
        size_t frame_count_;
        std::string input_path_;
        hailo_utils::InputType input_type_;
        hailo_utils::VisualizationParams vis_params_;
        static constexpr size_t kMaxQueueSize = 60;
        using BatchQueue = hailo_utils::BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>;
        using ResultQueue = hailo_utils::BoundedTSQueue<hailo_utils::InferenceResult>;
        std::shared_ptr<BatchQueue> preprocessed_batch_queue = std::make_shared<BatchQueue>(kMaxQueueSize);
        std::shared_ptr<ResultQueue> results_queue = std::make_shared<ResultQueue>(kMaxQueueSize);

        // State tracking
        std::atomic<bool> first_frame_received_;
        std::atomic<bool> kill_;
};
