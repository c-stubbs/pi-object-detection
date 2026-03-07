#include <chrono>
#include <opencv2/opencv.hpp>

#include <cstring>
#include <opencv2/videoio.hpp>
#include <thread>

#include "object_detection.h"

#include "hailo_apps/cpp/common/toolbox.hpp"
#include "hailo_apps/cpp/common/hailo_infer.hpp"
#include "hailo_apps/cpp/object_detection/utils/utils.hpp"

ObjectDetection::ObjectDetection(ObjectDetectionConfig config)
{
    logger_ = Logger("ObjectDetection", config.log_level_);
    
    // Set config to class variables
    src_url_ = config.src_url_;
    mtx_url_ = config.mtx_url_;
    model_name_ = config.model_name_;
    batch_size_ = config.batch_size_;
    target_fps_ = config.target_fps_;

    first_frame_received_ = false;
}

ObjectDetection::~ObjectDetection()
{
    kill_ = true;
    cap_.release();
}

void ObjectDetection::run()
{
    int fps = 30;
    
    std::string pipeline("appsrc ! videoconvert" 
                         " ! video/x-raw,format=I420" 
                         " ! x264enc speed-preset=ultrafast bitrate=600 key-int-max=60" 
                         " ! video/x-h264,profile=baseline"
                         " ! rtspclientsink location="+ mtx_url_);
 
    writer_ = cv::VideoWriter(pipeline, cv::CAP_GSTREAMER, 0, fps, cv::Size(720, 480), true);

    if (!writer_.isOpened())
    {
        logger_.error("Cannot open RTSP writer");
    }

    if (!cap_.open(src_url_, cv::CAP_FFMPEG))
    {
        logger_.error("Cannot open RTSP stream");
    }
 
    initializeHailo();
}

hailo_status ObjectDetection::initializeHailo()
{

    input_type_.is_video = true;
    auto model = HailoInfer(model_name_, batch_size_);

    // --- Preprocessing

    // Lambda for preprocessing
    hailo_utils::PreprocessCallback preprocess_cb = [this](const std::vector<cv::Mat>& org_frames, std::vector<cv::Mat>& preproc_frames, auto target_width, auto target_height)
    {
        this->preprocessCallback(org_frames, preproc_frames, target_width, target_height);
    };

    
    std::thread([&] ()
    {
        return hailo_utils::run_preprocess(this->input_path_, this->model_name_, model, this->input_type_, this->cap_, this->batch_size_, this->target_fps_, this->preprocessed_batch_queue, preprocess_cb);
    }).detach();

    hailo_utils::ModelInputQueuesMap input_queues = {
        { model.get_infer_model()->get_input_names().at(0), this->preprocessed_batch_queue }
    };

    // Loop here until the first frame has been received
    while (!first_frame_received_)
    {
        if (kill_)
        {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    // --- Inference

    std::thread([&] () 
    {
        return hailo_utils::run_inference_async(model, this->inference_time_, input_queues, this->results_queue);
    }).detach();

    // --- Post Processing

    // Callback for post processing
    hailo_utils::PostprocessCallback post_cb = [this](auto frame_to_draw, auto output_data_and_infos)
    {
        this->postprocessCallback(frame_to_draw, output_data_and_infos, this->vis_params_);
    };

    std::thread([&] ()
    {
        return hailo_utils::run_post_process(this->input_type_, this->org_height_, this->org_width_, this->frame_count_, this->cap_, this->target_fps_, this->batch_size_, this->save_stream_output_, this->output_dir_, this->output_resolution_, this->results_queue, post_cb);
    }).detach();


    for (;;)
    {
    }

    return HAILO_SUCCESS;
}

// Task-specific preprocessing callback
void ObjectDetection::preprocessCallback(const std::vector<cv::Mat>& org_frames,
                         std::vector<cv::Mat>& preprocessed_frames,
                         uint32_t target_width, uint32_t target_height)
{
    preprocessed_frames.clear();
    preprocessed_frames.reserve(org_frames.size());

    for (const auto &src_bgr : org_frames) {
        // Skip invalid frames but keep vector alignment (optional: push empty)
        if (src_bgr.empty()) {
            preprocessed_frames.emplace_back();
            continue;
        }
        cv::Mat rgb;
        // 1) Convert to RGB
        if (src_bgr.channels() == 3) {
            cv::cvtColor(src_bgr, rgb, cv::COLOR_BGR2RGB);
        } else if (src_bgr.channels() == 4) {
            // If someone passed BGRA, drop alpha
            cv::cvtColor(src_bgr, rgb, cv::COLOR_BGRA2RGB);
        } else if (src_bgr.channels() == 1) {
            // If grayscale sneaks in, promote to 3 channels
            cv::cvtColor(src_bgr, rgb, cv::COLOR_GRAY2RGB);
        } else {
            // Fallback: force 3 channels by duplicating/merging
            std::vector<cv::Mat> ch(3, src_bgr);
            cv::merge(ch, rgb);
            cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB); // ensure RGB order
        }
        // 2) Resize to target
        if (rgb.cols != static_cast<int>(target_width) || rgb.rows != static_cast<int>(target_height)) {
            cv::resize(rgb, rgb, cv::Size(static_cast<int>(target_width),
                                          static_cast<int>(target_height)),
                       0.0, 0.0, cv::INTER_AREA);
        }
        // 3) Ensure contiguous buffer
        if (!rgb.isContinuous()) {
            rgb = rgb.clone();
        }
        logger_.info("rgb size: {}x{}", rgb.cols, rgb.rows);
        // 4) Push to output vector
        preprocessed_frames.push_back(std::move(rgb));
        logger_.info("preprocessed_frames size: {}", preprocessed_frames.size());
        first_frame_received_ = true;
    }
}

// Task-specific postprocessing callback
void ObjectDetection::postprocessCallback(
    cv::Mat &frame_to_draw,
    const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> &output_data_and_infos,
    const hailo_utils::VisualizationParams &vis)
{
    const size_t class_count = 80;
    auto bboxes = parse_nms_data(output_data_and_infos[0].first, class_count);

    draw_bounding_boxes(frame_to_draw, bboxes, vis);

    // Write the frame to RTSP
    writer_.write(frame_to_draw);
}

