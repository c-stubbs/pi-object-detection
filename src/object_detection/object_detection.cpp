#include <chrono>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <thread>

#include "object_detection.h"
#include "hailo_apps/cpp/object_detection/utils/utils.hpp"
#include "hailo_apps/cpp/common/labels/coco_eighty.hpp"

ObjectDetection::ObjectDetection(ObjectDetectionConfig config) : model_(config.model_name_, config.batch_size_)
{
    logger_ = Logger("ObjectDetection", config.log_level_);
    
    // Set config to class variables
    src_url_ = config.src_url_;
    org_height_ = config.src_height_;
    org_width_ = config.src_width_;
    mtx_url_ = config.mtx_url_;
    model_name_ = config.model_name_;
    batch_size_ = config.batch_size_;
    target_fps_ = config.target_fps_;
    restream_grayscale_ = config.restream_grayscale_;
    valid_objects_ = processValidObjects(config.valid_objects_);

    kill_ = false;
    
    // Hailo initialization
    first_frame_received_ = false;
    input_type_.is_video = true;
    input_queues_ = {{ model_.get_infer_model()->get_input_names().at(0), preprocessed_batch_queue }};
    save_stream_output_ = false;
    vis_params_.score_thresh = 0.15;
}

ObjectDetection::~ObjectDetection()
{
    kill_ = true;
    cap_.release();
}

void ObjectDetection::run()
{
    if (!setupStreamHandlers()) return;
    if (!setupPreprocessing()) return;
    if (!setupInference()) return;
    if (!setupPostprocessing()) return;
    keepRunning();
}

bool ObjectDetection::setupStreamHandlers()
{    
    std::string pipeline("appsrc ! videoconvert" 
                         " ! video/x-raw,format=I420"
                         " ! clockoverlay time-format=\"%Y-%m-%d %H:%M:%S\" halignment=right valignment=top "
                         " ! x264enc speed-preset=ultrafast bitrate=600 key-int-max=60" 
                         " ! video/x-h264,profile=baseline"
                         " ! rtspclientsink location="+ mtx_url_);
 
    writer_ = cv::VideoWriter(pipeline, cv::CAP_GSTREAMER, 0, target_fps_, cv::Size(720, 480), true);

    if (!writer_.isOpened())
    {
        logger_.error("Cannot open RTSP writer");
        return false;
    }

    if (!cap_.open(src_url_, cv::CAP_FFMPEG))
    {
        logger_.error("Cannot open RTSP stream");
        return false;
    }

    return true;
}

bool ObjectDetection::setupPreprocessing()
{
    hailo_utils::PreprocessCallback preprocess_callback = [this](const std::vector<cv::Mat>& org_frames, std::vector<cv::Mat>& preproc_frames, auto target_width, auto target_height)
    {
        this->preprocessCallback(org_frames, preproc_frames, target_width, target_height);
    };
 
    std::thread([this, preprocess_callback]
    {
        hailo_utils::run_preprocess(this->input_path_, this->model_name_, this->model_, this->input_type_, this->cap_, this->batch_size_, this->target_fps_, this->preprocessed_batch_queue, preprocess_callback);
    }
    ).detach();

    return true;
}

bool ObjectDetection::setupInference()
{
    std::thread([this] 
    {
        // Loop here until the first frame has been received
        while (!first_frame_received_)
        {
            if (kill_)
            {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        hailo_utils::run_inference_async(this->model_, this->inference_time_, this->input_queues_, this->results_queue);
    }
    ).detach();

    return true;
}

bool ObjectDetection::setupPostprocessing()
{
    hailo_utils::PostprocessCallback postprocess_callback = [this](cv::Mat& frame_to_draw, const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>>& output_data_and_infos)
    {
        this->postprocessCallback(frame_to_draw, output_data_and_infos, vis_params_);
    };

    std::thread([this, postprocess_callback]
    {
        // Loop here until the first frame has been received
        while (!first_frame_received_)
        {
            if (kill_)
            {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        hailo_utils::run_post_process(this->input_type_, this->org_height_, this->org_width_, this->frame_count_, this->cap_, this->target_fps_, this->batch_size_, this->save_stream_output_, this->save_stream_output_dir_, this->save_stream_output_res_, this->results_queue, postprocess_callback);
    }).detach();

    return true;
}

void ObjectDetection::keepRunning()
{
    while (!kill_)
    {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

std::vector<size_t> ObjectDetection::processValidObjects(std::vector<std::string> valid_strings)
{
    std::vector<size_t> valid_objects;
    for (auto type : valid_strings)
    {
        for (const auto& [class_id, class_name] : common::coco_eighty)
        {
            if (type == class_name)
            {
                valid_objects.push_back(class_id);
                logger_.info("Adding detection type {} ({}) to valid objects list", class_id, class_name);
            }
        }
    }
    return valid_objects;

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
        logger_.debug("Image channels before conversion: {}", src_bgr.channels());
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
        logger_.debug("Image channels after conversion: {}", rgb.channels());
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
        logger_.debug("rgb size: {}x{}", rgb.cols, rgb.rows);
        // 4) Push to output vector
        preprocessed_frames.push_back(std::move(rgb));
        logger_.debug("preprocessed_frames size: {}", preprocessed_frames.size());
        first_frame_received_ = true;
    }
}

// Task-specific postprocessing callback
void ObjectDetection::postprocessCallback(
    cv::Mat &frame_to_draw,
    const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> &output_data_and_infos,
    const hailo_utils::VisualizationParams &vis)
{
    if (restream_grayscale_)
    {
        cv::cvtColor(frame_to_draw, frame_to_draw, cv::COLOR_BGR2GRAY);
        cv::cvtColor(frame_to_draw, frame_to_draw, cv::COLOR_GRAY2BGR);
    }
    const size_t class_count = 80;
    auto bboxes = parse_nms_data(output_data_and_infos[0].first, class_count);
    //std::vector<size_t> class_whitelist = {1, 16, 17};
    for (auto& box : bboxes)
    {
        if (std::find(valid_objects_.begin(), valid_objects_.end(), box.class_id) == valid_objects_.end())
        {
            box.bbox.score = 0.0;
        }
    }

    draw_bounding_boxes(frame_to_draw, bboxes, vis);

    // Write the frame to RTSP
    writer_.write(frame_to_draw);
}

