#include <opencv2/opencv.hpp>

#include <cstring>

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
    
    framerate_ = static_cast<double>(target_fps_);
    batch_size_st_ = static_cast<size_t>(batch_size_);
}

void ObjectDetection::run()
{
    // cv::VideoCapture cap;
    int fps = 30;

    std::string pipeline("appsrc ! videoconvert" 
                         " ! video/x-raw,format=I420" 
                         " ! x264enc speed-preset=ultrafast bitrate=600 key-int-max=60" 
                         " ! video/x-h264,profile=baseline"
                         " ! rtspclientsink location=rtsp://" + mtx_url_ + ":8554/test");
 
    cv::VideoWriter wrt(pipeline, cv::CAP_GSTREAMER, 0, fps, cv::Size(720, 480), true);

    if (!wrt.isOpened())
    {
        logger_.error("Cannot open RTSP writer");
    }

    // Use ffmpeg backend. Could also use gstreamer but ffmpeg requires less thinking on my part.
    if (!cap_.open(src_url_, cv::CAP_FFMPEG))
    {
        logger_.error("Cannot open RTSP stream");
    }

    cv::Mat frame;

    while (true)
    {
        // Read in frame from RTSP source
        if (!cap_.read(frame))
        {
            logger_.error("Failed to read frame");
            break;
        }

        logger_.trace("Frame Info: {}x{}", frame.rows, frame.cols);
        // ESC to quit
        if (cv::waitKey(1) == 27)
            break;

        // TODO: Run object detection here

        // Write the frame to RTSP
        wrt.write(frame);
    }
    cap_.release();
    cv::destroyAllWindows();
}

hailo_status ObjectDetection::initializeHailo()
{ 
    // TODO: manually set the input_type here

    model = std::make_unique<HailoInfer>(model_name_, batch_size_);

    // Lambda for preprocessing
    auto preprocess_cb = [this](auto org_frames, auto preproc_frames, auto target_width, auto target_height)
    {
        this->preprocessCallback(org_frames, preproc_frames, target_width, target_height);
    };

    // Lambda for post processing
    auto post_cb = [this](auto frame_to_draw, auto output_data_and_infos)
    {
        this->postprocessCallback(frame_to_draw, output_data_and_infos, this->vis_params);
    };

    auto preprocess_thread = std::async(hailo_utils::run_preprocess,
                                        std::ref(input_path),
                                        std::ref(model_name_),
                                        std::ref(*model),
                                        std::ref(input_type_),
                                        std::ref(cap_),
                                        std::ref(batch_size_st_),
                                        std::ref(framerate_),
                                        preprocessed_batch_queue,
                                        preprocess_cb);

    hailo_utils::ModelInputQueuesMap input_queues = {
        { model->get_infer_model()->get_input_names().at(0), preprocessed_batch_queue }
    };
    auto inference_thread = std::async(hailo_utils::run_inference_async,
                                    std::ref(*model),
                                    std::ref(inference_time_),
                                    std::ref(input_queues),
                                    results_queue);

    auto output_parser_thread = std::async(hailo_utils::run_post_process,
                                std::ref(input_type_),
                                std::ref(org_height_),
                                std::ref(org_width_),
                                std::ref(frame_count_),
                                std::ref(cap_),
                                std::ref(framerate_),
                                std::ref(batch_size_st_),
                                std::ref(save_stream_output_),
                                std::ref(output_dir_),
                                std::ref(output_resolution_),
                                results_queue,
                                post_cb);

    hailo_status status = wait_and_check_threads(
        preprocess_thread,    "Preprocess",
        inference_thread,     "Inference",
        output_parser_thread, "Postprocess "
    );
    if (HAILO_SUCCESS != status) {
        return status;
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
        // 4) Push to output vector
        preprocessed_frames.push_back(std::move(rgb));
    }
}

// Task-specific postprocessing callback
void ObjectDetection::postprocessCallback(
    cv::Mat &frame_to_draw,
    const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> &output_data_and_infos,
    const hailo_utils::VisualizationParams &vis)
{
    const size_t class_count = 80;
    auto bboxes = parse_nms_data(output_data_and_infos[0].first, class_count); // TODO: this is in utils

    draw_bounding_boxes(frame_to_draw, bboxes, vis);
}

