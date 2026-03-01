#include <opencv2/opencv.hpp>

#include <cstring>

#include "hailo/hailort.hpp"

#include "object_detection.h"

#include "hailo_apps/cpp/common/toolbox.hpp"
#include "hailo_apps/cpp/common/hailo_infer.hpp"  

ObjectDetection::ObjectDetection(ObjectDetectionConfig config)
{
    logger_ = Logger("ObjectDetection", config.log_level_);
    
    // Set config to class variables
    src_url_ = config.src_url_;
    mtx_url_ = config.mtx_url_;
    model_name_ = config.model_name_;
    batch_size_ = config.batch_size_;
    target_fps_ = config.target_fps_;
}

void ObjectDetection::run()
{
    cv::VideoCapture cap;
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
    if (!cap.open(src_url_, cv::CAP_FFMPEG))
    {
        logger_.error("Cannot open RTSP stream");
    }

    cv::Mat frame;

    while (true)
    {
        // Read in frame from RTSP source
        if (!cap.read(frame))
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
    cap.release();
    cv::destroyAllWindows();
}

void ObjectDetection::preprocessAsync(cv::VideoCapture& capture, std::shared_ptr<ObjectDetection::BatchQueue> preprocessed_batch_queue)
{
    std::vector<cv::Mat> org_frames;
    std::vector<cv::Mat> preproc_frames;

    const bool limit_fps = (target_fps_ > 0.0);
    using clock = std::chrono::steady_clock;

    clock::duration frame_interval{};
    clock::time_point next_frame_time{};

    if (limit_fps)
    {
        frame_interval = std::chrono::duration_cast<clock::duration>(
                std::chrono::duration<double>(1.0 / target_fps_));
        next_frame_time = clock::now() + frame_interval;
    }

    while (true)
    {
        if (limit_fps)
        {
            auto now = clock::now();
            if (now < next_frame_time)
            {
                std::this_thread::sleep_until(next_frame_time);
            }
        }

        cv::Mat org_frame;
        capture >> org_frame;
        if (org_frame.empty())
        {
            preprocessed_batch_queue->stop();
            break;
        }

        org_frames.push_back(org_frame);
        if (org_frames.size() == batch_size_)
        {
            preproc_frames.clear();
            reformatFrames(org_frames, preproc_frames);
            preprocessed_batch_queue->push(std::make_pair(org_frames, preproc_frames));
            org_frames.clear();
        }

        if (limit_fps)
        {
            next_frame_time += frame_interval;
        }

    }
}

void ObjectDetection::reformatFrames(const std::vector<cv::Mat>& org_frames,
                                         std::vector<cv::Mat>& preprocessed_frames)                                         
{
    preprocessed_frames.clear();
    preprocessed_frames.reserve(org_frames.size());

    for (const auto &src_bgr : org_frames)
    {
        if (src_bgr.empty())
        {
            preprocessed_frames.emplace_back();
            continue;
        }
        cv::Mat rgb;

        // 1) Convert to RGB
        if (src_bgr.channels() == 3)
        {
            cv::cvtColor(src_bgr, rgb, cv::COLOR_BGR2RGB);
        }
        else if (src_bgr.channels() == 4)
        {
            cv::cvtColor(src_bgr, rgb, cv::COLOR_BGRA2RGB);
        }
        else if (src_bgr.channels() == 1)
        {
            cv::cvtColor(src_bgr, rgb, cv::COLOR_GRAY2RGB);
        }
        else
        {
            std::vector<cv::Mat> ch(3, src_bgr);
            cv::merge(ch, rgb);
            cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);
        }

        // 2) Resize to Target
        if (rgb.cols != model_input_width_ || rgb.rows != model_input_height_)
        {
            cv::resize(rgb, rgb, cv::Size(model_input_width_,
                                          model_input_height_),
                                          0.0, 0.0, cv::INTER_AREA);
        }

        // 3) Ensure contiguous buffer
        if (!rgb.isContinuous())
        {
            rgb = rgb.clone();
        }

        // 4) Push to output vector
        preprocessed_frames.push_back(std::move(rgb));
    }

}


void ObjectDetection::inferAsync()
{

}

void ObjectDetection::postprocessAsync()
{

}
