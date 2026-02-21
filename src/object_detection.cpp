#include <opencv2/opencv.hpp>

#include <cstring>

#include "hailo/hailort.hpp"

#include "object_detection.h"

ObjectDetection::ObjectDetection(ObjectDetectionConfig config)
{
    logger_ = Logger("ObjectDetection", config.log_level_);
    
    // Set config to class variables
    src_url_ = config.src_url_;
    mtx_url_ = config.mtx_url_;
    model_name_ = config.model_name_;
    batch_size_ = config.batch_size_;
    target_fps_ = config.target_fps_;

    initializeHailo();
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
    int fps = 30; // TODO: make class variable or config

    std::vector<cv::Mat> org_frames;
    std::vector<cv::Mat> preproc_frames;

    const bool limit_fps = (fps > 0.0);
    using clock = std::chrono::steady_clock;

    clock::duration frame_interval{};
    clock::time_point next_frame_time{};

    if (limit_fps)
    {
        frame_interval = std::chrono::duration_cast<clock::duration>(
                std::chrono::duration<double>(1.0 / fps));
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
        if (org_frames.size() == batch_size)
        {
            preproc_frames.clear();
            // TODO: put preprocess logic here
            preprocessed_batch_queue->push(std::make_pair(org_frames, preproc_frames));
            org_frames.clear();
        }

        if (limit_fps)
        {
            next_frame_time += frame_interval;
        }

    }
}

void ObjectDetection::inferAsync()
{

}

void ObjectDetection::postprocessAsync()
{

}

void ObjectDetection::initializeHailo()
{
    logger_.info("Initializing HAILO device.");

    this->vdevice = hailort::VDevice::create().expect("Failed to create VDevice");
    this->batch_size = batch_size_;
    this->infer_model = vdevice->create_infer_model(model_name_).expect("Failed to create infer model");
    this->infer_model->set_batch_size(batch_size);
    
    for (auto& output_vstream_info : this->infer_model->hef().get_output_vstream_infos().release())
    {
        std::string name(output_vstream_info.name);
        this->output_vstream_info_by_name[name] = output_vstream_info;
    }

    this->configured_infer_model = this->infer_model->configure().expect("Failed to create configured infer model");
    this->multiple_bindings = std::vector<hailort::ConfiguredInferModel::Bindings>();
    
    logger_.info("HAILO device initialized.");
}

