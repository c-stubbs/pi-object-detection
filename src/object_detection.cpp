#include <opencv2/opencv.hpp>

#include <iostream>
#include <chrono>
#include <atomic>
#include <thread>
#include <cstring>

#include "object_detection.h"

ObjectDetection::ObjectDetection(ObjectDetectionConfig config)
{
    logger_ = Logger("ObjectDetection", config.log_level_);
    src_url_ = config.src_url_;
}

void ObjectDetection::run()
{
    cv::VideoCapture cap;
    int fps = 30;

    std::string pipeline("appsrc ! videoconvert" 
                         " ! video/x-raw,format=I420" 
                         " ! x264enc speed-preset=ultrafast bitrate=600 key-int-max=60" 
                         " ! video/x-h264,profile=baseline"
                         " ! rtspclientsink location=rtsp://127.0.0.1:8554/mystream");

    cv::VideoWriter wrt(pipeline, cv::CAP_GSTREAMER, 0, fps, cv::Size(720, 480), true);

    if (!wrt.isOpened())
    {
        logger_.error("Cannot open RTSP writer");
    }

    // Try forcing FFmpeg backend (optional but recommended)
    if (!cap.open(src_url_, cv::CAP_FFMPEG))
    {
        logger_.error("Cannot open RTSP stream");
    }

    cv::Mat frame;

    while (true)
    {
        if (!cap.read(frame))
        {
            logger_.error("Failed to read frame");
            break;
        }

        logger_.trace("Frame Info: {}x{}", frame.rows, frame.cols);
        // ESC to quit
        if (cv::waitKey(1) == 27)
            break;

        wrt.write(frame);

    }

    cap.release();
    cv::destroyAllWindows();
}
