#include <opencv2/opencv.hpp>
#include <cstdio>
#include <iostream>

int main()
{
    // ---------- RTSP INPUT ----------
    std::string rtsp_in =
        "rtsp://user:pass@CAMERA_IP/stream";

    cv::VideoCapture cap(rtsp_in);
    if (!cap.isOpened())
    {
        std::cerr << "Failed to open RTSP input\n";
        return -1;
    }

    cv::Mat frame;
    cap.read(frame);

    int width = frame.cols;
    int height = frame.rows;
    int fps = 30; // adjust if needed

    // ---------- FFmpeg COMMAND ----------
    std::string ffmpeg_cmd =
        "ffmpeg -loglevel error "
        "-f rawvideo "
        "-pix_fmt bgr24 "
        "-s " +
        std::to_string(width) + "x" + std::to_string(height) + " "
                                                               "-r " +
        std::to_string(fps) + " "
                              "-i - "
                              "-c:v libx264 "
                              "-preset ultrafast "
                              "-tune zerolatency "
                              "-f rtsp "
                              "rtsp://127.0.0.1:8554/processed";

    FILE *ffmpeg = popen(ffmpeg_cmd.c_str(), "w");
    if (!ffmpeg)
    {
        std::cerr << "Failed to start FFmpeg\n";
        return -1;
    }

    // ---------- MAIN LOOP ----------
    while (cap.read(frame))
    {
        if (frame.empty())
            continue;

        // ----- OpenCV processing -----
        cv::putText(frame,
                    "OpenCV + FFmpeg RTSP",
                    {40, 50},
                    cv::FONT_HERSHEY_SIMPLEX,
                    1.2,
                    {0, 255, 0},
                    2);

        // ----- Write raw frame to FFmpeg -----
        fwrite(frame.data,
               1,
               frame.total() * frame.elemSize(),
               ffmpeg);
    }

    pclose(ffmpeg);
    return 0;
}
