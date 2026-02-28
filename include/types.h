#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

using InputMap = std::map<std::string, std::vector<cv::Mat>>;

// who thought this was a good idea??
using ModelInputQueuesMap = std::vector<
    std::pair<std::string,
        std::shared_ptr<BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>>>>;




