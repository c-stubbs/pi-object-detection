#pragma once

#include <opencv2/opencv.hpp>

#ifdef HEADLESS_OPENCV

namespace cv {

    inline void imshow(const std::string&, InputArray) {}

    inline int waitKey(int delay) {return -1;}

    inline void destroyAllWindows() {}
}

#endif
