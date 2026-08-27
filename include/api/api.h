#pragma once

#include "httplib.h"
#include <memory>

class Api {

    public:
        Api();
        void onPutConfidenceThreshold(std::unique_ptr<std::function<void()>> callback);

    private:
        httplib::Server server_;
        std::unique_ptr<std::function<void()>> on_put_confidence_threshold_;
};
