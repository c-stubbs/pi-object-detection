#include "api.h"

Api::Api()
{
    server_.Get("/hi",
            [this](const httplib::Request& req, httplib::Response& res)
            {
                res.set_content("Hello World!", "text/plain");
            });

    server_.Get("/test",
            [this](const httplib::Request& req, httplib::Response& res)
            {
                if (this->on_put_confidence_threshold_)
                    (*on_put_confidence_threshold_)();

                res.set_content("Passed", "text/plain");
            });

    server_.listen("0.0.0.0", 8080);
}

void Api::onPutConfidenceThreshold(std::unique_ptr<std::function<void()>> callback)
{
    if (callback)
        this->on_put_confidence_threshold_ = std::move(callback);
    // else error TODO
}


