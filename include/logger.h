#pragma once

#include <spdlog/spdlog.h>

class Logger
{
public:
    Logger(std::string name, std::string level)
    {
        auto sink_ = std::make_shared<spdlog::sinks::ansicolor_stderr_sink_mt>();
        logger_ = std::make_shared<spdlog::logger>(name, sink_);

        if (level == "trace")
            logger_->set_level(spdlog::level::trace);
        else if (level == "debug")
            logger_->set_level(spdlog::level::debug);
        else if (level == "info")
            logger_->set_level(spdlog::level::info);
        else if (level == "warn")
            logger_->set_level(spdlog::level::warn);
        else if (level == "error")
            logger_->set_level(spdlog::level::err);
        else if (level == "critical")
            logger_->set_level(spdlog::level::critical);
        else
        {
            logger_->set_level(spdlog::level::info);
            warn("Log level invalid. Defaulting to INFO.");
        }
    }

    Logger()
    {
    }

    template <typename... Args>
    void trace(fmt::format_string<Args...> words, Args &&...args)
    {
        if (logger_)
            logger_->trace(words, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void debug(fmt::format_string<Args...> words, Args &&...args)
    {
        if (logger_)
            logger_->debug(words, std::forward<Args>(args)...);
    }
    template <typename... Args>
    void info(fmt::format_string<Args...> words, Args &&...args)
    {
        if (logger_)
            logger_->info(words, std::forward<Args>(args)...);
    }
    template <typename... Args>
    void warn(fmt::format_string<Args...> words, Args &&...args)
    {
        if (logger_)
            logger_->warn(words, std::forward<Args>(args)...);
    }
    template <typename... Args>
    void error(fmt::format_string<Args...> words, Args &&...args)
    {
        if (logger_)
            logger_->error(words, std::forward<Args>(args)...);
    }
    template <typename... Args>
    void critical(fmt::format_string<Args...> words, Args &&...args)
    {
        if (logger_)
            logger_->critical(words, std::forward<Args>(args)...);
    }

private:
    std::shared_ptr<spdlog::logger> logger_;
};