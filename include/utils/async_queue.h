#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>

// A thread safe fifo queue
template<typename T>
class AsyncQueue {
    public:
        explicit AsyncQueue(size_t max_size) : max_size_(max_size), stopped_(false) {}
        ~AsyncQueue() { stop(); }
        AsyncQueue(const AsyncQueue&) = delete;
        AsyncQueue& operator=(const AsyncQueue&) = delete;

        /// Add an item to the back of the queue
        void push(T item)
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cond_not_full_.wait(lock, [this] {return (queue_.size() < max_size_) || (stopped_); });
            if (stopped_) return;

            queue_.push(std::move(item));
            cond_not_empty_.notify_one();
        }

        /// Get the oldest item in the queue
        std::optional<T> pop()
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cond_not_empty_.wait(lock, [this] {return !queue_.empty() || stopped_; });
            if (stopped_ && queue_.empty())
            {
                return std::nullopt;
            }

            std::optional<T> top_item(std::move(queue_.front())); // TODO: is this right?
            queue_.pop();
            cond_not_full_.notify_one();
            return top_item;
        }

        /// Stop the queue from accepting or producing new items
        void stop()
        {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stopped_ = true;
            }
            cond_not_empty_.notify_all();
            cond_not_full_.notify_all();
        }

        /// Return true if the queue has no items
        bool empty() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return queue_.empty();
        }

    private:
        std::queue<T> queue_;
        std::mutex mutex_;
        std::condition_variable cond_not_empty_;
        std::condition_variable cond_not_full_;
        const size_t max_size_;
        bool stopped_;
};
