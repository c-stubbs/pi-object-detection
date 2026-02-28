#pragma once

#include <opencv2/opencv.hpp>
#include <hailo/hailort.hpp>

#include "async_queue.h"
#include "inference_config.h"
#include "logger.h"

class Inference{

    public:
        Inference(InferenceConfig config);
        using InputMap = std::map<std::string, std::vector<cv::Mat>>;
        using ModelInputQueuesMap = std::vector<
            std::pair<std::string,
                std::shared_ptr<AsyncQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>>>>;
        struct InferenceResult {
                cv::Mat org_frame;
                std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> output_data_and_infos;
                std::vector<std::shared_ptr<uint8_t>> output_guards;
            };
   
        using BatchQueue = AsyncQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>;
        static std::shared_ptr<uint8_t> page_aligned_alloc(size_t size, void* buff = nullptr) {
                auto addr = mmap(buff, size, PROT_WRITE | PROT_READ, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
                if (MAP_FAILED == addr) throw std::bad_alloc();
                return std::shared_ptr<uint8_t>(reinterpret_cast<uint8_t*>(addr), [size](void *addr) { munmap(addr, size); });
}


    private:
        // Config
        int target_fps_;
        int batch_size_;
        std::string model_name_;

        std::unique_ptr<hailort::VDevice> vdevice_;
        std::shared_ptr<hailort::InferModel> infer_model_;
        hailort::ConfiguredInferModel configured_infer_model_;
        std::vector<hailort::ConfiguredInferModel::Bindings> multiple_bindings_;
        hailort::AsyncInferJob last_infer_job_;
        std::map<std::string, hailo_vstream_info_t> output_vstream_info_by_name_;
        int model_input_width_;
        int model_input_height_;
        
        Logger logger_;

        void initialize();
        void preprocessAsync(cv::VideoCapture& capture, std::shared_ptr<BatchQueue> preprocessed_batch_queue);
        void reformatFrames(const std::vector<cv::Mat>& org_frames, std::vector<cv::Mat>& preprocessed_frames);
        void inferAsync(std::chrono::duration<double>& inference_time, ModelInputQueuesMap &named_input_queues, std::shared_ptr<AsyncQueue<InferenceResult>> results_queue);
        void postprocessAsync();

        void infer(const InputMap &inputs, 
                   std::function<void(const hailort::AsyncInferCompletionInfo&,
                   const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> &,
                   const std::vector<std::shared_ptr<uint8_t>> &)> callback);

        void set_input_buffers(const InputMap &inputs, std::vector<std::shared_ptr<cv::Mat>> &image_guards);
        std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> prepare_output_buffers(std::vector<std::shared_ptr<uint8_t>> &output_guards);


        void run_async( const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> &output_data_and_infos,
                        const std::vector<std::shared_ptr<uint8_t>> &output_guards,
                        const std::vector<std::shared_ptr<cv::Mat>> &input_image_guards,
                        std::function<void( const hailort::AsyncInferCompletionInfo&,
                                            const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> &,
                                            const std::vector<std::shared_ptr<uint8_t>> &)> callback);
};
