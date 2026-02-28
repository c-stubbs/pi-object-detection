#include "inference.h"

Inference::Inference(InferenceConfig config)
{
    logger_ = Logger("Inference", config.log_level_);
    target_fps_ = 0; // TODO
    batch_size_ = 0; // TODO
    model_name_ = "yolov8n"; // TODO
    initialize();
}

void Inference::preprocessAsync(cv::VideoCapture& capture, std::shared_ptr<Inference::BatchQueue> preprocessed_batch_queue)
{
    std::vector<cv::Mat> org_frames;
    std::vector<cv::Mat> preproc_frames;

    const bool limit_fps = (target_fps_ > 0.0);
    using clock = std::chrono::steady_clock;

    clock::duration frame_interval{};
    clock::time_point next_frame_time{};

    if (limit_fps)
    {
        frame_interval = std::chrono::duration_cast<clock::duration>(
                std::chrono::duration<double>(1.0 / target_fps_));
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
        if (org_frames.size() == batch_size_)
        {
            preproc_frames.clear();
            reformatFrames(org_frames, preproc_frames);
            preprocessed_batch_queue->push(std::make_pair(org_frames, preproc_frames));
            org_frames.clear();
        }

        if (limit_fps)
        {
            next_frame_time += frame_interval;
        }

    }
}

void Inference::reformatFrames(const std::vector<cv::Mat>& org_frames,
                                         std::vector<cv::Mat>& preprocessed_frames)                                         
{
    preprocessed_frames.clear();
    preprocessed_frames.reserve(org_frames.size());

    for (const auto &src_bgr : org_frames)
    {
        if (src_bgr.empty())
        {
            preprocessed_frames.emplace_back();
            continue;
        }
        cv::Mat rgb;

        // 1) Convert to RGB
        if (src_bgr.channels() == 3)
        {
            cv::cvtColor(src_bgr, rgb, cv::COLOR_BGR2RGB);
        }
        else if (src_bgr.channels() == 4)
        {
            cv::cvtColor(src_bgr, rgb, cv::COLOR_BGRA2RGB);
        }
        else if (src_bgr.channels() == 1)
        {
            cv::cvtColor(src_bgr, rgb, cv::COLOR_GRAY2RGB);
        }
        else
        {
            std::vector<cv::Mat> ch(3, src_bgr);
            cv::merge(ch, rgb);
            cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);
        }

        // 2) Resize to Target
        if (rgb.cols != model_input_width_ || rgb.rows != model_input_height_)
        {
            cv::resize(rgb, rgb, cv::Size(model_input_width_,
                                          model_input_height_),
                                          0.0, 0.0, cv::INTER_AREA);
        }

        // 3) Ensure contiguous buffer
        if (!rgb.isContinuous())
        {
            rgb = rgb.clone();
        }

        // 4) Push to output vector
        preprocessed_frames.push_back(std::move(rgb));
    }

}


void Inference::inferAsync(std::chrono::duration<double>& inference_time, ModelInputQueuesMap &named_input_queues, std::shared_ptr<AsyncQueue<InferenceResult>> results_queue)
{
    const auto start_time = std::chrono::high_resolution_clock::now();
    const size_t outputs_per_binding = this->infer_model_->get_output_names().size();
    if (named_input_queues.empty())
    {
        logger_.error("Named input queue is empty!");
        return;
    } 
    bool jobs_submitted = false;

    while (true)
    { 
        //build InputMap and capture originals in one pass
        InputMap inputs_map;
        std::vector<cv::Mat> org_frames;
        bool have_org = false;

        for (const auto &[input_name, queue] : named_input_queues) {
            std::optional<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>> pack;
            pack = queue->pop();
            if (!pack.has_value()) goto done; // TODO: is goto really the best choice here?

            if (!have_org) {
                org_frames = std::move(pack.value().first);
                have_org = true;
            }
            inputs_map.emplace(input_name, std::move(pack.value().second));
        }

        this->infer(
            inputs_map,
            [org_frames = std::move(org_frames), results_queue, outputs_per_binding]
            (const hailort::AsyncInferCompletionInfo &,
             const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> &flat_outputs,
             const std::vector<std::shared_ptr<uint8_t>> &flat_guards)
            {
                const size_t batch_size = org_frames.size();
                for (size_t i = 0; i < batch_size; ++i) {
                    InferenceResult out;
                    out.org_frame = org_frames[i];

                    const size_t start = i * outputs_per_binding;
                    const size_t end   = start + outputs_per_binding;
                    out.output_data_and_infos.insert(out.output_data_and_infos.end(),
                                                     flat_outputs.begin() + start,
                                                     flat_outputs.begin() + end);
                    out.output_guards.insert(out.output_guards.end(),
                                             flat_guards.begin() + start,
                                             flat_guards.begin() + end);

                    results_queue->push(std::move(out));
                }
            }
        );

        jobs_submitted = true;
    }

done:
    if (jobs_submitted) this->wait_for_last_job();
    results_queue->stop();
    inference_time = std::chrono::high_resolution_clock::now() - start_time;
    return HAILO_SUCCESS;
}

void Inference::postprocessAsync()
{

}

void Inference::initialize()
{
    logger_.info("Initializing HAILO device.");

    this->vdevice_ = hailort::VDevice::create().expect("Failed to create VDevice");
    this->infer_model_ = vdevice_->create_infer_model(model_name_).expect("Failed to create infer model");
    this->infer_model_->set_batch_size(batch_size_);
    
    auto model_input_shape = this->infer_model_->hef().get_input_vstream_infos().release()[0].shape;
    this->model_input_width_ = static_cast<int>(model_input_shape.width);
    this->model_input_height_ = static_cast<int>(model_input_shape.height);
    
    for (auto& output_vstream_info : this->infer_model_->hef().get_output_vstream_infos().release())
    {
        std::string name(output_vstream_info.name);
        this->output_vstream_info_by_name_[name] = output_vstream_info;
    }

    this->configured_infer_model_ = this->infer_model_->configure().expect("Failed to create configured infer model");
    this->multiple_bindings_ = std::vector<hailort::ConfiguredInferModel::Bindings>();
    
    logger_.info("HAILO device initialized.");
}

void Inference::infer(
    const InputMap &inputs,
    std::function<void(const hailort::AsyncInferCompletionInfo&,
                       const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> &,
                       const std::vector<std::shared_ptr<uint8_t>> &)> callback) {

    std::vector<std::shared_ptr<cv::Mat>> input_image_guards;
    set_input_buffers(inputs, input_image_guards);
    std::vector<std::shared_ptr<uint8_t>> output_guards;
    auto output_data_and_infos = prepare_output_buffers(output_guards);
    this->run_async(output_data_and_infos, output_guards, input_image_guards, callback);
}

void Inference::set_input_buffers(
    const InputMap &inputs,
    std::vector<std::shared_ptr<cv::Mat>> &image_guards)
{
    this->multiple_bindings_.clear();
    const auto &model_inputs = infer_model_->get_input_names();

    for (size_t i = 0; i < this->batch_size_; ++i) {
        auto bindings = this->configured_infer_model_.create_bindings().expect("Failed");
        for (const auto &input_name : model_inputs) {
            const cv::Mat &input = inputs.at(input_name)[i];
            size_t frame_size = infer_model_->input(input_name)->get_frame_size();
            auto status = bindings.input(input_name)->set_buffer(hailort::MemoryView(input.data, frame_size));
            if (HAILO_SUCCESS != status) {
                std::cerr << "Failed to set input buffer for '" << input_name
                          << "', status = " << status << std::endl;
            }
            // keep input data alive until the async job completes
            image_guards.push_back(std::make_shared<cv::Mat>(input));

        }
        this->multiple_bindings_.push_back(std::move(bindings));
    }
}

std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> Inference::prepare_output_buffers(
    std::vector<std::shared_ptr<uint8_t>> &output_guards) {

    std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> result;
    for (auto& binding : this->multiple_bindings_) {
        for (const auto &output_name : this->infer_model_->get_output_names()) {
            size_t frame_size = this->infer_model_->output(output_name)->get_frame_size();
            auto buffer = page_aligned_alloc(frame_size);
            output_guards.push_back(buffer);
            auto status = binding.output(output_name)->set_buffer(hailort::MemoryView(buffer.get(), frame_size));
            if (HAILO_SUCCESS != status) {
                std::cerr << "Failed to set output buffer, status = " << status << std::endl;
            }
            result.emplace_back(buffer.get(), output_vstream_info_by_name_[output_name]);
        }
    }
    return result;
}

void Inference::run_async(
    const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> &output_data_and_infos,
    const std::vector<std::shared_ptr<uint8_t>> &output_guards,
    const std::vector<std::shared_ptr<cv::Mat>> &input_image_guards,
    std::function<void(const hailort::AsyncInferCompletionInfo&,
                       const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> &,
                       const std::vector<std::shared_ptr<uint8_t>> &)> callback)
{
    auto status = configured_infer_model_.wait_for_async_ready(std::chrono::milliseconds(50000), this->batch_size_);
    if (HAILO_SUCCESS != status) {
        std::cerr << "Failed wait_for_async_ready, status = " << status << std::endl;
    }
    auto job = configured_infer_model_.run_async(
        this->multiple_bindings_,
        [callback, output_data_and_infos, input_image_guards, output_guards](const hailort::AsyncInferCompletionInfo& info)
        {
            // callback sent by the applicative side
            callback(info, output_data_and_infos, output_guards);
        }
    );
    if (!job) {
        std::cerr << "Failed to start async infer job, status = " << job.status() << std::endl;
    }
    job->detach();
    last_infer_job_ = std::move(job.release());
}
