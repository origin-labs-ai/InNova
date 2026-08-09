#pragma once
#include "quant/trainer.h"
#include <string>
#include <fstream>
#include <mutex>

namespace quant {

// C21: Lightweight TensorBoard-compatible event writer (no protobuf dependency)
class EventWriter {
public:
    EventWriter(const std::string& logdir);
    ~EventWriter();
    void write_scalar(const std::string& tag, float value, int step);
    void write_scalars(const TrainMetrics& metrics, int step);
    void flush();
private:
    std::string logdir_;
    std::ofstream file_;
    std::mutex mtx_;
    void write_event(const std::string& tag, float value, int step, int64_t wall_time);
};

// C22: WandB-compatible lightweight logger (writes metrics.json instead of API calls)
class WandBLogger {
public:
    WandBLogger(const std::string& project, const std::string& run_name = "");
    ~WandBLogger();
    void log(const std::string& key, float value, int step);
    void log_metrics(const TrainMetrics& metrics, int step);
    void flush();
private:
    std::string run_name_;
    std::ofstream file_;
    std::mutex mtx_;
};

} // namespace quant
