#pragma once

#include "vesper/target.h"
#include "vesper/types.h"

#include <cstdint>
#include <string>

namespace vesper {

enum class ReportEngine {
    Vesper,
    LlamaCpp,
};

enum class ReportBackend {
    CPU,
    HIP,
    Vulkan,
};

enum class ReportStatus {
    Ok,
    Unsupported,
};

struct DecodeReport {
    ReportEngine engine = ReportEngine::Vesper;
    ReportBackend backend = ReportBackend::CPU;
    std::string model;
    std::string quant;
    std::string arch;
    int prompt_tokens = 0;
    int new_tokens = 0;
    double prefill_tps = 0.0;
    double decode_tps = 0.0;
    std::uint64_t bytes_per_token = 0;
    double achieved_gbs = 0.0;
    double peak_gbs = kPeakBandwidthGBs;
    int context = 0;
    ReportStatus status = ReportStatus::Ok;

    void fill_roofline(double decode_ms);
    std::string line() const;
};

const char* report_engine_name(ReportEngine engine);
const char* report_backend_name(ReportBackend backend);
const char* report_status_name(ReportStatus status);

ReportBackend report_backend(Device device);
void print_report(const DecodeReport& report);

}  // namespace vesper
