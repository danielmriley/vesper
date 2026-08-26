#include "vesper/report.h"

#include "vesper/types.h"

#include <iostream>
#include <sstream>

namespace vesper {

const char* report_engine_name(ReportEngine engine) {
    switch (engine) {
        case ReportEngine::Vesper:
            return "vesper";
        case ReportEngine::LlamaCpp:
            return "llamacpp";
    }
    throw std::logic_error("unhandled ReportEngine");
}

const char* report_backend_name(ReportBackend backend) {
    switch (backend) {
        case ReportBackend::CPU:
            return "cpu";
        case ReportBackend::HIP:
            return "hip";
        case ReportBackend::Vulkan:
            return "vulkan";
    }
    throw std::logic_error("unhandled ReportBackend");
}

const char* report_status_name(ReportStatus status) {
    switch (status) {
        case ReportStatus::Ok:
            return "ok";
        case ReportStatus::Unsupported:
            return "unsupported";
    }
    throw std::logic_error("unhandled ReportStatus");
}

ReportBackend report_backend(Device device) {
    switch (device) {
        case Device::CPU:
            return ReportBackend::CPU;
        case Device::HIP:
            return ReportBackend::HIP;
    }
    throw std::logic_error("unhandled Device");
}

void DecodeReport::fill_roofline(double decode_ms) {
    const double seconds = decode_ms / 1000.0;
    const double moved = static_cast<double>(bytes_per_token) * static_cast<double>(new_tokens);
    achieved_gbs = (seconds > 0.0) ? (moved / seconds) / 1e9 : 0.0;
    peak_gbs = kPeakBandwidthGBs;
}

std::string DecodeReport::line() const {
    check(!model.empty(), "DecodeReport missing model");
    check(!quant.empty(), "DecodeReport missing quant");
    check(!arch.empty(), "DecodeReport missing arch");
    std::ostringstream out;
    out << "engine=" << report_engine_name(engine)
        << " backend=" << report_backend_name(backend)
        << " model=" << model
        << " quant=" << quant
        << " arch=" << arch
        << " prompt_tokens=" << prompt_tokens
        << " new_tokens=" << new_tokens
        << " prefill_tps=" << prefill_tps
        << " decode_tps=" << decode_tps
        << " bytes_per_token=" << bytes_per_token
        << " achieved_gbs=" << achieved_gbs
        << " peak_gbs=" << peak_gbs
        << " context=" << context
        << " status=" << report_status_name(status)
        << " graphs=" << graphs
        << " ids=" << (ids.empty() ? "-" : ids);
    return out.str();
}

void print_report(const DecodeReport& report) {
    std::cout << report.line() << "\n";
}

}  // namespace vesper
