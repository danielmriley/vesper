#pragma once

#include <cstddef>
#include <string>

namespace vesper {

#ifdef VESPER_USE_HIP
inline constexpr bool kHipBuilt = true;
#else
inline constexpr bool kHipBuilt = false;
#endif

// Probe without throwing. Sets GPU_MAX_HW_QUEUES=1 before the first HIP call.
bool hip_available();

// Require a live gfx1201 device. Safe to call more than once.
void hip_init();

std::string hip_arch();
std::string hip_device_name();

void* hip_alloc(std::size_t bytes);
void* hip_alloc_uninit(std::size_t bytes);
void hip_free(void* ptr);
void hip_copy_h2d(void* dst, const void* src, std::size_t bytes);
void hip_copy_d2h(void* dst, const void* src, std::size_t bytes);
void hip_copy_d2d(void* dst, const void* src, std::size_t bytes);
void hip_fill(float* dst, float value, std::size_t n_elems);
void hip_synchronize();

// Capture one GDN+FFN layer after the first eager token. Replay is the same
// pointers every later token. Attention stays eager: K/V slots move with pos.
bool hip_graph_ready(int slot);
void hip_graph_capture_begin(int slot);
void hip_graph_capture_end(int slot);
void hip_graph_launch(int slot);
void hip_graph_destroy_all();

}  // namespace vesper
