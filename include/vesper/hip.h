#pragma once

#include <cstddef>
#include <string>

#ifdef VESPER_USE_HIP
#include <hip/hip_runtime_api.h>
#endif

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
// H2D/D2H join the decode stream and wait. D2D is async on that stream so
// a graph can record it. Default-stream memcpy does not wait for
// hipStreamNonBlocking work.
void hip_copy_h2d(void* dst, const void* src, std::size_t bytes);
void hip_copy_d2h(void* dst, const void* src, std::size_t bytes);
void hip_copy_d2d(void* dst, const void* src, std::size_t bytes);
void hip_fill(float* dst, float value, std::size_t n_elems);
void hip_synchronize();

#ifdef VESPER_USE_HIP
// Decode work and graph capture share this stream. Default-stream capture is illegal.
hipStream_t hip_stream();
#endif

// Async H2D of one int on the decode stream. host must stay live until the stream syncs.
void hip_upload_i32(int* dst, const int* host);

// Capture decode in layer chunks at Engine HIP init, not inside the
// timed generate loop. token/pos live in device memory so attention
// K/V scatter and RoPE stay legal on replay. try_* return false and
// disable further capture if RDNA rejects graphs. reset() clears that
// disable so init can retry a smaller chunk.
bool hip_graph_ready(int slot);
bool hip_graph_try_begin(int slot);
bool hip_graph_try_end(int slot);
void hip_graph_abort();
void hip_graph_reset();
void hip_graph_launch(int slot);
void hip_graph_destroy_all();

}  // namespace vesper
