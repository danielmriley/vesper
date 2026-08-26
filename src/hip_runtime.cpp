#include "vesper/hip.h"

#include "vesper/target.h"
#include "vesper/types.h"

#ifdef VESPER_USE_HIP
#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>
#endif

namespace vesper {
namespace {

#ifdef VESPER_USE_HIP

#define VESPER_HIP_CHECK(cmd)                                                              \
    do {                                                                                   \
        const hipError_t err_ = (cmd);                                                     \
        if (err_ != hipSuccess) {                                                          \
            fail(std::string("HIP: ") + hipGetErrorString(err_) + " (" + #cmd + ")");      \
        }                                                                                  \
    } while (0)

std::once_flag g_env_once;
std::once_flag g_init_once;
bool g_available = false;
std::string g_arch;
std::string g_name;
hipStream_t g_stream = nullptr;

void pin_idle_queues() {
    const char* existing = std::getenv("GPU_MAX_HW_QUEUES");
    if (existing == nullptr || existing[0] == '\0') {
        // R9700 HIP idle-power bug. Must be set before the first runtime call.
        setenv("GPU_MAX_HW_QUEUES", "1", 1);
    }
}

std::string arch_from_prop(const hipDeviceProp_t& prop) {
    const std::string raw = prop.gcnArchName;
    const auto colon = raw.find(':');
    return colon == std::string::npos ? raw : raw.substr(0, colon);
}

void probe_unlocked() {
    pin_idle_queues();
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count <= 0) {
        g_available = false;
        return;
    }
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, 0) != hipSuccess) {
        g_available = false;
        return;
    }
    g_arch = arch_from_prop(prop);
    g_name = prop.name;
    g_available = (g_arch == kHipArch);
}

void require_gfx1201() {
    if (g_arch != kHipArch) {
        fail("v1 HIP target is " + std::string(kHipArch) + " (Radeon AI Pro R9700); found " +
             (g_arch.empty() ? std::string("no device") : g_arch));
    }
    hipDeviceProp_t prop{};
    VESPER_HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    if (prop.warpSize != kWavefront) {
        fail("expected wave" + std::to_string(kWavefront) + " on " + std::string(kHipArch) +
             ", device reports " + std::to_string(prop.warpSize));
    }
}

#endif  // VESPER_USE_HIP

}  // namespace

bool hip_available() {
#ifdef VESPER_USE_HIP
    std::call_once(g_env_once, probe_unlocked);
    return g_available;
#else
    return false;
#endif
}

void hip_init() {
#ifdef VESPER_USE_HIP
    if (!hip_available()) {
        if (g_arch.empty()) {
            fail("HIP was built but no GPU is visible");
        }
        fail("v1 HIP target is " + std::string(kHipArch) +
             " (Radeon AI Pro R9700); found " + g_arch);
    }
    std::call_once(g_init_once, []() {
        VESPER_HIP_CHECK(hipSetDevice(0));
        require_gfx1201();
        VESPER_HIP_CHECK(hipStreamCreateWithFlags(&g_stream, hipStreamNonBlocking));
    });
#else
    fail("HIP is not built; configure with -DVESPER_USE_HIP=ON");
#endif
}

std::string hip_arch() {
#ifdef VESPER_USE_HIP
    (void)hip_available();
    return g_arch;
#else
    return {};
#endif
}

std::string hip_device_name() {
#ifdef VESPER_USE_HIP
    (void)hip_available();
    return g_name;
#else
    return {};
#endif
}

#ifdef VESPER_USE_HIP
void* hip_malloc_aligned(std::size_t bytes, bool zero) {
    hip_init();
    const std::size_t aligned = (bytes + static_cast<std::size_t>(kCachelineBytes) - 1) &
                                ~(static_cast<std::size_t>(kCachelineBytes) - 1);
    const std::size_t request = aligned == 0 ? static_cast<std::size_t>(kCachelineBytes) : aligned;
    void* ptr = nullptr;
    VESPER_HIP_CHECK(hipMalloc(&ptr, request));
    if ((reinterpret_cast<std::uintptr_t>(ptr) % static_cast<unsigned>(kCachelineBytes)) != 0) {
        hipFree(ptr);
        fail("hipMalloc was not 256-byte aligned");
    }
    if (zero) {
        VESPER_HIP_CHECK(hipMemset(ptr, 0, request));
    }
    return ptr;
}

void hip_copy_chunked(void* dst, const void* src, std::size_t bytes, hipMemcpyKind kind) {
    hip_init();
    auto* d = static_cast<std::byte*>(dst);
    const auto* s = static_cast<const std::byte*>(src);
    std::size_t off = 0;
    while (off < bytes) {
        const std::size_t n = std::min(bytes - off, kHipCopyChunkBytes);
        VESPER_HIP_CHECK(hipMemcpy(d + off, s + off, n, kind));
        off += n;
    }
}
#endif

void* hip_alloc(std::size_t bytes) {
#ifdef VESPER_USE_HIP
    return hip_malloc_aligned(bytes, true);
#else
    (void)bytes;
    fail("HIP is not built; configure with -DVESPER_USE_HIP=ON");
#endif
}

void* hip_alloc_uninit(std::size_t bytes) {
#ifdef VESPER_USE_HIP
    return hip_malloc_aligned(bytes, false);
#else
    (void)bytes;
    fail("HIP is not built; configure with -DVESPER_USE_HIP=ON");
#endif
}

void hip_free(void* ptr) {
#ifdef VESPER_USE_HIP
    if (ptr == nullptr) {
        return;
    }
    VESPER_HIP_CHECK(hipFree(ptr));
#else
    (void)ptr;
#endif
}

void hip_copy_h2d(void* dst, const void* src, std::size_t bytes) {
#ifdef VESPER_USE_HIP
    hip_copy_chunked(dst, src, bytes, hipMemcpyHostToDevice);
#else
    (void)dst;
    (void)src;
    (void)bytes;
    fail("HIP is not built");
#endif
}

void hip_copy_d2h(void* dst, const void* src, std::size_t bytes) {
#ifdef VESPER_USE_HIP
    hip_copy_chunked(dst, src, bytes, hipMemcpyDeviceToHost);
#else
    (void)dst;
    (void)src;
    (void)bytes;
    fail("HIP is not built");
#endif
}

void hip_copy_d2d(void* dst, const void* src, std::size_t bytes) {
#ifdef VESPER_USE_HIP
    hip_init();
    VESPER_HIP_CHECK(hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice, g_stream));
#else
    (void)dst;
    (void)src;
    (void)bytes;
    fail("HIP is not built");
#endif
}

void hip_fill(float* dst, float value, std::size_t n_elems) {
#ifdef VESPER_USE_HIP
    if (n_elems == 0) {
        return;
    }
    std::vector<float> host(n_elems, value);
    hip_copy_h2d(dst, host.data(), n_elems * sizeof(float));
#else
    (void)dst;
    (void)value;
    (void)n_elems;
    fail("HIP is not built");
#endif
}

void hip_synchronize() {
#ifdef VESPER_USE_HIP
    hip_init();
    if (g_stream != nullptr) {
        VESPER_HIP_CHECK(hipStreamSynchronize(g_stream));
    }
    VESPER_HIP_CHECK(hipDeviceSynchronize());
#endif
}

#ifdef VESPER_USE_HIP
struct HipGraphSlot {
    hipGraphExec_t exec = nullptr;
    bool ready = false;
};

std::vector<HipGraphSlot> g_graphs;
int g_capturing = -1;
bool g_graphs_off = false;

void graph_ensure(int slot) {
    check(slot >= 0, "hip graph slot");
    if (static_cast<int>(g_graphs.size()) <= slot) {
        g_graphs.resize(static_cast<std::size_t>(slot) + 1u);
    }
}

void disable_graphs() {
    g_graphs_off = true;
    g_capturing = -1;
}
#endif

#ifdef VESPER_USE_HIP
hipStream_t hip_stream() {
    hip_init();
    return g_stream;
}
#endif

void hip_upload_i32(int* dst, const int* host) {
#ifdef VESPER_USE_HIP
    hip_init();
    check(dst != nullptr && host != nullptr, "hip_upload_i32 null");
    VESPER_HIP_CHECK(hipMemcpyAsync(dst, host, sizeof(int), hipMemcpyHostToDevice, g_stream));
#else
    (void)dst;
    (void)host;
    fail("HIP is not built");
#endif
}

bool hip_graph_ready(int slot) {
#ifdef VESPER_USE_HIP
    if (g_graphs_off || slot < 0 || slot >= static_cast<int>(g_graphs.size())) {
        return false;
    }
    return g_graphs[static_cast<std::size_t>(slot)].ready;
#else
    (void)slot;
    return false;
#endif
}

bool hip_graph_try_begin(int slot) {
#ifdef VESPER_USE_HIP
    if (g_graphs_off || slot < 0) {
        return false;
    }
    hip_init();
    if (g_capturing >= 0) {
        disable_graphs();
        return false;
    }
    graph_ensure(slot);
    const hipError_t rc = hipStreamBeginCapture(g_stream, hipStreamCaptureModeGlobal);
    if (rc != hipSuccess) {
        disable_graphs();
        return false;
    }
    g_capturing = slot;
    return true;
#else
    (void)slot;
    return false;
#endif
}

bool hip_graph_try_end(int slot) {
#ifdef VESPER_USE_HIP
    if (g_graphs_off || g_capturing != slot) {
        disable_graphs();
        return false;
    }
    hipGraph_t graph = nullptr;
    const hipError_t end_rc = hipStreamEndCapture(g_stream, &graph);
    g_capturing = -1;
    if (end_rc != hipSuccess) {
        disable_graphs();
        return false;
    }
    hipGraphExec_t exec = nullptr;
    const hipError_t inst = hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    (void)hipGraphDestroy(graph);
    if (inst != hipSuccess) {
        disable_graphs();
        return false;
    }
    HipGraphSlot& dst = g_graphs[static_cast<std::size_t>(slot)];
    if (dst.exec != nullptr) {
        (void)hipGraphExecDestroy(dst.exec);
        dst.exec = nullptr;
        dst.ready = false;
    }
    dst.exec = exec;
    dst.ready = true;
    return true;
#else
    (void)slot;
    return false;
#endif
}

void hip_graph_abort() {
#ifdef VESPER_USE_HIP
    if (g_capturing < 0) {
        return;
    }
    hipGraph_t graph = nullptr;
    (void)hipStreamEndCapture(g_stream, &graph);
    if (graph != nullptr) {
        (void)hipGraphDestroy(graph);
    }
    disable_graphs();
#endif
}

void hip_graph_launch(int slot) {
#ifdef VESPER_USE_HIP
    check(hip_graph_ready(slot), "HIP graph launch of empty slot");
    VESPER_HIP_CHECK(hipGraphLaunch(g_graphs[static_cast<std::size_t>(slot)].exec, g_stream));
#else
    (void)slot;
    fail("HIP is not built");
#endif
}

void hip_graph_destroy_all() {
#ifdef VESPER_USE_HIP
    if (g_capturing >= 0) {
        hip_graph_abort();
    }
    for (HipGraphSlot& slot : g_graphs) {
        if (slot.exec != nullptr) {
            (void)hipGraphExecDestroy(slot.exec);
            slot.exec = nullptr;
        }
        slot.ready = false;
    }
    g_graphs.clear();
    g_capturing = -1;
#endif
}

}  // namespace vesper
