#include "vesper/rdna4.h"

#include "vesper/dotq.h"
#include "vesper/hip.h"
#include "vesper/multi_row.h"
#include "vesper/q4k.h"
#include "vesper/q5k.h"
#include "vesper/q6k.h"
#include "vesper/q8.h"
#include "vesper/q8x.h"
#include "vesper/target.h"
#include "vesper/types.h"
#include "vesper/weight.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace vesper {
namespace rdna4 {
namespace {

#define VESPER_HIP_CHECK(cmd)                                                              \
    do {                                                                                   \
        const hipError_t err_ = (cmd);                                                     \
        if (err_ != hipSuccess) {                                                          \
            fail(std::string("HIP: ") + hipGetErrorString(err_) + " (" + #cmd + ")");      \
        }                                                                                  \
    } while (0)

// __shfl_down leaves the reduction in lane 0. Broadcast it: Q8_1
// quantize, GDN delta, attn, and the block reduce consume every lane.
__device__ float wave32_sum(float v) {
    v += __shfl_down(v, 16, kWavefront);
    v += __shfl_down(v, 8, kWavefront);
    v += __shfl_down(v, 4, kWavefront);
    v += __shfl_down(v, 2, kWavefront);
    v += __shfl_down(v, 1, kWavefront);
    return __shfl(v, 0, kWavefront);
}

__device__ float wave32_max(float v) {
    v = fmaxf(v, __shfl_down(v, 16, kWavefront));
    v = fmaxf(v, __shfl_down(v, 8, kWavefront));
    v = fmaxf(v, __shfl_down(v, 4, kWavefront));
    v = fmaxf(v, __shfl_down(v, 2, kWavefront));
    v = fmaxf(v, __shfl_down(v, 1, kWavefront));
    return __shfl(v, 0, kWavefront);
}

// One shuffle per wave, then nwarps LDS slots. Do not copy CDNA wave64.
__device__ float block_reduce_sum(float value, float* smem) {
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid % kWavefront;
    const int warp = tid / kWavefront;
    const int nwarps = static_cast<int>(blockDim.x) / kWavefront;
    value = wave32_sum(value);
    if (lane == 0) {
        smem[warp] = value;
    }
    __syncthreads();
    float v = 0.0f;
    if (tid < kWavefront) {
        v = (tid < nwarps) ? smem[tid] : 0.0f;
        v = wave32_sum(v);
    }
    if (tid == 0) {
        smem[0] = v;
    }
    __syncthreads();
    return smem[0];
}

__device__ float block_reduce_max(float value, float* smem) {
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid % kWavefront;
    const int warp = tid / kWavefront;
    const int nwarps = static_cast<int>(blockDim.x) / kWavefront;
    value = wave32_max(value);
    if (lane == 0) {
        smem[warp] = value;
    }
    __syncthreads();
    float v = -INFINITY;
    if (tid < kWavefront) {
        v = (tid < nwarps) ? smem[tid] : -INFINITY;
        v = wave32_max(v);
    }
    if (tid == 0) {
        smem[0] = v;
    }
    __syncthreads();
    return smem[0];
}

__device__ int decode_i32(int val, const int* ptr) {
    return ptr != nullptr ? *ptr : val;
}

// llama.cpp MMVQ uses hardware __half / __half2. The software IEEE path
// in src/q8.cpp stays the CPU oracle. GGUF scales are normal numbers.
__device__ __forceinline__ float load_f16(const void* p) {
    return __half2float(*reinterpret_cast<const __half*>(p));
}

__device__ __forceinline__ float2 load_f16x2(const void* p) {
    return __half22float2(*reinterpret_cast<const __half2*>(p));
}

// Packed Q8/Q6 block d, unique per thread. Q4 d/dmin is cached: every
// thread on a super rereads the same 4 bytes. Q8_1 xd stays a cached
// float load.
__device__ __forceinline__ float load_wf16(const void* p) {
#if defined(__has_builtin) && __has_builtin(__builtin_nontemporal_load)
    const __half h = __builtin_nontemporal_load(reinterpret_cast<const __half*>(p));
    return __half2float(h);
#else
    return load_f16(p);
#endif
}

__device__ __forceinline__ float2 load_wf16x2(const void* p) {
#if defined(__has_builtin) && __has_builtin(__builtin_nontemporal_load)
    const __half2 h = __builtin_nontemporal_load(reinterpret_cast<const __half2*>(p));
    return __half22float2(h);
#else
    return load_f16x2(p);
#endif
}

// llama.cpp RDNA4 MMVQ: 1 output row, up to 8 waves. Q8_1 x lives in a
// process scratch. Official launch is often 5 or 6 waves.
static_assert(kQ8XBlockElems == kWavefront, "fused Q8_1 uses one lane per block elem");

struct Q8XDevice {
    std::int8_t* qs = nullptr;
    float* d = nullptr;
    // Allocated so fused kernels keep the same signature as the CPU oracle.
    // HIP MMVQ never reads it.
    float* sum = nullptr;
    int cap = 0;
    int ready_cols = 0;
};

Q8XDevice g_q8x;

void q8x_ensure(int cols) {
    if (cols <= g_q8x.cap) {
        return;
    }
    if (g_q8x.qs != nullptr) {
        hip_synchronize();
        hip_free(g_q8x.qs);
        hip_free(g_q8x.d);
        hip_free(g_q8x.sum);
    }
    const int blocks = cols / kQ8XBlockElems;
    g_q8x.qs = static_cast<std::int8_t*>(hip_alloc(static_cast<std::size_t>(cols)));
    g_q8x.d = static_cast<float*>(hip_alloc(sizeof(float) * static_cast<std::size_t>(blocks)));
    g_q8x.sum = static_cast<float*>(hip_alloc(sizeof(float) * static_cast<std::size_t>(blocks)));
    g_q8x.cap = cols;
    g_q8x.ready_cols = 0;
}

__device__ void quantize_q8x_value(std::int8_t* __restrict__ qs, float* __restrict__ d,
                                   float* __restrict__ sum, int b, int lane, float v) {
    const float amax = wave32_max(fabsf(v));
    const float db = amax / 127.0f;
    const float inv = (db > 0.0f) ? (1.0f / db) : 0.0f;
    int q = static_cast<int>(roundf(v * inv));
    if (q > 127) {
        q = 127;
    }
    if (q < -127) {
        q = -127;
    }
    qs[b * kQ8XBlockElems + lane] = static_cast<std::int8_t>(q);
    if (lane == 0) {
        d[b] = db;
    }
    // HIP MMVQ matches llama.cpp vec_dot_q4_K_q8_1_impl_vmmq: the Q4/Q5 min
    // term is dp4a(0x01010101, u). A wave reduce of sum(qs) is unused on
    // every official fused write and on FFN down's standalone quantize.
    (void)sum;
}

__device__ void quantize_q8x_block(std::int8_t* __restrict__ qs, float* __restrict__ d,
                                   float* __restrict__ sum, const float* __restrict__ x, int b,
                                   int lane) {
    quantize_q8x_value(qs, d, sum, b, lane, x[b * kQ8XBlockElems + lane]);
}

template<int NElems, int NThreads>
__device__ void quantize_q8x_from_slice_unroll(std::int8_t* __restrict__ qs, float* __restrict__ d,
                                               float* __restrict__ sum, const float* __restrict__ base,
                                               int elem0) {
    static_assert(NElems > 0 && (NElems % kQ8XBlockElems) == 0, "Q8_1 unroll width");
    static_assert(NThreads > 0 && (NThreads % kWavefront) == 0, "Q8_1 unroll threads");
    constexpr int nblocks = NElems / kQ8XBlockElems;
    constexpr int nwarps = NThreads / kWavefront;
    static_assert(nblocks % nwarps == 0, "Q8_1 unroll trips");
    constexpr int trips = nblocks / nwarps;
    const int lane = static_cast<int>(threadIdx.x) % kWavefront;
    const int warp = static_cast<int>(threadIdx.x) / kWavefront;
    const int block0 = elem0 / kQ8XBlockElems;
#pragma unroll
    for (int t = 0; t < trips; ++t) {
        quantize_q8x_block(qs, d, sum, base, block0 + warp + t * nwarps, lane);
    }
}

__device__ void quantize_q8x_from_slice(std::int8_t* __restrict__ qs, float* __restrict__ d,
                                        float* __restrict__ sum, const float* __restrict__ base,
                                        int elem0, int n_elems) {
    const int lane = static_cast<int>(threadIdx.x) % kWavefront;
    const int warp = static_cast<int>(threadIdx.x) / kWavefront;
    const int nwarps = static_cast<int>(blockDim.x) / kWavefront;
    const int nblocks = n_elems / kQ8XBlockElems;
    const int block0 = elem0 / kQ8XBlockElems;
    for (int b = warp; b < nblocks; b += nwarps) {
        quantize_q8x_block(qs, d, sum, base, block0 + b, lane);
    }
}

__device__ void quantize_q8x_from_vec(std::int8_t* __restrict__ qs, float* __restrict__ d,
                                      float* __restrict__ sum, const float* __restrict__ x, int n) {
    quantize_q8x_from_slice(qs, d, sum, x, 0, n);
}

__global__ void __launch_bounds__(256)
quantize_q8x_kernel(std::int8_t* __restrict__ qs, float* __restrict__ d, float* __restrict__ sum,
                    const float* __restrict__ x, int nblocks) {
    const int b = static_cast<int>(blockIdx.x) * kQuantizeRowsPerWg +
                  static_cast<int>(threadIdx.x) / kWavefront;
    const int lane = static_cast<int>(threadIdx.x) % kWavefront;
    if (b >= nblocks) {
        return;
    }
    quantize_q8x_block(qs, d, sum, x, b, lane);
}

// Official 27B writes Q8_1 in the kernel that produces x for the next
// packed GEMV: copy/add/final rmsnorm (5120), attn_decode (6144), and
// rmsnorm_silu_mul (6144). The matching GEMV skips its quantize launch
// once. down is one WG per row, so it still quantizes.
void bind_fused_q8x(int n, bool write_is_x, std::int8_t** qs, float** d, float** sum) {
    *qs = nullptr;
    *d = nullptr;
    *sum = nullptr;
    if (!write_is_x || !q8x_can_fuse(n)) {
        q8x_clear_ready(&g_q8x.ready_cols);
        return;
    }
    q8x_ensure(n);
    *qs = g_q8x.qs;
    *d = g_q8x.d;
    *sum = g_q8x.sum;
}

void q8x_quantize(const float* x, int cols) {
    check(cols > 0 && (cols % kQ8XBlockElems) == 0, "q8x: cols must be a multiple of 32");
    if (q8x_take_ready(&g_q8x.ready_cols, cols)) {
        return;
    }
    q8x_ensure(cols);
    const int nblocks = cols / kQ8XBlockElems;
    const int grid = (nblocks + kQuantizeRowsPerWg - 1) / kQuantizeRowsPerWg;
    hipLaunchKernelGGL(quantize_q8x_kernel, dim3(grid), dim3(kGemvWorkgroup), 0, hip_stream(), g_q8x.qs,
                       g_q8x.d, g_q8x.sum, x, nblocks);
    VESPER_HIP_CHECK(hipGetLastError());
}

__device__ void write_gemv(float* y, int row, float acc, const float* addend) {
    if (addend != nullptr) {
        acc += addend[row];
    }
    y[row] = acc;
}

__device__ int gemv_row() {
    return static_cast<int>(blockIdx.x);
}

__device__ int gemv_tid() {
    return static_cast<int>(threadIdx.x);
}

__device__ int gemv_lane() {
    return static_cast<int>(threadIdx.x) % kWavefront;
}

__device__ int gemv_warp() {
    return static_cast<int>(threadIdx.x) / kWavefront;
}

// llama.cpp MMVQ: one barrier, only tid 0 needs the sum. block_reduce_sum
// broadcasts to every lane and syncs again. That extra barrier is on every
// official Q4/Q8/Q6 workgroup, including lm_head's 248320.
__device__ float reduce_wg(float acc) {
    __shared__ float smem[kGemvWaves];
    const int tid = gemv_tid();
    const int lane = tid % kWavefront;
    const int warp = tid / kWavefront;
    const int nwarps = static_cast<int>(blockDim.x) / kWavefront;
    acc = wave32_sum(acc);
    if (lane == 0) {
        smem[warp] = acc;
    }
    __syncthreads();
    if (tid != 0) {
        return 0.0f;
    }
    float v = 0.0f;
    for (int w = 0; w < nwarps; ++w) {
        v += smem[w];
    }
    return v;
}

// Official SwiGLU is two Q4 dots in one WG. Two reduce_wg calls were two
// barriers on every ffn_up row (17408 WGs). Only tid 0 reads the pair.
__device__ void reduce_wg2(float a, float b, float* out_a, float* out_b) {
    __shared__ float smem[2 * kGemvWaves];
    const int tid = gemv_tid();
    const int lane = tid % kWavefront;
    const int warp = tid / kWavefront;
    const int nwarps = static_cast<int>(blockDim.x) / kWavefront;
    a = wave32_sum(a);
    b = wave32_sum(b);
    if (lane == 0) {
        smem[warp] = a;
        smem[kGemvWaves + warp] = b;
    }
    __syncthreads();
    if (tid != 0) {
        return;
    }
    float va = 0.0f;
    float vb = 0.0f;
    for (int w = 0; w < nwarps; ++w) {
        va += smem[w];
        vb += smem[kGemvWaves + w];
    }
    *out_a = va;
    *out_b = vb;
}

static_assert((kQ8VdrMmvq * kGemvWorkgroup) % kQ8Qi == 0,
              "Q8 MMVQ thread mapping needs workgroup divisible by QI/VDR");
static_assert(kQ4MmvqThreadsPerSuper * kQ4MmvqSuperStride == kGemvWorkgroup,
              "Q4 pair map must cover the workgroup");
static_assert(kQ4MmvqMidThreadsPerSuper == 4, "Q4 mid map is four threads per super");
static_assert(kQ4MmvqMidThreadsPerSuper * kQ4MmvqMidSuperStride == kGemvWorkgroup,
              "Q4 mid map must cover the workgroup");
static_assert(kQ4MmvqOctThreadsPerSuper == 2, "Q4 oct map is two threads per super");
static_assert(kQ4MmvqOctThreadsPerSuper * kQ4MmvqOctSuperStride == kGemvWorkgroup,
              "Q4 oct map must cover the workgroup");
static_assert(kQ4MmvqDownThreadsPerSuper == 1, "Q4 down map is one thread per super");
static_assert(kQ4MmvqDownThreadsPerSuper * kQ4MmvqDownSuperStride == kGemvWorkgroup,
              "Q4 down map must cover the workgroup");
static_assert(q4_mmvq_threads(16) == kQ4MmvqThreadsPerSuper, "narrow Q4 stays on the pair map");
static_assert(q4_mmvq_threads(20) == kQ4MmvqMidThreadsPerSuper,
              "official SwiGLU uses the quad map");
static_assert(q4_mmvq_threads(64) == kQ4MmvqMidThreadsPerSuper, "64 supers fit the mid map");
static_assert(q4_mmvq_threads(68) == kQ4MmvqOctThreadsPerSuper,
              "official FFN down uses one oct per thread");
static_assert(q4_mmvq_threads(128) == kQ4MmvqOctThreadsPerSuper, "128 supers fit the oct map");
static_assert(q4_mmvq_threads(129) == kQ4MmvqDownThreadsPerSuper,
              "129 supers fall through to one thread per super");
static_assert((20 + kQ4MmvqMidSuperStride - 1) / kQ4MmvqMidSuperStride == 1,
              "official SwiGLU is one K-trip on the quad map");
static_assert((68 + kQ4MmvqOctSuperStride - 1) / kQ4MmvqOctSuperStride == 1,
              "official FFN down is one K-trip on the oct map");
static_assert(kQ8MmvqThreadsPerBlock == 1,
              "HIP Q8 MMVQ owns a full block per thread");
static_assert(kQ8MmvqBlocksPerThread == 2, "HIP Q8 MMVQ does two consecutive blocks");
static_assert(kQ8MmvqThreadsPerBlock * kQ8MmvqPerIter == kGemvWorkgroup,
              "Q8 block map must cover the workgroup");
static_assert((320 + kQ8MmvqPerIter * kQ8MmvqBlocksPerThread - 1) /
                      (kQ8MmvqPerIter * kQ8MmvqBlocksPerThread) ==
                  1,
              "Q8 K 10240 is one K-trip at two blocks per thread");
static_assert(kQ8Qi % kQ8MmvqThreadsPerBlock == 0,
              "Q8 block ints must split across the threads on a block");
static_assert((kQ8Qi / kQ8MmvqThreadsPerBlock) % kQ8VdrMmvq == 0,
              "each Q8 thread does a whole number of VDR slices");
static_assert(kQ6MmvqThreadsPerSuper == 4,
              "HIP Q6 MMVQ is eight iqs per thread");
static_assert((kQ6KQi / kQ6MmvqThreadsPerSuper) == 8,
              "each Q6 thread does one oct");
static_assert(kQ6KQi % kQ6MmvqThreadsPerSuper == 0,
              "Q6 iqs must split across the threads on a super");
static_assert(kQ6MmvqThreadsPerSuper * kQ6MmvqSuperStride == kGemvWorkgroup,
              "Q6 oct map must cover the workgroup");
static_assert(q4_mmvq_launch(5120) == 96, "official SwiGLU launch is 3 waves");
static_assert(q4_mmvq_launch(17408) == 160, "official FFN down launch is 5 waves");
static_assert(q8_mmvq_launch(5120) == 96, "official Q8 K 5120 launch is 3 waves");
static_assert(q8_mmvq_launch(6144) == 96, "official Q8 K 6144 launch is 3 waves");
static_assert(q8_mmvq_launch(10240) == 160, "Q8 K 10240 launch is 5 waves");
static_assert(mmvq_launch_bounds(q4_mmvq_launch(5120)) == kMmvqLaunch96,
              "official SwiGLU launch_bounds is 96");
static_assert(mmvq_launch_bounds(q4_mmvq_launch(17408)) == kMmvqLaunch160,
              "official FFN down launch_bounds is 160");
static_assert(mmvq_launch_bounds(q8_mmvq_launch(5120)) == kMmvqLaunch96,
              "official Q8 K 5120 launch_bounds is 96");
static_assert(mmvq_launch_bounds(q8_mmvq_launch(6144)) == kMmvqLaunch96,
              "official Q8 K 6144 launch_bounds is 96");
static_assert(mmvq_launch_bounds(q8_mmvq_launch(10240)) == kMmvqLaunch160,
              "Q8 K 10240 launch_bounds is 160");
static_assert(mmvq_launch_bounds(q6_mmvq_launch(5120)) == kMmvqLaunch96,
              "official lm_head launch_bounds is 96");
static_assert(mmvq_launch_bounds(q6_mmvq_launch(6144)) == kMmvqLaunch96,
              "official o_proj launch_bounds is 96");
static_assert(mmvq_launch_bounds(128) == kMmvqLaunch256,
              "4-wave leftover keeps the 8-wave VGPR budget");
static_assert(q8_mmvq_tight(5120), "official Q8 K 5120 has no leftover block");
static_assert(q8_mmvq_tight(6144), "official Q8 K 6144 has no leftover block");
static_assert(q8_mmvq_tight(10240), "Q8 K 10240 has no leftover block");
static_assert(!q8_mmvq_tight(96), "3-block Q8 walk keeps the tail check");
static_assert(q6_mmvq_launch(5120) == 96, "official lm_head launch is 3 waves");
static_assert(q6_mmvq_launch(6144) == 96, "official o_proj launch is 3 waves");
static_assert(kGdnConvKernel == 4, "official GDN conv is k=4");
static_assert(kOfficialHidden == 5120, "official hidden is 5120");
static_assert(kOfficialHidden % kGemvWorkgroup == 0, "official hidden fills 256-thread trips");
static_assert((kOfficialHidden / 4) % kGemvWorkgroup == 0,
              "official hidden is an integer number of float4 trips");
static_assert(kOfficialGdnDim == 128, "official GDN dim is 128");
static_assert(row_workgroup(kOfficialGdnDim) == kOfficialGdnDim,
              "official GDN row WG is one thread per dim");
static_assert(kOfficialHeadDim == 256, "official attn head dim is 256");
static_assert(kOfficialVocab == 248320, "official vocab is 248320");
static_assert(argmax_one_trip(kOfficialVocab), "official argmax is leftover-free");
static_assert(argmax_partial_blocks(kOfficialVocab) == kOfficialVocab / kGemvWorkgroup,
              "official argmax is 970 partial WGs");
static_assert(gdn_delta_tight(kOfficialGdnDim), "official GDN fills 4 shards");
static_assert(gdn_delta_tight(kOfficialHeadDim), "official attn fills 8 shards");
static_assert(!gdn_delta_tight(16), "tiny dim 16 has a partial shard");
static_assert(kOfficialRopeDim == 64, "official rope dim is 64");
static_assert((kOfficialRopeDim % 2) == 0, "official rope dim is even");
static_assert(kOfficialRopeDim <= kOfficialHeadDim, "official rope sits inside head dim");
static_assert(kGemvWorkgroup == kOfficialHeadDim, "attn prepare is one thread per official dim");
static_assert(q8x_warp_trips(kOfficialHidden, kGemvWorkgroup) == 20,
              "official hidden Q8_1 is 20 blocks per warp");
static_assert(q8x_warp_trips(kOfficialGdnDim, row_workgroup(kOfficialGdnDim)) == 1,
              "official GDN Q8_1 is one block per warp");
static_assert(q4_mmvq_one_trip(5120), "official SwiGLU Q4 is leftover-free");
static_assert(q4_mmvq_one_trip(17408), "official FFN down Q4 is leftover-free");
static_assert(!q4_mmvq_one_trip(257 * 256), "257 Q4 supers need a leftover trip");
static_assert(q8_soa_row_bytes(160) == 160u * kQ8BlockBytes, "HIP Q8 K 5120 SoA is same size");
static_assert(q8_soa_row_bytes(192) == 192u * kQ8BlockBytes, "HIP Q8 K 6144 SoA is same size");
static_assert(q4k_soa_row_bytes(20) == 20u * 144u, "HIP Q4 SwiGLU SoA is same size");
static_assert(q4k_soa_row_bytes(68) == 68u * 144u, "HIP Q4 down SoA is same size");
static_assert(q8_mmvq_one_trip(5120), "official Q8 K 5120 is leftover-free");
static_assert(q8_mmvq_one_trip(6144), "official Q8 K 6144 is leftover-free");
static_assert(q8_mmvq_one_trip(10240), "Q8 K 10240 is leftover-free");
static_assert(!q8_mmvq_one_trip(257 * 32 * kQ8MmvqBlocksPerThread),
              "257 Q8 thread slots need a leftover trip");
static_assert(q6k_soa_row_bytes(24) == 24u * 210u, "HIP Q6 o_proj SoA is same size");
static_assert(q6k_soa_row_bytes(20) == 4208u, "HIP Q6 lm_head SoA pads d by 8 B");
static_assert(q6_mmvq_one_trip(5120), "official lm_head Q6 is leftover-free");
static_assert(q6_mmvq_one_trip(6144), "official o_proj Q6 is leftover-free");
static_assert(!q6_mmvq_one_trip(65 * 256), "65 Q6 supers need a leftover trip");

__device__ void q4k_decode_scales(const unsigned char* packed, int* scales, int* mins);
__device__ float silu_dev(float g);
__device__ float exp_dev(float x);

__device__ float q5k_lane_dot_q8(const unsigned char* __restrict__ blk,
                                const std::int8_t* __restrict__ xq, const float* __restrict__ xd,
                                int lane) {
    const float2 dm = load_wf16x2(blk);
    const float d = dm.x;
    const float dmin = dm.y;
    int scales[8];
    int mins[8];
    q4k_decode_scales(blk + 4, scales, mins);
    const unsigned char* qh = blk + 16;
    const unsigned char* qs = blk + 48;
    float acc = 0.0f;
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        const int is = 2 * il;
        unsigned char hm = static_cast<unsigned char>(1u << (2 * il));
        const unsigned char byte = qs[32 * il + lane];
        const int qlo = (byte & 0x0f) + ((qh[lane] & hm) ? 16 : 0);
        hm = static_cast<unsigned char>(hm << 1);
        const int qhi = (byte >> 4) + ((qh[lane] & hm) ? 16 : 0);
        const float xlo = xd[is] * static_cast<float>(xq[64 * il + lane]);
        const float xhi = xd[is + 1] * static_cast<float>(xq[64 * il + 32 + lane]);
        acc += (d * static_cast<float>(scales[is]) * static_cast<float>(qlo) -
                dmin * static_cast<float>(mins[is])) *
               xlo;
        acc += (d * static_cast<float>(scales[is + 1]) * static_cast<float>(qhi) -
                dmin * static_cast<float>(mins[is + 1])) *
               xhi;
    }
    return acc;
}

__device__ float q6k_lane_dot_q8(const unsigned char* __restrict__ packed, int rows, int supers,
                                int row, int s, const std::int8_t* __restrict__ xq,
                                const float* __restrict__ xd, int iqs) {
    const int half = iqs >= 16 ? 1 : 0;
    return q6k_dot_q8_oct_half(q6k_soa_ql(packed, rows, supers, row, s, half),
                               q6k_soa_qh(packed, rows, supers, row, s),
                               q6k_soa_scales(packed, rows, supers, row, s),
                               load_f16(q6k_soa_d(packed, rows, supers, row, s)), xq, xd, iqs);
}

__device__ __forceinline__ void maybe_stage_q8x(const std::int8_t* __restrict__& xq,
                                               const float* __restrict__& xd, int cols) {
    // Cap is 0. Official 5120 / 17408 stay in L2 with llama.cpp MMVQ.
    // Do not compute LDS bytes from runtime cols on that path. lm_head
    // is 248320 WGs.
    if constexpr (kLdsQ8xMaxBytes <= 0) {
        (void)xq;
        (void)xd;
        (void)cols;
        return;
    }
    if (q8x_lds_bytes(cols) > kLdsQ8xMaxBytes) {
        return;
    }
    extern __shared__ unsigned char sm[];
    std::int8_t* qs_s = reinterpret_cast<std::int8_t*>(sm);
    float* xd_s = reinterpret_cast<float*>(sm + ((cols + 3) & ~3));
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);
    for (int i = tid; i < cols; i += nthreads) {
        qs_s[i] = xq[i];
    }
    const int nxd = cols / 32;
    for (int i = tid; i < nxd; i += nthreads) {
        xd_s[i] = xd[i];
    }
    __syncthreads();
    xq = qs_s;
    xd = xd_s;
}

__device__ bool bind_multi_row(int r0, int r1, int r2, int r3, float* y0, float* y1, float* y2,
                               float* y3, const unsigned char* p0, const unsigned char* p1,
                               const unsigned char* p2, const unsigned char* p3, float*& y,
                               const unsigned char*& packed, int& local, int& nrows) {
    vesper::MultiRowPick pick;
    if (!vesper::pick_multi_row(gemv_row(), r0, r1, r2, r3, &pick)) {
        return false;
    }
    float* ys[4] = {y0, y1, y2, y3};
    const unsigned char* ps[4] = {p0, p1, p2, p3};
    const int rs[4] = {r0, r1, r2, r3};
    y = ys[pick.slot];
    packed = ps[pick.slot];
    local = pick.local;
    nrows = rs[pick.slot];
    return true;
}

template<bool Tight>
__device__ float mmvq_q8_pair_at(const unsigned char* __restrict__ packed,
                                 const std::int8_t* __restrict__ xqs, const float* __restrict__ xds,
                                 int row, int rows, int k0, int nblocks) {
    float acc = 0.0f;
    if constexpr (Tight) {
        float xd0 = 0.0f;
        float xd1 = 0.0f;
        load_f32x2(xds, k0, &xd0, &xd1);
        // Pair-major 4 B scale tile. Cached: NT is for qs.
        const unsigned char* sp = q8_soa_scale(packed, rows, nblocks, row, k0);
        const float d0 = load_f16(sp);
        const float d1 = load_f16(sp + 2);
        // Pair tile is 64 B. NT both 32 B qs halves before ALU.
        acc = q8_dot_q8_2_t<true>(
            reinterpret_cast<const std::int8_t*>(q8_soa_qs(packed, rows, nblocks, row, k0)), d0,
            xqs + k0 * kQ8BlockElems, xd0,
            reinterpret_cast<const std::int8_t*>(q8_soa_qs(packed, rows, nblocks, row, k0 + 1)), d1,
            xqs + (k0 + 1) * kQ8BlockElems, xd1);
    } else {
#pragma unroll
        for (int t = 0; t < kQ8MmvqBlocksPerThread; ++t) {
            const int kbx = k0 + t;
            if (kbx >= nblocks) {
                continue;
            }
            acc += q8_dot_q8_t<true>(
                reinterpret_cast<const std::int8_t*>(q8_soa_qs(packed, rows, nblocks, row, kbx)),
                load_f16(q8_soa_scale(packed, rows, nblocks, row, kbx)), xqs + kbx * kQ8BlockElems,
                xds[kbx]);
        }
    }
    return acc;
}

template<bool Tight, bool OneTrip>
__device__ float mmvq_q8_partial(const unsigned char* __restrict__ packed,
                                 const std::int8_t* __restrict__ xqs, const float* __restrict__ xds,
                                 int row, int rows, int cols) {
    const int tid = gemv_tid();
    const int nblocks = cols / kQ8BlockElems;
    constexpr int bpt = kQ8MmvqBlocksPerThread;
    constexpr int stride = kQ8MmvqPerIter * bpt;
    if constexpr (Tight) {
        static_assert(bpt == 2, "tight Q8 path emits two consecutive dots");
    }
    const int k0 = bpt * (tid / kQ8MmvqThreadsPerBlock);
    if (k0 >= nblocks) {
        return 0.0f;
    }
    // Official K is one trip. Host instantiates OneTrip so the leftover
    // compare on runtime cols is not in that kernel. Wider rows keep the walk.
    float acc = mmvq_q8_pair_at<Tight>(packed, xqs, xds, row, rows, k0, nblocks);
    if constexpr (!OneTrip) {
#pragma unroll
        for (int k = k0 + stride; k < nblocks; k += stride) {
            acc += mmvq_q8_pair_at<Tight>(packed, xqs, xds, row, rows, k, nblocks);
        }
    }
    return acc;
}

template<bool Tight, bool OneTrip>
__device__ void mmvq_q8_row(float* __restrict__ y, const unsigned char* __restrict__ packed,
                            const std::int8_t* __restrict__ xqs, const float* __restrict__ xds,
                            int row, int rows, int cols, const float* __restrict__ addend) {
    const float acc = reduce_wg(mmvq_q8_partial<Tight, OneTrip>(packed, xqs, xds, row, rows, cols));
    if (gemv_tid() == 0) {
        write_gemv(y, row, acc, addend);
    }
}

template<int Threads>
__device__ float q4k_mmvq_dot(const unsigned char* __restrict__ packed, int rows, int row, int supers,
                              const std::int8_t* __restrict__ xqs, const float* __restrict__ xds,
                              int s, int tid) {
    const unsigned char* __restrict__ hdr = q4k_soa_hdr(packed, rows, supers, row, s);
    if constexpr (Threads == kQ4MmvqDownThreadsPerSuper) {
        return q4k_dot_q8_super_halves(hdr, q4k_soa_qs(packed, rows, supers, row, s, 0),
                                      q4k_soa_qs(packed, rows, supers, row, s, 1), xqs + s * 256,
                                      xds + s * 8);
    }
    int iqs = 0;
    if constexpr (Threads == kQ4MmvqThreadsPerSuper) {
        iqs = 4 * (tid % Threads);
    } else if constexpr (Threads == kQ4MmvqMidThreadsPerSuper) {
        iqs = 8 * (tid % Threads);
    } else {
        static_assert(Threads == kQ4MmvqOctThreadsPerSuper, "unhandled Q4 MMVQ threads");
        iqs = 16 * (tid % Threads);
    }
    // qs is 64 B halves. iqs>=16 is half 1 remapped to iqs-16.
    const int half = iqs >= 16 ? 1 : 0;
    const int iqs_r = iqs - 16 * half;
    const unsigned char* __restrict__ qs = q4k_soa_qs(packed, rows, supers, row, s, half);
    const std::int8_t* __restrict__ xq = xqs + s * 256 + half * 128;
    const float* __restrict__ xd = xds + s * 8 + half * 4;
    if constexpr (Threads == kQ4MmvqThreadsPerSuper) {
        return q4k_dot_q8_pair_parts(hdr, qs, xq, xd, iqs_r);
    } else if constexpr (Threads == kQ4MmvqMidThreadsPerSuper) {
        return q4k_dot_q8_quad_parts(hdr, qs, xq, xd, iqs_r);
    } else {
        return q4k_dot_q8_oct_parts(hdr, qs, xq, xd, iqs_r);
    }
}

template<int Threads, bool OneTrip>
__device__ float q4k_mmvq_walk(const unsigned char* __restrict__ packed, int rows, int row,
                               const std::int8_t* __restrict__ xqs, const float* __restrict__ xds,
                               int supers, int tid) {
    static_assert(kGemvWorkgroup % Threads == 0, "Q4 map must divide the workgroup");
    constexpr int stride = kGemvWorkgroup / Threads;
    const int s = tid / Threads;
    if (s >= supers) {
        return 0.0f;
    }
    // Official SwiGLU (20, mid) and down (68, oct) are one trip. Host
    // instantiates OneTrip so the leftover compare on runtime supers is
    // not in that kernel.
    float acc = q4k_mmvq_dot<Threads>(packed, rows, row, supers, xqs, xds, s, tid);
    if constexpr (!OneTrip) {
#pragma unroll
        for (int t = s + stride; t < supers; t += stride) {
            acc += q4k_mmvq_dot<Threads>(packed, rows, row, supers, xqs, xds, t, tid);
        }
    }
    return acc;
}

template<int Threads, bool OneTrip>
__device__ void mmvq_q4k_row(float* __restrict__ y, const unsigned char* __restrict__ packed,
                             const std::int8_t* __restrict__ xqs, const float* __restrict__ xds,
                             int row, int rows, int cols, const float* __restrict__ addend) {
    const int tid = gemv_tid();
    const int supers = cols / 256;
    const float acc =
        reduce_wg(q4k_mmvq_walk<Threads, OneTrip>(packed, rows, row, xqs, xds, supers, tid));
    if (tid == 0) {
        write_gemv(y, row, acc, addend);
    }
}

__device__ void mmvq_q5k_row(float* __restrict__ y, const unsigned char* __restrict__ packed,
                             const std::int8_t* __restrict__ xqs, const float* __restrict__ xds,
                             int row, int cols, const float* __restrict__ addend) {
    const int tid = gemv_tid();
    const int lane = gemv_lane();
    const int warp = gemv_warp();
    const int supers = cols / 256;
    const unsigned char* rowp =
        packed + static_cast<std::size_t>(row) * static_cast<std::size_t>(supers) * 176u;
    float acc = 0.0f;
#pragma unroll
    for (int s = warp; s < supers; s += kGemvWaves) {
        acc += q5k_lane_dot_q8(rowp + s * 176, xqs + s * 256, xds + s * 8, lane);
    }
    acc = reduce_wg(acc);
    if (tid == 0) {
        write_gemv(y, row, acc, addend);
    }
}

template<bool OneTrip>
__device__ float q6k_mmvq_walk(const unsigned char* __restrict__ packed, int rows, int row,
                               const std::int8_t* __restrict__ xqs, const float* __restrict__ xds,
                               int supers, int tid) {
    const int iqs = (kQ6KQi / kQ6MmvqThreadsPerSuper) * (tid % kQ6MmvqThreadsPerSuper);
    const int s = tid / kQ6MmvqThreadsPerSuper;
    if (s >= supers) {
        return 0.0f;
    }
    float acc = q6k_lane_dot_q8(packed, rows, supers, row, s, xqs + s * 256, xds + s * 8, iqs);
    if constexpr (!OneTrip) {
#pragma unroll
        for (int t = s + kQ6MmvqSuperStride; t < supers; t += kQ6MmvqSuperStride) {
            acc += q6k_lane_dot_q8(packed, rows, supers, row, t, xqs + t * 256, xds + t * 8, iqs);
        }
    }
    return acc;
}

template<bool OneTrip>
__device__ void mmvq_q6k_row(float* __restrict__ y, const unsigned char* __restrict__ packed,
                             const std::int8_t* __restrict__ xqs, const float* __restrict__ xds,
                             int row, int rows, int cols, const float* __restrict__ addend) {
    const int tid = gemv_tid();
    const int supers = cols / 256;
    const float acc = reduce_wg(q6k_mmvq_walk<OneTrip>(packed, rows, row, xqs, xds, supers, tid));
    if (tid == 0) {
        write_gemv(y, row, acc, addend);
    }
}

// llama.cpp RDNA4: 1 output row, up to 8 waves. Official Q4_K_M attn/SSM is Q8_0.
// Tight is official even block counts: two dots, no tail check.
// OneTrip is official K: leftover walk is not compiled into that kernel.
// Launch is the host block size. Official 96/160 match __launch_bounds__
// so the 64 B NT qs hoist is not compiled under the 8-wave 256 cap.
template<bool Tight, bool OneTrip, int Launch>
__global__ void __launch_bounds__(Launch)
gemv_q8_mmvq_kernel(float* __restrict__ y, const unsigned char* __restrict__ packed,
                    const std::int8_t* __restrict__ xq, const float* __restrict__ xd, int rows,
                    int cols, const float* __restrict__ addend) {
    const std::int8_t* __restrict__ xqs = xq;
    const float* __restrict__ xds = xd;
    maybe_stage_q8x(xqs, xds, cols);
    const int row = gemv_row();
    if (row >= rows) {
        return;
    }
    mmvq_q8_row<Tight, OneTrip>(y, packed, xqs, xds, row, rows, cols, addend);
}

int gemv_block(int in_features) {
    if (in_features <= 64) {
        return 64;
    }
    if (in_features <= 256) {
        return 128;
    }
    return kGemvWorkgroup;
}

// One workgroup per output row. float4 loads cover 16 B; 16 threads cover a
// 256 B gfx1201 cacheline. Several independent FP32 accumulators.
__global__ void __launch_bounds__(256)
gemv_f32_kernel(float* __restrict__ y, const float* __restrict__ weight,
                const float* __restrict__ x, int out_features, int in_features,
                const float* __restrict__ addend) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= out_features) {
        return;
    }

    const float* row_w = weight + static_cast<std::size_t>(row) * in_features;
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    // float4 only when every row is 16-byte aligned (in % 4 == 0).
    if ((in_features & 3) == 0) {
        const int n4 = in_features >> 2;
        const float4* w4 = reinterpret_cast<const float4*>(row_w);
        const float4* x4 = reinterpret_cast<const float4*>(x);
        for (int j = tid; j < n4; j += nthreads) {
            const float4 wv = w4[j];
            const float4 xv = x4[j];
            acc0 += wv.x * xv.x;
            acc1 += wv.y * xv.y;
            acc2 += wv.z * xv.z;
            acc3 += wv.w * xv.w;
        }
    } else {
        for (int j = tid; j < in_features; j += nthreads) {
            acc0 += row_w[j] * x[j];
        }
    }
    const float acc = acc0 + acc1 + acc2 + acc3;

    __shared__ float smem[kGemvWorkgroup];
    const float sum = block_reduce_sum(acc, smem);
    if (tid == 0) {
        write_gemv(y, row, sum, addend);
    }
}

// Official hidden is 5120: 256 threads × 4-wide × 5 trips. Each thread
// owns the same elements across residual write and ss, so add/copy and
// the square-sum share one pass. A grid barrier is not required.
template<int N>
__device__ void hidden_float4_ss(const float* __restrict__ x, int tid, float& ss) {
    static_assert(N > 0 && (N % 4) == 0, "official hidden float4");
    constexpr int trips = (N / 4) / kGemvWorkgroup;
    static_assert(trips * kGemvWorkgroup == N / 4, "official hidden float4 trips");
    const float4* x4 = reinterpret_cast<const float4*>(x);
#pragma unroll
    for (int t = 0; t < trips; ++t) {
        const float4 xv = x4[tid + t * kGemvWorkgroup];
        ss += xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
    }
}

template<int N>
__device__ void hidden_float4_store(float* __restrict__ out, const float* __restrict__ src,
                                    const float* __restrict__ weight, float inv, int tid) {
    static_assert(N > 0 && (N % 4) == 0, "official hidden float4");
    constexpr int trips = (N / 4) / kGemvWorkgroup;
    float4* o4 = reinterpret_cast<float4*>(out);
    const float4* s4 = reinterpret_cast<const float4*>(src);
    const float4* w4 = reinterpret_cast<const float4*>(weight);
#pragma unroll
    for (int t = 0; t < trips; ++t) {
        const int j = tid + t * kGemvWorkgroup;
        const float4 sv = s4[j];
        const float4 wv = w4[j];
        float4 y;
        y.x = sv.x * inv * wv.x;
        y.y = sv.y * inv * wv.y;
        y.z = sv.z * inv * wv.z;
        y.w = sv.w * inv * wv.w;
        o4[j] = y;
    }
}

template<int N>
__device__ void hidden_float4_copy_ss(float* __restrict__ residual, const float* __restrict__ x,
                                      int tid, float& ss) {
    static_assert(N > 0 && (N % 4) == 0, "official hidden float4");
    constexpr int trips = (N / 4) / kGemvWorkgroup;
    float4* r4 = reinterpret_cast<float4*>(residual);
    const float4* x4 = reinterpret_cast<const float4*>(x);
#pragma unroll
    for (int t = 0; t < trips; ++t) {
        const int j = tid + t * kGemvWorkgroup;
        const float4 xv = x4[j];
        r4[j] = xv;
        ss += xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
    }
}

template<int N>
__device__ void hidden_float4_add_ss(float* __restrict__ residual, const float* __restrict__ x,
                                     int tid, float& ss) {
    static_assert(N > 0 && (N % 4) == 0, "official hidden float4");
    constexpr int trips = (N / 4) / kGemvWorkgroup;
    float4* r4 = reinterpret_cast<float4*>(residual);
    const float4* x4 = reinterpret_cast<const float4*>(x);
#pragma unroll
    for (int t = 0; t < trips; ++t) {
        const int j = tid + t * kGemvWorkgroup;
        float4 rv = r4[j];
        const float4 xv = x4[j];
        rv.x += xv.x;
        rv.y += xv.y;
        rv.z += xv.z;
        rv.w += xv.w;
        r4[j] = rv;
        ss += rv.x * rv.x + rv.y * rv.y + rv.z * rv.z + rv.w * rv.w;
    }
}

template<int N>
__global__ void __launch_bounds__(256)
rmsnorm_f32_kernel(float* __restrict__ out, const float* __restrict__ x,
                   const float* __restrict__ weight, int n, float eps, std::int8_t* __restrict__ qs,
                   float* __restrict__ d, float* __restrict__ sum) {
    __shared__ float smem[kGemvWorkgroup];
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);
    const int width = N > 0 ? N : n;

    float ss = 0.0f;
    if constexpr (N > 0) {
        hidden_float4_ss<N>(x, tid, ss);
    } else {
        for (int i = tid; i < n; i += nthreads) {
            const float v = x[i];
            ss += v * v;
        }
    }
    ss = block_reduce_sum(ss, smem);
    const float inv = rsqrtf(ss / static_cast<float>(width) + eps);
    if constexpr (N > 0) {
        hidden_float4_store<N>(out, x, weight, inv, tid);
    } else {
        for (int i = tid; i < n; i += nthreads) {
            out[i] = x[i] * inv * weight[i];
        }
    }
    if (qs != nullptr) {
        __syncthreads();
        if constexpr (N > 0) {
            quantize_q8x_from_slice_unroll<N, kGemvWorkgroup>(qs, d, sum, out, 0);
        } else {
            quantize_q8x_from_vec(qs, d, sum, out, width);
        }
    }
}

__global__ void rope_neox_kernel(float* __restrict__ q, float* __restrict__ k, int n_q_heads,
                                 int n_kv_heads, int head_dim, int rotary_dim, int pos_val,
                                 const int* pos_ptr, float theta) {
    const int pos = decode_i32(pos_val, pos_ptr);
    const int half = rotary_dim / 2;
    const int total = (n_q_heads + n_kv_heads) * half;
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total) {
        return;
    }

    const bool on_q = idx < n_q_heads * half;
    const int local = on_q ? idx : idx - n_q_heads * half;
    const int head = local / half;
    const int i = local - head * half;
    float* vec = on_q ? q : k;
    float* pair = vec + head * head_dim;

    const float freq =
        1.0f / powf(theta, static_cast<float>(2 * i) / static_cast<float>(rotary_dim));
    const float angle = static_cast<float>(pos) * freq;
    const float c = cosf(angle);
    const float s = sinf(angle);
    const float x0 = pair[i];
    const float x1 = pair[i + half];
    pair[i] = x0 * c - x1 * s;
    pair[i + half] = x0 * s + x1 * c;
}

__global__ void __launch_bounds__(256)
rope_neox_k_norm_kernel(float* __restrict__ q, float* __restrict__ k,
                        const float* __restrict__ k_weight, int n_q_heads, int n_kv_heads,
                        int head_dim, int rotary_dim, int pos_val, const int* pos_ptr, float theta,
                        float eps) {
    const int pos = decode_i32(pos_val, pos_ptr);
    const int h = static_cast<int>(blockIdx.x);
    const int total_heads = n_q_heads + n_kv_heads;
    if (h >= total_heads) {
        return;
    }
    const bool on_q = h < n_q_heads;
    const int local = on_q ? h : h - n_q_heads;
    float* head = (on_q ? q : k) + local * head_dim;
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);
    if (!on_q) {
        __shared__ float smem[kGemvWorkgroup];
        float ss = 0.0f;
        for (int i = tid; i < head_dim; i += nthreads) {
            const float v = head[i];
            ss += v * v;
        }
        ss = block_reduce_sum(ss, smem);
        const float inv = rsqrtf(ss / static_cast<float>(head_dim) + eps);
        for (int i = tid; i < head_dim; i += nthreads) {
            head[i] = head[i] * inv * k_weight[i];
        }
        __syncthreads();
    }
    const int half = rotary_dim / 2;
    for (int i = tid; i < half; i += nthreads) {
        const float freq =
            1.0f / powf(theta, static_cast<float>(2 * i) / static_cast<float>(rotary_dim));
        const float angle = static_cast<float>(pos) * freq;
        const float c = cosf(angle);
        const float s = sinf(angle);
        const float x0 = head[i];
        const float x1 = head[i + half];
        head[i] = x0 * c - x1 * s;
        head[i + half] = x0 * s + x1 * c;
    }
}

// Official gated attn: split+q_norm, k_norm+rope, scatter. One grid, no
// cross-block reads. attn_decode stays a later launch so it sees *pos+1.
// Official is head_dim 256 / rope 64: one element per thread, one store.
template<int HeadDim, int RotaryDim>
__global__ void __launch_bounds__(256)
attn_prepare_kernel(float* __restrict__ q, float* __restrict__ gate, float* __restrict__ k,
                    float* __restrict__ v, const float* __restrict__ q_full,
                    const float* __restrict__ q_weight, const float* __restrict__ k_weight,
                    float* __restrict__ k_base, float* __restrict__ v_base, const int* pos_ptr,
                    int n_q_heads, int n_kv_heads, int head_dim, int rotary_dim, float theta,
                    float eps) {
    const int pos = *pos_ptr;
    const int h = static_cast<int>(blockIdx.x);
    if (h >= n_q_heads + n_kv_heads) {
        return;
    }
    const int hd = HeadDim > 0 ? HeadDim : head_dim;
    const int rd = RotaryDim > 0 ? RotaryDim : rotary_dim;
    const bool on_q = h < n_q_heads;
    const int local = on_q ? h : h - n_q_heads;
    float* head = (on_q ? q : k) + local * hd;
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);
    if constexpr (HeadDim > 0) {
        (void)nthreads;
    }
    __shared__ float smem[kGemvWorkgroup];
    if (on_q) {
        const float* src = q_full + local * 2 * hd;
        float* gh = gate + local * hd;
        if constexpr (HeadDim > 0) {
            const float val = src[tid];
            gh[tid] = src[HeadDim + tid];
            const float ss = block_reduce_sum(val * val, smem);
            const float inv = rsqrtf(ss / static_cast<float>(HeadDim) + eps);
            head[tid] = val * inv * q_weight[tid];
        } else {
            float ss = 0.0f;
            for (int i = tid; i < head_dim; i += nthreads) {
                const float val = src[i];
                head[i] = val;
                gh[i] = src[head_dim + i];
                ss += val * val;
            }
            ss = block_reduce_sum(ss, smem);
            const float inv = rsqrtf(ss / static_cast<float>(head_dim) + eps);
            for (int i = tid; i < head_dim; i += nthreads) {
                head[i] = head[i] * inv * q_weight[i];
            }
        }
    } else if constexpr (HeadDim > 0) {
        const float val = head[tid];
        const float ss = block_reduce_sum(val * val, smem);
        const float inv = rsqrtf(ss / static_cast<float>(HeadDim) + eps);
        head[tid] = val * inv * k_weight[tid];
    } else {
        float ss = 0.0f;
        for (int i = tid; i < head_dim; i += nthreads) {
            const float val = head[i];
            ss += val * val;
        }
        ss = block_reduce_sum(ss, smem);
        const float inv = rsqrtf(ss / static_cast<float>(head_dim) + eps);
        for (int i = tid; i < head_dim; i += nthreads) {
            head[i] = head[i] * inv * k_weight[i];
        }
    }
    __syncthreads();
    const int half = rd / 2;
    if constexpr (RotaryDim > 0) {
        if (tid < half) {
            const float freq =
                1.0f / powf(theta, static_cast<float>(2 * tid) / static_cast<float>(RotaryDim));
            const float angle = static_cast<float>(pos) * freq;
            const float c = cosf(angle);
            const float s = sinf(angle);
            const float x0 = head[tid];
            const float x1 = head[tid + half];
            head[tid] = x0 * c - x1 * s;
            head[tid + half] = x0 * s + x1 * c;
        }
    } else {
        for (int i = tid; i < half; i += nthreads) {
            const float freq =
                1.0f / powf(theta, static_cast<float>(2 * i) / static_cast<float>(rotary_dim));
            const float angle = static_cast<float>(pos) * freq;
            const float c = cosf(angle);
            const float s = sinf(angle);
            const float x0 = head[i];
            const float x1 = head[i + half];
            head[i] = x0 * c - x1 * s;
            head[i + half] = x0 * s + x1 * c;
        }
    }
    if (on_q) {
        return;
    }
    __syncthreads();
    const int kv_dim = n_kv_heads * hd;
    if constexpr (HeadDim > 0) {
        const std::size_t off = static_cast<std::size_t>(pos) * static_cast<std::size_t>(kv_dim) +
                                static_cast<std::size_t>(local) * static_cast<std::size_t>(HeadDim) +
                                static_cast<std::size_t>(tid);
        k_base[off] = head[tid];
        v_base[off] = v[local * HeadDim + tid];
    } else {
        for (int i = tid; i < head_dim; i += nthreads) {
            const std::size_t off = static_cast<std::size_t>(pos) * static_cast<std::size_t>(kv_dim) +
                                    static_cast<std::size_t>(local) * static_cast<std::size_t>(head_dim) +
                                    static_cast<std::size_t>(i);
            k_base[off] = head[i];
            v_base[off] = v[local * head_dim + i];
        }
    }
}

__global__ void swiglu_kernel(float* __restrict__ out, const float* __restrict__ gate,
                              const float* __restrict__ up, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) {
        return;
    }
    const float g = gate[i];
    out[i] = silu_dev(g) * up[i];
}

__global__ void __launch_bounds__(256)
softmax_kernel(float* __restrict__ x, int n) {
    __shared__ float smem[kGemvWorkgroup];
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);

    float max_v = -INFINITY;
    for (int i = tid; i < n; i += nthreads) {
        max_v = fmaxf(max_v, x[i]);
    }
    max_v = block_reduce_max(max_v, smem);

    float sum = 0.0f;
    for (int i = tid; i < n; i += nthreads) {
        const float e = exp_dev(x[i] - max_v);
        x[i] = e;
        sum += e;
    }
    sum = block_reduce_sum(sum, smem);
    const float inv = 1.0f / sum;
    for (int i = tid; i < n; i += nthreads) {
        x[i] *= inv;
    }
}

__global__ void attn_scores_kernel(float* __restrict__ scores, const float* __restrict__ q,
                                   const float* __restrict__ k, int seq, int n_kv_heads,
                                   int kv_head, int head_dim) {
    const int t = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (t >= seq) {
        return;
    }
    const float scale = rsqrtf(static_cast<float>(head_dim));
    const float* key =
        k + (static_cast<std::size_t>(t) * n_kv_heads + kv_head) * head_dim;
    float dot = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
        dot += q[i] * key[i];
    }
    scores[t] = dot * scale;
}

__global__ void attn_mix_kernel(float* __restrict__ out, const float* __restrict__ scores,
                                const float* __restrict__ v, int seq, int n_kv_heads,
                                int kv_head, int head_dim) {
    const int i = static_cast<int>(threadIdx.x);
    if (i >= head_dim) {
        return;
    }
    float acc = 0.0f;
    for (int t = 0; t < seq; ++t) {
        const float* val =
            v + (static_cast<std::size_t>(t) * n_kv_heads + kv_head) * head_dim;
        acc += scores[t] * val[i];
    }
    out[i] = acc;
}

__global__ void add_inplace_kernel(float* __restrict__ dst, const float* __restrict__ src,
                                   int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        dst[i] += src[i];
    }
}

int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

__device__ void q4k_decode_scales(const unsigned char* packed, int* scales, int* mins) {
    const int w0 = load_i32(packed, 0);
    const int w1 = load_i32(packed, 1);
    const int w2 = load_i32(packed, 2);
    for (int j = 0; j < 4; ++j) {
        const int lo = (w0 >> (8 * j)) & 0xff;
        const int mid = (w1 >> (8 * j)) & 0xff;
        const int hi = (w2 >> (8 * j)) & 0xff;
        scales[j] = lo & 0x3f;
        mins[j] = mid & 0x3f;
        scales[j + 4] = (hi & 0x0f) | ((lo >> 6) << 4);
        mins[j + 4] = (hi >> 4) | ((mid >> 6) << 4);
    }
}

// 1 row / WG. Host instantiates one map: pair (8), mid/quad (4),
// oct (2), or down/super (1). Official SwiGLU is mid, official FFN
// down is oct. Launch follows working waves. OneTrip drops the leftover
// K walk. Launch is also __launch_bounds__: official 96/160 get that
// VGPR budget so the 64 B NT hoist is not spilled under 256.
template<int Threads, bool OneTrip, int Launch>
__global__ void __launch_bounds__(Launch)
gemv_q4k_mmvq_kernel(float* __restrict__ y, const unsigned char* __restrict__ packed,
                     const std::int8_t* __restrict__ xq, const float* __restrict__ xd, int rows,
                     int cols, const float* __restrict__ addend) {
    const std::int8_t* __restrict__ xqs = xq;
    const float* __restrict__ xds = xd;
    maybe_stage_q8x(xqs, xds, cols);
    const int row = gemv_row();
    if (row >= rows) {
        return;
    }
    mmvq_q4k_row<Threads, OneTrip>(y, packed, xqs, xds, row, rows, cols, addend);
}

__global__ void __launch_bounds__(256)
gemv_q5k_mmvq_kernel(float* __restrict__ y, const unsigned char* __restrict__ packed,
                     const std::int8_t* __restrict__ xq, const float* __restrict__ xd, int rows,
                     int cols, const float* __restrict__ addend) {
    const std::int8_t* __restrict__ xqs = xq;
    const float* __restrict__ xds = xd;
    maybe_stage_q8x(xqs, xds, cols);
    const int row = gemv_row();
    if (row >= rows) {
        return;
    }
    mmvq_q5k_row(y, packed, xqs, xds, row, cols, addend);
}

template<bool OneTrip, int Launch>
__global__ void __launch_bounds__(Launch)
gemv_q6k_mmvq_kernel(float* __restrict__ y, const unsigned char* __restrict__ packed,
                     const std::int8_t* __restrict__ xq, const float* __restrict__ xd, int rows,
                     int cols, const float* __restrict__ addend) {
    const std::int8_t* __restrict__ xqs = xq;
    const float* __restrict__ xds = xd;
    maybe_stage_q8x(xqs, xds, cols);
    const int row = gemv_row();
    if (row >= rows) {
        return;
    }
    mmvq_q6k_row<OneTrip>(y, packed, xqs, xds, row, rows, cols, addend);
}

// One grid over Q/K/V or GDN qkv/z/beta/alpha. Stage Q8_1 x once per WG, then
// pick the output matrix from blockIdx. Host already quantized x once.
template<bool Tight, bool OneTrip, int Launch>
__global__ void __launch_bounds__(Launch)
gemv_q8_mmvq_multi_kernel(float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
                          float* __restrict__ y3, const unsigned char* __restrict__ p0,
                          const unsigned char* __restrict__ p1, const unsigned char* __restrict__ p2,
                          const unsigned char* __restrict__ p3, const std::int8_t* __restrict__ xq,
                          const float* __restrict__ xd, int r0, int r1, int r2, int r3, int cols) {
    const std::int8_t* __restrict__ xqs = xq;
    const float* __restrict__ xds = xd;
    maybe_stage_q8x(xqs, xds, cols);
    float* y = nullptr;
    const unsigned char* packed = nullptr;
    int local = 0;
    int nrows = 0;
    if (!bind_multi_row(r0, r1, r2, r3, y0, y1, y2, y3, p0, p1, p2, p3, y, packed, local, nrows)) {
        return;
    }
    mmvq_q8_row<Tight, OneTrip>(y, packed, xqs, xds, local, nrows, cols, nullptr);
}

template<int Threads, bool OneTrip, int Launch>
__global__ void __launch_bounds__(Launch)
gemv_q4k_mmvq_multi_kernel(float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
                           float* __restrict__ y3, const unsigned char* __restrict__ p0,
                           const unsigned char* __restrict__ p1, const unsigned char* __restrict__ p2,
                           const unsigned char* __restrict__ p3, const std::int8_t* __restrict__ xq,
                           const float* __restrict__ xd, int r0, int r1, int r2, int r3, int cols) {
    const std::int8_t* __restrict__ xqs = xq;
    const float* __restrict__ xds = xd;
    maybe_stage_q8x(xqs, xds, cols);
    float* y = nullptr;
    const unsigned char* packed = nullptr;
    int local = 0;
    int nrows = 0;
    if (!bind_multi_row(r0, r1, r2, r3, y0, y1, y2, y3, p0, p1, p2, p3, y, packed, local, nrows)) {
        return;
    }
    mmvq_q4k_row<Threads, OneTrip>(y, packed, xqs, xds, local, nrows, cols, nullptr);
}

__global__ void __launch_bounds__(256)
gemv_q5k_mmvq_multi_kernel(float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
                           float* __restrict__ y3, const unsigned char* __restrict__ p0,
                           const unsigned char* __restrict__ p1, const unsigned char* __restrict__ p2,
                           const unsigned char* __restrict__ p3, const std::int8_t* __restrict__ xq,
                           const float* __restrict__ xd, int r0, int r1, int r2, int r3, int cols) {
    const std::int8_t* __restrict__ xqs = xq;
    const float* __restrict__ xds = xd;
    maybe_stage_q8x(xqs, xds, cols);
    float* y = nullptr;
    const unsigned char* packed = nullptr;
    int local = 0;
    int nrows = 0;
    if (!bind_multi_row(r0, r1, r2, r3, y0, y1, y2, y3, p0, p1, p2, p3, y, packed, local, nrows)) {
        return;
    }
    (void)nrows;
    mmvq_q5k_row(y, packed, xqs, xds, local, cols, nullptr);
}

template<bool OneTrip, int Launch>
__global__ void __launch_bounds__(Launch)
gemv_q6k_mmvq_multi_kernel(float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
                           float* __restrict__ y3, const unsigned char* __restrict__ p0,
                           const unsigned char* __restrict__ p1, const unsigned char* __restrict__ p2,
                           const unsigned char* __restrict__ p3, const std::int8_t* __restrict__ xq,
                           const float* __restrict__ xd, int r0, int r1, int r2, int r3, int cols) {
    const std::int8_t* __restrict__ xqs = xq;
    const float* __restrict__ xds = xd;
    maybe_stage_q8x(xqs, xds, cols);
    float* y = nullptr;
    const unsigned char* packed = nullptr;
    int local = 0;
    int nrows = 0;
    if (!bind_multi_row(r0, r1, r2, r3, y0, y1, y2, y3, p0, p1, p2, p3, y, packed, local, nrows)) {
        return;
    }
    mmvq_q6k_row<OneTrip>(y, packed, xqs, xds, local, nrows, cols, nullptr);
}

__global__ void sigmoid_kernel(float* __restrict__ x, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) {
        return;
    }
    const float v = x[i];
    if (v >= 0.0f) {
        const float z = exp_dev(-v);
        x[i] = 1.0f / (1.0f + z);
    } else {
        const float z = exp_dev(v);
        x[i] = z / (1.0f + z);
    }
}

__global__ void silu_kernel(float* __restrict__ x, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        const float v = x[i];
        x[i] = v / (1.0f + exp_dev(-v));
    }
}

__global__ void softplus_kernel(float* __restrict__ x, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) {
        return;
    }
    const float v = x[i];
    if (v > 20.0f) {
        return;
    }
    if (v < -20.0f) {
        x[i] = exp_dev(v);
        return;
    }
    x[i] = log1pf(exp_dev(v));
}

__global__ void exp_kernel(float* __restrict__ x, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        x[i] = exp_dev(x[i]);
    }
}

__global__ void mul_inplace_kernel(float* __restrict__ dst, const float* __restrict__ src, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        dst[i] *= src[i];
    }
}

__global__ void scale_inplace_kernel(float* __restrict__ x, float scale, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        x[i] *= scale;
    }
}

__global__ void l2_normalize_rows_kernel(float* __restrict__ x, int rows, int dim, float eps) {
    const int r = static_cast<int>(blockIdx.x);
    if (r >= rows) {
        return;
    }
    float* row = x + r * dim;
    float ss = 0.0f;
    for (int i = static_cast<int>(threadIdx.x); i < dim; i += static_cast<int>(blockDim.x)) {
        ss += row[i] * row[i];
    }
    ss = wave32_sum(ss);
    __shared__ float inv;
    if (threadIdx.x == 0) {
        inv = rsqrtf(ss + eps);
    }
    __syncthreads();
    for (int i = static_cast<int>(threadIdx.x); i < dim; i += static_cast<int>(blockDim.x)) {
        row[i] *= inv;
    }
}

template<int Kernel>
__device__ float gdn_conv_step(float* __restrict__ state, const float* __restrict__ x,
                               const float* __restrict__ weight, int d) {
    static_assert(Kernel >= 2, "GDN conv kernel must be >= 2");
    constexpr int hist = Kernel - 1;
    const float* w = weight + d * Kernel;
    float* st = state + d * hist;
    float acc = 0.0f;
    if constexpr (Kernel == kGdnConvKernel) {
        float w0 = 0.0f;
        float w1 = 0.0f;
        float w2 = 0.0f;
        float w3 = 0.0f;
        load_f32x2(w, 0, &w0, &w1);
        load_f32x2(w, 2, &w2, &w3);
        const float x0 = x[d];
        acc = w0 * st[0] + w1 * st[1] + w2 * st[2] + w3 * x0;
        st[0] = st[1];
        st[1] = st[2];
        st[2] = x0;
    } else {
#pragma unroll
        for (int t = 0; t < hist; ++t) {
            acc += w[t] * st[t];
        }
        acc += w[hist] * x[d];
#pragma unroll
        for (int t = 0; t < hist - 1; ++t) {
            st[t] = st[t + 1];
        }
        st[hist - 1] = x[d];
    }
    return silu_dev(acc);
}

__device__ float gdn_conv_step_runtime(float* __restrict__ state, const float* __restrict__ x,
                                       const float* __restrict__ weight, int d, int kernel) {
    const int hist = kernel - 1;
    float acc = 0.0f;
    for (int t = 0; t < hist; ++t) {
        acc += weight[d * kernel + t] * state[d * hist + t];
    }
    acc += weight[d * kernel + hist] * x[d];
    for (int t = 0; t < hist - 1; ++t) {
        state[d * hist + t] = state[d * hist + t + 1];
    }
    state[d * hist + (hist - 1)] = x[d];
    return silu_dev(acc);
}

template <int Kernel>
__device__ float gdn_conv_at(float* __restrict__ state, const float* __restrict__ x,
                             const float* __restrict__ weight, int d, int kernel) {
    if constexpr (Kernel >= 2) {
        (void)kernel;
        return gdn_conv_step<Kernel>(state, x, weight, d);
    }
    return gdn_conv_step_runtime(state, x, weight, d, kernel);
}

template<int Kernel>
__global__ void gdn_conv_update_kernel(float* __restrict__ y, float* __restrict__ state,
                                      const float* __restrict__ x, const float* __restrict__ weight,
                                      int conv_dim, int kernel) {
    const int d = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (d >= conv_dim) {
        return;
    }
    if constexpr (Kernel >= 2) {
        y[d] = gdn_conv_step<Kernel>(state, x, weight, d);
        (void)kernel;
    } else {
        y[d] = gdn_conv_step_runtime(state, x, weight, d, kernel);
    }
}

// llama.cpp gated_delta_net.cu: one warp owns one S column in registers.
// rec is [n_v][dv][dk] so column j is contiguous. dim <= kGdnDeltaMaxDim.
// Rows is 4 on official GDN (128), 8 only when dim > 128. Tight is
// dim == Rows * wave: official 128/256 drop i < dim.
template <int Rows, bool Tight>
__global__ void __launch_bounds__(kWavefront * kGdnDeltaWarps)
gdn_delta_rule_kernel(float* __restrict__ y, float* __restrict__ rec,
                      const float* __restrict__ q, const float* __restrict__ k,
                      const float* __restrict__ v, const float* __restrict__ decay,
                      const float* __restrict__ beta, int n_heads, int dim) {
    static_assert(Rows >= 1 && Rows <= kGdnDeltaRowsPerLane, "GDN shard rows");
    constexpr int width = Rows * kWavefront;
    const int n = Tight ? width : dim;
    const int h = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    const int warp = static_cast<int>(threadIdx.y);
    const int col = static_cast<int>(blockIdx.y) * kGdnDeltaWarps + warp;
    if (h >= n_heads || col >= n) {
        return;
    }
    if constexpr (Tight) {
        (void)dim;
    }
    float* scol = rec + (static_cast<std::size_t>(h) * n + col) * n;
    const float* qh = q + h * n;
    const float* kh = k + h * n;
    float s_shard[Rows];
    float k_reg[Rows];
    float q_reg[Rows];
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
        const int i = r * kWavefront + lane;
        if constexpr (Tight) {
            s_shard[r] = scol[i];
            k_reg[r] = kh[i];
            q_reg[r] = qh[i];
        } else {
            const bool on = i < dim;
            s_shard[r] = on ? scol[i] : 0.0f;
            k_reg[r] = on ? kh[i] : 0.0f;
            q_reg[r] = on ? qh[i] : 0.0f;
        }
    }
    float kv = 0.0f;
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
        kv += s_shard[r] * k_reg[r];
    }
    kv = wave32_sum(kv);
    const float g = decay[h];
    const float delta = beta[h] * (v[h * n + col] - g * kv);
    float attn = 0.0f;
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
        s_shard[r] = g * s_shard[r] + k_reg[r] * delta;
        attn += s_shard[r] * q_reg[r];
    }
    attn = wave32_sum(attn);
    if (lane == 0) {
        y[h * n + col] = attn;
    }
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
        const int i = r * kWavefront + lane;
        if constexpr (Tight) {
            scol[i] = s_shard[r];
        } else if (i < dim) {
            scol[i] = s_shard[r];
        }
    }
}

__global__ void __launch_bounds__(256)
rmsnorm_rows_kernel(float* __restrict__ x, const float* __restrict__ weight, int rows, int dim,
                    float eps) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) {
        return;
    }
    float* y = x + row * dim;
    __shared__ float smem[kGemvWorkgroup];
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);
    float ss = 0.0f;
    for (int i = tid; i < dim; i += nthreads) {
        const float v = y[i];
        ss += v * v;
    }
    ss = block_reduce_sum(ss, smem);
    const float inv = rsqrtf(ss / static_cast<float>(dim) + eps);
    for (int i = tid; i < dim; i += nthreads) {
        y[i] = y[i] * inv * weight[i];
    }
}

__global__ void split_gated_q_kernel(float* __restrict__ q, float* __restrict__ gate,
                                     const float* __restrict__ q_full, int n_heads, int head_dim) {
    const int h = static_cast<int>(blockIdx.x);
    if (h >= n_heads) {
        return;
    }
    const float* src = q_full + h * 2 * head_dim;
    for (int i = static_cast<int>(threadIdx.x); i < head_dim; i += static_cast<int>(blockDim.x)) {
        q[h * head_dim + i] = src[i];
        gate[h * head_dim + i] = src[head_dim + i];
    }
}

__global__ void __launch_bounds__(256)
split_gated_q_norm_kernel(float* __restrict__ q, float* __restrict__ gate,
                          const float* __restrict__ q_full, const float* __restrict__ weight,
                          int n_heads, int head_dim, float eps) {
    const int h = static_cast<int>(blockIdx.x);
    if (h >= n_heads) {
        return;
    }
    const float* src = q_full + h * 2 * head_dim;
    float* qh = q + h * head_dim;
    float* gh = gate + h * head_dim;
    __shared__ float smem[kGemvWorkgroup];
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);
    float ss = 0.0f;
    for (int i = tid; i < head_dim; i += nthreads) {
        const float v = src[i];
        qh[i] = v;
        gh[i] = src[head_dim + i];
        ss += v * v;
    }
    ss = block_reduce_sum(ss, smem);
    const float inv = rsqrtf(ss / static_cast<float>(head_dim) + eps);
    for (int i = tid; i < head_dim; i += nthreads) {
        qh[i] = qh[i] * inv * weight[i];
    }
}

__device__ float silu_dev(float g);

template<int Dim>
__global__ void __launch_bounds__(256)
rmsnorm_silu_mul_kernel(float* __restrict__ y, const float* __restrict__ z,
                        const float* __restrict__ weight, int rows, int dim, float eps,
                        std::int8_t* __restrict__ qs, float* __restrict__ d, float* __restrict__ sum) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) {
        return;
    }
    const int width = Dim > 0 ? Dim : dim;
    float* yr = y + row * width;
    const float* zr = z + row * width;
    __shared__ float smem[kGemvWorkgroup];
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);
    float ss = 0.0f;
    if constexpr (Dim > 0) {
        const float v = yr[tid];
        ss = block_reduce_sum(v * v, smem);
        const float inv = rsqrtf(ss / static_cast<float>(Dim) + eps);
        yr[tid] = v * inv * weight[tid] * silu_dev(zr[tid]);
    } else {
        for (int i = tid; i < dim; i += nthreads) {
            const float v = yr[i];
            ss += v * v;
        }
        ss = block_reduce_sum(ss, smem);
        const float inv = rsqrtf(ss / static_cast<float>(dim) + eps);
        for (int i = tid; i < dim; i += nthreads) {
            yr[i] = yr[i] * inv * weight[i] * silu_dev(zr[i]);
        }
    }
    if (qs != nullptr) {
        __syncthreads();
        if constexpr (Dim > 0) {
            quantize_q8x_from_slice_unroll<Dim, row_workgroup(Dim)>(qs, d, sum, y, row * Dim);
        } else {
            quantize_q8x_from_slice(qs, d, sum, y, row * width, width);
        }
    }
}

template<int Kernel>
__global__ void gdn_conv_split_kernel(float* __restrict__ q, float* __restrict__ k,
                                      float* __restrict__ v, float* __restrict__ state,
                                      const float* __restrict__ x, const float* __restrict__ weight,
                                      int key_dim, int value_dim, int kernel) {
    const int qkv_dim = 2 * key_dim + value_dim;
    const int d = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (d >= qkv_dim) {
        return;
    }
    float y = 0.0f;
    if constexpr (Kernel >= 2) {
        y = gdn_conv_step<Kernel>(state, x, weight, d);
        (void)kernel;
    } else {
        y = gdn_conv_step_runtime(state, x, weight, d, kernel);
    }
    if (d < key_dim) {
        q[d] = y;
    } else if (d < 2 * key_dim) {
        k[d - key_dim] = y;
    } else {
        v[d - 2 * key_dim] = y;
    }
}

__global__ void tile_heads_kernel(float* __restrict__ dst, const float* __restrict__ src, int n_dst,
                                  int n_src, int dim) {
    const int d = static_cast<int>(blockIdx.x);
    if (d >= n_dst) {
        return;
    }
    const int s = d % n_src;
    for (int i = static_cast<int>(threadIdx.x); i < dim; i += static_cast<int>(blockDim.x)) {
        dst[d * dim + i] = src[s * dim + i];
    }
}

// One wave per Q head. Online softmax over seq with shuffle dots, no
// LDS barrier per token. Official 27B attn is head_dim 256 (Rows=8).
// Tight is head_dim == Rows * wave: the seq walk drops i < head_dim.
template <int Rows, bool Tight>
__global__ void __launch_bounds__(kWavefront)
attn_decode_kernel(float* __restrict__ out, const float* __restrict__ q, const float* __restrict__ k,
                   const float* __restrict__ v, const float* __restrict__ gate, int seq_val,
                   const int* pos_ptr, int n_q_heads, int n_kv_heads, int head_dim,
                   std::int8_t* __restrict__ qs, float* __restrict__ d, float* __restrict__ sum) {
    static_assert(Rows >= 1 && Rows <= kGdnDeltaRowsPerLane, "attn shard rows");
    constexpr int hd = Rows * kWavefront;
    const int width = Tight ? hd : head_dim;
    const int seq = pos_ptr != nullptr ? (*pos_ptr + 1) : seq_val;
    const int qh = static_cast<int>(blockIdx.x);
    if (qh >= n_q_heads) {
        return;
    }
    if constexpr (Tight) {
        (void)head_dim;
    }
    const int group = n_q_heads / n_kv_heads;
    const int kvh = qh / group;
    const int lane = static_cast<int>(threadIdx.x);
    const float* qh_ptr = q + qh * width;
    float q_reg[Rows];
    float acc[Rows];
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
        const int i = r * kWavefront + lane;
        if constexpr (Tight) {
            q_reg[r] = qh_ptr[i];
        } else {
            q_reg[r] = (i < head_dim) ? qh_ptr[i] : 0.0f;
        }
        acc[r] = 0.0f;
    }
    const float scale = rsqrtf(static_cast<float>(width));
    float m = -INFINITY;
    float l = 0.0f;
    for (int t = 0; t < seq; ++t) {
        const float* kt = k + (static_cast<std::size_t>(t) * n_kv_heads + kvh) * width;
        const float* vt = v + (static_cast<std::size_t>(t) * n_kv_heads + kvh) * width;
        float partial = 0.0f;
        float v_reg[Rows];
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            const int i = r * kWavefront + lane;
            if constexpr (Tight) {
                v_reg[r] = vt[i];
                partial += q_reg[r] * kt[i];
            } else {
                const bool on = i < head_dim;
                const float kv = on ? kt[i] : 0.0f;
                v_reg[r] = on ? vt[i] : 0.0f;
                partial += q_reg[r] * kv;
            }
        }
        const float score = wave32_sum(partial) * scale;
        const float m_new = fmaxf(m, score);
        const float alpha = exp_dev(m - m_new);
        const float exp_s = exp_dev(score - m_new);
        l = l * alpha + exp_s;
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            acc[r] = acc[r] * alpha + exp_s * v_reg[r];
        }
        m = m_new;
    }
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
        const int i = r * kWavefront + lane;
        float y = 0.0f;
        if constexpr (Tight) {
            y = acc[r] / l;
            if (gate != nullptr) {
                const float g = gate[qh * hd + i];
                const float s = (g >= 0.0f) ? (1.0f / (1.0f + exp_dev(-g))) : (exp_dev(g) / (1.0f + exp_dev(g)));
                y *= s;
            }
            out[qh * hd + i] = y;
            if (qs != nullptr) {
                quantize_q8x_value(qs, d, sum, qh * Rows + r, lane, y);
            }
        } else if (i < head_dim) {
            y = acc[r] / l;
            if (gate != nullptr) {
                const float g = gate[qh * head_dim + i];
                const float s = (g >= 0.0f) ? (1.0f / (1.0f + exp_dev(-g))) : (exp_dev(g) / (1.0f + exp_dev(g)));
                y *= s;
            }
            out[qh * head_dim + i] = y;
            if (qs != nullptr) {
                quantize_q8x_value(qs, d, sum, qh * (head_dim / kQ8XBlockElems) + r, lane, y);
            }
        }
    }
}

__device__ float exp_dev(float x) {
    return __expf(x);
}

__device__ float silu_dev(float g) {
    // llama.cpp AMD SiLU uses the hardware exp. CPU stays expf.
    return g / (1.0f + exp_dev(-g));
}

template<int N>
__global__ void __launch_bounds__(256)
add_rmsnorm_kernel(float* __restrict__ x, float* __restrict__ residual, const float* __restrict__ weight,
                   int n, float eps, std::int8_t* __restrict__ qs, float* __restrict__ d,
                   float* __restrict__ sum) {
    __shared__ float smem[kGemvWorkgroup];
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);
    const int width = N > 0 ? N : n;
    float ss = 0.0f;
    if constexpr (N > 0) {
        hidden_float4_add_ss<N>(residual, x, tid, ss);
    } else {
        for (int i = tid; i < n; i += nthreads) {
            residual[i] += x[i];
            const float v = residual[i];
            ss += v * v;
        }
    }
    ss = block_reduce_sum(ss, smem);
    const float inv = rsqrtf(ss / static_cast<float>(width) + eps);
    if constexpr (N > 0) {
        hidden_float4_store<N>(x, residual, weight, inv, tid);
    } else {
        for (int i = tid; i < n; i += nthreads) {
            x[i] = residual[i] * inv * weight[i];
        }
    }
    if (qs != nullptr) {
        __syncthreads();
        if constexpr (N > 0) {
            quantize_q8x_from_slice_unroll<N, kGemvWorkgroup>(qs, d, sum, x, 0);
        } else {
            quantize_q8x_from_vec(qs, d, sum, x, width);
        }
    }
}

template<int N>
__global__ void __launch_bounds__(256)
copy_rmsnorm_kernel(float* __restrict__ x, float* __restrict__ residual,
                    const float* __restrict__ weight, int n, float eps, std::int8_t* __restrict__ qs,
                    float* __restrict__ d, float* __restrict__ sum) {
    __shared__ float smem[kGemvWorkgroup];
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);
    const int width = N > 0 ? N : n;
    float ss = 0.0f;
    if constexpr (N > 0) {
        hidden_float4_copy_ss<N>(residual, x, tid, ss);
    } else {
        for (int i = tid; i < n; i += nthreads) {
            residual[i] = x[i];
            const float v = residual[i];
            ss += v * v;
        }
    }
    ss = block_reduce_sum(ss, smem);
    const float inv = rsqrtf(ss / static_cast<float>(width) + eps);
    if constexpr (N > 0) {
        hidden_float4_store<N>(x, residual, weight, inv, tid);
    } else {
        for (int i = tid; i < n; i += nthreads) {
            x[i] = residual[i] * inv * weight[i];
        }
    }
    if (qs != nullptr) {
        __syncthreads();
        if constexpr (N > 0) {
            quantize_q8x_from_slice_unroll<N, kGemvWorkgroup>(qs, d, sum, x, 0);
        } else {
            quantize_q8x_from_vec(qs, d, sum, x, width);
        }
    }
}

__global__ void silu_mul_kernel(float* __restrict__ y, const float* __restrict__ z, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) {
        return;
    }
    y[i] *= silu_dev(z[i]);
}

__device__ void gdn_apply_gate(float* __restrict__ decay, float* __restrict__ beta,
                               const float* __restrict__ alpha, const float* __restrict__ dt,
                               const float* __restrict__ a, int i) {
    float t = alpha[i] + dt[i];
    if (t > 20.0f) {
        // softplus(t) ~= t
    } else if (t < -20.0f) {
        t = exp_dev(t);
    } else {
        t = log1pf(exp_dev(t));
    }
    decay[i] = exp_dev(a[i] * t);
    const float b = beta[i];
    if (b >= 0.0f) {
        beta[i] = 1.0f / (1.0f + exp_dev(-b));
    } else {
        const float z = exp_dev(b);
        beta[i] = z / (1.0f + z);
    }
}

__global__ void gdn_gates_kernel(float* __restrict__ decay, float* __restrict__ beta,
                                 const float* __restrict__ alpha, const float* __restrict__ dt,
                                 const float* __restrict__ a, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) {
        return;
    }
    gdn_apply_gate(decay, beta, alpha, dt, a, i);
}

__global__ void split_qkv_kernel(float* __restrict__ q, float* __restrict__ k, float* __restrict__ v,
                                 const float* __restrict__ qkv, int key_dim, int value_dim) {
    const int total = 2 * key_dim + value_dim;
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= total) {
        return;
    }
    if (i < key_dim) {
        q[i] = qkv[i];
        return;
    }
    if (i < 2 * key_dim) {
        k[i - key_dim] = qkv[i];
        return;
    }
    v[i - 2 * key_dim] = qkv[i];
}

__global__ void __launch_bounds__(256)
gemv_swiglu_f32_kernel(float* __restrict__ hidden, const float* __restrict__ gate,
                       const float* __restrict__ up, const float* __restrict__ x, int rows,
                       int cols) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) {
        return;
    }
    const float* grow = gate + static_cast<std::size_t>(row) * cols;
    const float* urow = up + static_cast<std::size_t>(row) * cols;
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);
    float ag = 0.0f;
    float au = 0.0f;
    if ((cols & 3) == 0) {
        const int n4 = cols >> 2;
        const float4* g4 = reinterpret_cast<const float4*>(grow);
        const float4* u4 = reinterpret_cast<const float4*>(urow);
        const float4* x4 = reinterpret_cast<const float4*>(x);
        for (int j = tid; j < n4; j += nthreads) {
            const float4 gv = g4[j];
            const float4 uv = u4[j];
            const float4 xv = x4[j];
            ag += gv.x * xv.x + gv.y * xv.y + gv.z * xv.z + gv.w * xv.w;
            au += uv.x * xv.x + uv.y * xv.y + uv.z * xv.z + uv.w * xv.w;
        }
    } else {
        for (int j = tid; j < cols; j += nthreads) {
            ag += grow[j] * x[j];
            au += urow[j] * x[j];
        }
    }
    __shared__ float smem[kGemvWorkgroup];
    ag = block_reduce_sum(ag, smem);
    au = block_reduce_sum(au, smem);
    if (tid == 0) {
        hidden[row] = silu_dev(ag) * au;
    }
}

template<bool Tight, bool OneTrip, int Launch>
__global__ void __launch_bounds__(Launch)
gemv_swiglu_q8_mmvq_kernel(float* __restrict__ hidden, const unsigned char* __restrict__ gate,
                           const unsigned char* __restrict__ up, const std::int8_t* __restrict__ xq,
                           const float* __restrict__ xd, int rows, int cols) {
    const std::int8_t* __restrict__ xqs = xq;
    const float* __restrict__ xds = xd;
    maybe_stage_q8x(xqs, xds, cols);
    const int row = gemv_row();
    if (row >= rows) {
        return;
    }
    float ag = mmvq_q8_partial<Tight, OneTrip>(gate, xqs, xds, row, rows, cols);
    float au = mmvq_q8_partial<Tight, OneTrip>(up, xqs, xds, row, rows, cols);
    reduce_wg2(ag, au, &ag, &au);
    if (gemv_tid() == 0) {
        hidden[row] = silu_dev(ag) * au;
    }
}

template<int Threads, bool OneTrip, int Launch>
__global__ void __launch_bounds__(Launch)
gemv_swiglu_q4k_mmvq_kernel(float* __restrict__ hidden, const unsigned char* __restrict__ gate,
                            const unsigned char* __restrict__ up, const std::int8_t* __restrict__ xq,
                            const float* __restrict__ xd, int rows, int cols) {
    const std::int8_t* __restrict__ xqs = xq;
    const float* __restrict__ xds = xd;
    maybe_stage_q8x(xqs, xds, cols);
    const int row = gemv_row();
    const int tid = gemv_tid();
    if (row >= rows) {
        return;
    }
    const int supers = cols / 256;
    float ag = q4k_mmvq_walk<Threads, OneTrip>(gate, rows, row, xqs, xds, supers, tid);
    float au = q4k_mmvq_walk<Threads, OneTrip>(up, rows, row, xqs, xds, supers, tid);
    reduce_wg2(ag, au, &ag, &au);
    if (tid == 0) {
        hidden[row] = silu_dev(ag) * au;
    }
}

__global__ void __launch_bounds__(256)
gemv_swiglu_q5k_mmvq_kernel(float* __restrict__ hidden, const unsigned char* __restrict__ gate,
                            const unsigned char* __restrict__ up, const std::int8_t* __restrict__ xq,
                            const float* __restrict__ xd, int rows, int cols) {
    const std::int8_t* __restrict__ xqs = xq;
    const float* __restrict__ xds = xd;
    maybe_stage_q8x(xqs, xds, cols);
    const int row = gemv_row();
    const int tid = gemv_tid();
    const int lane = gemv_lane();
    const int warp = gemv_warp();
    if (row >= rows) {
        return;
    }
    const int supers = cols / 256;
    const unsigned char* __restrict__ gp =
        gate + static_cast<std::size_t>(row) * static_cast<std::size_t>(supers) * 176u;
    const unsigned char* __restrict__ upw =
        up + static_cast<std::size_t>(row) * static_cast<std::size_t>(supers) * 176u;
    float ag = 0.0f;
    float au = 0.0f;
#pragma unroll
    for (int s = warp; s < supers; s += kGemvWaves) {
        ag += q5k_lane_dot_q8(gp + s * 176, xqs + s * 256, xds + s * 8, lane);
        au += q5k_lane_dot_q8(upw + s * 176, xqs + s * 256, xds + s * 8, lane);
    }
    reduce_wg2(ag, au, &ag, &au);
    if (tid == 0) {
        hidden[row] = silu_dev(ag) * au;
    }
}

template<bool OneTrip, int Launch>
__global__ void __launch_bounds__(Launch)
gemv_swiglu_q6k_mmvq_kernel(float* __restrict__ hidden, const unsigned char* __restrict__ gate,
                            const unsigned char* __restrict__ up, const std::int8_t* __restrict__ xq,
                            const float* __restrict__ xd, int rows, int cols) {
    const std::int8_t* __restrict__ xqs = xq;
    const float* __restrict__ xds = xd;
    maybe_stage_q8x(xqs, xds, cols);
    const int row = gemv_row();
    const int tid = gemv_tid();
    if (row >= rows) {
        return;
    }
    const int supers = cols / 256;
    float ag = q6k_mmvq_walk<OneTrip>(gate, rows, row, xqs, xds, supers, tid);
    float au = q6k_mmvq_walk<OneTrip>(up, rows, row, xqs, xds, supers, tid);
    reduce_wg2(ag, au, &ag, &au);
    if (tid == 0) {
        hidden[row] = silu_dev(ag) * au;
    }
}

__global__ void __launch_bounds__(256)
gemv_f32_multi_kernel(float* __restrict__ y0, const float* __restrict__ w0, int r0,
                      float* __restrict__ y1, const float* __restrict__ w1, int r1,
                      float* __restrict__ y2, const float* __restrict__ w2, int r2,
                      float* __restrict__ y3, const float* __restrict__ w3, int r3,
                      const float* __restrict__ x, int cols) {
    const int row = static_cast<int>(blockIdx.x);
    const int total = r0 + r1 + r2 + r3;
    if (row >= total) {
        return;
    }
    float* y = y0;
    const float* weight = w0;
    int local = row;
    if (row >= r0 + r1 + r2) {
        y = y3;
        weight = w3;
        local = row - (r0 + r1 + r2);
    } else if (row >= r0 + r1) {
        y = y2;
        weight = w2;
        local = row - (r0 + r1);
    } else if (row >= r0) {
        y = y1;
        weight = w1;
        local = row - r0;
    }
    const float* row_w = weight + static_cast<std::size_t>(local) * cols;
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);
    float acc = 0.0f;
    if ((cols & 3) == 0) {
        const int n4 = cols >> 2;
        const float4* w4 = reinterpret_cast<const float4*>(row_w);
        const float4* x4 = reinterpret_cast<const float4*>(x);
        for (int j = tid; j < n4; j += nthreads) {
            const float4 wv = w4[j];
            const float4 xv = x4[j];
            acc += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
        }
    } else {
        for (int j = tid; j < cols; j += nthreads) {
            acc += row_w[j] * x[j];
        }
    }
    __shared__ float smem[kGemvWorkgroup];
    acc = block_reduce_sum(acc, smem);
    if (tid == 0) {
        y[local] = acc;
    }
}

__global__ void tile_l2_scale_kernel(float* __restrict__ dst, const float* __restrict__ src,
                                     int n_dst, int n_src, int dim, float eps, float scale) {
    const int d = static_cast<int>(blockIdx.x);
    if (d >= n_dst) {
        return;
    }
    const int s = d % n_src;
    float* row = dst + d * dim;
    const float* in = src + s * dim;
    __shared__ float smem[kGemvWorkgroup];
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);
    float ss = 0.0f;
    for (int i = tid; i < dim; i += nthreads) {
        const float v = in[i];
        row[i] = v;
        ss += v * v;
    }
    ss = block_reduce_sum(ss, smem);
    const float inv = rsqrtf(ss + eps) * scale;
    for (int i = tid; i < dim; i += nthreads) {
        row[i] *= inv;
    }
}

__global__ void tile_l2_pair_kernel(float* __restrict__ q_dst, const float* __restrict__ q_src,
                                    float* __restrict__ k_dst, const float* __restrict__ k_src,
                                    int n_dst, int n_src, int dim, float eps, float q_scale,
                                    float k_scale) {
    const int idx = static_cast<int>(blockIdx.x);
    if (idx >= 2 * n_dst) {
        return;
    }
    const bool on_q = idx < n_dst;
    const int d = on_q ? idx : idx - n_dst;
    const int s = d % n_src;
    float* row = (on_q ? q_dst : k_dst) + d * dim;
    const float* in = (on_q ? q_src : k_src) + s * dim;
    const float scale = on_q ? q_scale : k_scale;
    __shared__ float smem[kGemvWorkgroup];
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);
    float ss = 0.0f;
    for (int i = tid; i < dim; i += nthreads) {
        const float v = in[i];
        row[i] = v;
        ss += v * v;
    }
    ss = block_reduce_sum(ss, smem);
    const float inv = rsqrtf(ss + eps) * scale;
    for (int i = tid; i < dim; i += nthreads) {
        row[i] *= inv;
    }
}

// tile_l2_pair and gdn_gates do not depend on each other. One grid.
template<int Dim>
__global__ void __launch_bounds__(256)
gdn_tile_gates_kernel(float* __restrict__ q_dst, const float* __restrict__ q_src,
                      float* __restrict__ k_dst, const float* __restrict__ k_src,
                      float* __restrict__ decay, float* __restrict__ beta,
                      const float* __restrict__ alpha, const float* __restrict__ dt,
                      const float* __restrict__ a, int n_dst, int n_src, int dim, float eps,
                      float q_scale, float k_scale) {
    const int idx = static_cast<int>(blockIdx.x);
    const int width = Dim > 0 ? Dim : dim;
    if (idx < 2 * n_dst) {
        const bool on_q = idx < n_dst;
        const int d = on_q ? idx : idx - n_dst;
        const int s = d % n_src;
        float* row = (on_q ? q_dst : k_dst) + d * width;
        const float* in = (on_q ? q_src : k_src) + s * width;
        const float scale = on_q ? q_scale : k_scale;
        __shared__ float smem[kGemvWorkgroup];
        const int tid = static_cast<int>(threadIdx.x);
        const int nthreads = static_cast<int>(blockDim.x);
        if constexpr (Dim > 0) {
            const float val = in[tid];
            const float ss = block_reduce_sum(val * val, smem);
            row[tid] = val * rsqrtf(ss + eps) * scale;
        } else {
            float ss = 0.0f;
            for (int i = tid; i < dim; i += nthreads) {
                const float val = in[i];
                row[i] = val;
                ss += val * val;
            }
            ss = block_reduce_sum(ss, smem);
            const float inv = rsqrtf(ss + eps) * scale;
            for (int i = tid; i < dim; i += nthreads) {
                row[i] *= inv;
            }
        }
    }
    if (idx < n_dst && threadIdx.x == 0) {
        gdn_apply_gate(decay, beta, alpha, dt, a, idx);
    }
}

// conv_split + tile_gates. Official 27B: 80 WGs of 128 (16 q src + 16 k src +
// 48 v dest) instead of a 10240-thread conv plus 96 dest WGs. Drops the q/k
// write and one graph node per GDN layer (48 on the pin). Each conv element
// updates state once.
template <int Dim, int Kernel>
__global__ void __launch_bounds__(256)
gdn_conv_tile_gates_kernel(float* __restrict__ q_dst, float* __restrict__ k_dst,
                           float* __restrict__ v, float* __restrict__ state,
                           const float* __restrict__ x, const float* __restrict__ weight,
                           float* __restrict__ decay, float* __restrict__ beta,
                           const float* __restrict__ alpha, const float* __restrict__ dt,
                           const float* __restrict__ a, int key_dim, int n_src, int n_dst, int dim,
                           int kernel, float eps, float q_scale, float k_scale) {
    const int idx = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);
    const int width = Dim > 0 ? Dim : dim;
    const int n_qk = n_src * 2;
    __shared__ float smem[kGemvWorkgroup];
    if constexpr (Dim > 0) {
        (void)dim;
        (void)nthreads;
    }
    if (idx < n_qk) {
        const bool on_q = idx < n_src;
        const int s = on_q ? idx : idx - n_src;
        const int base = on_q ? s * width : key_dim + s * width;
        float* dst = on_q ? q_dst : k_dst;
        const float scale = on_q ? q_scale : k_scale;
        if constexpr (Dim > 0) {
            const float y = gdn_conv_at<Kernel>(state, x, weight, base + tid, kernel);
            const float ss = block_reduce_sum(y * y, smem);
            const float out = y * rsqrtf(ss + eps) * scale;
            for (int d = s; d < n_dst; d += n_src) {
                dst[d * Dim + tid] = out;
            }
        } else {
            float* first = dst + s * width;
            float ss = 0.0f;
            for (int i = tid; i < dim; i += nthreads) {
                const float y = gdn_conv_at<Kernel>(state, x, weight, base + i, kernel);
                first[i] = y;
                ss += y * y;
            }
            ss = block_reduce_sum(ss, smem);
            const float inv = rsqrtf(ss + eps) * scale;
            for (int i = tid; i < dim; i += nthreads) {
                first[i] *= inv;
            }
            for (int d = s + n_src; d < n_dst; d += n_src) {
                for (int i = tid; i < dim; i += nthreads) {
                    dst[d * width + i] = first[i];
                }
            }
        }
        return;
    }
    const int h = idx - n_qk;
    const int vbase = 2 * key_dim + h * width;
    if constexpr (Dim > 0) {
        v[h * Dim + tid] = gdn_conv_at<Kernel>(state, x, weight, vbase + tid, kernel);
    } else {
        for (int i = tid; i < dim; i += nthreads) {
            v[h * width + i] = gdn_conv_at<Kernel>(state, x, weight, vbase + i, kernel);
        }
    }
    if (tid == 0) {
        gdn_apply_gate(decay, beta, alpha, dt, a, h);
    }
}

int gemv_grid(int rows) {
    return (rows + kGemvRowsPerWg - 1) / kGemvRowsPerWg;
}

unsigned q8x_shared(int cols) {
    if constexpr (kLdsQ8xMaxBytes <= 0) {
        (void)cols;
        return 0;
    }
    const int n = q8x_lds_bytes(cols);
    if (n <= 0 || n > kLdsQ8xMaxBytes) {
        return 0;
    }
    return static_cast<unsigned>(n);
}

struct ArgMaxScratch {
    float* vals = nullptr;
    int* idxs = nullptr;
    int* out = nullptr;
    int cap = 0;
};

ArgMaxScratch g_argmax;

void argmax_ensure(int nblocks) {
    if (nblocks <= g_argmax.cap) {
        return;
    }
    if (g_argmax.vals != nullptr) {
        hip_synchronize();
        hip_free(g_argmax.vals);
        hip_free(g_argmax.idxs);
        hip_free(g_argmax.out);
    }
    g_argmax.vals = static_cast<float*>(hip_alloc(sizeof(float) * static_cast<std::size_t>(nblocks)));
    g_argmax.idxs = static_cast<int*>(hip_alloc(sizeof(int) * static_cast<std::size_t>(nblocks)));
    g_argmax.out = static_cast<int*>(hip_alloc(sizeof(int)));
    g_argmax.cap = nblocks;
}

__device__ void better_max_idx(float& v, int& i, float ov, int oi) {
    if (ov > v || (ov == v && oi < i)) {
        v = ov;
        i = oi;
    }
}

__device__ void block_reduce_max_idx(float& v, int& idx, float* sv, int* si) {
    const int tid = static_cast<int>(threadIdx.x);
    sv[tid] = v;
    si[tid] = idx;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) / 2; stride >= kWavefront; stride >>= 1) {
        if (tid < stride) {
            better_max_idx(sv[tid], si[tid], sv[tid + stride], si[tid + stride]);
        }
        __syncthreads();
    }
    if (tid < kWavefront) {
        float wv = sv[tid];
        int wi = si[tid];
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_down(wv, off, kWavefront);
            const int oi = __shfl_down(wi, off, kWavefront);
            if (tid + off < kWavefront) {
                better_max_idx(wv, wi, ov, oi);
            }
        }
        if (tid == 0) {
            sv[0] = wv;
            si[0] = wi;
        }
    }
    __syncthreads();
    v = sv[0];
    idx = si[0];
}

template<bool OneTrip>
__global__ void __launch_bounds__(256)
argmax_partial_kernel(const float* __restrict__ x, int n, float* __restrict__ out_v,
                      int* __restrict__ out_i) {
    const int tid = static_cast<int>(threadIdx.x);
    float best = -INFINITY;
    int bi = 0;
    if constexpr (OneTrip) {
        (void)n;
        const int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + tid;
        best = x[i];
        bi = i;
    } else {
        const int stride = static_cast<int>(gridDim.x) * static_cast<int>(blockDim.x);
        for (int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + tid; i < n;
             i += stride) {
            const float v = x[i];
            if (v > best) {
                best = v;
                bi = i;
            }
        }
    }
    __shared__ float sv[kGemvWorkgroup];
    __shared__ int si[kGemvWorkgroup];
    block_reduce_max_idx(best, bi, sv, si);
    if (tid == 0) {
        out_v[blockIdx.x] = best;
        out_i[blockIdx.x] = bi;
    }
}

__global__ void __launch_bounds__(256)
argmax_final_kernel(const float* __restrict__ vals, const int* __restrict__ idxs, int n,
                    int* __restrict__ out) {
    const int tid = static_cast<int>(threadIdx.x);
    float best = -INFINITY;
    int bi = 0;
    for (int i = tid; i < n; i += static_cast<int>(blockDim.x)) {
        better_max_idx(best, bi, vals[i], idxs[i]);
    }
    __shared__ float sv[kGemvWorkgroup];
    __shared__ int si[kGemvWorkgroup];
    block_reduce_max_idx(best, bi, sv, si);
    if (tid == 0) {
        *out = bi;
    }
}

// Decode writes the next id and commits in the same final WG. Prefill still
// uses argmax_write so the first generated token is seeded before the loop.
__global__ void __launch_bounds__(256)
argmax_final_commit_kernel(const float* __restrict__ vals, const int* __restrict__ idxs, int n,
                           int* __restrict__ token, int* __restrict__ ids, int* __restrict__ index,
                           int* __restrict__ pos) {
    const int tid = static_cast<int>(threadIdx.x);
    float best = -INFINITY;
    int bi = 0;
    for (int i = tid; i < n; i += static_cast<int>(blockDim.x)) {
        better_max_idx(best, bi, vals[i], idxs[i]);
    }
    __shared__ float sv[kGemvWorkgroup];
    __shared__ int si[kGemvWorkgroup];
    block_reduce_max_idx(best, bi, sv, si);
    if (tid == 0) {
        *token = bi;
        const int i = *index;
        ids[i] = bi;
        *index = i + 1;
        *pos += 1;
    }
}

__global__ void seed_generated_kernel(int* __restrict__ ids, int* __restrict__ index,
                                      const int* __restrict__ token) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ids[0] = *token;
        *index = 1;
    }
}

__global__ void commit_generated_kernel(int* __restrict__ ids, int* __restrict__ index,
                                        const int* __restrict__ token, int* __restrict__ pos) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const int i = *index;
        ids[i] = *token;
        *index = i + 1;
        *pos += 1;
    }
}

template<int Threads, bool OneTrip, int Launch>
void hip_gemv_q4k(float* y, const unsigned char* packed, int rows, int cols, const float* addend,
                  dim3 grid, dim3 block, unsigned shmem) {
    hipLaunchKernelGGL(gemv_q4k_mmvq_kernel<Threads, OneTrip, Launch>, grid, block, shmem,
                       hip_stream(), y, packed, g_q8x.qs, g_q8x.d, rows, cols, addend);
}

template<int Threads, bool OneTrip>
void hip_gemv_q4k_bounds(float* y, const unsigned char* packed, int rows, int cols,
                         const float* addend, dim3 grid, dim3 block, unsigned shmem) {
    const int launch = static_cast<int>(block.x);
    if (launch == kMmvqLaunch96) {
        hip_gemv_q4k<Threads, OneTrip, kMmvqLaunch96>(y, packed, rows, cols, addend, grid, block,
                                                      shmem);
    } else if (launch == kMmvqLaunch160) {
        hip_gemv_q4k<Threads, OneTrip, kMmvqLaunch160>(y, packed, rows, cols, addend, grid, block,
                                                       shmem);
    } else {
        hip_gemv_q4k<Threads, OneTrip, kMmvqLaunch256>(y, packed, rows, cols, addend, grid, block,
                                                       shmem);
    }
}

template<int Threads>
void launch_gemv_q4k_mmvq_map(float* y, const unsigned char* packed, int rows, int cols,
                              const float* addend, dim3 grid, dim3 block, unsigned shmem) {
    if (q4_mmvq_one_trip(cols)) {
        hip_gemv_q4k_bounds<Threads, true>(y, packed, rows, cols, addend, grid, block, shmem);
    } else {
        hip_gemv_q4k_bounds<Threads, false>(y, packed, rows, cols, addend, grid, block, shmem);
    }
}

void launch_gemv_q4k_mmvq(float* y, const unsigned char* packed, int rows, int cols,
                          const float* addend) {
    const int threads = q4_mmvq_threads(cols / 256);
    const dim3 grid(gemv_grid(rows));
    const dim3 block(q4_mmvq_launch(cols));
    const unsigned shmem = q8x_shared(cols);
    switch (threads) {
        case kQ4MmvqThreadsPerSuper:
            launch_gemv_q4k_mmvq_map<kQ4MmvqThreadsPerSuper>(y, packed, rows, cols, addend, grid,
                                                             block, shmem);
            return;
        case kQ4MmvqMidThreadsPerSuper:
            launch_gemv_q4k_mmvq_map<kQ4MmvqMidThreadsPerSuper>(y, packed, rows, cols, addend, grid,
                                                                block, shmem);
            return;
        case kQ4MmvqOctThreadsPerSuper:
            launch_gemv_q4k_mmvq_map<kQ4MmvqOctThreadsPerSuper>(y, packed, rows, cols, addend, grid,
                                                                block, shmem);
            return;
        case kQ4MmvqDownThreadsPerSuper:
            launch_gemv_q4k_mmvq_map<kQ4MmvqDownThreadsPerSuper>(y, packed, rows, cols, addend, grid,
                                                                 block, shmem);
            return;
    }
    throw std::logic_error("gemv_q4k: unhandled Q4 MMVQ map");
}

template<int Threads, bool OneTrip, int Launch>
void hip_gemv_q4k_multi(float* y0, float* y1, float* y2, float* y3, const unsigned char* p0,
                        const unsigned char* p1, const unsigned char* p2, const unsigned char* p3,
                        int r0, int r1, int r2, int r3, int cols, dim3 grid, dim3 block,
                        unsigned shmem) {
    hipLaunchKernelGGL(gemv_q4k_mmvq_multi_kernel<Threads, OneTrip, Launch>, grid, block, shmem,
                       hip_stream(), y0, y1, y2, y3, p0, p1, p2, p3, g_q8x.qs, g_q8x.d, r0, r1, r2,
                       r3, cols);
}

template<int Threads, bool OneTrip>
void hip_gemv_q4k_multi_bounds(float* y0, float* y1, float* y2, float* y3, const unsigned char* p0,
                               const unsigned char* p1, const unsigned char* p2,
                               const unsigned char* p3, int r0, int r1, int r2, int r3, int cols,
                               dim3 grid, dim3 block, unsigned shmem) {
    const int launch = static_cast<int>(block.x);
    if (launch == kMmvqLaunch96) {
        hip_gemv_q4k_multi<Threads, OneTrip, kMmvqLaunch96>(y0, y1, y2, y3, p0, p1, p2, p3, r0, r1,
                                                            r2, r3, cols, grid, block, shmem);
    } else if (launch == kMmvqLaunch160) {
        hip_gemv_q4k_multi<Threads, OneTrip, kMmvqLaunch160>(y0, y1, y2, y3, p0, p1, p2, p3, r0, r1,
                                                             r2, r3, cols, grid, block, shmem);
    } else {
        hip_gemv_q4k_multi<Threads, OneTrip, kMmvqLaunch256>(y0, y1, y2, y3, p0, p1, p2, p3, r0, r1,
                                                             r2, r3, cols, grid, block, shmem);
    }
}

template<int Threads>
void launch_gemv_q4k_mmvq_multi_map(float* y0, float* y1, float* y2, float* y3,
                                    const unsigned char* p0, const unsigned char* p1,
                                    const unsigned char* p2, const unsigned char* p3, int r0, int r1,
                                    int r2, int r3, int cols, dim3 grid, dim3 block,
                                    unsigned shmem) {
    if (q4_mmvq_one_trip(cols)) {
        hip_gemv_q4k_multi_bounds<Threads, true>(y0, y1, y2, y3, p0, p1, p2, p3, r0, r1, r2, r3,
                                                 cols, grid, block, shmem);
    } else {
        hip_gemv_q4k_multi_bounds<Threads, false>(y0, y1, y2, y3, p0, p1, p2, p3, r0, r1, r2, r3,
                                                  cols, grid, block, shmem);
    }
}

void launch_gemv_q4k_mmvq_multi(float* y0, float* y1, float* y2, float* y3, const unsigned char* p0,
                                const unsigned char* p1, const unsigned char* p2,
                                const unsigned char* p3, int r0, int r1, int r2, int r3, int cols,
                                dim3 grid, unsigned shmem) {
    const int threads = q4_mmvq_threads(cols / 256);
    const dim3 block(q4_mmvq_launch(cols));
    switch (threads) {
        case kQ4MmvqThreadsPerSuper:
            launch_gemv_q4k_mmvq_multi_map<kQ4MmvqThreadsPerSuper>(y0, y1, y2, y3, p0, p1, p2, p3,
                                                                   r0, r1, r2, r3, cols, grid, block,
                                                                   shmem);
            return;
        case kQ4MmvqMidThreadsPerSuper:
            launch_gemv_q4k_mmvq_multi_map<kQ4MmvqMidThreadsPerSuper>(
                y0, y1, y2, y3, p0, p1, p2, p3, r0, r1, r2, r3, cols, grid, block, shmem);
            return;
        case kQ4MmvqOctThreadsPerSuper:
            launch_gemv_q4k_mmvq_multi_map<kQ4MmvqOctThreadsPerSuper>(
                y0, y1, y2, y3, p0, p1, p2, p3, r0, r1, r2, r3, cols, grid, block, shmem);
            return;
        case kQ4MmvqDownThreadsPerSuper:
            launch_gemv_q4k_mmvq_multi_map<kQ4MmvqDownThreadsPerSuper>(
                y0, y1, y2, y3, p0, p1, p2, p3, r0, r1, r2, r3, cols, grid, block, shmem);
            return;
    }
    throw std::logic_error("gemv_multi q4k: unhandled Q4 MMVQ map");
}

template<int Threads, bool OneTrip, int Launch>
void hip_gemv_swiglu_q4k(float* hidden, const unsigned char* gate, const unsigned char* up, int rows,
                         int cols, dim3 grid, dim3 block, unsigned shmem) {
    hipLaunchKernelGGL(gemv_swiglu_q4k_mmvq_kernel<Threads, OneTrip, Launch>, grid, block, shmem,
                       hip_stream(), hidden, gate, up, g_q8x.qs, g_q8x.d, rows, cols);
}

template<int Threads, bool OneTrip>
void hip_gemv_swiglu_q4k_bounds(float* hidden, const unsigned char* gate, const unsigned char* up,
                                int rows, int cols, dim3 grid, dim3 block, unsigned shmem) {
    const int launch = static_cast<int>(block.x);
    if (launch == kMmvqLaunch96) {
        hip_gemv_swiglu_q4k<Threads, OneTrip, kMmvqLaunch96>(hidden, gate, up, rows, cols, grid,
                                                             block, shmem);
    } else if (launch == kMmvqLaunch160) {
        hip_gemv_swiglu_q4k<Threads, OneTrip, kMmvqLaunch160>(hidden, gate, up, rows, cols, grid,
                                                              block, shmem);
    } else {
        hip_gemv_swiglu_q4k<Threads, OneTrip, kMmvqLaunch256>(hidden, gate, up, rows, cols, grid,
                                                              block, shmem);
    }
}

template<int Threads>
void launch_gemv_swiglu_q4k_map(float* hidden, const unsigned char* gate, const unsigned char* up,
                                int rows, int cols, dim3 grid, dim3 block, unsigned shmem) {
    if (q4_mmvq_one_trip(cols)) {
        hip_gemv_swiglu_q4k_bounds<Threads, true>(hidden, gate, up, rows, cols, grid, block, shmem);
    } else {
        hip_gemv_swiglu_q4k_bounds<Threads, false>(hidden, gate, up, rows, cols, grid, block, shmem);
    }
}

void launch_gemv_swiglu_q4k(float* hidden, const unsigned char* gate, const unsigned char* up,
                            int rows, int cols) {
    const int threads = q4_mmvq_threads(cols / 256);
    const dim3 grid(gemv_grid(rows));
    const dim3 block(q4_mmvq_launch(cols));
    const unsigned shmem = q8x_shared(cols);
    switch (threads) {
        case kQ4MmvqThreadsPerSuper:
            launch_gemv_swiglu_q4k_map<kQ4MmvqThreadsPerSuper>(hidden, gate, up, rows, cols, grid,
                                                               block, shmem);
            return;
        case kQ4MmvqMidThreadsPerSuper:
            launch_gemv_swiglu_q4k_map<kQ4MmvqMidThreadsPerSuper>(hidden, gate, up, rows, cols, grid,
                                                                  block, shmem);
            return;
        case kQ4MmvqOctThreadsPerSuper:
            launch_gemv_swiglu_q4k_map<kQ4MmvqOctThreadsPerSuper>(hidden, gate, up, rows, cols, grid,
                                                                  block, shmem);
            return;
        case kQ4MmvqDownThreadsPerSuper:
            launch_gemv_swiglu_q4k_map<kQ4MmvqDownThreadsPerSuper>(hidden, gate, up, rows, cols, grid,
                                                                   block, shmem);
            return;
    }
    throw std::logic_error("gemv_swiglu q4k: unhandled Q4 MMVQ map");
}

template<bool Tight, bool OneTrip, int Launch>
void hip_gemv_q8(float* y, const unsigned char* packed, int rows, int cols, const float* addend,
                 dim3 grid, dim3 block, unsigned shmem) {
    hipLaunchKernelGGL(gemv_q8_mmvq_kernel<Tight, OneTrip, Launch>, grid, block, shmem, hip_stream(),
                       y, packed, g_q8x.qs, g_q8x.d, rows, cols, addend);
}

template<bool Tight, bool OneTrip>
void hip_gemv_q8_bounds(float* y, const unsigned char* packed, int rows, int cols,
                        const float* addend, dim3 grid, dim3 block, unsigned shmem) {
    const int launch = static_cast<int>(block.x);
    if (launch == kMmvqLaunch96) {
        hip_gemv_q8<Tight, OneTrip, kMmvqLaunch96>(y, packed, rows, cols, addend, grid, block,
                                                   shmem);
    } else if (launch == kMmvqLaunch160) {
        hip_gemv_q8<Tight, OneTrip, kMmvqLaunch160>(y, packed, rows, cols, addend, grid, block,
                                                    shmem);
    } else {
        hip_gemv_q8<Tight, OneTrip, kMmvqLaunch256>(y, packed, rows, cols, addend, grid, block,
                                                    shmem);
    }
}

template<bool Tight>
void launch_gemv_q8_mmvq_tight(float* y, const unsigned char* packed, int rows, int cols,
                               const float* addend, dim3 grid, dim3 block, unsigned shmem) {
    if (q8_mmvq_one_trip(cols)) {
        hip_gemv_q8_bounds<Tight, true>(y, packed, rows, cols, addend, grid, block, shmem);
    } else {
        hip_gemv_q8_bounds<Tight, false>(y, packed, rows, cols, addend, grid, block, shmem);
    }
}

void launch_gemv_q8_mmvq(float* y, const unsigned char* packed, int rows, int cols,
                         const float* addend) {
    const dim3 grid(gemv_grid(rows));
    const dim3 block(q8_mmvq_launch(cols));
    const unsigned shmem = q8x_shared(cols);
    if (q8_mmvq_tight(cols)) {
        launch_gemv_q8_mmvq_tight<true>(y, packed, rows, cols, addend, grid, block, shmem);
        return;
    }
    launch_gemv_q8_mmvq_tight<false>(y, packed, rows, cols, addend, grid, block, shmem);
}

template<bool Tight, bool OneTrip, int Launch>
void hip_gemv_q8_multi(float* y0, float* y1, float* y2, float* y3, const unsigned char* p0,
                       const unsigned char* p1, const unsigned char* p2, const unsigned char* p3,
                       int r0, int r1, int r2, int r3, int cols, dim3 grid, dim3 block,
                       unsigned shmem) {
    hipLaunchKernelGGL(gemv_q8_mmvq_multi_kernel<Tight, OneTrip, Launch>, grid, block, shmem,
                       hip_stream(), y0, y1, y2, y3, p0, p1, p2, p3, g_q8x.qs, g_q8x.d, r0, r1, r2,
                       r3, cols);
}

template<bool Tight, bool OneTrip>
void hip_gemv_q8_multi_bounds(float* y0, float* y1, float* y2, float* y3, const unsigned char* p0,
                              const unsigned char* p1, const unsigned char* p2,
                              const unsigned char* p3, int r0, int r1, int r2, int r3, int cols,
                              dim3 grid, dim3 block, unsigned shmem) {
    const int launch = static_cast<int>(block.x);
    if (launch == kMmvqLaunch96) {
        hip_gemv_q8_multi<Tight, OneTrip, kMmvqLaunch96>(y0, y1, y2, y3, p0, p1, p2, p3, r0, r1, r2,
                                                         r3, cols, grid, block, shmem);
    } else if (launch == kMmvqLaunch160) {
        hip_gemv_q8_multi<Tight, OneTrip, kMmvqLaunch160>(y0, y1, y2, y3, p0, p1, p2, p3, r0, r1, r2,
                                                          r3, cols, grid, block, shmem);
    } else {
        hip_gemv_q8_multi<Tight, OneTrip, kMmvqLaunch256>(y0, y1, y2, y3, p0, p1, p2, p3, r0, r1, r2,
                                                          r3, cols, grid, block, shmem);
    }
}

template<bool Tight>
void launch_gemv_q8_mmvq_multi_tight(float* y0, float* y1, float* y2, float* y3,
                                     const unsigned char* p0, const unsigned char* p1,
                                     const unsigned char* p2, const unsigned char* p3, int r0,
                                     int r1, int r2, int r3, int cols, dim3 grid, dim3 block,
                                     unsigned shmem) {
    if (q8_mmvq_one_trip(cols)) {
        hip_gemv_q8_multi_bounds<Tight, true>(y0, y1, y2, y3, p0, p1, p2, p3, r0, r1, r2, r3, cols,
                                              grid, block, shmem);
    } else {
        hip_gemv_q8_multi_bounds<Tight, false>(y0, y1, y2, y3, p0, p1, p2, p3, r0, r1, r2, r3, cols,
                                               grid, block, shmem);
    }
}

void launch_gemv_q8_mmvq_multi(float* y0, float* y1, float* y2, float* y3, const unsigned char* p0,
                               const unsigned char* p1, const unsigned char* p2,
                               const unsigned char* p3, int r0, int r1, int r2, int r3, int cols,
                               dim3 grid, unsigned shmem) {
    const dim3 block(q8_mmvq_launch(cols));
    if (q8_mmvq_tight(cols)) {
        launch_gemv_q8_mmvq_multi_tight<true>(y0, y1, y2, y3, p0, p1, p2, p3, r0, r1, r2, r3, cols,
                                              grid, block, shmem);
        return;
    }
    launch_gemv_q8_mmvq_multi_tight<false>(y0, y1, y2, y3, p0, p1, p2, p3, r0, r1, r2, r3, cols,
                                           grid, block, shmem);
}

template<bool Tight, bool OneTrip, int Launch>
void hip_gemv_swiglu_q8(float* hidden, const unsigned char* gate, const unsigned char* up, int rows,
                        int cols, dim3 grid, dim3 block, unsigned shmem) {
    hipLaunchKernelGGL(gemv_swiglu_q8_mmvq_kernel<Tight, OneTrip, Launch>, grid, block, shmem,
                       hip_stream(), hidden, gate, up, g_q8x.qs, g_q8x.d, rows, cols);
}

template<bool Tight, bool OneTrip>
void hip_gemv_swiglu_q8_bounds(float* hidden, const unsigned char* gate, const unsigned char* up,
                               int rows, int cols, dim3 grid, dim3 block, unsigned shmem) {
    const int launch = static_cast<int>(block.x);
    if (launch == kMmvqLaunch96) {
        hip_gemv_swiglu_q8<Tight, OneTrip, kMmvqLaunch96>(hidden, gate, up, rows, cols, grid, block,
                                                          shmem);
    } else if (launch == kMmvqLaunch160) {
        hip_gemv_swiglu_q8<Tight, OneTrip, kMmvqLaunch160>(hidden, gate, up, rows, cols, grid, block,
                                                           shmem);
    } else {
        hip_gemv_swiglu_q8<Tight, OneTrip, kMmvqLaunch256>(hidden, gate, up, rows, cols, grid, block,
                                                           shmem);
    }
}

template<bool Tight>
void launch_gemv_swiglu_q8_tight(float* hidden, const unsigned char* gate, const unsigned char* up,
                                 int rows, int cols, dim3 grid, dim3 block, unsigned shmem) {
    if (q8_mmvq_one_trip(cols)) {
        hip_gemv_swiglu_q8_bounds<Tight, true>(hidden, gate, up, rows, cols, grid, block, shmem);
    } else {
        hip_gemv_swiglu_q8_bounds<Tight, false>(hidden, gate, up, rows, cols, grid, block, shmem);
    }
}

void launch_gemv_swiglu_q8(float* hidden, const unsigned char* gate, const unsigned char* up,
                           int rows, int cols) {
    const dim3 grid(gemv_grid(rows));
    const dim3 block(q8_mmvq_launch(cols));
    const unsigned shmem = q8x_shared(cols);
    if (q8_mmvq_tight(cols)) {
        launch_gemv_swiglu_q8_tight<true>(hidden, gate, up, rows, cols, grid, block, shmem);
        return;
    }
    launch_gemv_swiglu_q8_tight<false>(hidden, gate, up, rows, cols, grid, block, shmem);
}

template<bool OneTrip, int Launch>
void hip_gemv_q6k(float* y, const unsigned char* packed, int rows, int cols, const float* addend,
                  dim3 grid, dim3 block, unsigned shmem) {
    hipLaunchKernelGGL(gemv_q6k_mmvq_kernel<OneTrip, Launch>, grid, block, shmem, hip_stream(), y,
                       packed, g_q8x.qs, g_q8x.d, rows, cols, addend);
}

template<bool OneTrip>
void hip_gemv_q6k_bounds(float* y, const unsigned char* packed, int rows, int cols,
                         const float* addend, dim3 grid, dim3 block, unsigned shmem) {
    const int launch = static_cast<int>(block.x);
    if (launch == kMmvqLaunch96) {
        hip_gemv_q6k<OneTrip, kMmvqLaunch96>(y, packed, rows, cols, addend, grid, block, shmem);
    } else if (launch == kMmvqLaunch160) {
        hip_gemv_q6k<OneTrip, kMmvqLaunch160>(y, packed, rows, cols, addend, grid, block, shmem);
    } else {
        hip_gemv_q6k<OneTrip, kMmvqLaunch256>(y, packed, rows, cols, addend, grid, block, shmem);
    }
}

void launch_gemv_q6k_mmvq(float* y, const unsigned char* packed, int rows, int cols,
                          const float* addend) {
    const dim3 grid(gemv_grid(rows));
    const dim3 block(q6_mmvq_launch(cols));
    const unsigned shmem = q8x_shared(cols);
    if (q6_mmvq_one_trip(cols)) {
        hip_gemv_q6k_bounds<true>(y, packed, rows, cols, addend, grid, block, shmem);
    } else {
        hip_gemv_q6k_bounds<false>(y, packed, rows, cols, addend, grid, block, shmem);
    }
}

template<bool OneTrip, int Launch>
void hip_gemv_q6k_multi(float* y0, float* y1, float* y2, float* y3, const unsigned char* p0,
                        const unsigned char* p1, const unsigned char* p2, const unsigned char* p3,
                        int r0, int r1, int r2, int r3, int cols, dim3 grid, dim3 block,
                        unsigned shmem) {
    hipLaunchKernelGGL(gemv_q6k_mmvq_multi_kernel<OneTrip, Launch>, grid, block, shmem, hip_stream(),
                       y0, y1, y2, y3, p0, p1, p2, p3, g_q8x.qs, g_q8x.d, r0, r1, r2, r3, cols);
}

template<bool OneTrip>
void hip_gemv_q6k_multi_bounds(float* y0, float* y1, float* y2, float* y3, const unsigned char* p0,
                               const unsigned char* p1, const unsigned char* p2,
                               const unsigned char* p3, int r0, int r1, int r2, int r3, int cols,
                               dim3 grid, dim3 block, unsigned shmem) {
    const int launch = static_cast<int>(block.x);
    if (launch == kMmvqLaunch96) {
        hip_gemv_q6k_multi<OneTrip, kMmvqLaunch96>(y0, y1, y2, y3, p0, p1, p2, p3, r0, r1, r2, r3,
                                                   cols, grid, block, shmem);
    } else if (launch == kMmvqLaunch160) {
        hip_gemv_q6k_multi<OneTrip, kMmvqLaunch160>(y0, y1, y2, y3, p0, p1, p2, p3, r0, r1, r2, r3,
                                                    cols, grid, block, shmem);
    } else {
        hip_gemv_q6k_multi<OneTrip, kMmvqLaunch256>(y0, y1, y2, y3, p0, p1, p2, p3, r0, r1, r2, r3,
                                                    cols, grid, block, shmem);
    }
}

void launch_gemv_q6k_mmvq_multi(float* y0, float* y1, float* y2, float* y3, const unsigned char* p0,
                                const unsigned char* p1, const unsigned char* p2,
                                const unsigned char* p3, int r0, int r1, int r2, int r3, int cols,
                                dim3 grid, unsigned shmem) {
    const dim3 block(q6_mmvq_launch(cols));
    if (q6_mmvq_one_trip(cols)) {
        hip_gemv_q6k_multi_bounds<true>(y0, y1, y2, y3, p0, p1, p2, p3, r0, r1, r2, r3, cols, grid,
                                        block, shmem);
    } else {
        hip_gemv_q6k_multi_bounds<false>(y0, y1, y2, y3, p0, p1, p2, p3, r0, r1, r2, r3, cols, grid,
                                         block, shmem);
    }
}

template<bool OneTrip, int Launch>
void hip_gemv_swiglu_q6k(float* hidden, const unsigned char* gate, const unsigned char* up, int rows,
                         int cols, dim3 grid, dim3 block, unsigned shmem) {
    hipLaunchKernelGGL(gemv_swiglu_q6k_mmvq_kernel<OneTrip, Launch>, grid, block, shmem,
                       hip_stream(), hidden, gate, up, g_q8x.qs, g_q8x.d, rows, cols);
}

template<bool OneTrip>
void hip_gemv_swiglu_q6k_bounds(float* hidden, const unsigned char* gate, const unsigned char* up,
                                int rows, int cols, dim3 grid, dim3 block, unsigned shmem) {
    const int launch = static_cast<int>(block.x);
    if (launch == kMmvqLaunch96) {
        hip_gemv_swiglu_q6k<OneTrip, kMmvqLaunch96>(hidden, gate, up, rows, cols, grid, block,
                                                    shmem);
    } else if (launch == kMmvqLaunch160) {
        hip_gemv_swiglu_q6k<OneTrip, kMmvqLaunch160>(hidden, gate, up, rows, cols, grid, block,
                                                     shmem);
    } else {
        hip_gemv_swiglu_q6k<OneTrip, kMmvqLaunch256>(hidden, gate, up, rows, cols, grid, block,
                                                     shmem);
    }
}

void launch_gemv_swiglu_q6k(float* hidden, const unsigned char* gate, const unsigned char* up,
                            int rows, int cols) {
    const dim3 grid(gemv_grid(rows));
    const dim3 block(q6_mmvq_launch(cols));
    const unsigned shmem = q8x_shared(cols);
    if (q6_mmvq_one_trip(cols)) {
        hip_gemv_swiglu_q6k_bounds<true>(hidden, gate, up, rows, cols, grid, block, shmem);
    } else {
        hip_gemv_swiglu_q6k_bounds<false>(hidden, gate, up, rows, cols, grid, block, shmem);
    }
}

}  // namespace

void gemv(float* y, const float* weight, const float* x, int out_features, int in_features,
          const float* addend) {
    hip_init();
    check(out_features > 0 && in_features > 0, "gemv: empty shape");
    const int block = gemv_block(in_features);
    hipLaunchKernelGGL(gemv_f32_kernel, dim3(out_features), dim3(block), 0, hip_stream(), y, weight, x,
                       out_features, in_features, addend);
    VESPER_HIP_CHECK(hipGetLastError());
}

void gemv_q8(float* y, const std::byte* packed, const float* x, int rows, int cols,
             const float* addend) {
    hip_init();
    check(rows > 0 && cols > 0 && (cols % 32) == 0, "gemv_q8: cols must be a multiple of 32");
    q8x_quantize(x, cols);
    launch_gemv_q8_mmvq(y, reinterpret_cast<const unsigned char*>(packed), rows, cols, addend);
    VESPER_HIP_CHECK(hipGetLastError());
}

void gemv_q4k(float* y, const std::byte* packed, const float* x, int rows, int cols,
              const float* addend) {
    hip_init();
    check(rows > 0 && cols > 0 && (cols % 256) == 0, "gemv_q4k: cols must be a multiple of 256");
    q8x_quantize(x, cols);
    launch_gemv_q4k_mmvq(y, reinterpret_cast<const unsigned char*>(packed), rows, cols, addend);
    VESPER_HIP_CHECK(hipGetLastError());
}

void gemv_q5k(float* y, const std::byte* packed, const float* x, int rows, int cols,
              const float* addend) {
    hip_init();
    check(rows > 0 && cols > 0 && (cols % 256) == 0, "gemv_q5k: cols must be a multiple of 256");
    q8x_quantize(x, cols);
    const unsigned char* p = reinterpret_cast<const unsigned char*>(packed);
    hipLaunchKernelGGL(gemv_q5k_mmvq_kernel, dim3(gemv_grid(rows)), dim3(kGemvWorkgroup), q8x_shared(cols), hip_stream(), y, p,
                       g_q8x.qs, g_q8x.d, rows, cols, addend);
    VESPER_HIP_CHECK(hipGetLastError());
}

void gemv_q6k(float* y, const std::byte* packed, const float* x, int rows, int cols,
              const float* addend) {
    hip_init();
    check(rows > 0 && cols > 0 && (cols % 256) == 0, "gemv_q6k: cols must be a multiple of 256");
    q8x_quantize(x, cols);
    launch_gemv_q6k_mmvq(y, reinterpret_cast<const unsigned char*>(packed), rows, cols, addend);
    VESPER_HIP_CHECK(hipGetLastError());
}

void quantize_activation(const float* x, int cols) {
    hip_init();
    q8x_quantize(x, cols);
}

void gemv_prepared(float* y, const WeightMatrix& w, const float* x, const float* addend) {
    hip_init();
    const int rows = w.rows();
    const int cols = w.cols();
    check(rows > 0 && cols > 0, "gemv_prepared: empty shape");
    switch (w.kind()) {
        case WeightKind::F32:
            gemv(y, w.f32_data(), x, rows, cols, addend);
            return;
        case WeightKind::Q8_0:
            check((cols % 32) == 0, "gemv_prepared q8: cols must be a multiple of 32");
            launch_gemv_q8_mmvq(y, reinterpret_cast<const unsigned char*>(w.packed()), rows, cols,
                                addend);
            VESPER_HIP_CHECK(hipGetLastError());
            return;
        case WeightKind::Q4_K:
            check((cols % 256) == 0, "gemv_prepared q4k: cols must be a multiple of 256");
            launch_gemv_q4k_mmvq(y, reinterpret_cast<const unsigned char*>(w.packed()), rows, cols,
                                 addend);
            VESPER_HIP_CHECK(hipGetLastError());
            return;
        case WeightKind::Q5_K:
            check((cols % 256) == 0, "gemv_prepared q5k: cols must be a multiple of 256");
            hipLaunchKernelGGL(gemv_q5k_mmvq_kernel, dim3(gemv_grid(rows)), dim3(kGemvWorkgroup),
                               q8x_shared(cols), hip_stream(), y,
                               reinterpret_cast<const unsigned char*>(w.packed()), g_q8x.qs,
                               g_q8x.d, rows, cols, addend);
            VESPER_HIP_CHECK(hipGetLastError());
            return;
        case WeightKind::Q6_K:
            check((cols % 256) == 0, "gemv_prepared q6k: cols must be a multiple of 256");
            launch_gemv_q6k_mmvq(y, reinterpret_cast<const unsigned char*>(w.packed()), rows, cols,
                                 addend);
            VESPER_HIP_CHECK(hipGetLastError());
            return;
    }
    throw std::logic_error("gemv_prepared: unhandled WeightKind");
}

void launch_gemv_multi(float* y0, const WeightMatrix& w0, float* y1, const WeightMatrix& w1,
                       float* y2, const WeightMatrix& w2, float* y3, const WeightMatrix* w3,
                       const float* x) {
    hip_init();
    const int r0 = w0.rows();
    const int r1 = w1.rows();
    const int r2 = w2.rows();
    const int r3 = (w3 != nullptr) ? w3->rows() : 0;
    const int cols = w0.cols();
    const int total = r0 + r1 + r2 + r3;
    check(total > 0 && cols > 0, "gemv_multi empty shape");
    check(w0.kind() == w1.kind() && w1.kind() == w2.kind(), "gemv_multi kind mismatch");
    if (w3 != nullptr) {
        check(w3->kind() == w0.kind(), "gemv_multi kind mismatch");
    }
    switch (w0.kind()) {
        case WeightKind::F32: {
            const float* f3 = (w3 != nullptr) ? w3->f32_data() : nullptr;
            hipLaunchKernelGGL(gemv_f32_multi_kernel, dim3(total), dim3(gemv_block(cols)), 0, hip_stream(), y0,
                               w0.f32_data(), r0, y1, w1.f32_data(), r1, y2, w2.f32_data(), r2, y3,
                               f3, r3, x, cols);
            VESPER_HIP_CHECK(hipGetLastError());
            return;
        }
        case WeightKind::Q8_0:
        case WeightKind::Q4_K:
        case WeightKind::Q5_K:
        case WeightKind::Q6_K: {
            if (w0.kind() == WeightKind::Q8_0) {
                check((cols % 32) == 0, "gemv_multi q8 cols");
            } else {
                check((cols % 256) == 0, "gemv_multi k-quant cols");
            }
            q8x_quantize(x, cols);
            const unsigned char* p0 = reinterpret_cast<const unsigned char*>(w0.packed());
            const unsigned char* p1 = reinterpret_cast<const unsigned char*>(w1.packed());
            const unsigned char* p2 = reinterpret_cast<const unsigned char*>(w2.packed());
            const unsigned char* p3 =
                (w3 != nullptr) ? reinterpret_cast<const unsigned char*>(w3->packed()) : nullptr;
            const dim3 grid(gemv_grid(total));
            const unsigned shmem = q8x_shared(cols);
            switch (w0.kind()) {
                case WeightKind::Q8_0:
                    launch_gemv_q8_mmvq_multi(y0, y1, y2, y3, p0, p1, p2, p3, r0, r1, r2, r3, cols,
                                              grid, shmem);
                    VESPER_HIP_CHECK(hipGetLastError());
                    return;
                case WeightKind::Q4_K:
                    launch_gemv_q4k_mmvq_multi(y0, y1, y2, y3, p0, p1, p2, p3, r0, r1, r2, r3,
                                               cols, grid, shmem);
                    VESPER_HIP_CHECK(hipGetLastError());
                    return;
                case WeightKind::Q5_K:
                    hipLaunchKernelGGL(gemv_q5k_mmvq_multi_kernel, grid, dim3(kGemvWorkgroup), shmem,
                                       hip_stream(), y0, y1, y2, y3, p0, p1, p2, p3, g_q8x.qs,
                                       g_q8x.d, r0, r1, r2, r3, cols);
                    VESPER_HIP_CHECK(hipGetLastError());
                    return;
                case WeightKind::Q6_K:
                    launch_gemv_q6k_mmvq_multi(y0, y1, y2, y3, p0, p1, p2, p3, r0, r1, r2, r3, cols,
                                               grid, shmem);
                    VESPER_HIP_CHECK(hipGetLastError());
                    return;
                case WeightKind::F32:
                    throw std::logic_error("gemv_multi: F32 in packed launch");
            }
            throw std::logic_error("gemv_multi: unhandled packed WeightKind");
        }
    }
    throw std::logic_error("gemv_multi: unhandled WeightKind");
}

void gemv3(float* y0, const WeightMatrix& w0, float* y1, const WeightMatrix& w1, float* y2,
           const WeightMatrix& w2, const float* x) {
    if (w0.kind() == w1.kind() && w1.kind() == w2.kind()) {
        launch_gemv_multi(y0, w0, y1, w1, y2, w2, nullptr, nullptr, x);
        return;
    }
    hip_init();
    check(w0.cols() == w1.cols() && w1.cols() == w2.cols(), "gemv3 cols mismatch");
    q8x_quantize(x, w0.cols());
    gemv_prepared(y0, w0, x);
    gemv_prepared(y1, w1, x);
    gemv_prepared(y2, w2, x);
}

void gemv4(float* y0, const WeightMatrix& w0, float* y1, const WeightMatrix& w1, float* y2,
           const WeightMatrix& w2, float* y3, const WeightMatrix& w3, const float* x) {
    if (w0.kind() == w1.kind() && w1.kind() == w2.kind() && w2.kind() == w3.kind()) {
        launch_gemv_multi(y0, w0, y1, w1, y2, w2, y3, &w3, x);
        return;
    }
    hip_init();
    check(w0.cols() == w1.cols() && w1.cols() == w2.cols() && w2.cols() == w3.cols(),
          "gemv4 cols mismatch");
    q8x_quantize(x, w0.cols());
    gemv_prepared(y0, w0, x);
    gemv_prepared(y1, w1, x);
    gemv_prepared(y2, w2, x);
    gemv_prepared(y3, w3, x);
}

void tile_l2_scale(float* dst, const float* src, int n_dst, int n_src, int dim, float eps,
                   float scale) {
    hip_init();
    check(n_dst > 0 && n_src > 0 && dim > 0, "tile_l2_scale empty shape");
    hipLaunchKernelGGL(tile_l2_scale_kernel, dim3(n_dst), dim3(row_workgroup(dim)), 0, hip_stream(), dst, src,
                       n_dst, n_src, dim, eps, scale);
    VESPER_HIP_CHECK(hipGetLastError());
}

void tile_l2_pair(float* q_dst, const float* q_src, float* k_dst, const float* k_src, int n_dst,
                  int n_src, int dim, float eps, float q_scale, float k_scale) {
    hip_init();
    check(n_dst > 0 && n_src > 0 && dim > 0, "tile_l2_pair empty shape");
    hipLaunchKernelGGL(tile_l2_pair_kernel, dim3(2 * n_dst), dim3(row_workgroup(dim)), 0, hip_stream(), q_dst,
                       q_src, k_dst, k_src, n_dst, n_src, dim, eps, q_scale, k_scale);
    VESPER_HIP_CHECK(hipGetLastError());
}

void gdn_tile_gates(float* q_dst, const float* q_src, float* k_dst, const float* k_src,
                    float* decay, float* beta, const float* alpha, const float* dt, const float* a,
                    int n_dst, int n_src, int dim, float eps, float q_scale, float k_scale) {
    hip_init();
    check(n_dst > 0 && n_src > 0 && dim > 0, "gdn_tile_gates empty shape");
    const dim3 grid(2 * n_dst);
    const dim3 block(row_workgroup(dim));
    if (dim == kOfficialGdnDim) {
        hipLaunchKernelGGL(gdn_tile_gates_kernel<kOfficialGdnDim>, grid, block, 0, hip_stream(),
                           q_dst, q_src, k_dst, k_src, decay, beta, alpha, dt, a, n_dst, n_src, dim,
                           eps, q_scale, k_scale);
        VESPER_HIP_CHECK(hipGetLastError());
        return;
    }
    hipLaunchKernelGGL(gdn_tile_gates_kernel<0>, grid, block, 0, hip_stream(), q_dst, q_src, k_dst,
                       k_src, decay, beta, alpha, dt, a, n_dst, n_src, dim, eps, q_scale, k_scale);
    VESPER_HIP_CHECK(hipGetLastError());
}

void gdn_conv_tile_gates(float* q_dst, float* k_dst, float* v, float* state, const float* x,
                         const float* weight, float* decay, float* beta, const float* alpha,
                         const float* dt, const float* a, int key_dim, int value_dim, int n_dst,
                         int n_src, int dim, int kernel, float eps, float q_scale, float k_scale) {
    hip_init();
    check(n_dst > 0 && n_src > 0 && dim > 0 && kernel >= 2, "gdn_conv_tile_gates empty shape");
    check(key_dim == n_src * dim && value_dim == n_dst * dim, "gdn_conv_tile_gates dim mismatch");
    const dim3 grid(2 * n_src + n_dst);
    const dim3 block(row_workgroup(dim));
    if (dim == kOfficialGdnDim && kernel == kGdnConvKernel) {
        hipLaunchKernelGGL(gdn_conv_tile_gates_kernel<kOfficialGdnDim, kGdnConvKernel>, grid, block,
                           0, hip_stream(), q_dst, k_dst, v, state, x, weight, decay, beta, alpha,
                           dt, a, key_dim, n_src, n_dst, dim, kernel, eps, q_scale, k_scale);
        VESPER_HIP_CHECK(hipGetLastError());
        return;
    }
    if (kernel == kGdnConvKernel) {
        hipLaunchKernelGGL(gdn_conv_tile_gates_kernel<0, kGdnConvKernel>, grid, block, 0,
                           hip_stream(), q_dst, k_dst, v, state, x, weight, decay, beta, alpha, dt,
                           a, key_dim, n_src, n_dst, dim, kernel, eps, q_scale, k_scale);
        VESPER_HIP_CHECK(hipGetLastError());
        return;
    }
    hipLaunchKernelGGL(gdn_conv_tile_gates_kernel<0, 0>, grid, block, 0, hip_stream(), q_dst, k_dst,
                       v, state, x, weight, decay, beta, alpha, dt, a, key_dim, n_src, n_dst, dim,
                       kernel, eps, q_scale, k_scale);
    VESPER_HIP_CHECK(hipGetLastError());
}

void rmsnorm(float* out, const float* x, const float* weight, int n, float eps) {
    hip_init();
    check(n > 0, "rmsnorm: empty");
    std::int8_t* qs = nullptr;
    float* d = nullptr;
    float* sum = nullptr;
    bind_fused_q8x(n, out == x, &qs, &d, &sum);
    if (n == kOfficialHidden) {
        hipLaunchKernelGGL(rmsnorm_f32_kernel<kOfficialHidden>, dim3(1), dim3(kGemvWorkgroup), 0,
                           hip_stream(), out, x, weight, n, eps, qs, d, sum);
        VESPER_HIP_CHECK(hipGetLastError());
        if (qs != nullptr) {
            q8x_mark_ready(&g_q8x.ready_cols, n);
        }
        return;
    }
    hipLaunchKernelGGL(rmsnorm_f32_kernel<0>, dim3(1), dim3(kGemvWorkgroup), 0, hip_stream(), out, x,
                       weight, n, eps, qs, d, sum);
    VESPER_HIP_CHECK(hipGetLastError());
    if (qs != nullptr) {
        q8x_mark_ready(&g_q8x.ready_cols, n);
    }
}

void rmsnorm_rows(float* x, const float* weight, int rows, int dim, float eps) {
    hip_init();
    check(rows > 0 && dim > 0, "rmsnorm_rows: empty");
    hipLaunchKernelGGL(rmsnorm_rows_kernel, dim3(rows), dim3(row_workgroup(dim)), 0, hip_stream(), x, weight,
                       rows, dim, eps);
    VESPER_HIP_CHECK(hipGetLastError());
}

void split_gated_q(float* q, float* gate, const float* q_full, int n_heads, int head_dim) {
    hip_init();
    check(n_heads > 0 && head_dim > 0, "split_gated_q empty shape");
    hipLaunchKernelGGL(split_gated_q_kernel, dim3(n_heads), dim3(kGemvWorkgroup), 0, hip_stream(), q, gate,
                       q_full, n_heads, head_dim);
    VESPER_HIP_CHECK(hipGetLastError());
}

void split_gated_q_norm(float* q, float* gate, const float* q_full, const float* weight, int n_heads,
                        int head_dim, float eps) {
    hip_init();
    check(n_heads > 0 && head_dim > 0, "split_gated_q_norm empty shape");
    hipLaunchKernelGGL(split_gated_q_norm_kernel, dim3(n_heads), dim3(kGemvWorkgroup), 0, hip_stream(), q, gate,
                       q_full, weight, n_heads, head_dim, eps);
    VESPER_HIP_CHECK(hipGetLastError());
}

void rmsnorm_silu_mul(float* y, const float* z, const float* weight, int rows, int dim, float eps) {
    hip_init();
    check(rows > 0 && dim > 0, "rmsnorm_silu_mul empty shape");
    std::int8_t* qs = nullptr;
    float* d = nullptr;
    float* sum = nullptr;
    const int cols = rows * dim;
    bind_fused_q8x(cols, q8x_can_fuse(dim), &qs, &d, &sum);
    const dim3 grid(rows);
    const dim3 block(row_workgroup(dim));
    if (dim == kOfficialGdnDim) {
        hipLaunchKernelGGL(rmsnorm_silu_mul_kernel<kOfficialGdnDim>, grid, block, 0, hip_stream(), y,
                           z, weight, rows, dim, eps, qs, d, sum);
        VESPER_HIP_CHECK(hipGetLastError());
        if (qs != nullptr) {
            q8x_mark_ready(&g_q8x.ready_cols, cols);
        }
        return;
    }
    hipLaunchKernelGGL(rmsnorm_silu_mul_kernel<0>, grid, block, 0, hip_stream(), y, z, weight, rows,
                       dim, eps, qs, d, sum);
    VESPER_HIP_CHECK(hipGetLastError());
    if (qs != nullptr) {
        q8x_mark_ready(&g_q8x.ready_cols, cols);
    }
}

void tile_heads(float* dst, const float* src, int n_dst, int n_src, int dim) {
    hip_init();
    check(n_dst > 0 && n_src > 0 && dim > 0, "tile_heads empty shape");
    hipLaunchKernelGGL(tile_heads_kernel, dim3(n_dst), dim3(kGemvWorkgroup), 0, hip_stream(), dst, src, n_dst,
                       n_src, dim);
    VESPER_HIP_CHECK(hipGetLastError());
}

template <int Rows>
void launch_attn_decode(float* out, const float* q, const float* k, const float* v, const float* gate,
                        int seq_val, const int* pos, int n_q_heads, int n_kv_heads, int head_dim,
                        std::int8_t* qs, float* d, float* sum) {
    if (gdn_delta_tight(head_dim)) {
        hipLaunchKernelGGL(attn_decode_kernel<Rows, true>, dim3(n_q_heads), dim3(kWavefront), 0,
                           hip_stream(), out, q, k, v, gate, seq_val, pos, n_q_heads, n_kv_heads,
                           head_dim, qs, d, sum);
    } else {
        hipLaunchKernelGGL(attn_decode_kernel<Rows, false>, dim3(n_q_heads), dim3(kWavefront), 0,
                           hip_stream(), out, q, k, v, gate, seq_val, pos, n_q_heads, n_kv_heads,
                           head_dim, qs, d, sum);
    }
}

void launch_attn_decode_sharded(float* out, const float* q, const float* k, const float* v,
                                const float* gate, int seq_val, const int* pos, int n_q_heads,
                                int n_kv_heads, int head_dim, std::int8_t* qs, float* d, float* sum) {
    switch (gdn_delta_shard_rows(head_dim)) {
        case 1:
            launch_attn_decode<1>(out, q, k, v, gate, seq_val, pos, n_q_heads, n_kv_heads, head_dim,
                                  qs, d, sum);
            return;
        case 2:
            launch_attn_decode<2>(out, q, k, v, gate, seq_val, pos, n_q_heads, n_kv_heads, head_dim,
                                  qs, d, sum);
            return;
        case 4:
            launch_attn_decode<4>(out, q, k, v, gate, seq_val, pos, n_q_heads, n_kv_heads, head_dim,
                                  qs, d, sum);
            return;
        case 8:
            launch_attn_decode<8>(out, q, k, v, gate, seq_val, pos, n_q_heads, n_kv_heads, head_dim,
                                  qs, d, sum);
            return;
    }
    throw std::logic_error("unhandled attn shard rows");
}

void attn_decode_launch(float* out, const float* q, const float* k, const float* v, const float* gate,
                        int seq_val, const int* pos, int n_q_heads, int n_kv_heads, int head_dim) {
    std::int8_t* qs = nullptr;
    float* d = nullptr;
    float* sum = nullptr;
    const int cols = n_q_heads * head_dim;
    bind_fused_q8x(cols, q8x_can_fuse(head_dim), &qs, &d, &sum);
    launch_attn_decode_sharded(out, q, k, v, gate, seq_val, pos, n_q_heads, n_kv_heads, head_dim, qs,
                               d, sum);
    VESPER_HIP_CHECK(hipGetLastError());
    if (qs != nullptr) {
        q8x_mark_ready(&g_q8x.ready_cols, cols);
    }
}

void attn_decode(float* out, const float* q, const float* k, const float* v, const float* gate,
                 int seq, int n_q_heads, int n_kv_heads, int head_dim) {
    hip_init();
    check(seq > 0 && n_q_heads > 0 && n_kv_heads > 0 && head_dim > 0, "attn_decode empty shape");
    check(n_q_heads % n_kv_heads == 0, "attn_decode GQA");
    check(head_dim <= kGdnDeltaMaxDim, "attn_decode: head_dim exceeds wave shards");
    attn_decode_launch(out, q, k, v, gate, seq, nullptr, n_q_heads, n_kv_heads, head_dim);
}

void attn_decode(float* out, const float* q, const float* k, const float* v, const float* gate,
                 const int* pos, int n_q_heads, int n_kv_heads, int head_dim) {
    hip_init();
    check(pos != nullptr && n_q_heads > 0 && n_kv_heads > 0 && head_dim > 0,
          "attn_decode empty shape");
    check(n_q_heads % n_kv_heads == 0, "attn_decode GQA");
    check(head_dim <= kGdnDeltaMaxDim, "attn_decode: head_dim exceeds wave shards");
    attn_decode_launch(out, q, k, v, gate, 0, pos, n_q_heads, n_kv_heads, head_dim);
}

void add_rmsnorm(float* x, float* residual, const float* weight, int n, float eps) {
    hip_init();
    check(n > 0, "add_rmsnorm: empty");
    std::int8_t* qs = nullptr;
    float* d = nullptr;
    float* sum = nullptr;
    bind_fused_q8x(n, true, &qs, &d, &sum);
    if (n == kOfficialHidden) {
        hipLaunchKernelGGL(add_rmsnorm_kernel<kOfficialHidden>, dim3(1), dim3(kGemvWorkgroup), 0,
                           hip_stream(), x, residual, weight, n, eps, qs, d, sum);
        VESPER_HIP_CHECK(hipGetLastError());
        if (qs != nullptr) {
            q8x_mark_ready(&g_q8x.ready_cols, n);
        }
        return;
    }
    hipLaunchKernelGGL(add_rmsnorm_kernel<0>, dim3(1), dim3(kGemvWorkgroup), 0, hip_stream(), x,
                       residual, weight, n, eps, qs, d, sum);
    VESPER_HIP_CHECK(hipGetLastError());
    if (qs != nullptr) {
        q8x_mark_ready(&g_q8x.ready_cols, n);
    }
}

void copy_rmsnorm(float* x, float* residual, const float* weight, int n, float eps) {
    hip_init();
    check(n > 0, "copy_rmsnorm: empty");
    std::int8_t* qs = nullptr;
    float* d = nullptr;
    float* sum = nullptr;
    bind_fused_q8x(n, true, &qs, &d, &sum);
    if (n == kOfficialHidden) {
        hipLaunchKernelGGL(copy_rmsnorm_kernel<kOfficialHidden>, dim3(1), dim3(kGemvWorkgroup), 0,
                           hip_stream(), x, residual, weight, n, eps, qs, d, sum);
        VESPER_HIP_CHECK(hipGetLastError());
        if (qs != nullptr) {
            q8x_mark_ready(&g_q8x.ready_cols, n);
        }
        return;
    }
    hipLaunchKernelGGL(copy_rmsnorm_kernel<0>, dim3(1), dim3(kGemvWorkgroup), 0, hip_stream(), x,
                       residual, weight, n, eps, qs, d, sum);
    VESPER_HIP_CHECK(hipGetLastError());
    if (qs != nullptr) {
        q8x_mark_ready(&g_q8x.ready_cols, n);
    }
}

void silu_mul(float* y, const float* z, int n) {
    hip_init();
    if (n <= 0) {
        return;
    }
    const int block = 256;
    hipLaunchKernelGGL(silu_mul_kernel, dim3(ceil_div(n, block)), dim3(block), 0, hip_stream(), y, z, n);
    VESPER_HIP_CHECK(hipGetLastError());
}

void gdn_gates(float* decay, float* beta, const float* alpha, const float* dt, const float* a,
               int n) {
    hip_init();
    if (n <= 0) {
        return;
    }
    const int block = 256;
    hipLaunchKernelGGL(gdn_gates_kernel, dim3(ceil_div(n, block)), dim3(block), 0, hip_stream(), decay, beta,
                       alpha, dt, a, n);
    VESPER_HIP_CHECK(hipGetLastError());
}

void split_qkv(float* q, float* k, float* v, const float* qkv, int key_dim, int value_dim) {
    hip_init();
    check(key_dim > 0 && value_dim > 0, "split_qkv empty shape");
    const int total = 2 * key_dim + value_dim;
    const int block = 256;
    hipLaunchKernelGGL(split_qkv_kernel, dim3(ceil_div(total, block)), dim3(block), 0, hip_stream(), q, k, v,
                       qkv, key_dim, value_dim);
    VESPER_HIP_CHECK(hipGetLastError());
}

void gemv_swiglu(float* hidden, const WeightMatrix& gate, const WeightMatrix& up, const float* x) {
    hip_init();
    check(gate.kind() == up.kind(), "gemv_swiglu kind mismatch");
    check(gate.rows() == up.rows() && gate.cols() == up.cols(), "gemv_swiglu shape mismatch");
    const int rows = gate.rows();
    const int cols = gate.cols();
    check(rows > 0 && cols > 0, "gemv_swiglu empty shape");
    switch (gate.kind()) {
        case WeightKind::F32:
            hipLaunchKernelGGL(gemv_swiglu_f32_kernel, dim3(rows), dim3(gemv_block(cols)), 0,
                               hip_stream(), hidden, gate.f32_data(), up.f32_data(), x, rows,
                               cols);
            VESPER_HIP_CHECK(hipGetLastError());
            return;
        case WeightKind::Q8_0: {
            check((cols % 32) == 0, "gemv_swiglu q8: cols must be a multiple of 32");
            q8x_quantize(x, cols);
            launch_gemv_swiglu_q8(hidden, reinterpret_cast<const unsigned char*>(gate.packed()),
                                  reinterpret_cast<const unsigned char*>(up.packed()), rows, cols);
            VESPER_HIP_CHECK(hipGetLastError());
            return;
        }
        case WeightKind::Q4_K: {
            check((cols % 256) == 0, "gemv_swiglu q4k: cols must be a multiple of 256");
            q8x_quantize(x, cols);
            launch_gemv_swiglu_q4k(hidden, reinterpret_cast<const unsigned char*>(gate.packed()),
                                   reinterpret_cast<const unsigned char*>(up.packed()), rows,
                                   cols);
            VESPER_HIP_CHECK(hipGetLastError());
            return;
        }
        case WeightKind::Q5_K: {
            check((cols % 256) == 0, "gemv_swiglu q5k: cols must be a multiple of 256");
            q8x_quantize(x, cols);
            hipLaunchKernelGGL(gemv_swiglu_q5k_mmvq_kernel, dim3(gemv_grid(rows)),
                               dim3(kGemvWorkgroup), q8x_shared(cols), hip_stream(), hidden,
                               reinterpret_cast<const unsigned char*>(gate.packed()),
                               reinterpret_cast<const unsigned char*>(up.packed()), g_q8x.qs,
                               g_q8x.d, rows, cols);
            VESPER_HIP_CHECK(hipGetLastError());
            return;
        }
        case WeightKind::Q6_K: {
            check((cols % 256) == 0, "gemv_swiglu q6k: cols must be a multiple of 256");
            q8x_quantize(x, cols);
            launch_gemv_swiglu_q6k(hidden, reinterpret_cast<const unsigned char*>(gate.packed()),
                                   reinterpret_cast<const unsigned char*>(up.packed()), rows, cols);
            VESPER_HIP_CHECK(hipGetLastError());
            return;
        }
    }
    throw std::logic_error("gemv_swiglu: unhandled WeightKind");
}

void rope_neox(float* q, float* k, int n_q_heads, int n_kv_heads, int head_dim, int rotary_dim,
               int pos, float theta) {
    hip_init();
    check(rotary_dim > 0 && (rotary_dim % 2) == 0 && rotary_dim <= head_dim,
          "rope rotary_dim must be even and <= head_dim");
    const int total = (n_q_heads + n_kv_heads) * (rotary_dim / 2);
    if (total <= 0) {
        return;
    }
    const int block = 64;
    hipLaunchKernelGGL(rope_neox_kernel, dim3(ceil_div(total, block)), dim3(block), 0, hip_stream(), q, k,
                       n_q_heads, n_kv_heads, head_dim, rotary_dim, pos, nullptr, theta);
    VESPER_HIP_CHECK(hipGetLastError());
}

void rope_neox(float* q, float* k, int n_q_heads, int n_kv_heads, int head_dim, int rotary_dim,
               const int* pos, float theta) {
    hip_init();
    check(pos != nullptr, "rope pos");
    check(rotary_dim > 0 && (rotary_dim % 2) == 0 && rotary_dim <= head_dim,
          "rope rotary_dim must be even and <= head_dim");
    const int total = (n_q_heads + n_kv_heads) * (rotary_dim / 2);
    if (total <= 0) {
        return;
    }
    const int block = 64;
    hipLaunchKernelGGL(rope_neox_kernel, dim3(ceil_div(total, block)), dim3(block), 0, hip_stream(), q, k,
                       n_q_heads, n_kv_heads, head_dim, rotary_dim, 0, pos, theta);
    VESPER_HIP_CHECK(hipGetLastError());
}

void rope_neox_k_norm(float* q, float* k, const float* k_weight, int n_q_heads, int n_kv_heads,
                      int head_dim, int rotary_dim, int pos, float theta, float eps) {
    hip_init();
    check(rotary_dim > 0 && (rotary_dim % 2) == 0 && rotary_dim <= head_dim,
          "rope rotary_dim must be even and <= head_dim");
    check(n_q_heads > 0 && n_kv_heads > 0 && head_dim > 0, "rope_neox_k_norm empty shape");
    hipLaunchKernelGGL(rope_neox_k_norm_kernel, dim3(n_q_heads + n_kv_heads), dim3(kGemvWorkgroup),
                       0, hip_stream(), q, k, k_weight, n_q_heads, n_kv_heads, head_dim, rotary_dim,
                       pos, nullptr, theta, eps);
    VESPER_HIP_CHECK(hipGetLastError());
}

void rope_neox_k_norm(float* q, float* k, const float* k_weight, int n_q_heads, int n_kv_heads,
                      int head_dim, int rotary_dim, const int* pos, float theta, float eps) {
    hip_init();
    check(pos != nullptr, "rope pos");
    check(rotary_dim > 0 && (rotary_dim % 2) == 0 && rotary_dim <= head_dim,
          "rope rotary_dim must be even and <= head_dim");
    check(n_q_heads > 0 && n_kv_heads > 0 && head_dim > 0, "rope_neox_k_norm empty shape");
    hipLaunchKernelGGL(rope_neox_k_norm_kernel, dim3(n_q_heads + n_kv_heads), dim3(kGemvWorkgroup),
                       0, hip_stream(), q, k, k_weight, n_q_heads, n_kv_heads, head_dim, rotary_dim,
                       0, pos, theta, eps);
    VESPER_HIP_CHECK(hipGetLastError());
}

void attn_prepare(float* q, float* gate, float* k, float* v, const float* q_full,
                  const float* q_weight, const float* k_weight, float* k_base, float* v_base,
                  const int* pos, int n_q_heads, int n_kv_heads, int head_dim, int rotary_dim,
                  float theta, float eps) {
    hip_init();
    check(pos != nullptr, "attn_prepare pos");
    check(rotary_dim > 0 && (rotary_dim % 2) == 0 && rotary_dim <= head_dim,
          "attn_prepare rotary_dim must be even and <= head_dim");
    check(n_q_heads > 0 && n_kv_heads > 0 && head_dim > 0, "attn_prepare empty shape");
    check(n_q_heads % n_kv_heads == 0, "attn_prepare GQA");
    const dim3 grid(n_q_heads + n_kv_heads);
    const dim3 block(kGemvWorkgroup);
    if (head_dim == kOfficialHeadDim && rotary_dim == kOfficialRopeDim) {
        hipLaunchKernelGGL(attn_prepare_kernel<kOfficialHeadDim, kOfficialRopeDim>, grid, block, 0,
                           hip_stream(), q, gate, k, v, q_full, q_weight, k_weight, k_base, v_base,
                           pos, n_q_heads, n_kv_heads, head_dim, rotary_dim, theta, eps);
        VESPER_HIP_CHECK(hipGetLastError());
        return;
    }
    hipLaunchKernelGGL(attn_prepare_kernel<0, 0>, grid, block, 0, hip_stream(), q, gate, k, v,
                       q_full, q_weight, k_weight, k_base, v_base, pos, n_q_heads, n_kv_heads,
                       head_dim, rotary_dim, theta, eps);
    VESPER_HIP_CHECK(hipGetLastError());
}

void swiglu(float* out, const float* gate, const float* up, int n) {
    hip_init();
    if (n <= 0) {
        return;
    }
    const int block = 256;
    hipLaunchKernelGGL(swiglu_kernel, dim3(ceil_div(n, block)), dim3(block), 0, hip_stream(), out, gate,
                       up, n);
    VESPER_HIP_CHECK(hipGetLastError());
}

void softmax_inplace(float* x, int n) {
    hip_init();
    check(n > 0, "softmax: empty");
    hipLaunchKernelGGL(softmax_kernel, dim3(1), dim3(kGemvWorkgroup), 0, hip_stream(), x, n);
    VESPER_HIP_CHECK(hipGetLastError());
}

__global__ void embed_f32_kernel(float* __restrict__ out, const float* __restrict__ table,
                                const int* __restrict__ token, int hidden) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= hidden) {
        return;
    }
    out[i] = table[static_cast<std::size_t>(*token) * static_cast<std::size_t>(hidden) +
                   static_cast<std::size_t>(i)];
}

__global__ void scatter_row_kernel(float* __restrict__ base, const float* __restrict__ row,
                                  const int* __restrict__ pos, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) {
        return;
    }
    base[static_cast<std::size_t>(*pos) * static_cast<std::size_t>(n) + static_cast<std::size_t>(i)] =
        row[i];
}

__global__ void scatter_kv_kernel(float* __restrict__ k_base, float* __restrict__ v_base,
                                 const float* __restrict__ k, const float* __restrict__ v,
                                 const int* __restrict__ pos, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) {
        return;
    }
    const std::size_t off =
        static_cast<std::size_t>(*pos) * static_cast<std::size_t>(n) + static_cast<std::size_t>(i);
    k_base[off] = k[i];
    v_base[off] = v[i];
}

__global__ void embed_q8_kernel(float* __restrict__ out, const unsigned char* __restrict__ packed,
                               int token_val, const int* token_ptr, int cols, int rows) {
    const int token = decode_i32(token_val, token_ptr);
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= cols) {
        return;
    }
    const int nblocks = cols / kQ8BlockElems;
    const int b = i / kQ8BlockElems;
    const unsigned char* qs = q8_soa_qs(packed, rows, nblocks, token, b);
    out[i] = load_f16(q8_soa_scale(packed, rows, nblocks, token, b)) *
             static_cast<float>(static_cast<std::int8_t>(qs[i % kQ8BlockElems]));
}

__global__ void embed_q4k_kernel(float* __restrict__ out, const unsigned char* __restrict__ packed,
                                int token_val, const int* token_ptr, int cols, int rows) {
    const int token = decode_i32(token_val, token_ptr);
    const int supers = cols / kQ4KBlockElems;
    const int s = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    if (s >= supers) {
        return;
    }
    const unsigned char* hdr = q4k_soa_hdr(packed, rows, supers, token, s);
    const float2 dm = load_f16x2(hdr);
    int scales[8];
    int mins[8];
    q4k_decode_scales(hdr + 4, scales, mins);
    const int group = tid / 64;
    const int r = tid % 32;
    const int hi = (tid / 32) & 1;
    const int is = 2 * group + hi;
    const int half = group / 2;
    const unsigned char* qs = q4k_soa_qs(packed, rows, supers, token, s, half);
    const unsigned char byte = qs[(group % 2) * 32 + r];
    const int q = hi ? (byte >> 4) : (byte & 0x0f);
    out[s * kQ4KBlockElems + tid] =
        dm.x * static_cast<float>(scales[is]) * static_cast<float>(q) -
        dm.y * static_cast<float>(mins[is]);
}

__global__ void embed_q5k_kernel(float* __restrict__ out, const unsigned char* __restrict__ packed,
                                int token_val, const int* token_ptr, int cols) {
    const int token = decode_i32(token_val, token_ptr);
    const int supers = cols / kQ5KBlockElems;
    const int s = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    if (s >= supers) {
        return;
    }
    const unsigned char* blk =
        packed + (static_cast<std::size_t>(token) * supers + s) * kQ5KBlockBytes;
    const float2 dm = load_wf16x2(blk);
    int scales[8];
    int mins[8];
    q4k_decode_scales(blk + 4, scales, mins);
    const int il = tid / 64;
    const int r = tid % 32;
    const int hi = (tid / 32) & 1;
    const int is = 2 * il + hi;
    const unsigned char hm = static_cast<unsigned char>(1u << (2 * il + hi));
    const unsigned char byte = blk[48 + il * 32 + r];
    int q = hi ? (byte >> 4) : (byte & 0x0f);
    if (blk[16 + r] & hm) {
        q += 16;
    }
    out[s * kQ5KBlockElems + tid] =
        dm.x * static_cast<float>(scales[is]) * static_cast<float>(q) -
        dm.y * static_cast<float>(mins[is]);
}

__global__ void embed_q6k_kernel(float* __restrict__ out, const unsigned char* __restrict__ packed,
                                int token_val, const int* token_ptr, int cols, int rows) {
    const int token = decode_i32(token_val, token_ptr);
    const int supers = cols / kQ6KBlockElems;
    const int s = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    if (s >= supers) {
        return;
    }
    const unsigned char* qh = q6k_soa_qh(packed, rows, supers, token, s);
    const unsigned char* scales = q6k_soa_scales(packed, rows, supers, token, s);
    const float d = load_f16(q6k_soa_d(packed, rows, supers, token, s));
    const int ip = tid / 128;
    const int off = tid % 128;
    const int il = off % 32;
    const int slot = off / 32;
    const int is = 8 * ip + il / 16;
    const unsigned char* ql = q6k_soa_ql(packed, rows, supers, token, s, ip);
    const unsigned char ql0 = ql[il];
    const unsigned char ql1 = ql[32 + il];
    const unsigned char qhb = qh[32 * ip + il];
    int q = 0;
    int sc = 0;
    if (slot == 0) {
        q = (ql0 & 0x0f) | static_cast<int>((qhb >> 0) & 3) << 4;
        sc = static_cast<signed char>(scales[is + 0]);
    } else if (slot == 1) {
        q = (ql1 & 0x0f) | static_cast<int>((qhb >> 2) & 3) << 4;
        sc = static_cast<signed char>(scales[is + 2]);
    } else if (slot == 2) {
        q = (ql0 >> 4) | static_cast<int>((qhb >> 4) & 3) << 4;
        sc = static_cast<signed char>(scales[is + 4]);
    } else {
        q = (ql1 >> 4) | static_cast<int>((qhb >> 6) & 3) << 4;
        sc = static_cast<signed char>(scales[is + 6]);
    }
    out[s * kQ6KBlockElems + tid] =
        d * static_cast<float>(sc) * static_cast<float>(q - 32);
}

void embed_row(float* out, const float* table, int token, int hidden) {
    hip_init();
    hip_copy_d2d(out, table + static_cast<std::size_t>(token) * hidden,
                 static_cast<std::size_t>(hidden) * sizeof(float));
}

void embed_row(float* out, const float* table, const int* token, int hidden) {
    hip_init();
    check(token != nullptr && hidden > 0, "embed empty");
    const int block = 256;
    hipLaunchKernelGGL(embed_f32_kernel, dim3(ceil_div(hidden, block)), dim3(block), 0, hip_stream(), out,
                       table, token, hidden);
    VESPER_HIP_CHECK(hipGetLastError());
}

void scatter_row(float* base, const float* row, const int* pos, int n) {
    hip_init();
    check(base != nullptr && row != nullptr && pos != nullptr, "scatter_row null");
    if (n <= 0) {
        return;
    }
    const int block = 256;
    hipLaunchKernelGGL(scatter_row_kernel, dim3(ceil_div(n, block)), dim3(block), 0, hip_stream(), base, row,
                       pos, n);
    VESPER_HIP_CHECK(hipGetLastError());
}

void scatter_kv(float* k_base, float* v_base, const float* k, const float* v, const int* pos, int n) {
    hip_init();
    check(k_base != nullptr && v_base != nullptr && k != nullptr && v != nullptr && pos != nullptr,
          "scatter_kv null");
    if (n <= 0) {
        return;
    }
    const int block = 256;
    hipLaunchKernelGGL(scatter_kv_kernel, dim3(ceil_div(n, block)), dim3(block), 0, hip_stream(), k_base,
                       v_base, k, v, pos, n);
    VESPER_HIP_CHECK(hipGetLastError());
}

namespace {

void embed_packed(float* out, const WeightMatrix& table, int token_val, const int* token_ptr) {
    const int cols = table.cols();
    switch (table.kind()) {
        case WeightKind::F32:
            if (token_ptr != nullptr) {
                embed_row(out, table.f32_data(), token_ptr, cols);
            } else {
                embed_row(out, table.f32_data(), token_val, cols);
            }
            return;
        case WeightKind::Q8_0:
            check(cols > 0 && (cols % kQ8BlockElems) == 0, "embed Q8_0 cols");
            hipLaunchKernelGGL(embed_q8_kernel, dim3(ceil_div(cols, kGemvWorkgroup)),
                               dim3(kGemvWorkgroup), 0, hip_stream(), out,
                               reinterpret_cast<const unsigned char*>(table.packed()), token_val,
                               token_ptr, cols, table.rows());
            VESPER_HIP_CHECK(hipGetLastError());
            return;
        case WeightKind::Q4_K:
            check(cols > 0 && (cols % kQ4KBlockElems) == 0, "embed Q4_K cols");
            hipLaunchKernelGGL(embed_q4k_kernel, dim3(cols / kQ4KBlockElems), dim3(kQ4KBlockElems),
                               0, hip_stream(), out, reinterpret_cast<const unsigned char*>(table.packed()),
                               token_val, token_ptr, cols, table.rows());
            VESPER_HIP_CHECK(hipGetLastError());
            return;
        case WeightKind::Q5_K:
            check(cols > 0 && (cols % kQ5KBlockElems) == 0, "embed Q5_K cols");
            hipLaunchKernelGGL(embed_q5k_kernel, dim3(cols / kQ5KBlockElems), dim3(kQ5KBlockElems),
                               0, hip_stream(), out, reinterpret_cast<const unsigned char*>(table.packed()),
                               token_val, token_ptr, cols);
            VESPER_HIP_CHECK(hipGetLastError());
            return;
        case WeightKind::Q6_K:
            check(cols > 0 && (cols % kQ6KBlockElems) == 0, "embed Q6_K cols");
            hipLaunchKernelGGL(embed_q6k_kernel, dim3(cols / kQ6KBlockElems), dim3(kQ6KBlockElems),
                               0, hip_stream(), out, reinterpret_cast<const unsigned char*>(table.packed()),
                               token_val, token_ptr, cols, table.rows());
            VESPER_HIP_CHECK(hipGetLastError());
            return;
    }
    throw std::logic_error("unhandled WeightKind");
}

}  // namespace

void embed_row(float* out, const WeightMatrix& table, int token) {
    hip_init();
    check(token >= 0 && token < table.rows(), "embed token out of range");
    embed_packed(out, table, token, nullptr);
}

void embed_row(float* out, const WeightMatrix& table, const int* token) {
    hip_init();
    check(token != nullptr, "embed token");
    embed_packed(out, table, 0, token);
}

void attn_scores(float* scores, const float* q, const float* k, int seq, int n_kv_heads,
                 int kv_head, int head_dim) {
    hip_init();
    if (seq <= 0) {
        return;
    }
    const int block = 256;
    hipLaunchKernelGGL(attn_scores_kernel, dim3(ceil_div(seq, block)), dim3(block), 0, hip_stream(),
                       scores, q, k, seq, n_kv_heads, kv_head, head_dim);
    VESPER_HIP_CHECK(hipGetLastError());
}

void attn_mix(float* out, const float* scores, const float* v, int seq, int n_kv_heads,
              int kv_head, int head_dim) {
    hip_init();
    check(head_dim <= kGemvWorkgroup, "attn_mix: head_dim exceeds workgroup");
    hipLaunchKernelGGL(attn_mix_kernel, dim3(1), dim3(head_dim), 0, hip_stream(), out, scores, v, seq,
                       n_kv_heads, kv_head, head_dim);
    VESPER_HIP_CHECK(hipGetLastError());
}

void add_inplace(float* dst, const float* src, int n) {
    hip_init();
    if (n <= 0) {
        return;
    }
    const int block = 256;
    hipLaunchKernelGGL(add_inplace_kernel, dim3(ceil_div(n, block)), dim3(block), 0, hip_stream(), dst,
                       src, n);
    VESPER_HIP_CHECK(hipGetLastError());
}

void copy(float* dst, const float* src, int n) {
    hip_init();
    if (n <= 0) {
        return;
    }
    hip_copy_d2d(dst, src, static_cast<std::size_t>(n) * sizeof(float));
}

void sigmoid_inplace(float* x, int n) {
    hip_init();
    if (n <= 0) {
        return;
    }
    const int block = 256;
    hipLaunchKernelGGL(sigmoid_kernel, dim3(ceil_div(n, block)), dim3(block), 0, hip_stream(), x, n);
    VESPER_HIP_CHECK(hipGetLastError());
}

void silu_inplace(float* x, int n) {
    hip_init();
    if (n <= 0) {
        return;
    }
    const int block = 256;
    hipLaunchKernelGGL(silu_kernel, dim3(ceil_div(n, block)), dim3(block), 0, hip_stream(), x, n);
    VESPER_HIP_CHECK(hipGetLastError());
}

void softplus_inplace(float* x, int n) {
    hip_init();
    if (n <= 0) {
        return;
    }
    const int block = 256;
    hipLaunchKernelGGL(softplus_kernel, dim3(ceil_div(n, block)), dim3(block), 0, hip_stream(), x, n);
    VESPER_HIP_CHECK(hipGetLastError());
}

void exp_inplace(float* x, int n) {
    hip_init();
    if (n <= 0) {
        return;
    }
    const int block = 256;
    hipLaunchKernelGGL(exp_kernel, dim3(ceil_div(n, block)), dim3(block), 0, hip_stream(), x, n);
    VESPER_HIP_CHECK(hipGetLastError());
}

void mul_inplace(float* dst, const float* src, int n) {
    hip_init();
    if (n <= 0) {
        return;
    }
    const int block = 256;
    hipLaunchKernelGGL(mul_inplace_kernel, dim3(ceil_div(n, block)), dim3(block), 0, hip_stream(), dst, src,
                       n);
    VESPER_HIP_CHECK(hipGetLastError());
}

void scale_inplace(float* x, float scale, int n) {
    hip_init();
    if (n <= 0) {
        return;
    }
    const int block = 256;
    hipLaunchKernelGGL(scale_inplace_kernel, dim3(ceil_div(n, block)), dim3(block), 0, hip_stream(), x,
                       scale, n);
    VESPER_HIP_CHECK(hipGetLastError());
}

void l2_normalize_rows(float* x, int rows, int dim, float eps) {
    hip_init();
    if (rows <= 0 || dim <= 0) {
        return;
    }
    hipLaunchKernelGGL(l2_normalize_rows_kernel, dim3(rows), dim3(kWavefront), 0, hip_stream(), x, rows,
                       dim, eps);
    VESPER_HIP_CHECK(hipGetLastError());
}

void gdn_conv_update(float* y, float* state, const float* x, const float* weight, int conv_dim,
                     int kernel) {
    hip_init();
    check(kernel >= 2 && conv_dim > 0, "gdn conv shape");
    const int block = 256;
    const dim3 grid(ceil_div(conv_dim, block));
    if (kernel == kGdnConvKernel) {
        hipLaunchKernelGGL(gdn_conv_update_kernel<kGdnConvKernel>, grid, dim3(block), 0, hip_stream(),
                           y, state, x, weight, conv_dim, kernel);
        VESPER_HIP_CHECK(hipGetLastError());
        return;
    }
    hipLaunchKernelGGL(gdn_conv_update_kernel<0>, grid, dim3(block), 0, hip_stream(), y, state, x,
                       weight, conv_dim, kernel);
    VESPER_HIP_CHECK(hipGetLastError());
}

void gdn_conv_split(float* q, float* k, float* v, float* state, const float* x, const float* weight,
                    int key_dim, int value_dim, int kernel) {
    hip_init();
    check(kernel >= 2 && key_dim > 0 && value_dim > 0, "gdn conv split shape");
    const int qkv_dim = 2 * key_dim + value_dim;
    const int block = 256;
    const dim3 grid(ceil_div(qkv_dim, block));
    if (kernel == kGdnConvKernel) {
        hipLaunchKernelGGL(gdn_conv_split_kernel<kGdnConvKernel>, grid, dim3(block), 0, hip_stream(),
                           q, k, v, state, x, weight, key_dim, value_dim, kernel);
        VESPER_HIP_CHECK(hipGetLastError());
        return;
    }
    hipLaunchKernelGGL(gdn_conv_split_kernel<0>, grid, dim3(block), 0, hip_stream(), q, k, v, state,
                       x, weight, key_dim, value_dim, kernel);
    VESPER_HIP_CHECK(hipGetLastError());
}

template <int Rows>
void launch_gdn_delta_rule(float* y, float* rec, const float* q, const float* k, const float* v,
                           const float* decay, const float* beta, int n_heads, int dim) {
    const int col_blocks = (dim + kGdnDeltaWarps - 1) / kGdnDeltaWarps;
    if (gdn_delta_tight(dim)) {
        hipLaunchKernelGGL(gdn_delta_rule_kernel<Rows, true>, dim3(n_heads, col_blocks),
                           dim3(kWavefront, kGdnDeltaWarps), 0, hip_stream(), y, rec, q, k, v, decay,
                           beta, n_heads, dim);
    } else {
        hipLaunchKernelGGL(gdn_delta_rule_kernel<Rows, false>, dim3(n_heads, col_blocks),
                           dim3(kWavefront, kGdnDeltaWarps), 0, hip_stream(), y, rec, q, k, v, decay,
                           beta, n_heads, dim);
    }
}

void gdn_delta_rule(float* y, float* rec, const float* q, const float* k, const float* v,
                    const float* decay, const float* beta, int n_heads, int dim) {
    hip_init();
    check(n_heads > 0 && dim > 0 && dim <= kGdnDeltaMaxDim, "gdn delta shape");
    switch (gdn_delta_shard_rows(dim)) {
        case 1:
            launch_gdn_delta_rule<1>(y, rec, q, k, v, decay, beta, n_heads, dim);
            VESPER_HIP_CHECK(hipGetLastError());
            return;
        case 2:
            launch_gdn_delta_rule<2>(y, rec, q, k, v, decay, beta, n_heads, dim);
            VESPER_HIP_CHECK(hipGetLastError());
            return;
        case 4:
            launch_gdn_delta_rule<4>(y, rec, q, k, v, decay, beta, n_heads, dim);
            VESPER_HIP_CHECK(hipGetLastError());
            return;
        case 8:
            launch_gdn_delta_rule<8>(y, rec, q, k, v, decay, beta, n_heads, dim);
            VESPER_HIP_CHECK(hipGetLastError());
            return;
    }
    throw std::logic_error("unhandled GDN shard rows");
}

void warmup_decode(int max_cols, int vocab) {
    hip_init();
    if (max_cols > 0) {
        q8x_ensure(max_cols);
    }
    q8x_clear_ready(&g_q8x.ready_cols);
    if (vocab > 0) {
        argmax_ensure(argmax_partial_blocks(vocab));
    }
}

int launch_argmax_partial(const float* x, int n) {
    const int block = kGemvWorkgroup;
    if (argmax_one_trip(n)) {
        const int nblocks = n / block;
        argmax_ensure(nblocks);
        hipLaunchKernelGGL(argmax_partial_kernel<true>, dim3(nblocks), dim3(block), 0, hip_stream(),
                           x, n, g_argmax.vals, g_argmax.idxs);
        return nblocks;
    }
    const int nblocks = argmax_partial_blocks(n);
    argmax_ensure(nblocks);
    hipLaunchKernelGGL(argmax_partial_kernel<false>, dim3(nblocks), dim3(block), 0, hip_stream(), x,
                       n, g_argmax.vals, g_argmax.idxs);
    return nblocks;
}

void argmax_write(int* dst, const float* x, int n) {
    hip_init();
    check(dst != nullptr && n > 0, "argmax_write empty");
    const int nblocks = launch_argmax_partial(x, n);
    VESPER_HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(argmax_final_kernel, dim3(1), dim3(kGemvWorkgroup), 0, hip_stream(),
                       g_argmax.vals, g_argmax.idxs, nblocks, dst);
    VESPER_HIP_CHECK(hipGetLastError());
}

int argmax(const float* x, int n) {
    argmax_write(g_argmax.out, x, n);
    int best = 0;
    hip_copy_d2h(&best, g_argmax.out, sizeof(int));
    return best;
}

void seed_generated(int* ids, int* index, const int* token) {
    hip_init();
    check(ids != nullptr && index != nullptr && token != nullptr, "seed_generated null");
    hipLaunchKernelGGL(seed_generated_kernel, dim3(1), dim3(1), 0, hip_stream(), ids, index, token);
    VESPER_HIP_CHECK(hipGetLastError());
}

void commit_generated(int* ids, int* index, const int* token, int* pos) {
    hip_init();
    check(ids != nullptr && index != nullptr && token != nullptr && pos != nullptr,
          "commit_generated null");
    hipLaunchKernelGGL(commit_generated_kernel, dim3(1), dim3(1), 0, hip_stream(), ids, index, token,
                       pos);
    VESPER_HIP_CHECK(hipGetLastError());
}

void argmax_write_commit(int* ids, int* index, int* token, int* pos, const float* x, int n) {
    hip_init();
    check(ids != nullptr && index != nullptr && token != nullptr && pos != nullptr && n > 0,
          "argmax_write_commit empty");
    const int nblocks = launch_argmax_partial(x, n);
    VESPER_HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(argmax_final_commit_kernel, dim3(1), dim3(kGemvWorkgroup), 0, hip_stream(),
                       g_argmax.vals, g_argmax.idxs, nblocks, token, ids, index, pos);
    VESPER_HIP_CHECK(hipGetLastError());
}

}  // namespace rdna4
}  // namespace vesper

