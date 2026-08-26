#include "vesper/rdna4.h"

#include "vesper/types.h"
#include "vesper/weight.h"

namespace vesper {
namespace rdna4 {
namespace {

[[noreturn]] void no_hip() {
    fail("HIP is not built; configure with -DVESPER_USE_HIP=ON");
}

}  // namespace

void rmsnorm(float*, const float*, const float*, int, float) { no_hip(); }
void rmsnorm_rows(float*, const float*, int, int, float) { no_hip(); }
void split_gated_q(float*, float*, const float*, int, int) { no_hip(); }
void split_gated_q_norm(float*, float*, const float*, const float*, int, int, float) { no_hip(); }
void rmsnorm_silu_mul(float*, const float*, const float*, int, int, float) { no_hip(); }
void gdn_conv_split(float*, float*, float*, float*, const float*, const float*, int, int, int) {
    no_hip();
}
void tile_heads(float*, const float*, int, int, int) { no_hip(); }
void attn_decode(float*, const float*, const float*, const float*, const float*, int, int, int,
                 int) {
    no_hip();
}
void gemv_swiglu(float*, const WeightMatrix&, const WeightMatrix&, const float*) { no_hip(); }
void add_rmsnorm(float*, float*, const float*, int, float) { no_hip(); }
void copy_rmsnorm(float*, float*, const float*, int, float) { no_hip(); }
void silu_mul(float*, const float*, int) { no_hip(); }
void gdn_gates(float*, float*, const float*, const float*, const float*, int) { no_hip(); }
void split_qkv(float*, float*, float*, const float*, int, int) { no_hip(); }
void rope_neox(float*, float*, int, int, int, int, int, float) { no_hip(); }
void rope_neox_k_norm(float*, float*, const float*, int, int, int, int, int, float, float) {
    no_hip();
}
void gemv(float*, const float*, const float*, int, int, const float*) { no_hip(); }
void gemv_q8(float*, const std::byte*, const float*, int, int, const float*) { no_hip(); }
void gemv_q4k(float*, const std::byte*, const float*, int, int, const float*) { no_hip(); }
void gemv_q5k(float*, const std::byte*, const float*, int, int, const float*) { no_hip(); }
void gemv_q6k(float*, const std::byte*, const float*, int, int, const float*) { no_hip(); }
void gemv3(float*, const WeightMatrix&, float*, const WeightMatrix&, float*, const WeightMatrix&,
           const float*) {
    no_hip();
}
void gemv4(float*, const WeightMatrix&, float*, const WeightMatrix&, float*, const WeightMatrix&,
           float*, const WeightMatrix&, const float*) {
    no_hip();
}
void tile_l2_scale(float*, const float*, int, int, int, float, float) { no_hip(); }
void tile_l2_pair(float*, const float*, float*, const float*, int, int, int, float, float, float) {
    no_hip();
}
void swiglu(float*, const float*, const float*, int) { no_hip(); }
void softmax_inplace(float*, int) { no_hip(); }
void sigmoid_inplace(float*, int) { no_hip(); }
void silu_inplace(float*, int) { no_hip(); }
void softplus_inplace(float*, int) { no_hip(); }
void exp_inplace(float*, int) { no_hip(); }
void mul_inplace(float*, const float*, int) { no_hip(); }
void scale_inplace(float*, float, int) { no_hip(); }
void l2_normalize_rows(float*, int, int, float) { no_hip(); }
void embed_row(float*, const float*, int, int) { no_hip(); }
void attn_scores(float*, const float*, const float*, int, int, int, int) { no_hip(); }
void attn_mix(float*, const float*, const float*, int, int, int, int) { no_hip(); }
void add_inplace(float*, const float*, int) { no_hip(); }
void copy(float*, const float*, int) { no_hip(); }
void gdn_conv_update(float*, float*, const float*, const float*, int, int) { no_hip(); }
void gdn_delta_rule(float*, float*, const float*, const float*, const float*, const float*,
                    const float*, int, int) {
    no_hip();
}
int argmax(const float*, int) {
    no_hip();
}

}  // namespace rdna4
}  // namespace vesper
