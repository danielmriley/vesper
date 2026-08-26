#include "vesper/rdna4.h"

#include "vesper/types.h"

namespace vesper {
namespace rdna4 {
namespace {

[[noreturn]] void no_hip() {
    fail("HIP is not built; configure with -DVESPER_USE_HIP=ON");
}

}  // namespace

void rmsnorm(float*, const float*, const float*, int, float) { no_hip(); }
void rope_neox(float*, float*, int, int, int, int, int, float) { no_hip(); }
void gemv(float*, const float*, const float*, int, int) { no_hip(); }
void gemv_q8(float*, const std::byte*, const float*, int, int) { no_hip(); }
void gemv_q4k(float*, const std::byte*, const float*, int, int) { no_hip(); }
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

}  // namespace rdna4
}  // namespace vesper
