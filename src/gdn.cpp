#include "vesper/gdn.h"

#include "vesper/kernels.h"
#include "vesper/rdna4.h"
#include "vesper/types.h"

#include <cmath>

namespace vesper {
namespace {

void gdn_conv_update_cpu(float* y, float* state, const float* x, const float* weight,
                         int conv_dim, int kernel) {
    check(kernel >= 2, "gdn conv kernel must be >= 2");
    const int hist = kernel - 1;
    for (int d = 0; d < conv_dim; ++d) {
        float acc = 0.0f;
        for (int t = 0; t < hist; ++t) {
            acc += weight[d * kernel + t] * state[d * hist + t];
        }
        acc += weight[d * kernel + hist] * x[d];
        y[d] = acc / (1.0f + std::exp(-acc));
        for (int t = 0; t < hist - 1; ++t) {
            state[d * hist + t] = state[d * hist + t + 1];
        }
        state[d * hist + (hist - 1)] = x[d];
    }
}

void gdn_delta_rule_cpu(float* y, float* rec, const float* q, const float* k, const float* v,
                        const float* decay, const float* beta, int n_heads, int dim) {
    for (int h = 0; h < n_heads; ++h) {
        float* S = rec + static_cast<std::size_t>(h) * dim * dim;
        const float* qh = q + h * dim;
        const float* kh = k + h * dim;
        const float* vh = v + h * dim;
        const float g = decay[h];
        const float b = beta[h];
        for (int j = 0; j < dim; ++j) {
            float* col = S + j * dim;
            float retrieved = 0.0f;
            for (int i = 0; i < dim; ++i) {
                retrieved += col[i] * kh[i];
            }
            const float delta = b * (vh[j] - g * retrieved);
            float acc = 0.0f;
            for (int i = 0; i < dim; ++i) {
                const float s = g * col[i] + kh[i] * delta;
                col[i] = s;
                acc += s * qh[i];
            }
            y[h * dim + j] = acc;
        }
    }
}

}  // namespace

void gdn_scratch_init(GdnScratch* scratch, const ModelConfig& cfg, Device device) {
    check(scratch != nullptr, "gdn scratch is null");
    scratch->qkv = Buffer(static_cast<std::size_t>(cfg.gdn_qkv_dim()), device);
    scratch->z = Buffer(static_cast<std::size_t>(cfg.gdn_value_dim()), device);
    scratch->beta = Buffer(static_cast<std::size_t>(cfg.gdn_v_heads), device);
    scratch->alpha = Buffer(static_cast<std::size_t>(cfg.gdn_v_heads), device);
    scratch->conv_y = Buffer(static_cast<std::size_t>(cfg.gdn_qkv_dim()), device);
    scratch->q = Buffer(static_cast<std::size_t>(cfg.gdn_key_dim()), device);
    scratch->k = Buffer(static_cast<std::size_t>(cfg.gdn_key_dim()), device);
    scratch->v = Buffer(static_cast<std::size_t>(cfg.gdn_value_dim()), device);
    scratch->q_rep = Buffer(static_cast<std::size_t>(cfg.gdn_v_heads * cfg.gdn_head_dim), device);
    scratch->k_rep = Buffer(static_cast<std::size_t>(cfg.gdn_v_heads * cfg.gdn_head_dim), device);
    scratch->decay = Buffer(static_cast<std::size_t>(cfg.gdn_v_heads), device);
    scratch->y = Buffer(static_cast<std::size_t>(cfg.gdn_value_dim()), device);
}

void gdn_conv_update(Device device, float* y, float* state, const float* x, const float* weight,
                     int conv_dim, int kernel) {
    switch (device) {
        case Device::CPU:
            gdn_conv_update_cpu(y, state, x, weight, conv_dim, kernel);
            return;
        case Device::HIP:
            rdna4::gdn_conv_update(y, state, x, weight, conv_dim, kernel);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void gdn_conv_split(Device device, float* q, float* k, float* v, float* conv_y, float* state,
                    const float* x, const float* weight, int key_dim, int value_dim, int kernel) {
    switch (device) {
        case Device::CPU:
            gdn_conv_update_cpu(conv_y, state, x, weight, 2 * key_dim + value_dim, kernel);
            split_qkv(q, k, v, conv_y, key_dim, value_dim);
            return;
        case Device::HIP:
            rdna4::gdn_conv_split(q, k, v, state, x, weight, key_dim, value_dim, kernel);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void gdn_delta_rule(Device device, float* y, float* rec, const float* q, const float* k,
                    const float* v, const float* decay, const float* beta, int n_heads, int dim) {
    switch (device) {
        case Device::CPU:
            gdn_delta_rule_cpu(y, rec, q, k, v, decay, beta, n_heads, dim);
            return;
        case Device::HIP:
            rdna4::gdn_delta_rule(y, rec, q, k, v, decay, beta, n_heads, dim);
            return;
    }
    throw std::logic_error("unhandled Device");
}

void gdn_layer(Device device, float* y, const float* x, const LayerWeights& layer,
               const ModelConfig& cfg, float* rec, float* conv, GdnScratch* scratch) {
    check(scratch != nullptr, "gdn_layer needs scratch");
    const int nk = cfg.gdn_qk_heads;
    const int nv = cfg.gdn_v_heads;
    const int dim = cfg.gdn_head_dim;
    const int key_dim = cfg.gdn_key_dim();
    const int value_dim = cfg.gdn_value_dim();

    gemv4(device, scratch->qkv.data(), layer.qkv_proj, scratch->z.data(), layer.z_proj,
          scratch->beta.data(), layer.beta_proj, scratch->alpha.data(), layer.alpha_proj, x);

    gdn_conv_split(device, scratch->q.data(), scratch->k.data(), scratch->v.data(),
                   scratch->conv_y.data(), conv, scratch->qkv.data(), layer.conv1d.data(), key_dim,
                   value_dim, cfg.gdn_conv_kernel);

    tile_l2_pair(device, scratch->q_rep.data(), scratch->q.data(), scratch->k_rep.data(),
                 scratch->k.data(), nv, nk, dim, 1e-6f, 1.0f / std::sqrt(static_cast<float>(dim)),
                 1.0f);

    gdn_gates(device, scratch->decay.data(), scratch->beta.data(), scratch->alpha.data(),
              layer.ssm_dt.data(), layer.ssm_a.data(), nv);

    gdn_delta_rule(device, scratch->y.data(), rec, scratch->q_rep.data(), scratch->k_rep.data(),
                   scratch->v.data(), scratch->decay.data(), scratch->beta.data(), nv, dim);

    rmsnorm_silu_mul(device, scratch->y.data(), scratch->z.data(), layer.ssm_norm.data(), nv, dim,
                     cfg.rms_eps);
    gemv(device, y, layer.ssm_out, scratch->y.data());
}

}  // namespace vesper
