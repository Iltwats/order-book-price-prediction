// Unified high-speed streaming C++17 inference engine for Alpha Connectome.
// Compiles offline at startup via system `c++ -O3 -march=native` (available in
// the official scoring container via `build-essential` and on macOS/Linux hosts).
// Zero external dependencies beyond the C++17 standard library; portable SIMD
// via ARM NEON (macOS/ARM64) and AVX2+FMA / auto-vectorization (x86-64 EPYC).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

namespace {

static inline float fast_sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + std::exp(-x));
    float e = std::exp(x);
    return e / (1.0f + e);
}

static inline float exact_gelu(float x) {
    return 0.5f * x * (1.0f + std::erf(x * 0.7071067811865475f));
}

// Register-tiled GEMV: y[0..M) = bias[0..M) + sum_{i=0}^{N-1} x[i] * A_t[i * M + j]
// Requires M to be a multiple of 16 (or handles tail cleanly).
static inline void gemv_tiled_add(const float* __restrict A_t, const float* __restrict x,
                                  int N, int M, float* __restrict y) {
#if defined(__ARM_NEON)
    int j = 0;
    for (; j + 31 < M; j += 32) {
        float32x4_t a0 = vld1q_f32(y + j + 0),  a1 = vld1q_f32(y + j + 4);
        float32x4_t a2 = vld1q_f32(y + j + 8),  a3 = vld1q_f32(y + j + 12);
        float32x4_t a4 = vld1q_f32(y + j + 16), a5 = vld1q_f32(y + j + 20);
        float32x4_t a6 = vld1q_f32(y + j + 24), a7 = vld1q_f32(y + j + 28);
        const float* p = A_t + j;
        for (int i = 0; i < N; ++i, p += M) {
            float32x4_t vx = vdupq_n_f32(x[i]);
            a0 = vfmaq_f32(a0, vld1q_f32(p + 0), vx);
            a1 = vfmaq_f32(a1, vld1q_f32(p + 4), vx);
            a2 = vfmaq_f32(a2, vld1q_f32(p + 8), vx);
            a3 = vfmaq_f32(a3, vld1q_f32(p + 12), vx);
            a4 = vfmaq_f32(a4, vld1q_f32(p + 16), vx);
            a5 = vfmaq_f32(a5, vld1q_f32(p + 20), vx);
            a6 = vfmaq_f32(a6, vld1q_f32(p + 24), vx);
            a7 = vfmaq_f32(a7, vld1q_f32(p + 28), vx);
        }
        vst1q_f32(y + j + 0, a0);  vst1q_f32(y + j + 4, a1);
        vst1q_f32(y + j + 8, a2);  vst1q_f32(y + j + 12, a3);
        vst1q_f32(y + j + 16, a4); vst1q_f32(y + j + 20, a5);
        vst1q_f32(y + j + 24, a6); vst1q_f32(y + j + 28, a7);
    }
    for (; j + 15 < M; j += 16) {
        float32x4_t a0 = vld1q_f32(y + j + 0), a1 = vld1q_f32(y + j + 4);
        float32x4_t a2 = vld1q_f32(y + j + 8), a3 = vld1q_f32(y + j + 12);
        const float* p = A_t + j;
        for (int i = 0; i < N; ++i, p += M) {
            float32x4_t vx = vdupq_n_f32(x[i]);
            a0 = vfmaq_f32(a0, vld1q_f32(p + 0), vx);
            a1 = vfmaq_f32(a1, vld1q_f32(p + 4), vx);
            a2 = vfmaq_f32(a2, vld1q_f32(p + 8), vx);
            a3 = vfmaq_f32(a3, vld1q_f32(p + 12), vx);
        }
        vst1q_f32(y + j + 0, a0); vst1q_f32(y + j + 4, a1);
        vst1q_f32(y + j + 8, a2); vst1q_f32(y + j + 12, a3);
    }
    for (; j < M; ++j) {
        float sum = y[j];
        const float* p = A_t + j;
        for (int i = 0; i < N; ++i, p += M) sum += (*p) * x[i];
        y[j] = sum;
    }
#elif defined(__AVX2__) && defined(__FMA__)
    int j = 0;
    for (; j + 63 < M; j += 64) {
        __m256 a0 = _mm256_loadu_ps(y + j + 0),  a1 = _mm256_loadu_ps(y + j + 8);
        __m256 a2 = _mm256_loadu_ps(y + j + 16), a3 = _mm256_loadu_ps(y + j + 24);
        __m256 a4 = _mm256_loadu_ps(y + j + 32), a5 = _mm256_loadu_ps(y + j + 40);
        __m256 a6 = _mm256_loadu_ps(y + j + 48), a7 = _mm256_loadu_ps(y + j + 56);
        const float* p = A_t + j;
        for (int i = 0; i < N; ++i, p += M) {
            __m256 vx = _mm256_set1_ps(x[i]);
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 0),  vx, a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 8),  vx, a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 16), vx, a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 24), vx, a3);
            a4 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 32), vx, a4);
            a5 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 40), vx, a5);
            a6 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 48), vx, a6);
            a7 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 56), vx, a7);
        }
        _mm256_storeu_ps(y + j + 0, a0);  _mm256_storeu_ps(y + j + 8, a1);
        _mm256_storeu_ps(y + j + 16, a2); _mm256_storeu_ps(y + j + 24, a3);
        _mm256_storeu_ps(y + j + 32, a4); _mm256_storeu_ps(y + j + 40, a5);
        _mm256_storeu_ps(y + j + 48, a6); _mm256_storeu_ps(y + j + 56, a7);
    }
    for (; j + 15 < M; j += 16) {
        __m256 a0 = _mm256_loadu_ps(y + j + 0), a1 = _mm256_loadu_ps(y + j + 8);
        const float* p = A_t + j;
        for (int i = 0; i < N; ++i, p += M) {
            __m256 vx = _mm256_set1_ps(x[i]);
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 0), vx, a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 8), vx, a1);
        }
        _mm256_storeu_ps(y + j + 0, a0); _mm256_storeu_ps(y + j + 8, a1);
    }
    for (; j < M; ++j) {
        float sum = y[j];
        const float* p = A_t + j;
        for (int i = 0; i < N; ++i, p += M) sum += (*p) * x[i];
        y[j] = sum;
    }
#else
    for (int i = 0; i < N; ++i) {
        float xi = x[i];
        const float* row = A_t + i * M;
        for (int j = 0; j < M; ++j) y[j] += row[j] * xi;
    }
#endif
}

static inline void gemv_set(const float* __restrict A_t, const float* __restrict x,
                            int N, int M, const float* __restrict bias, float* __restrict y) {
    if (bias) std::memcpy(y, bias, M * sizeof(float));
    else std::memset(y, 0, M * sizeof(float));
    gemv_tiled_add(A_t, x, N, M, y);
}

// Insertion sort for N=11 quote prices + aligned volumes (exact stable ascending sort matching torch.sort!)
static inline void sort11(const float* p_in, const float* v_in, float* p_out, float* v_out) {
    int idx[11];
    for (int i = 0; i < 11; ++i) idx[i] = i;
    for (int i = 1; i < 11; ++i) {
        int key = idx[i];
        float kp = p_in[key];
        int j = i - 1;
        while (j >= 0 && p_in[idx[j]] > kp) {
            idx[j + 1] = idx[j];
            --j;
        }
        idx[j + 1] = key;
    }
    for (int i = 0; i < 11; ++i) {
        p_out[i] = p_in[idx[i]];
        v_out[i] = v_in[idx[i]];
    }
}

// ============================================================================
// Feature Extractor State (builds BOTH 169-ch engineered LOB features & 34-ch snap40 base)
// ============================================================================
struct FeatureState {
    float dyn_ring[32][16];
    int step = 0;

    void reset() {
        std::memset(dyn_ring, 0, sizeof(dyn_ring));
        step = 0;
    }

    // Fills feat169 [169] AND snap34 [34] (the 27 book/trade stats + 7 best quotes used by snap40!)
    void update(const float* __restrict x, float* __restrict feat169, float* __restrict snap34) {
        float i0_bp[11], i0_bv[11], i0_ap[11], i0_av[11];
        float i1_bp[11], i1_bv[11], i1_ap[11], i1_av[11];
        sort11(x + 0,  x + 22, i0_bp, i0_bv);
        sort11(x + 11, x + 33, i0_ap, i0_av);
        sort11(x + 52, x + 74, i1_bp, i1_bv);
        sort11(x + 63, x + 85, i1_ap, i1_av);

        // Microstructure scalars matching build_lob_engineered_features_torch bit-for-bit
        float i0_mid = 0.5f * (i0_bp[10] + i0_ap[0]);
        float i0_spr = i0_ap[0] - i0_bp[10];
        float i0_wmid = (i0_bp[10] * std::fabs(i0_av[0]) + i0_ap[0] * std::fabs(i0_bv[10])) /
                        (std::fabs(i0_bv[10]) + std::fabs(i0_av[0]) + 1e-4f);
        float i0_bv_abs = 0, i0_av_abs = 0, i0_bv_s = 0, i0_av_s = 0;
        for (int i = 0; i < 11; ++i) {
            i0_bv_abs += std::fabs(i0_bv[i]); i0_av_abs += std::fabs(i0_av[i]);
            i0_bv_s += i0_bv[i];              i0_av_s += i0_av[i];
        }
        float i0_imb_all = (i0_bv_s - i0_av_s) / (i0_bv_abs + i0_av_abs + 1e-4f);
        float i0_bv3_s = i0_bv[8] + i0_bv[9] + i0_bv[10];
        float i0_av3_s = i0_av[0] + i0_av[1] + i0_av[2];
        float i0_bv3_a = std::fabs(i0_bv[8]) + std::fabs(i0_bv[9]) + std::fabs(i0_bv[10]);
        float i0_av3_a = std::fabs(i0_av[0]) + std::fabs(i0_av[1]) + std::fabs(i0_av[2]);
        float i0_imb_top3 = (i0_bv3_s - i0_av3_s) / (i0_bv3_a + i0_av3_a + 1e-4f);

        float i1_mid = 0.5f * (i1_bp[10] + i1_ap[0]);
        float i1_spr = i1_ap[0] - i1_bp[10];
        float i1_bv_abs = 0, i1_av_abs = 0, i1_bv_s = 0, i1_av_s = 0;
        for (int i = 0; i < 11; ++i) {
            i1_bv_abs += std::fabs(i1_bv[i]); i1_av_abs += std::fabs(i1_av[i]);
            i1_bv_s += i1_bv[i];              i1_av_s += i1_av[i];
        }
        float i1_imb_all = (i1_bv_s - i1_av_s) / (i1_bv_abs + i1_av_abs + 1e-4f);
        float i1_bv3_s = i1_bv[8] + i1_bv[9] + i1_bv[10];
        float i1_av3_s = i1_av[0] + i1_av[1] + i1_av[2];
        float i1_bv3_a = std::fabs(i1_bv[8]) + std::fabs(i1_bv[9]) + std::fabs(i1_bv[10]);
        float i1_av3_a = std::fabs(i1_av[0]) + std::fabs(i1_av[1]) + std::fabs(i1_av[2]);
        float i1_imb_top3 = (i1_bv3_s - i1_av3_s) / (i1_bv3_a + i1_av3_a + 1e-4f);

        // Dynamic 16-channel vector
        float cur_dyn[16] = {
            i0_mid, i0_wmid, i0_spr, i0_imb_all, i0_imb_top3,
            x[44], x[45], x[46], x[47],
            i1_mid, i1_spr, i1_imb_all,
            x[96], x[97], x[98], x[99]
        };
        std::memcpy(dyn_ring[step & 31], cur_dyn, sizeof(cur_dyn));
        const float* lag1  = dyn_ring[(step >= 1  ? step - 1  : 0) & 31];
        const float* lag4  = dyn_ring[(step >= 4  ? step - 4  : 0) & 31];
        const float* lag16 = dyn_ring[(step >= 16 ? step - 16 : 0) & 31];

        if (feat169) {
            std::memcpy(feat169 + 0,   i0_bp, 11 * sizeof(float));
            std::memcpy(feat169 + 11,  i0_ap, 11 * sizeof(float));
            std::memcpy(feat169 + 22,  i0_bv, 11 * sizeof(float));
            std::memcpy(feat169 + 33,  i0_av, 11 * sizeof(float));
            std::memcpy(feat169 + 44,  x + 44, 8 * sizeof(float));
            std::memcpy(feat169 + 52,  i1_bp, 11 * sizeof(float));
            std::memcpy(feat169 + 63,  i1_ap, 11 * sizeof(float));
            std::memcpy(feat169 + 74,  i1_bv, 11 * sizeof(float));
            std::memcpy(feat169 + 85,  i1_av, 11 * sizeof(float));
            std::memcpy(feat169 + 96,  x + 96, 8 * sizeof(float));
            std::memcpy(feat169 + 104, x + 104, 8 * sizeof(float));
            float micro9[9] = {i0_mid, i0_wmid, i0_spr, i0_imb_all, i0_imb_top3,
                               i1_mid, i1_spr, i1_imb_all, i1_imb_top3};
            std::memcpy(feat169 + 112, micro9, 9 * sizeof(float));
            for (int d = 0; d < 16; ++d) {
                feat169[121 + d] = cur_dyn[d] - lag1[d];
                feat169[137 + d] = cur_dyn[d] - lag4[d];
                feat169[153 + d] = cur_dyn[d] - lag16[d];
            }
        }

        if (snap34) {
            // Exact 27 snap book/trade stats + 7 best quotes matching snap40_bestq!
            float bp_m = 0, ap_m = 0, b1_m = 0, a1_m = 0;
            float bv_max = x[22], av_max = x[33];
            for (int i = 0; i < 11; ++i) {
                bp_m += i0_bp[i]; ap_m += i0_ap[i];
                b1_m += i1_bp[i]; a1_m += i1_ap[i];
                if (x[22 + i] > bv_max) bv_max = x[22 + i];
                if (x[33 + i] > av_max) av_max = x[33 + i];
            }
            bp_m /= 11.0f; ap_m /= 11.0f; b1_m /= 11.0f; a1_m /= 11.0f;
            float bp_v = 0, ap_v = 0;
            for (int i = 0; i < 11; ++i) {
                float db = i0_bp[i] - bp_m, da = i0_ap[i] - ap_m;
                bp_v += db * db; ap_v += da * da;
            }
            float bp_sd = std::sqrt(bp_v / 11.0f), ap_sd = std::sqrt(ap_v / 11.0f);
            float i0_imb_s = (i0_bv_s - i0_av_s) / (i0_bv_abs + i0_av_abs + 1e-5f);
            float i1_imb_s = (i1_bv_s - i1_av_s) / (i1_bv_abs + i1_av_abs + 1e-5f);
            float i1_dp_m = 0.25f * (x[96] + x[97] + x[98] + x[99]);
            float i1_dv_m = 0.25f * (x[100] + x[101] + x[102] + x[103]);

            snap34[0] = bp_m;  snap34[1] = bp_sd; snap34[2] = i0_bp[10]; snap34[3] = i0_bp[0];
            snap34[4] = ap_m;  snap34[5] = ap_sd; snap34[6] = i0_ap[10]; snap34[7] = i0_ap[0];
            snap34[8] = ap_m - bp_m;
            snap34[9] = i0_bv_s / 11.0f; snap34[10] = bv_max;
            snap34[11] = i0_av_s / 11.0f; snap34[12] = av_max;
            snap34[13] = i0_imb_s;
            std::memcpy(snap34 + 14, x + 44, 8 * sizeof(float));
            snap34[22] = b1_m; snap34[23] = a1_m; snap34[24] = i1_imb_s;
            snap34[25] = i1_dp_m; snap34[26] = i1_dv_m;
            // 7 best quotes
            snap34[27] = i0_bp[10]; snap34[28] = i0_bp[9]; snap34[29] = i0_bp[5];
            snap34[30] = i0_ap[0];  snap34[31] = i0_ap[1]; snap34[32] = i0_ap[5];
            snap34[33] = i0_ap[0] - i0_bp[10];
        }
        ++step;
    }
};

// ============================================================================
// Model 1: ONNX-compatible 2-layer GRU128 (e.g., correlation1024, [z, r, h], LBR=1)
// ============================================================================
struct OnnxGRU128 {
    alignas(64) float w0_t[112 * 384];
    alignas(64) float r0_t[128 * 384];
    alignas(64) float b0[768];
    alignas(64) float w1_t[128 * 384];
    alignas(64) float r1_t[128 * 384];
    alignas(64) float b1[768];
    alignas(64) float hw[128 * 2];
    alignas(64) float hb[2];
    alignas(64) float h0[128];
    alignas(64) float h1[128];

    void reset() {
        std::memset(h0, 0, sizeof(h0));
        std::memset(h1, 0, sizeof(h1));
    }

    static inline void step_layer(const float* W_t, const float* R_t, const float* B,
                                  const float* inp, int nin, float* h) {
        alignas(64) float xp[384], rp[384];
        gemv_set(W_t, inp, nin, 384, nullptr, xp);
        gemv_set(R_t, h, 128, 384, nullptr, rp);
        for (int j = 0; j < 128; ++j) {
            float z = fast_sigmoid((xp[j] + rp[j]) + (B[j] + B[384 + j]));
            float r = fast_sigmoid((xp[128 + j] + rp[128 + j]) + (B[128 + j] + B[512 + j]));
            float c = std::tanh((xp[256 + j] + B[256 + j]) + r * (rp[256 + j] + B[640 + j]));
            h[j] = (1.0f - z) * c + z * h[j];
        }
    }

    void step(const float* x112, float* out2) {
        step_layer(w0_t, r0_t, b0, x112, 112, h0);
        step_layer(w1_t, r1_t, b1, h0, 128, h1);
        float s0 = hb[0], s1 = hb[1];
        for (int j = 0; j < 128; ++j) {
            s0 += h1[j] * hw[2 * j];
            s1 += h1[j] * hw[2 * j + 1];
        }
        out2[0] = s0; out2[1] = s1;
    }
};

// ============================================================================
// Model 2: PyTorch ScratchSortedGRU (hidden=128, PyTorch [r, z, n] gate order!)
// ============================================================================
struct TorchSortedGRU128 {
    alignas(64) float in_mu[169], in_sd[169];
    alignas(64) float stem_w_t[169 * 128], stem_b[128];
    alignas(64) float w0_t[128 * 384], r0_t[128 * 384], bi0[384], bh0[384];
    alignas(64) float w1_t[128 * 384], r1_t[128 * 384], bi1[384], bh1[384];
    alignas(64) float h1_w_t[384 * 128], h1_b[128];
    alignas(64) float h2_w_t[128 * 2], h2_b[2];
    alignas(64) float h0[128], h1[128];

    void reset() {
        std::memset(h0, 0, sizeof(h0));
        std::memset(h1, 0, sizeof(h1));
    }

    static inline void step_torch_gru(const float* W_t, const float* R_t,
                                      const float* bi, const float* bh,
                                      const float* inp, float* h) {
        alignas(64) float xp[384], rp[384];
        gemv_set(W_t, inp, 128, 384, bi, xp);
        gemv_set(R_t, h,   128, 384, bh, rp);
        // PyTorch gate order is [r, z, n]: h' = (1 - z) * n + z * h
        for (int j = 0; j < 128; ++j) {
            float r = fast_sigmoid(xp[j] + rp[j]);
            float z = fast_sigmoid(xp[128 + j] + rp[128 + j]);
            float n = std::tanh(xp[256 + j] + r * rp[256 + j]);
            h[j] = (1.0f - z) * n + z * h[j];
        }
    }

    void step(const float* feat169, float* out2) {
        alignas(64) float z[169], s[128], g1_in[128], cat384[384], mid[128];
        for (int i = 0; i < 169; ++i) z[i] = (feat169[i] - in_mu[i]) / in_sd[i];
        gemv_set(stem_w_t, z, 169, 128, stem_b, s);
        for (int j = 0; j < 128; ++j) s[j] = exact_gelu(s[j]);
        step_torch_gru(w0_t, r0_t, bi0, bh0, s, h0);
        for (int j = 0; j < 128; ++j) g1_in[j] = h0[j] + s[j];
        step_torch_gru(w1_t, r1_t, bi1, bh1, g1_in, h1);
        std::memcpy(cat384 + 0,   s,  128 * sizeof(float));
        std::memcpy(cat384 + 128, h0, 128 * sizeof(float));
        std::memcpy(cat384 + 256, h1, 128 * sizeof(float));
        gemv_set(h1_w_t, cat384, 384, 128, h1_b, mid);
        for (int j = 0; j < 128; ++j) mid[j] = exact_gelu(mid[j]);
        float o0 = h2_b[0], o1 = h2_b[1];
        for (int j = 0; j < 128; ++j) {
            o0 += mid[j] * h2_w_t[2 * j];
            o1 += mid[j] * h2_w_t[2 * j + 1];
        }
        out2[0] = o0; out2[1] = o1;
    }
};

// ============================================================================
// Model 3: MultiScaleGatedTCN (channels=80, 10 dilated blocks, RF=2047)
// ============================================================================
struct StreamingTCN80 {
    static constexpr int CH = 80;
    static constexpr int NL = 10;
    static constexpr int BUF = 2048;
    alignas(64) float in_mu[169], in_sd[169];
    alignas(64) float stem_w_t[169 * CH], stem_b[CH];
    alignas(64) float kw_t[NL][3][CH * 2 * CH];
    alignas(64) float kb[NL][2 * CH];
    alignas(64) float pw_t[NL][CH * CH];
    alignas(64) float pb[NL][CH];
    alignas(64) float h1_w_t[3 * CH * CH], h1_b[CH];
    alignas(64) float h2_w_t[CH * 2], h2_b[2];
    // Per-layer compact ring buffers: max lag per layer is 2 * DILS[l], so total working set is 128 KB (fits in L1/L2 cache!) instead of 6.5 MB!
    static constexpr int RINGSZ[NL] = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
    static constexpr int ROFF[NL + 1] = {0, 4, 12, 28, 60, 124, 252, 508, 1020, 2044, 4092};
    alignas(64) float ring[4092][CH];  // 4092 * 80 * 4 = 1.30 MB total instead of 6.55 MB!
    int step_idx = 0;

    void reset() {
        std::memset(ring, 0, sizeof(ring));
        step_idx = 0;
    }

    void step(const float* feat169, float* out2) {
        static constexpr int DILS[NL] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
        alignas(64) float z[169], h0[CH], cur[CH], h_mid[CH], gate[2 * CH], gated[CH], proj[CH], cat[3 * CH], head_h[CH];
        for (int i = 0; i < 169; ++i) z[i] = (feat169[i] - in_mu[i]) / in_sd[i];
        gemv_set(stem_w_t, z, 169, CH, stem_b, h0);
        for (int j = 0; j < CH; ++j) cur[j] = h0[j] = exact_gelu(h0[j]);

        const int s = step_idx;
        for (int l = 0; l < NL; ++l) {
            const int mask = RINGSZ[l] - 1;
            const int base_off = ROFF[l];
            std::memcpy(ring[base_off + (s & mask)], cur, CH * sizeof(float));
            const int d = DILS[l];
            gemv_set(kw_t[l][2], cur, CH, 2 * CH, kb[l], gate);
            if (s >= d)     gemv_tiled_add(kw_t[l][1], ring[base_off + ((s - d) & mask)],     CH, 2 * CH, gate);
            if (s >= 2 * d) gemv_tiled_add(kw_t[l][0], ring[base_off + ((s - 2 * d) & mask)], CH, 2 * CH, gate);
            for (int j = 0; j < CH; ++j) {
                float a = gate[j], b = gate[CH + j];
                float silu_a = a * fast_sigmoid(a);
                gated[j] = silu_a * fast_sigmoid(b);
            }
            gemv_set(pw_t[l], gated, CH, CH, pb[l], proj);
            for (int j = 0; j < CH; ++j) cur[j] += proj[j];
            if (l == 4) std::memcpy(h_mid, cur, CH * sizeof(float));
        }
        step_idx = s + 1;
        // Warmup optimization: a 10-layer dilated conv with kernel=3 has finite causal receptive field = 2,047 steps.
        // Wait: step 99 (first required prediction!) IS within 2,047 steps of step 0!
        // So steps 0..98 DO enter the ring buffer for steps 99..2145!
        // What DOES NOT affect future steps is the final 3-block readout head (h1_w_t [240->80] + h2_w_t [80->2]) on steps 0..98!
        if (!out2) return;
        std::memcpy(cat + 0,      h0,    CH * sizeof(float));
        std::memcpy(cat + CH,     h_mid, CH * sizeof(float));
        std::memcpy(cat + 2 * CH, cur,   CH * sizeof(float));
        gemv_set(h1_w_t, cat, 3 * CH, CH, h1_b, head_h);
        for (int j = 0; j < CH; ++j) head_h[j] = exact_gelu(head_h[j]);
        float o0 = h2_b[0], o1 = h2_b[1];
        for (int j = 0; j < CH; ++j) {
            o0 += head_h[j] * h2_w_t[2 * j];
            o1 += head_h[j] * h2_w_t[2 * j + 1];
        }
        out2[0] = o0; out2[1] = o1;
    }
};

} // namespace

extern "C" {

struct EnsembleContext {
    FeatureState feat_state;
    int n_onnx_gru = 0;
    std::unique_ptr<OnnxGRU128[]> onnx_grus;
    bool has_sorted_gru = false;
    std::unique_ptr<TorchSortedGRU128> sorted_gru;
    bool has_tcn80 = false;
    std::unique_ptr<StreamingTCN80> tcn80;

    // Meta-calibrator: takes [q_models..., derived_q_feats..., snap34] -> residual MLP -> final 2 outputs!
    float blend_weights[8]{};  // per-model blend weights for base q
    int cal_in_dim = 0;
    int cal_hidden = 0;
    int cal_seeds = 0;
    std::unique_ptr<float[]> cal_mu;
    std::unique_ptr<float[]> cal_sd;
    std::unique_ptr<float[]> cal_w1_t; // [cal_seeds, cal_in_dim * cal_hidden]
    std::unique_ptr<float[]> cal_b1;   // [cal_seeds, cal_hidden]
    std::unique_ptr<float[]> cal_w2_t; // [cal_seeds, cal_hidden * 2]
    std::unique_ptr<float[]> cal_b2;   // [cal_seeds, 2]
};

void* ens_create(int n_onnx_gru, int has_sorted_gru, int has_tcn80) {
    auto* ctx = new EnsembleContext();
    ctx->n_onnx_gru = n_onnx_gru;
    if (n_onnx_gru > 0) ctx->onnx_grus = std::make_unique<OnnxGRU128[]>(n_onnx_gru);
    ctx->has_sorted_gru = (has_sorted_gru != 0);
    if (ctx->has_sorted_gru) ctx->sorted_gru = std::make_unique<TorchSortedGRU128>();
    ctx->has_tcn80 = (has_tcn80 != 0);
    if (ctx->has_tcn80) ctx->tcn80 = std::make_unique<StreamingTCN80>();
    return ctx;
}

void ens_load_onnx_gru(void* p, int idx, const float* w0, const float* r0, const float* b0,
                       const float* w1, const float* r1, const float* b1,
                       const float* hw, const float* hb, float blend_w) {
    auto* ctx = static_cast<EnsembleContext*>(p);
    auto& g = ctx->onnx_grus[idx];
    for (int i = 0; i < 112; ++i) for (int j = 0; j < 384; ++j) g.w0_t[i * 384 + j] = w0[j * 112 + i];
    for (int i = 0; i < 128; ++i) for (int j = 0; j < 384; ++j) g.r0_t[i * 384 + j] = r0[j * 128 + i];
    std::memcpy(g.b0, b0, 768 * sizeof(float));
    for (int i = 0; i < 128; ++i) for (int j = 0; j < 384; ++j) g.w1_t[i * 384 + j] = w1[j * 128 + i];
    for (int i = 0; i < 128; ++i) for (int j = 0; j < 384; ++j) g.r1_t[i * 384 + j] = r1[j * 128 + i];
    std::memcpy(g.b1, b1, 768 * sizeof(float));
    std::memcpy(g.hw, hw, 256 * sizeof(float));
    std::memcpy(g.hb, hb, 2 * sizeof(float));
    g.reset();
    ctx->blend_weights[idx] = blend_w;
}

void ens_load_sorted_gru(void* p, const float* in_mu, const float* in_sd,
                         const float* stem_w, const float* stem_b,
                         const float* w0, const float* r0, const float* bi0, const float* bh0,
                         const float* w1, const float* r1, const float* bi1, const float* bh1,
                         const float* h1_w, const float* h1_b, const float* h2_w, const float* h2_b,
                         float blend_w) {
    auto* ctx = static_cast<EnsembleContext*>(p);
    auto& g = *ctx->sorted_gru;
    std::memcpy(g.in_mu, in_mu, 169 * sizeof(float));
    std::memcpy(g.in_sd, in_sd, 169 * sizeof(float));
    for (int i = 0; i < 169; ++i) for (int j = 0; j < 128; ++j) g.stem_w_t[i * 128 + j] = stem_w[j * 169 + i];
    std::memcpy(g.stem_b, stem_b, 128 * sizeof(float));
    for (int i = 0; i < 128; ++i) for (int j = 0; j < 384; ++j) {
        g.w0_t[i * 384 + j] = w0[j * 128 + i];
        g.r0_t[i * 384 + j] = r0[j * 128 + i];
        g.w1_t[i * 384 + j] = w1[j * 128 + i];
        g.r1_t[i * 384 + j] = r1[j * 128 + i];
    }
    std::memcpy(g.bi0, bi0, 384 * sizeof(float)); std::memcpy(g.bh0, bh0, 384 * sizeof(float));
    std::memcpy(g.bi1, bi1, 384 * sizeof(float)); std::memcpy(g.bh1, bh1, 384 * sizeof(float));
    for (int i = 0; i < 384; ++i) for (int j = 0; j < 128; ++j) g.h1_w_t[i * 128 + j] = h1_w[j * 384 + i];
    std::memcpy(g.h1_b, h1_b, 128 * sizeof(float));
    for (int i = 0; i < 128; ++i) for (int j = 0; j < 2; ++j) g.h2_w_t[i * 2 + j] = h2_w[j * 128 + i];
    std::memcpy(g.h2_b, h2_b, 2 * sizeof(float));
    g.reset();
    ctx->blend_weights[ctx->n_onnx_gru] = blend_w;
}

void ens_load_tcn80(void* p, const float* in_mu, const float* in_sd,
                    const float* stem_w, const float* stem_b,
                    const float* all_conv_w, const float* all_conv_b,
                    const float* all_proj_w, const float* all_proj_b,
                    const float* h1_w, const float* h1_b, const float* h2_w, const float* h2_b,
                    float blend_w) {
    auto* ctx = static_cast<EnsembleContext*>(p);
    auto& t = *ctx->tcn80;
    constexpr int CH = 80;
    std::memcpy(t.in_mu, in_mu, 169 * sizeof(float));
    std::memcpy(t.in_sd, in_sd, 169 * sizeof(float));
    // stem_w is PyTorch Conv1d [CH, 169, 1]
    for (int i = 0; i < 169; ++i) for (int j = 0; j < CH; ++j) t.stem_w_t[i * CH + j] = stem_w[j * 169 + i];
    std::memcpy(t.stem_b, stem_b, CH * sizeof(float));
    // all_conv_w: [10, 2*CH, CH, 3]
    for (int l = 0; l < 10; ++l) {
        const float* cw = all_conv_w + l * (2 * CH * CH * 3);
        for (int k = 0; k < 3; ++k)
            for (int i = 0; i < CH; ++i)
                for (int j = 0; j < 2 * CH; ++j)
                    t.kw_t[l][k][i * (2 * CH) + j] = cw[(j * CH + i) * 3 + k];
        std::memcpy(t.kb[l], all_conv_b + l * (2 * CH), 2 * CH * sizeof(float));
        const float* pw = all_proj_w + l * (CH * CH);
        for (int i = 0; i < CH; ++i) for (int j = 0; j < CH; ++j) t.pw_t[l][i * CH + j] = pw[j * CH + i];
        std::memcpy(t.pb[l], all_proj_b + l * CH, CH * sizeof(float));
    }
    for (int i = 0; i < 3 * CH; ++i) for (int j = 0; j < CH; ++j) t.h1_w_t[i * CH + j] = h1_w[j * (3 * CH) + i];
    std::memcpy(t.h1_b, h1_b, CH * sizeof(float));
    for (int i = 0; i < CH; ++i) for (int j = 0; j < 2; ++j) t.h2_w_t[i * 2 + j] = h2_w[j * CH + i];
    std::memcpy(t.h2_b, h2_b, 2 * sizeof(float));
    t.reset();
    int slot = ctx->n_onnx_gru + (ctx->has_sorted_gru ? 1 : 0);
    ctx->blend_weights[slot] = blend_w;
}

void ens_load_calibrator(void* p, int in_dim, int hidden, int n_seeds,
                         const float* mu, const float* sd,
                         const float* w1, const float* b1,
                         const float* w2, const float* b2) {
    auto* ctx = static_cast<EnsembleContext*>(p);
    ctx->cal_in_dim = in_dim;
    ctx->cal_hidden = hidden;
    ctx->cal_seeds = n_seeds;
    ctx->cal_mu = std::make_unique<float[]>(in_dim);
    ctx->cal_sd = std::make_unique<float[]>(in_dim);
    std::memcpy(ctx->cal_mu.get(), mu, in_dim * sizeof(float));
    std::memcpy(ctx->cal_sd.get(), sd, in_dim * sizeof(float));
    ctx->cal_w1_t = std::make_unique<float[]>(n_seeds * in_dim * hidden);
    ctx->cal_b1 = std::make_unique<float[]>(n_seeds * hidden);
    ctx->cal_w2_t = std::make_unique<float[]>(n_seeds * hidden * 2);
    ctx->cal_b2 = std::make_unique<float[]>(n_seeds * 2);
    for (int s = 0; s < n_seeds; ++s) {
        const float* sw1 = w1 + s * (hidden * in_dim);
        float* tw1 = ctx->cal_w1_t.get() + s * (in_dim * hidden);
        for (int i = 0; i < in_dim; ++i) for (int j = 0; j < hidden; ++j) tw1[i * hidden + j] = sw1[j * in_dim + i];
        std::memcpy(ctx->cal_b1.get() + s * hidden, b1 + s * hidden, hidden * sizeof(float));
        const float* sw2 = w2 + s * (2 * hidden);
        float* tw2 = ctx->cal_w2_t.get() + s * (hidden * 2);
        for (int i = 0; i < hidden; ++i) for (int j = 0; j < 2; ++j) tw2[i * 2 + j] = sw2[j * hidden + i];
        std::memcpy(ctx->cal_b2.get() + s * 2, b2 + s * 2, 2 * sizeof(float));
    }
}

void ens_reset(void* p) {
    auto* ctx = static_cast<EnsembleContext*>(p);
    ctx->feat_state.reset();
    for (int i = 0; i < ctx->n_onnx_gru; ++i) ctx->onnx_grus[i].reset();
    if (ctx->has_sorted_gru) ctx->sorted_gru->reset();
    if (ctx->has_tcn80) ctx->tcn80->reset();
}

int ens_step(void* p, const float* __restrict x112, int need_calibrator, float* __restrict out2) {
    auto* ctx = static_cast<EnsembleContext*>(p);
    for (int i = 0; i < 112; ++i) {
        if (!std::isfinite(x112[i])) return -1;
    }
    alignas(64) float feat169[169];
    alignas(64) float snap34[34];
    bool need_f169 = ctx->has_sorted_gru || ctx->has_tcn80;
    ctx->feat_state.update(x112, need_f169 ? feat169 : nullptr, need_calibrator ? snap34 : nullptr);

    float model_preds[8][2];
    int m_count = 0;
    for (int i = 0; i < ctx->n_onnx_gru; ++i) {
        ctx->onnx_grus[i].step(x112, model_preds[m_count++]);
    }
    if (ctx->has_sorted_gru) {
        ctx->sorted_gru->step(feat169, model_preds[m_count++]);
    }
    if (ctx->has_tcn80) {
        ctx->tcn80->step(feat169, model_preds[m_count++]);
    }

    if (!need_calibrator) {
        out2[0] = 0.0f; out2[1] = 0.0f;
        return 0;
    }

    float base0 = 0.0f, base1 = 0.0f;
    for (int m = 0; m < m_count; ++m) {
        float q0 = std::max(-2.0f, std::min(2.0f, model_preds[m][0]));
        float q1 = std::max(-2.0f, std::min(2.0f, model_preds[m][1]));
        base0 += ctx->blend_weights[m] * q0;
        base1 += ctx->blend_weights[m] * q1;
    }
    base0 = std::max(-2.0f, std::min(2.0f, base0));
    base1 = std::max(-2.0f, std::min(2.0f, base1));

    if (ctx->cal_seeds == 0) {
        out2[0] = base0; out2[1] = base1;
        return 0;
    }

    // Build calibrator input vector: [blended q_feats (6), individual model clipped preds (2*m_count if >1), snap34 (34)]
    alignas(64) float z_in[64];
    int pos = 0;
    z_in[pos++] = base0;
    z_in[pos++] = base1;
    z_in[pos++] = base0 - base1;
    z_in[pos++] = base0 + base1;
    z_in[pos++] = std::fabs(base0);
    z_in[pos++] = std::fabs(base1);
    if (m_count > 1 && ctx->cal_in_dim == 6 + 2 * m_count + 34) {
        for (int m = 0; m < m_count; ++m) {
            z_in[pos++] = std::max(-2.0f, std::min(2.0f, model_preds[m][0]));
            z_in[pos++] = std::max(-2.0f, std::min(2.0f, model_preds[m][1]));
        }
    }
    std::memcpy(z_in + pos, snap34, 34 * sizeof(float));
    pos += 34;

    for (int i = 0; i < pos; ++i) {
        float zi = (z_in[i] - ctx->cal_mu[i]) / ctx->cal_sd[i];
        z_in[i] = std::max(-8.0f, std::min(8.0f, zi));
    }

    float d0 = 0.0f, d1 = 0.0f;
    alignas(64) float h_buf[256];
    const int H = ctx->cal_hidden;
    for (int s = 0; s < ctx->cal_seeds; ++s) {
        const float* w1_t = ctx->cal_w1_t.get() + s * (pos * H);
        const float* b1   = ctx->cal_b1.get()   + s * H;
        const float* w2_t = ctx->cal_w2_t.get() + s * (H * 2);
        const float* b2   = ctx->cal_b2.get()   + s * 2;
        gemv_set(w1_t, z_in, pos, H, b1, h_buf);
        float s0 = b2[0], s1 = b2[1];
        for (int j = 0; j < H; ++j) {
            float g = exact_gelu(h_buf[j]);
            s0 += g * w2_t[2 * j];
            s1 += g * w2_t[2 * j + 1];
        }
        d0 += s0; d1 += s1;
    }
    float inv_s = 1.0f / static_cast<float>(ctx->cal_seeds);
    out2[0] = base0 + d0 * inv_s;
    out2[1] = base1 + d1 * inv_s;
    return (std::isfinite(out2[0]) && std::isfinite(out2[1])) ? 0 : -2;
}

void ens_destroy(void* p) {
    delete static_cast<EnsembleContext*>(p);
}

} // extern "C"
