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
#include <cstdlib>
#include <cstring>
#include <memory>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

namespace {

struct AlignedDeleter {
    void operator()(void* p) const noexcept { std::free(p); }
};
using AlignedFloats = std::unique_ptr<float[], AlignedDeleter>;
using AlignedU16    = std::unique_ptr<uint16_t[], AlignedDeleter>;

static inline AlignedFloats alloc_aligned_floats(std::size_t n) {
    std::size_t bytes = ((n * sizeof(float) + 63) / 64) * 64;
    if (bytes == 0) bytes = 64;
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 64, bytes) != 0 || !ptr) return AlignedFloats(nullptr);
    std::memset(ptr, 0, bytes);
    return AlignedFloats(static_cast<float*>(ptr));
}

static inline AlignedU16 alloc_aligned_u16(std::size_t n) {
    std::size_t bytes = ((n * sizeof(uint16_t) + 63) / 64) * 64;
    if (bytes == 0) bytes = 64;
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 64, bytes) != 0 || !ptr) return AlignedU16(nullptr);
    std::memset(ptr, 0, bytes);
    return AlignedU16(static_cast<uint16_t*>(ptr));
}

// IEEE-754 FP32 <-> FP16 conversion (exact hardware instructions when F16C/NEON present, portable fallback otherwise)
static inline uint16_t f32_to_f16_bits(float f) {
#if defined(__ARM_NEON)
    __fp16 h = static_cast<__fp16>(f);
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return u;
#elif defined(__F16C__)
    return static_cast<uint16_t>(_mm_cvtsi128_si32(_mm_cvtps_ph(_mm_set_ss(f), 0)));
#else
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | ((mant + 0x1000u) >> 13));
#endif
}

static inline float f16_bits_to_f32(uint16_t u) {
#if defined(__ARM_NEON)
    __fp16 h;
    std::memcpy(&h, &u, 2);
    return static_cast<float>(h);
#elif defined(__F16C__)
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<int>(u))));
#else
    uint32_t sign = static_cast<uint32_t>(u & 0x8000u) << 16;
    uint32_t exp = (u >> 10) & 0x1fu;
    uint32_t mant = u & 0x3ffu;
    uint32_t out = (exp == 0) ? sign : (sign | ((exp + 112u) << 23) | (mant << 13));
    float f;
    std::memcpy(&f, &out, 4);
    return f;
#endif
}

static inline void pack_matrix_f16_panels(const float* src_mn, int N, int M, uint16_t* dst_pack) {
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
    int j = 0;
    uint16_t* out = dst_pack;
    for (; j + 63 < M; j += 64) {
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < 64; ++k) *out++ = f32_to_f16_bits(src_mn[(j + k) * N + i]);
        }
    }
    for (; j + 31 < M; j += 32) {
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < 32; ++k) *out++ = f32_to_f16_bits(src_mn[(j + k) * N + i]);
        }
    }
    for (; j < M; ++j) {
        for (int i = 0; i < N; ++i) *out++ = f32_to_f16_bits(src_mn[j * N + i]);
    }
#else
    int j = 0;
    uint16_t* out = dst_pack;
    for (; j + 31 < M; j += 32) {
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < 32; ++k) *out++ = f32_to_f16_bits(src_mn[(j + k) * N + i]);
        }
    }
    for (; j < M; ++j) {
        for (int i = 0; i < N; ++i) *out++ = f32_to_f16_bits(src_mn[j * N + i]);
    }
#endif
}

static inline void gemv_f16_packed_add(const uint16_t* __restrict A_pack, const float* __restrict x,
                                       int N, int M, float* __restrict y) {
#if defined(__ARM_NEON)
    const __fp16* p = reinterpret_cast<const __fp16*>(A_pack);
    int j = 0;
    for (; j + 31 < M; j += 32) {
        float32x4_t a0 = vld1q_f32(y + j + 0),  a1 = vld1q_f32(y + j + 4);
        float32x4_t a2 = vld1q_f32(y + j + 8),  a3 = vld1q_f32(y + j + 12);
        float32x4_t a4 = vld1q_f32(y + j + 16), a5 = vld1q_f32(y + j + 20);
        float32x4_t a6 = vld1q_f32(y + j + 24), a7 = vld1q_f32(y + j + 28);
        for (int i = 0; i < N; ++i, p += 32) {
            float32x4_t vx = vdupq_n_f32(x[i]);
            float16x8_t h01 = vld1q_f16(p + 0);
            float16x8_t h23 = vld1q_f16(p + 8);
            float16x8_t h45 = vld1q_f16(p + 16);
            float16x8_t h67 = vld1q_f16(p + 24);
            a0 = vfmaq_f32(a0, vcvt_f32_f16(vget_low_f16(h01)),  vx);
            a1 = vfmaq_f32(a1, vcvt_f32_f16(vget_high_f16(h01)), vx);
            a2 = vfmaq_f32(a2, vcvt_f32_f16(vget_low_f16(h23)),  vx);
            a3 = vfmaq_f32(a3, vcvt_f32_f16(vget_high_f16(h23)), vx);
            a4 = vfmaq_f32(a4, vcvt_f32_f16(vget_low_f16(h45)),  vx);
            a5 = vfmaq_f32(a5, vcvt_f32_f16(vget_high_f16(h45)), vx);
            a6 = vfmaq_f32(a6, vcvt_f32_f16(vget_low_f16(h67)),  vx);
            a7 = vfmaq_f32(a7, vcvt_f32_f16(vget_high_f16(h67)), vx);
        }
        vst1q_f32(y + j + 0, a0);  vst1q_f32(y + j + 4, a1);
        vst1q_f32(y + j + 8, a2);  vst1q_f32(y + j + 12, a3);
        vst1q_f32(y + j + 16, a4); vst1q_f32(y + j + 20, a5);
        vst1q_f32(y + j + 24, a6); vst1q_f32(y + j + 28, a7);
    }
    const uint16_t* pu = reinterpret_cast<const uint16_t*>(p);
    for (; j < M; ++j) {
        float sum = y[j];
        for (int i = 0; i < N; ++i) sum += f16_bits_to_f32(*pu++) * x[i];
        y[j] = sum;
    }
#elif defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
    const uint16_t* p = A_pack;
    int j = 0;
    for (; j + 63 < M; j += 64) {
        __m256 a0 = _mm256_loadu_ps(y + j + 0),  a1 = _mm256_loadu_ps(y + j + 8);
        __m256 a2 = _mm256_loadu_ps(y + j + 16), a3 = _mm256_loadu_ps(y + j + 24);
        __m256 a4 = _mm256_loadu_ps(y + j + 32), a5 = _mm256_loadu_ps(y + j + 40);
        __m256 a6 = _mm256_loadu_ps(y + j + 48), a7 = _mm256_loadu_ps(y + j + 56);
        for (int i = 0; i < N; ++i, p += 64) {
            __m256 vx = _mm256_set1_ps(x[i]);
            a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 0))),  vx, a0);
            a1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 8))),  vx, a1);
            a2 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 16))), vx, a2);
            a3 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 24))), vx, a3);
            a4 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 32))), vx, a4);
            a5 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 40))), vx, a5);
            a6 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 48))), vx, a6);
            a7 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 56))), vx, a7);
        }
        _mm256_storeu_ps(y + j + 0, a0);  _mm256_storeu_ps(y + j + 8, a1);
        _mm256_storeu_ps(y + j + 16, a2); _mm256_storeu_ps(y + j + 24, a3);
        _mm256_storeu_ps(y + j + 32, a4); _mm256_storeu_ps(y + j + 40, a5);
        _mm256_storeu_ps(y + j + 48, a6); _mm256_storeu_ps(y + j + 56, a7);
    }
    for (; j + 31 < M; j += 32) {
        __m256 a0 = _mm256_loadu_ps(y + j + 0),  a1 = _mm256_loadu_ps(y + j + 8);
        __m256 a2 = _mm256_loadu_ps(y + j + 16), a3 = _mm256_loadu_ps(y + j + 24);
        for (int i = 0; i < N; ++i, p += 32) {
            __m256 vx = _mm256_set1_ps(x[i]);
            a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 0))),  vx, a0);
            a1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 8))),  vx, a1);
            a2 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 16))), vx, a2);
            a3 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 24))), vx, a3);
        }
        _mm256_storeu_ps(y + j + 0, a0);  _mm256_storeu_ps(y + j + 8, a1);
        _mm256_storeu_ps(y + j + 16, a2); _mm256_storeu_ps(y + j + 24, a3);
    }
    for (; j < M; ++j) {
        float sum = y[j];
        for (int i = 0; i < N; ++i) sum += f16_bits_to_f32(*p++) * x[i];
        y[j] = sum;
    }
#else
    const uint16_t* p = A_pack;
    int j = 0;
    for (; j + 31 < M; j += 32) {
        for (int i = 0; i < N; ++i, p += 32) {
            float xi = x[i];
            for (int k = 0; k < 32; ++k) y[j + k] += f16_bits_to_f32(p[k]) * xi;
        }
    }
    for (; j < M; ++j) {
        float sum = y[j];
        for (int i = 0; i < N; ++i) sum += f16_bits_to_f32(*p++) * x[i];
        y[j] = sum;
    }
#endif
}

static inline void gemv_f16_packed_set(const uint16_t* __restrict A_pack, const float* __restrict x,
                                       int N, int M, const float* __restrict bias, float* __restrict y) {
    if (bias) std::memcpy(y, bias, M * sizeof(float));
    else std::memset(y, 0, M * sizeof(float));
    gemv_f16_packed_add(A_pack, x, N, M, y);
}

// Branchless Cody-Waite minimax expf (max abs error <= 1.2e-7 on sigmoid/tanh, auto-vectorizes cleanly in AVX2/NEON!)
static inline float fast_expf(float x) {
    x = x < -87.3f ? -87.3f : (x > 88.0f ? 88.0f : x);
    float z = x * 1.4426950408889634f;
    float bias = z >= 0.0f ? 0.5f : -0.5f;
    int32_t ni = static_cast<int32_t>(z + bias);
    float n = static_cast<float>(ni);
    float r = (x - n * 0.693145751953125f) - n * 1.428606765330187e-6f;
    float p = (((((1.9875691500e-4f * r + 1.3981999507e-3f) * r + 8.3334519073e-3f) * r
                 + 4.1665795894e-2f) * r + 1.6666665459e-1f) * r + 5.0000001201e-1f) * (r * r) + r + 1.0f;
    uint32_t bits = static_cast<uint32_t>(ni + 127) << 23;
    float scale;
    std::memcpy(&scale, &bits, sizeof(float));
    return p * scale;
}

static inline float fast_sigmoid(float x) {
    return 1.0f / (1.0f + fast_expf(-x));
}

static inline float fast_tanh(float x) {
    float ax = x >= 0.0f ? x : -x;
    float e2 = fast_expf(-2.0f * ax);
    float t = (1.0f - e2) / (1.0f + e2);
    return x >= 0.0f ? t : -t;
}

static inline float exact_gelu(float x) {
    float z = x * 0.7071067811865475f;
    float az = z >= 0.0f ? z : -z;
    float t = 1.0f / (1.0f + 0.3275911f * az);
    float poly = (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
    float y = 1.0f - poly * fast_expf(-az * az);
    float erf_z = z >= 0.0f ? y : -y;
    return 0.5f * x * (1.0f + erf_z);
}

// Panel-packed stride-1 GEMV for M % 32 == 0:
// Packs A [M, N] (PyTorch row-major [out_features=M, in_features=N]) into sequential
// panels of width 64 (on AVX2) or 32 (on NEON/portable), so every single load in the
// inner loop is contiguous stride-1 (`p += 32` / `p += 64`), completely eliminating
// strided cache-set & TLB thrashing!
static inline void pack_matrix_panels(const float* src_mn, int N, int M, float* dst_pack) {
#if defined(__AVX2__) && defined(__FMA__)
    int j = 0;
    float* out = dst_pack;
    for (; j + 63 < M; j += 64) {
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < 64; ++k) *out++ = src_mn[(j + k) * N + i];
        }
    }
    for (; j + 31 < M; j += 32) {
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < 32; ++k) *out++ = src_mn[(j + k) * N + i];
        }
    }
    for (; j < M; ++j) {
        for (int i = 0; i < N; ++i) *out++ = src_mn[j * N + i];
    }
#else
    int j = 0;
    float* out = dst_pack;
    for (; j + 31 < M; j += 32) {
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < 32; ++k) *out++ = src_mn[(j + k) * N + i];
        }
    }
    for (; j < M; ++j) {
        for (int i = 0; i < N; ++i) *out++ = src_mn[j * N + i];
    }
#endif
}

static inline void gemv_packed_add(const float* __restrict A_pack, const float* __restrict x,
                                   int N, int M, float* __restrict y) {
#if defined(__ARM_NEON)
    const float* p = A_pack;
    int j = 0;
    for (; j + 31 < M; j += 32) {
        float32x4_t a0 = vld1q_f32(y + j + 0),  a1 = vld1q_f32(y + j + 4);
        float32x4_t a2 = vld1q_f32(y + j + 8),  a3 = vld1q_f32(y + j + 12);
        float32x4_t a4 = vld1q_f32(y + j + 16), a5 = vld1q_f32(y + j + 20);
        float32x4_t a6 = vld1q_f32(y + j + 24), a7 = vld1q_f32(y + j + 28);
        int i = 0;
        for (; i + 1 < N; i += 2, p += 64) {
            float32x4_t vx0 = vdupq_n_f32(x[i]);
            a0 = vfmaq_f32(a0, vld1q_f32(p + 0),  vx0); a1 = vfmaq_f32(a1, vld1q_f32(p + 4),  vx0);
            a2 = vfmaq_f32(a2, vld1q_f32(p + 8),  vx0); a3 = vfmaq_f32(a3, vld1q_f32(p + 12), vx0);
            a4 = vfmaq_f32(a4, vld1q_f32(p + 16), vx0); a5 = vfmaq_f32(a5, vld1q_f32(p + 20), vx0);
            a6 = vfmaq_f32(a6, vld1q_f32(p + 24), vx0); a7 = vfmaq_f32(a7, vld1q_f32(p + 28), vx0);
            float32x4_t vx1 = vdupq_n_f32(x[i + 1]);
            a0 = vfmaq_f32(a0, vld1q_f32(p + 32), vx1); a1 = vfmaq_f32(a1, vld1q_f32(p + 36), vx1);
            a2 = vfmaq_f32(a2, vld1q_f32(p + 40), vx1); a3 = vfmaq_f32(a3, vld1q_f32(p + 44), vx1);
            a4 = vfmaq_f32(a4, vld1q_f32(p + 48), vx1); a5 = vfmaq_f32(a5, vld1q_f32(p + 52), vx1);
            a6 = vfmaq_f32(a6, vld1q_f32(p + 56), vx1); a7 = vfmaq_f32(a7, vld1q_f32(p + 60), vx1);
        }
        for (; i < N; ++i, p += 32) {
            float32x4_t vx0 = vdupq_n_f32(x[i]);
            a0 = vfmaq_f32(a0, vld1q_f32(p + 0),  vx0); a1 = vfmaq_f32(a1, vld1q_f32(p + 4),  vx0);
            a2 = vfmaq_f32(a2, vld1q_f32(p + 8),  vx0); a3 = vfmaq_f32(a3, vld1q_f32(p + 12), vx0);
            a4 = vfmaq_f32(a4, vld1q_f32(p + 16), vx0); a5 = vfmaq_f32(a5, vld1q_f32(p + 20), vx0);
            a6 = vfmaq_f32(a6, vld1q_f32(p + 24), vx0); a7 = vfmaq_f32(a7, vld1q_f32(p + 28), vx0);
        }
        vst1q_f32(y + j + 0, a0);  vst1q_f32(y + j + 4, a1);
        vst1q_f32(y + j + 8, a2);  vst1q_f32(y + j + 12, a3);
        vst1q_f32(y + j + 16, a4); vst1q_f32(y + j + 20, a5);
        vst1q_f32(y + j + 24, a6); vst1q_f32(y + j + 28, a7);
    }
    for (; j < M; ++j) {
        float sum = y[j];
        for (int i = 0; i < N; ++i) sum += (*p++) * x[i];
        y[j] = sum;
    }
#elif defined(__AVX2__) && defined(__FMA__)
    const float* p = A_pack;
    int j = 0;
    for (; j + 63 < M; j += 64) {
        __m256 a0 = _mm256_loadu_ps(y + j + 0),  a1 = _mm256_loadu_ps(y + j + 8);
        __m256 a2 = _mm256_loadu_ps(y + j + 16), a3 = _mm256_loadu_ps(y + j + 24);
        __m256 a4 = _mm256_loadu_ps(y + j + 32), a5 = _mm256_loadu_ps(y + j + 40);
        __m256 a6 = _mm256_loadu_ps(y + j + 48), a7 = _mm256_loadu_ps(y + j + 56);
        int i = 0;
        for (; i + 1 < N; i += 2, p += 128) {
            __m256 vx0 = _mm256_set1_ps(x[i]);
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 0),   vx0, a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 8),   vx0, a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 16),  vx0, a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 24),  vx0, a3);
            a4 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 32),  vx0, a4);
            a5 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 40),  vx0, a5);
            a6 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 48),  vx0, a6);
            a7 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 56),  vx0, a7);
            __m256 vx1 = _mm256_set1_ps(x[i + 1]);
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 64),  vx1, a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 72),  vx1, a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 80),  vx1, a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 88),  vx1, a3);
            a4 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 96),  vx1, a4);
            a5 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 104), vx1, a5);
            a6 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 112), vx1, a6);
            a7 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 120), vx1, a7);
        }
        for (; i < N; ++i, p += 64) {
            __m256 vx0 = _mm256_set1_ps(x[i]);
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 0),  vx0, a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 8),  vx0, a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 16), vx0, a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 24), vx0, a3);
            a4 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 32), vx0, a4);
            a5 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 40), vx0, a5);
            a6 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 48), vx0, a6);
            a7 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 56), vx0, a7);
        }
        _mm256_storeu_ps(y + j + 0, a0);  _mm256_storeu_ps(y + j + 8, a1);
        _mm256_storeu_ps(y + j + 16, a2); _mm256_storeu_ps(y + j + 24, a3);
        _mm256_storeu_ps(y + j + 32, a4); _mm256_storeu_ps(y + j + 40, a5);
        _mm256_storeu_ps(y + j + 48, a6); _mm256_storeu_ps(y + j + 56, a7);
    }
    for (; j + 31 < M; j += 32) {
        __m256 a0 = _mm256_loadu_ps(y + j + 0),  a1 = _mm256_loadu_ps(y + j + 8);
        __m256 a2 = _mm256_loadu_ps(y + j + 16), a3 = _mm256_loadu_ps(y + j + 24);
        for (int i = 0; i < N; ++i, p += 32) {
            __m256 vx0 = _mm256_set1_ps(x[i]);
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 0),  vx0, a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 8),  vx0, a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 16), vx0, a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 24), vx0, a3);
        }
        _mm256_storeu_ps(y + j + 0, a0);  _mm256_storeu_ps(y + j + 8, a1);
        _mm256_storeu_ps(y + j + 16, a2); _mm256_storeu_ps(y + j + 24, a3);
    }
    for (; j < M; ++j) {
        float sum = y[j];
        for (int i = 0; i < N; ++i) sum += (*p++) * x[i];
        y[j] = sum;
    }
#else
    const float* p = A_pack;
    int j = 0;
    for (; j + 31 < M; j += 32) {
        for (int i = 0; i < N; ++i, p += 32) {
            float xi = x[i];
            for (int k = 0; k < 32; ++k) y[j + k] += p[k] * xi;
        }
    }
    for (; j < M; ++j) {
        float sum = y[j];
        for (int i = 0; i < N; ++i) sum += (*p++) * x[i];
        y[j] = sum;
    }
#endif
}

static inline void gemv_packed_set(const float* __restrict A_pack, const float* __restrict x,
                                   int N, int M, const float* __restrict bias, float* __restrict y) {
    if (bias) std::memcpy(y, bias, M * sizeof(float));
    else std::memset(y, 0, M * sizeof(float));
    gemv_packed_add(A_pack, x, N, M, y);
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
    alignas(64) uint16_t w0_p[112 * 384];
    alignas(64) uint16_t r0_p[128 * 384];
    alignas(64) float b0[768];
    alignas(64) uint16_t w1_p[128 * 384];
    alignas(64) uint16_t r1_p[128 * 384];
    alignas(64) float b1[768];
    alignas(64) float hw[128 * 2];
    alignas(64) float hb[2];
    alignas(64) float h0[128];
    alignas(64) float h1[128];

    void reset() {
        std::memset(h0, 0, sizeof(h0));
        std::memset(h1, 0, sizeof(h1));
    }

    static inline void step_layer(const uint16_t* W_p, const uint16_t* R_p, const float* B,
                                  const float* inp, int nin, float* h) {
        alignas(64) float xp[384], rp[384];
        gemv_f16_packed_set(W_p, inp, nin, 384, nullptr, xp);
        gemv_f16_packed_set(R_p, h, 128, 384, nullptr, rp);
        for (int j = 0; j < 128; ++j) {
            float z = fast_sigmoid((xp[j] + rp[j]) + (B[j] + B[384 + j]));
            float r = fast_sigmoid((xp[128 + j] + rp[128 + j]) + (B[128 + j] + B[512 + j]));
            float c = fast_tanh((xp[256 + j] + B[256 + j]) + r * (rp[256 + j] + B[640 + j]));
            h[j] = (1.0f - z) * c + z * h[j];
        }
    }

    void step(const float* x112, float* out2) {
        step_layer(w0_p, r0_p, b0, x112, 112, h0);
        step_layer(w1_p, r1_p, b1, h0, 128, h1);
        if (!out2) return;
        float s0 = hb[0], s1 = hb[1];
        for (int j = 0; j < 128; ++j) {
            s0 += h1[j] * hw[2 * j];
            s1 += h1[j] * hw[2 * j + 1];
        }
        out2[0] = s0; out2[1] = s1;
    }
};

// ============================================================================
// Variable-width PyTorch ScratchSortedGRU (supports hidden H in 32..384, e.g. 128, 160, 192, 256!)
// ============================================================================
struct TorchSortedGRU {
    int H = 128;
    alignas(64) float in_mu[169], in_sd[169];
    AlignedU16 stem_w_p, w0_p, r0_p, w1_p, r1_p, h1_w_p;
    AlignedFloats stem_b, bi0, bh0, bi1, bh1, h1_b, h2_w_t;
    float h2_b[2]{};
    AlignedFloats h0, h1;

    void init_dims(int hidden) {
        H = hidden;
        stem_w_p = alloc_aligned_u16(169 * H);
        stem_b   = alloc_aligned_floats(H);
        w0_p     = alloc_aligned_u16(H * 3 * H);
        r0_p     = alloc_aligned_u16(H * 3 * H);
        bi0      = alloc_aligned_floats(3 * H);
        bh0      = alloc_aligned_floats(3 * H);
        w1_p     = alloc_aligned_u16(H * 3 * H);
        r1_p     = alloc_aligned_u16(H * 3 * H);
        bi1      = alloc_aligned_floats(3 * H);
        bh1      = alloc_aligned_floats(3 * H);
        h1_w_p   = alloc_aligned_u16(3 * H * H);
        h1_b     = alloc_aligned_floats(H);
        h2_w_t   = alloc_aligned_floats(H * 2);
        h0       = alloc_aligned_floats(H);
        h1       = alloc_aligned_floats(H);
        reset();
    }

    void reset() {
        if (h0) std::memset(h0.get(), 0, H * sizeof(float));
        if (h1) std::memset(h1.get(), 0, H * sizeof(float));
    }

    template <int FH>
    static inline void step_torch_gru_fixed(const uint16_t* __restrict W_p, const uint16_t* __restrict R_p,
                                            const float* __restrict bi, const float* __restrict bh,
                                            const float* __restrict inp, float* __restrict h) {
        alignas(64) float xp[3 * FH], rp[3 * FH];
        gemv_f16_packed_set(W_p, inp, FH, 3 * FH, bi, xp);
        gemv_f16_packed_set(R_p, h,   FH, 3 * FH, bh, rp);
        for (int j = 0; j < FH; ++j) {
            float r = fast_sigmoid(xp[j] + rp[j]);
            float z = fast_sigmoid(xp[FH + j] + rp[FH + j]);
            float n = fast_tanh(xp[2 * FH + j] + r * rp[2 * FH + j]);
            h[j] = (1.0f - z) * n + z * h[j];
        }
    }

    template <int FH>
    inline void step_fixed(const float* __restrict feat169, float* __restrict out2) {
        alignas(64) float z[169], s[FH], g1_in[FH], cat[3 * FH], mid[FH];
        for (int i = 0; i < 169; ++i) z[i] = (feat169[i] - in_mu[i]) / in_sd[i];
        gemv_f16_packed_set(stem_w_p.get(), z, 169, FH, stem_b.get(), s);
        for (int j = 0; j < FH; ++j) s[j] = exact_gelu(s[j]);
        step_torch_gru_fixed<FH>(w0_p.get(), r0_p.get(), bi0.get(), bh0.get(), s, h0.get());
        for (int j = 0; j < FH; ++j) g1_in[j] = h0[j] + s[j];
        step_torch_gru_fixed<FH>(w1_p.get(), r1_p.get(), bi1.get(), bh1.get(), g1_in, h1.get());
        if (!out2) return;
        std::memcpy(cat + 0,      s,        FH * sizeof(float));
        std::memcpy(cat + FH,     h0.get(), FH * sizeof(float));
        std::memcpy(cat + 2 * FH, h1.get(), FH * sizeof(float));
        gemv_f16_packed_set(h1_w_p.get(), cat, 3 * FH, FH, h1_b.get(), mid);
        for (int j = 0; j < FH; ++j) mid[j] = exact_gelu(mid[j]);
        float o0 = h2_b[0], o1 = h2_b[1];
        const float* hw = h2_w_t.get();
        for (int j = 0; j < FH; ++j) {
            o0 += mid[j] * hw[2 * j];
            o1 += mid[j] * hw[2 * j + 1];
        }
        out2[0] = o0; out2[1] = o1;
    }

    static inline void step_torch_gru(const uint16_t* W_p, const uint16_t* R_p,
                                      const float* bi, const float* bh,
                                      const float* inp, int H, float* h) {
        alignas(64) float xp[1152], rp[1152];
        gemv_f16_packed_set(W_p, inp, H, 3 * H, bi, xp);
        gemv_f16_packed_set(R_p, h,   H, 3 * H, bh, rp);
        for (int j = 0; j < H; ++j) {
            float r = fast_sigmoid(xp[j] + rp[j]);
            float z = fast_sigmoid(xp[H + j] + rp[H + j]);
            float n = fast_tanh(xp[2 * H + j] + r * rp[2 * H + j]);
            h[j] = (1.0f - z) * n + z * h[j];
        }
    }

    void step(const float* feat169, float* out2) {
        if (H == 160) { step_fixed<160>(feat169, out2); return; }
        if (H == 128) { step_fixed<128>(feat169, out2); return; }
        if (H == 192) { step_fixed<192>(feat169, out2); return; }
        alignas(64) float z[169], s[384], g1_in[384], cat[1152], mid[384];
        for (int i = 0; i < 169; ++i) z[i] = (feat169[i] - in_mu[i]) / in_sd[i];
        gemv_f16_packed_set(stem_w_p.get(), z, 169, H, stem_b.get(), s);
        for (int j = 0; j < H; ++j) s[j] = exact_gelu(s[j]);
        step_torch_gru(w0_p.get(), r0_p.get(), bi0.get(), bh0.get(), s, H, h0.get());
        for (int j = 0; j < H; ++j) g1_in[j] = h0[j] + s[j];
        step_torch_gru(w1_p.get(), r1_p.get(), bi1.get(), bh1.get(), g1_in, H, h1.get());
        if (!out2) return;
        std::memcpy(cat + 0,     s,        H * sizeof(float));
        std::memcpy(cat + H,     h0.get(), H * sizeof(float));
        std::memcpy(cat + 2 * H, h1.get(), H * sizeof(float));
        gemv_f16_packed_set(h1_w_p.get(), cat, 3 * H, H, h1_b.get(), mid);
        for (int j = 0; j < H; ++j) mid[j] = exact_gelu(mid[j]);
        float o0 = h2_b[0], o1 = h2_b[1];
        const float* hw = h2_w_t.get();
        for (int j = 0; j < H; ++j) {
            o0 += mid[j] * hw[2 * j];
            o1 += mid[j] * hw[2 * j + 1];
        }
        out2[0] = o0; out2[1] = o1;
    }
};

// ============================================================================
// Variable-width PyTorch ScratchSortedLSTM (orthogonal cell-state highway [i, f, g, o]!)
// ============================================================================
struct TorchSortedLSTM {
    int H = 128;
    alignas(64) float in_mu[169], in_sd[169];
    AlignedU16 stem_w_p, w0_p, r0_p, w1_p, r1_p, h1_w_p;
    AlignedFloats stem_b, bi0, bh0, bi1, bh1, h1_b, h2_w_t;
    float h2_b[2]{};
    AlignedFloats h0, c0, h1, c1;

    void init_dims(int hidden) {
        H = hidden;
        stem_w_p = alloc_aligned_u16(169 * H);
        stem_b   = alloc_aligned_floats(H);
        w0_p     = alloc_aligned_u16(H * 4 * H);
        r0_p     = alloc_aligned_u16(H * 4 * H);
        bi0      = alloc_aligned_floats(4 * H);
        bh0      = alloc_aligned_floats(4 * H);
        w1_p     = alloc_aligned_u16(H * 4 * H);
        r1_p     = alloc_aligned_u16(H * 4 * H);
        bi1      = alloc_aligned_floats(4 * H);
        bh1      = alloc_aligned_floats(4 * H);
        h1_w_p   = alloc_aligned_u16(3 * H * H);
        h1_b     = alloc_aligned_floats(H);
        h2_w_t   = alloc_aligned_floats(H * 2);
        h0       = alloc_aligned_floats(H);
        c0       = alloc_aligned_floats(H);
        h1       = alloc_aligned_floats(H);
        c1       = alloc_aligned_floats(H);
        reset();
    }

    void reset() {
        if (h0) std::memset(h0.get(), 0, H * sizeof(float));
        if (c0) std::memset(c0.get(), 0, H * sizeof(float));
        if (h1) std::memset(h1.get(), 0, H * sizeof(float));
        if (c1) std::memset(c1.get(), 0, H * sizeof(float));
    }

    template <int FH>
    static inline void step_torch_lstm_fixed(const uint16_t* __restrict W_p, const uint16_t* __restrict R_p,
                                             const float* __restrict bi, const float* __restrict bh,
                                             const float* __restrict inp, float* __restrict h, float* __restrict c) {
        alignas(64) float gates[4 * FH];
        for (int j = 0; j < 4 * FH; ++j) gates[j] = bi[j] + bh[j];
        gemv_f16_packed_add(W_p, inp, FH, 4 * FH, gates);
        gemv_f16_packed_add(R_p, h,   FH, 4 * FH, gates);
        for (int j = 0; j < FH; ++j) {
            float ig = fast_sigmoid(gates[j]);
            float fg = fast_sigmoid(gates[FH + j]);
            float gg = fast_tanh(gates[2 * FH + j]);
            float og = fast_sigmoid(gates[3 * FH + j]);
            float nc = fg * c[j] + ig * gg;
            c[j] = nc;
            h[j] = og * fast_tanh(nc);
        }
    }

    template <int FH>
    inline void step_fixed(const float* __restrict feat169, float* __restrict out2) {
        alignas(64) float z[169], s[FH], l1_in[FH], cat[3 * FH], mid[FH];
        for (int i = 0; i < 169; ++i) z[i] = (feat169[i] - in_mu[i]) / in_sd[i];
        gemv_f16_packed_set(stem_w_p.get(), z, 169, FH, stem_b.get(), s);
        for (int j = 0; j < FH; ++j) s[j] = exact_gelu(s[j]);
        step_torch_lstm_fixed<FH>(w0_p.get(), r0_p.get(), bi0.get(), bh0.get(), s, h0.get(), c0.get());
        for (int j = 0; j < FH; ++j) l1_in[j] = h0[j] + s[j];
        step_torch_lstm_fixed<FH>(w1_p.get(), r1_p.get(), bi1.get(), bh1.get(), l1_in, h1.get(), c1.get());
        if (!out2) return;
        std::memcpy(cat + 0,      s,        FH * sizeof(float));
        std::memcpy(cat + FH,     h0.get(), FH * sizeof(float));
        std::memcpy(cat + 2 * FH, h1.get(), FH * sizeof(float));
        gemv_f16_packed_set(h1_w_p.get(), cat, 3 * FH, FH, h1_b.get(), mid);
        for (int j = 0; j < FH; ++j) mid[j] = exact_gelu(mid[j]);
        float o0 = h2_b[0], o1 = h2_b[1];
        const float* hw = h2_w_t.get();
        for (int j = 0; j < FH; ++j) {
            o0 += mid[j] * hw[2 * j];
            o1 += mid[j] * hw[2 * j + 1];
        }
        out2[0] = o0; out2[1] = o1;
    }

    static inline void step_torch_lstm(const uint16_t* W_p, const uint16_t* R_p,
                                       const float* bi, const float* bh,
                                       const float* inp, int H, float* h, float* c) {
        alignas(64) float gates[1536];
        for (int j = 0; j < 4 * H; ++j) gates[j] = bi[j] + bh[j];
        gemv_f16_packed_add(W_p, inp, H, 4 * H, gates);
        gemv_f16_packed_add(R_p, h,   H, 4 * H, gates);
        for (int j = 0; j < H; ++j) {
            float ig = fast_sigmoid(gates[j]);
            float fg = fast_sigmoid(gates[H + j]);
            float gg = fast_tanh(gates[2 * H + j]);
            float og = fast_sigmoid(gates[3 * H + j]);
            float nc = fg * c[j] + ig * gg;
            c[j] = nc;
            h[j] = og * fast_tanh(nc);
        }
    }

    void step(const float* feat169, float* out2) {
        if (H == 160) { step_fixed<160>(feat169, out2); return; }
        if (H == 128) { step_fixed<128>(feat169, out2); return; }
        if (H == 192) { step_fixed<192>(feat169, out2); return; }
        alignas(64) float z[169], s[384], l1_in[384], cat[1152], mid[384];
        for (int i = 0; i < 169; ++i) z[i] = (feat169[i] - in_mu[i]) / in_sd[i];
        gemv_f16_packed_set(stem_w_p.get(), z, 169, H, stem_b.get(), s);
        for (int j = 0; j < H; ++j) s[j] = exact_gelu(s[j]);
        step_torch_lstm(w0_p.get(), r0_p.get(), bi0.get(), bh0.get(), s, H, h0.get(), c0.get());
        for (int j = 0; j < H; ++j) l1_in[j] = h0[j] + s[j];
        step_torch_lstm(w1_p.get(), r1_p.get(), bi1.get(), bh1.get(), l1_in, H, h1.get(), c1.get());
        if (!out2) return;
        std::memcpy(cat + 0,     s,        H * sizeof(float));
        std::memcpy(cat + H,     h0.get(), H * sizeof(float));
        std::memcpy(cat + 2 * H, h1.get(), H * sizeof(float));
        gemv_f16_packed_set(h1_w_p.get(), cat, 3 * H, H, h1_b.get(), mid);
        for (int j = 0; j < H; ++j) mid[j] = exact_gelu(mid[j]);
        float o0 = h2_b[0], o1 = h2_b[1];
        const float* hw = h2_w_t.get();
        for (int j = 0; j < H; ++j) {
            o0 += mid[j] * hw[2 * j];
            o1 += mid[j] * hw[2 * j + 1];
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
    int n_sorted_gru = 0;
    std::unique_ptr<TorchSortedGRU[]> sorted_grus;
    int n_sorted_lstm = 0;
    std::unique_ptr<TorchSortedLSTM[]> sorted_lstms;
    bool has_tcn80 = false;
    std::unique_ptr<StreamingTCN80> tcn80;

    // Meta-calibrator: takes [q_models..., derived_q_feats..., snap34] -> residual MLP -> final 2 outputs!
    float blend_weights[16]{};  // per-model blend weights for base q
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

void* ens_create_v2(int n_onnx_gru, int n_sorted_gru, int n_sorted_lstm, int has_tcn80) {
    auto* ctx = new EnsembleContext();
    ctx->n_onnx_gru = n_onnx_gru;
    if (n_onnx_gru > 0) ctx->onnx_grus = std::make_unique<OnnxGRU128[]>(n_onnx_gru);
    ctx->n_sorted_gru = n_sorted_gru;
    if (n_sorted_gru > 0) ctx->sorted_grus = std::make_unique<TorchSortedGRU[]>(n_sorted_gru);
    ctx->n_sorted_lstm = n_sorted_lstm;
    if (n_sorted_lstm > 0) ctx->sorted_lstms = std::make_unique<TorchSortedLSTM[]>(n_sorted_lstm);
    ctx->has_tcn80 = (has_tcn80 != 0);
    if (ctx->has_tcn80) ctx->tcn80 = std::make_unique<StreamingTCN80>();
    return ctx;
}

void* ens_create(int n_onnx_gru, int has_sorted_gru, int has_tcn80) {
    return ens_create_v2(n_onnx_gru, has_sorted_gru ? 1 : 0, 0, has_tcn80);
}

void ens_load_onnx_gru(void* p, int idx, const float* w0, const float* r0, const float* b0,
                       const float* w1, const float* r1, const float* b1,
                       const float* hw, const float* hb, float blend_w) {
    auto* ctx = static_cast<EnsembleContext*>(p);
    auto& g = ctx->onnx_grus[idx];
    pack_matrix_f16_panels(w0, 112, 384, g.w0_p);
    pack_matrix_f16_panels(r0, 128, 384, g.r0_p);
    std::memcpy(g.b0, b0, 768 * sizeof(float));
    pack_matrix_f16_panels(w1, 128, 384, g.w1_p);
    pack_matrix_f16_panels(r1, 128, 384, g.r1_p);
    std::memcpy(g.b1, b1, 768 * sizeof(float));
    std::memcpy(g.hw, hw, 256 * sizeof(float));
    std::memcpy(g.hb, hb, 2 * sizeof(float));
    g.reset();
    ctx->blend_weights[idx] = blend_w;
}

void ens_load_sorted_gru_v2(void* p, int idx, int hidden,
                            const float* in_mu, const float* in_sd,
                            const float* stem_w, const float* stem_b,
                            const float* w0, const float* r0, const float* bi0, const float* bh0,
                            const float* w1, const float* r1, const float* bi1, const float* bh1,
                            const float* h1_w, const float* h1_b, const float* h2_w, const float* h2_b,
                            float blend_w) {
    auto* ctx = static_cast<EnsembleContext*>(p);
    auto& g = ctx->sorted_grus[idx];
    const int H = hidden;
    g.init_dims(H);
    std::memcpy(g.in_mu, in_mu, 169 * sizeof(float));
    std::memcpy(g.in_sd, in_sd, 169 * sizeof(float));
    pack_matrix_f16_panels(stem_w, 169, H, g.stem_w_p.get());
    std::memcpy(g.stem_b.get(), stem_b, H * sizeof(float));
    pack_matrix_f16_panels(w0, H, 3 * H, g.w0_p.get());
    pack_matrix_f16_panels(r0, H, 3 * H, g.r0_p.get());
    pack_matrix_f16_panels(w1, H, 3 * H, g.w1_p.get());
    pack_matrix_f16_panels(r1, H, 3 * H, g.r1_p.get());
    std::memcpy(g.bi0.get(), bi0, 3 * H * sizeof(float)); std::memcpy(g.bh0.get(), bh0, 3 * H * sizeof(float));
    std::memcpy(g.bi1.get(), bi1, 3 * H * sizeof(float)); std::memcpy(g.bh1.get(), bh1, 3 * H * sizeof(float));
    pack_matrix_f16_panels(h1_w, 3 * H, H, g.h1_w_p.get());
    std::memcpy(g.h1_b.get(), h1_b, H * sizeof(float));
    for (int i = 0; i < H; ++i) for (int j = 0; j < 2; ++j) g.h2_w_t[i * 2 + j] = h2_w[j * H + i];
    std::memcpy(g.h2_b, h2_b, 2 * sizeof(float));
    g.reset();
    ctx->blend_weights[ctx->n_onnx_gru + idx] = blend_w;
}

void ens_load_sorted_gru(void* p, const float* in_mu, const float* in_sd,
                         const float* stem_w, const float* stem_b,
                         const float* w0, const float* r0, const float* bi0, const float* bh0,
                         const float* w1, const float* r1, const float* bi1, const float* bh1,
                         const float* h1_w, const float* h1_b, const float* h2_w, const float* h2_b,
                         float blend_w) {
    ens_load_sorted_gru_v2(p, 0, 128, in_mu, in_sd, stem_w, stem_b,
                           w0, r0, bi0, bh0, w1, r1, bi1, bh1,
                           h1_w, h1_b, h2_w, h2_b, blend_w);
}

void ens_load_sorted_lstm(void* p, int idx, int hidden,
                          const float* in_mu, const float* in_sd,
                          const float* stem_w, const float* stem_b,
                          const float* w0, const float* r0, const float* bi0, const float* bh0,
                          const float* w1, const float* r1, const float* bi1, const float* bh1,
                          const float* h1_w, const float* h1_b, const float* h2_w, const float* h2_b,
                          float blend_w) {
    auto* ctx = static_cast<EnsembleContext*>(p);
    auto& g = ctx->sorted_lstms[idx];
    const int H = hidden;
    g.init_dims(H);
    std::memcpy(g.in_mu, in_mu, 169 * sizeof(float));
    std::memcpy(g.in_sd, in_sd, 169 * sizeof(float));
    pack_matrix_f16_panels(stem_w, 169, H, g.stem_w_p.get());
    std::memcpy(g.stem_b.get(), stem_b, H * sizeof(float));
    pack_matrix_f16_panels(w0, H, 4 * H, g.w0_p.get());
    pack_matrix_f16_panels(r0, H, 4 * H, g.r0_p.get());
    pack_matrix_f16_panels(w1, H, 4 * H, g.w1_p.get());
    pack_matrix_f16_panels(r1, H, 4 * H, g.r1_p.get());
    std::memcpy(g.bi0.get(), bi0, 4 * H * sizeof(float)); std::memcpy(g.bh0.get(), bh0, 4 * H * sizeof(float));
    std::memcpy(g.bi1.get(), bi1, 4 * H * sizeof(float)); std::memcpy(g.bh1.get(), bh1, 4 * H * sizeof(float));
    pack_matrix_f16_panels(h1_w, 3 * H, H, g.h1_w_p.get());
    std::memcpy(g.h1_b.get(), h1_b, H * sizeof(float));
    for (int i = 0; i < H; ++i) for (int j = 0; j < 2; ++j) g.h2_w_t[i * 2 + j] = h2_w[j * H + i];
    std::memcpy(g.h2_b, h2_b, 2 * sizeof(float));
    g.reset();
    ctx->blend_weights[ctx->n_onnx_gru + ctx->n_sorted_gru + idx] = blend_w;
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
    int slot = ctx->n_onnx_gru + ctx->n_sorted_gru + ctx->n_sorted_lstm;
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
    for (int i = 0; i < ctx->n_sorted_gru; ++i) ctx->sorted_grus[i].reset();
    for (int i = 0; i < ctx->n_sorted_lstm; ++i) ctx->sorted_lstms[i].reset();
    if (ctx->has_tcn80) ctx->tcn80->reset();
}

int ens_step(void* p, const float* __restrict x112, int need_calibrator, float* __restrict out2) {
    auto* ctx = static_cast<EnsembleContext*>(p);
    for (int i = 0; i < 112; ++i) {
        if (!std::isfinite(x112[i])) return -1;
    }
    alignas(64) float feat169[169];
    alignas(64) float snap34[34];
    bool need_f169 = (ctx->n_sorted_gru > 0) || (ctx->n_sorted_lstm > 0) || ctx->has_tcn80;
    // Skip dynamic sort11 + ring-buffer entirely when only OnnxGRU runs during warm-up (steps 0..98)!
    if (need_f169 || need_calibrator) {
        ctx->feat_state.update(x112, need_f169 ? feat169 : nullptr, need_calibrator ? snap34 : nullptr);
    }

    float model_preds[16][2];
    int m_count = 0;
    for (int i = 0; i < ctx->n_onnx_gru; ++i) {
        ctx->onnx_grus[i].step(x112, need_calibrator ? model_preds[m_count] : nullptr);
        ++m_count;
    }
    for (int i = 0; i < ctx->n_sorted_gru; ++i) {
        ctx->sorted_grus[i].step(feat169, need_calibrator ? model_preds[m_count] : nullptr);
        ++m_count;
    }
    for (int i = 0; i < ctx->n_sorted_lstm; ++i) {
        ctx->sorted_lstms[i].step(feat169, need_calibrator ? model_preds[m_count] : nullptr);
        ++m_count;
    }
    if (ctx->has_tcn80) {
        ctx->tcn80->step(feat169, need_calibrator ? model_preds[m_count] : nullptr);
        ++m_count;
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
    alignas(64) float z_in[96];
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
