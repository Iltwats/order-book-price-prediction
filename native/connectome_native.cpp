#include "connectome_native.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <memory>
#include <new>

namespace {
constexpr size_t H = 128;
constexpr size_t G = 3 * H;
constexpr size_t W0 = 0;
constexpr size_t R0 = W0 + G * 112;
constexpr size_t B0 = R0 + G * H;
constexpr size_t W1 = B0 + 6 * H;
constexpr size_t R1 = W1 + G * H;
constexpr size_t B1 = R1 + G * H;
constexpr size_t HEAD_W = B1 + 6 * H;
constexpr size_t HEAD_B = HEAD_W + H * 2;
constexpr size_t WEIGHT_COUNT = HEAD_B + 2;
static_assert(WEIGHT_COUNT == 192258);

// Error reporting uses fixed string literals and never allocates.
thread_local const char* last_error = "";

struct Context {
  // Each buffer starts at its own cache-line boundary. C++17 aligned new/delete
  // preserve this over-alignment for the make_unique allocation below.
  alignas(64) std::array<float, WEIGHT_COUNT> weights{};
  alignas(64) std::array<float, H> h0{};
  alignas(64) std::array<float, H> h1{};
  alignas(64) std::array<float, G> input_projection{};
  alignas(64) std::array<float, G> recurrent_projection{};
  alignas(64) std::array<float, H> next_hidden{};
  bool packed = false;
};
static_assert(alignof(Context) >= 64);
static_assert(offsetof(Context, weights) % 64 == 0);
static_assert(offsetof(Context, h0) % 64 == 0);
static_assert(offsetof(Context, h1) % 64 == 0);
static_assert(offsetof(Context, input_projection) % 64 == 0);
static_assert(offsetof(Context, recurrent_projection) % 64 == 0);
static_assert(offsetof(Context, next_hidden) % 64 == 0);

int fail(const char* message) {
  last_error = message;
  return -1;
}

void pack_matrix(const float* source, float* target, size_t inputs) {
  for (size_t i = 0; i < inputs; ++i)
    for (size_t j = 0; j < G; ++j)
      target[i * G + j] = source[j * inputs + i];
}

void matvec(const float* matrix, const float* input, size_t inputs,
            float* result, bool packed) {
  if (packed) {
    std::fill_n(result, G, 0.0f);
    for (size_t i = 0; i < inputs; ++i) {
      const float x = input[i];
      const float* row = matrix + i * G;
      for (size_t j = 0; j < G; ++j)
        result[j] += row[j] * x;
    }
  } else {
    for (size_t j = 0; j < G; ++j) {
      const float* row = matrix + j * inputs;
      float sum = 0.0f;
      for (size_t i = 0; i < inputs; ++i)
        sum += row[i] * input[i];
      result[j] = sum;
    }
  }
}

float sigmoid(float value) {
  // Stable on both tails; no approximate exp or activation replacements.
  if (value >= 0.0f)
    return 1.0f / (1.0f + std::exp(-value));
  const float e = std::exp(value);
  return e / (1.0f + e);
}

void gru(Context& context, const float* input, size_t inputs,
         size_t w_offset, size_t r_offset, size_t b_offset,
         std::array<float, H>& hidden) {
  const float* weights = context.weights.data();
  float* x = context.input_projection.data();
  float* r = context.recurrent_projection.data();
  const float* b = weights + b_offset;
  matvec(weights + w_offset, input, inputs, x, context.packed);
  matvec(weights + r_offset, hidden.data(), H, r, context.packed);
  for (size_t j = 0; j < H; ++j) {
    const float z = sigmoid((x[j] + r[j]) + (b[j] + b[3 * H + j]));
    const float reset = sigmoid((x[H + j] + r[H + j]) +
                                (b[H + j] + b[4 * H + j]));
    // ONNX linear_before_reset=1. Keep recurrent candidate bias INSIDE reset.
    const float candidate = std::tanh((x[2 * H + j] + b[2 * H + j]) +
                                     reset * (r[2 * H + j] + b[5 * H + j]));
    context.next_hidden[j] = (1.0f - z) * candidate + z * hidden[j];
  }
  std::copy(context.next_hidden.begin(), context.next_hidden.end(), hidden.begin());
}
}  // namespace

extern "C" {
int cc_abi_version(void) { return 1; }

void* cc_create(const float* weights, size_t count, int layout) {
  last_error = "";
  if (!weights || count != WEIGHT_COUNT || (layout != 0 && layout != 1)) {
    fail("expected 192258 FP32 weights and layout 0 (row) or 1 (packed)");
    return nullptr;
  }
  for (size_t i = 0; i < count; ++i) {
    if (!std::isfinite(weights[i])) {
      fail("weights contain a nonfinite value");
      return nullptr;
    }
  }
  try {
    auto context = std::make_unique<Context>();
    context->packed = layout == 1;
    std::copy_n(weights, count, context->weights.data());
    if (context->packed) {
      pack_matrix(weights + W0, context->weights.data() + W0, 112);
      pack_matrix(weights + R0, context->weights.data() + R0, H);
      pack_matrix(weights + W1, context->weights.data() + W1, H);
      pack_matrix(weights + R1, context->weights.data() + R1, H);
    }
    return context.release();
  } catch (const std::bad_alloc&) {
    fail("could not allocate native context");
  } catch (const std::exception&) {
    fail("native context creation failed");
  }
  return nullptr;
}

int cc_reset(void* opaque) {
  if (!opaque) return fail("null native context");
  auto& context = *static_cast<Context*>(opaque);
  context.h0.fill(0.0f);
  context.h1.fill(0.0f);
  return 0;
}

int cc_step(void* opaque, const float* features, size_t feature_count,
            float* prediction, size_t prediction_count) {
  if (!opaque || !features || !prediction || feature_count != 112 || prediction_count != 2)
    return fail("invalid native step buffers");
  for (size_t i = 0; i < feature_count; ++i)
    if (!std::isfinite(features[i])) return fail("features contain a nonfinite value");
  auto& context = *static_cast<Context*>(opaque);
  gru(context, features, 112, W0, R0, B0, context.h0);
  gru(context, context.h0.data(), H, W1, R1, B1, context.h1);
  for (size_t target = 0; target < 2; ++target) {
    float sum = 0.0f;
    for (size_t j = 0; j < H; ++j)
      sum += context.h1[j] * context.weights[HEAD_W + 2 * j + target];
    prediction[target] = sum + context.weights[HEAD_B + target];
    if (!std::isfinite(prediction[target]))
      return fail("native prediction became nonfinite; reset context before reuse");
  }
  return 0;
}

int cc_hidden(void* opaque, float* hidden0, size_t count0,
              float* hidden1, size_t count1) {
  if (!opaque || !hidden0 || !hidden1 || count0 != H || count1 != H)
    return fail("invalid hidden-state buffers");
  const auto& context = *static_cast<const Context*>(opaque);
  std::copy(context.h0.begin(), context.h0.end(), hidden0);
  std::copy(context.h1.begin(), context.h1.end(), hidden1);
  return 0;
}

void cc_destroy(void* opaque) { delete static_cast<Context*>(opaque); }
const char* cc_last_error(void) { return last_error; }
}
