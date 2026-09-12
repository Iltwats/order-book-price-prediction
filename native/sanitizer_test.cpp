#include "connectome_native.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
constexpr size_t H = 128, G = 384, COUNT = 192258;
constexpr size_t W0 = 0, R0 = 43008, B0 = 92160;
constexpr size_t W1 = 92928, R1 = 142080, B1 = 191232;
constexpr size_t HEAD_W = 192000, HEAD_B = 192256;

void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

// Independent double-precision, row-major reference. It deliberately does not
// reuse native packing, workspaces, or its FP32 activation implementation.
void reference_gru(const std::vector<float>& weights, const double* input,
                   size_t inputs, size_t w, size_t r, size_t b,
                   std::array<double, H>& hidden) {
  std::array<double, G> x_projection{}, h_projection{};
  for (size_t gate = 0; gate < G; ++gate) {
    for (size_t j = 0; j < inputs; ++j)
      x_projection[gate] += double(weights[w + gate * inputs + j]) * input[j];
    for (size_t j = 0; j < H; ++j)
      h_projection[gate] += double(weights[r + gate * H + j]) * hidden[j];
  }
  std::array<double, H> next{};
  for (size_t j = 0; j < H; ++j) {
    const double z = 1.0 / (1.0 + std::exp(-(x_projection[j] + h_projection[j] + weights[b + j] + weights[b + 3 * H + j])));
    const double reset = 1.0 / (1.0 + std::exp(-(x_projection[H + j] + h_projection[H + j] + weights[b + H + j] + weights[b + 4 * H + j])));
    const double candidate = std::tanh(x_projection[2 * H + j] + weights[b + 2 * H + j] +
                                      reset * (h_projection[2 * H + j] + weights[b + 5 * H + j]));
    next[j] = (1.0 - z) * candidate + z * hidden[j];
  }
  hidden = next;
}

void run() {
  check(cc_abi_version() == 1, "wrong ABI version");
  check(cc_create(nullptr, COUNT, 0) == nullptr, "null weights accepted");
  std::vector<float> weights(COUNT);
  for (size_t j = 0; j < COUNT; ++j)
    weights[j] = static_cast<float>(0.015 * std::sin(double(j) * 0.117));
  check(cc_create(weights.data(), COUNT - 1, 0) == nullptr, "wrong weight count accepted");
  check(cc_create(weights.data(), COUNT, 2) == nullptr, "wrong layout accepted");
  const float original = weights[11];
  weights[11] = std::numeric_limits<float>::quiet_NaN();
  check(cc_create(weights.data(), COUNT, 1) == nullptr, "NaN weights accepted");
  weights[11] = original;
  void* row = cc_create(weights.data(), COUNT, 0);
  void* packed = cc_create(weights.data(), COUNT, 1);
  check(row && packed, "context creation failed");
  std::array<double, H> reference0{}, reference1{};
  std::array<float, 112> features{};
  std::array<double, 112> reference_features{};
  std::array<float, 2> row_output{}, packed_output{};
  std::array<float, H> hidden0{}, hidden1{};
  double worst = 0.0;
  for (int repeat = 0; repeat < 2; ++repeat) {
    check(cc_reset(row) == 0 && cc_reset(packed) == 0, "reset failed");
    reference0.fill(0.0);
    reference1.fill(0.0);
    check(cc_hidden(packed, hidden0.data(), H, hidden1.data(), H) == 0, "state read failed");
    check(std::all_of(hidden0.begin(), hidden0.end(), [](float x) { return x == 0.0f; }), "reset left stale state");
    for (int step = 0; step < 256; ++step) {
      for (size_t j = 0; j < features.size(); ++j) {
        features[j] = static_cast<float>(2.0 * std::sin((step + 1) * 0.13 + j * 0.21));
        reference_features[j] = features[j];
      }
      check(cc_step(row, features.data(), features.size(), row_output.data(), 2) == 0, "row inference failed");
      check(cc_step(packed, features.data(), features.size(), packed_output.data(), 2) == 0, "packed inference failed");
      check(row_output == packed_output, "row/packed FP32 results differ");
      reference_gru(weights, reference_features.data(), 112, W0, R0, B0, reference0);
      reference_gru(weights, reference0.data(), H, W1, R1, B1, reference1);
      check(cc_hidden(packed, hidden0.data(), H, hidden1.data(), H) == 0, "state read failed");
      for (size_t j = 0; j < H; ++j) {
        worst = std::max(worst, std::abs(double(hidden0[j]) - reference0[j]));
        worst = std::max(worst, std::abs(double(hidden1[j]) - reference1[j]));
      }
      for (size_t target = 0; target < 2; ++target) {
        double expected = weights[HEAD_B + target];
        for (size_t j = 0; j < H; ++j)
          expected += reference1[j] * weights[HEAD_W + 2 * j + target];
        worst = std::max(worst, std::abs(double(packed_output[target]) - expected));
      }
      check(worst < 1e-5, "native differs from independent double reference");
    }
  }
  check(cc_step(packed, features.data(), 111, packed_output.data(), 2) != 0, "invalid feature count accepted");
  features[0] = std::numeric_limits<float>::infinity();
  check(cc_step(packed, features.data(), 112, packed_output.data(), 2) != 0, "nonfinite input accepted");
  check(cc_hidden(packed, hidden0.data(), H - 1, hidden1.data(), H) != 0, "invalid state count accepted");
  cc_destroy(row);
  cc_destroy(packed);
  cc_destroy(nullptr);
  check(cc_reset(nullptr) != 0, "null reset accepted");
  std::cout << "native reference/layout/reset/buffer checks passed; max_abs_error=" << worst << '\n';
}
}  // namespace

int main() {
  try {
    run();
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "native test failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
