#ifndef CONNECTOME_NATIVE_H
#define CONNECTOME_NATIVE_H

#include <stddef.h>

#if defined(_WIN32)
#define CC_EXPORT __declspec(dllexport)
#else
#define CC_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ABI 1. Flat FP32 weights, in order:
 * W0[384,112], R0[384,128], B0[768], W1[384,128], R1[384,128],
 * B1[768], head_W[128,2], head_B[2]. GRU gate order is z,r,h;
 * B = [Wbz,Wbr,Wbh,Rbz,Rbr,Rbh]. All source matrices are row-major.
 * layout 0 keeps source rows; layout 1 packs matrices input-major.
 * A context is stateful and must not be used concurrently.
 */
CC_EXPORT int cc_abi_version(void);
CC_EXPORT void* cc_create(const float* weights, size_t count, int layout);
CC_EXPORT int cc_reset(void* context);
CC_EXPORT int cc_step(void* context, const float* features, size_t feature_count,
                      float* prediction, size_t prediction_count);
CC_EXPORT int cc_hidden(void* context, float* hidden0, size_t count0,
                        float* hidden1, size_t count1);
CC_EXPORT void cc_destroy(void* context);
CC_EXPORT const char* cc_last_error(void);

#ifdef __cplusplus
}
#endif
#endif
