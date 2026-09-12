# Experimental native GRU backend

This C++17 backend evaluates the supplied two-layer, 128-unit FP32 GRU with
ONNX gate order `z,r,h`, `linear_before_reset=1`, and a 128-by-2 linear head.
It has a C ABI and fixed workspaces: `cc_step` performs no heap allocation.
The weights, each hidden state, and each scratch buffer have explicit
`alignas(64)` alignment, verified by compile-time member-offset assertions.
C++17 aligned allocation preserves the context's 64-byte alignment.
Its two layouts use the same weights/equations: `row` keeps ONNX weight rows;
`packed` transposes each gate matrix once at construction for contiguous
output-lane accumulation. No fast-math or local-ISA compiler flags are used.

It remains a **candidate**, not an automatically promoted submission backend.
Floating-point reductions and libm activations can differ from ONNX Runtime;
verify hidden-state/prediction parity across complete sequences, full validation
WP, and one-core Linux throughput before deployment.

The current native candidate failed the specified `rtol=1e-5, atol=1e-6`
prediction/hidden-state parity gate on a complete real sequence. The aligned
rebuild has the same measured errors and remains unpromoted. Linux ARM64
evidence is in `artifacts/linux_arm64/align64`; the earlier unaligned Linux
matrix remains in `artifacts/linux_arm64`. Host artifacts and source from before
the alignment correction are preserved in `artifacts/native_before_align64`.

From the project root:

```sh
python native/build.py --test --sanitize
```

The system compiler creates `artifacts/native/libconnectome_native.dylib` on
macOS or `.so` on Linux. The independent C++ reference test compares both layouts
against double-precision recurrence, repeats sequence resets, and checks invalid
buffers/nonfinite inputs. The sanitizer executable enables AddressSanitizer and
UndefinedBehaviorSanitizer. The build manifest records compiler, flags, hashes,
architecture, and check results. A macOS ARM library cannot be used by the Linux
judge: rebuild on matching Linux architecture and repeat runtime checks there.

Offline weight preparation uses `connectome.model_io.export_native_weights`.
The runtime `connectome.native.NativeBackend(weights_path, library_path,
layout="packed")` requires only NumPy and the standard library. `step` returns
an owned two-element float32 array; hidden-state exports are copies. Instances
are stateful and must not be used concurrently.

The `.npy` weight format is one contiguous 192,258-element float32 array in this
order (source matrices are row-major):

| Key | Shape | Start offset |
| --- | --- | ---: |
| w0 | 384 × 112 | 0 |
| r0 | 384 × 128 | 43,008 |
| b0 | 768 | 92,160 |
| w1 | 384 × 128 | 92,928 |
| r1 | 384 × 128 | 142,080 |
| b1 | 768 | 191,232 |
| head_w | 128 × 2 | 192,000 |
| head_b | 2 | 192,256 |

Each bias contains `[Wbz, Wbr, Wbh, Rbz, Rbr, Rbh]`. The recurrent candidate
bias stays inside the reset-gate multiplication. Concatenated 256-unit heads,
alternate activations, dynamic input layouts, and external ONNX tensor files
are deliberately excluded from this native format.
