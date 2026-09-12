# Linux ARM64 inference checks

The dependency image uses Python 3.11 on Debian Bookworm, the project's pinned
NumPy/ONNX Runtime/PyArrow requirements, ONNX 1.17.0, pytest 8.3.5, and the system
C++17 compiler. It deliberately excludes Torch and all training packages.
The Docker build context contains only the Dockerfile and requirements;
datasets, virtual environments, model artifacts, and source are not copied.

Build dependencies once, then run verification:

```sh
python scripts/run_linux_checks.py --build --phase prepare
python scripts/run_linux_checks.py --phase benchmark --smoke-count 1 --repetitions 1
python scripts/run_linux_checks.py --phase benchmark --smoke-count 32 --repetitions 3
```

Docker Desktop must be running on macOS. The image build needs network access
to acquire dependencies. Every verification/benchmark container then runs with
`--network none`, a one-CPU quota, exactly 16,000,000,000 bytes RAM and no additional swap. Source and
datasets are mounted read-only at `/workspace`; only
`artifacts/linux_arm64` is writable, along with temporary container storage.
The runner verifies cgroup CPU/memory limits and the absence of active
non-loopback network interfaces inside the container, and records their observed
values. Dormant kernel tunnel placeholders can exist with Docker's `none`
network and are recorded separately from active interfaces.
Use `--image sha256:...` with the immutable image ID saved in a host report to
rerun the exact dependency image. This also avoids reliance on a mutable tag;
the final recorded checks used the saved image ID after Docker's tag lookup
became inconsistent while the image itself remained available.

The prepare phase builds the native `.so`, runs its independent reference and
ASan/UBSan checks, runs inference/metric/export tests, and compares baseline,
simplified ONNX, and native outputs over every row of validation group 0.
Training tests skip because Torch is absent; those tests must pass separately
in the host training environment.

Benchmark groups come directly from the manifest's fixed `splits.smoke`
membership. Host-specific data paths stored in that manifest are not used
inside Docker. Each repetition processes complete 20,000-row sequences through
the callback, including warm-up and sequence resets. Startup is separate and
the sequence arrays are preloaded, so these timings exclude Parquet loading.
Original graph modes are `reference`, `run`, `iobinding`, `ortvaluevector`,
`native_row`, and `native_packed`. Simplified graph modes are the four ONNX
options; native weights are unchanged by graph simplification.

The API-study mode named `reference` uses the new `StreamingPredictor` with
the original ONNX `run(None, ...)` call. It is **not** the published callback.
After freezing a candidate configuration, compare the actual
`baseline/solution.py` class against the configured candidate with strict input
validation enabled:

```sh
python scripts/run_linux_checks.py --phase tests
python scripts/run_linux_checks.py --phase solutions --candidate-config artifacts/candidate/config.json
python scripts/run_linux_checks.py --phase solutions --candidate-config artifacts/candidate/config.json --affinity
python scripts/run_linux_checks.py --phase stream --candidate-config artifacts/candidate/config.json --stream-count 256
python scripts/run_linux_checks.py --phase zip --archive artifacts/candidate/submission.zip
```

Replace the example configuration path with the selected candidate's actual
file. The comparison requires `strict_inputs: true`, includes the configured
postprocessing, and alternates baseline/candidate ordering across three
repetitions of the 32 complete smoke sequences. The optional affinity variant
pins the process to one CPU from its allowed CPU set; it never assumes CPU 0
is available. No hardware cache counters are collected. The image does not
install `perf`, and the reports explicitly record this limitation.

The `stream` phase reads one complete sequence at a time from the fixed 256
screen groups, runs the exact strict production configuration, and scores it.
Its elapsed time includes startup, Parquet reads, callbacks, output validation,
and scoring. It reports process peak RSS and uses no prediction cache or bulk
sequence graph. The `zip` phase extracts the selected archive and exercises it
in a fresh isolated Python process using real input rows, checking warm-up,
sequence resets, finite predictions, and independent output ownership.

For a short optional affinity API comparison use `--phase benchmark
--smoke-count 1 --repetitions 3 --modes iobinding ortvaluevector --affinity`.
Historical reports created before the decimal-memory correction retain their
observed 17,179,869,184-byte (16 GiB) limits. New runs use exactly 16 GB.

Logs, image/container identifiers, compiler flags/hashes, parity reports,
per-mode timing reports, and summaries are saved under
`artifacts/linux_arm64`. Reports are saved after each benchmark mode so an
interrupted long run retains completed evidence. Native remains unpromoted
even when it runs successfully: correctness/score and throughput gates are
separate decisions.

These are **Linux ARM64 measurements on this Mac's shared physical host**.
The CPU quota limits container consumption but does not reserve an exclusive
physical core. Concurrent host work, thermal state, Docker virtualization,
and the unknown judge architecture affect transferability. Do not present
these measurements as Linux x86 or hidden-judge performance, or as proof of
the 60-minute test limit without the hidden test row count and hardware.
