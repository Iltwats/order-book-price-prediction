# Linux ARM64 measurements — 2026-09-12

The final candidate uses the original ONNX graph with the selected affine
readout and calibration folded into its head, I/O binding, strict input checks,
no runtime postprocessing, and no affinity pinning.

- Model SHA256: `d920986b8c42b256c652e9f695e6ca52af1be7c3ecc88cd0596de41ed5653a45`
- Configuration SHA256: `4f3c27303d5d35b5fdc4a32e0365d084731654e1e50511118d0481d8e29b3904`
- Image ID: `sha256:be7a836b27b3eecd1c1e83b0675ff189c520678059526dbf77d90bbe20d4493f`
- Python 3.11.16, ONNX Runtime 1.23.2, NumPy 2.2.6; Linux aarch64, glibc 2.36.

Final runs verified one CPU quota (`cpu.max=100000 100000`), exactly
16,000,000,000 bytes RAM, no additional swap, no active non-loopback network
interfaces, and no Torch installation. Source/data were mounted read-only;
only the report directory and temporary storage were writable.

## Actual callback comparison

The reference is the actual `baseline/solution.py` class. Each of three
repetitions processed the same 32 complete 20,000-row smoke sequences through
the public callback, including the warm-up rows and normal sequence resets.
The candidate retained strict input validation. Data loading and startup were
outside the timed callback region; baseline/candidate order alternated.

| Callback | Median seconds / 640,000 rows | Median callbacks/s |
| --- | ---: | ---: |
| Published baseline | 18.305398 | 34,962.364 |
| Selected strict candidate | 16.714034 | 38,291.174 |

Measured candidate throughput was 1.095211× the published baseline (+9.52%).
Candidate repetition durations were 16.829556, 16.703286 and 16.714034 seconds;
baseline durations were 18.434011, 18.305398 and 18.182348 seconds. The preloaded
process peak RSS was 638,365,696 bytes. Full phase time was 106.195 seconds.
Evidence: `solutions_32seq_unpinned.json`, `summary_solutions_32seq.json`, and
`host_solutions_32seq.json`.

## Sustained streamed evaluation

The exact production configuration processed 256 complete fixed screen
sequences, 5,120,000 rows, with one sequence loaded at a time and no prediction
cache. Elapsed time includes startup, Parquet reads, strict callbacks, output
validation and official scoring.

- Elapsed: 141.246490 seconds; 36,248.688 rows/s.
- Process peak RSS: 370,044,928 bytes, including runtime/dataset/scorer overhead.
- Weighted Pearson: 0.6389160985024305.
- t0: 0.6384605868532468; t1: 0.6393716101516141.

Evidence: `stream_256seq.json`, `summary_stream_256seq.json`, and
`host_stream_256seq.json`. This is a screen score, not a confirmation score.

## Extracted ZIP

The fresh isolated Python process passed 768 real-input callbacks, warm-up
checks, two resets, finite outputs and independent output ownership. Network
access was disabled both by the container and by the child-process socket
guard. Verification took 0.156899 seconds.

- Archive: `dist/candidate.zip`, 722,475 bytes.
- SHA256: `9d8862659bbe823532879708fa2765ac023d2cae8984501919e1293a022c8de4`.
- Evidence: `fresh_zip.json` and `host_zip_32seq.json` (the generic runner's
  filename includes its default smoke count; the ZIP check itself used 768 rows).

## Experimental native backend

The native backend remains unpromoted. Its explicit 64-byte-aligned rebuild
passed independent reference, reset, invalid-buffer, ASan and UBSan checks.
The associated inference suite passed 79 tests, with the Torch training module
skipped because it is absent from the image.

Over all 20,000 rows of validation group 0, both native layouts were bitwise
equal to each other, but failed the `rtol=1e-5, atol=1e-6` ORT parity gate:
maximum prediction error 7.390975952e-5; maximum hidden-state error
6.312131882e-5. There were 15,066 prediction-row tolerance failures and 39,673
hidden-array comparison failures. Alignment did not change these errors.
The simplified ONNX graph was bitwise equal to the original in that test.

The aligned 1-sequence × 3 diagnostic measured native row/packed medians of
11,251.177 / 32,252.474 callbacks/s. This short diagnostic and the earlier
32-sequence matrix do not support native promotion. Aligned evidence is under
`align64/`; the earlier Linux matrix is retained in this directory. Host
artifacts/source from before alignment are retained in
`../native_before_align64`.

## Measurement limits

These are Linux ARM64 measurements on this Mac's shared physical host through
Docker, not Linux x86 or hidden-judge measurements. Concurrent host work was
allowed, so the CPU quota did not reserve an exclusive physical core. No cache
or hardware performance counters were collected; `perf` is not installed.
The optional one-sequence affinity diagnostic was variable and did not justify
changing the final no-affinity configuration.

Historical API-matrix reports retain the observed 17,179,869,184-byte (16 GiB)
limit used before the decimal-memory correction. Final production checks and
the aligned rebuild used the exact 16 GB limit above. The API-study mode named
`reference` is the new wrapper around `run(None, ...)`, not the published class;
only `solutions_32seq_unpinned.json` is the actual published-callback comparison.

The hidden test row count and CPU architecture remain unknown. These checks
do not prove completion within the hidden judge's 60-minute limit.
