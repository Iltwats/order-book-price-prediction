# Dynamics branch: Linux runtime gate failed

Measured 2026-09-13. The nonzero dynamics prototype delivered **29,472.36 callbacks/s**, compared with **35,902.53** for the actual published baseline and **38,439.61** for the selected incumbent. Its throughput was **17.91% below the baseline** and **23.33% below the incumbent**. It was slower than the baseline in all three repetitions, so it fails the current requirement that a new candidate be no slower than the original baseline. Keep this prototype out of fitting, score selection and deployment. Prioritize the validated upper-GRU training work. [Complete benchmark report](../artifacts/round2/causal_linux8/benchmark_report.json)

These are throughput reductions. Expressed as time per callback, the increases are **21.82% versus the baseline** and **30.43% versus the incumbent**; those percentages have different denominators.

## What was measured

The test used eight complete 20,000-row smoke sequences: `1562, 595, 750, 1192, 561, 1128, 584, 658`. Each arm processed all **160,000 rows per repetition**, including warm-up, for three unprofiled repetitions. Across four arms, that is 1,920,000 timed callbacks. The data loader read sequence metadata and features only; it did not request targets or `is_scored`. No normalization statistics were fitted and no prediction score was computed. This engineering probe consumed no candidate screening or full-development slot. [Frozen specification](../artifacts/round2/causal_linux8/specification.json), [benchmark implementation](../scripts/benchmark_causal_branch.py)

The baseline arm instantiated the actual no-argument `PredictionModel` from `baseline/solution.py`, using its existing checks. The incumbent and both branch fixtures used the current strict `StreamingPredictor` with CPU I/O binding. Both branch fixtures preserve the selected model's original nodes and initializers and append the same `112→16`, difference/EMA8, `48→16→2` dynamics path with 33 auxiliary FP32 values. The zero fixture has a zero final residual layer; the nonzero fixture has deterministic untrained output weights so the residual MLP is not represented solely by a zero-output fixture. No operator profile was collected, so this does not establish exactly which graph nodes ORT executed. [Prototype](../connectome/causal_branch.py), [design](CAUSAL_BRANCH_DESIGN.md)

Each arm/repetition ran in a fresh worker process. Timing included construction of each `DataPoint`, the public callback, its own validation, state updates and prediction copying. It excluded feature preload, model startup and a separate 256-row warm-up. Each timed sequence reset through the normal public interface. Prediction ownership was checked after later callbacks. These timings are not streamed end-to-end evaluation times or startup-inclusive submission times.

The three orders were baseline/incumbent/zero/nonzero; incumbent/zero/nonzero/baseline; and zero/nonzero/baseline/incumbent. Each arm occupied three distinct positions. Four arms with three repetitions cannot achieve perfect four-position balance, and the host was not exclusive.

## Environment and correctness

The recorded environment was `Linux-6.10.14-linuxkit-aarch64-with-glibc2.36`, architecture `aarch64`, Python **3.11.16**, NumPy **2.2.6**, and ONNX Runtime **1.23.2**. All arms used `CPUExecutionProvider` with one intra-op and one inter-op thread. The benchmark set the common numerical-runtime thread limits to one and enabled no CPU affinity.

The enforced cgroup settings were `cpu.max = 100000 100000`, `memory.max = 16000000000` bytes and `memory.swap.max = 0`. The allowed CPU set was `0-3`; the one-CPU quota did not assign an exclusive core. There were no active non-loopback network interfaces. Dormant tunnel interfaces were present but down. Hardware performance/cache counters were not collected. The report records `host_exclusive=false` and `judge_runtime_verified=false`; hidden-judge workload and runtime remain unknown.

Before timing, a separate worker compared the zero fixture with the selected incumbent on **all 160,000 real rows**. Raw predictions, including warm-up outputs, and both GRU states were **bitwise identical on every row**. First-observation filter resets and retained-output ownership passed. The raw prediction stream hash was `0807d6387fab36168b995f8bfe01ebe89270efef09c30e18cb2b0832700bc374`. This proves the zero branch preserved the incumbent on these sequences; it does not establish the nonzero fixture's prediction quality. [Parity receipt](../artifacts/round2/causal_linux8/zero_parity.json)

## Repetitions, startup and memory

Times below are rounded for display; the linked JSON preserves the original values. Peak RSS is the process high-water mark sampled immediately after callbacks, including imports, feature preload, model startup and warm-up. It is not model-only memory. Later environment snapshots are also retained in the raw report.

| Arm | Rep | Callback seconds | Callbacks/s | Startup ms | Peak RSS bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Actual baseline | 1 | 4.409284752 | 36,287.06 | 3.057167 | 406,319,104 |
| Actual baseline | 2 | 4.580273377 | 34,932.41 | 2.466542 | 406,437,888 |
| Actual baseline | 3 | 4.456510211 | 35,902.53 | 2.382083 | 406,441,984 |
| Selected incumbent | 1 | 4.373749585 | 36,581.88 | 2.998375 | 406,441,984 |
| Selected incumbent | 2 | 4.116883210 | 38,864.35 | 3.152208 | 406,409,216 |
| Selected incumbent | 3 | 4.162372835 | 38,439.61 | 3.449542 | 406,441,984 |
| Zero dynamics | 1 | 5.376904044 | 29,756.90 | 3.576375 | 406,773,760 |
| Zero dynamics | 2 | 5.890387961 | 27,162.90 | 3.396959 | 406,765,568 |
| Zero dynamics | 3 | 5.143233877 | 31,108.83 | 3.647625 | 406,720,512 |
| Nonzero dynamics | 1 | 5.546770877 | 28,845.61 | 4.580417 | 406,728,704 |
| Nonzero dynamics | 2 | 5.428815669 | 29,472.36 | 3.421291 | 406,716,416 |
| Nonzero dynamics | 3 | 5.370921210 | 29,790.05 | 3.358458 | 406,695,936 |

Median callback time was **27.853189 µs** for the baseline, **26.014830 µs** for the incumbent, **33.605650 µs** for the zero branch and **33.930098 µs** for the nonzero branch. The measured nonzero branch added **7.915268 µs** over the incumbent, while this run left only **1.838359 µs** before reaching the baseline's time. It would need to remove approximately **6.076909 µs per callback** to meet that baseline gate.

Median startup was 2.466542 ms, 3.152208 ms, 3.576375 ms and 3.421291 ms respectively. Preloading the eight sequences took 0.188295–0.242506 seconds per timed worker and was excluded. Post-callback peak RSS ranged from **406,319,104 to 406,773,760 bytes**, comfortably within the measured memory limit. Memory capacity and startup are not the failing gates here.

## Decision and smallest credible follow-up

The prototype adds only 2,592 analytical MACs per row, about 1.36% of the incumbent's recurrent/head arithmetic, yet increased complete-callback time by 30.43%. Small operator scheduling, intermediate tensors, state handling and dispatch are plausible contributors; this unprofiled run does not identify their individual costs or measure cache misses. The branch already shares one Python-to-ORT call with the GRUs, so merely combining Python calls is not a sufficient proposed fix.

Both fixtures used identity normalization with `normalization_folded=false`. ORT may eliminate subtract-zero/divide-one operations. Timing can therefore be optimistic relative to a trained branch with nonidentity normalization. Folding those future statistics cannot be assumed to recover the present deficit: the benchmark already offered identity operations for optimization, and prior synthetic checks showed that FP32 folding can exceed the strict auxiliary-state parity tolerance.

If branch engineering is resumed, the smallest credible step is one unscored operator profile of this fixed nonzero graph, followed by one focused reduction of its filter/elementwise operators. If profiling identifies branch dispatch as the bottleneck, a small native **branch-only** projection/filter/MLP kernel is a possible later experiment while retaining the verified ONNX GRU. It would introduce packaging/ABI work and possibly another call boundary; it has no presumed speed advantage. Verify any normalization folding or reassociation against the same numerical budget and rerun complete callbacks after the change. Do not train a branch, expand to 32 sequences, or start a layout sweep before a revised implementation meets the eight-sequence timing gate.

The current priority remains upper-GRU training, which can change learned representations while retaining the existing inference architecture. This probe supplied no evidence about achievable WP and no support for claiming the live ≥0.75 goal is reached. The released model and submission remain unchanged.

## Evidence identities

- Benchmark report SHA256: `36049a8d6e511103b615ef38204cb9a80ca35033b0f0653c901c3c4250d5a5eb`.
- Specification SHA256: `99153ed9f8b38210fec8161c16bff541257648c39bb690a0d6d6769fe016ccf4`.
- Selected model SHA256: `d920986b8c42b256c652e9f695e6ca52af1be7c3ecc88cd0596de41ed5653a45`.
- Zero fixture SHA256: `a34a14ea56d8077e2abde4c4bd3953de72176be76e7b226a17af6f0c3282c844`.
- Nonzero fixture SHA256: `79911bb7b70c45ba8afe1cce48da5c274430e97af6aacff1b9cdc03a94bbb4ca`.
