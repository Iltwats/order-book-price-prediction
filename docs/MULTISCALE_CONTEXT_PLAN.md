# Discussion and plan for improving the GRU

Discussion and research date: 2026-10-24. Status: proposed experiments; no new model has been trained or scored during this discussion.

We can expose recent changes and slower summaries together and train a model to combine them. The first test with our current GRU should isolate whether a longer backward window helps it learn temporal dependencies. An explicit branch with several timescales is a separate modeling experiment, with additional inference and export work.

## How the discussion led to this plan

We started with the idea that the model receives a price feed, recognizes historical patterns and predicts future movement. The discussion made that idea more precise: a pattern is useful when it helps predict the supplied targets on sequences that were excluded from fitting, and its meaning can depend on other market features.

| Question or intuition | What we established | Consequence for the next step |
| --- | --- | --- |
| If a price rises, falls and repeats an earlier pattern, could what followed repeat too? | This is a testable predictive relationship. Similar price paths can occur under different volumes, order books and cross-instrument conditions. | Learn interactions among the available inputs and assess them on held-out sequences. A matching chart shape alone is insufficient evidence. |
| Would different volume justify different predictions for the same price pattern? | The user's expectation was yes: volume changes the context. The model can learn such conditional relationships through its weights. | Preserve the full market snapshot when testing temporal features; do not reduce the task to price history alone. |
| What exactly are `t0` and `t1`? | The live organizer documentation explicitly withholds their construction. Both concern `i0`; neither their names nor common forecasting conventions establish their horizons, units or sign conventions. | Learn the supplied numeric labels directly. Do not relabel them as short-term and long-term returns or replace them with guessed price targets. |
| How does a model learn which inputs matter? | Training evaluates predictions against targets through a loss, computes gradients and adjusts trainable weights. A GRU also learns how to update its hidden state. Held-out comparisons and input ablations test whether those learned relationships are useful. | Keep the training objective aligned with Global WP and use controlled comparisons. Weight magnitudes alone do not prove feature importance. |
| Should we provide a short history or a long history? | The user expected both to matter: recent moves can suggest an immediate pattern, while broader history can establish recurring relationships and context. | Test how the model learns across time, then test explicit fast and slow summaries together. |
| Does longer history mean looking back six months for each prediction? | Learning recurring relationships across the training corpus differs from retaining observations for one prediction. Calendar timestamps are absent and sequences are independent. | Measure context in event rows and reset state between sequences. Do not invent month-based windows or stitch sequence IDs into a timeline. |
| Is our GRU limited to 256 rows of memory? | No. It carries state through a sequence; 256 is its current backward window. The lower GRU is frozen in the prepared training method. | Change the backward window in a separate experiment to isolate temporal credit assignment. This does not change the amount of forward history available. |
| Have others combined recent and longer signals? | The original papers and author code below offer concrete mechanisms. Our earlier dynamics branch was stopped on runtime before any predictive test. | Borrow mechanisms as hypotheses and account for the prior runtime problem. Neither published gains nor the old timing failure settle their predictive value here. |

This led to a staged proposal: preserve the prepared coverage experiment, compare 256-row and 1,024-row backward windows, and then assess an explicit combination of fast and slow summaries if it remains a useful challenger. The specific widths, half-lives and continuation threshold below are proposed settings. The discussion does not establish them as optimal or as an implemented improvement.

## Which baseline we mean

The supplied baseline is `baseline/baseline.onnx`. The strongest existing model in the saved development rescore is `correlation1024_seed20260912`, at **0.672091945 Global WP** on 1,400 development sequences. It is the starting model and unchanged teacher in the prepared GPU plan. Those are existing results, not outcomes of the proposed context experiments. [Saved model comparison](GLOBAL_WP_RESULTS.md)

The root `submission_config.json` still selects the earlier calibrated `artifacts/selected/model.onnx`. Before a new context experiment, bind its comparator to an exact checkpoint and configuration after the prepared comparison is resolved. Packaging with the root config does not automatically select the development winner. [Submission configuration](../submission_config.json)

## What longer history means here

There are two useful meanings of historical information. Training on many examples lets the model learn recurring relationships into its weights. At prediction time, the model also uses observations from the current sequence to identify the present situation. Training examples may span different market periods even when the current observation window is short.

The supplied columns contain no calendar timestamps. We cannot identify six-month trends, match calendar dates, or join sequences into a continuous history. Context must be measured in event rows, with state reset between independent 20,000-row sequences. The first required prediction has only 100 observed rows. Both targets concern instrument `i0`; their construction and horizons remain undisclosed. [Organizer data documentation](https://wundernn.io/connectome/docs/data_overview)

A bounded check during the discussion read eight previously used fitting sequences, with 159,208 post-warm-up rows, and left the reserved TRAIN groups untouched. The sampled within-sequence target correlations ranged from approximately -0.91 to -0.60. Numerical spacings varied across sequences, at approximately 0.125, 0.469 and 0.657. These observations do not identify a target formula or prove a tick-count interpretation. They correct the earlier assumption that 0.657 was a universal spacing. The sampled TRAIN row groups were `2309, 9078, 4507, 3165, 3508, 2523, 1866, 5932`.

## What published work supports

Four targeted Exa searches returned 14 results. The retained evidence covers four original works, rather than 14 independent methods. Author repositories were checked; the LSTNet and MLPLOB model implementations were also read. These methods have not been reproduced locally.

| Original work | Relevant mechanism or finding | What transfers and what differs |
| --- | --- | --- |
| [DeepLOB, 2019](https://arxiv.org/html/1808.03668v6), with [author code](https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books) | Temporal filters of different sizes feed recurrent memory, allowing local dynamics and longer dependencies to be modeled together. | Supports testing several timescales. Its ordered book representation and movement-classification labels differ from our anonymized columns and two regression targets. We cannot copy its spatial ordering assumptions. |
| [LSTNet, 2018](https://arxiv.org/abs/1703.07015), with [author GRU implementation](https://github.com/laiguokun/LSTNet/blob/master/models/LSTNet.py) | Convolutional features feed an ordinary GRU and a skip-connected GRU; their outputs are combined with an autoregressive component. | A concrete example of combining local and longer temporal paths. Its general forecasting benchmarks and periodic skip structure do not establish useful periods in our event stream. The released environment is old and is not an installation recipe for this project. |
| [Kolm, Turiel and Westray, 2023](https://onlinelibrary.wiley.com/doi/full/10.1111/mafi.12413) | In their multi-horizon return study, order-flow inputs helped simple recurrent models compete with more complex architectures using raw book states. | Supports a representation hypothesis: changes may be easier to learn from than levels alone. Our column meanings do not support assuming standard level-by-level order-flow formulas are valid. Same-column changes are a narrower candidate. |
| [TLOB, 2025](https://arxiv.org/abs/2502.15757), with [author repository](https://github.com/LeonardoBerti00/TLOB) and [MLPLOB code](https://github.com/LeonardoBerti00/TLOB/blob/main/models/mlplob.py) | The authors compare attention models with a competitive MLP that mixes features and time. | Supports keeping a small learned combination as a serious control. Their classification scores do not predict a Global WP gain, and their windowed implementation is not a streaming CPU deployment measurement. |

The common lesson is to test complementary temporal representations and their inputs. None of these sources establishes that our targets need months of context, that a transformer is necessary, or that a particular window will improve our score.

## What our existing work establishes

The current model already carries forward state through an entire sequence. The 256-row setting limits backward credit assignment: each Torch chunk starts from a detached exact ONNX boundary state. It does not reset inference memory. The current runner freezes the lower GRU and updates the upper GRU and head. [Chunk implementation](../connectome/partial_gru.py), [training runner](../connectome/gpu_training.py)

The prepared v2.1 plan asks a different question: four passes over 1,024 sequences versus one pass over 4,096 sequences, with 512 updates and a 256-row backward window in both arms. Keep that declared comparison intact. Longer windows require a separate plan. [Current GPU plan](GPU_TRAINING.md)

An earlier dynamics branch used projected inputs, one-step differences and an EMA8 deviation. It was stopped before fitting because its untrained nonzero fixture increased callback time by 30.43% against the incumbent in a Linux ARM64 benchmark. It supplied no predictive-quality result. Its historical gate required matching the original baseline's speed; the current competition limit is an absolute 60 minutes on x86-64. The old timing is useful engineering evidence, but cannot certify current feasibility or reject the statistical hypothesis. [Benchmark receipt](../artifacts/round2/causal_linux8/benchmark_report.json)

The archived design and interpretation remain available with:

```sh
git show pre-cleanup:docs/CAUSAL_BRANCH_DESIGN.md
git show pre-cleanup:docs/CAUSAL_BRANCH_RESULTS.md
```

## First hypothesis and test with the current GRU

**Hypothesis:** a 256-row backward window prevents some useful longer dependencies from being learned by the upper GRU. A longer window can improve held-out Global WP while retaining the same inference architecture.

After the prepared coverage comparison, declare a separate pair of arms. Initialize both from the same selected comparator, with the same unchanged teacher. Reuse the original 1,024 fitting groups, excluding the 512 reserved TRAIN groups. Use one pass, batch size eight and 128 complete-sequence optimizer updates in each arm. Keep group order, seed, pooled objective, learning rate, regularization and postprocessing identical. Change only the backward window: **256 versus 1,024 rows**. Both arms retain full forward state.

Qualify raw Torch/ONNX local discrepancies and CPU/device gradients at each window length before fitting. Longer chunks may fail existing numerical limits or consume too much training memory; either result is an engineering limitation, not a finding about predictive signal. Use at most two optimizer updates for the timing pilot, then estimate the complete run before committing compute. A device and spending cap remain to be set before any paid launch.

This comparison tests learning across time. It does not measure an optimal input lookback, and success would not establish that 1,024 is globally optimal. Failure would not rule out better temporal features: the lower GRU remains frozen.

## Second hypothesis and an explicit combination of timescales

**Hypothesis:** the incumbent compresses recent changes and slower conditions into its state imperfectly. Providing separate causal summaries will add predictive information beyond an equally sized readout of current inputs and GRU state.

Keep the pretrained GRU unchanged and add a small residual predictor. A concrete first design is an eight-dimensional learned projection `u` of the 112 current features, two causal exponential moving averages with half-lives of **16 and 1,024 event rows**, and a `160 -> 16 -> 2` residual MLP. Its inputs are the upper GRU state (128 values), current projection (8), one-step change (8), deviation from the fast average (8), and fast-minus-slow average (8). These timescales and widths are proposed starting settings, not measured optima.

The learned head can use recent movement differently depending on slower conditions; an explicit attention mechanism is not required to express that interaction. Initialize only its final residual layer to zero so the initial predictions match the incumbent. Update averages on every row, initialize them from the first observed projection, and reset all auxiliary state at sequence boundaries. Fit input normalization only on the declared fitting sequences. Never use previous true targets as inputs.

For half-life `H`, update `m = m + (1 - 2**(-1/H)) * (u - m)`. This is a decaying summary, not a complete H-row window. At early steps it contains only the history actually observed.

Compare four arms with the same residual-network dimensions, seed and training allocation:

| Arm | Inputs in addition to current projection and GRU state |
| --- | --- |
| Capacity control | Fast and slow feature slots set to zero |
| Fast | One-step change and deviation from the fast average |
| Slow | Fast-minus-slow average |
| Both | All fast and slow summaries |

Also report the unchanged incumbent. If Both improves on the capacity control but does not improve on Fast or Slow, we have not shown that combining timescales adds value. Removing inputs can establish incremental predictive usefulness under this protocol; learned weight or gate magnitudes alone cannot.

The runtime currently expects only two GRU states, and the exporter only replaces existing tensors. A branch needs explicit auxiliary-state, export, evaluation and packaging support. Start with one unscored operator profile and nonzero-fixture callback benchmark informed by the earlier slowdown. Verify causality, sequence resets, short-history behavior and zero-residual parity before training. [Runtime](../connectome/runtime.py), [export contract](../connectome/conversion_audit.py), [packaging](../connectome/packaging.py)

**CPU probe result (2026-10-24/25).** Before building the branch, we tested
whether the proposed timescales carry signal beyond the incumbent's outputs.
Activity and output summaries with half-lives of 16 and 1,024 rows, plus `a0`–`a7`,
were fitted out of fold on cached development predictions. Every fast, slow and
combined arm scored at or below its capacity control. The probe used no GRU state
and no end-to-end training, so it does not rule the hypothesis out, but it found
nothing to justify the branch's runtime and export work now. The outputs-only
control itself gained, which led to the scored-row output calibration. Details:
[experiment log](EXPERIMENT_LOG.md).

## How to judge the experiments

Use the same fixed sequence groups for every paired comparison. Permit one 256-sequence screen per completed arm; advance the predeclared leading comparison to the 1,400-sequence development split, then repeat that comparison with a second training seed. Keep the already-consumed confirmation ledger closed and the reserved TRAIN groups sealed. The supplied pretrained model's earlier exposure to that reserve is unknown, so it cannot be called an independently unseen set for the incumbent.

A proposed development continuation rule is an average Global WP gain of at least **0.0005**, positive deltas in both seeds, and a paired sequence-cluster 95% interval above zero for the mean paired gain across seeds. Also report each target and performance outside the visible scoring mask. This threshold is a project decision proposed for this experiment, not a guarantee of public or private improvement. Reusing development sequences does not make them fresh acceptance data.

For a branch candidate, measure complete callbacks and streamed end-to-end inference on representative x86-64 hardware with one CPU thread. Report startup, memory, package size and the estimated 39.4-million-row runtime with headroom under 60 minutes. A timing extrapolation remains an estimate, and ARM64 results are not judge-runtime proof. Do not repeatedly retune timescales, checkpoints or calibration against the same screen under this small comparison.

## Implementation sequence and decision points

| Step | Deliverable and decision | Status |
| --- | --- | --- |
| Complete the existing coverage comparison | Qualify CUDA, measure a bounded pilot, then run the declared repeat-versus-coverage arms without changing their frozen settings. Use the declared comparison to select the starting point for the next experiment. | Plan prepared and CPU qualification recorded; CUDA and fitting pending |
| Freeze a separate context experiment | Bind the exact comparator, teacher, fitting groups, order, seed, loss, two window sizes, update count, evaluation schedule and compute cap. Preserve the reserve and the existing plan. | Pending |
| Qualify both backward windows | Check raw Torch/ONNX discrepancies, CPU/device gradients and training time/memory for 256 and 1,024 rows. Stop a failing or unaffordable arm before a full run. | Pending |
| Fit and compare the current GRU arms | Produce both checkpoints, aligned held-out predictions, per-target and pooled Global WP, and paired sequence-cluster uncertainty. Repeat the leading comparison with a second seed under the stated rule. | Pending |
| Decide whether a separate temporal branch is warranted | A longer-window gain supports improving the existing training method. No gain leaves the representation hypothesis open because the lower encoder stayed frozen. Proceed with the branch only under its own declared comparison and budget. | Pending |
| Test the explicit combination if pursued | Pass engineering checks, then compare the capacity control, Fast, Slow and Both arms. Keep a combination only if its incremental predictive value justifies its measured cost. | Pending |
| Verify the chosen inference artifact | Evaluate the exported model with its actual state handling and postprocessing; check x86-64 runtime, memory, package size and offline behavior before changing submission selection. | Pending |

The next implementation milestone is an independently frozen and qualified context comparison after the prepared coverage experiment. A model becomes an improvement only after the corresponding predictive and deployment checks pass.

## Validation record

| Item | Status |
| --- | --- |
| Discussion, reasoning and proposed implementation sequence recorded | Complete |
| Primary sources and public author repositories inspected | Complete |
| Current state, backward-window implementation and archived branch outcome checked | Complete |
| New source links and local documentation links checked | Complete |
| New candidate implementation, CUDA qualification, training and scores | Pending |
| Candidate x86-64 runtime, export and submission packaging | Pending |
| CPU probe of fast/slow summaries on cached outputs ([log](EXPERIMENT_LOG.md)) | Complete: no added signal |

This record captures the discussion and proposed experiments. Model code, frozen plans, split manifests and submission selection were not changed. The README links here and corrects the target-spacing summary using the bounded check above.
