# What we are doing

Our target is at least 0.75 on the live leaderboard. The latest reported live score remains 0.6083. We have improved local development performance, but that is a different dataset. We cannot honestly claim that the current changes will achieve the live target.

Yes, we are still using the supplied pretrained GRU. Its first recurrent layer stays unchanged. We train its second layer and prediction head using only the supplied training data. The final model keeps the same architecture, input features, memory state and number of inference operations.

The main improvement comes from changing what training optimizes. Ordinary weighted squared error reduced validation performance. Training directly toward clipped, weighted correlation worked better. We first tested 256 training sequences, then expanded to 1,024 sequences with two independently shuffled training runs. Both larger runs improved the fixed screening score.

We selected the stronger run and evaluated it on all 1,400 development sequences. Its score was 0.630019, compared with 0.629256 for the incumbent. The measured gain was +0.000764. Previously inspected confirmation data cannot provide fresh independent evidence, and development selection introduces uncertainty that bootstrap intervals do not remove.

Execution still uses Python with ONNX Runtime performing the CPU calculations. The larger candidate processed about 38,860 callbacks per second in Linux benchmarks, approximately thirteen percent faster than the original baseline. Sustained execution processed 5.12 million rows in 140 seconds and used roughly 370 MB. The judge's hardware and total workload remain unknown.

The new ZIP contains inference code and trained weights, with datasets excluded. Offline verification passed. We retain earlier archives and commit changes in steps. Next, a bounded training diagnostic should test whether gradients need longer history or whether the frozen first layer limits learning. Those measurements will guide another declared experiment without changing the current model or treating the 0.75 target as achieved.
