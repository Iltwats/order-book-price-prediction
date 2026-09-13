# What we are doing

Our goal is to reach at least 0.75 on the live leaderboard. That remains a target. We are using the supplied pretrained GRU, with our calibration. Its first recurrent layer stays unchanged. We allow the second layer and prediction head to learn useful changes from the supplied training data.

Numerical consistency was difficult. Torch and ONNX can produce slightly different answers even with identical weights. Some hidden states remember tiny rounding differences for thousands of rows. We preserved the failed checks and investigated their cause. We then built a separate training method that keeps the forward predictions and memory states inside ONNX Runtime.

Torch calculates learning directions over chunks of 256 rows. Every chunk begins with the matching ONNX state, and its visible values are corrected to the ONNX results. This keeps training tied to the deployed model while limiting accumulated numerical differences. We checked the local errors and compared gradients against higher precision calculations. All eight prescribed training sequences passed.

A pilot trained sixteen sequences in 13.16 seconds on this Mac. The first layer stayed unchanged, and all local numerical checks passed. Simple scaling suggests about three and a half minutes for one configuration using 256 sequences. That estimate covers training only. Larger runs, validation, and packaging still need measurement. The hidden judge's CPU and total workload remain unknown.

We keep training and validation separate. Training labels guide weight updates; development scores decide whether a finished candidate improves on our current model. Previously inspected confirmation results cannot provide fresh independent evidence. Next, we will train the registered candidates, check their exported models, measure actual scores, and test execution speed. We will report gains only when measured and keep the current submission available throughout these experiments. The pilot itself establishes feasibility, not a leaderboard improvement or a promised final score.
