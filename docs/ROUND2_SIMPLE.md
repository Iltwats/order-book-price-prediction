# Round two in plain language

We kept the supplied pretrained model: two GRU layers with 128 hidden values each. Its recurrent weights remain unchanged. This round focused on improving the final prediction layer, which turns those hidden values into the two required answers.

We implemented five controlled experiments. Two adjusted the selected model's existing outputs, comparing squared error with correlation training. Three trained the final hidden state readout directly for correlation, using different strengths of regularization. Every correction started from our selected model.

Training used only supplied training sequences. We processed complete sequences, including warm-up rows, and saved compact statistical summaries instead of enormous hidden-state files. Each target and sequence received the intended weighting. We preserved the selected model's average training prediction through a fixed, training-derived offset.

The resulting coefficients fit into the existing final layer and add no inference operations. Linux tests measured around 40,700 callbacks each second using one CPU.

All five candidates completed development screening. Four lost. The output correction trained for correlation reached full-development WP of 0.629306913, compared with 0.629255535 for the selected model. Its improvement was 0.000051378, with an uncertainty interval from minus 0.000041751 to plus 0.000141743.

The predeclared repeat using 512 training sequences improved, reaching 0.629272806: a gain of 0.000017271, with an interval from minus 0.000003088 to plus 0.000036346. Both intervals include zero. This was a sensitivity check, not independent confirmation. The previously used confirmation set remains closed.

We created and verified an experimental ZIP while preserving the current submission and default model. Your screenshot's public score of 0.6083 and rank 26 concern different data; these findings cannot guarantee a leaderboard gain. We have 145 passing tests. Strict Torch versus ONNX agreement remains unresolved, so recurrent fine-tuning has not started. No external pretrained weights were introduced, and datasets remain excluded from commits. Evidence for adopting this change remains weak.
