# What we should do next

We are keeping the supplied GRU as our starting model. Its recurrent weights already capture useful patterns, and the current submission is small and fast. Most of our improvement so far came from correcting prediction scale. The next goal is to improve the information in predictions, while preserving that speed.

First, we should train the final prediction layer using an objective closer to the competition's weighted correlation score. This can change the predictions without adding inference operations. We can store compact statistics for each sequence instead of saving every hidden state, substantially reducing storage. The official clipped score will still decide whether this works.

Second, we should test a small correction using recent feature changes and relationships between the two instruments. These features will use only information already observed. We will preserve the original feature order and avoid assuming that anonymous columns represent particular order-book levels.

Third, we should verify the conversion between the trainable GRU and ONNX, then cautiously fine-tune its upper layer. The earlier numerical mismatch remains unresolved; we will investigate it explicitly before relying on new training results.

The Hugging Face research found two worthwhile secondary candidates. IBM's PatchTSMixer has small weights trained on electricity measurements. Chronos-2-Synth uses synthetic training data and could become an offline teacher. Neither is a ready-made predictor for our anonymous targets. Several other models include external market data, which conflicts with the competition rules.

We should run a limited experiment set, compare every candidate with our current calibrated model, and benchmark complete CPU callbacks. The initial round caps screening at twelve candidate model configurations. The old confirmation set has already been inspected and cannot become a fresh holdout again. No external weights have been adopted, no new score gain is claimed, and the existing submission remains available throughout the next implementation cycle.
