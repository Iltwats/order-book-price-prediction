Implemented, using the pretrained model you supplied. Both GRU layers and all 192,000 recurrent parameters remain unchanged. I trained a small correction to the final prediction head on 1,024 supplied training sequences, then folded its calibration into the weights. This improves predictions without adding inference operations.

Python still provides the required PredictionModel interface. The expensive calculations run inside ONNX Runtime, which uses optimized native kernels. I reduced callback overhead through persistent input/output binding and reusable hidden-state buffers. Every row updates the model, warm-up returns nothing, and changing sequences clears its memory.

I also implemented the optional C++ backend with aligned memory, prepacked weights, reusable scratch buffers, and one native call per row. Packing helped, but this implementation remained slower than optimized ONNX and failed our strict numerical comparison. It is available for experimentation, but the submission uses ONNX. Core pinning also failed to show a repeatable benefit, so it stays disabled.

The frozen model scored 0.64100 on confirmation, compared with 0.59967 for the original baseline. The improvement was 0.04133, with a 95% interval from 0.03746 to 0.04537. Confirmation was evaluated once after selection. Combined validation scored 0.63222 versus the original 0.58959; that combined figure includes development data used for selection.

On Linux ARM64 with one CPU and 16 GB RAM, the complete callback was 9.52% faster. Streaming 5.12 million rows used about 370 MB of memory. The extracted offline submission passed verification, and 89 host tests passed.

The final ZIP is 0.72 MB. The workspace includes the report, frozen manifests, selected model, C++ experiment, and training/export tools. Full recurrent fine-tuning remains experimental because its reconstructed runtime misses the strict numerical tolerance.

The remaining deployment uncertainty is the judge's CPU and hidden test size. Local measurements cannot certify its 60-minute deadline. No paid training hardware or external training data was used.
