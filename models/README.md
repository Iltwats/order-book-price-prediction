# Pretrained Model Artifacts

This directory contains both the packaged deployment ensembles (`.npz`) and the raw FP32 PyTorch / ONNX backbone checkpoints (`.pt` / `.onnx`):

## Packaged Deployment Ensembles

| Directory | Contents | Dev OOF WP | Public WP | 1-vCPU Judge Time |
| --- | --- | ---: | ---: | ---: |
| `gpu_3model_gru160_lstm160_snap40/` | `correlation1024` + `ScratchSortedGRU-160` + `ScratchSortedLSTM-160` + 5-seed `snap40` `MLP128` calibrator (`[0.2137, 0.4331, 0.3532]`, FP16-panel packed `.npz`) | **0.68062** | **0.65694** | `51m27s` |
| `gpu_2model_gru160_snap40/` | `correlation1024` + `ScratchSortedGRU-160` + 5-seed `snap40` `MLP128` calibrator (`[0.2663, 0.7337]`, FP16-panel packed `.npz`) | **0.68023** | **0.65662** | `33m51s` |

Switch between the 3-model and 2-model ensembles at inference or packaging time by editing `"weights"` in [`../submission_config.json`](../submission_config.json).

## Standalone FP32 Backbone Checkpoints

| Directory | Architecture & Training | Standalone Raw Dev WP | Public WP |
| --- | --- | ---: | ---: |
| `sgru_h160_10ep/final_checkpoint.pt` | `ScratchSortedGRU` (`hidden=160`, 413,602 params, ~10 epochs on `train.parquet`, Polyak EMA `0.995`) | **0.67499** | — |
| `slstm_h160_10ep/final_checkpoint.pt` | `ScratchSortedLSTM` (`hidden=160`, 550,242 params, ~10 epochs on `train.parquet`, Polyak EMA `0.995`) | **0.67437** | — |
| `correlation1024_seed20260912/model.onnx` | 2-layer ONNX `GRU128` (192,258 params) fine-tuned with sequence correlation loss | `0.67209` | `0.65151` |
| `sgru_largetick_h128/final_checkpoint.pt` | Large-tick specialist `ScratchSortedGRU` (`hidden=128`, trained strictly on `0.4685 + 0.6569` sequences) | `0.59421` (`0.40686` on `0.6569`) | — |

