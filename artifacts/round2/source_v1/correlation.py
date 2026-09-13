"""Offline, sequence-balanced readout fitting with bounded sufficient statistics.

The correlation loss uses *unclipped predictions*. The official clipped WP,
evaluated on development, remains the selection criterion. The original ORT
encoder is frozen; no training dependency enters the submission runtime.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch

from .data import file_fingerprint
from .evaluation import environment, file_hash, save_json
from .model_io import load_weights
from .training import OnnxSequenceEncoder, _iter_sequences, export_frozen_head, load_torch_model
from utils import FEATURE_COLUMNS, WARMUP

FORMAT = "connectome.correlation-moments.v1"
VAR_FLOOR = 1e-16  # Official standard-deviation floor is 1e-8.
MOMENT_KEYS = ("covariance", "cross", "target_variance", "feature_mean", "target_mean")


def sequence_moments(features, targets):
    """Target-specific FP64 moments; inputs already exclude warm-up rows.

    All weights are divided by the side's full-sequence total. A block with
    either one-sign target is excluded, independently of prediction quality.
    """
    h = np.asarray(features, np.float64)
    raw_y = np.asarray(targets, np.float32)
    if not np.isfinite(raw_y).all():
        raise ValueError("nonfinite training targets")
    y = np.clip(raw_y, -2, 2).astype(np.float64)
    if h.ndim != 2 or y.shape != (len(h), 2) or not len(h) or not h.shape[1]:
        raise ValueError("expected nonempty features [rows,width] and targets [rows,2]")
    if not np.isfinite(h).all() or not np.isfinite(y).all():
        raise ValueError("nonfinite training features or targets")
    if not all(np.any(y[:, k] > 0) and np.any(y[:, k] < 0) for k in range(2)):
        return None
    cov, cross, variance, mean_h, mean_y = [], [], [], [], []
    for k in range(2):
        w = np.abs(y[:, k])
        total = w.sum()
        # Eligibility depends on signs, not total weight. Tiny-weight sides
        # remain in the block and contribute zero correlation via q's floor.
        w /= total
        mh, my = w @ h, w @ y[:, k]
        hc, yc = h - mh, y[:, k] - my
        c = hc.T @ (w[:, None] * hc)
        cov.append((c + c.T) * 0.5)
        cross.append(hc.T @ (w * yc))
        variance.append(w @ (yc * yc))
        mean_h.append(mh)
        mean_y.append(my)
    return dict(zip(MOMENT_KEYS, (np.stack(cov), np.stack(cross), np.asarray(variance),
                                 np.stack(mean_h), np.asarray(mean_y))))


def _validate_groups(train_path, groups):
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(train_path)
    # This extractor is deliberately for the supplied training table, not a
    # generic route around confirmation guards in the evaluation CLI.
    if "is_scored" in parquet.schema_arrow.names:
        raise ValueError("moment extraction requires training data without is_scored")
    groups = list(map(int, groups))
    if not groups or len(groups) != len(set(groups)) or min(groups) < 0 or max(groups) >= parquet.num_row_groups:
        raise ValueError("training groups must be distinct existing row groups")
    return groups


def extract_moments(model_path, train_path, groups, output_dir, *, progress=True):
    """One streamed pass writes final-state and output moments, never row caches.

    A completed cache is content-hashed and cannot be silently overwritten or
    reused for another checkpoint, dataset, feature order, or sequence list.
    """
    started = time.perf_counter()
    groups = _validate_groups(train_path, groups)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("moment output directory already exists")
    weights = load_weights(model_path)
    output_dir.mkdir(parents=True)
    provenance = {"format": FORMAT, "source_model": str(Path(model_path).resolve()),
                  "training_path": str(Path(train_path).resolve()),
                  "model_sha256": file_hash(model_path), "training_data": file_fingerprint(train_path),
                  "groups": groups, "features": list(FEATURE_COLUMNS), "warmup": WARMUP,
                  "code_sha256": file_hash(__file__), "training_code_sha256": file_hash(Path(__file__).with_name("training.py")),
                  "target_rule": "clip FP32 targets to [-2,2]; absolute weights normalized per sequence/target",
                  "mask_rule": "all post-warmup training rows; both target signs required",
                  "complete": False}
    save_json(output_dir / "manifest.json", provenance)
    encoder = OnnxSequenceEncoder(model_path, output_dir / "encoder.onnx")
    shapes = {"covariance": (2, 128, 128), "cross": (2, 128), "target_variance": (2,),
              "feature_mean": (2, 128), "target_mean": (2,)}
    arrays = {key: np.lib.format.open_memmap(output_dir / f"{key}.npy", mode="w+", dtype=np.float64,
                                           shape=(len(groups), *shape)) for key, shape in shapes.items()}
    output_rows = {key: [] for key in MOMENT_KEYS}
    included, excluded, seq_ids = [], [], []
    for index, (seq, x, y) in enumerate(_iter_sequences(train_path, groups)):
        _, upper, predicted = encoder(x)
        moments = sequence_moments(upper[WARMUP:], y[WARMUP:])
        if moments is None:
            excluded.append({"row_group": groups[index], "seq_ix": seq})
        else:
            output_moments = sequence_moments(predicted[WARMUP:], y[WARMUP:])
            slot = len(included)
            for key in MOMENT_KEYS:
                arrays[key][slot] = moments[key]
                output_rows[key].append(output_moments[key])
            included.append(groups[index])
            seq_ids.append(int(seq))
        if progress and ((index + 1) % 32 == 0 or index + 1 == len(groups)):
            elapsed = time.perf_counter() - started
            print(f"moments {index+1}/{len(groups)} sequences; {elapsed:.1f}s; "
                  f"{(index+1)*20000/elapsed:.0f} rows/s", flush=True)
    if file_fingerprint(train_path) != provenance["training_data"]:
        raise ValueError("training dataset changed during moment extraction")
    if not included:
        raise ValueError("no eligible training blocks")
    for array in arrays.values():
        array.flush()
    del arrays
    for key, values in output_rows.items():
        np.save(output_dir / f"outputs_{key}.npy", np.stack(values), allow_pickle=False)
    np.savez(output_dir / "initial_head.npz", beta=weights["head_w"].T, bias=weights["head_b"])
    files = {p.name: file_hash(p) for p in output_dir.glob("*.np*")}
    provenance.update(complete=True, included_groups=included, excluded=excluded, seq_ids=seq_ids,
                      eligible_sequences=len(included), encoder=encoder.provenance, files=files,
                      seconds=time.perf_counter()-started, environment=environment())
    save_json(output_dir / "manifest.json", provenance)
    return provenance


def load_moments(path, *, representation="hidden", model_path=None, groups=None, train_path=None, verify=True):
    path = Path(path).resolve()
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("format") != FORMAT or not manifest.get("complete"):
        raise ValueError("incomplete or incompatible moment cache")
    if representation not in {"hidden", "outputs"}:
        raise ValueError("representation must be hidden or outputs")
    if model_path is not None and manifest["model_sha256"] != file_hash(model_path):
        raise ValueError("moment model hash differs")
    if groups is not None and list(map(int, groups)) != manifest["groups"]:
        raise ValueError("moment training groups differ")
    source_data = Path(manifest["training_path"])
    if train_path is not None and Path(train_path).resolve() != source_data:
        raise ValueError("moment training dataset path differs")
    if file_fingerprint(source_data) != manifest["training_data"]:
        raise ValueError("moment training dataset fingerprint differs")
    if manifest["features"] != list(FEATURE_COLUMNS) or manifest["warmup"] != WARMUP:
        raise ValueError("moment feature/warm-up contract differs")
    if verify:
        for name, expected in manifest["files"].items():
            if file_hash(path / name) != expected:
                raise ValueError(f"moment content hash differs: {name}")
    prefix = "outputs_" if representation == "outputs" else ""
    count = manifest["eligible_sequences"]
    moments = {key: np.load(path / f"{prefix}{key}.npy", mmap_mode="r", allow_pickle=False)[:count]
               for key in MOMENT_KEYS}
    with np.load(path / "initial_head.npz", allow_pickle=False) as head:
        beta = np.eye(2, dtype=np.float64) if representation == "outputs" else head["beta"].astype(np.float64)
        bias = np.zeros(2, dtype=np.float64) if representation == "outputs" else head["bias"].astype(np.float64)
    return moments, beta, bias, manifest


def moment_loss(beta, moments, initial_beta, initial_bias, *, objective="correlation", penalty=1e-2):
    """Differentiable mean over sequences/sides with training-centered offset.

    The fixed center is the equal-sequence mean of each target's weighted
    feature means. Inference uses a single derived bias, never per-sequence
    target statistics. Both objective controls use the same parameterization.
    """
    if objective not in {"correlation", "mse"} or penalty < 0 or not np.isfinite(penalty):
        raise ValueError("invalid objective or penalty")
    c, v, q, mh, my = [moments[key] for key in MOMENT_KEYS]
    cb = torch.einsum("bkij,kj->bki", c, beta)
    variance = (cb * beta[None]).sum(-1)
    covariance = (v * beta[None]).sum(-1)
    rho = covariance / torch.sqrt(variance.clamp_min(VAR_FLOOR) * q.clamp_min(VAR_FLOOR))
    rho = torch.where((variance > VAR_FLOOR) & (q > VAR_FLOOR), rho.clamp(-1, 1), torch.zeros_like(rho))
    center = mh.mean(0)
    bias = initial_bias + ((initial_beta - beta) * center).sum(-1)
    predicted_mean = (mh * beta[None]).sum(-1) + bias[None]
    mse = variance - 2 * covariance + q + (predicted_mean - my).square()
    anchor = ((beta-initial_beta).square().sum(-1) / initial_beta.square().sum(-1).clamp_min(1e-12)).mean()
    data_loss = (1-rho).mean() if objective == "correlation" else mse.mean()
    return data_loss + penalty * anchor, {"rho": rho, "mse": mse, "anchor": anchor,
                                        "variance": variance, "bias": bias, "center": center}


def optimize_moments(moments, initial_beta, initial_bias, *, objective="correlation", penalty=1e-2,
                     max_iterations=200, threads=1):
    if max_iterations < 1 or threads < 1:
        raise ValueError("positive iteration and thread limits required")
    torch.set_num_threads(threads)
    tensors = {key: torch.tensor(np.asarray(moments[key]), dtype=torch.float64) for key in MOMENT_KEYS}
    beta0, bias0 = torch.tensor(initial_beta, dtype=torch.float64), torch.tensor(initial_bias, dtype=torch.float64)
    if not all(torch.isfinite(value).all() for value in [*tensors.values(), beta0, bias0]):
        raise ValueError("nonfinite moments or initial head")
    n, sides, width = tensors["feature_mean"].shape
    expected = {"covariance": (n, 2, width, width), "cross": (n, 2, width),
                "target_variance": (n, 2), "feature_mean": (n, 2, width), "target_mean": (n, 2)}
    if n < 1 or sides != 2 or beta0.shape != (2, width) or bias0.shape != (2,) or any(tuple(tensors[k].shape) != v for k,v in expected.items()):
        raise ValueError("incompatible moment or head shapes")
    beta = beta0.clone().requires_grad_()
    optimizer = torch.optim.LBFGS([beta], lr=1, max_iter=max_iterations, max_eval=max_iterations*2,
                                 history_size=20, tolerance_grad=1e-9, tolerance_change=1e-12,
                                 line_search_fn="strong_wolfe")
    history = []
    initial_loss, initial_details = moment_loss(beta, tensors, beta0, bias0, objective=objective, penalty=penalty)
    best = {"loss": float(initial_loss.detach()), "beta": beta.detach().clone()}

    def closure():
        optimizer.zero_grad()
        loss, details = moment_loss(beta, tensors, beta0, bias0, objective=objective, penalty=penalty)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite readout objective")
        loss.backward()
        if beta.grad is None or not torch.isfinite(beta.grad).all():
            raise FloatingPointError("nonfinite readout gradient")
        value = float(loss.detach())
        history.append(value)
        if value < best["loss"]:
            best.update(loss=value, beta=beta.detach().clone())
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        loss, details = moment_loss(beta, tensors, beta0, bias0, objective=objective, penalty=penalty)
        if float(loss) < best["loss"]:
            best.update(loss=float(loss), beta=beta.detach().clone())
        beta = best["beta"]
        loss, details = moment_loss(beta, tensors, beta0, bias0, objective=objective, penalty=penalty)
    return beta.numpy(), details["bias"].numpy(), {
        "initial_loss": float(initial_loss.detach()), "final_loss": float(loss),
        "initial_training_unclipped_rho": float(initial_details["rho"].detach().mean()),
        "training_unclipped_rho": float(details["rho"].mean()), "training_weighted_mse": float(details["mse"].mean()),
        "relative_anchor": float(details["anchor"]), "closure_evaluations": len(history),
        "optimizer_iterations": int(optimizer.state[next(iter(optimizer.state))].get("n_iter", 0)) if optimizer.state else 0,
        "history": history, "minimum_prediction_variance": float(details["variance"].min()),
        "zero_variance_sides": int((details["variance"] <= VAR_FLOOR).sum()),
        "center": details["center"].numpy().tolist()}


def fit_correlation_head(model_path, moments_path, output_dir, *, representation="hidden",
                         objective="correlation", penalty=1e-2, max_iterations=200, threads=1,
                         seed=20260912):
    started = time.perf_counter()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("candidate output directory already exists")
    moments, beta0, bias0, cache = load_moments(moments_path, representation=representation, model_path=model_path)
    beta, bias, result = optimize_moments(moments, beta0, bias0, objective=objective, penalty=penalty,
                                        max_iterations=max_iterations, threads=threads)
    model = load_torch_model(model_path)
    with torch.no_grad():
        if representation == "outputs":
            source_w = model.head.weight.detach().numpy().astype(np.float64)
            source_b = model.head.bias.detach().numpy().astype(np.float64)
            model.head.weight.copy_(torch.from_numpy((beta @ source_w).astype(np.float32)))
            model.head.bias.copy_(torch.from_numpy((beta @ source_b + bias).astype(np.float32)))
        else:
            model.head.weight.copy_(torch.from_numpy(beta.astype(np.float32)))
            model.head.bias.copy_(torch.from_numpy(bias.astype(np.float32)))
    model_file = export_frozen_head(model_path, model, output_dir / "model.onnx")
    before, after = load_weights(model_path), load_weights(model_file)
    recurrence_equal = all(np.array_equal(before[k], after[k]) for k in ("w0", "r0", "b0", "w1", "r1", "b1"))
    if not recurrence_equal:
        raise AssertionError("frozen readout export changed recurrent weights")
    parameters = {"representation": representation, "objective": objective, "penalty": penalty,
                  "max_iterations": max_iterations, "threads": threads,
                  "offset_policy": "training_centered_selected"}
    specification = {"groups": cache["groups"], "seed": seed, "epochs": 0,
                     "source": {"sha256": file_hash(model_path)},
                     "data": {"path": cache["training_path"], **cache["training_data"]},
                     "parameters": parameters}
    report = {"method": "frozen_moment_readout", "representation": representation, "objective": objective,
              "complete": True, "specification": specification,
              "penalty": penalty, "max_iterations": max_iterations, "model_path": model_file,
              "model_sha256": file_hash(model_file), "parent_model_sha256": file_hash(model_path),
              "moment_manifest_sha256": file_hash(Path(moments_path) / "manifest.json"),
              "training_groups": cache["groups"], "eligible_training_sequences": cache["eligible_sequences"],
              "training_seq_ids": cache["seq_ids"], "target_rule": cache["target_rule"],
              "offset_policy": "fixed selected offset in training-centered coordinates; one exported bias per target",
              "code_sha256": file_hash(__file__), "training_code_sha256": file_hash(Path(__file__).with_name("training.py")),
              "beta": beta.tolist(), "bias": bias.tolist(), "recurrence_bitwise_equal": recurrence_equal,
              "coefficient_coordinates": "selected output transform" if representation == "outputs" else "final GRU hidden state",
              "exported_head_bias": after["head_b"].tolist(),
              "optimization": result, "seconds": time.perf_counter()-started, "environment": environment(),
              "selection_metric": "official clipped development WP; training surrogate is not a score claim"}
    save_json(output_dir / "report.json", report)
    return report
