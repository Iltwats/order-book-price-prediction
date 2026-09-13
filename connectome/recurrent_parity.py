"""Byte-exact callback and bulk parity receipts without computing any score."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq

from .data import file_fingerprint, read_sequence
from .evaluation import environment, file_hash, save_json
from .runtime import OnnxBackend, session_options
from utils import SEQUENCE_LENGTH

FORMAT = "connectome.recurrent-parity.v1"
CHECKS = ("callback_predictions", "callback_all_hidden_states",
          "bulk_all_predictions", "bulk_final_hidden_states")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _sources():
    root = Path(__file__).resolve().parent
    paths = [Path(__file__), *[root / name for name in ("data.py", "evaluation.py", "runtime.py")], root.parent / "utils.py"]
    return {str(path.resolve()): file_hash(path) for path in paths}


def _groups(groups):
    values = list(groups)
    if (not values or any(isinstance(g, bool) or not isinstance(g, (int, np.integer)) or g < 0 for g in values)
            or len(set(values)) != len(values)):
        raise ValueError("parity needs distinct nonnegative integer row groups")
    return [int(g) for g in values]


def _array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _require_bytes(actual, expected, name):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or actual.dtype != np.float32 or expected.dtype != np.float32:
        raise AssertionError(f"{name}: incompatible FP32 shape or dtype")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError(f"{name}: nonfinite values")
    if actual.tobytes(order="C") != expected.tobytes(order="C"):
        # uint32 comparisons retain signed-zero distinctions that array_equal loses.
        left, right = (np.ascontiguousarray(v).view(np.uint32) for v in (actual, expected))
        raise AssertionError(f"{name}: byte mismatch in {np.count_nonzero(left != right)} FP32 values")


def check_parity(dataset_path, original_model, candidate_model=None, groups=(0,),
                 mode="iobinding", bulk_model=None, report_path=None):
    """Require byte equality over complete sequences before any scoring.

    Reference and candidate callbacks retain every output and hidden state for
    one sequence only. The bulk path must match all prediction rows and both
    final hidden states. Targets, masks and prediction eligibility do not enter
    comparisons. A failed check raises and writes a failed receipt when asked.
    """
    if mode != "iobinding" or bulk_model is None:
        raise ValueError("recurrent parity requires iobinding and a bulk model")
    groups = _groups(groups)
    dataset_path, original_model = map(lambda p: Path(p).resolve(), (dataset_path, original_model))
    candidate_model, bulk_model = Path(candidate_model or original_model).resolve(), Path(bulk_model).resolve()
    if report_path is not None and Path(report_path).exists():
        raise FileExistsError("preserve existing parity receipt")
    started = time.monotonic()
    report = {"format": FORMAT, "passed": False, "bitwise_equal": False, "bulk_bitwise_equal": False, "mode": mode,
              "groups": groups, "groups_sha256": _digest(groups), "per_sequence": [],
              "dataset": {"path": str(dataset_path), "fingerprint": file_fingerprint(dataset_path)},
              "original_sha256": file_hash(original_model), "candidate_sha256": file_hash(candidate_model),
              "bulk_model_sha256": file_hash(bulk_model), "implementation": _sources(),
              "checks": {name: False for name in CHECKS}, "comparison": "finite FP32 shape/dtype and exact C-order bytes; signed zeros differ",
              "score_computed": False}
    parquet = reference = candidate = None
    try:
        parquet = pq.ParquetFile(dataset_path)
        if max(groups) >= parquet.num_row_groups:
            raise ValueError("parity row group out of range")
        reference, candidate = OnnxBackend(original_model, "reference"), OnnxBackend(candidate_model, mode)
        bulk = ort.InferenceSession(str(bulk_model), sess_options=session_options(), providers=["CPUExecutionProvider"])
        if [x.name for x in bulk.get_outputs()] != ["prediction", "next_hidden_0", "next_hidden_1"]:
            raise ValueError("bulk model must expose prediction and two final hidden states in order")
        seen = set()
        for group in groups:
            sequence = read_sequence(parquet, group)
            features = sequence.features
            if (features.shape != (SEQUENCE_LENGTH, 112) or features.dtype != np.float32
                    or not np.isfinite(features).all() or sequence.seq_ix in seen):
                raise ValueError("parity requires distinct complete finite FP32 sequences")
            seen.add(sequence.seq_ix)
            feature_hash = _array_hash(features)
            reference.reset(); candidate.reset()
            initial = reference.hidden_states()
            if len(initial) != 2 or len(candidate.hidden_states()) != 2:
                raise ValueError("two recurrent states required")
            outputs = [np.empty((SEQUENCE_LENGTH, 2), np.float32) for _ in range(2)]
            states = [[np.empty((SEQUENCE_LENGTH, *state.shape), np.float32) for state in initial] for _ in range(2)]
            for step, row in enumerate(features):
                for slot, backend in enumerate((reference, candidate)):
                    output = backend.step(row)
                    if output.shape != (2,) or output.dtype != np.float32:
                        raise AssertionError("callback prediction shape or dtype changed")
                    outputs[slot][step] = output
                    hidden = backend.hidden_states()
                    if len(hidden) != 2:
                        raise AssertionError("callback state count changed")
                    for layer, value in enumerate(hidden):
                        if value.shape != initial[layer].shape or value.dtype != np.float32:
                            raise AssertionError("callback hidden shape or dtype changed")
                        states[slot][layer][step] = value
            _require_bytes(outputs[1], outputs[0], f"group {group} callback predictions")
            for layer in range(2):
                _require_bytes(states[1][layer], states[0][layer], f"group {group} callback hidden {layer} all rows")
            bulk_outputs = bulk.run(None, {"features": features[None],
                                           "hidden_0": np.zeros_like(initial[0]), "hidden_1": np.zeros_like(initial[1])})
            if len(bulk_outputs) != 3:
                raise AssertionError("bulk output count changed")
            _require_bytes(bulk_outputs[0], outputs[0][None], f"group {group} bulk predictions all rows")
            for layer in range(2):
                _require_bytes(bulk_outputs[layer+1], states[0][layer][-1], f"group {group} bulk final hidden {layer}")
            if _array_hash(features) != feature_hash:
                raise AssertionError("inference mutated sequence inputs")
            report["per_sequence"].append({"row_group": group, "seq_ix": int(sequence.seq_ix),
                "rows": SEQUENCE_LENGTH, "features_sha256": feature_hash,
                "reference_predictions_sha256": _array_hash(outputs[0]),
                "reference_hidden_sha256": [_array_hash(value) for value in states[0]],
                "checks": {name: True for name in CHECKS}})
            print(f"byte parity sequence {group}: callback all states and bulk outputs/final states passed", flush=True)
        if file_fingerprint(dataset_path) != report["dataset"]["fingerprint"]:
            raise ValueError("dataset changed during parity")
        for path, key in ((original_model, "original_sha256"), (candidate_model, "candidate_sha256"), (bulk_model, "bulk_model_sha256")):
            if file_hash(path) != report[key]:
                raise ValueError("model changed during parity")
        if _sources() != report["implementation"]:
            raise ValueError("parity implementation changed during execution")
        report.update(passed=True, bitwise_equal=True, bulk_bitwise_equal=True, checks={name: True for name in CHECKS})
        report["dataset"]["selected_features_sha256"] = _digest([
            {key: row[key] for key in ("row_group", "seq_ix", "features_sha256")} for row in report["per_sequence"]])
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report.update(seconds=time.monotonic()-started, environment=environment())
        if report_path is not None:
            save_json(report_path, report)
        for resource in (parquet, reference, candidate):
            if resource is not None and hasattr(resource, "close"):
                resource.close()
    return report


def require_parity(report_path, dataset_path, original_model, *, groups, bulk_model,
                   candidate_model=None, mode="iobinding"):
    """Verify a completed receipt and current provenance without scoring rows."""
    report = json.loads(Path(report_path).read_text())
    groups = _groups(groups)
    if (report.get("format") != FORMAT or report.get("passed") is not True or report.get("bitwise_equal") is not True
            or report.get("bulk_bitwise_equal") is not True
            or report.get("score_computed") is not False or report.get("mode") != mode or mode != "iobinding"
            or report.get("groups") != groups or report.get("groups_sha256") != _digest(groups)
            or report.get("checks") != {name: True for name in CHECKS}
            or report.get("implementation") != _sources()):
        raise ValueError("complete matching byte-parity receipt required")
    rows = report.get("per_sequence", [])
    if ([row["row_group"] for row in rows] != groups or len({row["seq_ix"] for row in rows}) != len(groups)
            or any(row["rows"] != SEQUENCE_LENGTH or row["checks"] != {name: True for name in CHECKS} for row in rows)
            or report["dataset"].get("selected_features_sha256") != _digest([
                {key: row[key] for key in ("row_group", "seq_ix", "features_sha256")} for row in rows])):
        raise ValueError("incomplete parity sequence coverage")
    if (Path(report["dataset"]["path"]).resolve() != Path(dataset_path).resolve()
            or report["dataset"]["fingerprint"] != file_fingerprint(dataset_path)):
        raise ValueError("parity dataset changed")
    for path, key in ((original_model, "original_sha256"), (candidate_model or original_model, "candidate_sha256"), (bulk_model, "bulk_model_sha256")):
        if file_hash(path) != report[key]:
            raise ValueError("parity model changed")
    return report
