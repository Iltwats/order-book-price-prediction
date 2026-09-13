"""Complete synthetic sequences exercise byte parity and frozen receipt reuse."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from connectome import recurrent_parity as parity


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    files = {}
    for name in ("dataset", "original", "candidate", "bulk"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        files[name] = path
    state = SimpleNamespace(fault=None, resets=[], calls=[], bulk_calls=0, closed=0)
    features = []
    for group in range(2):
        value = np.zeros((20000, 112), np.float32)
        value[0, 0] = group+1
        features.append(value)

    class Parquet:
        num_row_groups = 2
        def close(self):
            state.closed += 1

    def sequence(_parquet, group):
        # No targets/need/mask fields: parity must never attempt scoring.
        return SimpleNamespace(seq_ix=400+group, features=features[group])

    class Backend:
        def __init__(self, path, mode):
            self.mode = mode
            self.states = [np.zeros((1, 1, 2), np.float32) for _ in range(2)]
            self.output = np.zeros(2, np.float32)
            self.step_number = 0

        def reset(self):
            state.resets.append(self.mode)
            self.step_number = 0
            for value in self.states:
                value.fill(0)

        def step(self, features):
            self.step_number += 1
            self.states[0][0, 0, 0] = self.step_number
            self.states[1][0, 0, 0] += features[0]
            self.states[1][0, 0, 1] = 0
            self.output[:] = [self.step_number, 0]
            if state.fault == "nonfinite_equal" and self.step_number == 7:
                self.output[0] = np.nan
            if self.mode == "iobinding" and self.step_number == 6:
                if state.fault == "callback_prediction":
                    self.output[0] = np.nextafter(self.output[0], np.float32(np.inf))
                elif state.fault == "callback_signed_zero":
                    self.output[1] = -0.
                elif state.fault == "callback_hidden":
                    self.states[1][0, 0, 1] = np.nextafter(np.float32(0), np.float32(1))
            return self.output  # Deliberately reused output storage.

        def hidden_states(self):
            return self.states  # Deliberately reused state storage.

        def close(self):
            state.closed += 1

    class Bulk:
        def get_outputs(self):
            return [SimpleNamespace(name=name) for name in ("prediction", "next_hidden_0", "next_hidden_1")]

        def run(self, _, feed):
            state.bulk_calls += 1
            assert not feed["hidden_0"].any() and not feed["hidden_1"].any()
            count = feed["features"].shape[1]
            prediction = np.zeros((1, count, 2), np.float32)
            prediction[0, :, 0] = np.arange(1, count+1, dtype=np.float32)
            final0 = np.array([[[count, 0]]], np.float32)
            final1 = np.array([[[feed["features"][0, :, 0].sum(), 0]]], np.float32)
            if state.fault == "bulk_prediction":
                prediction[0, 55, 0] = np.nextafter(prediction[0, 55, 0], np.float32(np.inf))
            elif state.fault == "bulk_signed_zero":
                prediction[0, 55, 1] = -0.
            elif state.fault == "bulk_final_hidden":
                final1[0, 0, 1] = np.nextafter(np.float32(0), np.float32(1))
            return [prediction, final0, final1]

    monkeypatch.setattr(parity.pq, "ParquetFile", lambda _: Parquet())
    monkeypatch.setattr(parity, "read_sequence", sequence)
    monkeypatch.setattr(parity, "OnnxBackend", Backend)
    monkeypatch.setattr(parity.ort, "InferenceSession", lambda *args, **kwargs: Bulk())
    state.files, state.features, state.report = files, features, tmp_path / "report.json"
    return state


def run(state, groups=(0, 1)):
    f = state.files
    return parity.check_parity(f["dataset"], f["original"], f["candidate"], groups=groups,
                              bulk_model=f["bulk"], report_path=state.report)


def require(state, groups=(0, 1)):
    f = state.files
    return parity.require_parity(state.report, f["dataset"], f["original"], candidate_model=f["candidate"],
                                 groups=groups, bulk_model=f["bulk"])


def test_all_rows_states_and_bulk_are_byte_equal_with_bounded_reused_buffers(runtime):
    report = run(runtime)
    assert report["passed"] and report["bitwise_equal"] and report["bulk_bitwise_equal"]
    assert report["score_computed"] is False
    assert all(report["checks"].values())
    assert [row["rows"] for row in report["per_sequence"]] == [20000, 20000]
    assert [row["seq_ix"] for row in report["per_sequence"]] == [400, 401]
    assert runtime.resets == ["reference", "iobinding", "reference", "iobinding"]
    assert runtime.bulk_calls == 2 and runtime.closed == 3
    assert require(runtime) == report
    assert all(row["features_sha256"] == parity._array_hash(x) for row, x in zip(report["per_sequence"], runtime.features))
    before = runtime.report.read_bytes()
    with pytest.raises(FileExistsError, match="preserve"):
        run(runtime)
    assert runtime.report.read_bytes() == before


@pytest.mark.parametrize("fault,match", [
    ("callback_prediction", "callback predictions"),
    ("callback_hidden", "callback hidden 1 all rows"),
    ("callback_signed_zero", "callback predictions"),
    ("bulk_prediction", "bulk predictions all rows"),
    ("bulk_final_hidden", "bulk final hidden 1"),
    ("bulk_signed_zero", "bulk predictions all rows"),
    ("nonfinite_equal", "nonfinite values"),
])
def test_ulp_signed_zero_intermediate_hidden_and_bulk_final_fail_before_receipt_acceptance(runtime, fault, match):
    runtime.fault = fault
    with pytest.raises(AssertionError, match=match):
        run(runtime, groups=[0])
    report = json.loads(runtime.report.read_text())
    assert not report["passed"] and not report["bitwise_equal"] and not report["bulk_bitwise_equal"]
    assert report["score_computed"] is False and report["per_sequence"] == []
    with pytest.raises(ValueError, match="complete matching"):
        require(runtime, groups=[0])


def test_receipt_reuse_rejects_model_data_group_and_coverage_changes(runtime):
    run(runtime)
    with pytest.raises(ValueError, match="complete matching"):
        require(runtime, groups=[1, 0])
    original = runtime.files["bulk"].read_bytes()
    runtime.files["bulk"].write_bytes(b"changed model")
    with pytest.raises(ValueError, match="model changed"):
        require(runtime)
    runtime.files["bulk"].write_bytes(original)
    saved = json.loads(runtime.report.read_text())
    saved["per_sequence"][0]["rows"] = 19999
    runtime.report.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="sequence coverage"):
        require(runtime)
    saved["per_sequence"][0]["rows"] = 20000
    runtime.report.write_text(json.dumps(saved))
    runtime.files["dataset"].write_bytes(b"changed dataset")
    with pytest.raises(ValueError, match="dataset changed"):
        require(runtime)


def test_group_guards_and_finite_byte_checks(runtime):
    for groups in ([0, 0], [-1], [0.5], [], [True]):
        with pytest.raises(ValueError, match="distinct nonnegative"):
            run(runtime, groups)
    with pytest.raises(ValueError, match="out of range"):
        run(runtime, [2])
    with pytest.raises(AssertionError, match="dtype"):
        parity._require_bytes(np.zeros(2, np.float64), np.zeros(2, np.float32), "dtype")
    with pytest.raises(AssertionError, match="nonfinite"):
        parity._require_bytes(np.full(2, np.inf, np.float32), np.full(2, np.inf, np.float32), "finite")
