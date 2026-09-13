"""Benchmark guard/aggregation tests; never execute a timing or inference run."""
import numpy as np
import pyarrow as pa
import pytest

from scripts.benchmark_causal_branch import (
    ARMS, load_features, run_orders, select_groups, summarize, verify_ownership,
)
from utils import FEATURE_COLUMNS, SEQUENCE_LENGTH


def test_three_orders_counterbalance_each_arm_without_claiming_perfect_balance():
    orders = run_orders()
    assert len(orders) == 3
    assert all(set(order) == set(ARMS) for order in orders)
    assert all(len({order.index(name) for order in orders}) == 3 for name in ARMS)


def test_group_selection_stays_inside_smoke_and_outside_confirmation():
    manifest = {"splits": {"smoke": [4, 9, 2], "confirmation": [7]}}
    assert select_groups(manifest, count=2) == [4, 9]
    assert select_groups(manifest, [2, 4]) == [2, 4]
    for requested in ([4, 4], [7], [1], []):
        with pytest.raises(ValueError):
            select_groups(manifest, requested)
    with pytest.raises(ValueError, match="unavailable"):
        select_groups(manifest, count=4)


def test_feature_loader_requests_no_targets_or_score_mask(monkeypatch):
    row_count = SEQUENCE_LENGTH
    zero = np.zeros(row_count, np.float32)
    columns = {"seq_ix": np.full(row_count, 104, np.int32),
               "step_in_seq": np.arange(row_count, dtype=np.int32),
               "need_prediction": np.arange(row_count) >= 99,
               **{name: zero for name in FEATURE_COLUMNS}}
    table = pa.table(columns)
    calls = []
    class FakeParquet:
        def read_row_group(self, group, columns, use_threads):
            calls.append((group, columns, use_threads))
            assert "t0" not in columns and "t1" not in columns and "is_scored" not in columns
            return table.select(columns)
        def close(self):
            pass
    monkeypatch.setattr("pyarrow.parquet.ParquetFile", lambda _: FakeParquet())
    sequence = load_features("unused_synthetic_path", [3], {"3": 104})[0]
    assert sequence.features.shape == (20_000, 112)
    assert sequence.features.dtype == np.float32
    assert sequence.seq_ix == 104 and sequence.row_group == 3
    assert calls[0][2] is False
    with pytest.raises(ValueError, match="sequence ID"):
        load_features("unused_synthetic_path", [3], {"3": 999})


def test_summary_uses_complete_row_count_and_requires_three_runs():
    records = []
    for repetition, seconds in enumerate((4, 2, 3)):
        for name in ARMS:
            records.append({"name": name, "rows": 40_000, "seconds": seconds,
                            "startup_seconds": 0.1,
                            "rss": {"after_callbacks": {"process_peak_bytes": 1_000_000}}})
    result = summarize(records)
    assert result[ARMS[0]]["median_seconds"] == 3
    assert result[ARMS[0]]["median_callbacks_per_second"] == pytest.approx(40_000/3)
    assert result[ARMS[0]]["median_microseconds_per_callback"] == 75
    with pytest.raises(ValueError, match="three repetitions"):
        summarize(records[:-1])


def test_retained_output_mutation_is_detected():
    output = np.array([1, 2], np.float32)
    saved = [(output, output.tobytes())]
    verify_ownership(saved)
    output[0] = 4
    with pytest.raises(AssertionError, match="overwritten"):
        verify_ownership(saved)
