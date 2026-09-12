import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from connectome.calibration import (PredictionCache, apply_calibration, ema_grid, evaluate_cache,
                                    fit_affine_ridge, paired_bootstrap, scale_grid, search_calibration)
from connectome.data import iter_labels
from utils import BlockAccumulator, FEATURE_COLUMNS, SEQUENCE_LENGTH, WARMUP


class CalibrationTests(unittest.TestCase):
    def test_scale_affine_order_and_ema_uses_warmup_causally(self):
        raw = np.zeros((SEQUENCE_LENGTH, 2), dtype=np.float32)
        raw[0] = [2, -2]
        transformed = apply_calibration(raw, scales=[2, 3],
                                        affine={"matrix": [[0, 1], [1, 0]], "bias": [1, 2]})
        np.testing.assert_array_equal(transformed[0], [-5, 6])
        smoothed = apply_calibration(raw, ema_half_life=32, ema_blend=1)
        self.assertAlmostEqual(float(smoothed[WARMUP, 0]), 2 * 2 ** (-WARMUP / 32), places=5)
        changed = raw.copy()
        changed[500:] = [7, 9]
        other = apply_calibration(changed, ema_half_life=32, ema_blend=1)
        np.testing.assert_array_equal(smoothed[:500], other[:500])
        np.testing.assert_array_equal(apply_calibration(np.zeros_like(raw), ema_half_life=32,
                                                       ema_blend=1), np.zeros_like(raw))

    def test_ema_accepts_distinct_target_parameters(self):
        raw = np.random.default_rng(13).normal(size=(200, 2)).astype(np.float32)
        combined = apply_calibration(raw, ema_half_life=[2, 8], ema_blend=[0.1, 0.25])
        first = apply_calibration(raw, ema_half_life=2, ema_blend=0.1)
        second = apply_calibration(raw, ema_half_life=8, ema_blend=0.25)
        np.testing.assert_array_equal(combined[:, 0], first[:, 0])
        np.testing.assert_array_equal(combined[:, 1], second[:, 1])

    def test_train_only_affine_recovers_known_mapping_and_ignores_warmup(self):
        rng = np.random.default_rng(8)
        raw = rng.uniform(-0.5, 0.5, size=(SEQUENCE_LENGTH, 2)).astype(np.float32)
        matrix, bias = np.array([[0.8, 0.2], [-0.1, 1.2]]), np.array([0.1, -0.15])
        targets = (raw @ matrix.T + bias).astype(np.float32)
        targets[:WARMUP] = 1000
        with self.assertRaisesRegex(ValueError, "train split only"):
            fit_affine_ridge([(raw, targets)], source_split="dev")
        affine = fit_affine_ridge([(raw, targets)], source_split="train", ridge=1e-9, shrink=1)
        np.testing.assert_allclose(affine.matrix, matrix, atol=1e-6)
        np.testing.assert_allclose(affine.bias, bias, atol=1e-6)
        identity = fit_affine_ridge([(raw, targets)], source_split="train", shrink=0)
        np.testing.assert_array_equal(identity.matrix, np.eye(2))
        np.testing.assert_array_equal(identity.bias, np.zeros(2))

    def test_cache_complete_guard_and_warmup_finiteness(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cache"
            cache = PredictionCache.create(path, [7, 3], [70, 30], {"split": "dev"})
            with self.assertRaisesRegex(ValueError, "incomplete"):
                PredictionCache.open(path)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                cache.finalize()
            raw = np.ones((SEQUENCE_LENGTH, 2), dtype=np.float32)
            bad = raw.copy()
            bad[0] = np.nan
            with self.assertRaisesRegex(ValueError, "including warm-up"):
                cache.write(0, bad)
            cache.write(0, raw)
            cache.write(1, raw * 2)
            cache.finalize()
            loaded = PredictionCache.open(path)
            self.assertIsInstance(loaded.predictions, np.memmap)
            np.testing.assert_array_equal(loaded.read_group(3), raw * 2)
            with self.assertRaises(FileExistsError):
                PredictionCache.create(path, [7], [70], {})

    def test_paired_bootstrap_pairs_sequence_ids_and_is_batch_invariant(self):
        baseline = {"per_sequence": [{"seq_ix": 7, "eligible": True, "weighted_pearson": 0.1},
                                      {"seq_ix": 3, "eligible": True, "weighted_pearson": 0.7},
                                      {"seq_ix": 8, "eligible": False, "weighted_pearson": None}]}
        candidate = {"per_sequence": [{"seq_ix": 3, "eligible": True, "weighted_pearson": 0.8},
                                       {"seq_ix": 8, "eligible": False, "weighted_pearson": None},
                                       {"seq_ix": 7, "eligible": True, "weighted_pearson": 0.2}]}
        result = paired_bootstrap(candidate, baseline, n_resamples=1000, batch_size=17)
        self.assertAlmostEqual(result["delta"], 0.1)
        self.assertAlmostEqual(result["ci_low"], 0.1)
        self.assertAlmostEqual(result["ci_high"], 0.1)
        self.assertEqual(result, paired_bootstrap(candidate, baseline, n_resamples=1000, batch_size=103))
        candidate["per_sequence"][0]["eligible"] = False
        with self.assertRaisesRegex(ValueError, "eligibility"):
            paired_bootstrap(candidate, baseline)

    def test_exact_evaluation_uses_equal_block_weight_and_eligibility(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "valid.parquet"
            rng = np.random.default_rng(23)
            cache = PredictionCache.create(root / "cache", [0, 1, 2], [40, 41, 42], {})
            reference = BlockAccumulator()
            writer = None
            try:
                for group in range(3):
                    targets = rng.uniform(-3, 3, size=(SEQUENCE_LENGTH, 2)).astype(np.float32)
                    if group == 2:
                        targets[:, 1] = 1  # Whole block excluded despite t0 variation.
                    predictions = (targets + rng.normal(size=targets.shape) * (group + 1)).astype(np.float32)
                    mask = (np.arange(SEQUENCE_LENGTH) >= WARMUP) & (np.arange(SEQUENCE_LENGTH) % (group * 7 + 1) == 0)
                    arrays = {"seq_ix": np.full(SEQUENCE_LENGTH, 40 + group, np.int32),
                              "step_in_seq": np.arange(SEQUENCE_LENGTH, dtype=np.int32),
                              "need_prediction": np.arange(SEQUENCE_LENGTH) >= WARMUP, "is_scored": mask}
                    arrays.update({c: np.zeros(SEQUENCE_LENGTH, np.float32) for c in FEATURE_COLUMNS})
                    arrays.update(t0=targets[:, 0], t1=targets[:, 1])
                    table = pa.table(arrays)
                    if writer is None:
                        writer = pq.ParquetWriter(path, table.schema)
                    writer.write_table(table)
                    cache.write(group, predictions)
                    reference.add(targets, predictions, mask)
            finally:
                if writer is not None:
                    writer.close()
            cache.finalize()
            result = evaluate_cache(path, cache)
            self.assertEqual(result["postprocess"], {})
            self.assertTrue(result["postprocess_tracked"])
            for key, value in reference.result().items():
                self.assertEqual(result[key], value)
            self.assertEqual(result["eligible_blocks"], 2)
            self.assertFalse(result["per_sequence"][2]["eligible"])
            self.assertEqual(result["rows_seen"], 3 * SEQUENCE_LENGTH)
            tracked = evaluate_cache(path, cache, postprocess={"scales": [0.5, 0.75]})
            opaque = evaluate_cache(path, cache,
                transform=lambda raw: apply_calibration(raw, scales=[0.5, 0.75]))
            self.assertEqual(tracked["weighted_pearson"], opaque["weighted_pearson"])
            self.assertEqual(tracked["postprocess"], {"scales": [0.5, 0.75]})
            self.assertFalse(opaque["postprocess_tracked"])
            self.assertIsNone(opaque["postprocess"])
            with patch("connectome.calibration.iter_labels", wraps=iter_labels) as scan:
                search = search_calibration(path, cache, [0, 1, 2], root / "search.json",
                                            scale_values=(0.75, 1.0, 1.5), half_lives=(2, 8),
                                            blends=(0, 0.1), bootstrap_resamples=200)
                self.assertEqual(scan.call_count, 1)
            self.assertEqual(search["candidates_evaluated"], 11)
            self.assertTrue((root / "search.postprocess.json").exists())
            for candidate in search["candidates"]:
                independent = evaluate_cache(path, cache,
                    transform=lambda raw: apply_calibration(raw, **candidate["postprocess"]))
                self.assertEqual(candidate["weighted_pearson"], independent["weighted_pearson"])
            self.assertEqual(search["best"]["weighted_pearson"],
                             max(c["weighted_pearson"] for c in search["candidates"]))

    def test_default_grids_include_identity_and_requested_half_lives(self):
        self.assertIn({"scales": [1.0, 1.0]}, scale_grid())
        self.assertEqual(len(ema_grid()), 7)
        self.assertIn({}, ema_grid())
        self.assertEqual({v["ema_half_life"] for v in ema_grid() if v}, {2, 8, 32})


if __name__ == "__main__":
    unittest.main()
