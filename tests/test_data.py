import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from connectome.data import (build_manifest, file_fingerprint, iter_labels, iter_sequences, load_manifest,
                            make_splits, read_sequence, save_manifest)
from utils import FEATURE_COLUMNS, SEQUENCE_LENGTH, WARMUP


def write_fixture(path, ids, *, valid=False, rows=SEQUENCE_LENGTH, nonfinite=False,
                  bad_steps=False, warmup_mask=False):
    writer = None
    try:
        for seq_ix in ids:
            step = np.arange(rows, dtype=np.int32)
            if bad_steps:
                step[[100, 101]] = step[[101, 100]]
            arrays = {"seq_ix": np.full(rows, seq_ix, dtype=np.int32),
                      "step_in_seq": step, "need_prediction": np.arange(rows) >= WARMUP}
            if valid:
                arrays["is_scored"] = (np.arange(rows) >= WARMUP) & (np.arange(rows) % 3 == 0)
                if warmup_mask:
                    arrays["is_scored"][0] = True
            for index, column in enumerate(FEATURE_COLUMNS):
                arrays[column] = np.full(rows, index / 100, dtype=np.float32)
            if nonfinite:
                arrays[FEATURE_COLUMNS[0]][100] = np.nan
            arrays["t0"] = np.sin(np.arange(rows) / 7).astype(np.float32)
            arrays["t1"] = np.cos(np.arange(rows) / 11).astype(np.float32)
            table = pa.table(arrays)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, write_statistics=False)
            writer.write_table(table, row_group_size=rows)
    finally:
        if writer is not None:
            writer.close()


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_feature_order_and_complete_sequence(self):
        path = self.root / "valid.parquet"
        write_fixture(path, [17, 23], valid=True)
        sequences = list(iter_sequences(path, [1, 0]))
        self.assertEqual([s.seq_ix for s in sequences], [23, 17])
        sequence = sequences[0]
        self.assertEqual(sequence.features.shape, (SEQUENCE_LENGTH, 112))
        np.testing.assert_array_equal(sequence.features[0], np.arange(112, dtype=np.float32) / 100)
        self.assertEqual(sequence.targets.shape, (SEQUENCE_LENGTH, 2))
        self.assertEqual(sequence.need.sum(), SEQUENCE_LENGTH - WARMUP)
        self.assertFalse(sequence.mask[:WARMUP].any())
        labels = list(iter_labels(path, [1, 0]))
        self.assertEqual([s.seq_ix for s in labels], [23, 17])
        np.testing.assert_array_equal(labels[0].targets, sequence.targets)
        np.testing.assert_array_equal(labels[0].need, sequence.need)
        np.testing.assert_array_equal(labels[0].mask, sequence.mask)

    def test_noncausal_order_nonfinite_and_warmup_selection_rejected(self):
        for option, message in [("bad_steps", "step order"), ("nonfinite", "nonfinite"),
                                ("warmup_mask", "warm-up")]:
            with self.subTest(option=option):
                path = self.root / f"{option}.parquet"
                write_fixture(path, [7], valid=True, **{option: True})
                with pq.ParquetFile(path) as parquet:
                    with self.assertRaisesRegex(ValueError, message):
                        read_sequence(parquet, 0)

    def test_fixed_nested_split_keeps_inspected_groups_in_development(self):
        splits = make_splits(1873)
        self.assertEqual(splits, make_splits(1873))
        self.assertEqual([len(splits[k]) for k in ("dev", "confirmation", "smoke", "screen")],
                         [1400, 473, 32, 256])
        self.assertFalse(set(splits["dev"]) & set(splits["confirmation"]))
        self.assertEqual(set(splits["dev"]) | set(splits["confirmation"]), set(range(1873)))
        self.assertTrue({0, 468, 936, 1404, 1872} <= set(splits["dev"]))
        self.assertTrue(set(splits["smoke"]) <= set(splits["screen"]) <= set(splits["dev"]))

    def test_manifest_narrow_ids_fingerprint_and_bounded_audit(self):
        train, valid = self.root / "train.parquet", self.root / "valid.parquet"
        write_fixture(train, [101, 102])
        write_fixture(valid, [201, 202, 203, 204], valid=True)
        manifest = build_manifest(train, valid, development_count=3, confirmation_count=1,
                                  smoke_count=1, screen_count=2)
        self.assertTrue(manifest["audit"]["train_validation_ids_disjoint"])
        self.assertEqual([g["seq_ix"] for g in manifest["datasets"]["train"]["groups"]], [101, 102])
        sampled = {g["row_group"] for g in manifest["audit"]["sampled_full_groups"]["valid"]}
        self.assertFalse(sampled & set(manifest["splits"]["confirmation"]))
        output = self.root / "manifest.json"
        save_manifest(manifest, output)
        self.assertEqual(load_manifest(output), manifest)
        with valid.open("ab") as stream:
            stream.write(b"changed")
        self.assertNotEqual(file_fingerprint(valid), manifest["datasets"]["valid"]["fingerprint"])
        with self.assertRaisesRegex(ValueError, "dataset changed"):
            load_manifest(output)

    def test_duplicate_and_cross_split_identifiers_rejected(self):
        for train_ids, valid_ids, error in [([101, 101], [201, 202], "duplicate"),
                                           ([101, 102], [102, 202], "overlap")]:
            with self.subTest(error=error):
                train, valid = self.root / "train.parquet", self.root / "valid.parquet"
                write_fixture(train, train_ids)
                write_fixture(valid, valid_ids, valid=True)
                with self.assertRaisesRegex(ValueError, error):
                    build_manifest(train, valid, development_count=1, confirmation_count=1,
                                   smoke_count=1, screen_count=1)

    def test_all_group_metadata_rejects_wrong_sequence_length(self):
        train, valid = self.root / "train.parquet", self.root / "valid.parquet"
        write_fixture(train, [101], rows=SEQUENCE_LENGTH - 1)
        write_fixture(valid, [201, 202], valid=True)
        with self.assertRaisesRegex(ValueError, "expected 20000"):
            build_manifest(train, valid, development_count=1, confirmation_count=1,
                           smoke_count=1, screen_count=1)


if __name__ == "__main__":
    unittest.main()
