"""Sequence-preserving reads and reproducible, bounded-memory dataset manifests."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq

from utils import FEATURE_COLUMNS, SEQUENCE_LENGTH, TARGET_COLUMNS, WARMUP, validate_sequence

MANIFEST_VERSION = 1
INSPECTED_VALID_GROUPS = (0, 468, 936, 1404, 1872)


@dataclass(frozen=True)
class Sequence:
    row_group: int
    seq_ix: int
    features: np.ndarray
    targets: np.ndarray
    need: np.ndarray
    mask: np.ndarray | None = None

    @property
    def need_prediction(self) -> np.ndarray:
        return self.need


@dataclass(frozen=True)
class SequenceLabels:
    row_group: int
    seq_ix: int
    targets: np.ndarray
    need: np.ndarray
    mask: np.ndarray | None


def _expected_schema(validation: bool) -> list[str]:
    return ["seq_ix", "step_in_seq", "need_prediction", *(["is_scored"] if validation else []),
            *FEATURE_COLUMNS, *TARGET_COLUMNS]


def _check_schema(parquet: pq.ParquetFile, *, mask_only: bool = False) -> None:
    names = parquet.schema_arrow.names
    if mask_only:
        expected = ["seq_ix", "step_in_seq", "is_scored"]
    else:
        expected = _expected_schema("is_scored" in names)
    if names != expected:
        raise ValueError("unexpected schema or column order; use the exact numerical feature order")
    for field in parquet.schema_arrow:
        expected_type = ("int32" if field.name in {"seq_ix", "step_in_seq"} else
                         "bool" if field.name in {"need_prediction", "is_scored"} else "float")
        if str(field.type) != expected_type:
            raise ValueError(f"unexpected dtype for {field.name}: {field.type}, expected {expected_type}")


def read_sequence(parquet: pq.ParquetFile, group: int) -> Sequence:
    """Read and validate one complete labeled sequence, without retaining other groups."""
    _check_schema(parquet)
    group = int(group)
    if not 0 <= group < parquet.num_row_groups:
        raise IndexError(f"row group {group} out of range")
    table = parquet.read_row_group(group, use_threads=False)
    if any(table[name].null_count for name in table.column_names):
        raise ValueError(f"null values in row group {group}")
    seq_ix, need = validate_sequence(table)
    features = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32, copy=False)
    targets = np.column_stack([table[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32, copy=False)
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError(f"nonfinite features or targets in row group {group}")
    mask = table["is_scored"].to_numpy().astype(bool) if "is_scored" in table.column_names else None
    if mask is not None and mask[:WARMUP].any():
        raise ValueError(f"scoring mask selects warm-up in row group {group}")
    return Sequence(group, seq_ix, features, targets, need, mask)


def iter_sequences(path: str | Path, groups=None) -> Iterator[Sequence]:
    """Read groups in the requested order; rows within a sequence are never shuffled."""
    parquet = pq.ParquetFile(path)
    try:
        for group in range(parquet.num_row_groups) if groups is None else groups:
            yield read_sequence(parquet, int(group))
    finally:
        parquet.close()


def iter_labels(path: str | Path, groups=None) -> Iterator[SequenceLabels]:
    """Read only IDs, order, warm-up, mask and labels for cached prediction scoring."""
    parquet = pq.ParquetFile(path)
    try:
        _check_schema(parquet)
        columns = ["seq_ix", "step_in_seq", "need_prediction"]
        if "is_scored" in parquet.schema_arrow.names:
            columns.append("is_scored")
        columns.extend(TARGET_COLUMNS)
        for group in range(parquet.num_row_groups) if groups is None else groups:
            group = int(group)
            if not 0 <= group < parquet.num_row_groups:
                raise IndexError(f"row group {group} out of range")
            table = parquet.read_row_group(group, columns=columns, use_threads=False)
            if any(table[name].null_count for name in table.column_names):
                raise ValueError(f"null values in label row group {group}")
            seq_ix, need = validate_sequence(table)
            targets = np.column_stack([table[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32, copy=False)
            if not np.isfinite(targets).all():
                raise ValueError(f"nonfinite targets in row group {group}")
            mask = table["is_scored"].to_numpy().astype(bool) if "is_scored" in table.column_names else None
            if mask is not None and mask[:WARMUP].any():
                raise ValueError(f"scoring mask selects warm-up in row group {group}")
            yield SequenceLabels(group, seq_ix, targets, need, mask)
    finally:
        parquet.close()


def file_fingerprint(path: str | Path, sample_bytes: int = 65536) -> dict:
    """Fast change detector, explicitly not a cryptographic hash of the full dataset."""
    path = Path(path).resolve()
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        digest.update(stream.read(sample_bytes))
        stream.seek(max(0, before.st_size - sample_bytes))
        digest.update(stream.read(sample_bytes))
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"file changed during fingerprinting: {path}")
    return {"size_bytes": before.st_size, "mtime_ns": before.st_mtime_ns,
            "sample_sha256": digest.hexdigest(), "sample_bytes_per_edge": sample_bytes,
            "method": "size+mtime_ns+sha256(first_and_last_bytes); not a full-file hash"}


def _inventory(path: str | Path, *, read_ids: bool = True, mask_only: bool = False) -> dict:
    path = Path(path).resolve()
    fingerprint = file_fingerprint(path)
    parquet = pq.ParquetFile(path)
    try:
        _check_schema(parquet, mask_only=mask_only)
        metadata = parquet.metadata
        groups, seen = [], set()
        for group in range(metadata.num_row_groups):
            rows = metadata.row_group(group).num_rows
            if rows != SEQUENCE_LENGTH:
                raise ValueError(f"{path.name} row group {group} has {rows} rows, expected {SEQUENCE_LENGTH}")
            item = {"row_group": group, "rows": rows}
            if read_ids:
                column = parquet.read_row_group(group, columns=["seq_ix"], use_threads=False)["seq_ix"]
                if column.null_count:
                    raise ValueError(f"null sequence identifier in {path.name} group {group}")
                ids = column.to_numpy()
                seq_ix = int(ids[0])
                if not np.all(ids == seq_ix):
                    raise ValueError(f"multiple sequence identifiers in {path.name} group {group}")
                if seq_ix in seen:
                    raise ValueError(f"duplicate sequence identifier {seq_ix} in {path.name}")
                seen.add(seq_ix)
                item["seq_ix"] = seq_ix
            groups.append(item)
        if file_fingerprint(path) != fingerprint:
            raise ValueError(f"file changed during inventory: {path}")
        return {"path": str(path), "fingerprint": fingerprint,
                "num_rows": metadata.num_rows, "num_row_groups": metadata.num_row_groups,
                "schema": [{"name": f.name, "dtype": str(f.type)} for f in parquet.schema_arrow],
                "groups": groups, "all_group_row_counts_checked": True,
                "all_group_identifiers_checked": read_ids}
    finally:
        parquet.close()


def make_splits(num_valid_groups: int, *, seed: int = 20260912,
                development_count: int = 1400, confirmation_count: int = 473,
                smoke_count: int = 32, screen_count: int = 256,
                force_dev=INSPECTED_VALID_GROUPS) -> dict[str, list[int]]:
    if development_count + confirmation_count != num_valid_groups:
        raise ValueError("development and confirmation counts must cover all validation sequences")
    if not 0 < smoke_count <= screen_count <= development_count or confirmation_count <= 0:
        raise ValueError("require 0 < smoke <= screen <= development and a nonempty confirmation split")
    forced = sorted({int(g) for g in force_dev if 0 <= int(g) < num_valid_groups})
    if len(forced) > development_count:
        raise ValueError("too many forced development groups")
    rng = np.random.default_rng(seed)
    remainder = rng.permutation([g for g in range(num_valid_groups) if g not in set(forced)]).tolist()
    dev = forced + remainder[:development_count - len(forced)]
    rng.shuffle(dev)
    confirmation = remainder[development_count - len(forced):]
    return {"dev": dev, "confirmation": confirmation,
            "smoke": dev[:smoke_count], "screen": dev[:screen_count]}


def build_manifest(train_path: str | Path, valid_path: str | Path,
                   mask_path: str | Path | None = None, *, seed: int = 20260912,
                   development_count: int = 1400, confirmation_count: int = 473,
                   smoke_count: int = 32, screen_count: int = 256) -> dict:
    """Inventory all narrow ID columns once; fully validate only bounded sample groups.

    Confirmation groups are never opened for feature/label inspection here. The ID
    scan checks uniqueness/disjointness, not hidden temporal overlap between IDs.
    """
    train = _inventory(train_path)
    valid = _inventory(valid_path)
    train_ids = {g["seq_ix"] for g in train["groups"]}
    valid_ids = {g["seq_ix"] for g in valid["groups"]}
    overlap = train_ids & valid_ids
    if overlap:
        raise ValueError(f"train/validation sequence IDs overlap: {sorted(overlap)[:10]}")
    splits = make_splits(valid["num_row_groups"], seed=seed, development_count=development_count,
                         confirmation_count=confirmation_count, smoke_count=smoke_count,
                         screen_count=screen_count)
    splits["train"] = list(range(train["num_row_groups"]))
    sample_train = sorted({0, train["num_row_groups"] // 2})
    sample_valid = [g for g in INSPECTED_VALID_GROUPS if g in set(splits["dev"])]
    if not sample_valid:
        sample_valid = splits["dev"][:1]
    samples = {"train": [], "valid": []}
    datasets = {"train": train, "valid": valid}
    mask_parquet = None
    if mask_path is not None:
        datasets["valid_mask"] = _inventory(mask_path, read_ids=False, mask_only=True)
        if datasets["valid_mask"]["num_rows"] != valid["num_rows"] or datasets["valid_mask"]["num_row_groups"] != valid["num_row_groups"]:
            raise ValueError("validation mask metadata does not match validation")
        mask_parquet = pq.ParquetFile(mask_path)
    try:
        for name, selected in [("train", sample_train), ("valid", sample_valid)]:
            for sequence in iter_sequences(datasets[name]["path"], selected):
                item = {"row_group": sequence.row_group, "seq_ix": sequence.seq_ix,
                        "finite_features_and_targets": True, "steps_and_warmup_checked": True,
                        "feature_min": float(sequence.features.min()),
                        "feature_max": float(sequence.features.max()),
                        "label_clip_fraction": (np.abs(sequence.targets) > 2).mean(axis=0).tolist()}
                if sequence.mask is not None:
                    item["selected_rows"] = int(sequence.mask.sum())
                    if mask_parquet is not None:
                        table = mask_parquet.read_row_group(sequence.row_group, use_threads=False)
                        matches = (not any(table[c].null_count for c in table.column_names)
                                   and np.all(table["seq_ix"].to_numpy() == sequence.seq_ix)
                                   and np.array_equal(table["step_in_seq"].to_numpy(), np.arange(SEQUENCE_LENGTH))
                                   and np.array_equal(table["is_scored"].to_numpy(), sequence.mask))
                        if not matches:
                            raise ValueError(f"standalone validation mask mismatch in group {sequence.row_group}")
                        item["standalone_mask_matches"] = True
                samples[name].append(item)
    finally:
        if mask_parquet is not None:
            mask_parquet.close()
    return {"version": MANIFEST_VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
            "seed": seed, "sequence_length": SEQUENCE_LENGTH, "warmup": WARMUP,
            "feature_columns": FEATURE_COLUMNS, "datasets": datasets, "splits": splits,
            "audit": {"train_validation_ids_disjoint": True,
                      "sampled_full_groups": samples,
                      "full_features_steps_labels_scan_performed": False,
                      "standalone_mask_checked_only_on_sampled_development_groups": mask_path is not None,
                      "confirmation_model_outputs_inspected": False}}


def save_manifest(manifest: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)


def load_manifest(path: str | Path, *, verify_fingerprints: bool = True) -> dict:
    manifest = json.loads(Path(path).read_text())
    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError("unsupported manifest version")
    if verify_fingerprints:
        for item in manifest["datasets"].values():
            if file_fingerprint(item["path"]) != item["fingerprint"]:
                raise ValueError(f"dataset changed since manifest creation: {item['path']}")
    return manifest
