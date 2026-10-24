"""Run `python -m connectome --help` for the reproducible research workflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit", help="inventory datasets and freeze development/confirmation splits")
    audit.add_argument("--train", default="datasets/train.parquet")
    audit.add_argument("--valid", default="datasets/valid.parquet")
    audit.add_argument("--mask", default="datasets/valid_mask.parquet")
    audit.add_argument("--output", default="artifacts/manifest.json")
    prepare = sub.add_parser("prepare", help="inspect weights and create fixed-step/sequence/native artifacts")
    prepare.add_argument("--model", default="baseline/baseline.onnx")
    prepare.add_argument("--output", default="artifacts/models")
    for name in ("evaluate", "benchmark", "parity", "calibrate"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", default="datasets/valid.parquet")
        p.add_argument("--manifest", default="artifacts/manifest.json")
        p.add_argument("--split", choices=["smoke", "screen", "dev", "confirmation", "full"], default="smoke")
        p.add_argument("--groups", help="comma-separated row groups; still checked against confirmation split")
        p.add_argument("--limit", type=int)
        p.add_argument("--model", default="baseline/baseline.onnx")
        p.add_argument("--output", default=f"artifacts/{name}.json")
        p.add_argument("--frozen-config", help="required for candidate access to confirmation labels")
        if name in {"evaluate", "parity"}:
            p.add_argument("--mode", choices=["run", "reference", "iobinding", "ortvaluevector", "native_row", "native_packed"], default="run")
            p.add_argument("--bulk-model", help="offline only; requires --parity-report when evaluating")
        if name == "evaluate":
            p.add_argument("--cache")
            p.add_argument("--postprocess")
            p.add_argument("--parity-report")
        elif name == "benchmark":
            p.add_argument("--modes", default="run,iobinding,ortvaluevector")
            p.add_argument("--repetitions", type=int, default=3)
            p.add_argument("--affinity", action="store_true")
            p.add_argument("--native-weights")
            p.add_argument("--native-library")
        elif name == "parity":
            p.add_argument("--original", default="baseline/baseline.onnx")
            p.add_argument("--native-weights")
            p.add_argument("--native-library")
        else:
            p.add_argument("--cache", required=True)
    for name in ("fit-head", "fit-affine", "fine-tune", "distill"):
        p = sub.add_parser(name)
        p.add_argument("--model", default="baseline/baseline.onnx")
        p.add_argument("--train", default="datasets/train.parquet")
        p.add_argument("--output", default=f"artifacts/{name}")
        p.add_argument("--max-sequences", type=int)
        p.add_argument("--seed", type=int, default=20260912)
        p.add_argument("--device", default="cpu")
        p.add_argument("--threads", type=int, default=1)
        p.add_argument("--resume")
        if name == "fit-head":
            p.add_argument("--head-state", choices=["last", "both"], default="last")
            p.add_argument("--optimizer", choices=["adamw", "ridge"], default="adamw")
            p.add_argument("--penalty", type=float, default=1e-3)
            p.add_argument("--learning-rate", type=float, default=1e-3)
            p.add_argument("--minibatch-size", type=int, default=4096)
        elif name == "fit-affine":
            p.add_argument("--penalty", type=float, default=1e-2)
        else:
            p.add_argument("--epochs", type=int, default=3)
            p.add_argument("--learning-rate", type=float, default=1e-5 if name == "fine-tune" else 1e-3)
            p.add_argument("--teacher-weight", type=float, default=.1)
            p.add_argument("--batch-size", type=int, default=8)
            p.add_argument("--chunk-size", type=int, default=256)
            if name == "distill":
                p.add_argument("--hidden-size", type=int, choices=[64, 96], default=96)
    freeze = sub.add_parser("freeze", help="bind final configuration to model/config hashes before confirmation")
    freeze.add_argument("--config", default="submission_config.json")
    freeze.add_argument("--output", default="artifacts/frozen_candidate.json")
    freeze.add_argument("--manifest", default="artifacts/manifest.json")
    freeze.add_argument("--dev-report", help="optional complete development report bound into the freeze")
    select = sub.add_parser("select", help="choose highest development WP among documented feasible candidates")
    select.add_argument("--candidates", required=True, help="JSON array of candidate records")
    select.add_argument("--manifest", default="artifacts/manifest.json")
    select.add_argument("--output", default="artifacts/development_selection.json")
    confirm = sub.add_parser("confirm", help="make the single frozen candidate versus original baseline decision")
    confirm.add_argument("--frozen-config", default="artifacts/frozen_candidate.json")
    confirm.add_argument("--candidate-report", required=True)
    confirm.add_argument("--baseline-report", default="artifacts/baseline_confirmation.json")
    confirm.add_argument("--output", default="artifacts/final_selection.json")
    package = sub.add_parser("package", help="build and verify a small inference-only ZIP")
    package.add_argument("--config", default="submission_config.json")
    package.add_argument("--output", default="dist/submission.zip")
    return root


def _groups(args):
    from .data import load_manifest
    from .evaluation import file_hash
    manifest = load_manifest(args.manifest, verify_fingerprints=False)
    # A mounted dataset may have a different path, but its contents must match.
    from .data import file_fingerprint
    expected = manifest["datasets"]["valid"]["fingerprint"]
    actual = file_fingerprint(args.dataset)
    if any(actual[k] != expected[k] for k in ("size_bytes", "sample_sha256")):
        raise ValueError("dataset does not match the frozen manifest")
    groups = ([int(x) for x in args.groups.split(",")] if args.groups else
              list(range(manifest["datasets"]["valid"]["num_row_groups"])) if args.split == "full" else
              manifest["splits"][args.split])
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        groups = groups[:args.limit]
    if not groups or len(set(groups)) != len(groups) or any(g < 0 or g >= manifest["datasets"]["valid"]["num_row_groups"] for g in groups):
        raise ValueError("groups must be unique, nonempty and within the manifest")
    args._candidate_confirmation = False
    if set(groups) & set(manifest["splits"]["confirmation"]):
        original = Path(__file__).resolve().parents[1] / "baseline" / "baseline.onnx"
        is_original = file_hash(args.model) == file_hash(original)
        if not (args.command == "evaluate" and is_original and not args.postprocess):
            if not args.frozen_config or args.command != "evaluate":
                raise ValueError("confirmation is reserved for a frozen final candidate; use development groups")
            from .selection import validate_frozen
            frozen = validate_frozen(args.frozen_config)
            if Path(frozen["manifest_path"]) != Path(args.manifest).resolve():
                raise ValueError("confirmation manifest differs from frozen candidate")
            if set(groups) != set(frozen["confirmation_groups"]):
                raise ValueError("candidate confirmation must use the exact full confirmation split")
            args._candidate_confirmation = True
    return groups


def main(argv=None):
    args = parser().parse_args(argv)
    from .evaluation import file_hash, save_json
    from .global_metric import METRIC_ID
    if args.command == "audit":
        from .data import build_manifest, save_manifest
        result = build_manifest(args.train, args.valid, args.mask)
        save_manifest(result, args.output)
        print(f"Frozen dataset manifest: {args.output}")
        return
    if args.command == "prepare":
        from .model_io import inspect_model, simplify_model, export_sequence_model, export_native_weights
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        result = inspect_model(args.model)
        simplify_model(args.model, output / "simplified.onnx")
        export_sequence_model(args.model, output / "sequence.onnx")
        export_native_weights(args.model, output / "weights.npy")
        save_json(output / "model_manifest.json", result)
    elif args.command in {"evaluate", "benchmark", "parity", "calibrate"}:
        groups = _groups(args)
        if args.command == "evaluate":
            from .evaluation import evaluate
            if args.mode.startswith("native"):
                raise ValueError("use parity/benchmark for unpromoted native candidates")
            if args.bulk_model:
                if not args.parity_report:
                    raise ValueError("offline bulk evaluation requires a successful --parity-report")
                parity = json.loads(Path(args.parity_report).read_text())
                if (not parity.get("passed") or parity.get("original_sha256") != file_hash(args.model)
                        or parity.get("bulk_model_sha256") != file_hash(args.bulk_model)):
                    raise ValueError("parity report does not validate this model")
            postprocess = json.loads(Path(args.postprocess).read_text()) if args.postprocess else None
            if args._candidate_confirmation:
                # Finish predictable failures before consuming the one confirmation
                # receipt. Nothing below this preflight has inspected candidate outputs.
                if args.cache and Path(args.cache).exists():
                    raise FileExistsError("confirmation cache path already exists")
                if Path(args.output).exists():
                    raise FileExistsError("confirmation report must use a new output path")
                from .runtime import OnnxBackend, StreamingPredictor
                preflight = StreamingPredictor(OnnxBackend(args.model, args.mode), postprocess)
                preflight.backend.close()
                del preflight
                from .selection import claim_confirmation
                claim_confirmation(args.frozen_config, model_path=args.model, groups=groups,
                                   report_path=args.output, postprocess=postprocess, mode=args.mode,
                                   bulk_model=args.bulk_model, parity_report=args.parity_report, metric_id=METRIC_ID)
            result = evaluate(args.dataset, args.model, groups, args.mode, args.cache, args.output,
                              postprocess, args.bulk_model)
        elif args.command == "benchmark":
            from .evaluation import benchmark
            result = benchmark(args.dataset, args.model, groups, args.modes.split(","), args.repetitions,
                               args.native_weights, args.native_library, args.affinity, args.output)
        elif args.command == "parity":
            from .evaluation import check_parity
            result = check_parity(args.dataset, args.original, args.model, groups, args.mode,
                                  args.native_weights, args.native_library, args.bulk_model, args.output)
        else:
            from .calibration import PredictionCache, search_calibration
            result = search_calibration(args.dataset, PredictionCache.open(args.cache), groups, args.output)
    elif args.command in {"fit-head", "fit-affine", "fine-tune", "distill"}:
        from . import training
        shared = dict(max_sequences=args.max_sequences, seed=args.seed, device=args.device,
                      threads=args.threads, resume_from=args.resume)
        if args.command == "fit-head":
            result = training.fit_head(args.model, args.train, args.output, **shared,
                                       head_state=args.head_state, optimizer=args.optimizer, penalty=args.penalty,
                                       learning_rate=args.learning_rate, minibatch_size=args.minibatch_size)
        elif args.command == "fit-affine":
            result = training.fit_affine(args.model, args.train, args.output, **shared, ridge=args.penalty)
        else:
            shared.update(epochs=args.epochs, learning_rate=args.learning_rate, distill_weight=args.teacher_weight,
                          batch_size=args.batch_size, chunk_size=args.chunk_size)
            function = training.fine_tune if args.command == "fine-tune" else training.train_student
            if args.command == "distill":
                shared["hidden_size"] = args.hidden_size
            result = function(args.model, args.train, args.output, **shared)
    elif args.command == "freeze":
        from .selection import freeze_candidate
        result = freeze_candidate(args.config, args.output, manifest_path=args.manifest,
                                  development_report=args.dev_report)
    elif args.command == "select":
        from .selection import select_development_candidate
        result = select_development_candidate(json.loads(Path(args.candidates).read_text()),
                                               manifest_path=args.manifest, output_path=args.output)
    elif args.command == "confirm":
        from .selection import compare_confirmation_once
        result = compare_confirmation_once(args.frozen_config, args.candidate_report,
                                           args.baseline_report, args.output)
    else:
        from .packaging import package_submission
        result = package_submission(args.config, args.output)
    # Large per-sequence reports stay in their artifact files.
    print(json.dumps({k:v for k,v in result.items() if k not in {"per_sequence", "candidates", "baseline", "best"}},
                     indent=2, default=str, allow_nan=False))


if __name__ == "__main__":
    main()
