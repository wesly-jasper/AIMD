"""Measure the detectors on a labelled dataset, repeatably, and compare runs.

Every model or threshold change is judged against a baseline run on the same
held-out split (docs/EVALUATION.md). Usage (from backend/, with the venv python):

    # 1. Assign leakage-safe splits by source image (writes a new manifest)
    python scripts/evaluate.py split sets/web/manifest.json --out sets/web/manifest.split.json \\
        --val 0.4 --test 0.6

    # 2. Run every file through the production detectors and decision rules
    python scripts/evaluate.py run sets/web/manifest.split.json --out runs/baseline

    # 3. Metrics with 95% intervals, per branch and subgroup
    python scripts/evaluate.py metrics runs/baseline --split test

    # 4. Fit calibration on the validation split, then re-run and compare on test
    python scripts/evaluate.py calibrate runs/baseline --out models/calibration.json
    CALIBRATION_PATH=models/calibration.json python scripts/evaluate.py run ... --out runs/calibrated
    python scripts/evaluate.py compare runs/baseline runs/calibrated --split test

Several manifests can be given to ``run``; their items are evaluated together.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.services.evaluation.dataset import (  # noqa: E402
    assign_splits,
    load_manifest,
    validate,
    write_manifest,
)


def _pct(value) -> str:
    if not isinstance(value, dict) or value.get("rate") is None:
        return "–"
    lo, hi = value["ci95"]
    return f"{value['k']}/{value['n']} = {value['rate']:.0%} [{lo:.0%}–{hi:.0%}]"


def cmd_split(args) -> int:
    items = load_manifest(args.manifest)
    fractions = {k: v for k, v in (("train", args.train), ("val", args.val), ("test", args.test)) if v}
    assign_splits(items, fractions, seed=args.seed, by_generator=args.by_generator, overwrite=True)
    problems = validate(items)
    if problems:
        print("\n".join(problems[:20]))
        return 1
    out = Path(args.out)
    if out.resolve().parent != Path(args.manifest).resolve().parent:
        print("--out must be in the same directory as the manifest (file paths are relative to it)")
        return 1
    write_manifest(items, out)
    counts: dict = {}
    for i in items:
        counts.setdefault(i.split, {}).setdefault(i.label, 0)
        counts[i.split][i.label] += 1
    print(json.dumps(counts, indent=1))
    return 0


def cmd_run(args) -> int:
    from app.services.evaluation.runner import run_evaluation

    items = [item for path in args.manifests for item in load_manifest(path)]
    if args.split:
        items = [i for i in items if i.split == args.split]
    if args.limit:
        items = items[: args.limit]
    engine = None
    if args.only:
        # A subset of the production detectors, for screening one question
        # quickly. Verdicts of the other branch are then incomplete by design.
        from app.services.detection.detection_container import get_detection_container
        from app.services.detection.engine.image_detection_engine import ImageDetectionEngine

        wanted = set(args.only.split(","))
        chosen = [d for d in get_detection_container().router.image_engine.detectors
                  if type(d).__name__ in wanted]
        missing = wanted - {type(d).__name__ for d in chosen}
        if missing:
            print(f"not configured: {sorted(missing)}")
            return 1
        engine = ImageDetectionEngine(detectors=chosen)
    problems = validate(items)
    if problems:
        print("Dataset problems:\n  " + "\n  ".join(problems[:20]))
        return 1

    def progress(n, total, rec):
        v = rec.get("verdicts", {})
        print(
            f"[{n}/{total}] {rec['id'][:40]:40} {rec['label']:12} "
            f"manip={v.get('manipulation')!s:34} ai={v.get('ai_generation')!s:24} "
            f"{rec.get('seconds', 0):.1f}s{'  ERROR ' + rec['error'][:80] if rec.get('error') else ''}",
            flush=True,
        )

    path = run_evaluation(items, args.out, engine=engine, progress=progress)
    print(f"predictions: {path}")
    return 0


def _metrics(run_dir: Path, split):
    from app.services.evaluation.metrics import compute_report
    from app.services.evaluation.runner import load_predictions

    return compute_report(load_predictions(run_dir / "predictions.jsonl"), split=split)


def _print_report(report) -> None:
    print(f"\nSplit: {report['split']}  files: {report['files']}  run errors: {report['run_errors']}  "
          f"silent failures: {report['silent_failures']}")
    for branch in ("manipulation", "ai_generation"):
        b = report[branch]
        print(f"\n{branch}: {b['positives']} positive / {b['negatives']} negative")
        print(f"  recall             {_pct(b['recall'])}")
        print(f"  false-positive     {_pct(b['false_positive_rate'])}")
        print(f"  precision          {_pct(b['precision'])}")
        print(f"  balanced accuracy  {b['balanced_accuracy']}")
        print(f"  abstained (pos/neg) {_pct(b['abstained_positive'])} / {_pct(b['abstained_negative'])}")
        print(f"  score AUC          {b['score_auc']} {b['score_auc_ci95'] or ''}")
        for name, m in (b.get("per_detector") or {}).items():
            print(f"  [{name} on its own] recall {_pct(m['recall'])}  FPR {_pct(m['false_positive_rate'])}  "
                  f"balanced {m['balanced_accuracy']}  AUC {m['score_auc']} {m['score_auc_ci95'] or ''}")
            for key, sub in m["subgroups"].items():
                if key.startswith(("generator=", "label=")):
                    parts = [f"recall {_pct(sub['recall'])}" if "recall" in sub else "",
                             f"FPR {_pct(sub['false_positive_rate'])}" if "false_positive_rate" in sub else ""]
                    print(f"      {key:36} {'  '.join(p for p in parts if p)}")
        for key, sub in b["subgroups"].items():
            parts = [f"recall {_pct(sub['recall'])}" if "recall" in sub else "",
                     f"FPR {_pct(sub['false_positive_rate'])}" if "false_positive_rate" in sub else ""]
            print(f"    {key:38} {'  '.join(p for p in parts if p)}")
    loc = report["localisation"]
    if loc.get("n"):
        print(f"\nlocalisation: n={loc['n']} mean IoU {loc['mean_iou']} median {loc['median_iou']} "
              f"mean F1 {loc['mean_f1']} IoU>=0.5 {_pct(loc['iou_at_least_0_5'])}")
    print("\ndetectors:")
    for name, d in report["detectors"].items():
        print(f"  {name:28} {json.dumps(d)}")


def cmd_metrics(args) -> int:
    run_dir = Path(args.run)
    report = _metrics(run_dir, args.split)
    target = run_dir / f"metrics.{args.split or 'all'}.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print_report(report)
    print(f"\nwritten: {target}")
    return 0


def cmd_compare(args) -> int:
    from app.services.evaluation.metrics import compare

    base = _metrics(Path(args.baseline), args.split)
    cand = _metrics(Path(args.candidate), args.split)
    result = compare(base, cand)
    for branch in ("manipulation", "ai_generation"):
        print(f"\n{branch}")
        for key, row in result[branch].items():
            print(f"  {key:20} {row.get('baseline')!s:>8} -> {row.get('candidate')!s:<8} "
                  f"{row.get('delta', ''):>8}  {row.get('verdict', '')}")
    print(f"\nlocalisation mean IoU: {result['localisation_mean_iou']}")
    print(f"regressions: {result['regressions'] or 'none'}")
    return 1 if result["regressions"] else 0


def cmd_calibrate(args) -> int:
    from app.services.detection import calibration
    from app.services.evaluation.metrics import BRANCH_SCORE_DETECTOR
    from app.services.evaluation.runner import load_predictions

    run_dir = Path(args.run)
    records = [r for r in load_predictions(run_dir / "predictions.jsonl") if r.get("split") == args.split]
    info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    fitted = {}
    for branch, detector in BRANCH_SCORE_DETECTOR.items():
        rows = [
            (r["detectors"][detector]["raw_score"], r["targets"][branch], r["detectors"][detector]["version"])
            for r in records
            if r["targets"].get(branch) is not None
            and r["detectors"].get(detector, {}).get("raw_score") is not None
        ]
        versions = {v for _, _, v in rows}
        if len(versions) != 1:
            print(f"{detector}: skipped ({len(rows)} scored rows, versions {sorted(map(str, versions))})")
            continue
        scores, targets = [s for s, _, _ in rows], [t for _, t, _ in rows]
        if all(targets) or not any(targets):
            print(f"{detector}: skipped (needs both classes on the {args.split} split)")
            continue
        try:
            cal = calibration.fit(
                detector, versions.pop(), scores, targets, target_fpr=args.target_fpr,
                fitted_on={"dataset_sha256": info.get("dataset_sha256"), "split": args.split,
                           "run": run_dir.name, "commit": info.get("commit")},
            )
        except ValueError as exc:
            print(f"{detector}: not calibrated -- {exc}")
            continue
        if cal.validation["ece_calibrated"] > cal.validation["ece_raw"]:
            print(f"{detector}: warning -- calibration error rose on {args.split} "
                  f"({cal.validation['ece_raw']} -> {cal.validation['ece_calibrated']})")
        fitted[detector] = cal
        print(f"{detector}: a={cal.a:.3f} b={cal.b:.3f} threshold={cal.threshold:.4f} "
              f"validation={cal.validation} n={cal.fitted_on['positives']}+/{cal.fitted_on['negatives']}-")
    if not fitted:
        print("nothing was calibrated")
        return 1
    calibration.save(fitted, args.out)
    print(f"written: {args.out} -- set CALIBRATION_PATH to apply it, then evaluate on the test split")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("split", help="assign train/val/test by source image")
    p.add_argument("manifest")
    p.add_argument("--out", required=True)
    p.add_argument("--train", type=float, default=0.0)
    p.add_argument("--val", type=float, default=0.4)
    p.add_argument("--test", type=float, default=0.6)
    p.add_argument("--seed", default="aimd-eval-v1")
    p.add_argument("--by-generator", action="store_true",
                   help="keep each AI generator wholly in one split")
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("run", help="predict every file")
    p.add_argument("manifests", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--split", help="only this split")
    p.add_argument("--limit", type=int)
    p.add_argument("--only", help="comma-separated detector classes to run (screening)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("metrics", help="metrics for a run")
    p.add_argument("run")
    p.add_argument("--split")
    p.set_defaults(func=cmd_metrics)

    p = sub.add_parser("compare", help="candidate run vs baseline run")
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.add_argument("--split", default="test")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("calibrate", help="fit calibration on a run's validation split")
    p.add_argument("run")
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--target-fpr", type=float, default=0.05)
    p.set_defaults(func=cmd_calibrate)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
