"""The frozen holdout: locked by content, never fitted on, never re-split."""
import importlib.util
import json
from pathlib import Path

from app.services.evaluation import holdout
from app.services.evaluation.dataset import EvalItem

BACKEND = Path(__file__).resolve().parents[2]


def _evaluate_module():
    spec = importlib.util.spec_from_file_location("evaluate_cli", BACKEND / "scripts" / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _items(tmp_path, n=4, split="holdout"):
    items = []
    for i in range(n):
        path = tmp_path / f"f{i}.bin"
        path.write_bytes(f"file {i}".encode())
        items.append(EvalItem(id=f"f{i}", path=path, label="real" if i % 2 else "ai_generated",
                              group=f"g{i}", split=split))
    return items


def test_lock_lists_holdout_files_by_content(tmp_path):
    items = _items(tmp_path)
    items[0].split = "val"
    lock = holdout.build_lock(items)
    assert lock["count"] == 3
    assert holdout.locked_ids(lock) == {"f1", "f2", "f3"}
    assert holdout.check(items, lock) == []


def test_changed_or_relabelled_holdout_files_are_caught(tmp_path):
    items = _items(tmp_path)
    lock = holdout.build_lock(items)
    items[1].path.write_bytes(b"edited after freezing")
    items[2].split = "val"
    problems = holdout.check(items, lock)
    assert any("f1" in p and "not in the lock" in p for p in problems)
    assert any("f2" in p and "'val'" in p for p in problems)


def test_a_renamed_holdout_file_is_still_locked(tmp_path):
    items = _items(tmp_path)
    lock = holdout.build_lock(items)
    renamed = EvalItem(id="other-name", path=items[0].path, label="real", group="g", split="test")
    assert holdout.is_locked(renamed, lock)


def test_freeze_refuses_to_replace_a_lock(tmp_path):
    cli = _evaluate_module()
    manifest = tmp_path / "manifest.json"
    (tmp_path / "files").mkdir()
    (tmp_path / "files" / "a.jpg").write_bytes(b"a")
    manifest.write_text(json.dumps([{"file": "a.jpg", "label": "real", "group": "a", "split": "holdout"}]))
    lock = tmp_path / "lock.json"
    args = type("A", (), {"lock": str(lock), "replace": False, "note": "", "manifests": [str(manifest)]})
    assert cli.cmd_freeze(args) == 0 and lock.is_file()
    assert cli.cmd_freeze(args) == 1


def test_calibrate_refuses_the_holdout(tmp_path):
    cli = _evaluate_module()
    args = type("A", (), {"run": [str(tmp_path)], "split": "holdout", "lock": None,
                          "out": str(tmp_path / "c.json"), "target_fpr": 0.05})
    assert cli.cmd_calibrate(args) == 1


def test_calibrate_refuses_locked_files_under_another_split(tmp_path):
    cli = _evaluate_module()
    items = _items(tmp_path)
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(holdout.build_lock(items)))
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text("{}")
    (run / "predictions.jsonl").write_text(json.dumps({"id": "f0", "split": "val"}) + "\n")
    args = type("A", (), {"run": [str(run)], "manifest": None, "split": "val", "lock": str(lock_path),
                          "out": str(tmp_path / "c.json"), "target_fpr": 0.05})
    assert cli.cmd_calibrate(args) == 1


def test_calibrate_merges_runs_and_counts_re_encoded_copies_once(tmp_path):
    cli = _evaluate_module()
    runs = []
    for name, detector in (("a", "BFreeDetector"), ("b", "UnivFDDetector")):
        run = tmp_path / name
        run.mkdir()
        (run / "run.json").write_text(json.dumps({"dataset_sha256": name, "commit": "c"}))
        rows = []
        for i in range(12):
            positive = i % 2 == 0
            for tags, suffix in ((["original"], ".jpg"), (["jpeg_q70"], "__jpeg_q70.jpg")):
                rows.append({"id": f"img{i}{suffix}", "split": "val", "tags": tags,
                             "targets": {"ai_generation": positive, "manipulation": None},
                             "detectors": {detector: {"raw_score": (0.8 if positive else 0.2) + i / 100,
                                                      "version": "1", "execution_status": "COMPLETED"}}})
        (run / "predictions.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        runs.append(str(run))
    out = tmp_path / "cal.json"
    args = type("A", (), {"run": runs, "manifest": None, "split": "val", "lock": str(tmp_path / "none.json"),
                          "out": str(out), "target_fpr": 0.2})
    assert cli.cmd_calibrate(args) == 0
    saved = json.loads(out.read_text())
    assert set(saved) == {"BFreeDetector", "UnivFDDetector"}  # one detector from each run
    fitted = saved["BFreeDetector"]["fitted_on"]
    assert (fitted["positives"], fitted["negatives"]) == (12, 12)
    assert (fitted["independent_positives"], fitted["independent_negatives"]) == (6, 6)
    assert fitted["runs"] == ["a", "b"]
