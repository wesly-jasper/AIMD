"""
The evaluation runner: every file through the production detectors and the
production decision rules, with every prediction saved.

It runs in process -- no HTTP, no database -- but the verdicts are produced by
``AnalysisPipeline._decide`` and ``_decide_ai_generation`` themselves, applied
to the same results the pipeline would store. A runner that re-implemented the
decision rule would measure its own copy, not the product.

Output directory:

    run.json            what ran: commit, config, detector versions, dataset hash
    predictions.jsonl   one line per file: targets, verdicts, every detector's
                        status / raw score / reason / time, localisation scores
    metrics.json        written by ``scripts/evaluate.py metrics``

Resumable: files already in predictions.jsonl are skipped, so an interrupted
run continues where it stopped.
"""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

from app.core.config import settings
from app.schemas.detection import DetectionResult
from app.services.detection.engine.image_detection_engine import (
    NON_MANIPULATION_FAMILIES,
    ImageDetectionEngine,
)
from app.services.evaluation.dataset import EvalItem, manifest_sha256
from app.services.evaluation.metrics import load_binary, localisation_scores

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"})
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm"})
#: TruFor's localization map is thresholded here for IoU, as in the reliability report.
LOCALISATION_THRESHOLD = 0.5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def as_stored_row(result: DetectionResult) -> SimpleNamespace:
    """A ``DetectionResult`` shaped like the stored detection the decision rules read."""
    return SimpleNamespace(
        id=result.detector,
        detector=result.detector,
        detector_version=result.detector_version,
        media_type=result.media_type,
        assessment=result.assessment.value,
        availability=result.availability.value,
        execution_status=result.execution_status.value,
        score_semantics=result.score_semantics.value,
        confidence=result.confidence,
        manipulation_detected=result.manipulation_detected,
        regions=[r.model_dump() for r in result.regions],
        detector_metadata=result.metadata,
        status_reason=result.status_reason,
    )


def raw_score(result: DetectionResult) -> Optional[float]:
    """The detector's uncalibrated output: what calibration is fitted on."""
    if not result.produced_observations:
        return None
    calibration = result.metadata.get("calibration") or {}
    if calibration.get("raw_score") is not None:
        return float(calibration["raw_score"])
    return float(result.confidence)


def _localisation(result: DetectionResult, mask: Path) -> Optional[dict[str, float]]:
    artefact = ((result.metadata.get("artefacts") or {}).get("localization_map") or {}).get("path")
    if not artefact:
        return None
    map_path = Path(settings.artefacts_dir) / artefact
    if not map_path.is_file():
        return None
    truth = load_binary(mask)
    pred = load_binary(map_path, truth.shape, LOCALISATION_THRESHOLD)
    return {k: round(v, 4) for k, v in localisation_scores(pred, truth).items()}


def predict_item(item: EvalItem, engine: ImageDetectionEngine, video_engine: Any = None) -> dict[str, Any]:
    """One file through the detectors and the decision rules, as a saved record."""
    from app.services.orchestration.analysis_pipeline import AnalysisPipeline

    record: dict[str, Any] = {
        "id": item.id,
        "file": item.path.name,
        "label": item.label,
        "group": item.group,
        "kind": item.kind,
        "generator": item.generator,
        "split": item.split,
        "tags": item.tags,
        "targets": {
            "manipulation": item.manipulation_target,
            "ai_generation": item.ai_generation_target,
        },
        "verdicts": {},
        "detectors": {},
    }
    started = time.perf_counter()
    try:
        record["sha256"] = _sha256(item.path)
        suffix = item.path.suffix.lower()
        if suffix in VIDEO_SUFFIXES and video_engine is not None:
            output = video_engine.detect(item.path)
            record["media_type"] = "video"
            record["keyframe_coverage"] = output.get("keyframe_coverage")
            record["shot_boundaries"] = output.get("shot_boundaries")
        elif suffix in IMAGE_SUFFIXES:
            output = engine.detect(item.path)
        else:
            raise ValueError(f"unsupported media type: {item.path.suffix}")
        results: list[DetectionResult] = output["detections"]
        rows = [as_stored_row(r) for r in results]

        decision = AnalysisPipeline._decide(rows)
        ai = AnalysisPipeline._decide_ai_generation(rows)
        record["verdicts"] = {"manipulation": decision.category, "ai_generation": ai.get("status")}
        record["decision"] = {
            "rule": decision.rule,
            "evidence_strength": decision.evidence_strength,
            "families": sorted(decision.families),
            "ai_generation": {k: v for k, v in ai.items() if k not in ("rationale", "limitations")},
        }
        for r in results:
            record["detectors"][r.detector] = {
                "availability": r.availability.value,
                "execution_status": r.execution_status.value,
                "score_semantics": r.score_semantics.value,
                "raw_score": raw_score(r),
                "confidence": r.confidence,
                "manipulation_detected": r.manipulation_detected,
                "regions": len(r.regions),
                "reason": r.status_reason,
                "timed_out": bool(r.metadata.get("timed_out")),
                "duration_ms": r.metadata.get("runtime_ms"),
                "version": r.detector_version,
                "evidential": ImageDetectionEngine._family(r.detector) not in NON_MANIPULATION_FAMILIES,
                # The detector's numeric by-products (e.g. TruFor's localization
                # peak), so an alternative decision rule can be evaluated from
                # the saved run without running the models again.
                "signals": {k: v for k, v in r.metadata.items()
                            if isinstance(v, (int, float)) and not isinstance(v, bool)},
            }
            if item.mask_path is not None and r.detector == "TruForDetector" and r.produced_observations:
                loc = _localisation(r, item.mask_path)
                if loc is not None:
                    record["localisation"] = loc
    except Exception as exc:
        logger.warning("Evaluation of %s failed", item.id, exc_info=True)
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["seconds"] = round(time.perf_counter() - started, 3)
    return record


def load_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10,
            cwd=Path(__file__).resolve().parent,
        )
        commit = out.stdout.strip() or None
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
        return f"{commit}{'+dirty' if dirty else ''}" if commit else None
    except Exception:
        return None


def _run_info(items: list[EvalItem], engine: ImageDetectionEngine) -> dict[str, Any]:
    from app.services.orchestration.analysis_pipeline import AnalysisPipeline

    device = None
    try:
        import torch

        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        pass
    return {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "commit": _git_commit(),
        "app_version": settings.app_version,
        "config": AnalysisPipeline._config_snapshot(),
        "detectors": {
            type(d).__name__: getattr(d, "version", None) for d in engine.detectors
        },
        "dataset_sha256": manifest_sha256(items),
        "files": len(items),
        "host": {"python": platform.python_version(), "platform": platform.platform(), "device": device},
    }


def run_evaluation(
    items: list[EvalItem],
    out_dir: str | Path,
    *,
    engine: Optional[ImageDetectionEngine] = None,
    video_engine: Any = None,
    progress: Optional[Callable[[int, int, dict[str, Any]], None]] = None,
) -> Path:
    """Predict every item not already in ``out_dir/predictions.jsonl``. Returns that path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if engine is None:
        from app.services.detection.detection_container import get_detection_container

        router = get_detection_container().router
        engine, video_engine = router.image_engine, router.video_engine

    predictions = out / "predictions.jsonl"
    done = {p["id"] for p in load_predictions(predictions)}
    info_path = out / "run.json"
    if not info_path.exists():
        info_path.write_text(json.dumps(_run_info(items, engine), indent=2), encoding="utf-8")

    todo = [i for i in items if i.id not in done]
    with predictions.open("a", encoding="utf-8") as handle:
        for n, item in enumerate(todo, 1):
            record = predict_item(item, engine, video_engine)
            handle.write(json.dumps(record, default=str) + "\n")
            handle.flush()
            if progress:
                progress(n, len(todo), record)

    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["finished_at"] = datetime.now(timezone.utc).isoformat()
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return predictions
