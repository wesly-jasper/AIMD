"""
Failure modes, end to end: upload, analysis, API, and every report format.

A detector that is missing, runs out of GPU memory, returns an invalid number
or times out must leave a record that says so -- never a finding made from the
failure -- and the analysis, its API response and its reports must still be
produced, with every detector's status and the run's provenance intact.

B-Free is the detector under test throughout, with a stub runner; the rest of
the pipeline is real.
"""
import math
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.services.detection.detectors.image.bfree_detector import BFreeDetector
from app.services.detection.router.media_detection_router import MediaDetectionRouter
from app.services.detection.runtime import detector_health

client = TestClient(app)


class _Runner:
    """Behaves as told: a logit, an exception to raise, or a delay."""

    version_tag = "stub"

    def __init__(self, *, logit=2.0, score=None, raises=None, sleep=0.0):
        self.logit, self.score, self.raises, self.sleep = logit, score, raises, sleep
        self.calls = 0

    def predict(self, path):
        self.calls += 1
        if self.sleep:
            time.sleep(self.sleep)
        if self.raises is not None:
            raise self.raises
        score = self.score if self.score is not None else (
            1 / (1 + math.exp(-self.logit)) if math.isfinite(self.logit) else self.logit)
        return {"logit": self.logit, "score": score, "metadata": {}}


def _image(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    img = rng.integers(60, 200, size=(96, 128, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


@pytest.fixture
def bfree_with(monkeypatch):
    """Replace B-Free in the real image route with one built on ``runner``."""
    def install(detector):
        original = MediaDetectionRouter.route

        def route(self, file_path, media_type):
            result = original(self, file_path, media_type)
            if media_type == "image":
                kept = [d for d in result.get("detections", [])
                        if getattr(d, "detector", None) != "BFreeDetector"]
                from app.services.detection.runtime import run_detector
                result["detections"] = kept + [run_detector(
                    detector, file_path, timeout_s=settings.detector_timeout_seconds,
                    retry_on_memory_error=settings.detector_retry_on_memory_error,
                )]
            return result

        monkeypatch.setattr(MediaDetectionRouter, "route", route)
        detector_health.reset()
        return detector

    yield install
    detector_health.reset()


def _analyse(seed: int) -> tuple[str, dict]:
    media_id = client.post(
        "/api/v1/media/upload", files={"file": (f"f{seed}.png", _image(seed), "image/png")},
    ).json()["media_id"]
    response = client.post("/api/v1/analysis/", json={"media_id": media_id})
    assert response.status_code in (200, 201, 202), response.text
    analysis_id = response.json()["analysis_id"]
    detail = client.get(f"/api/v1/analysis/{analysis_id}")
    assert detail.status_code == 200, detail.text
    return analysis_id, detail.json()


def _bfree(detail: dict) -> dict:
    return next(d for d in detail["detections"] if d["detector"] == "BFreeDetector")


def _reports_render(analysis_id: str) -> dict:
    """Every report format is served, and the JSON one is strict JSON."""
    out = {}
    for fmt in ("json", "md", "html"):
        r = client.get(f"/api/v1/analysis/{analysis_id}/report", params={"format": fmt})
        assert r.status_code == 200, (fmt, r.text[:300])
        out[fmt] = r.json() if fmt == "json" else r.text
    return out


def _no_finding_from(detail: dict):
    ai = detail["ai_generation"]
    assert ai["status"] not in ("AI_GENERATION_INDICATED", "NOT_INDICATED", "CONFLICTING"), ai["status"]


@pytest.mark.parametrize("logit,score", [
    (float("nan"), None),
    (float("inf"), None),
    (2.0, float("nan")),
    (2.0, 1.7),     # not a probability
    (2.0, -0.2),
])
def test_an_invalid_score_is_a_failure_not_a_finding(bfree_with, monkeypatch, logit, score):
    monkeypatch.setattr(settings, "ai_generation_advisory_detectors", ["UnivFDDetector"])
    bfree_with(BFreeDetector(_Runner(logit=logit, score=score)))
    analysis_id, detail = _analyse(11)
    b = _bfree(detail)
    assert b["execution_status"] == "FAILED", b
    assert "invalid" in (b.get("status_reason") or "").lower() or "finite" in (b.get("status_reason") or "").lower()
    if not any(d["detector"] == "UnivFDDetector" and d["execution_status"] == "COMPLETED"
               for d in detail["detections"]):
        _no_finding_from(detail)
    reports = _reports_render(analysis_id)
    assert "BFreeDetector" in reports["md"]


def test_gpu_memory_pressure_is_retried_then_recorded(bfree_with, monkeypatch):
    monkeypatch.setattr(settings, "detector_retry_on_memory_error", True)
    runner = _Runner(raises=RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    bfree_with(BFreeDetector(runner))
    analysis_id, detail = _analyse(12)
    b = _bfree(detail)
    assert b["execution_status"] == "FAILED"
    assert runner.calls == 2  # one retry after releasing cached memory
    assert "memory" in (b.get("status_reason") or "").lower()
    assert detail["status"] in ("COMPLETED", "PARTIAL", "COMPLETED_WITH_ERRORS"), detail["status"]
    _reports_render(analysis_id)


def test_a_detector_that_hangs_times_out_and_the_analysis_finishes(bfree_with, monkeypatch):
    monkeypatch.setattr(settings, "detector_timeout_seconds", 0.5)
    bfree_with(BFreeDetector(_Runner(sleep=3.0)))
    started = time.perf_counter()
    analysis_id, detail = _analyse(13)
    assert time.perf_counter() - started < 30
    b = _bfree(detail)
    assert b["execution_status"] == "FAILED"
    assert "time" in (b.get("status_reason") or "").lower()
    _reports_render(analysis_id)


def test_missing_weights_are_unavailable_with_a_reason(bfree_with, monkeypatch, tmp_path):
    from app.services.detection.detector_factory import DetectorFactory

    monkeypatch.setattr(settings, "ai_generation_advisory_detectors", ["UnivFDDetector"])
    monkeypatch.setattr(settings, "bfree_enabled", True)
    monkeypatch.setattr(settings, "bfree_model_path", tmp_path / "no-such-model")
    detector = DetectorFactory.create_bfree()
    bfree_with(detector)
    analysis_id, detail = _analyse(14)
    b = _bfree(detail)
    assert b["availability"] == "UNAVAILABLE"
    assert "FileNotFoundError" in (b.get("status_reason") or "")
    _no_finding_from(detail)  # UnivFD may have run, but it is advisory and never decides
    _reports_render(analysis_id)


def test_every_result_keeps_statuses_versions_and_the_verdict_policy(bfree_with, monkeypatch):
    bfree_with(BFreeDetector(_Runner(logit=3.0)))
    analysis_id, detail = _analyse(15)

    for d in detail["detections"]:
        assert d["detector_version"], d["detector"]
        assert d["availability"] and d["execution_status"], d["detector"]

    config = detail["run"]["config_snapshot"]
    policy = config["verdict_policy"]
    assert policy["version"] and len(policy["rules_sha256"]) == 64
    assert "calibration_files" in config and "fusion_calibration_path" in config

    reports = _reports_render(analysis_id)
    package = reports["json"]
    assert package["run"]["config_snapshot"]["verdict_policy"] == policy
    assert {d["detector"] for d in package["detector_statuses"]} >= {"BFreeDetector"}
    md = reports["md"]
    assert "Run Provenance" in md and policy["rules_sha256"][:12] in md
    assert "BFreeDetector" in reports["html"]
