"""
An advisory AI-generation detector, end to end: upload, analysis, API, report.

UnivFD and B-Free run with stub runners that disagree. B-Free decides the
branch; UnivFD is listed, marked as not voting, in the analysis response, the
evidence package and every served report format.
"""
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.services.detection.detectors.image.bfree_detector import BFreeDetector
from app.services.detection.detectors.image.univfd_detector import UnivFDDetector
from app.services.detection.router.media_detection_router import MediaDetectionRouter

client = TestClient(app)


class _Runner:
    def __init__(self, logit: float):
        self.version_tag = "stub"
        self._logit = logit

    def predict(self, path):
        score = float(1 / (1 + np.exp(-self._logit)))
        return {"logit": self._logit, "score": score, "metadata": {}}


def _image() -> bytes:
    img = np.full((96, 128, 3), 180, dtype=np.uint8)
    cv2.circle(img, (64, 48), 20, (30, 120, 200), -1)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


@pytest.fixture
def disagreeing_ai_detectors(monkeypatch):
    """UnivFD says not generated, B-Free says generated."""
    stubs = [UnivFDDetector(_Runner(-3.0)), BFreeDetector(_Runner(3.0))]
    original = MediaDetectionRouter.route

    def route(self, file_path, media_type):
        result = original(self, file_path, media_type)
        if media_type == "image":
            kept = [d for d in result.get("detections", [])
                    if getattr(d, "detector", None) not in ("UnivFDDetector", "BFreeDetector")]
            result["detections"] = kept + [s.detect(file_path) for s in stubs]
        return result

    monkeypatch.setattr(MediaDetectionRouter, "route", route)
    monkeypatch.setattr(settings, "ai_generation_advisory_detectors", ["UnivFDDetector"])


def test_advisory_detector_is_shown_but_does_not_vote_in_api_and_reports(disagreeing_ai_detectors):
    media_id = client.post(
        "/api/v1/media/upload", files={"file": ("ai.jpg", _image(), "image/jpeg")},
    ).json()["media_id"]
    analysis = client.post("/api/v1/analysis/", json={"media_id": media_id}).json()
    analysis_id = analysis["analysis_id"]

    ai = client.get(f"/api/v1/analysis/{analysis_id}").json()["ai_generation"]
    assert ai["status"] == "AI_GENERATION_INDICATED"  # B-Free decides, not CONFLICTING
    assert ai["detector"] == "BFreeDetector"
    voting = {d["detector"]: (d["status"], d["voting"]) for d in ai["detectors"]}
    assert voting == {
        "UnivFDDetector": ("NOT_INDICATED", False),
        "BFreeDetector": ("AI_GENERATION_INDICATED", True),
    }
    assert "does not vote" in ai["rationale"]

    report = client.get(f"/api/v1/analysis/{analysis_id}/report", params={"format": "json"}).json()
    package_ai = report["ai_generation_evidence"]
    assert package_ai["status"] == "AI_GENERATION_INDICATED"
    assert {d["detector"]: d["voting"] for d in package_ai["detectors"]} == {
        "UnivFDDetector": False, "BFreeDetector": True,
    }

    md = client.get(f"/api/v1/analysis/{analysis_id}/report", params={"format": "md"}).text
    univfd_line = next(line for line in md.splitlines() if line.strip().startswith("- UnivFDDetector:"))
    assert "advisory" in univfd_line and "does not vote" in univfd_line
    bfree_line = next(line for line in md.splitlines() if line.strip().startswith("- BFreeDetector:"))
    assert "advisory" not in bfree_line

    html = client.get(f"/api/v1/analysis/{analysis_id}/report", params={"format": "html"}).text
    assert "does not vote" in html


def test_without_an_advisory_list_the_same_files_conflict(disagreeing_ai_detectors, monkeypatch):
    monkeypatch.setattr(settings, "ai_generation_advisory_detectors", [])
    media_id = client.post(
        "/api/v1/media/upload", files={"file": ("ai2.jpg", _image(), "image/jpeg")},
    ).json()["media_id"]
    analysis_id = client.post("/api/v1/analysis/", json={"media_id": media_id}).json()["analysis_id"]
    ai = client.get(f"/api/v1/analysis/{analysis_id}").json()["ai_generation"]
    assert ai["status"] == "CONFLICTING"
    assert all(d["voting"] for d in ai["detectors"])
