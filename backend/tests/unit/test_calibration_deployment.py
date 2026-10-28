"""
The deployed calibration is the one fitted for the running detectors, and when it
is not -- a stale v3 path after B-Free moved to upscale mode, a missing file, a
tampered one -- nothing presents a raw score as calibrated evidence, and /health
says the deployment is not the validated configuration.
"""
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.core import health
from app.core.config import Settings, settings
from app.schemas.detection import ScoreSemantics
from app.services.detection import calibration
from app.services.detection.detectors.image.bfree_detector import BFreeDetector
from app.services.fusion.evidence_fusion import fuse

CAL_DIR = Path(__file__).resolve().parents[2] / "calibration"
V3, V4 = CAL_DIR / "fusion-calibration.v3.json", CAL_DIR / "fusion-calibration.v4.json"
WEIGHTS = "bfree-c6a9f89.w-5948ca78f4d9"  # the B-Free release both files were fitted on


def _fitted(path, detector="BFreeDetector"):
    return json.loads(path.read_text(encoding="utf-8"))[detector]["detector_version"]


class _Runner:
    """A B-Free runner whose version tag carries a small-input mode, like the real one."""

    def __init__(self, tag, logit=3.0):
        self.version_tag = tag
        self._logit = logit

    def predict(self, path):
        return {"logit": self._logit, "score": float(1 / (1 + np.exp(-self._logit))), "metadata": {}}


UPSCALE = BFreeDetector(_Runner(f"{WEIGHTS}.up504o+1.90"))
TILE = BFreeDetector(_Runner(WEIGHTS))


def _row(detector):
    return SimpleNamespace(detector="BFreeDetector", detector_version=detector.version, availability="AVAILABLE",
                           execution_status="COMPLETED", confidence=0.95, manipulation_detected=True,
                           detector_metadata={"image_size": [1024, 768]}, status_reason=None)


# ---- the committed files ---------------------------------------------------

def test_committed_files_match_their_checksums():
    for path in (V3, V4):
        status = calibration.file_status(path)
        assert status["state"] == "ok" and status["checksum"] == "verified", path.name


def test_v4_is_fitted_for_the_default_b_free_mode_and_v3_for_tile():
    defaults = Settings(_env_file=None)
    assert defaults.bfree_small_input_mode == "upscale"
    assert _fitted(V4) == f"1.0.0+{WEIGHTS}.up504o{defaults.bfree_small_input_logit_offset:+.2f}"
    assert _fitted(V4) == UPSCALE.version
    assert _fitted(V3) == TILE.version


# ---- a stale v3 path with B-Free in upscale mode ---------------------------

def test_stale_v3_is_reported_as_a_mismatch(monkeypatch):
    status = calibration.consistency(V3, {"BFreeDetector": UPSCALE.version})
    assert status["consistent"] is False
    assert status["detector_status"]["BFreeDetector"] == {
        "status": "version_mismatch", "fitted": _fitted(V3), "running": UPSCALE.version}
    assert calibration.consistency(V4, {"BFreeDetector": UPSCALE.version})["consistent"] is True


def test_stale_v3_gives_fusion_direction_only_with_the_reason(monkeypatch):
    monkeypatch.setattr(settings, "fusion_calibration_path", V3)
    ai = fuse([_row(UPSCALE)])["ai_generation"]
    (entry,) = ai["contributions"]
    assert entry["calibrated"] is False and entry["log10_lr"] is None
    assert _fitted(V3) in entry["reason"] and "direction only" in entry["reason"]
    assert ai["status"] == "NOT_ASSESSED"


def test_v4_quantifies_the_same_row(monkeypatch):
    monkeypatch.setattr(settings, "fusion_calibration_path", V4)
    out = fuse([_row(UPSCALE)])
    assert out["ai_generation"]["contributions"][0]["calibrated"] is True
    assert out["calibration"]["state"] == "ok" and out["calibration"]["checksum"] == "verified"


def test_stale_v3_never_labels_the_detector_score_calibrated(monkeypatch, tmp_path):
    # CALIBRATION_PATH also drives detector flags and the score label the UI shows.
    monkeypatch.setattr(settings, "calibration_path", V3)
    image = tmp_path / "x.png"
    image.write_bytes(b"not read by the stub runner")
    result = BFreeDetector(_Runner(f"{WEIGHTS}.up504o+1.90", logit=3.0)).detect(image)
    assert result.score_semantics != ScoreSemantics.CALIBRATED_PROBABILITY
    assert result.metadata["calibration"]["applied"] is False
    assert _fitted(V3) in result.metadata["calibration"]["reason"]


# ---- missing, unreadable, tampered -----------------------------------------

def test_a_missing_file_is_named_not_mistaken_for_no_calibration(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "fusion_calibration_path", tmp_path / "gone.json")
    out = fuse([_row(UPSCALE)])
    assert out["calibration"]["state"] == "missing"
    assert "configured calibration file is missing" in out["ai_generation"]["contributions"][0]["reason"]
    assert calibration.consistency(tmp_path / "gone.json", {})["consistent"] is False


def test_an_unreadable_file_is_reported(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert calibration.file_status(bad)["state"] == "unreadable"
    assert calibration.consistency(bad, {})["consistent"] is False


def test_a_file_that_fails_its_checksum_is_inconsistent_and_not_used(monkeypatch, tmp_path):
    shutil.copy(CAL_DIR / calibration.CHECKSUM_FILE, tmp_path / calibration.CHECKSUM_FILE)
    data = json.loads(V4.read_text(encoding="utf-8"))
    data["BFreeDetector"]["b"] += 1.0
    (tmp_path / V4.name).write_text(json.dumps(data), encoding="utf-8")
    status = calibration.consistency(tmp_path / V4.name, {"BFreeDetector": UPSCALE.version})
    assert status["checksum"] == "mismatch" and status["consistent"] is False
    monkeypatch.setattr(settings, "fusion_calibration_path", tmp_path / V4.name)
    entry = fuse([_row(UPSCALE)])["ai_generation"]["contributions"][0]
    assert entry["calibrated"] is False and "checksum" in entry["reason"]


def test_unset_is_consistent_and_says_so(monkeypatch):
    monkeypatch.setattr(settings, "fusion_calibration_path", None)
    monkeypatch.setattr(settings, "calibration_path", None)
    out = fuse([_row(UPSCALE)])
    assert out["calibration"]["state"] == "unset"
    assert "no calibration file is configured" in out["ai_generation"]["contributions"][0]["reason"]


# ---- /health ----------------------------------------------------------------

@pytest.mark.parametrize("path, detector, healthy", [(V4, UPSCALE, True), (V3, UPSCALE, False), (V3, TILE, True)])
def test_health_is_degraded_only_when_the_calibration_does_not_fit(monkeypatch, path, detector, healthy):
    monkeypatch.setattr(health, "_image_detectors", lambda: [detector])
    monkeypatch.setattr(health, "check_database", lambda: "ok")
    monkeypatch.setattr(settings, "fusion_calibration_path", path)
    monkeypatch.setattr(settings, "calibration_path", None)
    report = health.build_health_report()
    cal = report["detectors"]["calibration"]
    assert cal["consistent"] is healthy
    assert report["status"] == ("ok" if healthy else "degraded")
    assert cal["fusion_calibration_path"]["path"] == path.name  # no directory layout on an open route
