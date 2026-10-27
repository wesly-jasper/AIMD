"""
POST /api/v1/system/detectors/self-test and the operational /health report.

The self-test makes every detector run on a probe image, so "configured" can be
told apart from "works". It is admin-only: it occupies the GPU.
"""
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.core.auth import User, get_current_user
from app.main import app
from app.schemas.detection import DetectorExecutionStatus
from app.services.detection import detection_container
from app.services.detection.detectors.base_detector import BaseDetector
from app.services.detection.runtime import detector_health

client = TestClient(app)


class Works(BaseDetector):
    detector_id = "Works"

    def detect(self, file_path):
        return self.inconclusive("probe has no content worth a determination")


class Broken(BaseDetector):
    detector_id = "Broken"

    def detect(self, file_path):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")


def _with_detectors(monkeypatch, detectors):
    container = SimpleNamespace(
        router=SimpleNamespace(image_engine=SimpleNamespace(detectors=detectors)),
    )
    container.reload = lambda: container.router
    monkeypatch.setattr(detection_container, "_shared", container)
    detector_health.reset()


def _as_admin():
    app.dependency_overrides[get_current_user] = lambda: User(
        username="pytest_admin", role="admin", is_active=True
    )


def test_self_test_is_admin_only(monkeypatch):
    _with_detectors(monkeypatch, [Works()])
    assert client.post("/api/v1/system/detectors/self-test").status_code == 403


def test_self_test_runs_every_detector_and_reports_failures(monkeypatch):
    _with_detectors(monkeypatch, [Works(), Broken()])
    _as_admin()
    response = client.post("/api/v1/system/detectors/self-test")
    assert response.status_code == 200
    body = response.json()
    body = body.get("data", body)

    assert body["passed"] is False
    assert body["runs"]["Works"]["execution_status"] == "INCONCLUSIVE"
    broken = body["runs"]["Broken"]
    assert broken["availability"] == "AVAILABLE"
    assert broken["execution_status"] == DetectorExecutionStatus.FAILED.value
    assert body["detectors"]["Works"]["state"] == "OPERATIONAL"
    assert body["detectors"]["Broken"]["state"] == "DEGRADED"


def test_health_degrades_when_a_detector_fails_on_every_run(monkeypatch):
    from app.db import database

    class FakeSession:
        def execute(self, _statement):
            return None

        def close(self):
            return None

    monkeypatch.setattr(database, "SessionLocal", FakeSession)
    _with_detectors(monkeypatch, [Works(), Broken()])
    _as_admin()
    for _ in range(3):
        client.post("/api/v1/system/detectors/self-test")

    health = client.get("/health").json()
    health = health.get("data", health)
    assert health["status"] == "degraded"
    assert health["detectors"]["failing_detectors"] == ["Broken"]
    assert health["detectors"]["detectors"]["Broken"]["state"] == "FAILING"
    assert health["detectors"]["detectors"]["Works"]["state"] == "OPERATIONAL"
    detector_health.reset()
