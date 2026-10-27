"""
Detector runtime: timeouts, the out-of-memory retry, the health monitor, the
memory preflight, and TruFor's recovery from a model split across devices.

These are the failure modes the reliability evaluation hit (RELIABILITY_REPORT.md
section 4): each used to be silent or to need a restart.
"""
import threading
import time

import pytest
import torch

from app.schemas.detection import (
    DetectionResult,
    DetectorAvailability,
    DetectorExecutionStatus,
    ScoreSemantics,
)
from app.services.detection.detectors.base_detector import BaseDetector
from app.services.detection.runtime import (
    DetectorHealthMonitor,
    is_memory_error,
    memory_preflight,
    run_detector,
)


def _ok(detector_id="Probe", status=DetectorExecutionStatus.COMPLETED):
    return DetectionResult(
        detector=detector_id, media_type="image", manipulation_detected=False,
        confidence=0.1, manipulation_type="x",
        availability=DetectorAvailability.AVAILABLE, execution_status=status,
        score_semantics=ScoreSemantics.RAW_MODEL_SCORE,
    )


class Hangs(BaseDetector):
    detector_id = "Hangs"

    def __init__(self):
        self.release = threading.Event()

    def detect(self, file_path):
        self.release.wait(10)
        return _ok(self.detector_id)


class OutOfMemoryOnce(BaseDetector):
    """Raises an allocation error on the first call only, as a GPU does under pressure."""
    detector_id = "OutOfMemoryOnce"

    def __init__(self):
        self.calls = 0

    def detect(self, file_path):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 512.00 MiB")
        return _ok(self.detector_id)


class CatchesOutOfMemory(BaseDetector):
    """Swallows its own exception, as the TruFor and OpenCV detectors do."""
    detector_id = "CatchesOutOfMemory"

    def __init__(self):
        self.calls = 0

    def detect(self, file_path):
        self.calls += 1
        if self.calls == 1:
            try:
                raise MemoryError()
            except MemoryError as exc:
                return self.failed_from("analysis failed", exc)
        return _ok(self.detector_id)


class AlwaysBroken(BaseDetector):
    detector_id = "AlwaysBroken"

    def __init__(self):
        self.calls = 0

    def detect(self, file_path):
        self.calls += 1
        raise ValueError("deterministic bug")


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "x.jpg"
    path.write_bytes(b"\xff\xd8\xff")
    return path


def test_a_hung_detector_becomes_failed_not_a_hung_request(image):
    monitor = DetectorHealthMonitor()
    detector = Hangs()
    started = time.perf_counter()
    result = run_detector(detector, image, timeout_s=0.2, monitor=monitor)
    detector.release.set()

    assert time.perf_counter() - started < 5
    assert result.availability is DetectorAvailability.AVAILABLE
    assert result.execution_status is DetectorExecutionStatus.FAILED
    assert "timed out" in result.status_reason
    assert result.metadata["timed_out"] is True
    assert result.confidence == 0.0
    assert monitor.snapshot("Hangs")["timeouts"] == 1


def test_an_out_of_memory_exception_is_retried_once(image):
    monitor = DetectorHealthMonitor()
    detector = OutOfMemoryOnce()
    result = run_detector(detector, image, timeout_s=5, monitor=monitor)

    assert detector.calls == 2
    assert result.execution_status is DetectorExecutionStatus.COMPLETED
    assert result.metadata["retried_after_memory_error"] is True
    assert monitor.snapshot("OutOfMemoryOnce")["memory_retries"] == 1


def test_a_caught_out_of_memory_failure_is_retried_once(image):
    detector = CatchesOutOfMemory()
    result = run_detector(detector, image, timeout_s=5, monitor=DetectorHealthMonitor())
    assert detector.calls == 2
    assert result.execution_status is DetectorExecutionStatus.COMPLETED


def test_a_deterministic_failure_is_not_retried(image):
    detector = AlwaysBroken()
    result = run_detector(detector, image, timeout_s=5, monitor=DetectorHealthMonitor())
    assert detector.calls == 1
    assert result.availability is DetectorAvailability.AVAILABLE
    assert result.execution_status is DetectorExecutionStatus.FAILED
    assert result.status_reason.startswith("ValueError")
    # The message is logged, not recorded: it can carry paths.
    assert "deterministic bug" not in result.status_reason


def test_retry_can_be_disabled(image):
    detector = OutOfMemoryOnce()
    result = run_detector(
        detector, image, timeout_s=5, retry_on_memory_error=False,
        monitor=DetectorHealthMonitor(),
    )
    assert detector.calls == 1
    assert result.execution_status is DetectorExecutionStatus.FAILED
    assert "out of memory" in result.status_reason


@pytest.mark.parametrize("exc, expected", [
    (MemoryError(), True),
    (RuntimeError("CUDA out of memory"), True),
    (Exception("OpenCV(4.10) error: (-4:Insufficient memory) Failed to allocate"), True),
    (ValueError("bad shape"), False),
])
def test_memory_error_classification(exc, expected):
    assert is_memory_error(exc) is expected


def test_health_states_follow_actual_runs():
    monitor = DetectorHealthMonitor()
    name = "D"
    assert monitor.state(name) == "UNTESTED"
    assert monitor.state(name, DetectorAvailability.UNAVAILABLE) == "UNAVAILABLE"

    failed = _ok(name, DetectorExecutionStatus.FAILED)
    monitor.record(name, _ok(name), 10)
    assert monitor.state(name) == "OPERATIONAL"
    monitor.record(name, failed, 10)
    assert monitor.state(name, failing_after=3) == "DEGRADED"
    monitor.record(name, failed, 10)
    monitor.record(name, failed, 10)
    assert monitor.state(name, failing_after=3) == "FAILING"
    # A skip says nothing either way.
    monitor.record(name, _ok(name, DetectorExecutionStatus.SKIPPED), 1)
    assert monitor.state(name, failing_after=3) == "FAILING"
    # One good run clears it: the detector works again.
    monitor.record(name, _ok(name, DetectorExecutionStatus.INCONCLUSIVE), 10)
    assert monitor.state(name, failing_after=3) == "OPERATIONAL"
    snap = monitor.snapshot(name)
    assert snap["runs"] == 6 and snap["failed"] == 3 and snap["skipped"] == 1


def test_memory_preflight(monkeypatch):
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: type("VM", (), {"available": 512 * 1024 * 1024})()
    )
    assert memory_preflight(0) is None
    assert memory_preflight(256) is None
    reason = memory_preflight(2048)
    assert "512 MB free" in reason and "2048 MB required" in reason


def test_factory_refuses_to_load_a_model_without_memory(monkeypatch, tmp_path):
    from app.core.config import settings
    from app.services.detection import detector_factory

    monkeypatch.setattr(settings, "trufor_enabled", True)
    monkeypatch.setattr(settings, "trufor_model_path", tmp_path)
    monkeypatch.setattr(
        detector_factory, "memory_preflight", lambda mb: "insufficient free memory (test)"
    )
    detector = detector_factory.DetectorFactory.create_trufor()
    availability, reason = detector.availability()
    assert availability is DetectorAvailability.UNAVAILABLE
    assert "insufficient free memory" in reason


# ── TruFor recovery ───────────────────────────────────────────────────────────


class _TinyTruFor(torch.nn.Module):
    """TruFor's interface: (pred, conf, det, npp) for an image tensor."""

    def __init__(self, cfg=None):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 2, 1)
        self.register_buffer("scale", torch.ones(1))

    def forward(self, x):
        pred = self.conv(x) * self.scale
        conf = pred[:, :1]
        det = pred.mean().reshape(1, 1)
        return pred, conf, det, None


def _runner(tmp_path):
    from app.services.detection.models.trufor_runner import TruForRunner

    weights = tmp_path / "weights" / "trufor.pth.tar"
    weights.parent.mkdir(parents=True)
    torch.save({"state_dict": _TinyTruFor().state_dict()}, weights)
    runner = TruForRunner(tmp_path, device="cpu", model_builder=_TinyTruFor)
    runner.load()
    return runner


def test_trufor_moves_a_split_model_back_before_running(tmp_path):
    runner = _runner(tmp_path)
    # Simulate the failure: a buffer left on another device type. "meta" stands
    # in for the GPU on a CPU-only test host.
    runner.model.scale = runner.model.scale.to("meta")
    assert not runner._on_one_device()

    runner._infer(torch.rand(1, 3, 16, 16))

    assert runner._on_one_device()
    assert runner.recoveries + runner.reloads == 1


def test_trufor_reloads_when_the_model_cannot_be_moved_back(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    runner.model.scale = runner.model.scale.to("meta")
    monkeypatch.setattr(runner.model, "to", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("CUDA error")))

    runner._infer(torch.rand(1, 3, 16, 16))

    assert runner.reloads == 1
    assert runner._on_one_device()
    assert runner.identity()["reloads"] == 1


def test_trufor_does_not_wait_forever_behind_a_hung_call(tmp_path):
    runner = _runner(tmp_path)
    runner.lock_timeout_s = 0.1
    runner._lock.acquire()
    try:
        with pytest.raises(TimeoutError):
            runner._infer(torch.rand(1, 3, 16, 16))
    finally:
        runner._lock.release()


def test_repeated_runs_reuse_worker_threads(image):
    """A thread per call leaked native thread pools (6,400 threads in one run)."""
    class Quick(BaseDetector):
        detector_id = "Quick"

        def detect(self, file_path):
            return _ok(self.detector_id)

    detector = Quick()
    monitor = DetectorHealthMonitor()
    run_detector(detector, image, timeout_s=5, monitor=monitor)
    before = threading.active_count()
    for _ in range(200):
        run_detector(detector, image, timeout_s=5, monitor=monitor)
    assert threading.active_count() <= before + 4
