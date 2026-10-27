"""
/api/v1/system -- operator checks that the detectors actually work.

POST /system/detectors/self-test  -- run every image detector on a generated
                                     probe image and report what each did
                                     (administrators only: it occupies the GPU)

``/health`` reports configuration and the outcome of real runs. This route makes
a run happen on demand, so "the checkpoint is on disk" can be told apart from
"the model produces output" without waiting for a user's upload to find out.
"""
import tempfile
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, Query
from PIL import Image

from app.core.auth import require_role
from app.core.config import settings
from app.core.health import detector_operational_status
from app.core.safe_errors import scrub
from app.services.detection.detection_container import get_detection_container
from app.services.detection.runtime import run_detector

router = APIRouter(prefix="/system", tags=["System"])


def write_probe_image(directory: Path) -> Path:
    """
    A 512 px photograph-like probe: smooth gradients plus sensor-like noise,
    saved as JPEG so compression-reading detectors have blocks to read. Its
    content is irrelevant; the question is only whether each detector returns.
    """
    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:512, 0:512].astype(np.float32)
    base = np.stack([x / 2, y / 2, (x + y) / 4], axis=-1)
    pixels = np.clip(base + rng.normal(0, 6, base.shape), 0, 255).astype(np.uint8)
    path = directory / "aimd-self-test-probe.jpg"
    Image.fromarray(pixels).save(path, quality=90)
    return path


@router.post(
    "/detectors/self-test",
    summary="Run every image detector on a probe image",
    dependencies=[Depends(require_role("admin"))],
)
def detector_self_test(
    reload: bool = Query(
        False,
        description="Rebuild the detectors first, e.g. after freeing memory for a "
                    "model that was refused at startup.",
    ),
):
    container = get_detection_container()
    router_ = container.reload() if reload else container.router
    detectors = list(getattr(router_.image_engine, "detectors", []))

    runs = {}
    with tempfile.TemporaryDirectory(prefix="aimd-selftest-") as tmp:
        probe = write_probe_image(Path(tmp))
        for detector in detectors:
            result = run_detector(
                detector,
                probe,
                timeout_s=settings.detector_timeout_seconds,
                retry_on_memory_error=settings.detector_retry_on_memory_error,
            )
            runs[type(detector).__name__] = {
                "availability": result.availability.value,
                "execution_status": result.execution_status.value,
                "reason": scrub(result.status_reason),
                "timed_out": bool(result.metadata.get("timed_out")),
            }

    status = detector_operational_status(detectors)
    passed = all(
        r["execution_status"] in ("COMPLETED", "INCONCLUSIVE")
        or (r["availability"] == "UNAVAILABLE")
        for r in runs.values()
    )
    return {
        # UNAVAILABLE is a configuration fact, reported, not a self-test failure;
        # a detector that is present and FAILED or timed out is.
        "passed": passed,
        "reloaded": reload,
        "runs": runs,
        "detectors": status,
    }
