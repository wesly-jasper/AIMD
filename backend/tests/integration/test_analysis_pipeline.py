"""
End-to-end integration tests for the full AIMD forensic analysis pipeline.

Tests the complete flow:
  Upload Media → Trigger Analysis Pipeline → Verify All Downstream Artifacts
  (Metadata, Frames, Detection, Fingerprints, Similarity, Source, Provenance, Evidence, Report).
"""
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _make_test_image():
    """Create a test image with distinct geometric features."""
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    # Background pattern
    img[:, :] = (200, 200, 200)
    # Add a rectangle
    cv2.rectangle(img, (20, 20), (80, 80), (50, 50, 200), -1)
    # Add a circle
    cv2.circle(img, (120, 60), 25, (0, 150, 50), -1)
    success, encoded = cv2.imencode(".jpg", img)
    assert success
    return encoded.tobytes()


def _make_test_video(path, frame_count=6):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10, (160, 120))
    for i in range(frame_count):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        frame[:] = (i * 40, i * 40, i * 40)
        writer.write(frame)
    writer.release()


def test_full_pipeline_image():
    # 1. Upload media
    image_bytes = _make_test_image()
    upload_res = client.post(
        "/api/v1/media/upload",
        files={"file": ("pipeline_test.jpg", image_bytes, "image/jpeg")},
    )
    assert upload_res.status_code == 200
    upload_data = upload_res.json()
    media_id = upload_data["media_id"]
    assert media_id is not None
    assert upload_data["sha256"] is not None

    # 2. Trigger analysis pipeline
    analysis_res = client.post(
        "/api/v1/analysis/",
        json={"media_id": media_id},
    )
    assert analysis_res.status_code == 200
    analysis_data = analysis_res.json()
    analysis_id = analysis_data["analysis_id"]
    assert analysis_id is not None
    assert analysis_data["status"] in ("COMPLETED", "PARTIAL")

    # 3. Retrieve analysis details
    get_res = client.get(f"/api/v1/analysis/{analysis_id}")
    assert get_res.status_code == 200
    get_data = get_res.json()
    assert get_data["analysis_id"] == analysis_id
    assert get_data["media_id"] == media_id
    assert get_data["status"] in ("COMPLETED", "PARTIAL")
    assert "detections" in get_data
    assert len(get_data["detections"]) > 0

    # 4. Retrieve detection details
    det_res = client.get(f"/api/v1/detection/{analysis_id}")
    assert det_res.status_code == 200
    det_data = det_res.json()
    assert "detections" in det_data
    assert len(det_data["detections"]) > 0
    # Check that detectors have assessments
    for d in det_data["detections"]:
        assert d["assessment"] in ("CLEAN", "SUSPICIOUS", "INCONCLUSIVE", "UNAVAILABLE")

    # 5. Retrieve source tracing
    src_res = client.get(f"/api/v1/source/{analysis_id}")
    assert src_res.status_code == 200
    src_data = src_res.json()
    assert "provider_status" in src_data
    assert "candidates" in src_data

    # 6. Retrieve provenance graph
    prov_res = client.get(f"/api/v1/provenance/{analysis_id}")
    assert prov_res.status_code == 200
    prov_data = prov_res.json()
    assert "nodes" in prov_data
    assert "edges" in prov_data
    assert len(prov_data["nodes"]) >= 1

    # 7. Retrieve evidence package
    ev_res = client.get(f"/api/v1/evidence/{analysis_id}")
    assert ev_res.status_code == 200
    ev_data = ev_res.json()
    assert "case_information" in ev_data
    assert "media_evidence" in ev_data
    assert ev_data["media_evidence"]["evidence_category"] == "FACT"
    assert "classified_evidence" in ev_data
    assert "FACT" in ev_data["classified_evidence"]
    assert "limitations" in ev_data
    assert len(ev_data["limitations"]) > 0

    # 8. Retrieve forensic report
    rep_res = client.get(f"/api/v1/report/{analysis_id}")
    assert rep_res.status_code == 200
    rep_data = rep_res.json()
    assert "json" in rep_data
    assert "markdown" in rep_data
    assert "# AIMD FORENSIC ANALYSIS REPORT" in rep_data["markdown"]


def test_full_pipeline_video(tmp_path):
    video_path = tmp_path / "pipeline_test.mp4"
    _make_test_video(video_path, frame_count=6)
    video_bytes = video_path.read_bytes()

    # 1. Upload video
    upload_res = client.post(
        "/api/v1/media/upload",
        files={"file": ("pipeline_test.mp4", video_bytes, "video/mp4")},
    )
    assert upload_res.status_code == 200
    upload_data = upload_res.json()
    media_id = upload_data["media_id"]

    # 2. Run analysis pipeline
    analysis_res = client.post(
        "/api/v1/analysis/",
        json={"media_id": media_id},
    )
    assert analysis_res.status_code == 200
    analysis_id = analysis_res.json()["analysis_id"]

    # 3. Check report generated
    rep_res = client.get(f"/api/v1/report/{analysis_id}")
    assert rep_res.status_code == 200
    rep_data = rep_res.json()
    assert "markdown" in rep_data
    assert len(rep_data["markdown"]) > 100
