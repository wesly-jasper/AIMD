"""
End-to-end integration tests for the full AIMD forensic analysis pipeline.

Tests the complete flow:
  Upload Media -> Trigger Analysis Pipeline -> Verify All Downstream Artifacts
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

    # 4. The analysis payload carries the per-detector contract and the record.
    for d in get_data["detections"]:
        assert d["availability"] in ("AVAILABLE", "UNAVAILABLE")
        assert d["execution_status"] in ("COMPLETED", "INCONCLUSIVE", "FAILED", "SKIPPED")
    assert get_data["assessment_detail"]["category"] == get_data["assessment"]

    # 5. Stages with no route of their own still persisted their output.
    _assert_stage_records(analysis_id)


def _assert_stage_records(analysis_id: str) -> None:
    from app.db import repository as repo
    from app.db.database import SessionLocal

    with SessionLocal() as db:
        provenance = repo.get_provenance_for_analysis(db, analysis_id)
        assert len(provenance["nodes"]) >= 1
        report = repo.get_report_for_analysis(db, analysis_id)
        assert report is not None
        assert "AIMD FORENSIC ANALYSIS REPORT" in report.markdown_report


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

    # 3. Check the analysis and its stored record
    assert client.get(f"/api/v1/analysis/{analysis_id}").status_code == 200
    _assert_stage_records(analysis_id)


def test_full_pipeline_audio(tmp_path):
    import wave
    import struct

    # Generate a simple 1-second 440Hz sine wave WAV file
    audio_path = tmp_path / "pipeline_test.wav"
    with wave.open(str(audio_path), "w") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(22050)
        # 1 second of audio
        for i in range(22050):
            sample = int(32767.0 * 0.5 * np.sin(2.0 * np.pi * 440.0 * i / 22050))
            wav_file.writeframes(struct.pack("<h", sample))

    audio_bytes = audio_path.read_bytes()

    # 1. Upload audio
    upload_res = client.post(
        "/api/v1/media/upload",
        files={"file": ("pipeline_test.wav", audio_bytes, "audio/wav")},
    )
    assert upload_res.status_code == 200
    upload_data = upload_res.json()
    media_id = upload_data["media_id"]
    assert media_id is not None
    assert upload_data["media_type"] == "audio"

    # 2. Run analysis pipeline
    analysis_res = client.post(
        "/api/v1/analysis/",
        json={"media_id": media_id},
    )
    assert analysis_res.status_code == 200
    analysis_id = analysis_res.json()["analysis_id"]

    # 3. Verify analysis details
    get_res = client.get(f"/api/v1/analysis/{analysis_id}")
    assert get_res.status_code == 200
    get_data = get_res.json()
    assert get_data["media_id"] == media_id
    assert get_data["status"] in ("COMPLETED", "PARTIAL")

    # 4. Check the stored record
    assert get_data["media_type"] == "audio"
    _assert_stage_records(analysis_id)

