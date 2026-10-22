"""
Comprehensive Live E2E test verifying all API endpoints and data flows.
"""
import io
import wave
import struct
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoints():
    r1 = client.get("/health")
    assert r1.status_code == 200
    assert r1.json()["status"] == "ok"

    r2 = client.get("/api/v1/health")
    assert r2.status_code == 200
    assert r2.json()["status"] == "ok"


def test_e2e_image_investigation(tmp_path):
    # 1. Create and upload image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (50, 50), (255, 0, 0), -1)
    _, encoded = cv2.imencode(".png", img)
    img_bytes = encoded.tobytes()

    upload_res = client.post(
        "/api/v1/media/upload",
        files={"file": ("test_e2e.png", img_bytes, "image/png")},
    )
    assert upload_res.status_code == 200
    upload_data = upload_res.json()
    media_id = upload_data["media_id"]

    # 2. Test Media download/stream endpoint
    dl_res = client.get(f"/api/v1/media/{media_id}/download")
    assert dl_res.status_code == 200
    assert len(dl_res.content) == len(img_bytes)

    # 3. Test Media detail endpoint
    get_media_res = client.get(f"/api/v1/media/{media_id}")
    assert get_media_res.status_code == 200
    assert get_media_res.json()["media_id"] == media_id

    # 4. Run Analysis
    ana_res = client.post("/api/v1/analysis/", json={"media_id": media_id})
    assert ana_res.status_code == 200
    ana_id = ana_res.json()["analysis_id"]

    # 5. List analyses
    list_ana_res = client.get("/api/v1/analysis/")
    assert list_ana_res.status_code == 200
    items = list_ana_res.json()["items"]
    assert any(item["analysis_id"] == ana_id for item in items)

    # 6. Retrieve analysis details
    get_ana_res = client.get(f"/api/v1/analysis/{ana_id}")
    assert get_ana_res.status_code == 200
    ana_details = get_ana_res.json()
    assert ana_details["analysis_id"] == ana_id
    assert "detections" in ana_details

    # 7. The per-detector contract is in the analysis payload itself.
    assert len(ana_details["detections"]) > 0
    assert ana_details["assessment_detail"] is not None


def test_invalid_requests():
    # 404 on nonexistent media
    r = client.post("/api/v1/analysis/", json={"media_id": "nonexistent-id"})
    assert r.status_code == 404

    # 404 on nonexistent analysis
    r = client.get("/api/v1/analysis/nonexistent-id")
    assert r.status_code == 404

    # 404 on nonexistent media download
    r = client.get("/api/v1/media/nonexistent-id/download")
    assert r.status_code == 404
