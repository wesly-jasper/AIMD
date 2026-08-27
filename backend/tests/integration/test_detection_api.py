from fastapi.testclient import TestClient

from app.main import app


client=TestClient(app)


def test_detection_api_missing_file():

    response=client.post(
        "/api/v1/detection/analyze",
        json={
            "file_path":"missing.jpg",
            "media_type":"image"
        }
    )

    assert response.status_code==400

    data=response.json()

    assert "detail" in data


def test_detection_api_missing_media_type():

    response=client.post(
        "/api/v1/detection/analyze",
        json={
            "file_path":"test.jpg",
            "media_type":""
        }
    )

    assert response.status_code==400

    data=response.json()

    assert "detail" in data


def test_detection_api_invalid_request():

    response=client.post(
        "/api/v1/detection/analyze",
        json={
            "file_path":"test.jpg"
        }
    )

    assert response.status_code==422