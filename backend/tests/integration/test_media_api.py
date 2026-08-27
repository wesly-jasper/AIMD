from fastapi.testclient import TestClient
from app.main import app
import cv2
import numpy as np


client=TestClient(app)


def create_test_image():
    img=np.zeros((100,200,3),dtype=np.uint8)

    success,encoded=cv2.imencode(".jpg",img)

    assert success

    return encoded.tobytes()


def create_test_video(path,frame_count=5):
    fourcc=cv2.VideoWriter_fourcc(*"mp4v")

    writer=cv2.VideoWriter(
        str(path),
        fourcc,
        10,
        (200,100)
    )

    for i in range(frame_count):
        frame=np.zeros((100,200,3),dtype=np.uint8)
        frame[:]=i*30
        writer.write(frame)

    writer.release()


def test_upload_media():
    image=create_test_image()

    response=client.post(
        "/api/v1/media/upload",
        files={
            "file":("test.jpg",image,"image/jpeg")
        }
    )

    assert response.status_code==200

    data=response.json()

    assert "media_id" in data
    assert "original_filename" in data
    assert "stored_filename" in data
    assert "sha256" in data
    assert "metadata" in data

    assert data["original_filename"]=="test.jpg"
    assert data["metadata"]["media_type"]=="image"
    assert data["metadata"]["width"]==200
    assert data["metadata"]["height"]==100


def test_upload_video(tmp_path):
    video_path=tmp_path/"test.mp4"

    create_test_video(
        video_path,
        frame_count=5
    )

    video=video_path.read_bytes()

    response=client.post(
        "/api/v1/media/upload",
        files={
            "file":("test.mp4",video,"video/mp4")
        }
    )

    assert response.status_code==200

    data=response.json()

    assert "media_id" in data
    assert "metadata" in data
    assert "frames" in data
    assert "keyframes" in data

    assert data["original_filename"]=="test.mp4"
    assert data["metadata"]["media_type"]=="video"
    assert data["metadata"]["width"]==200
    assert data["metadata"]["height"]==100

    assert data["frames"] is not None
    assert data["frames"]["frame_count"]==5
    assert data["frames"]["saved_count"]==5

    assert data["keyframes"] is not None
    assert data["keyframes"]["total_frames"]==5
    assert data["keyframes"]["keyframe_count"]==1

    assert "fingerprint" in data
    assert "keyframe_fingerprints" in data

    assert data["fingerprint"] is not None
    assert "sha256" in data["fingerprint"]

    assert len(data["fingerprint"]["sha256"])==64

    assert len(data["keyframe_fingerprints"])==1

    assert "file" in data["keyframe_fingerprints"][0]
    assert "phash" in data["keyframe_fingerprints"][0]

    assert len(
        data["keyframe_fingerprints"][0]["phash"]
    )==64