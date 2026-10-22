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
    assert "sha256" in data
    # The upload response is a summary; internals such as the storage path and
    # the extraction output stay server-side. Dimensions come from GET /media/{id}.
    assert "file_path" not in data and "stored_filename" not in data

    assert data["original_filename"]=="test.jpg"
    assert data["media_type"]=="image"
    meta = client.get(f"/api/v1/media/{data['media_id']}").json()["metadata"]
    assert meta["width"]==200
    assert meta["height"]==100


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
    assert "frames" not in data

    assert data["original_filename"]=="test.mp4"
    assert data["media_type"]=="video"
    meta = client.get(f"/api/v1/media/{data['media_id']}").json()["metadata"]
    assert meta["width"]==200
    assert meta["height"]==100

    # Frames are still extracted at ingest; they are just not echoed back.
    from app.core.config import settings
    frames = list((settings.frames_dir / data["media_id"]).glob("*.jpg"))
    assert len(frames) == 5

