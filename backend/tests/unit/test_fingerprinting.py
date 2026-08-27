import cv2
import numpy as np

from app.services.fingerprinting.fingerprint import FingerprintService


def create_test_image(path):
    image=np.zeros(
        (100,200,3),
        dtype=np.uint8
    )

    success=cv2.imwrite(
        str(path),
        image
    )

    assert success


def test_generate_image_fingerprint(tmp_path):
    file_path=tmp_path/"test.jpg"

    create_test_image(file_path)

    service=FingerprintService()

    result=service.generate(file_path)

    assert "sha256" in result
    assert "phash" in result

    assert len(result["sha256"])==64
    assert len(result["phash"])==64


def test_generate_video_fingerprint(tmp_path):
    file_path=tmp_path/"test.mp4"

    file_path.write_bytes(
        b"test video"
    )

    service=FingerprintService()

    result=service.generate(file_path)

    assert "sha256" in result
    assert "phash" not in result

    assert len(result["sha256"])==64


def test_fingerprint_file_not_found(tmp_path):
    service=FingerprintService()

    try:
        service.generate(
            tmp_path/"missing.jpg"
        )
        assert False
    except FileNotFoundError:
        assert True