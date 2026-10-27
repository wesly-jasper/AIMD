import hashlib
from io import BytesIO

import cv2
import numpy as np
import pytest
from fastapi import UploadFile

from app.services.ingestion.media_ingestion import MediaIngestionService


def create_file(
    filename="test.jpg",
    content=b"test data",
    content_type="image/jpeg"
):
    return UploadFile(
        file=BytesIO(content),
        filename=filename,
        headers={
            "content-type": content_type
        }
    )


def create_test_image():

    image=np.zeros(
        (100,200,3),
        dtype=np.uint8
    )

    success,encoded=cv2.imencode(
        ".jpg",
        image
    )

    assert success

    return encoded.tobytes()


def test_sha256():

    service=MediaIngestionService()

    data=b"hello world"

    expected=hashlib.sha256(
        data
    ).hexdigest()

    assert service.generate_sha256(
        data
    )==expected


@pytest.mark.asyncio
async def test_invalid_file_type(tmp_path):
    """Text is refused whatever it is declared as: the bytes decide."""
    service=MediaIngestionService()
    service.upload_dir=tmp_path

    for declared in ("text/plain", "image/png"):
        file=create_file(filename="test.txt", content=b"hello", content_type=declared)
        with pytest.raises(ValueError, match="not a supported"):
            await service.ingest(file)
    assert list(tmp_path.iterdir())==[]


def test_valid_file_type():

    service=MediaIngestionService()

    file=create_file(
        filename="test.jpg",
        content=create_test_image(),
        content_type="image/jpeg"
    )

    service.validate_file(file)


@pytest.mark.asyncio
async def test_ingest(tmp_path):

    service=MediaIngestionService()

    service.upload_dir=tmp_path

    image_data=create_test_image()

    file=create_file(
        filename="test.jpg",
        content=image_data,
        content_type="image/jpeg"
    )

    result=await service.ingest(file)

    assert result["status"]=="UPLOADED"

    assert result["original_filename"]=="test.jpg"

    assert result["content_type"]=="image/jpeg"

    assert result["size_bytes"]==len(image_data)

    expected_hash=hashlib.sha256(
        image_data
    ).hexdigest()

    assert result["sha256"]==expected_hash

    assert "metadata" in result

    assert result["metadata"]["media_type"]=="image"

    assert result["metadata"]["width"]==200

    assert result["metadata"]["height"]==100

    saved_file=tmp_path/result["stored_filename"]

    assert saved_file.exists()


@pytest.mark.asyncio
async def test_ingest_includes_metadata(tmp_path):

    service=MediaIngestionService()

    service.upload_dir=tmp_path

    image_data=create_test_image()

    file=create_file(
        filename="test.jpg",
        content=image_data,
        content_type="image/jpeg"
    )

    result=await service.ingest(file)

    assert "metadata" in result

    assert result["metadata"]["media_type"]=="image"

    assert result["metadata"]["width"]==200

    assert result["metadata"]["height"]==100


def create_png():
    success, encoded = cv2.imencode(".png", np.full((40, 60, 3), 90, dtype=np.uint8))
    assert success
    return encoded.tobytes()


@pytest.mark.asyncio
@pytest.mark.parametrize("declared", ["application/octet-stream", "image/jpeg", ""])
async def test_a_real_png_is_accepted_and_typed_by_its_content(tmp_path, declared):
    """A missing or wrong declared type no longer refuses, or mislabels, a real image."""
    service=MediaIngestionService()
    service.upload_dir=tmp_path

    result=await service.ingest(create_file(filename="upload", content=create_png(), content_type=declared))

    assert result["content_type"]=="image/png"
    assert result["media_type"]=="image"
    assert result["stored_filename"].endswith(".png")
    assert result["metadata"]["width"]==60


@pytest.mark.asyncio
async def test_html_named_and_declared_as_jpeg_is_refused(tmp_path):
    service=MediaIngestionService()
    service.upload_dir=tmp_path
    page=b"<!DOCTYPE html><html><head><title>This Person Does Not Exist</title></head></html>"

    with pytest.raises(ValueError, match="not a supported"):
        await service.ingest(create_file(filename="face.jpg", content=page, content_type="image/jpeg"))


@pytest.mark.asyncio
async def test_a_truncated_jpeg_is_refused(tmp_path):
    """The signature alone is not enough: the image must decode."""
    service=MediaIngestionService()
    service.upload_dir=tmp_path
    rng=np.random.default_rng(1)
    ok, encoded=cv2.imencode(".jpg", rng.integers(0, 255, (400, 400, 3), dtype=np.uint8))
    truncated=encoded.tobytes()[:4000]

    with pytest.raises(ValueError, match="could not be decoded"):
        await service.ingest(create_file(filename="cut.jpg", content=truncated))
    assert list(tmp_path.iterdir())==[]


@pytest.mark.asyncio
async def test_a_video_signature_without_a_video_is_refused_and_removed(tmp_path):
    service=MediaIngestionService()
    service.upload_dir=tmp_path
    fake=b"\x00\x00\x00\x24ftypisom" + b"\x00" * 200

    with pytest.raises(ValueError, match="could not be decoded"):
        await service.ingest(create_file(filename="clip.mp4", content=fake, content_type="video/mp4"))
    assert list(tmp_path.iterdir())==[]


@pytest.mark.parametrize("head, expected", [
    (b"\xff\xd8\xff\xe0", ("image/jpeg", "image", ".jpg")),
    (b"II*\x00rest", ("image/tiff", "image", ".tiff")),
    (b"MM\x00*rest", ("image/tiff", "image", ".tiff")),
    (b"RIFF\x00\x00\x00\x00WEBPVP8 ", ("image/webp", "image", ".webp")),
    (b"\x00\x00\x00\x14ftypqt  ", ("video/quicktime", "video", ".mov")),
    (b"\x00\x00\x00\x24ftypmp42", ("video/mp4", "video", ".mp4")),  # box size not 0x18/0x1c/0x20
    (b"\x00\x00\x00\x20ftypM4A ", ("audio/mp4", "audio", ".m4a")),
    (b"\x1aE\xdf\xa3", ("video/x-matroska", "video", ".mkv")),
    (b"RIFF\x00\x00\x00\x00WAVEfmt ", ("audio/wav", "audio", ".wav")),
    (b"ID3\x04", ("audio/mpeg", "audio", ".mp3")),
    (b"\xff\xfb\x90", ("audio/mpeg", "audio", ".mp3")),
    (b"\xff\xf1\x50", ("audio/aac", "audio", ".aac")),
    (b"GIF89a", None),
    (b"%PDF-1.7", None),
])
def test_types_come_from_signatures(head, expected):
    from app.services.ingestion.media_ingestion import _sniff
    assert _sniff(head) == expected
