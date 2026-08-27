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


def test_invalid_file_type():

    service=MediaIngestionService()

    file=create_file(
        filename="test.txt",
        content=b"hello",
        content_type="text/plain"
    )

    with pytest.raises(ValueError):
        service.validate_file(file)


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