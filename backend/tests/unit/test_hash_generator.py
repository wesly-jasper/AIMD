import cv2
import numpy as np

from app.services.fingerprinting.hash_generator import HashGenerator


def create_test_image(path,value=0):
    image=np.zeros(
        (100,200,3),
        dtype=np.uint8
    )

    image[:]=value

    success=cv2.imwrite(
        str(path),
        image
    )

    assert success


def test_generate_sha256(tmp_path):
    file_path=tmp_path/"test.txt"

    file_path.write_bytes(
        b"hello world"
    )

    generator=HashGenerator()

    result=generator.generate_sha256(
        file_path
    )

    assert len(result)==64

    assert result==(
        "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    )


def test_sha256_same_file(tmp_path):
    file_path=tmp_path/"test.txt"

    file_path.write_bytes(
        b"hello world"
    )

    generator=HashGenerator()

    hash1=generator.generate_sha256(file_path)
    hash2=generator.generate_sha256(file_path)

    assert hash1==hash2


def test_generate_phash(tmp_path):
    file_path=tmp_path/"test.jpg"

    create_test_image(
        file_path,
        value=100
    )

    generator=HashGenerator()

    result=generator.generate_phash(
        file_path
    )

    assert len(result)==64
    assert all(
        bit in "01"
        for bit in result
    )


def test_file_not_found(tmp_path):
    generator=HashGenerator()

    try:
        generator.generate_sha256(
            tmp_path/"missing.txt"
        )
        assert False
    except FileNotFoundError:
        assert True


def test_phash_file_not_found(tmp_path):
    generator=HashGenerator()

    try:
        generator.generate_phash(
            tmp_path/"missing.jpg"
        )
        assert False
    except FileNotFoundError:
        assert True