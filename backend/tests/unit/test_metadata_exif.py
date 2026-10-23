"""
EXIF from real cameras must survive extraction as JSON.

Pillow returns EXIF rationals as ``IFDRational``, alone (ExposureTime, FNumber)
or inside tuples (LensSpecification, GPS degrees). Stored as they were, they made
``INSERT INTO media_metadata`` fail, and with it the upload of every camera photo.
"""
import io
import json
import math

from PIL import Image, TiffImagePlugin

from app.services.preprocessing.metadata_extractor import MetadataExtractor, json_safe

R = TiffImagePlugin.IFDRational


def camera_jpeg(size=(64, 48)) -> bytes:
    """A JPEG with the EXIF shapes a camera writes: rationals, tuples of them, GPS."""
    exif = Image.Exif()
    exif[0x010F] = "Canon"
    exif[0x0110] = "EOS 5D Mark IV"
    ifd = exif.get_ifd(0x8769)
    ifd[0x829A] = R(1, 250)                                  # ExposureTime
    ifd[0x829D] = R(28, 10)                                  # FNumber
    ifd[0xA432] = (R(24, 1), R(70, 1), R(28, 10), R(0, 0))   # LensSpecification, 0/0 allowed
    gps = exif.get_ifd(0x8825)
    gps[1] = "N"                                             # GPSLatitudeRef
    gps[2] = (R(51, 1), R(30, 1), R(0, 1))                   # GPSLatitude
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 90, 60)).save(buf, "JPEG", exif=exif)
    return buf.getvalue()


def test_json_safe_converts_rationals_tuples_and_blobs():
    assert json_safe(R(1, 250)) == 0.004
    assert json_safe(R(0, 0)) is None
    assert json_safe((R(51, 1), R(30, 1))) == [51.0, 30.0]
    assert json_safe(b"\x00\x01") is None
    assert json_safe(float("nan")) is None and json_safe(float("inf")) is None
    assert json_safe({1: "N", "nested": (R(1, 2),)}) == {"1": "N", "nested": [0.5]}
    assert json_safe("Canon\x00") == "Canon"
    assert json_safe(True) is True and json_safe(None) is None


def test_camera_exif_is_extracted_as_json(tmp_path):
    path = tmp_path / "camera.jpg"
    path.write_bytes(camera_jpeg())

    meta = MetadataExtractor().extract(path)
    json.dumps(meta, allow_nan=False)  # the column is JSON: this used to raise

    exif = meta["exif"]
    assert exif["Make"] == "Canon"
    assert math.isclose(exif["ExposureTime"], 0.004)  # rationals were dropped before
    assert math.isclose(exif["FNumber"], 2.8)
    assert exif["LensSpecification"] == [24.0, 70.0, 2.8, None]
    assert exif["GPSInfo"]["GPSLatitudeRef"] == "N"
    assert exif["GPSInfo"]["GPSLatitude"] == [51.0, 30.0, 0.0]
