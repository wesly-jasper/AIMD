"""Download Google's pinned MediaPipe Face Landmarker model asset.

The model is distributed under the Apache License 2.0. The matching model
cards are published alongside the official MediaPipe model documentation.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from urllib.request import urlopen

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
DEFAULT_DESTINATION = (
    Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"
)


def download_model(destination: Path) -> Path:
    destination = destination.resolve()
    if destination.is_file():
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if digest == MODEL_SHA256:
            return destination
        raise RuntimeError(f"Existing model asset failed SHA-256 verification: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    digest = hashlib.sha256()
    try:
        with urlopen(MODEL_URL, timeout=60) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != MODEL_SHA256:
            raise RuntimeError(f"Downloaded model SHA-256 mismatch: {actual}")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", nargs="?", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    path = download_model(args.destination)
    print(f"Verified MediaPipe Face Landmarker model: {path}")


if __name__ == "__main__":
    main()
