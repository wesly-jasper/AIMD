"""scripts/fetch_models.py: archive safety and the CocoGlide parquet importer."""
import importlib.util
import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "fetch_models.py"
_spec = importlib.util.spec_from_file_location("fetch_models", SCRIPT)
fetch_models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch_models)


def _png(value: int, mode: str = "RGB") -> bytes:
    shape = (8, 8, 3) if mode == "RGB" else (8, 8)
    buffer = io.BytesIO()
    Image.fromarray(np.full(shape, value, dtype=np.uint8), mode).save(buffer, "PNG")
    return buffer.getvalue()


def _parquet(path: Path, rows: list[dict]) -> Path:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    schema = pa.schema([
        ("key", pa.string()), ("image", image_type), ("mask", image_type),
        ("label", pa.int32()), ("prompt", pa.string()),
    ])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    return path


def _row(key, label, image, mask):
    return {
        "key": key, "label": label, "prompt": "teddy bear" if label else None,
        "image": {"bytes": image, "path": None},
        "mask": {"bytes": mask, "path": None} if mask is not None else None,
    }


def test_cocoglide_parquet_unpacks_fakes_masks_and_reals(tmp_path):
    fake, mask, real = _png(200), _png(255, "L"), _png(50)
    parquet = _parquet(tmp_path / "glide.parquet", [
        _row("glide/fake/glide_inpainting_val2017_1_up.png", 1, fake, mask),
        _row("glide/fake/glide_inpainting_val2017_2_up.png", 1, fake, mask),
        _row("glide/real/baseball bat_3.png", 0, real, None),
    ])
    manifest = {}

    fetch_models.install_cocoglide_parquet(tmp_path, parquet, "test", manifest)

    root = tmp_path / "CocoGlide"
    assert (root / "fake" / "glide_inpainting_val2017_1_up.png").read_bytes() == fake
    assert (root / "mask" / "glide_inpainting_val2017_1_up_mask.png").read_bytes() == mask
    assert (root / "real" / "baseball bat_3.png").read_bytes() == real
    assert manifest["cocoglide"]["counts"] == {"fake": 2, "real": 1, "mask": 2}
    assert manifest["cocoglide"]["verification"] == "unverified"
    assert len(manifest["cocoglide"]["parquet"]["sha256"]) == 64


@pytest.mark.parametrize("key", ["glide/fake/", "glide/fake/run.exe", "glide/fake/.hidden.png"])
def test_cocoglide_parquet_refuses_unusable_names(tmp_path, key):
    parquet = _parquet(tmp_path / "glide.parquet", [_row(key, 1, _png(1), _png(255, "L"))])

    with pytest.raises(fetch_models.VerificationError):
        fetch_models.install_cocoglide_parquet(tmp_path, parquet, "test", {})


def test_cocoglide_parquet_refuses_a_fake_without_its_mask(tmp_path):
    parquet = _parquet(tmp_path / "glide.parquet", [
        _row("glide/fake/a_up.png", 1, _png(1), None),
    ])

    with pytest.raises(fetch_models.VerificationError, match="masks"):
        fetch_models.install_cocoglide_parquet(tmp_path, parquet, "test", {})


def test_cocoglide_parquet_refuses_non_image_bytes(tmp_path):
    parquet = _parquet(tmp_path / "glide.parquet", [
        _row("glide/fake/a_up.png", 1, b"MZ\x90\x00not an image", _png(255, "L")),
    ])

    with pytest.raises(fetch_models.VerificationError, match="PNG nor JPEG"):
        fetch_models.install_cocoglide_parquet(tmp_path, parquet, "test", {})


def test_cocoglide_parquet_refuses_another_dataset(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    parquet = tmp_path / "other.parquet"
    pq.write_table(pa.table({"text": ["hello"]}), parquet)

    with pytest.raises(fetch_models.VerificationError, match="missing columns"):
        fetch_models.install_cocoglide_parquet(tmp_path, parquet, "test", {})


@pytest.mark.parametrize("name", ["../escape.txt", "/abs.txt", "C:/drive.txt", "a/../../b.txt"])
def test_extract_all_refuses_paths_outside_the_destination(tmp_path, name):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(name, "x")

    with pytest.raises(fetch_models.VerificationError, match="escapes"):
        fetch_models.extract_all(archive, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()
