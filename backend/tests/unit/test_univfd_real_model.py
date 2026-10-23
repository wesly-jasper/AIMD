"""
UnivFD and CLIP retrieval on the real weights.

Skipped until ``scripts/fetch_univfd.py`` has installed both files into
``backend/models/univfd/``. The done-when for Day 3 is checked here on real
photographs: CocoGlide's ``real/`` set, which holds several COCO images of the
same class -- the "looks similar, is unrelated" case CLIP alone cannot settle.
"""
import math
import uuid
from collections import defaultdict
from pathlib import Path

import pytest
from PIL import Image

from app.core.config import settings
from app.db import repository as repo
from app.db.database import SessionLocal
from app.schemas.detection import DetectorExecutionStatus
from app.services.detection.models import univfd_release as release
from app.services.fingerprinting.fingerprint import EMBEDDING_ALGORITHM, FingerprintService
from app.services.similarity.similarity_engine import VERIFIED_MATCH, SimilarityEngine

BACKEND = Path(__file__).resolve().parents[2]
UNIVFD_ROOT = BACKEND / "models" / "univfd"
COCOGLIDE_REAL = BACKEND / "models" / "trufor" / "CocoGlide" / "real"

pytestmark = pytest.mark.skipif(
    not ((UNIVFD_ROOT / release.CLIP_FILE).is_file()
         and (UNIVFD_ROOT / release.HEAD_FILE).is_file()),
    reason="UnivFD weights not installed (run scripts/fetch_univfd.py)",
)


@pytest.fixture(scope="module")
def backbone():
    from app.services.detection.models.clip_backbone import get_clip_backbone

    return get_clip_backbone(UNIVFD_ROOT, device=settings.device)


@pytest.fixture(scope="module")
def same_class_pair():
    """Two different CocoGlide real photographs of the same COCO class."""
    if not COCOGLIDE_REAL.is_dir():
        pytest.skip("CocoGlide is not installed (scripts/fetch_models.py)")
    by_class = defaultdict(list)
    for path in sorted(COCOGLIDE_REAL.glob("*.png")):
        by_class[path.stem.rsplit("_", 1)[0]].append(path)
    for paths in by_class.values():
        if len(paths) >= 2:
            return paths[0], paths[1]
    pytest.skip("no CocoGlide class has two real images")


def test_univfd_scores_a_real_photograph(tmp_path, same_class_pair):
    from app.services.detection.detectors.image.univfd_detector import UnivFDDetector
    from app.services.detection.models.univfd_runner import UnivFDRunner

    runner = UnivFDRunner(UNIVFD_ROOT, device=settings.device)
    runner.load()
    result = UnivFDDetector(runner, threshold=settings.univfd_threshold).detect(
        same_class_pair[0]
    )
    assert result.execution_status == DetectorExecutionStatus.COMPLETED
    assert 0.0 <= result.confidence <= 1.0
    assert math.isfinite(result.metadata["logit"])
    assert result.metadata["branch"] == "ai_generation"
    assert result.metadata["clip_sha256"] == release.CLIP_SHA256
    expected = "fp16" if runner.backbone.device.type == "cuda" else "fp32"
    assert result.metadata["clip_precision"] == expected


def test_resized_copy_embeds_close_and_another_photo_does_not(tmp_path, backbone,
                                                             same_class_pair):
    original, other = same_class_pair
    copy = tmp_path / "copy.jpg"
    with Image.open(original) as img:
        img.convert("RGB").resize((img.width // 2, img.height // 2)).save(
            copy, "JPEG", quality=70
        )
    a, c, b = (backbone.embed(p) for p in (original, copy, other))
    assert float(a @ c) >= settings.similarity_cosine_min
    assert float(a @ b) < float(a @ c)


def test_done_when_on_real_photographs(tmp_path, backbone, same_class_pair):
    original, other = same_class_pair
    owner = f"owner-{uuid.uuid4()}"
    service = FingerprintService(embedder=backbone)

    with SessionLocal() as db:
        ids = {}
        for path in (original, other):
            media_id = str(uuid.uuid4())
            repo.create_media(db, {
                "id": media_id, "original_filename": path.name,
                "stored_filename": path.name, "file_path": str(path),
                "content_type": "image/png", "media_type": "image",
                "size_bytes": path.stat().st_size, "sha256": "0" * 64,
                "uploaded_by": owner,
            })
            repo.bulk_create_fingerprints(
                db, service.generate(path, media_id=media_id)["_records"]
            )
            ids[path] = media_id

        copy = tmp_path / "copy.jpg"
        with Image.open(original) as img:
            img.convert("RGB").resize((img.width // 2, img.height // 2)).save(
                copy, "JPEG", quality=70
            )
        fps = service.generate(copy)
        query = {r["algorithm"]: r["value"] for r in fps["_records"]}
        version = next(r["algorithm_version"] for r in fps["_records"]
                       if r["algorithm"] == EMBEDDING_ALGORITHM)
        result = SimilarityEngine().search(str(copy), query, db, owner=owner,
                                           embedding_version=version)

    status = {m["source_media_id"]: m["verification_status"]
              for m in result["local_matches"]}
    assert status.get(ids[original]) == VERIFIED_MATCH, [(m["verification"]["reasons"], m["verification"].get("orb")) for m in result["local_matches"]]
    assert status.get(ids[other]) != VERIFIED_MATCH, (
        "a different photograph of the same class was verified as a copy"
    )
