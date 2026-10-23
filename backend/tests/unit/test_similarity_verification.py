"""
Similarity retrieval and verification.

Done-when for Day 3: a resized or re-compressed copy of an indexed image is
retrieved AND verified, while an unrelated image that looks similar is not.

These tests use the real pHash and the real ORB check on synthetic photo-like
images. The CLIP embedder is a stub, so a test can make two unrelated images sit
as close in embedding space as it likes -- the case verification must survive.
"""
import uuid

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from app.db import repository as repo
from app.db.database import SessionLocal
from app.services.fingerprinting.fingerprint import (
    EMBEDDING_ALGORITHM,
    FingerprintService,
    decode_embedding,
    encode_embedding,
)
from app.services.similarity.similarity_engine import (
    NO_MATCH,
    SIMILAR_UNVERIFIED,
    VERIFIED_MATCH,
    SimilarityEngine,
    orb_geometric_check,
    search_media_fingerprints,
    verify_candidate,
)


def photo(seed: int, size=(800, 600)) -> Image.Image:
    """Textured, photo-like content: smooth noise plus shapes, softly blurred."""
    rng = np.random.default_rng(seed)
    w, h = size
    base = rng.normal(128, 40, (h // 8, w // 8, 3)).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(base).resize(size, Image.Resampling.BICUBIC)
    draw = ImageDraw.Draw(img)
    for _ in range(60):
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        x1, y1 = x0 + int(rng.integers(10, 160)), y0 + int(rng.integers(10, 160))
        colour = tuple(int(c) for c in rng.integers(0, 256, 3))
        if rng.random() < 0.5:
            draw.ellipse((x0, y0, x1, y1), fill=colour)
        else:
            draw.rectangle((x0, y0, x1, y1), fill=colour)
    return img.filter(ImageFilter.GaussianBlur(1))


def resized_copy(img: Image.Image, path, scale=0.5, quality=70):
    w, h = img.size
    img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS).save(
        path, "JPEG", quality=quality
    )
    return path


class ConstantEmbedder:
    """Every image embeds to the same vector: CLIP saying 'these look alike'."""

    version_tag = "stub-constant/1"

    def embed(self, _path):
        v = np.ones(768, dtype=np.float32)
        return v / np.linalg.norm(v)


class SeededEmbedder:
    """A different random direction per file: unrelated in embedding space."""

    version_tag = "stub-seeded/1"

    def embed(self, path):
        seed = abs(hash(str(path))) % (2 ** 32)
        v = np.random.default_rng(seed).normal(size=768).astype(np.float32)
        return v / np.linalg.norm(v)


@pytest.fixture
def images(tmp_path):
    original = photo(1)
    paths = {
        "original": tmp_path / "original.png",
        "copy": tmp_path / "copy.jpg",
        "unrelated": tmp_path / "unrelated.png",
    }
    original.save(paths["original"])
    resized_copy(original, paths["copy"])
    photo(2).save(paths["unrelated"])
    return paths


@pytest.fixture
def db():
    with SessionLocal() as session:
        yield session


def _index(db, path, owner, embedder, media_id=None) -> str:
    """Ingest one file as ``owner``'s media with all its fingerprints."""
    media_id = media_id or str(uuid.uuid4())
    repo.create_media(db, {
        "id": media_id, "original_filename": path.name, "stored_filename": path.name,
        "file_path": str(path), "content_type": "image/png", "media_type": "image",
        "size_bytes": path.stat().st_size, "sha256": "0" * 64, "uploaded_by": owner,
    })
    fps = FingerprintService(embedder=embedder).generate(path, media_id=media_id)
    repo.bulk_create_fingerprints(db, fps["_records"])
    return media_id


def _query(path, embedder):
    fps = FingerprintService(embedder=embedder).generate(path)
    query = {r["algorithm"]: r["value"] for r in fps["_records"]}
    return query, fps.get(EMBEDDING_ALGORITHM + "_version")


# ── Embedding storage ────────────────────────────────────────────────────────

def test_embedding_round_trips_through_its_stored_encoding():
    v = SeededEmbedder().embed("x")
    back = decode_embedding(encode_embedding(v))
    assert back.dtype == np.float32 and back.shape == (768,)
    assert float(back @ v) > 0.9999


def test_an_embedder_adds_a_versioned_embedding_record(images):
    fps = FingerprintService(embedder=ConstantEmbedder()).generate(images["original"])
    [record] = [r for r in fps["_records"] if r["algorithm"] == EMBEDDING_ALGORITHM]
    assert record["algorithm_version"] == "stub-constant/1/retrieval224/l2"
    assert decode_embedding(record["value"]).shape == (768,)


def test_skip_leaves_out_stored_algorithms_and_the_embedding(images):
    class Exploding:
        version_tag = "never"

        def embed(self, _path):
            raise AssertionError("embedding computed although it was skipped")

    fps = FingerprintService(embedder=Exploding()).generate(
        images["original"], skip={"sha256", EMBEDDING_ALGORITHM}
    )
    algorithms = {r["algorithm"] for r in fps["_records"]}
    assert algorithms == {"phash", "dhash"}


def test_no_embedder_means_no_embedding_record(images, monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "similarity_embedding_enabled", False)
    fps = FingerprintService().generate(images["original"])
    assert EMBEDDING_ALGORITHM not in {r["algorithm"] for r in fps["_records"]}


# ── Geometric check ──────────────────────────────────────────────────────────

def test_orb_finds_one_homography_for_a_resized_recompressed_copy(images):
    orb = orb_geometric_check(images["original"], images["copy"])
    assert orb["inliers"] >= 100
    assert orb["inlier_ratio"] >= 0.5


def test_orb_finds_no_homography_between_unrelated_images(images):
    orb = orb_geometric_check(images["original"], images["unrelated"])
    assert orb["inliers"] < 20


# ── Verification ─────────────────────────────────────────────────────────────

def test_resized_copy_is_verified(images):
    candidate = {"sha256_exact": False, "phash_distance": 2, "cosine": 0.97,
                 "retrieved_by": ["phash", EMBEDDING_ALGORITHM]}
    status, verification = verify_candidate(images["copy"], candidate, images["original"])
    assert status == VERIFIED_MATCH
    assert verification["gates"] == {"phash": True, "cosine": True, "geometry": True}
    assert verification["reasons"] == []


def test_lookalike_with_high_cosine_is_not_verified(images):
    # CLIP and even pHash agree; the geometry does not.
    candidate = {"sha256_exact": False, "phash_distance": 8, "cosine": 0.99,
                 "retrieved_by": ["phash", EMBEDDING_ALGORITHM]}
    status, verification = verify_candidate(images["unrelated"], candidate, images["original"])
    assert status == SIMILAR_UNVERIFIED
    assert verification["gates"]["geometry"] is False
    assert any("ORB inliers" in r for r in verification["reasons"])


def test_every_gate_failing_is_no_match(images):
    candidate = {"sha256_exact": False, "phash_distance": 30, "cosine": 0.5,
                 "retrieved_by": ["phash"]}
    status, verification = verify_candidate(images["unrelated"], candidate, images["original"])
    assert status == NO_MATCH
    assert set(verification["gates"].values()) == {False}


def test_missing_candidate_file_cannot_verify(images, tmp_path):
    candidate = {"sha256_exact": False, "phash_distance": 0, "cosine": 0.99,
                 "retrieved_by": ["phash"]}
    status, verification = verify_candidate(
        images["copy"], candidate, tmp_path / "gone.png"
    )
    assert status == SIMILAR_UNVERIFIED
    assert verification["gates"]["geometry"] is None
    assert "candidate file is missing from storage" in verification["reasons"]


def test_missing_embedding_cannot_verify(images):
    candidate = {"sha256_exact": False, "phash_distance": 0, "cosine": None,
                 "retrieved_by": ["phash"]}
    status, verification = verify_candidate(images["copy"], candidate, images["original"])
    assert status == SIMILAR_UNVERIFIED
    assert verification["gates"]["cosine"] is None


def test_exact_sha256_is_verified_without_further_checks(tmp_path):
    candidate = {"sha256_exact": True, "phash_distance": None, "cosine": None,
                 "retrieved_by": ["sha256"]}
    status, verification = verify_candidate(tmp_path / "q.png", candidate, None)
    assert status == VERIFIED_MATCH
    assert verification["method"] == "sha256"


def test_thresholds_are_read_from_settings(images, monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "similarity_min_inliers", 100_000)
    candidate = {"sha256_exact": False, "phash_distance": 0, "cosine": 0.99,
                 "retrieved_by": ["phash"]}
    status, verification = verify_candidate(images["copy"], candidate, images["original"])
    assert status == SIMILAR_UNVERIFIED
    assert verification["thresholds"]["min_inliers"] == 100_000


# ── Retrieval ────────────────────────────────────────────────────────────────

def test_retrieval_ranks_and_scopes_to_the_owner(db, images):
    owner, other = f"owner-{uuid.uuid4()}", f"other-{uuid.uuid4()}"
    embedder = ConstantEmbedder()
    original = _index(db, images["original"], owner, embedder)
    unrelated = _index(db, images["unrelated"], owner, embedder)
    foreign = _index(db, images["original"], other, embedder)

    query, version = _query(images["copy"], embedder)
    candidates = search_media_fingerprints(query, db, owner, embedding_version=version)

    ids = [c["media_id"] for c in candidates]
    assert foreign not in ids, "another owner's media was retrieved"
    # Equal cosines (constant embedder), so the nearer pHash ranks first.
    assert ids == [original, unrelated]
    assert candidates[0]["phash_distance"] < candidates[1]["phash_distance"]
    assert candidates[0]["retrieved_by"] == ["phash", EMBEDDING_ALGORITHM]


def test_retrieval_needs_an_owner(db, images):
    query, _ = _query(images["copy"], ConstantEmbedder())
    assert search_media_fingerprints(query, db, owner=None) == []


def test_retrieval_ignores_embeddings_from_another_model(db, images):
    owner = f"owner-{uuid.uuid4()}"
    _index(db, images["unrelated"], owner, ConstantEmbedder())
    query, _ = _query(images["copy"], ConstantEmbedder())
    # The unrelated image is far in pHash, so only the embedding could retrieve it.
    assert search_media_fingerprints(query, db, owner, embedding_version="other/1") == []
    assert len(search_media_fingerprints(query, db, owner)) == 1


def test_exact_copy_ranks_first(db, images):
    owner = f"owner-{uuid.uuid4()}"
    embedder = SeededEmbedder()
    copy = _index(db, images["copy"], owner, embedder)
    original = _index(db, images["original"], owner, embedder)
    query, version = _query(images["original"], embedder)
    candidates = search_media_fingerprints(query, db, owner, embedding_version=version)
    assert candidates[0]["media_id"] == original
    assert candidates[0]["sha256_exact"] is True
    assert copy in [c["media_id"] for c in candidates]


# ── End to end: the done-when ────────────────────────────────────────────────

def test_copy_is_retrieved_and_verified_but_lookalike_is_not(db, images):
    owner = f"owner-{uuid.uuid4()}"
    embedder = ConstantEmbedder()  # the worst case: CLIP can't tell them apart
    original = _index(db, images["original"], owner, embedder)
    engine = SimilarityEngine()

    query, version = _query(images["copy"], embedder)
    result = engine.search(str(images["copy"]), query, db, owner=owner,
                           embedding_version=version)
    [match] = result["local_matches"]
    assert match["source_media_id"] == original
    assert match["verification_status"] == VERIFIED_MATCH
    assert match["verification"]["orb"]["inliers"] >= 20

    query, version = _query(images["unrelated"], embedder)
    result = engine.search(str(images["unrelated"]), query, db, owner=owner,
                           embedding_version=version)
    statuses = [m["verification_status"] for m in result["local_matches"]]
    assert VERIFIED_MATCH not in statuses
    assert statuses == [SIMILAR_UNVERIFIED]


def test_unrelated_image_with_unrelated_embedding_is_not_retrieved(db, images):
    owner = f"owner-{uuid.uuid4()}"
    embedder = SeededEmbedder()
    _index(db, images["original"], owner, embedder)
    query, version = _query(images["unrelated"], embedder)
    result = SimilarityEngine().search(str(images["unrelated"]), query, db,
                                       owner=owner, embedding_version=version)
    assert result["local_matches"] == []
