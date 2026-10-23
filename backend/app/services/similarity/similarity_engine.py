"""
Similarity engine.

Local search over this owner's stored fingerprints, then (optionally) an
external search provider.

Local search is two steps, and only the second may call a result a match:

  1. Retrieval (``search_media_fingerprints``) casts a wide net: an exact
     SHA-256, a pHash within ``SIMILARITY_CANDIDATE_PHASH_DISTANCE`` bits, or a
     CLIP embedding cosine of at least ``SIMILARITY_CANDIDATE_COSINE``. Ranked
     candidates come back with every measurement taken, not just the one that
     retrieved them.
  2. Verification (``verify_candidate``) labels each candidate:

       VERIFIED_MATCH      exact SHA-256, or pHash distance, embedding cosine
                           and ORB keypoints consistent with one homography all
                           pass their (tighter) thresholds.
       SIMILAR_UNVERIFIED  some check passed, or one could not run: a lead, not
                           a match.
       NO_MATCH            every check ran and none passed.

Neither the pHash nor the embedding is enough on its own. Two photographs of the
same kind of scene can share a coarse DCT pattern and sit close in CLIP space;
what they cannot share is hundreds of local features related by one geometric
transform. A resized or re-compressed copy keeps all three.

External search is performed only when a configured provider is present.
"""
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sqlalchemy.orm import Session

from app.services.fingerprinting.fingerprint import EMBEDDING_ALGORITHM, decode_embedding
from app.services.similarity.internet_search_provider import (
    BaseSourceSearchProvider,
    DisabledSourceSearchProvider,
    ProviderStatus,
    SourceSearchResult,
)

logger = logging.getLogger(__name__)

VERIFIED_MATCH = "VERIFIED_MATCH"
SIMILAR_UNVERIFIED = "SIMILAR_UNVERIFIED"
NO_MATCH = "NO_MATCH"

# ORB geometric check. Both images are scaled to at most ORB_MAX_SIDE on the
# long side first, so a 6000 px original and its 1500 px copy are compared at
# comparable scales and in bounded time.
ORB_FEATURES = 2000
ORB_RATIO = 0.75
ORB_MAX_SIDE = 1024
ORB_RANSAC_REPROJ_PX = 5.0
ORB_MIN_INLIER_RATIO = 0.25


def _hamming_distance(hash_a: str, hash_b: str) -> int:
    """Compute Hamming distance between two binary hash strings."""
    if len(hash_a) != len(hash_b):
        return len(hash_a)  # Treat incompatible lengths as maximum distance
    return sum(a != b for a, b in zip(hash_a, hash_b))


def _hamming_similarity(hash_a: str, hash_b: str) -> float:
    """Convert Hamming distance to [0,1] similarity score."""
    if not hash_a or not hash_b:
        return 0.0
    dist = _hamming_distance(hash_a, hash_b)
    max_bits = max(len(hash_a), len(hash_b))
    return 1.0 - (dist / max_bits)


def _thresholds() -> dict[str, float]:
    """Read at call time, so a changed setting applies without a restart."""
    from app.core.config import settings

    return {
        "phash_max_distance": settings.similarity_phash_max_distance,
        "cosine_min": settings.similarity_cosine_min,
        "min_inliers": settings.similarity_min_inliers,
        "min_inlier_ratio": ORB_MIN_INLIER_RATIO,
        "candidate_phash_distance": settings.similarity_candidate_phash_distance,
        "candidate_cosine": settings.similarity_candidate_cosine,
    }


def _unit(vec: np.ndarray) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else None


# ── Retrieval ─────────────────────────────────────────────────────────────────

def search_media_fingerprints(
    query_fps: dict[str, str],
    db: Session,
    owner: Optional[str],
    exclude_media_id: Optional[str] = None,
    top_k: int = 10,
    embedding_version: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Ranked candidate media for ``query_fps`` ({algorithm: value}), one owner's only.

    Each candidate is ``{media_id, sha256_exact, phash_distance, cosine,
    retrieved_by}``. ``phash_distance`` and ``cosine`` are ``None`` where either
    side has no such fingerprint. With ``embedding_version`` set, only
    embeddings from the same model and preprocessing are compared -- cosines
    across two different models mean nothing.

    Ranked exact matches first, then by cosine, then by pHash distance.
    """
    from app.db.repository import get_all_fingerprints

    if not owner:
        return []
    th = _thresholds()

    def stored(algorithm: str):
        return [
            fp for fp in get_all_fingerprints(db, algorithm, owner)
            if fp.media_id and fp.media_id != exclude_media_id
        ]

    exact: set[str] = set()
    if query_sha := query_fps.get("sha256"):
        exact = {fp.media_id for fp in stored("sha256") if fp.value == query_sha}

    phash_distance: dict[str, int] = {}
    if query_phash := query_fps.get("phash"):
        for fp in stored("phash"):
            if fp.value and len(fp.value) == len(query_phash):
                d = _hamming_distance(query_phash, fp.value)
                phash_distance[fp.media_id] = min(d, phash_distance.get(fp.media_id, d))

    cosine: dict[str, float] = {}
    query_vec = None
    if query_emb := query_fps.get(EMBEDDING_ALGORITHM):
        try:
            query_vec = _unit(decode_embedding(query_emb))
        except Exception as exc:
            logger.warning("Query embedding could not be decoded: %s", exc)
    if query_vec is not None:
        ids, rows = [], []
        for fp in stored(EMBEDDING_ALGORITHM):
            if embedding_version and fp.algorithm_version != embedding_version:
                continue
            try:
                vec = _unit(decode_embedding(fp.value))
            except Exception:
                continue
            if vec is not None and vec.shape == query_vec.shape:
                ids.append(fp.media_id)
                rows.append(vec)
        if rows:
            scores = np.stack(rows) @ query_vec
            for media_id, score in zip(ids, scores.tolist()):
                cosine[media_id] = max(score, cosine.get(media_id, -1.0))

    candidates = []
    for media_id in exact | set(phash_distance) | set(cosine):
        d = phash_distance.get(media_id)
        c = cosine.get(media_id)
        retrieved_by = []
        if media_id in exact:
            retrieved_by.append("sha256")
        if d is not None and d <= th["candidate_phash_distance"]:
            retrieved_by.append("phash")
        if c is not None and c >= th["candidate_cosine"]:
            retrieved_by.append(EMBEDDING_ALGORITHM)
        if not retrieved_by:
            continue
        candidates.append({
            "media_id": media_id,
            "sha256_exact": media_id in exact,
            "phash_distance": d,
            "cosine": None if c is None else round(float(c), 6),
            "retrieved_by": retrieved_by,
        })

    candidates.sort(key=lambda c: (
        not c["sha256_exact"],
        -(c["cosine"] if c["cosine"] is not None else -2.0),
        c["phash_distance"] if c["phash_distance"] is not None else 65,
    ))
    return candidates[:top_k]


# ── Verification ──────────────────────────────────────────────────────────────

def _load_gray(path: Path) -> np.ndarray:
    """Greyscale uint8, long side capped. PIL, not cv2.imread: it reads non-ASCII paths."""
    from PIL import Image, ImageOps

    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("L")
        img.thumbnail((ORB_MAX_SIDE, ORB_MAX_SIDE), Image.Resampling.LANCZOS)
        return np.asarray(img, dtype=np.uint8)


def orb_geometric_check(path_a: Path | str, path_b: Path | str) -> dict[str, Any]:
    """
    ORB keypoints matched with the ratio test, then a RANSAC homography.

    Returns keypoint counts, ``good_matches`` (after the ratio test),
    ``inliers`` (consistent with the homography) and ``inlier_ratio``.
    """
    import cv2

    gray_a, gray_b = _load_gray(Path(path_a)), _load_gray(Path(path_b))
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    kp_a, des_a = orb.detectAndCompute(gray_a, None)
    kp_b, des_b = orb.detectAndCompute(gray_b, None)
    out: dict[str, Any] = {
        "keypoints_query": len(kp_a),
        "keypoints_candidate": len(kp_b),
        "good_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
    }
    if des_a is None or des_b is None or len(kp_a) < 2 or len(kp_b) < 2:
        return out

    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des_a, des_b, k=2)
    good = [
        p[0] for p in pairs
        if len(p) == 2 and p[0].distance < ORB_RATIO * p[1].distance
    ]
    out["good_matches"] = len(good)
    if len(good) < 4:  # a homography needs four correspondences
        return out

    src = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, ORB_RANSAC_REPROJ_PX)
    inliers = int(mask.sum()) if mask is not None else 0
    out["inliers"] = inliers
    out["inlier_ratio"] = round(inliers / len(good), 4)
    return out


def verify_candidate(
    query_path: Path | str,
    candidate: dict[str, Any],
    candidate_path: Optional[Path | str],
) -> tuple[str, dict[str, Any]]:
    """
    ``(status, verification)`` for one ``search_media_fingerprints`` candidate.

    ``verification`` records every measurement, the thresholds it was held to,
    which gates passed (``None`` = could not run) and why the status is not
    VERIFIED_MATCH.
    """
    th = _thresholds()
    verification: dict[str, Any] = {
        "phash_distance": candidate.get("phash_distance"),
        "cosine": candidate.get("cosine"),
        "retrieved_by": candidate.get("retrieved_by", []),
        "thresholds": {
            "phash_max_distance": th["phash_max_distance"],
            "cosine_min": th["cosine_min"],
            "min_inliers": th["min_inliers"],
            "min_inlier_ratio": th["min_inlier_ratio"],
        },
        "gates": {},
        "reasons": [],
    }

    if candidate.get("sha256_exact"):
        verification["method"] = "sha256"
        verification["gates"] = {"sha256": True}
        return VERIFIED_MATCH, verification
    verification["method"] = "phash+embedding+orb"
    reasons = verification["reasons"]

    d = candidate.get("phash_distance")
    if d is None:
        phash_ok = None
        reasons.append("no comparable pHash for one of the images")
    else:
        phash_ok = d <= th["phash_max_distance"]
        if not phash_ok:
            reasons.append(f"pHash distance {d} > {th['phash_max_distance']}")

    c = candidate.get("cosine")
    if c is None:
        cosine_ok = None
        reasons.append("no comparable image embedding for one of the images")
    else:
        cosine_ok = c >= th["cosine_min"]
        if not cosine_ok:
            reasons.append(f"embedding cosine {c:.4f} < {th['cosine_min']}")

    geometry_ok = None
    if not candidate_path or not Path(candidate_path).is_file():
        reasons.append("candidate file is missing from storage")
    elif not Path(query_path).is_file():
        reasons.append("query file is missing from storage")
    else:
        try:
            orb = orb_geometric_check(query_path, candidate_path)
            verification["orb"] = orb
            geometry_ok = (
                orb["inliers"] >= th["min_inliers"]
                and orb["inlier_ratio"] >= th["min_inlier_ratio"]
            )
            if not geometry_ok:
                reasons.append(
                    f"ORB inliers {orb['inliers']} (ratio {orb['inlier_ratio']}) below "
                    f"{th['min_inliers']} / {th['min_inlier_ratio']}"
                )
        except Exception as exc:
            logger.warning("ORB verification failed: %s", exc)
            reasons.append(f"geometric check failed: {type(exc).__name__}")

    gates = {"phash": phash_ok, "cosine": cosine_ok, "geometry": geometry_ok}
    verification["gates"] = gates
    if all(g is True for g in gates.values()):
        return VERIFIED_MATCH, verification
    if any(g is not False for g in gates.values()):
        return SIMILAR_UNVERIFIED, verification
    return NO_MATCH, verification


class SimilarityMatch:
    """A ranked similarity match result."""

    def __init__(
        self,
        match_type: str,          # "local" | "external"
        source_media_id: str | None,
        external_url: str | None,
        similarity: float,
        method: str,              # "exact" | "perceptual" | "embedding" | "internet"
        algorithm: str | None = None,
        rank: int = 1,
        verification_status: str | None = None,
        verification: dict | None = None,
    ):
        self.match_type = match_type
        self.source_media_id = source_media_id
        self.external_url = external_url
        self.similarity = round(float(similarity), 4)
        self.method = method
        self.algorithm = algorithm
        self.rank = rank
        self.verification_status = verification_status
        self.verification = verification

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_type": self.match_type,
            "source_media_id": self.source_media_id,
            "external_url": self.external_url,
            "similarity": self.similarity,
            "method": self.method,
            "algorithm": self.algorithm,
            "rank": self.rank,
            "verification_status": self.verification_status,
            "verification": self.verification,
        }


class SimilarityEngine:
    """
    Coordinates multi-method similarity search.

    Local search is always attempted first using stored fingerprints.
    External search is attempted only if a configured provider is present.
    """

    def __init__(
        self,
        search_provider: BaseSourceSearchProvider | None = None,
    ):
        self.search_provider = search_provider or DisabledSourceSearchProvider()

    def search_local(
        self,
        file_path: str,
        query_fingerprints: dict[str, str],
        db: Session,
        exclude_media_id: str | None = None,
        owner: str | None = None,
        embedding_version: str | None = None,
        top_k: int = 10,
    ) -> tuple[list[SimilarityMatch], int]:
        """
        Retrieved and verified local matches, plus how many candidates
        verification rejected (NO_MATCH). Rejected candidates are not returned:
        they were near in one measure and contradicted by the others.
        """
        from app.db.repository import get_media

        candidates = search_media_fingerprints(
            query_fingerprints, db, owner, exclude_media_id,
            top_k=top_k, embedding_version=embedding_version,
        )
        matches: list[SimilarityMatch] = []
        rejected = 0
        for cand in candidates:
            media = get_media(db, cand["media_id"])
            status, verification = verify_candidate(
                file_path, cand, media.file_path if media else None
            )
            if status == NO_MATCH:
                rejected += 1
                continue
            if cand["sha256_exact"]:
                similarity, method, algorithm = 1.0, "exact", "sha256"
            elif cand["cosine"] is not None:
                similarity, method, algorithm = max(cand["cosine"], 0.0), "embedding", EMBEDDING_ALGORITHM
            else:
                similarity = 1.0 - cand["phash_distance"] / 64.0
                method, algorithm = "perceptual", "phash"
            matches.append(SimilarityMatch(
                match_type="local",
                source_media_id=cand["media_id"],
                external_url=None,
                similarity=similarity,
                method=method,
                algorithm=algorithm,
                rank=len(matches) + 1,
                verification_status=status,
                verification=verification,
            ))
        return matches, rejected

    def search_external(self, file_path: str) -> tuple[list[SimilarityMatch], str]:
        """
        Search external sources using the configured provider.

        Returns:
            (matches, provider_status)
        """
        if self.search_provider.status == ProviderStatus.UNAVAILABLE:
            return [], ProviderStatus.UNAVAILABLE.value

        try:
            raw_results: list[SourceSearchResult] = self.search_provider.search_by_image(file_path)
        except Exception as exc:
            logger.warning("External search provider failed: %s", exc)
            return [], ProviderStatus.FAILED.value

        matches = []
        for i, r in enumerate(raw_results):
            matches.append(SimilarityMatch(
                match_type="external",
                source_media_id=None,
                external_url=r.url,
                similarity=float(r.similarity),
                method="internet",
                algorithm=r.provider,
                rank=i + 1,
            ))

        return matches, ProviderStatus.CONFIGURED.value

    def search(
        self,
        file_path: str,
        query_fingerprints: dict[str, str] | None = None,
        db: Session | None = None,
        exclude_media_id: str | None = None,
        owner: str | None = None,
        embedding_version: str | None = None,
    ) -> dict[str, Any]:
        """
        Run full similarity search: local first, then external.

        Returns a dict with:
          local_matches: list of local SimilarityMatch dicts, each verified
          rejected_candidates: retrieved candidates verification ruled out
          external_matches: list of external SimilarityMatch dicts
          external_provider_status: UNAVAILABLE | CONFIGURED | FAILED
          total_matches: int
        """
        local_matches: list[SimilarityMatch] = []
        rejected = 0
        if db is not None and query_fingerprints:
            local_matches, rejected = self.search_local(
                file_path, query_fingerprints, db, exclude_media_id, owner,
                embedding_version=embedding_version,
            )

        external_matches, provider_status = self.search_external(file_path)

        all_matches = local_matches + external_matches
        return {
            "local_matches": [m.to_dict() for m in local_matches],
            "rejected_candidates": rejected,
            "external_matches": [m.to_dict() for m in external_matches],
            "external_provider_status": provider_status,
            "total_matches": len(all_matches),
        }
