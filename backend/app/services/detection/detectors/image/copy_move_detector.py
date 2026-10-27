"""
Copy-Move Forgery Detector using SIFT/ORB Feature Matching.

Detects cloned/copied regions within an image by finding matching
keypoint descriptors between different image regions.

Version 2 fixes why version 1 never fired (0 of 20 copy-moves in the
reliability evaluation). It matched every descriptor against the same set with
k=2. The nearest neighbour of a keypoint is itself, at distance 0, and that pair
was then discarded, so the ratio test almost never compared two real
candidates. It also verified matches with a fundamental matrix, which models
two camera views, not a region pasted elsewhere in one image.
"""
import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from app.schemas.detection import (
    DetectionRegion,
    DetectionResult,
    DetectorAvailability,
    DetectorExecutionStatus,
    ScoreSemantics,
)
from app.services.detection.detectors.base_detector import BaseDetector

logger = logging.getLogger(__name__)

#: A matched pair: (point in the source, point in the copy, descriptor distance).
Match = Tuple[Tuple[float, float], Tuple[float, float], float]


class CopyMoveDetector(BaseDetector):
    """
    Copy-Move Forgery Detection using feature matching.

    Algorithm:
    1. Detect keypoints (SIFT by default) on the image, downscaled to at most
       ``max_side`` pixels so a 24-megapixel photo costs what a 1600 px one does
    2. Match every descriptor to its nearest *other* keypoints; keep a pair when
       the best is clearly better than the second (Lowe's ratio, g2NN)
    3. Drop pairs closer than ``min_match_distance`` (texture, not a copy)
    4. Orient pairs consistently and fit a similarity transform with RANSAC;
       a pasted copy is a shift, possibly rotated and scaled
    5. Repeat on the remaining pairs for a second and third copy

    Evidence is the number of RANSAC inliers of the strongest transform. The
    default threshold (50) was set on the validation split of the evaluation
    set: its five copy-moves had 60 to 413 inliers, its 72 genuine photographs
    at most 38. Repetitive structure (windows, tiles, fences) is the known
    false-positive mode.
    """

    def __init__(
        self,
        detector_type: str = "SIFT",  # "SIFT" or "ORB"
        min_match_distance: float = 30.0,  # Minimum pixel distance between matched keypoints
        ransac_threshold: float = 4.0,
        min_inliers: int = 50,
        match_ratio_threshold: float = 0.6,  # Lowe's ratio test, on the 2nd/3rd neighbours
        max_side: int = 1600,
        min_keypoints: int = 100,
        max_transforms: int = 3,
    ):
        self.detector_type = detector_type
        self.min_match_distance = min_match_distance
        self.ransac_threshold = ransac_threshold
        self.min_inliers = min_inliers
        self.match_ratio_threshold = match_ratio_threshold
        self.max_side = max_side
        self.min_keypoints = min_keypoints
        self.max_transforms = max_transforms

        if detector_type == "SIFT":
            self.detector = cv2.SIFT_create(nfeatures=5000)
        elif detector_type == "ORB":
            self.detector = cv2.ORB_create(nfeatures=5000)
        else:
            raise ValueError(f"Unsupported detector type: {detector_type}")

    detector_id = "CopyMoveDetector"
    version = "2.0.0"
    supported_media_types = frozenset({"image"})
    manipulation_type = "copy_move"

    def detect(self, file_path: str) -> DetectionResult:
        path = Path(file_path)
        if not path.exists():
            # A missing evidence file is a caller error, not a detector outcome,
            # and raising keeps it from being recorded as a forensic result.
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"Unable to read image at {file_path}")

            h, w = img.shape[:2]
            scale = min(1.0, self.max_side / max(h, w))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if scale < 1.0:
                gray = cv2.resize(gray, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)

            keypoints, descriptors = self.detector.detectAndCompute(gray, None)
            if descriptors is None or len(keypoints) < self.min_keypoints:
                # A flat or tiny image: too little structure to have looked.
                return self.inconclusive(
                    f"Too few keypoints to search for copied regions ({len(keypoints or [])})"
                )

            matches = self._match_descriptors(descriptors, keypoints)
            transforms = self._transforms(matches)
            best = transforms[0]["inliers"] if transforms else 0
            metadata = {
                "detector_type": self.detector_type,
                "keypoints_detected": len(keypoints),
                "candidate_pairs": len(matches),
                "inliers": best,
                "min_inliers": self.min_inliers,
                "analysis_scale": round(scale, 4),
                "transforms": [
                    {k: v for k, v in t.items() if k not in ("src", "dst")} for t in transforms
                ],
            }

            flagged = [t for t in transforms if t["inliers"] >= self.min_inliers]
            regions: List[DetectionRegion] = []
            for n, t in enumerate(flagged):
                confidence = round(t["inliers"] / (t["inliers"] + self.min_inliers), 4)
                for role in ("src", "dst"):
                    regions.append(DetectionRegion(
                        type="copy_move_source" if role == "src" else "copy_move_copy",
                        confidence=confidence,
                        bbox=self._bbox(t[role], scale, (w, h)),
                        extra={"transform": n, "inliers": t["inliers"],
                               "shift_px": round(t["shift"] / scale, 1)},
                    ))

            # Ordinal: 0.5 at the threshold, rising with the inlier count.
            confidence = round(best / (best + self.min_inliers), 4) if best else 0.0
            return DetectionResult(
                detector=self.detector_id,
                detector_version=self.version,
                media_type="image",
                manipulation_detected=bool(flagged),
                confidence=confidence,
                manipulation_type=self.manipulation_type,
                availability=DetectorAvailability.AVAILABLE,
                execution_status=DetectorExecutionStatus.COMPLETED,
                # An inlier count against a threshold set on validation data,
                # not a trained model. Ordinal severity, not a probability.
                score_semantics=ScoreSemantics.HEURISTIC_SEVERITY,
                regions=regions,
                metadata=metadata,
            )

        except Exception as e:
            logger.warning(f"CopyMoveDetector failed: {e}")
            # AVAILABLE + FAILED: SIFT/ORB are present and the run broke.
            return self.failed_from("Copy-move analysis failed", e)

    def _match_descriptors(self, descriptors: np.ndarray, keypoints: List[cv2.KeyPoint]) -> List[Match]:
        """
        Pairs of distinct keypoints with near-identical descriptors, each pair once.

        k=4 so that, after the keypoint's match with itself is removed, the two
        nearest *other* keypoints remain for the ratio test.
        """
        if self.detector_type == "SIFT":
            matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        raw = matcher.knnMatch(descriptors, descriptors, k=min(4, len(keypoints)))

        seen: set[tuple[int, int]] = set()
        good: List[Match] = []
        for candidates in raw:
            others = [m for m in candidates if m.trainIdx != m.queryIdx]
            if len(others) < 2:
                continue
            best, second = others[0], others[1]
            if best.distance >= self.match_ratio_threshold * max(second.distance, 1e-6):
                continue
            pair = tuple(sorted((best.queryIdx, best.trainIdx)))
            if pair in seen:
                continue
            seen.add(pair)
            p, q = keypoints[pair[0]].pt, keypoints[pair[1]].pt
            if np.hypot(p[0] - q[0], p[1] - q[1]) < self.min_match_distance:
                continue
            # One orientation for every pair (the copy is to the right, or
            # below): a pasted region then gives one transform, not +T and -T.
            if (q[0] - p[0], q[1] - p[1]) < (0.0, 0.0):
                p, q = q, p
            good.append((p, q, float(best.distance)))
        return good

    def _transforms(self, matches: List[Match]) -> List[dict]:
        """Similarity transforms supported by the pairs, strongest first."""
        remaining = list(matches)
        found: List[dict] = []
        for _ in range(self.max_transforms):
            if len(remaining) < 4:
                break
            src = np.float32([m[0] for m in remaining])
            dst = np.float32([m[1] for m in remaining])
            M, mask = cv2.estimateAffinePartial2D(
                src, dst, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_threshold,
                maxIters=4000, confidence=0.995,
            )
            if M is None or mask is None:
                break
            inl = mask.ravel().astype(bool)
            shift = float(np.hypot(M[0, 2], M[1, 2]))
            # A transform near the identity, or one supported by points along a
            # line, is texture or noise rather than a pasted region.
            spread = float(np.ptp(src[inl], axis=0).min()) if inl.sum() >= 3 else 0.0
            if inl.sum() < 4 or shift < self.min_match_distance or spread < 8:
                break
            found.append({
                "inliers": int(inl.sum()),
                "shift": shift,
                "rotation_deg": round(float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))), 1),
                "scale": round(float(np.hypot(M[0, 0], M[1, 0])), 3),
                "src": src[inl], "dst": dst[inl],
            })
            remaining = [m for m, keep in zip(remaining, inl) if not keep]
        found.sort(key=lambda t: t["inliers"], reverse=True)
        return found

    @staticmethod
    def _bbox(points: np.ndarray, scale: float, size: Tuple[int, int]) -> List[float]:
        """Box around ``points`` (analysis scale) in original-image pixels, padded 8 px."""
        w, h = size
        x0, y0 = points.min(axis=0) / scale - 8
        x1, y1 = points.max(axis=0) / scale + 8
        x0, y0 = max(0.0, float(x0)), max(0.0, float(y0))
        x1, y1 = min(float(w), float(x1)), min(float(h), float(y1))
        return [round(x0, 1), round(y0, 1), round(x1 - x0, 1), round(y1 - y0, 1)]

    # `_inconclusive` and `_unavailable` were removed here: both are now on
    # BaseDetector, which every detector shares.
