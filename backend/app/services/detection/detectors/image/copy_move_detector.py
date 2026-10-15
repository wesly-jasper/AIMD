"""
Copy-Move Forgery Detector using SIFT/ORB Feature Matching.

Detects cloned/copied regions within an image by finding matching
keypoint descriptors between different image regions.
"""
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np

from app.schemas.detection import DetectionResult, DetectionRegion, Assessment
from app.services.detection.detectors.base_detector import BaseDetector

logger = logging.getLogger(__name__)


class CopyMoveDetector(BaseDetector):
    """
    Copy-Move Forgery Detection using feature matching.

    Algorithm:
    1. Detect keypoints using SIFT/ORB
    2. Match descriptors between all keypoint pairs
    3. Filter matches by spatial distance (remove nearby matches)
    4. Cluster matches to identify copied regions
    5. Use RANSAC to verify geometric consistency
    """

    def __init__(
        self,
        detector_type: str = "SIFT",  # "SIFT" or "ORB"
        min_match_distance: float = 30.0,  # Minimum pixel distance between matched keypoints
        ransac_threshold: float = 5.0,
        min_inliers: int = 8,
        match_ratio_threshold: float = 0.75,  # Lowe's ratio test
    ):
        self.detector_type = detector_type
        self.min_match_distance = min_match_distance
        self.ransac_threshold = ransac_threshold
        self.min_inliers = min_inliers
        self.match_ratio_threshold = match_ratio_threshold

        if detector_type == "SIFT":
            self.detector = cv2.SIFT_create(nfeatures=5000)
        elif detector_type == "ORB":
            self.detector = cv2.ORB_create(nfeatures=5000)
        else:
            raise ValueError(f"Unsupported detector type: {detector_type}")

    def detect(self, file_path: str) -> DetectionResult:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"Unable to read image at {file_path}")

            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect keypoints and descriptors
            keypoints, descriptors = self.detector.detectAndCompute(gray, None)

            if descriptors is None or len(keypoints) < 10:
                return self._inconclusive("Insufficient keypoints detected")

            # Match descriptors against themselves (find duplicated regions)
            matches = self._match_descriptors(descriptors, keypoints)

            if len(matches) < self.min_inliers:
                return self._inconclusive("Insufficient reliable matches for copy-move detection")

            # Cluster matches to find copied regions
            regions = self._cluster_matches(matches, keypoints, (w, h))

            if not regions:
                return self._inconclusive("No copy-move regions identified after clustering")

            # Compute overall confidence
            max_conf = max(r.confidence for r in regions)
            manipulation_detected = max_conf >= 0.50

            return DetectionResult(
                detector="CopyMoveDetector",
                media_type="image",
                manipulation_detected=manipulation_detected,
                confidence=round(max_conf, 4),
                manipulation_type="copy_move",
                assessment=Assessment.SUSPICIOUS if manipulation_detected else Assessment.CLEAN,
                regions=regions,
                metadata={
                    "status": "active_copy_move_detection",
                    "detector_type": self.detector_type,
                    "keypoints_detected": len(keypoints),
                    "raw_matches": len(matches),
                    "regions_found": len(regions),
                    "note": "OBSERVATION: Copy-move detection via feature matching. High confidence indicates cloned regions.",
                },
            )

        except Exception as e:
            logger.warning(f"CopyMoveDetector failed: {e}")
            return self._inconclusive(str(e))

    def _match_descriptors(
        self, descriptors: np.ndarray, keypoints: List[cv2.KeyPoint]
    ) -> List[Tuple[cv2.KeyPoint, cv2.KeyPoint, float]]:
        """Match descriptors using FLANN/BFMatcher with ratio test."""
        if self.detector_type == "SIFT":
            # FLANN for SIFT
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            # BFMatcher for ORB
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Match against itself
        raw_matches = matcher.knnMatch(descriptors, descriptors, k=2)

        good_matches = []
        for m, n in raw_matches:
            # Lowe's ratio test
            if m.distance < self.match_ratio_threshold * n.distance:
                # Exclude self-matches (same keypoint)
                if m.queryIdx != m.trainIdx:
                    kp1 = keypoints[m.queryIdx]
                    kp2 = keypoints[m.trainIdx]
                    # Check spatial distance
                    dist = np.sqrt(
                        (kp1.pt[0] - kp2.pt[0]) ** 2 + (kp1.pt[1] - kp2.pt[1]) ** 2
                    )
                    if dist >= self.min_match_distance:
                        good_matches.append((kp1, kp2, m.distance))

        return good_matches

    def _cluster_matches(
        self,
        matches: List[Tuple[cv2.KeyPoint, cv2.KeyPoint, float]],
        keypoints: List[cv2.KeyPoint],
        img_size: Tuple[int, int],
    ) -> List[DetectionRegion]:
        """Cluster matches into copy-move regions using RANSAC verification."""
        if len(matches) < self.min_inliers:
            return []

        w, h = img_size
        src_pts = np.float32([m[0].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([m[1].pt for m in matches]).reshape(-1, 1, 2)

        # Use RANSAC to find fundamental matrix and inliers
        try:
            F, mask = cv2.findFundamentalMat(
                src_pts, dst_pts, cv2.FM_RANSAC, self.ransac_threshold, 0.99
            )
        except Exception:
            return []

        if mask is None:
            return []

        inlier_matches = [matches[i] for i in range(len(matches)) if mask[i]]

        if len(inlier_matches) < self.min_inliers:
            return []

        # Group inliers into clusters using DBSCAN-like approach on keypoint positions
        regions = self._group_into_regions(inlier_matches, keypoints, (w, h))

        return regions

    def _group_into_regions(
        self,
        matches: List[Tuple[cv2.KeyPoint, cv2.KeyPoint, float]],
        keypoints: List[cv2.KeyPoint],
        img_size: Tuple[int, int],
    ) -> List[DetectionRegion]:
        """Group matched keypoints into distinct copy-move regions."""
        w, h = img_size
        regions = []

        if not matches:
            return regions

        # Collect all matched keypoint positions
        src_positions = np.array([m[0].pt for m in matches])
        dst_positions = np.array([m[1].pt for m in matches])
        all_positions = np.vstack([src_positions, dst_positions])

        # Simple clustering: use connected components on a grid
        grid_size = max(32, min(w, h) // 16)
        grid_w = (w + grid_size - 1) // grid_size
        grid_h = (h + grid_size - 1) // grid_size

        # Mark grid cells that have matched keypoints
        grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
        for pos in all_positions:
            gx = min(int(pos[0] // grid_size), grid_w - 1)
            gy = min(int(pos[1] // grid_size), grid_h - 1)
            grid[gy, gx] = 1

        # Find connected components
        num_labels, labels = cv2.connectedComponents(grid)

        for label_id in range(1, num_labels):
            component_mask = (labels == label_id).astype(np.uint8)
            if np.sum(component_mask) < 4:  # Too small
                continue

            # Get bounding box of this component
            ys, xs = np.where(component_mask)
            min_x, max_x = xs.min(), xs.max()
            min_y, max_y = ys.min(), ys.max()

            # Convert back to image coordinates
            x1 = min_x * grid_size
            y1 = min_y * grid_size
            x2 = min((max_x + 1) * grid_size, w)
            y2 = min((max_y + 1) * grid_size, h)

            region_w = x2 - x1
            region_h = y2 - y1

            if region_w < 16 or region_h < 16:
                continue

            # Count matches in this region
            region_matches = 0
            for m in matches:
                sx, sy = m[0].pt
                dx, dy = m[1].pt
                in_region = (
                    (x1 <= sx < x2 and y1 <= sy < y2) or
                    (x1 <= dx < x2 and y1 <= dy < y2)
                )
                if in_region:
                    region_matches += 1

            # Confidence based on number of inlier matches in region
            confidence = min(0.95, 0.3 + 0.6 * (region_matches / max(1, len(matches))))

            regions.append(DetectionRegion(
                type="copy_move_region",
                confidence=round(confidence, 3),
                bbox=[float(x1), float(y1), float(region_w), float(region_h)],
                extra={
                    "match_count": region_matches,
                    "grid_size": grid_size,
                }
            ))

        # Sort by confidence
        regions.sort(key=lambda r: r.confidence, reverse=True)
        return regions[:10]  # Top 10 regions

    def _inconclusive(self, reason: str) -> DetectionResult:
        return DetectionResult(
            detector="CopyMoveDetector",
            media_type="image",
            manipulation_detected=False,
            confidence=0.0,
            manipulation_type="copy_move",
            assessment=Assessment.INCONCLUSIVE,
            regions=[],
            metadata={
                "assessment": "INCONCLUSIVE",
                "reason": reason,
                "note": f"UNCERTAINTY: Copy-move detection unavailable - {reason}",
            },
        )