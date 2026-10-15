"""
Video Spatial-Temporal Tracker for Manipulation Region Tracking.

Tracks suspicious regions across video frames using optical flow,
feature matching, and temporal consistency analysis.
"""
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import cv2
import numpy as np
from dataclasses import dataclass, field

from app.schemas.detection import DetectionRegion

logger = logging.getLogger(__name__)


@dataclass
class TrackedRegion:
    """A region tracked across multiple frames."""
    region_id: int
    initial_bbox: List[float]  # [x, y, w, h] in first frame
    initial_frame: int
    bboxes: Dict[int, List[float]] = field(default_factory=dict)  # frame_idx -> bbox
    confidences: Dict[int, float] = field(default_factory=dict)
    features: Dict[int, np.ndarray] = field(default_factory=dict)  # frame_idx -> feature vector
    active: bool = True
    max_iou: float = 0.0
    temporal_consistency: float = 0.0

    def get_latest_bbox(self) -> Optional[List[float]]:
        if not self.bboxes:
            return None
        latest_frame = max(self.bboxes.keys())
        return self.bboxes[latest_frame]

    def get_latest_confidence(self) -> float:
        if not self.confidences:
            return 0.0
        latest_frame = max(self.confidences.keys())
        return self.confidences[latest_frame]


class VideoSpatialTemporalTracker:
    """
    Tracks manipulation regions across video frames.

    Features:
    - Optical flow tracking (Farneback / Lucas-Kanade)
    - Feature-based matching (ORB/SIFT) for re-detection
    - Temporal consistency scoring
    - Region merging/splitting detection
    - Motion trajectory analysis
    """

    def __init__(
        self,
        max_regions: int = 50,
        iou_threshold: float = 0.3,
        feature_match_threshold: float = 0.75,
        max_lost_frames: int = 5,
        optical_flow_method: str = "farneback",  # "farneback" or "lk"
        min_region_area: int = 100,
        consistency_window: int = 5,
    ):
        self.max_regions = max_regions
        self.iou_threshold = iou_threshold
        self.feature_match_threshold = feature_match_threshold
        self.max_lost_frames = max_lost_frames
        self.optical_flow_method = optical_flow_method
        self.min_region_area = min_region_area
        self.consistency_window = consistency_window

        self.tracked_regions: List[TrackedRegion] = []
        self.next_region_id = 0
        self.prev_frame: Optional[np.ndarray] = None
        self.prev_gray: Optional[np.ndarray] = None
        self.frame_idx = 0

        # Optical flow parameters
        self.farneback_params = dict(
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        self.lk_params = dict(
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

    def process_frame(
        self,
        frame: np.ndarray,
        detections: List[DetectionRegion],
        frame_idx: int,
    ) -> List[TrackedRegion]:
        """
        Process a new frame with detections, update tracking.

        Args:
            frame: Current frame (BGR)
            detections: List of DetectionRegion from current frame
            frame_idx: Frame index

        Returns:
            Updated list of tracked regions
        """
        self.frame_idx = frame_idx
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Predict positions using optical flow
        if self.prev_gray is not None:
            self._predict_with_optical_flow(gray)

        # Match detections to tracked regions
        self._match_detections(detections, frame, gray)

        # Update temporal consistency
        self._update_temporal_consistency()

        # Remove lost tracks
        self._prune_lost_tracks()

        # Create new tracks for unmatched detections
        self._create_new_tracks(detections, frame, gray)

        # Limit number of tracks
        if len(self.tracked_regions) > self.max_regions:
            self.tracked_regions.sort(key=lambda r: r.temporal_consistency, reverse=True)
            self.tracked_regions = self.tracked_regions[:self.max_regions]

        self.prev_frame = frame.copy()
        self.prev_gray = gray.copy()

        return [r for r in self.tracked_regions if r.active]

    def _predict_with_optical_flow(self, gray: np.ndarray):
        """Predict region positions using optical flow."""
        if not self.tracked_regions:
            return

        if self.optical_flow_method == "farneback":
            # Dense optical flow
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None, **self.farneback_params
            )

            for region in self.tracked_regions:
                if not region.active:
                    continue
                latest_bbox = region.get_latest_bbox()
                if latest_bbox is None:
                    continue

                x, y, w, h = latest_bbox
                x, y, w, h = int(x), int(y), int(w), int(h)

                # Get flow in region
                if y + h <= flow.shape[0] and x + w <= flow.shape[1]:
                    region_flow = flow[y:y+h, x:x+w]
                    mean_flow = np.mean(region_flow, axis=(0, 1))

                    # Predict new position
                    new_x = x + mean_flow[0]
                    new_y = y + mean_flow[1]
                    region.bboxes[self.frame_idx] = [float(new_x), float(new_y), float(w), float(h)]

        elif self.optical_flow_method == "lk":
            # Sparse optical flow on region centers
            for region in self.tracked_regions:
                if not region.active:
                    continue
                latest_bbox = region.get_latest_bbox()
                if latest_bbox is None:
                    continue

                x, y, w, h = latest_bbox
                center = np.array([[[x + w/2, y + h/2]]], dtype=np.float32)

                new_center, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, center, None, **self.lk_params
                )

                if status[0][0] == 1:
                    new_x, new_y = new_center[0][0]
                    region.bboxes[self.frame_idx] = [
                        float(new_x - w/2), float(new_y - h/2), float(w), float(h)
                    ]

    def _match_detections(
        self,
        detections: List[DetectionRegion],
        frame: np.ndarray,
        gray: np.ndarray,
    ):
        """Match current detections to tracked regions."""
        if not detections or not self.tracked_regions:
            return

        active_regions = [r for r in self.tracked_regions if r.active]
        if not active_regions:
            return

        # Compute IoU matrix
        iou_matrix = np.zeros((len(active_regions), len(detections)))
        for i, region in enumerate(active_regions):
            pred_bbox = region.get_latest_bbox()
            if pred_bbox is None:
                continue
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._compute_iou(pred_bbox, det.bbox)

        # Hungarian assignment (greedy for simplicity)
        matched_regions = set()
        matched_detections = set()

        # Sort by IoU descending
        pairs = []
        for i in range(len(active_regions)):
            for j in range(len(detections)):
                if iou_matrix[i, j] > self.iou_threshold:
                    pairs.append((iou_matrix[i, j], i, j))
        pairs.sort(reverse=True)

        for iou, i, j in pairs:
            if i in matched_regions or j in matched_detections:
                continue
            matched_regions.add(i)
            matched_detections.add(j)

            region = active_regions[i]
            det = detections[j]

            # Update region with detection
            region.bboxes[self.frame_idx] = det.bbox
            region.confidences[self.frame_idx] = det.confidence

            # Extract features for re-detection
            region.features[self.frame_idx] = self._extract_region_features(frame, det.bbox)

        # Mark unmatched regions as potentially lost
        for i, region in enumerate(active_regions):
            if i not in matched_regions:
                # Keep previous position but mark confidence decay
                if self.frame_idx - 1 in region.bboxes:
                    region.bboxes[self.frame_idx] = region.bboxes[self.frame_idx - 1]
                    region.confidences[self.frame_idx] = region.confidences.get(self.frame_idx - 1, 0) * 0.8

    def _extract_region_features(self, frame: np.ndarray, bbox: List[float]) -> np.ndarray:
        """Extract ORB features from region for re-detection."""
        x, y, w, h = [int(v) for v in bbox]
        x, y = max(0, x), max(0, y)
        w, h = min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)

        if w < 16 or h < 16:
            return np.array([])

        roi = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(nfeatures=100)
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        if descriptors is not None:
            return descriptors.flatten()[:500]  # Fixed size
        return np.array([])

    def _create_new_tracks(
        self,
        detections: List[DetectionRegion],
        frame: np.ndarray,
        gray: np.ndarray,
    ):
        """Create new tracks for unmatched high-confidence detections."""
        matched_detection_indices = set()

        # Find which detections were matched
        active_regions = [r for r in self.tracked_regions if r.active]
        for region in active_regions:
            if self.frame_idx in region.bboxes:
                # Check which detection this matches
                for j, det in enumerate(detections):
                    if self._compute_iou(region.bboxes[self.frame_idx], det.bbox) > self.iou_threshold:
                        matched_detection_indices.add(j)

        # Create new tracks for unmatched
        for j, det in enumerate(detections):
            if j in matched_detection_indices:
                continue
            if det.confidence < 0.4:
                continue
            if det.bbox[2] * det.bbox[3] < self.min_region_area:
                continue

            region = TrackedRegion(
                region_id=self.next_region_id,
                initial_bbox=det.bbox,
                initial_frame=self.frame_idx,
            )
            region.bboxes[self.frame_idx] = det.bbox
            region.confidences[self.frame_idx] = det.confidence
            region.features[self.frame_idx] = self._extract_region_features(frame, det.bbox)

            self.tracked_regions.append(region)
            self.next_region_id += 1

    def _update_temporal_consistency(self):
        """Compute temporal consistency score for each track."""
        for region in self.tracked_regions:
            if not region.active:
                continue

            frames = sorted(region.confidences.keys())
            if len(frames) < 2:
                region.temporal_consistency = 0.0
                continue

            # Consistency = average confidence over recent window
            recent_frames = frames[-self.consistency_window:]
            confs = [region.confidences[f] for f in recent_frames]
            region.temporal_consistency = np.mean(confs)

            # Also check position smoothness
            if len(frames) >= 3:
                positions = np.array([region.bboxes[f] for f in recent_frames])
                if len(positions) > 1:
                    diffs = np.diff(positions, axis=0)
                    smoothness = 1.0 / (1.0 + np.mean(np.abs(diffs)))
                    region.temporal_consistency = (region.temporal_consistency + smoothness) / 2.0

    def _prune_lost_tracks(self):
        """Remove tracks that have been lost for too long."""
        for region in self.tracked_regions:
            if not region.active:
                continue

            last_frame = max(region.bboxes.keys()) if region.bboxes else -1
            frames_lost = self.frame_idx - last_frame

            if frames_lost > self.max_lost_frames:
                region.active = False
            elif frames_lost > 0:
                # Decay confidence
                if last_frame in region.confidences:
                    decay = 0.85 ** frames_lost
                    region.confidences[self.frame_idx] = region.confidences[last_frame] * decay
                    region.bboxes[self.frame_idx] = region.bboxes[last_frame]

    def _compute_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Compute IoU between two bboxes [x, y, w, h]."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        inter = (xi2 - xi1) * (yi2 - yi1)
        union = w1 * h1 + w2 * h2 - inter
        return inter / union if union > 0 else 0.0

    def get_tracked_regions_for_frame(self, frame_idx: int) -> List[DetectionRegion]:
        """Get tracked regions as DetectionRegions for a specific frame."""
        regions = []
        for region in self.tracked_regions:
            if not region.active:
                continue
            if frame_idx not in region.bboxes:
                continue

            bbox = region.bboxes[frame_idx]
            conf = region.confidences.get(frame_idx, 0.0)

            regions.append(DetectionRegion(
                type="tracked_region",
                confidence=round(conf * region.temporal_consistency, 3),
                bbox=bbox,
                extra={
                    "track_id": region.region_id,
                    "initial_frame": region.initial_frame,
                    "frames_tracked": len(region.bboxes),
                    "temporal_consistency": round(region.temporal_consistency, 3),
                }
            ))

        return regions

    def reset(self):
        """Reset tracker state."""
        self.tracked_regions = []
        self.next_region_id = 0
        self.prev_frame = None
        self.prev_gray = None
        self.frame_idx = 0


class TemporalAnomalyDetector:
    """
    Detects temporal anomalies in tracked regions.
    - Sudden appearance/disappearance
    - Inconsistent motion patterns
    - Flickering confidence
    """

    def __init__(
        self,
        min_track_length: int = 5,
        flicker_threshold: float = 0.4,
        motion_smoothness_threshold: float = 0.3,
    ):
        self.min_track_length = min_track_length
        self.flicker_threshold = flicker_threshold
        self.motion_smoothness_threshold = motion_smoothness_threshold

    def analyze_tracks(self, tracked_regions: List[TrackedRegion]) -> List[Dict[str, Any]]:
        """Analyze tracks for temporal anomalies."""
        anomalies = []

        for region in tracked_regions:
            if not region.active:
                continue
            if len(region.bboxes) < self.min_track_length:
                continue

            frames = sorted(region.bboxes.keys())
            confidences = [region.confidences[f] for f in frames]
            bboxes = np.array([region.bboxes[f] for f in frames])

            # 1. Flickering detection (confidence oscillation)
            conf_diff = np.abs(np.diff(confidences))
            flicker_score = np.mean(conf_diff)
            if flicker_score > self.flicker_threshold:
                anomalies.append({
                    "track_id": region.region_id,
                    "type": "confidence_flicker",
                    "score": float(flicker_score),
                    "frames": frames,
                    "description": "Confidence oscillates significantly across frames",
                })

            # 2. Motion smoothness
            if len(bboxes) >= 3:
                velocities = np.diff(bboxes[:, :2], axis=0)  # x, y velocity
                accelerations = np.diff(velocities, axis=0)
                jerk = np.mean(np.abs(accelerations))
                if jerk > self.motion_smoothness_threshold:
                    anomalies.append({
                        "track_id": region.region_id,
                        "type": "motion_jerk",
                        "score": float(jerk),
                        "frames": frames,
                        "description": "Abrupt motion changes suggest tracking failure or manipulation",
                    })

            # 3. Sudden size changes
            areas = bboxes[:, 2] * bboxes[:, 3]
            area_ratios = areas[1:] / (areas[:-1] + 1e-10)
            max_area_change = np.max(np.abs(area_ratios - 1.0))
            if max_area_change > 0.5:
                anomalies.append({
                    "track_id": region.region_id,
                    "type": "sudden_size_change",
                    "score": float(max_area_change),
                    "frames": frames,
                    "description": "Region size changes abruptly",
                })

        return anomalies