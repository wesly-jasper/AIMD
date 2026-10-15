"""
Face Region Analyzer for Per-Facial-Region Deepfake Detection.

Analyzes specific facial regions (eyes, mouth, skin, boundaries) for
inconsistencies indicating deepfake manipulation.
"""
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
from PIL import Image

from app.schemas.detection import DetectionResult, DetectionRegion, Assessment
from app.services.detection.detectors.base_detector import BaseDetector

logger = logging.getLogger(__name__)


class FaceRegionAnalyzer(BaseDetector):
    """
    Per-facial-region deepfake analysis.

    Regions analyzed:
    - Eyes (blink consistency, gaze, reflections)
    - Mouth (lip sync, articulation, teeth)
    - Skin (texture, pores, frequency consistency)
    - Face boundaries (blending artifacts, edge consistency)
    - Forehead/cheeks (smoothness, unnatural symmetry)
    """

    # Facial landmark indices (MediaPipe 468 landmarks)
    FACE_REGIONS = {
        "left_eye": list(range(33, 42)) + list(range(133, 173)),
        "right_eye": list(range(263, 272)) + list(range(362, 402)),
        "mouth": list(range(61, 88)) + list(range(185, 212)),
        "nose": list(range(1, 11)) + list(range(19, 32)),
        "left_eyebrow": list(range(46, 56)) + list(range(276, 286)),
        "right_eyebrow": list(range(285, 295)) + list(range(334, 344)),
        "forehead": list(range(9, 20)) + list(range(67, 108)),
        "left_cheek": list(range(116, 143)),
        "right_cheek": list(range(345, 372)),
        "chin": list(range(152, 176)),
        "face_boundary": list(range(0, 27)) + list(range(172, 214)) + list(range(323, 361)) + list(range(446, 478)),
    }

    def __init__(
        self,
        use_mediapipe: bool = True,
        min_face_size: int = 64,
        region_weights: Dict[str, float] = None,
        frequency_analysis: bool = True,
        texture_analysis: bool = True,
        boundary_analysis: bool = True,
    ):
        self.use_mediapipe = use_mediapipe
        self.min_face_size = min_face_size
        self.region_weights = region_weights or {
            "left_eye": 1.5,
            "right_eye": 1.5,
            "mouth": 1.5,
            "nose": 0.8,
            "left_eyebrow": 1.0,
            "right_eyebrow": 1.0,
            "forehead": 1.0,
            "left_cheek": 1.2,
            "right_cheek": 1.2,
            "chin": 0.8,
            "face_boundary": 1.3,
        }
        self.frequency_analysis = frequency_analysis
        self.texture_analysis = texture_analysis
        self.boundary_analysis = boundary_analysis

        self._face_mesh = None
        if use_mediapipe:
            self._init_mediapipe()

    def _init_mediapipe(self):
        """Initialize MediaPipe Face Mesh."""
        try:
            import mediapipe as mp
            self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=5,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except ImportError:
            logger.warning("MediaPipe not available, falling back to OpenCV Haar")
            self.use_mediapipe = False

    def detect(self, file_path: str) -> DetectionResult:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"Unable to read image at {file_path}")

            h, w = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Detect faces and landmarks
            faces = self._detect_faces(rgb, (w, h))

            if not faces:
                return self._inconclusive("No faces detected")

            all_regions = []
            face_scores = []

            for face_idx, face_data in enumerate(faces):
                landmarks = face_data["landmarks"]
                face_bbox = face_data["bbox"]

                # Analyze each facial region
                face_regions = self._analyze_face_regions(img, landmarks, face_bbox, face_idx)
                all_regions.extend(face_regions)

                # Compute face-level score
                if face_regions:
                    weighted_score = sum(r.confidence * self.region_weights.get(r.extra.get("region_name", ""), 1.0)
                                         for r in face_regions)
                    total_weight = sum(self.region_weights.get(r.extra.get("region_name", ""), 1.0)
                                       for r in face_regions)
                    face_score = weighted_score / total_weight if total_weight > 0 else 0.0
                    face_scores.append(face_score)

            # Overall confidence
            max_conf = max(face_scores) if face_scores else 0.0
            manipulation_detected = max_conf >= 0.55

            return DetectionResult(
                detector="FaceRegionAnalyzer",
                media_type="image",
                manipulation_detected=manipulation_detected,
                confidence=round(max_conf, 4),
                manipulation_type="deepfake",
                assessment=Assessment.SUSPICIOUS if manipulation_detected else Assessment.CLEAN,
                regions=all_regions,
                metadata={
                    "status": "active_face_region_analysis",
                    "faces_detected": len(faces),
                    "regions_analyzed": len(all_regions),
                    "face_scores": [round(s, 3) for s in face_scores],
                    "region_weights": self.region_weights,
                    "note": "OBSERVATION: Per-facial-region analysis complete. High confidence in specific regions indicates localized manipulation.",
                },
            )

        except Exception as e:
            logger.warning(f"FaceRegionAnalyzer failed: {e}")
            return self._inconclusive(str(e))

    def _detect_faces(self, rgb: np.ndarray, img_size: Tuple[int, int]) -> List[Dict]:
        """Detect faces and extract landmarks."""
        w, h = img_size
        faces = []

        if self.use_mediapipe and self._mp_face_mesh:
            results = self._mp_face_mesh.process(rgb)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Convert landmarks to pixel coordinates
                    landmarks = []
                    for lm in face_landmarks.landmark:
                        x = int(lm.x * w)
                        y = int(lm.y * h)
                        landmarks.append((x, y))

                    # Compute face bbox
                    xs, ys = zip(*landmarks)
                    x1, x2 = max(0, min(xs)), min(w, max(xs))
                    y1, y2 = max(0, min(ys)), min(h, max(ys))

                    if (x2 - x1) >= self.min_face_size and (y2 - y1) >= self.min_face_size:
                        faces.append({
                            "landmarks": np.array(landmarks),
                            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        })
        else:
            # Fallback to Haar cascade
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            detected = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(self.min_face_size, self.min_face_size))
            for (x, y, w_f, h_f) in detected:
                faces.append({
                    "landmarks": None,
                    "bbox": [float(x), float(y), float(w_f), float(h_f)],
                })

        return faces

    def _analyze_face_regions(
        self,
        img: np.ndarray,
        landmarks: np.ndarray,
        face_bbox: List[float],
        face_idx: int,
    ) -> List[DetectionRegion]:
        """Analyze each facial region for manipulation."""
        regions = []
        x1, y1, w_f, h_f = face_bbox
        face_roi = img[int(y1):int(y1+h_f), int(x1):int(x1+w_f)]

        for region_name, landmark_indices in self.FACE_REGIONS.items():
            if landmarks is None:
                # Approximate region from face bbox
                region_bbox = self._approximate_region_bbox(region_name, face_bbox)
            else:
                region_bbox = self._get_region_bbox_from_landmarks(landmarks, landmark_indices, face_bbox)

            if region_bbox is None:
                continue

            rx, ry, rw, rh = region_bbox
            if rw < 8 or rh < 8:
                continue

            region_img = img[int(ry):int(ry+rh), int(rx):int(rx+rw)]
            if region_img.size == 0:
                continue

            # Analyze this region
            confidence = self._analyze_region(region_img, region_name, face_roi)

            if confidence >= 0.3:  # Only report suspicious regions
                regions.append(DetectionRegion(
                    type="face_region",
                    confidence=round(confidence, 3),
                    bbox=[float(rx), float(ry), float(rw), float(rh)],
                    extra={
                        "region_name": region_name,
                        "face_index": face_idx,
                        "analysis_types": self._get_analysis_types(region_name),
                    }
                ))

        return regions

    def _approximate_region_bbox(self, region_name: str, face_bbox: List[float]) -> Optional[List[float]]:
        """Approximate region bbox from face bbox (when landmarks unavailable)."""
        x1, y1, w_f, h_f = face_bbox

        # Relative positions (normalized to face bbox)
        approx = {
            "left_eye": (0.2, 0.25, 0.25, 0.15),
            "right_eye": (0.55, 0.25, 0.25, 0.15),
            "mouth": (0.3, 0.65, 0.4, 0.2),
            "nose": (0.35, 0.4, 0.3, 0.25),
            "left_eyebrow": (0.2, 0.15, 0.25, 0.1),
            "right_eyebrow": (0.55, 0.15, 0.25, 0.1),
            "forehead": (0.15, 0.05, 0.7, 0.2),
            "left_cheek": (0.05, 0.4, 0.3, 0.3),
            "right_cheek": (0.65, 0.4, 0.3, 0.3),
            "chin": (0.25, 0.8, 0.5, 0.2),
            "face_boundary": (0.0, 0.0, 1.0, 1.0),
        }

        if region_name not in approx:
            return None

        rx_rel, ry_rel, rw_rel, rh_rel = approx[region_name]
        return [
            x1 + rx_rel * w_f,
            y1 + ry_rel * h_f,
            rw_rel * w_f,
            rh_rel * h_f,
        ]

    def _get_region_bbox_from_landmarks(
        self, landmarks: np.ndarray, indices: List[int], face_bbox: List[float]
    ) -> Optional[List[float]]:
        """Get region bbox from specific landmark indices."""
        if len(indices) == 0:
            return None

        pts = landmarks[indices]
        xs, ys = pts[:, 0], pts[:, 1]

        x1, y1 = xs.min(), ys.min()
        x2, y2 = xs.max(), ys.max()

        # Add padding
        pad_x = (x2 - x1) * 0.15
        pad_y = (y2 - y1) * 0.15

        x1 = max(face_bbox[0], x1 - pad_x)
        y1 = max(face_bbox[1], y1 - pad_y)
        x2 = min(face_bbox[0] + face_bbox[2], x2 + pad_x)
        y2 = min(face_bbox[1] + face_bbox[3], y2 + pad_y)

        return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

    def _analyze_region(
        self, region_img: np.ndarray, region_name: str, face_roi: np.ndarray
    ) -> float:
        """Analyze a single facial region for manipulation signs."""
        scores = []

        if self.frequency_analysis:
            freq_score = self._analyze_frequency(region_img)
            scores.append(freq_score)

        if self.texture_analysis:
            tex_score = self._analyze_texture(region_img)
            scores.append(tex_score)

        if self.boundary_analysis and region_name == "face_boundary":
            bound_score = self._analyze_boundary(region_img, face_roi)
            scores.append(bound_score)

        if region_name in ["left_eye", "right_eye"]:
            eye_score = self._analyze_eye(region_img)
            scores.append(eye_score)

        if region_name == "mouth":
            mouth_score = self._analyze_mouth(region_img)
            scores.append(mouth_score)

        if not scores:
            return 0.0

        return np.mean(scores)

    def _analyze_frequency(self, region_img: np.ndarray) -> float:
        """Frequency domain analysis for region."""
        gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        if gray.shape[0] < 16 or gray.shape[1] < 16:
            return 0.0

        # 2D FFT
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.log(np.abs(fshift) + 1e-10)

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        # High frequency energy ratio
        y_grid, x_grid = np.ogrid[:h, :w]
        dist = np.sqrt((y_grid - cy)**2 + (x_grid - cx)**2)
        max_dist = np.sqrt(cy**2 + cx**2)

        hf_mask = dist > max_dist * 0.4
        hf_energy = np.mean(magnitude[hf_mask]) if np.any(hf_mask) else 0
        total_energy = np.mean(magnitude)

        # Unnatural high-frequency (synthetic) or low-frequency (over-smoothed)
        ratio = hf_energy / (total_energy + 1e-10)
        # Normalize: both too high and too low are suspicious
        anomaly = 2.0 * abs(ratio - 0.5)
        return min(1.0, anomaly)

    def _analyze_texture(self, region_img: np.ndarray) -> float:
        """Texture analysis using Local Binary Patterns and GLCM."""
        gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
        if gray.shape[0] < 16 or gray.shape[1] < 16:
            return 0.0

        # LBP
        lbp = self._compute_lbp(gray)
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256), density=True)
        # Uniform LBP patterns (natural texture has specific distribution)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        # Low entropy = over-smoothed (synthetic)
        texture_score = 1.0 - min(1.0, entropy / 8.0)  # Max entropy ~8 for 256 bins

        # Local variance
        var = cv2.boxFilter(gray.astype(np.float32)**2, -1, (8, 8)) - \
              cv2.boxFilter(gray.astype(np.float32), -1, (8, 8))**2
        mean_var = np.mean(var)
        var_score = min(1.0, 50.0 / (mean_var + 1e-10))  # Low variance = suspicious

        return (texture_score + var_score) / 2.0

    def _compute_lbp(self, gray: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
        """Compute Local Binary Pattern."""
        h, w = gray.shape
        lbp = np.zeros((h, w), dtype=np.uint8)

        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = gray[i, j]
                code = 0
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    r = int(i + radius * np.sin(angle))
                    c = int(j + radius * np.cos(angle))
                    if gray[r, c] >= center:
                        code |= (1 << k)
                lbp[i, j] = code

        return lbp

    def _analyze_boundary(self, region_img: np.ndarray, face_roi: np.ndarray) -> float:
        """Analyze face boundary for blending artifacts."""
        gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        if gray.shape[0] < 16 or gray.shape[1] < 16:
            return 0.0

        # Edge detection
        edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)

        # Check for unnatural edge patterns (too smooth or too sharp)
        edge_density = np.mean(edges > 0)

        # Gradient magnitude at boundary
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx**2 + gy**2)

        # High gradient at face boundary = blending artifact
        boundary_grad = np.mean(grad_mag)

        # Combine
        score = min(1.0, (boundary_grad * 10 + edge_density * 2) / 2.0)
        return score

    def _analyze_eye(self, region_img: np.ndarray) -> float:
        """Analyze eye region for deepfake artifacts."""
        gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        if gray.shape[0] < 16 or gray.shape[1] < 16:
            return 0.0

        scores = []

        # 1. Reflection consistency (specular highlights)
        # High reflections should be consistent
        bright_pixels = gray > 0.9
        if np.any(bright_pixels):
            reflection_score = np.mean(gray[bright_pixels])
            scores.append(1.0 - reflection_score)  # Unnaturally perfect reflections

        # 2. Iris/pupil circularity
        circles = cv2.HoughCircles(
            (gray * 255).astype(np.uint8),
            cv2.HOUGH_GRADIENT, dp=1, minDist=20,
            param1=50, param2=30, minRadius=5, maxRadius=min(gray.shape)//2
        )
        if circles is not None:
            # Check circularity of detected circles
            circles = np.uint16(np.around(circles))
            for c in circles[0, :]:
                # Perfect circles are suspicious in deepfakes
                scores.append(0.3)

        # 3. Blink/eye state consistency (would need temporal)
        # For single image, check symmetry
        h, w = gray.shape
        left_half = gray[:, :w//2]
        right_half = cv2.flip(gray[:, w//2:], 1)
        if left_half.shape == right_half.shape:
            symmetry_diff = np.mean(np.abs(left_half - right_half))
            scores.append(min(1.0, symmetry_diff * 10))

        return np.mean(scores) if scores else 0.0

    def _analyze_mouth(self, region_img: np.ndarray) -> float:
        """Analyze mouth region for lip-sync/articulation artifacts."""
        gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        if gray.shape[0] < 16 or gray.shape[1] < 16:
            return 0.0

        scores = []

        # 1. Lip edge sharpness (over-sharpened = synthetic)
        edges = cv2.Canny((gray * 255).astype(np.uint8), 30, 100)
        edge_density = np.mean(edges > 0)
        scores.append(min(1.0, edge_density * 5))

        # 2. Teeth region analysis (if visible)
        # High frequency in teeth region
        h, w = gray.shape
        upper = gray[:h//2, :]
        if upper.size > 0:
            f = np.fft.fft2(upper)
            mag = np.log(np.abs(np.fft.fftshift(f)) + 1e-10)
            hf_ratio = np.mean(mag[h//4:, w//4:3*w//4]) / (np.mean(mag) + 1e-10)
            scores.append(min(1.0, abs(hf_ratio - 0.5) * 2))

        return np.mean(scores) if scores else 0.0

    def _get_analysis_types(self, region_name: str) -> List[str]:
        """Get list of analysis types performed for region."""
        types = []
        if self.frequency_analysis:
            types.append("frequency")
        if self.texture_analysis:
            types.append("texture")
        if self.boundary_analysis and region_name == "face_boundary":
            types.append("boundary")
        if region_name in ["left_eye", "right_eye"]:
            types.append("eye_specific")
        if region_name == "mouth":
            types.append("mouth_specific")
        return types

    def _inconclusive(self, reason: str) -> DetectionResult:
        return DetectionResult(
            detector="FaceRegionAnalyzer",
            media_type="image",
            manipulation_detected=False,
            confidence=0.0,
            manipulation_type="deepfake",
            assessment=Assessment.INCONCLUSIVE,
            regions=[],
            metadata={
                "assessment": "INCONCLUSIVE",
                "reason": reason,
                "note": f"UNCERTAINTY: Face region analysis unavailable - {reason}",
            },
        )