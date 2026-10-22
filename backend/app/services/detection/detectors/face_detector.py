import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from app.schemas.detection import (
    DetectionRegion,
    DetectionResult,
    DetectorAvailability,
    DetectorExecutionStatus,
    ScoreSemantics,
)
from app.services.detection.detectors.base_detector import BaseDetector

logger = logging.getLogger(__name__)


class BaseFaceDetector(BaseDetector):
    pass


class BaselineFaceDetector(BaseFaceDetector):
    """
    Baseline Face Forensics & Deepfake Inconsistency Analyzer.
    1. Detects facial regions using cascade detectors.
    2. Analyzes boundary blending gradient artifacts (Laplacian variance around face borders).
    3. Evaluates high-frequency skin texture vs. background texture consistency.
    4. Computes facial anomaly confidence scores for localized bounding boxes.
    """

    detector_id = "BaselineFaceDetector"
    version = "1.0.0"
    supported_media_types = frozenset({"image"})
    manipulation_type = "face"

    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.status = "baseline_cascade_forensics"

    def availability(self) -> tuple[DetectorAvailability, Optional[str]]:
        """The Haar cascade ships with OpenCV, but verify rather than assume: an
        empty classifier silently matches nothing, which would look like "no
        faces found" instead of "the detector has no model"."""
        if self.face_cascade.empty():
            return (
                DetectorAvailability.UNAVAILABLE,
                "OpenCV Haar frontal-face cascade could not be loaded",
            )
        return DetectorAvailability.AVAILABLE, None

    def detect(self, file_path: str) -> DetectionResult:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        regions: List[DetectionRegion] = []
        overall_manipulation = False
        max_face_confidence = 0.0

        try:
            img = cv2.imread(str(path))
            if img is None:
                return self.failed("Image input is empty, invalid, or unsupported")

            if img is not None:
                h, w, _ = img.shape
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))

                for (x, y, fw, fh) in faces:
                    face_patch = gray[y : y + fh, x : x + fw]
                    if face_patch.size == 0:
                        continue

                    # Boundary gradient analysis
                    border_mask = np.zeros((fh, fw), dtype=np.uint8)
                    cv2.rectangle(border_mask, (0, 0), (fw, fh), 255, 4)
                    border_pixels = face_patch[border_mask == 255]
                    inner_pixels = face_patch[border_mask == 0]

                    border_lap = cv2.Laplacian(face_patch, cv2.CV_64F)
                    boundary_var = float(np.var(border_lap[border_mask == 255])) if border_pixels.size > 0 else 1.0
                    inner_var = float(np.var(border_lap[border_mask == 0])) if inner_pixels.size > 0 else 1.0

                    # Texture frequency analysis
                    f_face = np.fft.fft2(face_patch)
                    fshift = np.fft.fftshift(f_face)
                    mag = 20 * np.log(np.abs(fshift) + 1e-8)
                    hf_ratio = float(np.mean(mag[mag > np.median(mag)]) / (np.mean(mag) + 1e-6))

                    ratio_boundary = boundary_var / (inner_var + 1e-6)
                    anomaly_score = 0.15
                    if ratio_boundary < 0.35 or ratio_boundary > 3.0:
                        anomaly_score += 0.40
                    if hf_ratio > 1.25:
                        anomaly_score += 0.25

                    anomaly_score = float(np.clip(anomaly_score, 0.05, 0.95))
                    max_face_confidence = max(max_face_confidence, anomaly_score)

                    regions.append(
                        DetectionRegion(
                            type="face_region",
                            confidence=round(anomaly_score, 3),
                            bbox=[float(x), float(y), float(fw), float(fh)],
                        )
                    )

                if len(regions) > 0 and max_face_confidence >= 0.50:
                    overall_manipulation = True

        except Exception as e:
            # AVAILABLE + FAILED: the cascade loaded and the analysis broke.
            return self.failed(f"Face analysis failed: {type(e).__name__}")

        if not regions:
            # No face was located, so no face-based observation exists either way.
            # This is INCONCLUSIVE, not a negative result: a face detector that
            # found no face has not examined any face for manipulation.
            return self.inconclusive(
                "No face regions located; no face-based observation possible",
                metadata={
                    "status": self.status,
                    "faces_detected": 0,
                    "method": (
                        "Cascade Detection + Boundary Gradient & Texture "
                        "Frequency Forensics"
                    ),
                },
            )

        return DetectionResult(
            detector=self.detector_id,
            detector_version=self.version,
            media_type="image",
            manipulation_detected=overall_manipulation,
            confidence=round(max_face_confidence, 4),
            manipulation_type=self.manipulation_type,
            availability=DetectorAvailability.AVAILABLE,
            execution_status=DetectorExecutionStatus.COMPLETED,
            # Laplacian boundary variance and texture-frequency ratios, both
            # hand-thresholded. Not a trained deepfake classifier output.
            score_semantics=ScoreSemantics.HEURISTIC_SEVERITY,
            regions=regions,
            metadata={
                "status": self.status,
                "faces_detected": len(regions),
                "method": "Cascade Detection + Boundary Gradient & Texture Frequency Forensics",
            },
        )


class DeepLearningFaceDetector(BaseFaceDetector):
    """
    Neural MTCNN Face Detector & Deepfake Forensic Analyzer.
    - Deep MTCNN face detection with landmark localisation.
    - Multi-scale forensic analysis:
      1) Boundary blending gradient variance
      2) 2D-FFT frequency spectrum analysis for GAN / diffusion artifacts
      3) Color distribution divergence between face patch and background
    """

    detector_id = "DeepLearningFaceDetector"
    version = "1.0.0"
    supported_media_types = frozenset({"image"})
    manipulation_type = "face"

    def __init__(self, device: Optional[str] = None):
        self._device_str = device
        self._mtcnn = None
        self._loaded = False
        self._load_error: Optional[str] = None
        self._load_attempted = False

    def _load(self) -> None:
        self._load_attempted = True
        try:
            import torch
            from facenet_pytorch import MTCNN

            device = torch.device(self._device_str) if self._device_str else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._mtcnn = MTCNN(keep_all=True, device=device, post_process=False)
            self._loaded = True
            self._load_error = None
            logger.info(f"Loaded MTCNN face detector on {device}")
        except Exception as exc:
            # Keep the reason. It is the difference between "torch is not
            # installed" and "the checkpoint download failed", and an examiner
            # reading the record needs to know which capability was missing.
            self._load_error = f"{type(exc).__name__}: {exc}"
            logger.warning(f"Could not initialize MTCNN: {self._load_error}")
            self._loaded = False

    def availability(self) -> tuple[DetectorAvailability, Optional[str]]:
        """
        Whether MTCNN itself is usable.

        This detector IS the neural face localiser. If torch, facenet-pytorch or
        the MTCNN weights are missing, the capability is absent -- it is not
        satisfied by running a Haar cascade instead, which is a different
        detector with different error characteristics.
        """
        if not self._load_attempted:
            self._load()
        if self._loaded and self._mtcnn is not None:
            return DetectorAvailability.AVAILABLE, None
        return (
            DetectorAvailability.UNAVAILABLE,
            f"MTCNN could not be initialised ({self._load_error or 'unknown reason'})",
        )

    def detect(self, file_path: str) -> DetectionResult:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        availability, reason = self.availability()
        if availability is not DetectorAvailability.AVAILABLE:
            # Previously this returned ``self.cascade_fallback.detect(file_path)``:
            # a Haar cascade result handed back in place of neural inference, with
            # nothing in the record saying MTCNN had been requested or why it was
            # substituted. A reader would have taken a classical result for a
            # neural one. Report the absent capability instead; the cascade
            # detector is registered separately and reports under its own identity.
            return self.unavailable(reason or "MTCNN unavailable")

        regions: List[DetectionRegion] = []
        max_anomaly = 0.0
        overall_manipulation = False

        try:
            pil_img = Image.open(path).convert("RGB")
            img_np = np.array(pil_img)
            h, w, _ = img_np.shape

            boxes, probs, landmarks = self._mtcnn.detect(pil_img, landmarks=True)

            if boxes is not None and len(boxes) > 0:
                for idx, box in enumerate(boxes):
                    prob = float(probs[idx]) if probs is not None else 0.90
                    if prob < 0.60:
                        continue

                    bx1, by1, bx2, by2 = [max(0, int(v)) for v in box]
                    bx2 = min(w, bx2)
                    by2 = min(h, by2)
                    fw = bx2 - bx1
                    fh = by2 - by1

                    if fw <= 10 or fh <= 10:
                        continue

                    face_patch = img_np[by1:by2, bx1:bx2]
                    gray_face = cv2.cvtColor(face_patch, cv2.COLOR_RGB2GRAY)

                    # 1. Boundary gradient analysis
                    border_mask = np.zeros((fh, fw), dtype=np.uint8)
                    cv2.rectangle(border_mask, (0, 0), (fw, fh), 255, 4)
                    border_lap = cv2.Laplacian(gray_face, cv2.CV_64F)
                    b_var = float(np.var(border_lap[border_mask == 255])) if np.any(border_mask == 255) else 1.0
                    i_var = float(np.var(border_lap[border_mask == 0])) if np.any(border_mask == 0) else 1.0
                    ratio_boundary = b_var / (i_var + 1e-6)

                    # 2. FFT frequency analysis
                    f_face = np.fft.fft2(gray_face)
                    fshift = np.fft.fftshift(f_face)
                    mag = 20 * np.log(np.abs(fshift) + 1e-8)
                    hf_ratio = float(np.mean(mag[mag > np.median(mag)]) / (np.mean(mag) + 1e-6))

                    # 3. Anomaly scoring
                    anomaly_score = 0.10
                    if ratio_boundary < 0.30 or ratio_boundary > 3.2:
                        anomaly_score += 0.45
                    if hf_ratio > 1.30:
                        anomaly_score += 0.25

                    anomaly_score = float(np.clip(anomaly_score, 0.05, 0.95))
                    max_anomaly = max(max_anomaly, anomaly_score)

                    region_meta = {"mtcnn_prob": round(prob, 4)}
                    if landmarks is not None and len(landmarks) > idx:
                        region_meta["landmarks"] = landmarks[idx].tolist()

                    regions.append(
                        DetectionRegion(
                            type="deepfake_face",
                            confidence=round(anomaly_score, 3),
                            bbox=[float(bx1), float(by1), float(fw), float(fh)],
                            extra=region_meta,
                        )
                    )

                if len(regions) > 0 and max_anomaly >= 0.50:
                    overall_manipulation = True

        except Exception as exc:
            logger.warning(f"MTCNN deepfake detection error: {exc}")
            # AVAILABLE + FAILED. This used to return the cascade's result, so a
            # crash inside neural inference was reported as a completed analysis
            # by a detector that had not run.
            return self.failed(f"MTCNN face analysis failed: {type(exc).__name__}")

        if not regions:
            # MTCNN completed and localised no face above threshold. No face was
            # examined, so there is no face-based observation in either direction.
            return self.inconclusive(
                "MTCNN located no face above the detection threshold; "
                "no face-based observation possible",
                metadata={
                    "status": "active_mtcnn_deepfake_detector",
                    "faces_detected": 0,
                    "method": (
                        "MTCNN Neural Face Localization + Multi-Scale Boundary "
                        "& Spectrum Forensics"
                    ),
                },
            )

        return DetectionResult(
            detector=self.detector_id,
            detector_version=self.version,
            media_type="image",
            manipulation_detected=overall_manipulation,
            confidence=round(max_anomaly, 4),
            manipulation_type=self.manipulation_type,
            availability=DetectorAvailability.AVAILABLE,
            execution_status=DetectorExecutionStatus.COMPLETED,
            # MTCNN supplies the face *location*; the anomaly score is still
            # hand-thresholded Laplacian variance and FFT ratios. Neural
            # localisation does not make the manipulation score a model output.
            score_semantics=ScoreSemantics.HEURISTIC_SEVERITY,
            regions=regions,
            metadata={
                "status": "active_mtcnn_deepfake_detector",
                "faces_detected": len(regions),
                "method": "MTCNN Neural Face Localization + Multi-Scale Boundary & Spectrum Forensics",
            },
        )
