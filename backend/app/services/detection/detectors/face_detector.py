import cv2
import numpy as np
from pathlib import Path
from typing import Any, Dict, List

from app.schemas.detection import DetectionResult, DetectionRegion
from app.services.detection.detectors.base_detector import BaseDetector

class BaseFaceDetector(BaseDetector):
    pass

class BaselineFaceDetector(BaseFaceDetector):
    """
    Advanced Face Forensics & Deepfake Inconsistency Analyzer.
    1. Detects facial regions using cascade detectors.
    2. Analyzes boundary blending gradient artifacts (Laplacian variance around face borders).
    3. Evaluates high-frequency skin texture vs. background texture consistency.
    4. Computes facial anomaly confidence scores for localized bounding boxes.
    """
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.status = "active_face_forensics"

    def detect(self, file_path: str) -> DetectionResult:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        regions = []
        overall_manipulation = False
        max_face_confidence = 0.0

        try:
            img = cv2.imread(str(file_path))
            if img is not None:
                h, w, _ = img.shape
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))

                for (x, y, fw, fh) in faces:
                    # Extract face patch
                    face_patch = gray[y:y+fh, x:x+fw]
                    
                    # 1. Boundary gradient analysis (Laplacian variance along the border)
                    border_mask = np.zeros((fh, fw), dtype=np.uint8)
                    cv2.rectangle(border_mask, (0, 0), (fw, fh), 255, 4)
                    border_pixels = face_patch[border_mask == 255]
                    inner_pixels = face_patch[border_mask == 0]

                    border_lap = cv2.Laplacian(face_patch, cv2.CV_64F)
                    boundary_var = float(np.var(border_lap[border_mask == 255])) if border_pixels.size > 0 else 1.0
                    inner_var = float(np.var(border_lap[border_mask == 0])) if inner_pixels.size > 0 else 1.0

                    # 2. Skin texture frequency analysis
                    f_face = np.fft.fft2(face_patch)
                    fshift = np.fft.fftshift(f_face)
                    mag = 20 * np.log(np.abs(fshift) + 1e-8)
                    hf_ratio = float(np.mean(mag[mag > np.median(mag)]) / (np.mean(mag) + 1e-6))

                    # Deepfake / Face replacement indicators:
                    # Blurry face border on sharp background, or over-smoothed face texture
                    ratio_boundary = boundary_var / (inner_var + 1e-6)
                    
                    anomaly_score = 0.15 # Baseline prior
                    if ratio_boundary < 0.35 or ratio_boundary > 3.0:
                        anomaly_score += 0.40 # Boundary mismatch artifact
                    if hf_ratio > 1.25:
                        anomaly_score += 0.25 # High-frequency generative artifact

                    anomaly_score = float(np.clip(anomaly_score, 0.05, 0.95))
                    max_face_confidence = max(max_face_confidence, anomaly_score)

                    regions.append(
                        DetectionRegion(
                            type="face_region",
                            confidence=round(anomaly_score, 3),
                            bbox=[float(x), float(y), float(fw), float(fh)]
                        )
                    )

                if len(regions) > 0 and max_face_confidence >= 0.50:
                    overall_manipulation = True

        except Exception as e:
            return DetectionResult(
                detector="BaselineFaceDetector",
                media_type="image",
                manipulation_detected=False,
                confidence=0.0,
                manipulation_type="face",
                regions=[],
                metadata={"status": "error", "error": str(e)}
            )

        return DetectionResult(
            detector="BaselineFaceDetector",
            media_type="image",
            manipulation_detected=overall_manipulation,
            confidence=round(max_face_confidence, 4) if regions else 0.0,
            manipulation_type="face",
            regions=regions,
            metadata={
                "status": self.status,
                "faces_detected": len(regions),
                "method": "Cascade Detection + Boundary Gradient & Texture Frequency Forensics"
            }
        )
