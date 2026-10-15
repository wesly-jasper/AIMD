import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.schemas.detection import Assessment, DetectionRegion, DetectionResult
from app.services.detection.detectors.base_detector import BaseDetector

logger = logging.getLogger(__name__)


class BaseObjectDetector(BaseDetector):
    pass


class BaselineObjectDetector(BaseObjectDetector):
    """
    Object detector placeholder.

    Returns UNAVAILABLE -- no object detection model is configured.
    This is the honest response: we cannot determine manipulation
    from object presence/absence without an actual model.
    """

    DETECTOR_VERSION = "1.0-unavailable"

    def detect(self, file_path: str) -> DetectionResult:
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        return DetectionResult(
            detector="BaselineObjectDetector",
            media_type="image",
            manipulation_detected=False,
            confidence=0.0,
            manipulation_type="object",
            assessment=Assessment.UNAVAILABLE,
            regions=[],
            metadata={
                "assessment": "UNAVAILABLE",
                "status": "baseline",
                "detector_version": self.DETECTOR_VERSION,
                "note": (
                    "UNCERTAINTY: No object detection model is configured. "
                    "Object anomaly detection requires a trained model. "
                    "Configure a YOLOv8 or compatible detector via the plugin interface."
                ),
            },
        )


class YOLOv8ObjectDetector(BaseObjectDetector):
    """
    Real Object Detector using Ultralytics YOLOv8.
    Locates objects, spatial coordinates, bounding boxes, and object classes.
    Note: Object detection locates semantic objects and does not signify manipulation.
    """

    DETECTOR_VERSION = "2.0-yolov8"

    def __init__(self, model_name: str = "yolov8n.pt", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._loaded = False

    def load(self):
        try:
            from ultralytics import YOLO

            self._model = YOLO(self.model_name)
            self._loaded = True
            logger.info(f"YOLOv8 model {self.model_name} loaded")
        except Exception as exc:
            logger.warning(f"Could not load YOLOv8 model {self.model_name}: {exc}")
            self._loaded = False

    def detect(self, file_path: str) -> DetectionResult:
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self._loaded:
            self.load()

        if not self._loaded or self._model is None:
            return BaselineObjectDetector().detect(file_path)

        regions: List[DetectionRegion] = []

        try:
            results = self._model.predict(
                source=str(p),
                save=False,
                verbose=False,
                device=self.device if self.device else None,
            )

            if results and len(results) > 0:
                res = results[0]
                boxes = res.boxes
                if boxes is not None:
                    for box in boxes:
                        xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = self._model.names.get(cls_id, f"class_{cls_id}")

                        bx1, by1, bx2, by2 = xyxy
                        bw = bx2 - bx1
                        bh = by2 - by1

                        regions.append(
                            DetectionRegion(
                                type=f"object_{cls_name}",
                                confidence=round(conf, 4),
                                bbox=[float(bx1), float(by1), float(bw), float(bh)],
                                extra={"class_name": cls_name, "class_id": cls_id},
                            )
                        )

            return DetectionResult(
                detector="YOLOv8ObjectDetector",
                media_type="image",
                manipulation_detected=False,  # Object presence != manipulation
                confidence=0.0,
                manipulation_type="object",
                assessment=Assessment.CLEAN,
                regions=regions,
                metadata={
                    "status": "active_yolov8_detector",
                    "detector_version": self.DETECTOR_VERSION,
                    "model_name": self.model_name,
                    "objects_detected_count": len(regions),
                    "note": (
                        "Object detection successfully identified bounding coordinates and semantic categories. "
                        "Object detection does not indicate forgery or manipulation."
                    ),
                },
            )

        except Exception as exc:
            logger.warning(f"YOLOv8 detection failed: {exc}")
            return BaselineObjectDetector().detect(file_path)
