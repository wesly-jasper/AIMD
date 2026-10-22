import logging
from pathlib import Path
from typing import List, Optional

from app.schemas.detection import (
    DetectionRegion,
    DetectionResult,
    DetectorAvailability,
)
from app.services.detection.detectors.base_detector import BaseDetector

logger = logging.getLogger(__name__)


class BaseObjectDetector(BaseDetector):
    """
    Common base for object detectors.

    Object detection is NON-EVIDENTIAL in this system (see ``_SIGNAL_FAMILY`` in
    the image detection engine). Locating a dog in a photograph establishes that
    there is a dog in the photograph and nothing whatsoever about whether the
    photograph was manipulated. These detectors therefore never report
    ``manipulation_detected=True``, and they are also never allowed to report a
    *negative* finding: "no manipulation detected by the object detector" is not
    a statement anyone may lean on in either direction.
    """

    manipulation_type = "object"
    supported_media_types = frozenset({"image"})


class BaselineObjectDetector(BaseObjectDetector):
    """
    Object detector placeholder.

    No object detection model is configured, so the capability is absent. This is
    the honest response: manipulation cannot be determined from object
    presence or absence without an actual model -- and, as the base class notes,
    not from object presence even with one.
    """

    detector_id = "BaselineObjectDetector"
    version = "1.0-unavailable"

    def availability(self) -> tuple[DetectorAvailability, Optional[str]]:
        """Always absent. This class exists to *be* the unconfigured case."""
        return (
            DetectorAvailability.UNAVAILABLE,
            "No object detection model is configured. Configure a YOLOv8 or "
            "compatible detector via the plugin interface.",
        )

    def detect(self, file_path: str) -> DetectionResult:
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        _, reason = self.availability()
        return self.unavailable(reason or "No object detection model is configured")


class YOLOv8ObjectDetector(BaseObjectDetector):
    """
    Object detector using Ultralytics YOLOv8.

    Locates objects, spatial coordinates, bounding boxes and object classes. A
    completed run is recorded as INCONCLUSIVE, which is the literal truth: the
    model ran, it localised objects, and object localisation cannot support a
    determination about manipulation. The per-box numbers live on the regions,
    where they are plainly object-class confidences; the result's own
    ``confidence`` stays 0.0 with NOT_APPLICABLE semantics, because this detector
    produces no manipulation score at all.
    """

    detector_id = "YOLOv8ObjectDetector"
    version = "2.0-yolov8"

    def __init__(self, model_name: str = "yolov8n.pt", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._loaded = False
        self._load_attempted = False
        self._load_error: Optional[str] = None

    def load(self):
        self._load_attempted = True
        try:
            from ultralytics import YOLO

            self._model = YOLO(self.model_name)
            self._loaded = True
            logger.info(f"YOLOv8 model {self.model_name} loaded")
        except Exception as exc:
            # Keep the reason. "ultralytics is not installed" and "the weights
            # file is missing" are different deployment problems, and an examiner
            # reading the record needs to know which capability was absent.
            self._load_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                f"Could not load YOLOv8 model {self.model_name}: {self._load_error}"
            )
            self._loaded = False

    def availability(self) -> tuple[DetectorAvailability, Optional[str]]:
        """
        Whether YOLOv8 itself is usable.

        This detector IS the YOLOv8 localiser. If ultralytics or the weights are
        missing, the capability is absent -- it is not satisfied by returning
        ``BaselineObjectDetector``'s result, which is a different detector with a
        different identity.
        """
        if self._loaded and self._model is not None:
            return DetectorAvailability.AVAILABLE, None
        if not self._load_attempted:
            self.load()
        if self._loaded and self._model is not None:
            return DetectorAvailability.AVAILABLE, None
        return (
            DetectorAvailability.UNAVAILABLE,
            f"YOLOv8 model {self.model_name} could not be loaded "
            f"({self._load_error or 'unknown reason'})",
        )

    def detect(self, file_path: str) -> DetectionResult:
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        availability, reason = self.availability()
        if availability is not DetectorAvailability.AVAILABLE:
            # Previously this returned ``BaselineObjectDetector().detect(...)``:
            # a placeholder result handed back under a *different detector's*
            # identity, with nothing in the record saying that YOLOv8 had been
            # requested or why it was substituted. The record showed a detector
            # the pipeline never asked for and omitted the one it did.
            return self.unavailable(reason or "YOLOv8 is unavailable")

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

        except Exception as exc:
            logger.warning(f"YOLOv8 detection failed: {exc}")
            # AVAILABLE + FAILED. The model loaded and inference broke, which is a
            # defect to fix -- not the same as the model being absent, and it must
            # not be recorded under another detector's name either.
            return self.failed(f"YOLOv8 inference failed: {type(exc).__name__}")

        # A completed run with no determination to offer. Regions are preserved
        # because object locations are useful context for an examiner even though
        # they can never corroborate a manipulation finding.
        return self.inconclusive(
            f"YOLOv8 localised {len(regions)} object(s). Object detection "
            "identifies semantic categories and bounding coordinates; it does "
            "not indicate forgery or manipulation, in either direction.",
            regions=regions,
            metadata={
                "detector_version": self.version,
                "model_name": self.model_name,
                "objects_detected_count": len(regions),
            },
        )
