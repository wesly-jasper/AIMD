from pathlib import Path

from app.services.detection.router.media_detection_router import (
    MediaDetectionRouter
)


class MediaAnalysisService:

    def __init__(self,router=None):
        self.router=router or MediaDetectionRouter()

    def analyze(
        self,
        file_path,
        media_type
    ):

        file_path=Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        if not media_type:
            raise ValueError(
                "Media type is required"
            )

        return self.router.detect(
            file_path,
            media_type
        )