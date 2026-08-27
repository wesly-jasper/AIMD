from app.services.detection.engine.image_detection_engine import (
    ImageDetectionEngine
)


class MediaDetectionRouter:

    def __init__(
        self,
        image_engine=None,
        video_engine=None
    ):
        self.image_engine=image_engine
        self.video_engine=video_engine

    def detect(
        self,
        file_path,
        media_type
    ):

        if media_type=="image":

            if self.image_engine is None:
                raise RuntimeError(
                    "Image detection engine is not configured"
                )

            return self.image_engine.detect(
                file_path
            )

        if media_type=="video":

            if self.video_engine is None:
                raise RuntimeError(
                    "Video detection engine is not configured"
                )

            return self.video_engine.detect(
                file_path
            )

        raise ValueError(
            f"Unsupported media type: {media_type}"
        )

    # Alias for pipeline routing
    route = detect
    