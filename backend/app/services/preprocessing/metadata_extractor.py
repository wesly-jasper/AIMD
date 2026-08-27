from pathlib import Path
import cv2


class MetadataExtractor:

    def extract(self,file_path:Path)->dict:

        file_path=Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        extension=file_path.suffix.lower()

        image_extensions={
            ".jpg",
            ".jpeg",
            ".png",
            ".webp"
        }

        video_extensions={
            ".mp4",
            ".avi",
            ".mov",
            ".mkv"
        }

        if extension in image_extensions:
            return self._extract_image_metadata(file_path)

        if extension in video_extensions:
            return self._extract_video_metadata(file_path)

        raise ValueError(
            f"Unsupported media format: {extension}"
        )

    def _extract_image_metadata(self,file_path:Path)->dict:

        image=cv2.imread(str(file_path))

        if image is None:
            raise ValueError(
                "Unable to read image"
            )

        height,width,channels=image.shape

        return {
            "media_type":"image",
            "format":file_path.suffix.lower().replace(".",""),
            "width":width,
            "height":height,
            "channels":channels
        }

    def _extract_video_metadata(self,file_path:Path)->dict:

        capture=cv2.VideoCapture(str(file_path))

        if not capture.isOpened():
            raise ValueError(
                "Unable to open video"
            )

        fps=capture.get(
            cv2.CAP_PROP_FPS
        )

        frame_count=int(
            capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        width=int(
            capture.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height=int(
            capture.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        duration=0

        if fps>0:
            duration=frame_count/fps

        capture.release()

        return {
            "media_type":"video",
            "format":file_path.suffix.lower().replace(".",""),
            "width":width,
            "height":height,
            "fps":fps,
            "frame_count":frame_count,
            "duration_seconds":duration
        }