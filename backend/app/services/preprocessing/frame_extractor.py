"""
Frame extractor — extracts frames from video files with timestamp mapping.

Every extracted frame records:
  - media_id: the source media
  - frame_index: original frame number in the video
  - timestamp_seconds: exact position in the video
  - file_path: where the extracted image was saved
  - width/height: frame dimensions

Supports configurable sampling:
  - frame_interval: save every Nth frame (original default)
  - target_fps: sample at a specific frames-per-second rate

Does NOT load the entire video into memory.
"""
import logging
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


class FrameExtractor:

    def __init__(self, output_dir: Path | str):
        self.output_dir = Path(output_dir)

    def extract(
        self,
        file_path: Path | str,
        media_id: str,
        frame_interval: int = 1,
        target_fps: float | None = None,
    ) -> dict:
        """
        Extract frames from a video.

        Args:
            file_path: Path to the video file.
            media_id: ID of the source media (used for directory naming).
            frame_interval: Save every Nth original frame. Ignored if target_fps set.
            target_fps: If set, sample at this many frames per second.
                        E.g. target_fps=1.0 → one frame per second.

        Returns:
            dict with:
              video_fps: native FPS of the video
              total_frames: total frame count in video
              saved_count: number of frames saved
              frames: list of frame dicts (frame_index, timestamp_seconds,
                      file_path, width, height, media_id)
        """
        if frame_interval is not None and frame_interval <= 0:
            raise ValueError("Frame interval must be greater than zero")

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {file_path}")

        frame_dir = self.output_dir / media_id
        frame_dir.mkdir(parents=True, exist_ok=True)

        try:
            native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Determine which frame indices to save
            if target_fps is not None and target_fps > 0:
                # Sample at target_fps: save every (native_fps / target_fps)th frame
                step = max(1, round(native_fps / target_fps))
            else:
                step = max(1, int(frame_interval))

            saved_frames: list[dict] = []
            frame_index = 0
            saved_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_index % step == 0:
                    timestamp_seconds = round(frame_index / native_fps, 4)
                    frame_filename = f"frame_{frame_index:07d}.jpg"
                    frame_path = frame_dir / frame_filename

                    success = cv2.imwrite(
                        str(frame_path),
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 90],
                    )
                    if success:
                        saved_frames.append({
                            "media_id": media_id,
                            "frame_index": frame_index,
                            "timestamp_seconds": timestamp_seconds,
                            "file_path": str(frame_path),
                            "width": width,
                            "height": height,
                            "is_keyframe": False,
                            "scene_score": None,
                        })
                        saved_count += 1

                frame_index += 1

        finally:
            cap.release()

        logger.info(
            "Extracted %d frames from %s (total=%d, step=%d)",
            saved_count, file_path.name, total_frames, step,
        )

        return {
            "video_fps": round(native_fps, 3),
            "total_frames": total_frames,
            "frame_count": total_frames,
            "saved_count": saved_count,
            "frames": saved_frames,
        }