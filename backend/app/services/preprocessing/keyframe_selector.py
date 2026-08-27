from pathlib import Path
import shutil


class KeyframeSelector:

    def __init__(self,output_dir):
        self.output_dir=Path(output_dir)

    def select(self,frame_dir,media_id,frame_interval=10):
        frame_dir=Path(frame_dir)

        if not frame_dir.exists():
            raise FileNotFoundError(
                f"Frame directory not found: {frame_dir}"
            )

        if frame_interval<1:
            raise ValueError(
                "frame_interval must be at least 1"
            )

        frames=sorted(frame_dir.glob("*.jpg"))

        keyframe_dir=self.output_dir/media_id
        keyframe_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        keyframes=[]

        for index,frame in enumerate(frames):
            if index%frame_interval==0:
                keyframe_path=keyframe_dir/frame.name

                shutil.copy2(
                    frame,
                    keyframe_path
                )

                keyframes.append(keyframe_path)

        return {
            "total_frames":len(frames),
            "keyframe_count":len(keyframes),
            "keyframes":keyframes
        }