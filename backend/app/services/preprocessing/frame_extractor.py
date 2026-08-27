from pathlib import Path
import cv2


class FrameExtractor:
    def __init__(self,output_dir):
        self.output_dir=Path(output_dir)

    def extract(self,file_path,media_id,frame_interval=1):
        file_path=Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if frame_interval<1:
            raise ValueError("frame_interval must be at least 1")

        cap=cv2.VideoCapture(str(file_path))

        if not cap.isOpened():
            raise ValueError("Unable to open video")

        frame_dir=self.output_dir/media_id
        frame_dir.mkdir(parents=True,exist_ok=True)

        frame_count=0
        saved_count=0
        frame_paths=[]

        while True:
            ret,frame=cap.read()

            if not ret:
                break

            if frame_count%frame_interval==0:
                frame_path=frame_dir/f"frame_{saved_count+1:06d}.jpg"

                success=cv2.imwrite(str(frame_path),frame)

                if success:
                    frame_paths.append(frame_path)
                    saved_count+=1

            frame_count+=1

        cap.release()

        return {
            "frame_count":frame_count,
            "saved_count":saved_count,
            "frames":frame_paths
        }