from pathlib import Path
import hashlib
import uuid

from fastapi import UploadFile

from app.core.config import settings
from app.services.preprocessing.metadata_extractor import MetadataExtractor
from app.services.preprocessing.frame_extractor import FrameExtractor
from app.services.preprocessing.keyframe_selector import KeyframeSelector
from app.services.fingerprinting.fingerprint import FingerprintService


class MediaIngestionService:

    def __init__(self):
        self.upload_dir=settings.upload_dir
        self.max_file_size=settings.max_file_size_mb*1024*1024
        self.metadata_extractor=MetadataExtractor()
        self.frame_extractor=FrameExtractor(settings.frames_dir)
        self.keyframe_selector=KeyframeSelector(settings.keyframes_dir)
        self.fingerprint_service=FingerprintService()

    def validate_file(self,file:UploadFile):
        if not file.filename:
            raise ValueError("File name is missing")

        if not file.content_type:
            raise ValueError("File type is missing")

        allowed_types=(
            settings.allowed_image_types+
            settings.allowed_video_types+
            settings.allowed_audio_types
        )

        if file.content_type not in allowed_types:
            raise ValueError(
                f"Unsupported media type: {file.content_type}"
            )

    def generate_sha256(self,data:bytes)->str:
        return hashlib.sha256(data).hexdigest()

    async def ingest(self,file:UploadFile)->dict:
        self.validate_file(file)

        data=await file.read()

        if len(data)>self.max_file_size:
            raise ValueError(
                "File size exceeds the allowed limit"
            )

        media_id=str(uuid.uuid4())

        extension=Path(file.filename).suffix.lower()

        filename=f"{media_id}{extension}"

        self.upload_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        file_path=self.upload_dir/filename

        file_path.write_bytes(data)

        sha256=self.generate_sha256(data)

        metadata=self.metadata_extractor.extract(
            file_path
        )

        fingerprint=self.fingerprint_service.generate(
            file_path
        )

        frames=None
        keyframes=None
        keyframe_fingerprints=[]

        if metadata["media_type"]=="video":

            frames=self.frame_extractor.extract(
                file_path,
                media_id
            )

            keyframes=self.keyframe_selector.select(
                settings.frames_dir/media_id,
                media_id,
                frame_interval=10
            )

            for keyframe in keyframes["keyframes"]:

                keyframe_hash=self.fingerprint_service.generate(
                    keyframe
                )

                keyframe_fingerprints.append({
                    "file":str(keyframe),
                    "phash":keyframe_hash["phash"]
                })

        return {
            "media_id":media_id,
            "original_filename":file.filename,
            "stored_filename":filename,
            "content_type":file.content_type,
            "size_bytes":len(data),
            "sha256":sha256,
            "metadata":metadata,
            "fingerprint":fingerprint,
            "frames":frames,
            "keyframes":keyframes,
            "keyframe_fingerprints":keyframe_fingerprints,
            "file_path":str(file_path),
            "status":"UPLOADED"
        }