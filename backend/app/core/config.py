from pathlib import Path
from pydantic_settings import BaseSettings,SettingsConfigDict


BASE_DIR=Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name:str="AIMD"
    app_version:str="1.0.0"

    upload_dir:Path=BASE_DIR/"uploads"
    processed_dir:Path=BASE_DIR/"processed"

    frames_dir:Path=BASE_DIR/"processed"/"frames"
    keyframes_dir:Path=BASE_DIR/"processed"/"keyframes"

    trufor_model_path:Path|None=None
    trufor_enabled:bool=False

    max_file_size_mb:int=500

    allowed_image_types:list[str]=[
        "image/jpeg",
        "image/png",
        "image/webp"
    ]

    allowed_video_types:list[str]=[
        "video/mp4",
        "video/avi",
        "video/mov",
        "video/mkv"
    ]

    allowed_audio_types:list[str]=[
        "audio/wav",
        "audio/mp3",
        "audio/mpeg"
    ]

    model_config=SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


settings=Settings()