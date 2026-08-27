from pathlib import Path

from app.services.fingerprinting.hash_generator import HashGenerator


class FingerprintService:

    def __init__(self):
        self.hash_generator=HashGenerator()

    def generate(self,file_path):
        file_path=Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        sha256=self.hash_generator.generate_sha256(
            file_path
        )

        fingerprint={
            "sha256":sha256
        }

        image_extensions=[
            ".jpg",
            ".jpeg",
            ".png",
            ".webp"
        ]

        if file_path.suffix.lower() in image_extensions:
            fingerprint["phash"]=self.hash_generator.generate_phash(
                file_path
            )

        return fingerprint