from pathlib import Path
import hashlib

import cv2


class HashGenerator:

    def generate_sha256(self,file_path):
        file_path=Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        sha256=hashlib.sha256()

        with open(file_path,"rb") as file:
            while True:
                data=file.read(8192)

                if not data:
                    break

                sha256.update(data)

        return sha256.hexdigest()

    def generate_phash(self,file_path):
        file_path=Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        image=cv2.imread(str(file_path),cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(
                "Unable to read image"
            )

        image=cv2.resize(image,(32,32))

        dct=cv2.dct(image.astype("float32"))

        dct_low=dct[:8,:8]

        median=dct_low[1:,:].mean()

        hash_bits=dct_low>median

        return "".join(
            "1" if bit else "0"
            for bit in hash_bits.flatten()
        )