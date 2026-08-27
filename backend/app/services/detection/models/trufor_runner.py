from pathlib import Path


class TruForRunner:

    def __init__(self,model_path=None):
        self.model_path=model_path
        self.loaded=False

    def load(self):

        if not self.model_path:
            raise ValueError(
                "TruFor model path is not configured"
            )

        model_path=Path(self.model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"TruFor model not found: {model_path}"
            )

        self.loaded=True

    def predict(self,file_path):

        if not self.loaded:
            raise RuntimeError(
                "TruFor model is not loaded"
            )

        file_path=Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        raise NotImplementedError(
            "TruFor inference is not connected yet"
        )