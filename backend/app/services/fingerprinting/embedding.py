from abc import ABC, abstractmethod
from typing import List, Any
import cv2
from pathlib import Path

class BaseEmbeddingProvider(ABC):
    @abstractmethod
    def generate_embedding(self, file_path: str) -> List[float]:
        pass

class BaselineEmbeddingProvider(BaseEmbeddingProvider):
    """
    Baseline embedding provider using a color histogram.
    Placeholder for deep learning feature embeddings (e.g. ResNet, CLIP).
    """
    def generate_embedding(self, file_path: str) -> List[float]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        try:
            img = cv2.imread(str(file_path))
            if img is not None:
                # Calculate a simple 1D color histogram as a baseline "embedding"
                hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                cv2.normalize(hist, hist)
                return hist.flatten().tolist()
        except Exception:
            pass
            
        return []
