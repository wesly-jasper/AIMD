"""
Visual Embedding Providers.

Provides feature representations for media similarity and clustering.
Distinctly classifies heuristic visual features from deep semantic embeddings.
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from pathlib import Path
import cv2


class BaseEmbeddingProvider(ABC):
    """Abstract base class for all visual embedding providers."""
    algorithm_name: str = "BASE_EMBEDDING"

    @abstractmethod
    def generate_embedding(self, file_path: str) -> List[float]:
        """Generate a numerical feature vector for the given media file."""
        pass


class BaselineEmbeddingProvider(BaseEmbeddingProvider):
    """
    Baseline visual feature extractor.
    Extracts normalized 3D color histogram (8x8x8 bins = 512 dimensions) as a visual feature vector.
    
    Classification: BASELINE_VISUAL_FEATURE (heuristic color distribution, NOT semantic ML embedding).
    """
    algorithm_name: str = "BASELINE_VISUAL_FEATURE"

    def generate_embedding(self, file_path: str) -> List[float]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            img = cv2.imread(str(path))
            if img is not None:
                # 8x8x8 color histogram
                hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                cv2.normalize(hist, hist)
                return hist.flatten().tolist()
        except Exception:
            pass

        return []


class MLEmbeddingProvider(BaseEmbeddingProvider):
    """
    Deep Neural Network Embedding Provider (e.g. CLIP, ResNet, OpenCLIP).
    Can be configured with pre-trained weights for deep semantic representation.
    """
    algorithm_name: str = "SEMANTIC_NEURAL_EMBEDDING"

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name
        self._is_loaded = False

    def generate_embedding(self, file_path: str) -> List[float]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self._is_loaded or not self.model_name:
            raise RuntimeError(
                f"MLEmbeddingProvider ({self.model_name or 'unconfigured'}) weights not loaded. "
                "Configure a valid model checkpoint to generate deep neural embeddings."
            )
        return []
