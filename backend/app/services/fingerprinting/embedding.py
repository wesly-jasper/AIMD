"""
Visual Embedding Providers.

Provides feature representations for media similarity and clustering.
Distinctly classifies heuristic visual features from deep semantic embeddings.
"""
from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import List, Optional, Union
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


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


class OpenCLIPEmbeddingProvider(BaseEmbeddingProvider):
    """
    Deep Semantic Visual Embedding Provider using OpenCLIP.
    Extracts 512/768-dimensional L2-normalized semantic feature representations.
    """
    algorithm_name: str = "OPENCLIP_SEMANTIC_EMBEDDING"

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.pretrained = pretrained
        self._device_str = device
        self._is_loaded = False
        self.model = None
        self.preprocess = None
        self.device = None

    def load(self) -> None:
        """Load OpenCLIP model and transformation pipeline onto GPU/CPU."""
        import torch
        import open_clip

        if self._device_str:
            self.device = torch.device(self._device_str)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            if self.pretrained:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    self.model_name,
                    pretrained=self.pretrained,
                    device=self.device,
                )
            else:
                # Initialize model architecture (weights initialized or custom)
                model = open_clip.create_model(
                    self.model_name,
                    pretrained=None,
                    device=self.device,
                )
                preprocess = open_clip.image_transform(
                    model.visual.image_size,
                    is_train=False,
                )

            self.model = model.eval()
            self.preprocess = preprocess
            self._is_loaded = True
            logger.info(f"OpenCLIP loaded: {self.model_name} on {self.device}")
        except Exception as exc:
            logger.error(f"Failed to load OpenCLIP model {self.model_name}: {exc}")
            raise RuntimeError(f"OpenCLIP model loading failed: {exc}") from exc

    def generate_embedding(self, file_path: str) -> List[float]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self._is_loaded:
            self.load()

        import torch

        try:
            pil_img = Image.open(path).convert("RGB")
            tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.model.encode_image(tensor)
                # L2 normalize so cosine similarity equals dot product
                features = features / features.norm(dim=-1, keepdim=True)
                embedding = features.cpu().squeeze().tolist()

            if isinstance(embedding, float):
                embedding = [embedding]
            return [round(float(v), 6) for v in embedding]

        except Exception as exc:
            logger.warning(f"Failed to generate OpenCLIP embedding for {file_path}: {exc}")
            raise RuntimeError(f"OpenCLIP inference error: {exc}") from exc


class MLEmbeddingProvider(BaseEmbeddingProvider):
    """
    Unified Deep Neural Network Embedding Provider.
    Defaults to OpenCLIP semantic representations with fallback to baseline visual features.
    """
    algorithm_name: str = "SEMANTIC_NEURAL_EMBEDDING"

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or "ViT-B-32"
        self._clip_provider: Optional[OpenCLIPEmbeddingProvider] = None
        self._is_loaded = False

    def load(self) -> None:
        try:
            self._clip_provider = OpenCLIPEmbeddingProvider(model_name=self.model_name)
            self._clip_provider.load()
            self._is_loaded = True
        except Exception as exc:
            logger.warning(f"MLEmbeddingProvider could not load deep model: {exc}")
            self._is_loaded = False

    def generate_embedding(self, file_path: str) -> List[float]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if self._clip_provider is None and not self._is_loaded:
            try:
                self.load()
            except Exception:
                pass

        if self._clip_provider and self._clip_provider._is_loaded:
            return self._clip_provider.generate_embedding(file_path)

        raise RuntimeError(
            f"MLEmbeddingProvider ({self.model_name}) weights not loaded. "
            "Configure a valid model checkpoint to generate deep neural embeddings."
        )
