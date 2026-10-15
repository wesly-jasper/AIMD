import pytest
from PIL import Image
import numpy as np
from app.services.fingerprinting.embedding import (
    BaselineEmbeddingProvider,
    BaseEmbeddingProvider,
    OpenCLIPEmbeddingProvider,
    MLEmbeddingProvider,
)

def test_embedding_provider_missing_file():
    provider = BaselineEmbeddingProvider()
    with pytest.raises(FileNotFoundError):
        provider.generate_embedding("non_existent_image.png")

def test_embedding_provider_generation(tmp_path):
    provider = BaselineEmbeddingProvider()
    assert isinstance(provider, BaseEmbeddingProvider)
    
    img_path = tmp_path / "test_emb.png"
    img = Image.new("RGB", (64, 64), color="blue")
    img.save(str(img_path))
    
    embedding = provider.generate_embedding(str(img_path))
    assert isinstance(embedding, list)
    assert len(embedding) > 0

def test_openclip_embedding_provider(tmp_path):
    img_path = tmp_path / "test_clip.png"
    img = Image.new("RGB", (128, 128), color="green")
    img.save(str(img_path))

    # Initialize without remote weights for quick unit testing
    provider = OpenCLIPEmbeddingProvider(model_name="ViT-B-32", pretrained=None)
    provider.load()
    assert provider._is_loaded is True

    embedding = provider.generate_embedding(str(img_path))
    assert isinstance(embedding, list)
    assert len(embedding) == 512  # ViT-B-32 projection dimension
    
    # Check L2 normalization (sum of squares ~ 1.0)
    norm = np.linalg.norm(np.array(embedding))
    assert abs(norm - 1.0) < 1e-2

def test_openclip_missing_file():
    provider = OpenCLIPEmbeddingProvider()
    with pytest.raises(FileNotFoundError):
        provider.generate_embedding("missing.png")
