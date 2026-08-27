import pytest
from PIL import Image
from app.services.fingerprinting.embedding import BaselineEmbeddingProvider, BaseEmbeddingProvider

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
