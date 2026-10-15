from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from app.services.detection.models.trufor_runner import TruForRunner


class DummyTruForNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


def test_runner_without_model_path():
    runner = TruForRunner()
    with pytest.raises(ValueError):
        runner.load()


def test_runner_model_not_found(tmp_path):
    runner = TruForRunner(tmp_path / "trufor.pth")
    with pytest.raises(FileNotFoundError):
        runner.load()


def test_runner_not_loaded(tmp_path):
    image = tmp_path / "test.jpg"
    image.write_bytes(b"test")

    runner = TruForRunner(tmp_path / "trufor.pth")
    with pytest.raises(RuntimeError):
        runner.predict(image)


def test_runner_file_not_found(tmp_path):
    model_path = tmp_path / "trufor.pt"
    net = DummyTruForNet()
    torch.save(net, str(model_path))

    runner = TruForRunner(model_path)
    runner.load()

    with pytest.raises(FileNotFoundError):
        runner.predict(tmp_path / "non_existent.jpg")


def test_runner_pytorch_inference(tmp_path):
    model_path = tmp_path / "trufor.pt"
    net = DummyTruForNet()
    torch.save(net, str(model_path))

    img_path = tmp_path / "sample.png"
    sample_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    Image.fromarray(sample_img).save(img_path)

    runner = TruForRunner(model_path)
    runner.load()
    assert runner.loaded is True

    result = runner.predict(img_path, threshold=0.40)

    assert isinstance(result, dict)
    assert "manipulation_detected" in result
    assert isinstance(result["manipulation_detected"], bool)
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0
    assert "assessment" in result
    assert result["assessment"] in ["CLEAN", "SUSPICIOUS"]
    assert "regions" in result
    assert isinstance(result["regions"], list)
    assert "metadata" in result
    assert result["metadata"]["status"] == "MODEL_AVAILABLE"
    assert result["metadata"]["model_format"] == "pytorch"
