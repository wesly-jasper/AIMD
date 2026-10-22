import json
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


class UnsupportedOutputNet(nn.Module):
    def forward(self, x):
        return {"features": x}


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
    assert "regions" in result
    assert isinstance(result["regions"], list)
    assert "metadata" in result
    assert result["metadata"]["model_format"] == "pytorch"
    # The threshold the score was compared against is part of the record: the same
    # score is a detection or not depending on it.
    assert result["metadata"]["threshold"] == 0.40


def test_runner_reports_no_label_and_no_host_path(tmp_path):
    """
    The runner reports a score; it does not label the image, and it does not
    disclose where the checkpoint lives on this host.

    It used to return ``'assessment': 'SUSPICIOUS' if manipulation_detected else
    'CLEAN'``. A sub-threshold raw score from one model is not grounds to call an
    image clean -- this class cannot see which other detectors ran or whether any
    of them failed. It also returned ``metadata['status'] = 'MODEL_AVAILABLE'``, a
    third status vocabulary, and ``metadata['model_path']``, an absolute path that
    reached clients because TruForDetector passes this metadata through verbatim
    into the stored DetectorResult.
    """
    model_path = tmp_path / "trufor.pt"
    torch.save(DummyTruForNet(), str(model_path))

    img_path = tmp_path / "sample.png"
    Image.fromarray(
        np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
    ).save(img_path)

    runner = TruForRunner(model_path)
    runner.load()
    result = runner.predict(img_path, threshold=0.40)

    assert "assessment" not in result, "labelling is the detector's job, not the runner's"
    assert "status" not in result["metadata"], "a third status vocabulary is back"

    # The filename identifies which checkpoint produced the score; the directory
    # holding it is not forensic information about the evidence.
    assert result["metadata"]["model_file"] == "trufor.pt"
    assert "model_path" not in result["metadata"]

    serialised = json.dumps(result)
    assert str(tmp_path) not in serialised
    assert str(model_path) not in serialised


def test_runner_rejects_weights_without_trufor_architecture(tmp_path):
    model_path = tmp_path / "weights-only.pth"
    torch.save({"state_dict": DummyTruForNet().state_dict()}, model_path)
    runner = TruForRunner(model_path)

    with pytest.raises(RuntimeError, match="architecture"):
        runner.load()


def test_runner_does_not_turn_unsupported_output_into_clean(tmp_path):
    model_path = tmp_path / "unsupported-output.pt"
    torch.save(UnsupportedOutputNet(), model_path)
    image_path = tmp_path / "sample.png"
    Image.fromarray(np.zeros((12, 16, 3), dtype=np.uint8)).save(image_path)

    runner = TruForRunner(model_path)
    runner.load()

    with pytest.raises(ValueError, match="supported anomaly map or score"):
        runner.predict(image_path)
