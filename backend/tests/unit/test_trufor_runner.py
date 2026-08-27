import pytest

from app.services.detection.models.trufor_runner import TruForRunner


def test_runner_without_model_path():
    runner=TruForRunner()

    with pytest.raises(ValueError):
        runner.load()


def test_runner_model_not_found(tmp_path):
    runner=TruForRunner(
        tmp_path/"trufor.pth.tar"
    )

    with pytest.raises(FileNotFoundError):
        runner.load()


def test_runner_not_loaded(tmp_path):
    image=tmp_path/"test.jpg"
    image.write_bytes(b"test")

    runner=TruForRunner(
        tmp_path/"trufor.pth.tar"
    )

    with pytest.raises(RuntimeError):
        runner.predict(image)


def test_runner_predict_not_implemented(tmp_path):
    model=tmp_path/"trufor.pth.tar"
    model.write_bytes(b"mock model")

    image=tmp_path/"test.jpg"
    image.write_bytes(b"test")

    runner=TruForRunner(model)

    runner.load()

    with pytest.raises(NotImplementedError):
        runner.predict(image)