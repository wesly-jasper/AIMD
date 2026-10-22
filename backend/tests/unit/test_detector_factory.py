from app.core.config import settings
from app.services.detection.detector_factory import (
    DetectorFactory
)


def test_trufor_disabled(monkeypatch):
    # Set explicitly: a developer's backend/.env may enable TruFor.
    monkeypatch.setattr(settings, "trufor_enabled", False)

    result=DetectorFactory.create_trufor()

    assert result is None
