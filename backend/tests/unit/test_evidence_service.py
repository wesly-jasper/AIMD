import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.database import Base
from app.db import repository as repo
from app.services.evidence.evidence_service import EvidenceService


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_evidence_service_build_package(db_session):
    # 1. Create media record
    media = repo.create_media(db_session, {
        "id": "media_test_123",
        "original_filename": "test_sample.jpg",
        "stored_filename": "media_test_123.jpg",
        "file_path": "uploads/test_sample.jpg",
        "content_type": "image/jpeg",
        "media_type": "image",
        "size_bytes": 10240,
        "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    })

    # 2. Create analysis record
    analysis = repo.create_analysis(db_session, {
        "id": "analysis_test_456",
        "media_id": "media_test_123",
        "status": "COMPLETED",
        "assessment": "SUSPICIOUS",
        "overall_confidence": 0.85,
    })

    # 3. Create fingerprint record
    repo.create_fingerprint(db_session, {
        "media_id": "media_test_123",
        "analysis_id": "analysis_test_456",
        "algorithm": "sha256",
        "algorithm_version": "sha256",
        "value": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "scope": "full",
    })

    # 4. Create detection record
    repo.create_detection(db_session, {
        "analysis_id": "analysis_test_456",
        "detector": "ForensicImageDetector",
        "detector_version": "1.0",
        "media_type": "image",
        "manipulation_type": "image",
        "assessment": "SUSPICIOUS",
        "manipulation_detected": True,
        "confidence": 0.85,
        "detector_metadata": {"assessment": "SUSPICIOUS", "note": "High ELA discrepancy"},
        "regions": [],
    })

    # 5. Build evidence package
    service = EvidenceService()
    package = service.build_evidence_package("analysis_test_456", db_session)

    assert "case_information" in package
    assert package["case_information"]["analysis_id"] == "analysis_test_456"
    assert package["case_information"]["media_id"] == "media_test_123"

    assert "media_evidence" in package
    assert package["media_evidence"]["sha256"] == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert package["media_evidence"]["evidence_category"] == "FACT"

    assert "detection_evidence" in package
    assert len(package["detection_evidence"]) == 1
    assert package["detection_evidence"][0]["detector"] == "ForensicImageDetector"
    assert package["detection_evidence"][0]["assessment"] == "SUSPICIOUS"

    assert "classified_evidence" in package
    assert "FACT" in package["classified_evidence"]
    assert "OBSERVATION" in package["classified_evidence"]
    assert "UNCERTAINTY" in package["classified_evidence"]

    assert "limitations" in package
    assert len(package["limitations"]) > 0
