from app.services.evidence.evidence_service import EvidenceService

def test_evidence_service():
    service = EvidenceService()
    package = service.generate_evidence_package(
        media_id="media_test_123",
        file_path="uploads/test_sample.jpg",
        analysis_id="analysis_test_456"
    )
    
    assert "case_information" in package
    assert package["case_information"]["case_id"] == "CASE-analysis_test_456"
    assert package["case_information"]["media_id"] == "media_test_123"
    assert "media_evidence" in package
    assert "detection_evidence" in package
    assert "fingerprint_evidence" in package
    assert "source_evidence" in package
    assert "provenance_evidence" in package
    assert "limitations" in package
    assert len(package["limitations"]) > 0
