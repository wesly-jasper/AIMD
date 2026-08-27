from app.services.evidence.report_generator import ReportGenerator

def test_report_generator():
    generator = ReportGenerator()
    evidence_package = {
        "case_information": {
            "case_id": "CASE-001",
            "analysis_id": "ANALYSIS-001",
            "creation_time": "2024-05-12T10:00:00Z"
        },
        "fingerprint_evidence": {
            "sha256": "abc123sha256",
            "phash": "def456phash"
        },
        "detection_evidence": [
            {
                "detector": "BaselineFaceDetector",
                "manipulation_type": "face",
                "confidence": 0.1
            }
        ],
        "source_evidence": {
            "media_id": "https://source.com/img.jpg",
            "similarity": 0.92,
            "source_timestamp": "2024-05-01T00:00:00Z"
        },
        "limitations": [
            "Baseline detector used."
        ]
    }
    
    json_report = generator.generate_json_report(evidence_package)
    assert json_report == evidence_package
    
    md_report = generator.generate_markdown_report(evidence_package)
    assert "# AIMD FORENSIC ANALYSIS REPORT" in md_report
    assert "[FACT]" in md_report
    assert "[OBSERVATION]" in md_report
    assert "[INFERENCE]" in md_report
    assert "[UNCERTAINTY]" in md_report
    assert "abc123sha256" in md_report
