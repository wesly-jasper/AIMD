"""
Unit tests for SQLAlchemy database repository and models.
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.database import Base
from app.db import repository as repo


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_media_crud(db):
    media = repo.create_media(db, {
        "id": "med-001",
        "original_filename": "sample.jpg",
        "stored_filename": "med-001.jpg",
        "file_path": "/uploads/med-001.jpg",
        "content_type": "image/jpeg",
        "media_type": "image",
        "size_bytes": 4096,
        "sha256": "abc123sha256hash",
    })
    assert media.id == "med-001"

    fetched = repo.get_media(db, "med-001")
    assert fetched is not None
    assert fetched.original_filename == "sample.jpg"

    by_hash = repo.get_media_by_sha256(db, "abc123sha256hash")
    assert by_hash is not None
    assert by_hash.id == "med-001"

    media_list = repo.list_media(db)
    assert len(media_list) == 1


def test_media_metadata_upsert(db):
    repo.create_media(db, {
        "id": "med-002",
        "original_filename": "sample.mp4",
        "stored_filename": "med-002.mp4",
        "file_path": "/uploads/med-002.mp4",
        "content_type": "video/mp4",
        "media_type": "video",
        "size_bytes": 102400,
        "sha256": "video_sha256",
    })

    meta = repo.upsert_media_metadata(db, "med-002", {
        "format": "mp4",
        "width": 1920,
        "height": 1080,
        "duration_seconds": 12.5,
        "fps": 30.0,
        "frame_count": 375,
        "codec": "avc1",
        "exif_data": {"Camera": "Sony"},
    })
    assert meta.width == 1920
    assert meta.duration_seconds == 12.5

    fetched = repo.get_media(db, "med-002")
    assert fetched.metadata_record.fps == 30.0


def test_analysis_lifecycle(db):
    repo.create_media(db, {
        "id": "med-003",
        "original_filename": "image.png",
        "stored_filename": "med-003.png",
        "file_path": "/uploads/med-003.png",
        "content_type": "image/png",
        "media_type": "image",
        "size_bytes": 2048,
        "sha256": "img_sha256",
    })

    analysis = repo.create_analysis(db, {
        "id": "ana-001",
        "media_id": "med-003",
        "status": "QUEUED",
    })
    assert analysis.status == "QUEUED"

    repo.update_analysis_status(
        db,
        "ana-001",
        status="COMPLETED",
        assessment="SUSPICIOUS",
        confidence=0.88,
    )

    updated = repo.get_analysis(db, "ana-001")
    assert updated.status == "COMPLETED"
    assert updated.assessment == "SUSPICIOUS"
    assert updated.overall_confidence == 0.88
    assert updated.completed_at is not None


def test_detection_and_regions(db):
    repo.create_media(db, {
        "id": "med-004",
        "original_filename": "tampered.jpg",
        "stored_filename": "med-004.jpg",
        "file_path": "/uploads/med-004.jpg",
        "content_type": "image/jpeg",
        "media_type": "image",
        "size_bytes": 8192,
        "sha256": "tampered_sha256",
    })
    repo.create_analysis(db, {
        "id": "ana-002",
        "media_id": "med-004",
        "status": "PROCESSING",
    })

    det = repo.create_detection(db, {
        "analysis_id": "ana-002",
        "detector": "ForensicImageDetector",
        "detector_version": "1.0",
        "media_type": "image",
        "manipulation_type": "image",
        "assessment": "SUSPICIOUS",
        "manipulation_detected": True,
        "confidence": 0.85,
        "detector_metadata": {"method": "ELA"},
        "regions": [
            {
                "region_type": "ela_anomaly",
                "confidence": 0.82,
                "bbox": [10.0, 20.0, 50.0, 50.0],
            }
        ],
    })
    assert det.id is not None
    assert len(det.regions) == 1
    assert det.regions[0].region_type == "ela_anomaly"

    dets = repo.get_detections_for_analysis(db, "ana-002")
    assert len(dets) == 1
    assert len(dets[0].regions) == 1


def test_provenance_and_evidence(db):
    repo.create_media(db, {
        "id": "med-005",
        "original_filename": "evidence.jpg",
        "stored_filename": "med-005.jpg",
        "file_path": "/uploads/med-005.jpg",
        "content_type": "image/jpeg",
        "media_type": "image",
        "size_bytes": 5000,
        "sha256": "ev_sha256",
    })
    repo.create_analysis(db, {
        "id": "ana-003",
        "media_id": "med-005",
        "status": "COMPLETED",
        "assessment": "CLEAN",
        "overall_confidence": 0.05,
    })

    # Provenance
    nodes = [
        {"node_id": "node_1", "label": "Original Post", "node_type": "source", "node_metadata": {}},
        {"node_id": "med-005", "label": "Uploaded Media", "node_type": "uploaded", "node_metadata": {}},
    ]
    edges = [
        {"source_node_id": "node_1", "target_node_id": "med-005", "relationship": "reposted_from", "confidence": 0.95}
    ]
    graph = repo.upsert_provenance_graph(db, "ana-003", nodes, edges)
    assert len(graph["nodes"]) == 2
    assert len(graph["edges"]) == 1

    # Evidence items
    repo.bulk_create_evidence(db, [
        {
            "analysis_id": "ana-003",
            "category": "FACT",
            "description": "File SHA-256 is ev_sha256",
            "source_stage": "fingerprinting",
            "confidence": 1.0,
        },
        {
            "analysis_id": "ana-003",
            "category": "OBSERVATION",
            "description": "No visual anomalies detected",
            "source_stage": "detection",
            "confidence": 0.05,
        },
    ])
    items = repo.get_evidence_for_analysis(db, "ana-003")
    assert len(items) == 2
    assert {i.category for i in items} == {"FACT", "OBSERVATION"}
