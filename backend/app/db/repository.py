"""
Repository layer — all database CRUD operations.
Endpoints call services; services call repositories.
No business logic lives here.
"""
from datetime import datetime, timezone
from typing import Optional, List

from sqlalchemy.orm import Session

from app.db.models import (
    MediaRecord, MediaMetadataRecord, AnalysisRecord,
    FrameRecord, DetectionRecord, DetectionRegionRecord,
    FingerprintRecord, SimilarityMatchRecord, SourceRecord,
    ProvenanceNodeRecord, ProvenanceEdgeRecord,
    EvidenceItemRecord, ReportRecord,
)


# ── Media ─────────────────────────────────────────────────────────────────────

def create_media(db: Session, data: dict) -> MediaRecord:
    record = MediaRecord(**data)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_media(db: Session, media_id: str) -> Optional[MediaRecord]:
    return db.get(MediaRecord, media_id)


def get_media_by_sha256(db: Session, sha256: str) -> Optional[MediaRecord]:
    return (
        db.query(MediaRecord)
        .filter(MediaRecord.sha256 == sha256)
        .first()
    )


def list_media(db: Session, limit: int = 50, offset: int = 0) -> List[MediaRecord]:
    return db.query(MediaRecord).order_by(MediaRecord.created_at.desc()).offset(offset).limit(limit).all()


# ── Media Metadata ────────────────────────────────────────────────────────────

def upsert_media_metadata(db: Session, media_id: str, data: dict) -> MediaMetadataRecord:
    existing = db.query(MediaMetadataRecord).filter_by(media_id=media_id).first()
    if existing:
        for k, v in data.items():
            setattr(existing, k, v)
        db.commit()
        db.refresh(existing)
        return existing
    record = MediaMetadataRecord(media_id=media_id, **data)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


# ── Analysis ──────────────────────────────────────────────────────────────────

def create_analysis(db: Session, data: dict) -> AnalysisRecord:
    record = AnalysisRecord(**data)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_analysis(db: Session, analysis_id: str) -> Optional[AnalysisRecord]:
    return db.get(AnalysisRecord, analysis_id)


def update_analysis_status(
    db: Session,
    analysis_id: str,
    status: str,
    assessment: Optional[str] = None,
    confidence: Optional[float] = None,
    error_message: Optional[str] = None,
) -> Optional[AnalysisRecord]:
    record = db.get(AnalysisRecord, analysis_id)
    if not record:
        return None
    record.status = status
    if assessment is not None:
        record.assessment = assessment
    if confidence is not None:
        record.overall_confidence = confidence
    if error_message is not None:
        record.error_message = error_message
    if status in ("COMPLETED", "FAILED", "PARTIAL", "INCONCLUSIVE"):
        record.completed_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(record)
    return record


def list_analyses(db: Session, limit: int = 50, offset: int = 0) -> List[AnalysisRecord]:
    return (
        db.query(AnalysisRecord)
        .order_by(AnalysisRecord.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


# ── Frames ────────────────────────────────────────────────────────────────────

def create_frame(db: Session, data: dict) -> FrameRecord:
    record = FrameRecord(**data)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def bulk_create_frames(db: Session, frames: List[dict]) -> List[FrameRecord]:
    records = [FrameRecord(**f) for f in frames]
    db.add_all(records)
    db.commit()
    return records


def get_frames_for_media(db: Session, media_id: str) -> List[FrameRecord]:
    return (
        db.query(FrameRecord)
        .filter_by(media_id=media_id)
        .order_by(FrameRecord.frame_index)
        .all()
    )


def get_keyframes_for_media(db: Session, media_id: str) -> List[FrameRecord]:
    return (
        db.query(FrameRecord)
        .filter_by(media_id=media_id, is_keyframe=True)
        .order_by(FrameRecord.frame_index)
        .all()
    )


# ── Detections ────────────────────────────────────────────────────────────────

def create_detection(db: Session, data: dict) -> DetectionRecord:
    regions_data = data.pop("regions", [])
    record = DetectionRecord(**data)
    db.add(record)
    db.flush()  # Get the ID before adding regions
    for r in regions_data:
        region = DetectionRegionRecord(detection_id=record.id, **r)
        db.add(region)
    db.commit()
    db.refresh(record)
    return record


def get_detections_for_analysis(db: Session, analysis_id: str) -> List[DetectionRecord]:
    return (
        db.query(DetectionRecord)
        .filter_by(analysis_id=analysis_id)
        .all()
    )


# ── Fingerprints ──────────────────────────────────────────────────────────────

def create_fingerprint(db: Session, data: dict) -> FingerprintRecord:
    record = FingerprintRecord(**data)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def bulk_create_fingerprints(db: Session, fingerprints: List[dict]) -> List[FingerprintRecord]:
    records = [FingerprintRecord(**f) for f in fingerprints]
    db.add_all(records)
    db.commit()
    return records


def get_all_fingerprints(db: Session, algorithm: str) -> List[FingerprintRecord]:
    """Retrieve all stored fingerprints of a given algorithm for similarity search."""
    return db.query(FingerprintRecord).filter_by(algorithm=algorithm).all()


def get_fingerprints_for_media(db: Session, media_id: str) -> List[FingerprintRecord]:
    return db.query(FingerprintRecord).filter_by(media_id=media_id).all()


# ── Similarity Matches ────────────────────────────────────────────────────────

def create_similarity_match(db: Session, data: dict) -> SimilarityMatchRecord:
    record = SimilarityMatchRecord(**data)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def bulk_create_similarity_matches(db: Session, matches: List[dict]) -> List[SimilarityMatchRecord]:
    records = [SimilarityMatchRecord(**m) for m in matches]
    db.add_all(records)
    db.commit()
    return records


def get_similarity_matches_for_analysis(db: Session, analysis_id: str) -> List[SimilarityMatchRecord]:
    return (
        db.query(SimilarityMatchRecord)
        .filter_by(analysis_id=analysis_id)
        .order_by(SimilarityMatchRecord.rank)
        .all()
    )


# ── Sources ───────────────────────────────────────────────────────────────────

def create_source(db: Session, data: dict) -> SourceRecord:
    record = SourceRecord(**data)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def bulk_create_sources(db: Session, sources: List[dict]) -> List[SourceRecord]:
    records = [SourceRecord(**s) for s in sources]
    db.add_all(records)
    db.commit()
    return records


def get_sources_for_analysis(db: Session, analysis_id: str) -> List[SourceRecord]:
    return db.query(SourceRecord).filter_by(analysis_id=analysis_id).all()


# ── Provenance ────────────────────────────────────────────────────────────────

def upsert_provenance_graph(
    db: Session,
    analysis_id: str,
    nodes: List[dict],
    edges: List[dict],
):
    # Delete existing provenance for this analysis and recreate
    db.query(ProvenanceNodeRecord).filter_by(analysis_id=analysis_id).delete()
    db.query(ProvenanceEdgeRecord).filter_by(analysis_id=analysis_id).delete()
    for n in nodes:
        db.add(ProvenanceNodeRecord(analysis_id=analysis_id, **n))
    for e in edges:
        db.add(ProvenanceEdgeRecord(analysis_id=analysis_id, **e))
    db.commit()
    return get_provenance_for_analysis(db, analysis_id)


def get_provenance_for_analysis(db: Session, analysis_id: str) -> dict:
    nodes = db.query(ProvenanceNodeRecord).filter_by(analysis_id=analysis_id).all()
    edges = db.query(ProvenanceEdgeRecord).filter_by(analysis_id=analysis_id).all()
    return {
        "nodes": [
            {"id": n.node_id, "label": n.label, "type": n.node_type, "metadata": n.node_metadata}
            for n in nodes
        ],
        "edges": [
            {
                "source": e.source_node_id,
                "target": e.target_node_id,
                "relationship": e.relationship,
                "confidence": e.confidence,
                "evidence": e.evidence,
            }
            for e in edges
        ],
    }


# ── Evidence ──────────────────────────────────────────────────────────────────

def create_evidence_item(db: Session, data: dict) -> EvidenceItemRecord:
    record = EvidenceItemRecord(**data)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def bulk_create_evidence(db: Session, items: List[dict]) -> List[EvidenceItemRecord]:
    records = [EvidenceItemRecord(**i) for i in items]
    db.add_all(records)
    db.commit()
    return records


def get_evidence_for_analysis(db: Session, analysis_id: str) -> List[EvidenceItemRecord]:
    return (
        db.query(EvidenceItemRecord)
        .filter_by(analysis_id=analysis_id)
        .order_by(EvidenceItemRecord.created_at)
        .all()
    )


# ── Reports ───────────────────────────────────────────────────────────────────

def upsert_report(db: Session, analysis_id: str, data: dict) -> ReportRecord:
    existing = db.query(ReportRecord).filter_by(analysis_id=analysis_id).first()
    if existing:
        for k, v in data.items():
            setattr(existing, k, v)
        db.commit()
        db.refresh(existing)
        return existing
    record = ReportRecord(analysis_id=analysis_id, **data)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_report_for_analysis(db: Session, analysis_id: str) -> Optional[ReportRecord]:
    return db.query(ReportRecord).filter_by(analysis_id=analysis_id).first()
