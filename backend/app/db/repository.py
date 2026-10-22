"""
Repository layer -- all database CRUD operations.
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
    CaseRecord, ObservationRecord, FindingRecord, FindingObservationRecord,
    AssessmentRecord,
)

#: The only two bearings an observation may have on a finding.
FINDING_RELATIONS = frozenset({"SUPPORTS", "CONTRADICTS"})


# ── Cases ─────────────────────────────────────────────────────────────────────

def create_case(db: Session, data: dict) -> CaseRecord:
    record = CaseRecord(**data)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_case(db: Session, case_id: str) -> Optional[CaseRecord]:
    return db.get(CaseRecord, case_id)


def list_cases(db: Session, owner: str, limit: int = 50, offset: int = 0) -> List[CaseRecord]:
    return (
        db.query(CaseRecord)
        .filter(CaseRecord.created_by == owner)
        .order_by(CaseRecord.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


def count_media_by_case(db: Session, case_ids: List[str]) -> dict[str, int]:
    """Evidence count per case, in one query rather than one per case."""
    from sqlalchemy import func

    if not case_ids:
        return {}
    rows = (
        db.query(MediaRecord.case_id, func.count(MediaRecord.id))
        .filter(MediaRecord.case_id.in_(case_ids))
        .group_by(MediaRecord.case_id)
        .all()
    )
    return {case_id: count for case_id, count in rows}


# ── Ownership ─────────────────────────────────────────────────────────────────
#
# Every account sees only what it created. The owner of evidence is the account
# that uploaded it (``media.uploaded_by``); the owner of a folder is the account
# that opened it (``cases.created_by``); an analysis belongs to its media's owner.
# Rows with no recorded owner -- created before accounts existed -- belong to
# no one and are visible to no one.
#
# The getters return None for "not yours" exactly as for "does not exist", so a
# route cannot tell a caller which identifiers belong to someone else.

def get_owned_case(db: Session, case_id: str, owner: str) -> Optional[CaseRecord]:
    case = db.get(CaseRecord, case_id)
    return case if case is not None and case.created_by == owner else None


def get_owned_media(db: Session, media_id: str, owner: str) -> Optional[MediaRecord]:
    media = db.get(MediaRecord, media_id)
    return media if media is not None and media.uploaded_by == owner else None


def get_owned_analysis(db: Session, analysis_id: str, owner: str) -> Optional[AnalysisRecord]:
    analysis = db.get(AnalysisRecord, analysis_id)
    if analysis is None or analysis.media is None or analysis.media.uploaded_by != owner:
        return None
    return analysis


def list_owned_media(
    db: Session, owner: str, limit: int = 50, offset: int = 0, case_id: Optional[str] = None
) -> List[MediaRecord]:
    query = db.query(MediaRecord).filter(MediaRecord.uploaded_by == owner)
    if case_id is not None:
        query = query.filter(MediaRecord.case_id == case_id)
    return (
        query
        .order_by(MediaRecord.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


def count_uploads_since(db: Session, owner: str, since: datetime) -> int:
    return (
        db.query(MediaRecord)
        .filter(MediaRecord.uploaded_by == owner, MediaRecord.created_at >= since)
        .count()
    )


def list_owned_analyses(
    db: Session, owner: str, limit: int = 50, offset: int = 0, media_id: Optional[str] = None
) -> List[AnalysisRecord]:
    query = (
        db.query(AnalysisRecord)
        .join(MediaRecord, AnalysisRecord.media_id == MediaRecord.id)
        .filter(MediaRecord.uploaded_by == owner)
    )
    if media_id is not None:
        query = query.filter(AnalysisRecord.media_id == media_id)
    return (
        query.order_by(AnalysisRecord.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


def delete_media(db: Session, media: MediaRecord) -> None:
    """Delete a media record and, by cascade, everything derived from it."""
    db.delete(media)
    db.commit()


def list_media_for_case(db: Session, case_id: str) -> List[MediaRecord]:
    return (
        db.query(MediaRecord)
        .filter_by(case_id=case_id)
        .order_by(MediaRecord.created_at)
        .all()
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
    evidence_strength: Optional[str] = None,
    error_message: Optional[str] = None,
) -> Optional[AnalysisRecord]:
    record = db.get(AnalysisRecord, analysis_id)
    if not record:
        return None
    record.status = status
    if assessment is not None:
        record.assessment = assessment
    if evidence_strength is not None:
        record.evidence_strength = evidence_strength
    if error_message is not None:
        record.error_message = error_message
    if status in ("COMPLETED", "FAILED", "PARTIAL", "INCONCLUSIVE"):
        record.completed_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(record)
    return record


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



def update_analysis_run(db: Session, analysis_id: str, **fields) -> Optional[AnalysisRecord]:
    """Set run-provenance fields (software/detector versions, config, duration)."""
    allowed = {
        "requested_by", "software_version", "detector_set_version",
        "config_snapshot", "duration_ms",
    }
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"Not run-provenance fields: {sorted(unknown)}")
    record = db.get(AnalysisRecord, analysis_id)
    if not record:
        return None
    for k, v in fields.items():
        setattr(record, k, v)
    db.commit()
    db.refresh(record)
    return record


# ── Observations, findings, assessment ────────────────────────────────────────

def bulk_create_observations(db: Session, items: List[dict]) -> List[ObservationRecord]:
    records = [ObservationRecord(**i) for i in items]
    db.add_all(records)
    db.commit()
    return records


def get_observations_for_analysis(db: Session, analysis_id: str) -> List[ObservationRecord]:
    return (
        db.query(ObservationRecord)
        .filter_by(analysis_id=analysis_id)
        .order_by(ObservationRecord.created_at, ObservationRecord.id)
        .all()
    )


def create_finding(
    db: Session,
    analysis_id: str,
    statement: str,
    links: List[tuple[str, str]],
) -> FindingRecord:
    """
    A finding with its observation links, as ``(observation_id, relation)``.

    Every linked observation must belong to the same analysis: a finding
    supported by another analysis's measurement would be citing evidence about a
    different run, possibly of a different file.
    """
    bad = {rel for _, rel in links} - FINDING_RELATIONS
    if bad:
        raise ValueError(f"Unknown finding relation(s): {sorted(bad)}")

    obs_ids = {oid for oid, _ in links}
    if obs_ids:
        owned = {
            oid for (oid,) in db.query(ObservationRecord.id)
            .filter(ObservationRecord.id.in_(obs_ids))
            .filter(ObservationRecord.analysis_id == analysis_id)
        }
        foreign = obs_ids - owned
        if foreign:
            raise ValueError(
                f"{len(foreign)} linked observation(s) do not belong to analysis {analysis_id}"
            )

    finding = FindingRecord(analysis_id=analysis_id, statement=statement)
    db.add(finding)
    db.flush()
    for oid, rel in links:
        db.add(FindingObservationRecord(
            finding_id=finding.id, observation_id=oid, relation=rel
        ))
    db.commit()
    db.refresh(finding)
    return finding


def get_findings_for_analysis(db: Session, analysis_id: str) -> List[FindingRecord]:
    return (
        db.query(FindingRecord)
        .filter_by(analysis_id=analysis_id)
        .order_by(FindingRecord.created_at, FindingRecord.id)
        .all()
    )


def upsert_assessment(db: Session, analysis_id: str, data: dict) -> AssessmentRecord:
    existing = db.query(AssessmentRecord).filter_by(analysis_id=analysis_id).first()
    if existing:
        for k, v in data.items():
            setattr(existing, k, v)
        db.commit()
        db.refresh(existing)
        return existing
    record = AssessmentRecord(analysis_id=analysis_id, **data)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_assessment_for_analysis(db: Session, analysis_id: str) -> Optional[AssessmentRecord]:
    return db.query(AssessmentRecord).filter_by(analysis_id=analysis_id).first()


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


def get_all_fingerprints(db: Session, algorithm: str, owner: str) -> List[FingerprintRecord]:
    """
    Stored fingerprints of one algorithm, for similarity search -- one owner's only.

    This used to read every account's fingerprints, so a similarity match could
    hand one customer another customer's media id (and its place in the
    provenance graph). ``owner`` is required for that reason.
    """
    return (
        db.query(FingerprintRecord)
        .join(MediaRecord, FingerprintRecord.media_id == MediaRecord.id)
        .filter(FingerprintRecord.algorithm == algorithm, MediaRecord.uploaded_by == owner)
        .all()
    )


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
