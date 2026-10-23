"""
SQLAlchemy ORM models for AIMD.

Every investigation record is linked through analysis_id so the
full investigation can be reconstructed after server restart.
"""
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey, Integer,
    String, Text, JSON, UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship as sa_relationship

from app.db.database import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


# ── Users ─────────────────────────────────────────────────────────────────────

class UserRecord(Base):
    """
    An account. Previously held only in process memory, so a user registered at
    runtime vanished on restart and existed on one worker only.
    """
    __tablename__ = "users"

    username: Mapped[str] = mapped_column(String(150), primary_key=True)
    email: Mapped[Optional[str]] = mapped_column(String(254))
    hashed_password: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)  # admin | investigator
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


# ── Cases ─────────────────────────────────────────────────────────────────────

class CaseRecord(Base):
    """An investigation grouping evidence items under one reference."""
    __tablename__ = "cases"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String, nullable=False, default="OPEN")  # OPEN | CLOSED
    created_by: Mapped[Optional[str]] = mapped_column(String(150))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    media: Mapped[list["MediaRecord"]] = sa_relationship(back_populates="case")


# ── Media ─────────────────────────────────────────────────────────────────────

class MediaRecord(Base):
    """Represents an uploaded media file."""
    __tablename__ = "media"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    # Nullable because evidence ingested before cases existed has none. RESTRICT,
    # not CASCADE: deleting a case must never silently delete its evidence.
    case_id: Mapped[Optional[str]] = mapped_column(
        String, ForeignKey("cases.id", ondelete="RESTRICT"), index=True
    )
    uploaded_by: Mapped[Optional[str]] = mapped_column(String(150))
    original_filename: Mapped[str] = mapped_column(String, nullable=False)
    stored_filename: Mapped[str] = mapped_column(String, nullable=False)
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    content_type: Mapped[str] = mapped_column(String, nullable=False)
    media_type: Mapped[str] = mapped_column(String, nullable=False)  # image/video/audio
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    # Relationships
    case: Mapped[Optional["CaseRecord"]] = sa_relationship(back_populates="media")
    analyses: Mapped[list["AnalysisRecord"]] = sa_relationship(
        back_populates="media", cascade="all, delete-orphan"
    )
    metadata_record: Mapped[Optional["MediaMetadataRecord"]] = sa_relationship(
        back_populates="media", cascade="all, delete-orphan", uselist=False
    )
    frames: Mapped[list["FrameRecord"]] = sa_relationship(
        back_populates="media", cascade="all, delete-orphan"
    )
    fingerprints: Mapped[list["FingerprintRecord"]] = sa_relationship(
        back_populates="media", cascade="all, delete-orphan"
    )


# ── Media Metadata ────────────────────────────────────────────────────────────

class MediaMetadataRecord(Base):
    """Stores extracted technical metadata about a media file."""
    __tablename__ = "media_metadata"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    media_id: Mapped[str] = mapped_column(
        String, ForeignKey("media.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    # Common
    format: Mapped[Optional[str]] = mapped_column(String)
    width: Mapped[Optional[int]] = mapped_column(Integer)
    height: Mapped[Optional[int]] = mapped_column(Integer)
    # Video / Audio
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float)
    fps: Mapped[Optional[float]] = mapped_column(Float)
    frame_count: Mapped[Optional[int]] = mapped_column(Integer)
    codec: Mapped[Optional[str]] = mapped_column(String)
    bitrate: Mapped[Optional[int]] = mapped_column(Integer)
    # Audio
    sample_rate: Mapped[Optional[int]] = mapped_column(Integer)
    channels: Mapped[Optional[int]] = mapped_column(Integer)
    # Image EXIF
    exif_data: Mapped[Optional[dict]] = mapped_column(JSON)
    # Extra raw metadata
    raw_metadata: Mapped[Optional[dict]] = mapped_column(JSON)
    extracted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    media: Mapped["MediaRecord"] = sa_relationship(back_populates="metadata_record")


# ── Analysis ──────────────────────────────────────────────────────────────────

class AnalysisRecord(Base):
    """Represents a forensic analysis investigation tied to a media file."""
    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    media_id: Mapped[str] = mapped_column(
        String, ForeignKey("media.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[str] = mapped_column(
        String, nullable=False, default="QUEUED"
    )  # QUEUED|PROCESSING|COMPLETED|FAILED|PARTIAL|INCONCLUSIVE
    # NO_SIGNIFICANT_ANOMALIES_DETECTED | SUSPICIOUS | LIKELY_MANIPULATED
    # | LIKELY_AI_GENERATED | INCONCLUSIVE
    assessment: Mapped[Optional[str]] = mapped_column(String)

    # Ordinal support for the assessment: NONE|WEAK|MODERATE|STRONG|CONTRADICTORY.
    # Not a probability and not comparable across analyses as a magnitude.
    evidence_strength: Mapped[Optional[str]] = mapped_column(String)

    # ``overall_confidence`` -- a weighted mean of incomparable detector scores --
    # was dropped by migration b7e1f0c2d9a4. See that migration for why.
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # ── Run provenance ────────────────────────────────────────────────────────
    # What produced this result. Without these, two analyses of the same file
    # that disagree cannot be explained: the detectors, their versions or the
    # configuration may have changed between them. All are NULL on runs recorded
    # before they existed -- deliberately not backfilled, since the values in
    # force at the time are unknown.
    requested_by: Mapped[Optional[str]] = mapped_column(String(150))
    software_version: Mapped[Optional[str]] = mapped_column(String)
    # sha256 over the sorted (detector, version) pairs that were constructed.
    detector_set_version: Mapped[Optional[str]] = mapped_column(String(64))
    # The analysis-relevant settings, never credentials -- see
    # AnalysisPipeline._config_snapshot for the allow-list.
    config_snapshot: Mapped[Optional[dict]] = mapped_column(JSON)
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer)
    # {connector: {status, reason, hits, verified, duration_ms}}: which sources
    # were searched, so "no source found" is never confused with "not searched".
    source_searches: Mapped[Optional[dict]] = mapped_column(JSON)

    # Relationships
    media: Mapped["MediaRecord"] = sa_relationship(back_populates="analyses")
    observations: Mapped[list["ObservationRecord"]] = sa_relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )
    findings: Mapped[list["FindingRecord"]] = sa_relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )
    assessment_record: Mapped[Optional["AssessmentRecord"]] = sa_relationship(
        back_populates="analysis", cascade="all, delete-orphan", uselist=False
    )
    detections: Mapped[list["DetectionRecord"]] = sa_relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )
    fingerprints: Mapped[list["FingerprintRecord"]] = sa_relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )
    similarity_matches: Mapped[list["SimilarityMatchRecord"]] = sa_relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )
    sources: Mapped[list["SourceRecord"]] = sa_relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )
    provenance_nodes: Mapped[list["ProvenanceNodeRecord"]] = sa_relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )
    provenance_edges: Mapped[list["ProvenanceEdgeRecord"]] = sa_relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )
    evidence_items: Mapped[list["EvidenceItemRecord"]] = sa_relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )
    report: Mapped[Optional["ReportRecord"]] = sa_relationship(
        back_populates="analysis", cascade="all, delete-orphan", uselist=False
    )
    investigations: Mapped[list["InvestigationRecord"]] = sa_relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )


# ── Frames ────────────────────────────────────────────────────────────────────

class FrameRecord(Base):
    """A single extracted frame from a video."""
    __tablename__ = "frames"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    media_id: Mapped[str] = mapped_column(
        String, ForeignKey("media.id", ondelete="CASCADE"), nullable=False
    )
    frame_index: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    width: Mapped[Optional[int]] = mapped_column(Integer)
    height: Mapped[Optional[int]] = mapped_column(Integer)
    is_keyframe: Mapped[bool] = mapped_column(Boolean, default=False)
    scene_score: Mapped[Optional[float]] = mapped_column(Float)  # Frame difference score
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    media: Mapped["MediaRecord"] = sa_relationship(back_populates="frames")


# ── Detections ────────────────────────────────────────────────────────────────

class DetectionRecord(Base):
    """Output from a single detector run on a media file or frame."""
    __tablename__ = "detections"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    analysis_id: Mapped[str] = mapped_column(
        String, ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False
    )
    detector: Mapped[str] = mapped_column(String, nullable=False)
    detector_version: Mapped[Optional[str]] = mapped_column(String)
    media_type: Mapped[str] = mapped_column(String, nullable=False)
    manipulation_type: Mapped[str] = mapped_column(String, nullable=False)
    # Availability and execution are INDEPENDENT facts and are stored separately.
    # A detector may be AVAILABLE and FAILED, or COMPLETED and INCONCLUSIVE.
    # Collapsing them loses the difference between "we lack this capability",
    # "our capability is broken", and "it ran and found nothing" -- which have
    # different remedies and different meaning for the evidence.
    availability: Mapped[str] = mapped_column(
        String, nullable=False, default="AVAILABLE"
    )  # AVAILABLE | UNAVAILABLE
    execution_status: Mapped[str] = mapped_column(
        String, nullable=False, default="COMPLETED"
    )  # COMPLETED | INCONCLUSIVE | FAILED | SKIPPED

    # Why the detector was unavailable / failed / skipped.
    status_reason: Mapped[Optional[str]] = mapped_column(Text)

    # DEPRECATED detector-level label, derived from the two fields above.
    # Retained until consumers are migrated off it.
    assessment: Mapped[str] = mapped_column(String, nullable=False, default="INCONCLUSIVE")
    manipulation_detected: Mapped[Optional[bool]] = mapped_column(Boolean)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # How to read `confidence`. Without this the number is uninterpretable: a raw
    # model score, a calibrated probability and a heuristic severity are not
    # comparable and must not be presented on a shared scale.
    score_semantics: Mapped[str] = mapped_column(
        String, nullable=False, default="HEURISTIC_SEVERITY"
    )
    detector_metadata: Mapped[Optional[dict]] = mapped_column(JSON)
    # For video: suspicious segments
    suspicious_segments: Mapped[Optional[list]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    analysis: Mapped["AnalysisRecord"] = sa_relationship(back_populates="detections")
    regions: Mapped[list["DetectionRegionRecord"]] = sa_relationship(
        back_populates="detection", cascade="all, delete-orphan"
    )


class DetectionRegionRecord(Base):
    """A localized region of interest from a detection result."""
    __tablename__ = "detection_regions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    detection_id: Mapped[str] = mapped_column(
        String, ForeignKey("detections.id", ondelete="CASCADE"), nullable=False
    )
    region_type: Mapped[str] = mapped_column(String, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    bbox: Mapped[Optional[list]] = mapped_column(JSON)  # [x, y, w, h]
    mask_path: Mapped[Optional[str]] = mapped_column(String)
    start_frame: Mapped[Optional[int]] = mapped_column(Integer)
    end_frame: Mapped[Optional[int]] = mapped_column(Integer)
    start_timestamp: Mapped[Optional[float]] = mapped_column(Float)
    end_timestamp: Mapped[Optional[float]] = mapped_column(Float)
    extra_data: Mapped[Optional[dict]] = mapped_column(JSON)

    detection: Mapped["DetectionRecord"] = sa_relationship(back_populates="regions")


# ── Fingerprints ──────────────────────────────────────────────────────────────

class FingerprintRecord(Base):
    """A computed fingerprint/hash for a media file or frame."""
    __tablename__ = "fingerprints"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    media_id: Mapped[str] = mapped_column(
        String, ForeignKey("media.id", ondelete="CASCADE"), nullable=False
    )
    analysis_id: Mapped[Optional[str]] = mapped_column(
        String, ForeignKey("analyses.id", ondelete="CASCADE")
    )
    algorithm: Mapped[str] = mapped_column(String, nullable=False)  # sha256|phash|dhash|embedding
    algorithm_version: Mapped[Optional[str]] = mapped_column(String)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    scope: Mapped[str] = mapped_column(String, default="full")  # full|frame|keyframe|audio
    frame_index: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    media: Mapped["MediaRecord"] = sa_relationship(back_populates="fingerprints")
    analysis: Mapped[Optional["AnalysisRecord"]] = sa_relationship(back_populates="fingerprints")


# ── Similarity Matches ────────────────────────────────────────────────────────

class SimilarityMatchRecord(Base):
    """A ranked similarity match against another stored media or external source."""
    __tablename__ = "similarity_matches"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    analysis_id: Mapped[str] = mapped_column(
        String, ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False
    )
    match_type: Mapped[str] = mapped_column(String, nullable=False)  # local|external
    source_media_id: Mapped[Optional[str]] = mapped_column(String)  # local media ID if local
    external_url: Mapped[Optional[str]] = mapped_column(String)      # URL if external
    similarity: Mapped[float] = mapped_column(Float, nullable=False)
    method: Mapped[str] = mapped_column(String, nullable=False)      # sha256|phash|dhash|embedding
    algorithm: Mapped[Optional[str]] = mapped_column(String)
    rank: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    # VERIFIED_MATCH | SIMILAR_UNVERIFIED | NO_MATCH; NULL for matches recorded
    # before verification existed (they were never verified).
    verification_status: Mapped[Optional[str]] = mapped_column(String)
    # The measurements and thresholds behind the status, and why a gate failed.
    verification: Mapped[Optional[dict]] = mapped_column(JSON)
    discovered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    analysis: Mapped["AnalysisRecord"] = sa_relationship(back_populates="similarity_matches")


# ── Sources ───────────────────────────────────────────────────────────────────

class SourceRecord(Base):
    """A discovered source/occurrence of the media."""
    __tablename__ = "sources"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    analysis_id: Mapped[str] = mapped_column(
        String, ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False
    )
    url: Mapped[Optional[str]] = mapped_column(String)
    title: Mapped[Optional[str]] = mapped_column(String)
    domain: Mapped[Optional[str]] = mapped_column(String)
    source_timestamp: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    similarity: Mapped[Optional[float]] = mapped_column(Float)
    provider: Mapped[str] = mapped_column(String, nullable=False)  # search provider name
    retrieval_status: Mapped[str] = mapped_column(String, default="FOUND")
    is_earliest_known: Mapped[bool] = mapped_column(Boolean, default=False)
    # VERIFIED_MATCH | SIMILAR_UNVERIFIED, and the measurements behind it. NULL
    # for sources recorded before verification existed (never verified).
    verification_status: Mapped[Optional[str]] = mapped_column(String)
    verification: Mapped[Optional[dict]] = mapped_column(JSON)
    # What the source itself states (uploader, licence, file URL, ...), cited.
    citation: Mapped[Optional[dict]] = mapped_column(JSON)
    discovered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    analysis: Mapped["AnalysisRecord"] = sa_relationship(back_populates="sources")


# ── Reference corpus ──────────────────────────────────────────────────────────
#
# Public images indexed so an upload can be traced to where it was published.
# Not anyone's media: no owner, never returned by the owner-scoped media search,
# and every row carries the citation (page URL, uploader, upload time, licence)
# that a source claim must rest on.

class ReferenceImageRecord(Base):
    __tablename__ = "reference_images"
    __table_args__ = (
        UniqueConstraint("corpus", "external_id", name="uq_reference_images_corpus_external_id"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    corpus: Mapped[str] = mapped_column(String, nullable=False)  # e.g. wikimedia_commons
    # The corpus's own identifier, e.g. the Commons title "File:Example.jpg".
    external_id: Mapped[str] = mapped_column(String, nullable=False)
    page_url: Mapped[str] = mapped_column(String, nullable=False)
    file_url: Mapped[str] = mapped_column(String, nullable=False)
    # What was actually downloaded and fingerprinted; may be a scaled rendition.
    download_url: Mapped[str] = mapped_column(String, nullable=False)
    uploader: Mapped[Optional[str]] = mapped_column(String)
    # When the current version of the file -- the pixels indexed here -- was
    # uploaded. Not when the photograph was taken, and not its first appearance.
    uploaded_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    licence: Mapped[Optional[str]] = mapped_column(String)
    licence_url: Mapped[Optional[str]] = mapped_column(String)
    artist: Mapped[Optional[str]] = mapped_column(Text)
    # The corpus's hash of the original file, to notice when it is replaced.
    original_sha1: Mapped[Optional[str]] = mapped_column(String(40))
    original_width: Mapped[Optional[int]] = mapped_column(Integer)
    original_height: Mapped[Optional[int]] = mapped_column(Integer)
    original_mime: Mapped[Optional[str]] = mapped_column(String)
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    width: Mapped[Optional[int]] = mapped_column(Integer)
    height: Mapped[Optional[int]] = mapped_column(Integer)
    categories: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    # Descriptive metadata as the corpus states it (DateTimeOriginal, credit, ...):
    # claims by the uploader, not facts established here.
    source_metadata: Mapped[Optional[dict]] = mapped_column(JSON)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    fingerprints: Mapped[list["ReferenceFingerprintRecord"]] = sa_relationship(
        back_populates="reference", cascade="all, delete-orphan"
    )


class ReferenceFingerprintRecord(Base):
    """A fingerprint of a reference image, in the same form as ``fingerprints``."""
    __tablename__ = "reference_fingerprints"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    reference_id: Mapped[str] = mapped_column(
        String, ForeignKey("reference_images.id", ondelete="CASCADE"), nullable=False, index=True
    )
    algorithm: Mapped[str] = mapped_column(String, nullable=False, index=True)
    algorithm_version: Mapped[Optional[str]] = mapped_column(String)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    reference: Mapped["ReferenceImageRecord"] = sa_relationship(back_populates="fingerprints")


# ── Provenance ────────────────────────────────────────────────────────────────

class ProvenanceNodeRecord(Base):
    __tablename__ = "provenance_nodes"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    analysis_id: Mapped[str] = mapped_column(
        String, ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False
    )
    node_id: Mapped[str] = mapped_column(String, nullable=False)  # logical node ID
    label: Mapped[str] = mapped_column(String, nullable=False)
    node_type: Mapped[str] = mapped_column(String, nullable=False)  # uploaded|source|derivative
    node_metadata: Mapped[Optional[dict]] = mapped_column(JSON)

    analysis: Mapped["AnalysisRecord"] = sa_relationship(back_populates="provenance_nodes")


class ProvenanceEdgeRecord(Base):
    __tablename__ = "provenance_edges"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    analysis_id: Mapped[str] = mapped_column(
        String, ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False
    )
    source_node_id: Mapped[str] = mapped_column(String, nullable=False)
    target_node_id: Mapped[str] = mapped_column(String, nullable=False)
    relationship: Mapped[str] = mapped_column(String, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    evidence: Mapped[Optional[str]] = mapped_column(Text)

    analysis: Mapped["AnalysisRecord"] = sa_relationship(back_populates="provenance_edges")


# ── Evidence ──────────────────────────────────────────────────────────────────

class EvidenceItemRecord(Base):
    """A single forensic evidence item classified by FACT/OBSERVATION/INFERENCE/UNCERTAINTY."""
    __tablename__ = "evidence_items"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    analysis_id: Mapped[str] = mapped_column(
        String, ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False
    )
    category: Mapped[str] = mapped_column(
        String, nullable=False
    )  # FACT|OBSERVATION|INFERENCE|UNCERTAINTY
    description: Mapped[str] = mapped_column(Text, nullable=False)
    source_stage: Mapped[Optional[str]] = mapped_column(String)   # pipeline stage
    detector: Mapped[Optional[str]] = mapped_column(String)
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    frame_index: Mapped[Optional[int]] = mapped_column(Integer)
    timestamp_seconds: Mapped[Optional[float]] = mapped_column(Float)
    region_bbox: Mapped[Optional[list]] = mapped_column(JSON)
    extra_data: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    analysis: Mapped["AnalysisRecord"] = sa_relationship(back_populates="evidence_items")


# ── Report ────────────────────────────────────────────────────────────────────

class ReportRecord(Base):
    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    analysis_id: Mapped[str] = mapped_column(
        String, ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    json_report: Mapped[Optional[dict]] = mapped_column(JSON)
    markdown_report: Mapped[Optional[str]] = mapped_column(Text)
    conclusion: Mapped[Optional[str]] = mapped_column(Text)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    analysis: Mapped["AnalysisRecord"] = sa_relationship(back_populates="report")


# ── Observations, findings, assessment ────────────────────────────────────────
#
# These three were previously conflated in ``evidence_items`` (free text with a
# category) and two columns on ``analyses``. Separating them is what lets a
# conclusion be traced back to the measurements it rests on:
#
#   detection  -- a detector was invoked; did it run, and how?
#   observation-- one thing a detector that ran actually measured
#   finding    -- a statement, linked to the observations that SUPPORT it and,
#                 just as explicitly, to the ones that CONTRADICT it
#   assessment -- the analysis-level category, with the rule and rationale that
#                 produced it and the coverage gaps it was produced despite
#
# A detector that was UNAVAILABLE, FAILED or SKIPPED produces no observation.
# It is a coverage gap, recorded on the assessment, never a negative result.

class ObservationRecord(Base):
    __tablename__ = "observations"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    analysis_id: Mapped[str] = mapped_column(
        String, ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False, index=True
    )
    detection_id: Mapped[str] = mapped_column(
        String, ForeignKey("detections.id", ondelete="CASCADE"), nullable=False
    )
    detector: Mapped[str] = mapped_column(String, nullable=False)
    # Detectors sharing a physical basis are not independent corroboration.
    signal_family: Mapped[str] = mapped_column(String, nullable=False)
    # ANOMALY_OBSERVED | NO_ANOMALY_OBSERVED | INDETERMINATE
    polarity: Mapped[str] = mapped_column(String, nullable=False)
    score: Mapped[Optional[float]] = mapped_column(Float)
    score_semantics: Mapped[str] = mapped_column(String, nullable=False)
    localised: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    analysis: Mapped["AnalysisRecord"] = sa_relationship(back_populates="observations")


class FindingRecord(Base):
    __tablename__ = "findings"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    analysis_id: Mapped[str] = mapped_column(
        String, ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False, index=True
    )
    statement: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    analysis: Mapped["AnalysisRecord"] = sa_relationship(back_populates="findings")
    links: Mapped[list["FindingObservationRecord"]] = sa_relationship(
        back_populates="finding", cascade="all, delete-orphan"
    )


class FindingObservationRecord(Base):
    """An observation's bearing on a finding: SUPPORTS or CONTRADICTS."""
    __tablename__ = "finding_observations"

    finding_id: Mapped[str] = mapped_column(
        String, ForeignKey("findings.id", ondelete="CASCADE"), primary_key=True
    )
    observation_id: Mapped[str] = mapped_column(
        String, ForeignKey("observations.id", ondelete="CASCADE"), primary_key=True
    )
    relation: Mapped[str] = mapped_column(String, nullable=False)  # SUPPORTS | CONTRADICTS

    finding: Mapped["FindingRecord"] = sa_relationship(back_populates="links")


class AssessmentRecord(Base):
    __tablename__ = "assessments"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    analysis_id: Mapped[str] = mapped_column(
        String, ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    category: Mapped[str] = mapped_column(String, nullable=False)
    evidence_strength: Mapped[str] = mapped_column(String, nullable=False)
    # Stable identifier of the decision rule that fired, so the logic behind a
    # stored assessment can be found even after the wording of `rationale` changes.
    rule: Mapped[str] = mapped_column(String, nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    # Detection ids that produced no observation (UNAVAILABLE / FAILED / SKIPPED).
    coverage_gaps: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    # The AI-generation branch, reported beside the manipulation verdict above and
    # never blended into it. NULL for assessments made before the branch existed.
    ai_generation: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    analysis: Mapped["AnalysisRecord"] = sa_relationship(back_populates="assessment_record")


# ── Investigation agent ───────────────────────────────────────────────────────

class InvestigationRecord(Base):
    """
    One run of the Nemotron investigation agent over a finished analysis.

    Kept apart from the analysis's own results, which it reads and never
    changes. The summary is model output and is labelled INFERENCE; the tool
    trace is what AIMD's own tools returned and is labelled FACT or OBSERVATION
    per call. ``flags`` lists what the grounding guard removed from the summary
    and why, so a reader can see what the model claimed that no tool supported.
    """
    __tablename__ = "investigations"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    analysis_id: Mapped[str] = mapped_column(
        String, ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False, index=True
    )
    requested_by: Mapped[str] = mapped_column(String(150), nullable=False, index=True)
    status: Mapped[str] = mapped_column(
        String, nullable=False, default="RUNNING"
    )  # RUNNING|COMPLETED|UNAVAILABLE|FAILED
    model: Mapped[Optional[str]] = mapped_column(String)
    summary: Mapped[Optional[str]] = mapped_column(Text)
    summary_category: Mapped[str] = mapped_column(String, nullable=False, default="INFERENCE")
    # [{sentence, tool_call_ids}]: which tool results each sentence cites.
    citations: Mapped[Optional[list]] = mapped_column(JSON)
    flags: Mapped[Optional[list]] = mapped_column(JSON)
    # [{id, name, arguments, category, result | error, latency_ms}]
    tool_calls: Mapped[Optional[list]] = mapped_column(JSON)
    steps: Mapped[Optional[int]] = mapped_column(Integer)
    # {prompt_tokens, completion_tokens, total_tokens, calls: [{latency_ms, ...}]}
    usage: Mapped[Optional[dict]] = mapped_column(JSON)
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    analysis: Mapped["AnalysisRecord"] = sa_relationship(back_populates="investigations")


# ── Audit trail ───────────────────────────────────────────────────────────────

class AuditEventRecord(Base):
    """
    One entry in a hash-linked, append-only audit trail.

    ``event_hash`` is sha256 over the canonical serialisation of this event's
    fields *including* ``previous_event_hash``, so altering, deleting or
    reordering any earlier event breaks every hash after it. See
    app.services.audit.audit_log for the canonical form and the verifier.

    The subject ids are plain indexed strings, deliberately NOT foreign keys: an
    audit record must outlive the thing it describes, and a cascading FK would
    delete the record of a deletion along with its subject.

    ``occurred_at`` is stored as the exact ISO-8601 string that was hashed. A
    DateTime column would not round-trip byte-for-byte on every backend (SQLite
    drops the timezone), and verification would then fail on untampered rows.
    """
    __tablename__ = "audit_events"
    __table_args__ = (
        # Two writers that both read the same chain head would otherwise fork the
        # chain silently. With these, the second insert fails and is retried.
        UniqueConstraint("sequence", name="uq_audit_events_sequence"),
        UniqueConstraint("previous_event_hash", name="uq_audit_events_previous_hash"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    occurred_at: Mapped[str] = mapped_column(String(40), nullable=False)
    actor: Mapped[str] = mapped_column(String(150), nullable=False)
    action: Mapped[str] = mapped_column(String, nullable=False)
    case_id: Mapped[Optional[str]] = mapped_column(String, index=True)
    media_id: Mapped[Optional[str]] = mapped_column(String, index=True)
    analysis_id: Mapped[Optional[str]] = mapped_column(String, index=True)
    details: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    previous_event_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    event_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
