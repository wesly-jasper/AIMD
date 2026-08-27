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
    String, Text, JSON, Enum as SAEnum
)
from sqlalchemy.orm import Mapped, mapped_column, relationship as sa_relationship

from app.db.database import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


# ── Media ─────────────────────────────────────────────────────────────────────

class MediaRecord(Base):
    """Represents an uploaded media file."""
    __tablename__ = "media"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    original_filename: Mapped[str] = mapped_column(String, nullable=False)
    stored_filename: Mapped[str] = mapped_column(String, nullable=False)
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    content_type: Mapped[str] = mapped_column(String, nullable=False)
    media_type: Mapped[str] = mapped_column(String, nullable=False)  # image/video/audio
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    # Relationships
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
    assessment: Mapped[Optional[str]] = mapped_column(String)  # CLEAN|SUSPICIOUS|INCONCLUSIVE
    overall_confidence: Mapped[Optional[float]] = mapped_column(Float)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Relationships
    media: Mapped["MediaRecord"] = sa_relationship(back_populates="analyses")
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
    # CLEAN | SUSPICIOUS | INCONCLUSIVE | UNAVAILABLE
    assessment: Mapped[str] = mapped_column(String, nullable=False, default="INCONCLUSIVE")
    manipulation_detected: Mapped[Optional[bool]] = mapped_column(Boolean)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
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
    discovered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    analysis: Mapped["AnalysisRecord"] = sa_relationship(back_populates="sources")


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
