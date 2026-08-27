"""
Detection schemas.

Assessment values (use these instead of bare true/false):
  CLEAN         — analysis complete, no anomalies above threshold
  SUSPICIOUS    — significant forensic signals detected
  INCONCLUSIVE  — analysis ran but cannot make a determination
  UNAVAILABLE   — detector/model not configured or applicable
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class Assessment(str, Enum):
    CLEAN = "CLEAN"
    SUSPICIOUS = "SUSPICIOUS"
    INCONCLUSIVE = "INCONCLUSIVE"
    UNAVAILABLE = "UNAVAILABLE"


class DetectionRegion(BaseModel):
    """A localized region of forensic interest within a frame or audio segment."""
    type: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: Optional[list[float]] = None          # [x, y, width, height] in pixels
    mask_path: Optional[str] = None
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    start_timestamp: Optional[float] = None      # seconds
    end_timestamp: Optional[float] = None        # seconds
    extra: Optional[dict[str, Any]] = None


class DetectionResult(BaseModel):
    """Output from a single forensic detector."""
    detector: str
    media_type: str
    manipulation_detected: bool
    confidence: float = Field(ge=0.0, le=1.0)
    manipulation_type: str
    # Assessment is the preferred way to interpret results
    assessment: Assessment = Assessment.INCONCLUSIVE
    regions: list[DetectionRegion] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DetectionRequest(BaseModel):
    file_path: str
    media_type: str


class AnalysisRequest(BaseModel):
    """Request to run the full analysis pipeline on an already-uploaded media file."""
    media_id: str


class AnalysisResponse(BaseModel):
    """Response from a pipeline analysis request."""
    analysis_id: str
    media_id: str
    status: str  # QUEUED | PROCESSING | COMPLETED | FAILED | PARTIAL | INCONCLUSIVE
    assessment: Optional[str] = None
    overall_confidence: Optional[float] = None
    message: Optional[str] = None