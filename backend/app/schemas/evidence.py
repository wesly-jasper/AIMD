"""Evidence schemas -- structured forensic evidence."""
from pydantic import BaseModel, Field
from typing import Optional, Any
from datetime import datetime


class EvidenceItem(BaseModel):
    evidence_id: str
    category: str  # FACT | OBSERVATION | INFERENCE | UNCERTAINTY
    type: str | None = None
    finding: str
    severity: str = "INFO"  # INFO | LOW | MEDIUM | HIGH | CRITICAL
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    explanation: str | None = None
    source: str | None = None
    model: str | None = None
    frame_index: int | None = None
    timestamp_seconds: float | None = None
    bbox: list[float] | None = None
    extra_data: dict[str, Any] | None = None
    created_at: datetime | None = None


class EvidencePackage(BaseModel):
    analysis_id: str
    media_id: str
    overall_assessment: str
    overall_confidence: float | None = None
    evidence: list[EvidenceItem]
    evidence_count: int
    risk_score: float | None = None
    generated_at: datetime | None = None
