"""Report schemas."""
from pydantic import BaseModel
from typing import Optional, Any
from datetime import datetime


class ReportResponse(BaseModel):
    analysis_id: str
    json_report: dict[str, Any] | None = None
    markdown_report: str | None = None
    html_report: str | None = None
    pdf_url: str | None = None
    conclusion: str | None = None
    generated_at: datetime | None = None


class ChainOfCustody(BaseModel):
    case_id: str
    evidence_id: str
    file_hash: str
    acquisition_timestamp: datetime
    uploaded_filename: str
    storage_path: str
    processing_events: list[dict[str, Any]] = []
    model_versions: dict[str, str] = {}
