"""Provenance schemas."""
from pydantic import BaseModel
from typing import Optional, Any
from datetime import datetime


class ProvenanceNode(BaseModel):
    node_id: str
    label: str
    node_type: str  # uploaded | source | derivative
    metadata: dict[str, Any] | None = None


class ProvenanceEdge(BaseModel):
    source_node_id: str
    target_node_id: str
    relationship: str
    confidence: float
    evidence: str | None = None


class ProvenanceGraph(BaseModel):
    nodes: list[ProvenanceNode]
    edges: list[ProvenanceEdge]


class SourceRecordSchema(BaseModel):
    url: str | None = None
    title: str | None = None
    domain: str | None = None
    source_timestamp: datetime | None = None
    similarity: float | None = None
    provider: str
    is_earliest_known: bool = False
