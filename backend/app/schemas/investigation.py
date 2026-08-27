from pydantic import BaseModel
from typing import Optional

class GenerateFingerprintRequest(BaseModel):
    file_path: str
    media_id: str

class SimilaritySearchRequest(BaseModel):
    file_path: str
    media_id: str

class SourceTraceRequest(BaseModel):
    file_path: str
    media_id: str
    analysis_id: str
