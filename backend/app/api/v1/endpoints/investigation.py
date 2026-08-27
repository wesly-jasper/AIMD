from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from app.schemas.investigation import (
    GenerateFingerprintRequest,
    SimilaritySearchRequest,
    SourceTraceRequest
)
from app.services.similarity.internet_search_provider import BaselineInternetSearchProvider
from app.services.similarity.similarity_engine import SimilarityEngine
from app.services.provenance.source_tracer import SourceTracer
from app.services.evidence.evidence_service import EvidenceService
from app.services.evidence.report_generator import ReportGenerator
from app.services.provenance.graph import build_provenance_graph

router = APIRouter(
    tags=["Investigation"]
)

# Instantiate services
search_provider = BaselineInternetSearchProvider()
similarity_engine = SimilarityEngine(search_provider=search_provider)
source_tracer = SourceTracer(similarity_engine=similarity_engine)
evidence_service = EvidenceService(source_tracer=source_tracer)
report_generator = ReportGenerator()

# Mock DB for demonstration purposes of GET endpoints
# In reality, results would be stored in and retrieved from a database
mock_db = {}


from pathlib import Path
from app.services.fingerprinting.fingerprint import FingerprintService
from app.services.fingerprinting.embedding import BaselineEmbeddingProvider

fingerprint_service = FingerprintService()
embedding_provider = BaselineEmbeddingProvider()

@router.post("/fingerprint/generate")
def generate_fingerprint(request: GenerateFingerprintRequest):
    try:
        p = Path(request.file_path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {request.file_path}")
            
        fp = fingerprint_service.generate(p)
        emb = embedding_provider.generate_embedding(str(p))
        fp["embedding"] = emb
        
        return {
            "media_id": request.media_id,
            "fingerprints": fp
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/similarity/search")
def similarity_search(request: SimilaritySearchRequest):
    try:
        results = similarity_engine.search(request.file_path)
        return {"media_id": request.media_id, "matches": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/source/trace")
def trace_source(request: SourceTraceRequest):
    try:
        trace_result = source_tracer.trace_source(request.file_path)
        
        # Save to mock db
        mock_db[request.analysis_id] = {
            "media_id": request.media_id,
            "file_path": request.file_path,
            "trace_result": trace_result
        }
        
        return trace_result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/source/{analysis_id}")
def get_source(analysis_id: str):
    data = mock_db.get(analysis_id)
    if not data:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return data["trace_result"]


@router.get("/provenance/{analysis_id}")
def get_provenance(analysis_id: str):
    data = mock_db.get(analysis_id)
    if not data:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    graph = build_provenance_graph(data["media_id"], data["trace_result"])
    return graph.to_dict()


@router.get("/evidence/{analysis_id}")
def get_evidence(analysis_id: str):
    data = mock_db.get(analysis_id)
    if not data:
        raise HTTPException(status_code=404, detail="Analysis not found")
        
    evidence_package = evidence_service.generate_evidence_package(
        media_id=data["media_id"],
        file_path=data["file_path"],
        analysis_id=analysis_id
    )
    return evidence_package


@router.get("/report/{analysis_id}")
def get_report(analysis_id: str):
    data = mock_db.get(analysis_id)
    if not data:
        raise HTTPException(status_code=404, detail="Analysis not found")
        
    evidence_package = evidence_service.generate_evidence_package(
        media_id=data["media_id"],
        file_path=data["file_path"],
        analysis_id=analysis_id
    )
    
    return {
        "json": report_generator.generate_json_report(evidence_package),
        "markdown": report_generator.generate_markdown_report(evidence_package)
    }
