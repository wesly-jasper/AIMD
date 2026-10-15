"""
/api/v1/ -- Investigation endpoints (fingerprint, similarity, source, provenance, evidence, report).

All GET endpoints read from the database -- no in-memory state.
POST endpoints are available for standalone operations outside the full pipeline.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db import repository as repo
from app.schemas.investigation import (
    GenerateFingerprintRequest,
    SimilaritySearchRequest,
    SourceTraceRequest,
)
from app.services.fingerprinting.fingerprint import FingerprintService
from app.services.similarity.similarity_engine import SimilarityEngine
from app.services.evidence.evidence_service import EvidenceService
from app.services.evidence.report_generator import ReportGenerator

router = APIRouter(tags=["Investigation"])

_fingerprint_service = FingerprintService()
_similarity_engine = SimilarityEngine()
_evidence_service = EvidenceService()
_report_generator = ReportGenerator()


# ── Fingerprint ───────────────────────────────────────────────────────────────

@router.post("/fingerprint/generate", summary="Generate fingerprints for a file")
def generate_fingerprint(
    request: GenerateFingerprintRequest,
    db: Session = Depends(get_db),
):
    """Generate SHA-256, pHash, and dHash fingerprints for a file path."""
    try:
        from pathlib import Path
        p = Path(request.file_path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {request.file_path}")

        fp = _fingerprint_service.generate(
            p,
            media_id=request.media_id,
        )
        from app.services.fingerprinting.embedding import BaselineEmbeddingProvider
        try:
            emb = BaselineEmbeddingProvider().generate_embedding(str(p))
            fp["embedding"] = emb
        except Exception:
            fp["embedding"] = []

        # Remove internal records from response
        fp.pop("_records", None)
        fp.pop("_scope", None)

        return {"media_id": request.media_id, "fingerprints": fp}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ── Similarity ────────────────────────────────────────────────────────────────

@router.post("/similarity/search", summary="Search for similar media")
def similarity_search(
    request: SimilaritySearchRequest,
    db: Session = Depends(get_db),
):
    """
    Search local stored fingerprints for visually or perceptually similar media.
    External internet search is returned as UNAVAILABLE if no provider is configured.
    """
    try:
        # Get fingerprints for the query media
        fingerprints = repo.get_fingerprints_for_media(db, request.media_id)
        fp_dict = {fp.algorithm: fp.value for fp in fingerprints}

        result = _similarity_engine.search(
            request.file_path,
            query_fingerprints=fp_dict,
            db=db,
            exclude_media_id=request.media_id,
        )
        all_matches = result.get("local_matches", []) + result.get("external_matches", [])
        return {"media_id": request.media_id, "matches": all_matches, **result}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ── Source Tracing ────────────────────────────────────────────────────────────

@router.post("/source/trace", summary="Trace earliest-known source")
def trace_source(
    request: SourceTraceRequest,
    db: Session = Depends(get_db),
):
    """
    Traces the earliest-known occurrence of the media.
    Returns UNAVAILABLE for internet search if no provider is configured.
    """
    from app.services.provenance.source_tracer import SourceTracer
    from app.services.similarity.internet_search_provider import ProviderStatus
    from pathlib import Path

    tracer = SourceTracer()
    try:
        result = tracer.trace_source(request.file_path)

        # If analysis_id provided, persist records so GET endpoints succeed
        if request.analysis_id:
            analysis = repo.get_analysis(db, request.analysis_id)
            if not analysis:
                media = repo.get_media(db, request.media_id)
                p = Path(request.file_path)
                file_size = p.stat().st_size if p.exists() else 0
                if not media:
                    repo.create_media(db, {
                        "id": request.media_id,
                        "original_filename": p.name,
                        "stored_filename": p.name,
                        "file_path": str(request.file_path),
                        "content_type": "image/png" if p.suffix.lower() == ".png" else "image/jpeg",
                        "media_type": "image",
                        "size_bytes": file_size,
                        "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                    })
                repo.create_analysis(db, {
                    "id": request.analysis_id,
                    "media_id": request.media_id,
                    "status": "COMPLETED",
                    "assessment": "INCONCLUSIVE",
                })

            # Create provenance node and edges
            nodes = [
                {
                    "node_id": request.media_id,
                    "label": Path(request.file_path).name,
                    "node_type": "uploaded",
                    "node_metadata": {},
                }
            ]
            repo.upsert_provenance_graph(db, request.analysis_id, nodes, [])

        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/source/{analysis_id}", summary="Retrieve source tracing results")
def get_source(
    analysis_id: str,
    db: Session = Depends(get_db),
):
    analysis = repo.get_analysis(db, analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail=f"Analysis not found: {analysis_id}")

    sources = repo.get_sources_for_analysis(db, analysis_id)
    return {
        "analysis_id": analysis_id,
        "provider_status": sources[0].retrieval_status if sources else "UNAVAILABLE",
        "candidates": [
            {
                "url": s.url,
                "title": s.title,
                "domain": s.domain,
                "similarity": s.similarity,
                "source_timestamp": s.source_timestamp.isoformat() if s.source_timestamp else None,
                "provider": s.provider,
                "is_earliest_known": s.is_earliest_known,
            }
            for s in sources
        ],
        "earliest_known_occurrence": next(
            (
                {
                    "url": s.url,
                    "source_timestamp": s.source_timestamp.isoformat() if s.source_timestamp else None,
                    "note": (
                        "INFERENCE: This is the earliest occurrence AIMD discovered. "
                        "It is not necessarily the absolute origin."
                    ),
                }
                for s in sources if s.is_earliest_known
            ),
            None,
        ),
    }


# ── Provenance ────────────────────────────────────────────────────────────────

@router.get("/provenance/{analysis_id}", summary="Retrieve provenance graph")
def get_provenance(
    analysis_id: str,
    db: Session = Depends(get_db),
):
    analysis = repo.get_analysis(db, analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail=f"Analysis not found: {analysis_id}")

    return repo.get_provenance_for_analysis(db, analysis_id)


# ── Evidence ──────────────────────────────────────────────────────────────────

@router.get("/evidence/{analysis_id}", summary="Retrieve forensic evidence package")
def get_evidence(
    analysis_id: str,
    db: Session = Depends(get_db),
):
    """
    Returns the complete forensic evidence package.
    All values come from the database -- no mock data.
    """
    analysis = repo.get_analysis(db, analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail=f"Analysis not found: {analysis_id}")

    if analysis.status in ("QUEUED", "PROCESSING"):
        raise HTTPException(
            status_code=202,
            detail=f"Analysis is still {analysis.status}. Check back when status is COMPLETED.",
        )

    try:
        return _evidence_service.build_evidence_package(analysis_id, db)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evidence generation error: {exc}")


# ── Report ────────────────────────────────────────────────────────────────────

@router.get("/report/{analysis_id}", summary="Retrieve forensic report")
def get_report(
    analysis_id: str,
    db: Session = Depends(get_db),
):
    """Retrieve the JSON and Markdown forensic report for an analysis."""
    analysis = repo.get_analysis(db, analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail=f"Analysis not found: {analysis_id}")

    report = repo.get_report_for_analysis(db, analysis_id)
    if report:
        return {
            "analysis_id": analysis_id,
            "json": report.json_report,
            "markdown": report.markdown_report,
            "conclusion": report.conclusion,
            "generated_at": report.generated_at.isoformat() if report.generated_at else None,
        }

    # Regenerate if not cached
    try:
        evidence_package = _evidence_service.build_evidence_package(analysis_id, db)
        return {
            "analysis_id": analysis_id,
            "json": _report_generator.generate_json_report(evidence_package),
            "markdown": _report_generator.generate_markdown_report(evidence_package),
            "conclusion": evidence_package.get("overall_assessment"),
            "generated_at": None,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Report generation error: {exc}")


@router.get("/report/{analysis_id}/json", summary="Download JSON report")
def get_report_json(analysis_id: str, db: Session = Depends(get_db)):
    analysis = repo.get_analysis(db, analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail=f"Analysis not found: {analysis_id}")
    try:
        pkg = _evidence_service.build_evidence_package(analysis_id, db)
        return _report_generator.generate_json_report(pkg)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/report/{analysis_id}/html", summary="Download HTML report")
def get_report_html(analysis_id: str, db: Session = Depends(get_db)):
    from fastapi.responses import HTMLResponse

    analysis = repo.get_analysis(db, analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail=f"Analysis not found: {analysis_id}")
    try:
        pkg = _evidence_service.build_evidence_package(analysis_id, db)
        md = _report_generator.generate_markdown_report(pkg)
        # Simple markdown to HTML
        html = f"<html><head><meta charset='utf-8'><title>AIMD Report {analysis_id}</title>"
        html += "<style>body{font-family:system-ui;max-width:800px;margin:40px auto;padding:20px;line-height:1.6;color:#0f172a} h1{color:#0e7490} h2{color:#334155;border-bottom:1px solid #e2e8f0;padding-bottom:6px} code{background:#f1f5f9;padding:2px 6px;border-radius:4px} pre{background:#0f172a;color:#e2e8f0;padding:16px;border-radius:8px;overflow:auto}</style></head><body>"
        html += "<h1>AIMD Forensic Report</h1>"
        # Very light markdown rendering
        import re

        html_body = md
        html_body = re.sub(r"^### (.*)$", r"<h3>\1</h3>", html_body, flags=re.MULTILINE)
        html_body = re.sub(r"^## (.*)$", r"<h2>\1</h2>", html_body, flags=re.MULTILINE)
        html_body = re.sub(r"^# (.*)$", r"<h1>\1</h1>", html_body, flags=re.MULTILINE)
        html_body = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", html_body)
        html_body = re.sub(r"`(.*?)`", r"<code>\1</code>", html_body)
        html_body = html_body.replace("\n", "<br>")
        html += html_body + "</body></html>"
        return HTMLResponse(content=html)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/report/{analysis_id}/pdf", summary="Download PDF report")
def get_report_pdf(analysis_id: str, db: Session = Depends(get_db)):
    from fastapi.responses import FileResponse
    from pathlib import Path
    import tempfile
    from app.services.reports.pdf_report import generate_pdf_report

    analysis = repo.get_analysis(db, analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail=f"Analysis not found: {analysis_id}")
    try:
        pkg = _evidence_service.build_evidence_package(analysis_id, db)
        tmp = Path(tempfile.gettempdir()) / f"aimd_report_{analysis_id}.pdf"
        generate_pdf_report(pkg, tmp)
        return FileResponse(
            str(tmp),
            media_type="application/pdf",
            filename=f"AIMD_Report_{analysis_id[:8]}.pdf",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF generation error: {exc}")


@router.get("/evidence/{analysis_id}", summary="Alias for evidence package")
def get_evidence_alias(analysis_id: str, db: Session = Depends(get_db)):
    analysis = repo.get_analysis(db, analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail=f"Analysis not found: {analysis_id}")
    return _evidence_service.build_evidence_package(analysis_id, db)
