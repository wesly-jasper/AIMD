import os
import json
import logging
from pathlib import Path
from PIL import Image
from fastapi.encoders import jsonable_encoder

from app.services.ingestion.media_ingestion import MediaIngestionService
from app.services.detection.detection_container import create_detection_router
from app.services.analysis.media_analysis_service import MediaAnalysisService
from app.services.fingerprinting.fingerprint import FingerprintService
from app.services.fingerprinting.embedding import BaselineEmbeddingProvider
from app.services.similarity.internet_search_provider import BaselineInternetSearchProvider
from app.services.similarity.similarity_engine import SimilarityEngine
from app.services.provenance.source_tracer import SourceTracer
from app.services.provenance.graph import build_provenance_graph
from app.services.evidence.evidence_service import EvidenceService
from app.services.evidence.report_generator import ReportGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_full_pipeline_test(image_path: str):
    image_path = Path(image_path)
    if not image_path.exists():
        logger.info("Creating test image at %s", image_path)
        img = Image.new("RGB", (640, 480), color="darkblue")
        img.save(str(image_path))
    else:
        logger.info("Using existing test image at %s", image_path)

    # =========================================================================
    # PHASE 1: PREPROCESSING & FINGERPRINTING
    # =========================================================================
    logger.info("Starting Phase 1: Preprocessing & Fingerprinting")
    fp_service = FingerprintService()
    emb_provider = BaselineEmbeddingProvider()

    fingerprints = fp_service.generate(image_path)
    embeddings = emb_provider.generate_embedding(str(image_path))
    fingerprints["embedding_dimensions"] = len(embeddings)

    logger.info("Phase 1 complete: Generated %d fingerprints and %d-dimensional embeddings",
                len(fingerprints), len(embeddings))

    # =========================================================================
    # PHASE 2 & 3: MULTIMODAL DETECTION & LOCALIZATION
    # =========================================================================
    logger.info("Starting Phase 2 & 3: Multimodal Detection & Localization")
    detection_router = create_detection_router()
    analysis_service = MediaAnalysisService(router=detection_router)

    detection_results = analysis_service.analyze(str(image_path), "image")
    detections = detection_results.get("detections", [])

    logger.info("Phase 2 & 3 complete: Found %d detections", len(detections))

    # =========================================================================
    # PHASE 4: SIMILARITY SEARCH (INTERNET SCALE)
    # =========================================================================
    logger.info("Starting Phase 4: Similarity Search (Internet Scale)")
    search_provider = BaselineInternetSearchProvider()
    similarity_engine = SimilarityEngine(search_provider=search_provider)

    matches = similarity_engine.search(str(image_path))
    logger.info("Phase 4 complete: Found %d similar matches", len(matches))

    # =========================================================================
    # PHASE 5: SOURCE TRACING
    # =========================================================================
    logger.info("Starting Phase 5: Source Tracing")
    source_tracer = SourceTracer(similarity_engine=similarity_engine)
    trace_result = source_tracer.trace_source(str(image_path))

    earliest = trace_result.get("earliest_known_occurrence")
    logger.info("Phase 5 complete: Traced source to %s", earliest if earliest else "unknown")

    # =========================================================================
    # PHASE 6: PROVENANCE GRAPH RECONSTRUCTION
    # =========================================================================
    logger.info("Starting Phase 6: Provenance Graph Reconstruction")
    graph = build_provenance_graph(uploaded_media_id=image_path.name, source_trace_results=trace_result)
    graph_dict = graph.to_dict()
    logger.info("Phase 6 complete: Built provenance graph with %d nodes",
                len(graph_dict.get('nodes', [])) if isinstance(graph_dict, dict) else 0)

    # =========================================================================
    # PHASE 7: FORENSIC EVIDENCE PACKAGE & REPORT GENERATION
    # =========================================================================
    logger.info("Starting Phase 7: Forensic Evidence Package & Report Generation")
    evidence_service = EvidenceService(source_tracer=source_tracer)
    report_generator = ReportGenerator()

    evidence_package = evidence_service.generate_evidence_package(
        media_id=image_path.name,
        file_path=str(image_path),
        analysis_id="AIMD-DEMO-2026"
    )

    # Enrich with active detection results
    evidence_package["detection_evidence"] = jsonable_encoder(detections)
    evidence_package["fingerprint_evidence"] = fingerprints

    markdown_report = report_generator.generate_markdown_report(evidence_package)
    logger.info("Phase 7 complete: Generated evidence report (%d chars)", len(markdown_report))

    # =========================================================================
    # ASSERTIONS: Validate pipeline results
    # =========================================================================
    logger.info("Running validation assertions...")

    # Assert that we generated fingerprints
    assert isinstance(fingerprints, dict), "Fingerprints should be a dictionary"
    assert len(fingerprints) > 0, "Should have generated some fingerprints"
    assert "embedding_dimensions" in fingerprints, "Fingerprints should include embedding dimensions"

    # Assert that we got embeddings
    assert isinstance(embeddings, list), "Embeddings should be a list"
    assert len(embeddings) > 0, "Should have generated embeddings"
    assert fingerprints["embedding_dimensions"] == len(embeddings), "Embedding dimensions should match"

    # Assert that detection results are structured correctly
    assert isinstance(detection_results, dict), "Detection results should be a dictionary"
    assert "detections" in detection_results, "Detection results should contain detections key"
    assert isinstance(detections, list), "Detections should be a list"

    # Assert that we got similarity matches
    assert isinstance(matches, list), "Matches should be a list"

    # Assert that we got trace results
    assert isinstance(trace_result, dict), "Trace result should be a dictionary"

    # Assert that we got a provenance graph
    assert graph is not None, "Provenance graph should not be None"
    assert hasattr(graph, 'to_dict'), "Graph should have to_dict method"

    # Assert that we got evidence package
    assert isinstance(evidence_package, dict), "Evidence package should be a dictionary"
    assert "detection_evidence" in evidence_package, "Evidence package should contain detection evidence"
    assert "fingerprint_evidence" in evidence_package, "Evidence package should contain fingerprint evidence"

    # Assert that we got a markdown report
    assert isinstance(markdown_report, str), "Markdown report should be a string"
    assert len(markdown_report) > 0, "Markdown report should not be empty"
    assert "# Forensic Evidence Report" in markdown_report, "Report should contain title"

    logger.info("All validation assertions passed!")

if __name__ == "__main__":
    target_file = os.path.join(os.path.dirname(__file__), "test.jpg")
    run_full_pipeline_test(target_file)
