import os
import json
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

def run_full_pipeline_test(image_path: str):
    print("=" * 70)
    print("      AIMD FULL END-TO-END DIGITAL FORENSICS PIPELINE TEST")
    print("=" * 70)
    
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"[!] Test image not found at: {image_path}")
        print("[*] Generating a sample test image...")
        img = Image.new("RGB", (640, 480), color="darkblue")
        img.save(str(image_path))
        print(f"[+] Created sample image: {image_path}\n")
        
    print(f"[*] Target Media: {image_path.resolve()}\n")

    # =========================================================================
    # PHASE 1: PREPROCESSING & FINGERPRINTING
    # =========================================================================
    print(">>> [PHASE 1 & 4] Generating Cryptographic & Perceptual Fingerprints...")
    fp_service = FingerprintService()
    emb_provider = BaselineEmbeddingProvider()
    
    fingerprints = fp_service.generate(image_path)
    embeddings = emb_provider.generate_embedding(str(image_path))
    fingerprints["embedding_dimensions"] = len(embeddings)
    
    print(f"    - SHA-256 Hash : {fingerprints.get('sha256')}")
    print(f"    - pHash (Perceptual): {fingerprints.get('phash')}")
    print(f"    - Color Histogram Feature Vectors: {len(embeddings)} dimensions\n")

    # =========================================================================
    # PHASE 2 & 3: MULTIMODAL DETECTION & LOCALIZATION
    # =========================================================================
    print(">>> [PHASE 2 & 3] Running Multimodal Manipulation Detection & Localization...")
    detection_router = create_detection_router()
    analysis_service = MediaAnalysisService(router=detection_router)
    
    detection_results = analysis_service.analyze(str(image_path), "image")
    detections = detection_results.get("detections", [])
    
    print(f"    - Active Detectors Executed: {len(detections)}")
    for d in detections:
        d_dict = jsonable_encoder(d)
        print(f"      * Detector: {d_dict.get('detector'):<25} | Type: {d_dict.get('manipulation_type'):<10} | Confidence: {d_dict.get('confidence'):.2f} | Regions Found: {len(d_dict.get('regions', []))}")
        for r in d_dict.get('regions', []):
            print(f"        -> Region: {r.get('type')} at bbox: {r.get('bbox')}")
    print()

    # =========================================================================
    # PHASE 4: SIMILARITY SEARCH (INTERNET SCALE)
    # =========================================================================
    print(">>> [PHASE 4] Performing Internet-Scale Similarity Search...")
    search_provider = BaselineInternetSearchProvider()
    similarity_engine = SimilarityEngine(search_provider=search_provider)
    
    matches = similarity_engine.search(str(image_path))
    print(f"    - Found {len(matches)} potential visual match(es) across indexed sources:")
    for m in matches:
        print(f"      * Candidate URL: {m.get('url')} (Similarity: {m.get('similarity')*100:.1f}%)")
    print()

    # =========================================================================
    # PHASE 5: SOURCE TRACING
    # =========================================================================
    print(">>> [PHASE 5] Executing Source Tracing & Temporal Attribution...")
    source_tracer = SourceTracer(similarity_engine=similarity_engine)
    trace_result = source_tracer.trace_source(str(image_path))
    
    earliest = trace_result.get("earliest_known_occurrence")
    if earliest:
        print(f"    - [+] Earliest Known Occurrence Identified:")
        print(f"          Source ID   : {earliest.get('media_id')}")
        print(f"          Timestamp   : {earliest.get('source_timestamp')}")
        print(f"          Similarity  : {earliest.get('similarity')*100:.1f}%")
        print(f"          Attribution : {earliest.get('matching_method')}")
    else:
        print("    - No prior online occurrence discovered.")
    print()

    # =========================================================================
    # PHASE 6: PROVENANCE GRAPH RECONSTRUCTION
    # =========================================================================
    print(">>> [PHASE 6] Reconstructing Lineage & Provenance Graph...")
    graph = build_provenance_graph(uploaded_media_id=image_path.name, source_trace_results=trace_result)
    graph_dict = graph.to_dict()
    
    print(f"    - Graph Nodes ({len(graph_dict['nodes'])}):")
    for node in graph_dict['nodes']:
        print(f"      * [{node['id']}] - Label: '{node['label']}'")
        
    print(f"    - Graph Relationships / Edges ({len(graph_dict['edges'])}):")
    for edge in graph_dict['edges']:
        print(f"      * ({edge['source']}) ---[{edge['relationship']} (conf: {edge['confidence']:.2f})]---> ({edge['target']})")
    print()

    # =========================================================================
    # PHASE 7: FORENSIC EVIDENCE PACKAGE & REPORT GENERATION
    # =========================================================================
    print(">>> [PHASE 7] Compiling Forensic Evidence Package & Admissible Report...")
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
    
    print("=" * 70)
    print("                  FINAL FORENSIC REPORT")
    print("=" * 70)
    print(markdown_report)
    print("=" * 70)
    print("[+] FULL PIPELINE TEST COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    target_file = os.path.join(os.path.dirname(__file__), "test.jpg")
    if not os.path.exists(target_file):
        target_file = os.path.join(os.path.dirname(__file__), "test.jpg")
    run_full_pipeline_test(target_file)
