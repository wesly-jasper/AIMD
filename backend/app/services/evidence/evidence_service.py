from typing import Dict, Any, List
from datetime import datetime, timezone

class EvidenceService:
    def __init__(self, media_service=None, detection_service=None, fingerprinting_service=None, source_tracer=None):
        self.media_service = media_service
        self.detection_service = detection_service
        self.fingerprinting_service = fingerprinting_service
        self.source_tracer = source_tracer

    def generate_evidence_package(self, media_id: str, file_path: str, analysis_id: str) -> Dict[str, Any]:
        """
        Aggregates results from all phases into a comprehensive forensic evidence package.
        """
        # 1. Media Info
        # Normally retrieved from DB using media_id.
        media_info = {
            "media_id": media_id,
            "filename": file_path.split("/")[-1], # simplified
            "processing_time": datetime.now(timezone.utc).isoformat()
        }

        # 2. Fingerprints & Hashes
        # Normally generated via fingerprinting service
        fingerprints = {
            "sha256": "mock_sha256_hash",
            "phash": "mock_phash",
            "embeddings": []
        }

        # 3. Detection Results
        detection_results = []
        if self.detection_service:
            # We mock the actual call for structure
            try:
                result = self.detection_service.analyze(file_path)
                detection_results = result.get("detections", [])
            except Exception:
                pass

        # 4. Source & Provenance
        source_evidence = {}
        provenance = {}
        if self.source_tracer:
            try:
                trace_res = self.source_tracer.trace_source(file_path)
                source_evidence = trace_res.get("earliest_known_occurrence", {})
                from app.services.provenance.graph import build_provenance_graph
                graph = build_provenance_graph(media_id, trace_res)
                provenance = graph.to_dict()
            except Exception:
                pass

        # 5. Limitations
        limitations = [
            "Baseline detection models used; requires Deep Learning models for production-grade confidence.",
            "Internet source tracing mocked for testing; requires actual API integrations."
        ]

        # Assemble Evidence Package
        return {
            "case_information": {
                "case_id": f"CASE-{analysis_id}",
                "media_id": media_id,
                "analysis_id": analysis_id,
                "creation_time": datetime.now(timezone.utc).isoformat(),
            },
            "media_evidence": media_info,
            "detection_evidence": detection_results,
            "fingerprint_evidence": fingerprints,
            "source_evidence": source_evidence,
            "provenance_evidence": provenance,
            "limitations": limitations
        }
