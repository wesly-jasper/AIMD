from typing import Dict, Any

class ReportGenerator:
    """
    Generates human-readable and machine-readable reports from an evidence package.
    """
    def generate_json_report(self, evidence_package: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns the raw machine-readable JSON structure.
        """
        return evidence_package

    def generate_markdown_report(self, evidence_package: Dict[str, Any]) -> str:
        """
        Generates a human-readable markdown report distinguishing 
        FACT, OBSERVATION, INFERENCE, and UNCERTAINTY.
        """
        case_info = evidence_package.get("case_information", {})
        media_info = evidence_package.get("media_evidence", {})
        
        md = f"# AIMD FORENSIC ANALYSIS REPORT\n\n"
        
        md += f"## Case Information\n"
        md += f"- **Case ID:** {case_info.get('case_id')}\n"
        md += f"- **Analysis ID:** {case_info.get('analysis_id')}\n"
        md += f"- **Date:** {case_info.get('creation_time')}\n\n"
        
        md += f"## Integrity Information [FACT]\n"
        fps = evidence_package.get("fingerprint_evidence", {})
        md += f"- **SHA-256:** {fps.get('sha256')}\n"
        md += f"- **pHash:** {fps.get('phash')}\n\n"
        
        md += f"## Detection Findings [OBSERVATION]\n"
        detections = evidence_package.get("detection_evidence", [])
        if detections:
            for det in detections:
                det_dict = det.dict() if hasattr(det, 'dict') else det
                md += f"- **Detector:** {det_dict.get('detector')} | **Type:** {det_dict.get('manipulation_type')} | **Confidence:** {det_dict.get('confidence')}\n"
        else:
            md += "No detections executed.\n"
        md += "\n"
        
        md += f"## Source Candidates & Provenance [INFERENCE]\n"
        source = evidence_package.get("source_evidence", {})
        if source:
            md += f"- **Earliest Known Occurrence:** {source.get('media_id')} (Similarity: {source.get('similarity')})\n"
            md += f"- **Timestamp:** {source.get('source_timestamp')}\n"
        else:
            md += "No external sources found.\n"
        md += "\n"
        
        md += f"## Limitations [UNCERTAINTY]\n"
        for lim in evidence_package.get("limitations", []):
            md += f"- {lim}\n"
            
        return md
