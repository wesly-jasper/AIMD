"""
Forensic Report Generator — human-readable Markdown and structured JSON.

Structures reports into clear forensic sections with strict evidential labeling:
  [FACT]        - Cryptographic hashes, file sizes, EXIF data, container metadata.
  [OBSERVATION] - Algorithmically extracted signals (ELA, FFT, noise, spectral, face).
  [INFERENCE]   - Provenance relationships, earliest-known source estimates.
  [UNCERTAINTY] - Known detector limitations, missing models, inconclusive signals.
"""
import json
from datetime import datetime, timezone
from typing import Any, Dict


class ReportGenerator:
    """
    Generates human-readable Markdown and machine-readable JSON forensic reports.
    """

    def generate_json_report(self, evidence_package: Dict[str, Any]) -> Dict[str, Any]:
        """Returns the machine-readable evidence package."""
        return evidence_package

    def generate_markdown_report(self, evidence_package: Dict[str, Any]) -> str:
        """
        Generates a structured forensic report in GitHub-flavored Markdown.
        """
        case = evidence_package.get("case_information", {})
        media = evidence_package.get("media_evidence", {})
        fps = evidence_package.get("fingerprint_evidence", {})
        detections = evidence_package.get("detection_evidence", [])
        similarity = evidence_package.get("similarity_evidence", {})
        source = evidence_package.get("source_evidence", {})
        provenance = evidence_package.get("provenance_evidence", {})
        classified = evidence_package.get("classified_evidence", {})
        limitations = evidence_package.get("limitations", [])
        verdict = evidence_package.get("overall_assessment", "INCONCLUSIVE")
        confidence = evidence_package.get("overall_confidence")

        lines = [
            "# AIMD FORENSIC ANALYSIS REPORT",
            "",
            f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
            f"**Platform:** AIMD (AI Media Detection & Digital Forensics Platform)  ",
            f"**Case Reference:** {case.get('case_id', 'N/A')}",
            "",
            "---",
            "",
            "## 1. Executive Summary & Verdict",
            "",
            f"- **Forensic Assessment:** **`{verdict}`**",
            f"- **Composite Confidence:** `{confidence:.2%}`" if confidence is not None else "- **Composite Confidence:** `N/A`",
            f"- **Investigation Status:** `{case.get('analysis_status', 'COMPLETED')}`",
            f"- **Analysis ID:** `{case.get('analysis_id')}`",
            "",
        ]

        if verdict == "SUSPICIOUS":
            lines.append(
                "> **NOTICE [OBSERVATION]:** Forensic anomalies were identified above standard thresholds. "
                "Review the detection signals and localized regions below for specific findings."
            )
        elif verdict == "CLEAN":
            lines.append(
                "> **NOTICE [OBSERVATION]:** No definitive manipulation signals exceeded detection thresholds. "
                "Note: A 'CLEAN' assessment indicates absence of detected manipulation, not definitive proof of authenticity."
            )
        else:
            lines.append(
                "> **NOTICE [UNCERTAINTY]:** Analysis is INCONCLUSIVE. Available signals do not provide sufficient "
                "confidence for a conclusive assessment."
            )

        lines.extend([
            "",
            "---",
            "",
            "## 2. File & Metadata Evidence [FACT]",
            "",
            f"- **Original Filename:** `{media.get('original_filename', 'N/A')}`",
            f"- **Media Type:** `{media.get('media_type', 'N/A')}`",
            f"- **Content MIME:** `{media.get('content_type', 'N/A')}`",
            f"- **File Size:** `{media.get('size_bytes', 0):,} bytes`",
            f"- **SHA-256 (Cryptographic Hash):** `{media.get('sha256', 'N/A')}`",
            "",
        ])

        meta = media.get("metadata", {})
        if meta:
            lines.append("### Technical Metadata")
            if meta.get("width") and meta.get("height"):
                lines.append(f"- **Dimensions:** `{meta.get('width')} × {meta.get('height')} px`")
            if meta.get("duration_seconds"):
                lines.append(f"- **Duration:** `{meta.get('duration_seconds')} s`")
            if meta.get("fps"):
                lines.append(f"- **Frame Rate:** `{meta.get('fps')} FPS`")
            if meta.get("codec"):
                lines.append(f"- **Codec:** `{meta.get('codec')}`")
            if meta.get("exif_data"):
                lines.append(f"- **EXIF Attributes Found:** `{len(meta.get('exif_data'))}`")
            lines.append("")

        lines.extend([
            "---",
            "",
            "## 3. Fingerprints & Perceptual Hashes",
            "",
            "| Algorithm | Scope | Value | Category |",
            "| :--- | :--- | :--- | :--- |",
        ])

        if isinstance(fps, dict):
            for algo, fp_info in fps.items():
                if isinstance(fp_info, dict):
                    val = fp_info.get("value", "")
                    val_disp = f"`{val[:32]}...`" if len(val) > 32 else f"`{val}`"
                    cat = fp_info.get("evidence_category", "OBSERVATION")
                    scope = fp_info.get("scope", "full")
                    lines.append(f"| **{algo.upper()}** | {scope} | {val_disp} | [{cat}] |")
                elif isinstance(fp_info, str):
                    val_disp = f"`{fp_info[:32]}...`" if len(fp_info) > 32 else f"`{fp_info}`"
                    lines.append(f"| **{algo.upper()}** | full | {val_disp} | [OBSERVATION] |")
        lines.append("")

        lines.extend([
            "---",
            "",
            "## 4. Multimodal Detection Findings [OBSERVATION]",
            "",
        ])

        if detections:
            for det in detections:
                det_name = det.get("detector", "Unknown")
                m_type = det.get("manipulation_type", "unknown")
                ass = det.get("assessment", "INCONCLUSIVE")
                conf = det.get("confidence", 0.0)
                regions = det.get("regions", [])

                lines.extend([
                    f"### Detector: `{det_name}`",
                    f"- **Target:** `{m_type}` | **Assessment:** `{ass}` | **Confidence:** `{conf:.2%}`",
                ])

                meta_det = det.get("metadata", {})
                if meta_det.get("note"):
                    lines.append(f"- **Notes:** {meta_det.get('note')}")

                if regions:
                    lines.append(f"- **Localized Regions of Interest ({len(regions)}):**")
                    for r in regions[:5]:
                        r_type = r.get("type", "region")
                        r_conf = r.get("confidence", 0.0)
                        bbox = r.get("bbox")
                        t_start = r.get("start_timestamp")
                        loc_str = f"bbox={bbox}" if bbox else (f"time={t_start}s" if t_start is not None else "")
                        lines.append(f"  - `{r_type}` (confidence: {r_conf:.2%}{', ' + loc_str if loc_str else ''})")
                lines.append("")
        else:
            lines.append("No active detection modules produced findings.\n")

        lines.extend([
            "---",
            "",
            "## 5. Source Tracing & Similarity [INFERENCE]",
            "",
        ])

        # Similarity
        loc_matches = similarity.get("local_matches", [])
        ext_matches = similarity.get("external_matches", [])
        lines.append(f"- **Local Stored Matches:** `{len(loc_matches)}`")
        lines.append(f"- **External Source Matches:** `{len(ext_matches)}`")
        if loc_matches:
            for m in loc_matches[:3]:
                lines.append(f"  - Stored Media ID `{m.get('source_media_id')}` (similarity: `{m.get('similarity'):.2%}`, method: `{m.get('method')}`)")
        lines.append("")

        # Earliest known source
        earliest = source.get("earliest_known_occurrence")
        if earliest:
            lines.extend([
                "### Earliest-Known Discovered Occurrence",
                f"- **URL / Source:** `{earliest.get('url', 'N/A')}`",
                f"- **Discovered Timestamp:** `{earliest.get('source_timestamp', 'Unknown')}`",
                f"- **Match Similarity:** `{earliest.get('similarity', 0):.2%}`",
                f"- **Provider:** `{earliest.get('provider', 'N/A')}`",
                "> *Note:* This represents the earliest occurrence discovered by AIMD, not absolute cryptographic provenance.",
                "",
            ])
        else:
            lines.append(f"- **Internet Search Status:** `{source.get('provider_status', 'UNAVAILABLE')}` (No external origin discovered)\n")

        lines.extend([
            "---",
            "",
            "## 6. Provenance Graph [INFERENCE]",
            "",
            f"- **Nodes:** `{len(provenance.get('nodes', []))}` | **Edges:** `{len(provenance.get('edges', []))}`",
        ])
        for edge in provenance.get("edges", [])[:5]:
            lines.append(f"- `{edge.get('source')}` → `{edge.get('relationship')}` → `{edge.get('target')}` (confidence: `{edge.get('confidence', 0.5):.2%}`)")
        lines.append("")

        lines.extend([
            "---",
            "",
            "## 7. Categorized Forensic Evidence Log",
            "",
        ])

        for cat in ["FACT", "OBSERVATION", "INFERENCE", "UNCERTAINTY"]:
            items = classified.get(cat, [])
            if items:
                lines.append(f"### Category: [{cat}]")
                for item in items:
                    desc = item.get("description", "")
                    src_st = item.get("source_stage", "")
                    lines.append(f"- {desc} *(Source: {src_st})*")
                lines.append("")

        lines.extend([
            "---",
            "",
            "## 8. Epistemic Limitations & Uncertainty Analysis [UNCERTAINTY]",
            "",
        ])

        if limitations:
            for lim in limitations:
                lines.append(f"- {lim}")
        else:
            lines.append("- No specific operational limitations recorded.")

        lines.extend([
            "",
            "---",
            "",
            "*(End of Official Forensic Analysis Report — AIMD Platform)*",
        ])

        return "\n".join(lines)
