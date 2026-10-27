"""
The routes the result page reads evidence files from: a detector's heatmap
artefacts and the stored report in JSON, Markdown and HTML.

Both serve files, so both are guarded here against the ways file routes go
wrong: another user's analysis, a path that leaves the artefact root, bytes
replaced after analysis, and report text that would become markup.
"""
import hashlib

import pytest

from app.core.config import settings
from app.db import repository as repo
from app.db.database import SessionLocal
from tests.integration.test_ai_generation_branch import client
from tests.integration.test_investigation_agent import seed_analysis

PNG = b"\x89PNG\r\n\x1a\n" + b"heatmap-bytes"
OTHER = "someone-else@example.invalid"


@pytest.fixture
def artefact_root(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "artefacts_dir", tmp_path)
    (tmp_path / "trufor").mkdir()
    (tmp_path / "trufor" / "abc_localization.png").write_bytes(PNG)
    (tmp_path / "outside.png").write_bytes(PNG)
    return tmp_path


def _detection_with(analysis_id: str, artefacts: dict) -> str:
    with SessionLocal() as db:
        det = repo.create_detection(db, {
            "analysis_id": analysis_id, "detector": "trufor", "media_type": "image",
            "manipulation_type": "splicing", "availability": "AVAILABLE",
            "execution_status": "COMPLETED", "assessment": "SUSPICIOUS",
            "manipulation_detected": True, "confidence": 0.9,
            "score_semantics": "RAW_MODEL_SCORE",
            "detector_metadata": {"artefacts": artefacts},
        })
        return det.id


def _artefact(path: str, data: bytes = PNG) -> dict:
    return {"path": path, "sha256": hashlib.sha256(data).hexdigest(), "media_type": "image/png"}


def _url(analysis_id: str, detection_id: str, kind: str) -> str:
    return f"/api/v1/analysis/{analysis_id}/detections/{detection_id}/artefacts/{kind}"


# ── Artefacts ─────────────────────────────────────────────────────────────────

def test_a_recorded_heatmap_is_served_with_its_hash(artefact_root):
    analysis_id = seed_analysis()
    det = _detection_with(analysis_id, {"localization_map": _artefact("trufor/abc_localization.png")})
    r = client.get(_url(analysis_id, det, "localization_map"))
    assert r.status_code == 200 and r.content == PNG
    assert r.headers["content-type"] == "image/png"
    assert r.headers["x-content-sha256"] == hashlib.sha256(PNG).hexdigest()


def test_replaced_bytes_are_refused(artefact_root):
    analysis_id = seed_analysis()
    det = _detection_with(analysis_id, {"overlay": _artefact("trufor/abc_localization.png", b"other")})
    r = client.get(_url(analysis_id, det, "overlay"))
    assert r.status_code == 409 and "no longer match" in r.text


@pytest.mark.parametrize("path", ["../outside.png", "trufor/../../outside.png", "/etc/passwd", "C:/Windows/win.ini"])
def test_a_path_outside_the_artefact_root_is_never_served(artefact_root, path):
    analysis_id = seed_analysis()
    det = _detection_with(analysis_id, {"overlay": {"path": path}})
    assert client.get(_url(analysis_id, det, "overlay")).status_code == 404


def test_unknown_kinds_detections_and_owners_are_not_found(artefact_root):
    mine = seed_analysis()
    det = _detection_with(mine, {"localization_map": _artefact("trufor/abc_localization.png")})
    assert client.get(_url(mine, det, "reliability_map")).status_code == 404
    assert client.get(_url(mine, "no-such-detection", "localization_map")).status_code == 404

    theirs = seed_analysis(owner=OTHER)
    their_det = _detection_with(theirs, {"localization_map": _artefact("trufor/abc_localization.png")})
    assert client.get(_url(theirs, their_det, "localization_map")).status_code == 404
    # Nor can my analysis id reach their detection.
    assert client.get(_url(mine, their_det, "localization_map")).status_code == 404


# ── Reports ───────────────────────────────────────────────────────────────────

MARKDOWN = "\n".join([
    "# AIMD FORENSIC ANALYSIS REPORT",
    "",
    "**Generated:** 2026-10-23  ",
    "**Case Reference:** N/A",
    "",
    "---",
    "",
    "## 2. File & Metadata Evidence [FACT]",
    "",
    "- **Original Filename:** `<script>alert(1)</script>.jpg`",
    "- Title: <img src=x onerror=alert(1)>",
    "",
    "> **NOTICE [OBSERVATION]:** not *proof*.",
    "",
    "| Algorithm | Value |",
    "| :--- | :--- |",
    "| **PHASH** | `ff00` |",
])


def _with_report(analysis_id: str) -> None:
    with SessionLocal() as db:
        repo.upsert_report(db, analysis_id, {
            "json_report": {"overall_assessment": "INCONCLUSIVE", "case_information": {"analysis_id": analysis_id}},
            "markdown_report": MARKDOWN,
            "conclusion": "INCONCLUSIVE",
        })


def test_the_report_in_each_format():
    analysis_id = seed_analysis()
    _with_report(analysis_id)
    base = f"/api/v1/analysis/{analysis_id}/report"

    as_json = client.get(base)
    assert as_json.status_code == 200
    assert as_json.json()["case_information"]["analysis_id"] == analysis_id

    as_md = client.get(base, params={"format": "md"})
    assert as_md.headers["content-type"].startswith("text/markdown") and as_md.text == MARKDOWN

    as_html = client.get(base, params={"format": "html"})
    assert as_html.headers["content-type"].startswith("text/html")
    assert as_html.headers["content-security-policy"] == "default-src 'none'; style-src 'unsafe-inline'"
    page = as_html.text
    assert "<h1>AIMD FORENSIC ANALYSIS REPORT</h1>" in page
    assert "<table>" in page and "<th>Algorithm</th>" in page and "<strong>PHASH</strong>" in page
    assert "<blockquote>" in page and "<em>proof</em>" in page and "<hr>" in page
    assert "@media print" in page


def test_report_text_never_becomes_markup():
    analysis_id = seed_analysis()
    _with_report(analysis_id)
    page = client.get(f"/api/v1/analysis/{analysis_id}/report", params={"format": "html"}).text
    assert "<script>" not in page and "<img" not in page
    assert "<code>&lt;script&gt;alert(1)&lt;/script&gt;.jpg</code>" in page
    assert "&lt;img src=x onerror=alert(1)&gt;" in page


def test_report_guards():
    analysis_id = seed_analysis()
    base = f"/api/v1/analysis/{analysis_id}/report"
    assert client.get(base).status_code == 404                       # not written yet
    _with_report(analysis_id)
    assert client.get(base, params={"format": "pdf"}).status_code == 422

    theirs = seed_analysis(owner=OTHER)
    _with_report(theirs)
    assert client.get(f"/api/v1/analysis/{theirs}/report").status_code == 404


# ── The investigation in the served report ────────────────────────────────────

def _with_investigation(analysis_id: str, summary: str, flags=None) -> None:
    with SessionLocal() as db:
        inv = repo.create_investigation(db, {"analysis_id": analysis_id, "requested_by": "pytest"})
        repo.update_investigation(
            db, inv.id, status="COMPLETED", model="nvidia/nemotron-test",
            summary=summary, flags=flags or [],
            tool_calls=[{"id": "T1", "name": "get_analysis_overview", "category": "OBSERVATION",
                         "arguments": {}, "result": {"assessment": "INCONCLUSIVE"}, "latency_ms": 5}],
        )


def test_no_investigation_leaves_the_report_as_written():
    analysis_id = seed_analysis()
    _with_report(analysis_id)
    base = f"/api/v1/analysis/{analysis_id}/report"
    assert client.get(base).json()["investigation"] is None
    assert client.get(base, params={"format": "md"}).text == MARKDOWN


def test_a_completed_investigation_is_in_every_format_as_inference():
    analysis_id = seed_analysis()
    _with_report(analysis_id)
    _with_investigation(analysis_id, "The forgery model reported nothing [T1].\n<b>not markup</b>",
                        flags=[{"type": "unsupported_number", "value": "256", "sentence": "x"}])
    base = f"/api/v1/analysis/{analysis_id}/report"

    inv = client.get(base).json()["investigation"]
    assert inv["status"] == "COMPLETED" and inv["summary"]["category"] == "INFERENCE"
    assert inv["grounded"] is False
    assert "result" not in inv["tool_calls"][0]  # the trace's results stay out of the report

    md = client.get(base, params={"format": "md"}).text
    assert md.startswith(MARKDOWN)
    assert "## Investigation [INFERENCE]" in md and "nvidia/nemotron-test" in md
    assert "1 unsupported claim(s) removed: a number no tool returned" in md
    assert "The forgery model reported nothing [T1]." in md
    assert "- `T1` `get_analysis_overview` -- OBSERVATION" in md

    page = client.get(base, params={"format": "html"}).text
    assert "<h2>Investigation [INFERENCE]</h2>" in page
    assert "<b>" not in page and "&lt;b&gt;not markup&lt;/b&gt;" in page


def test_an_unfinished_investigation_is_named_without_a_summary():
    analysis_id = seed_analysis()
    _with_report(analysis_id)
    with SessionLocal() as db:
        repo.create_investigation(db, {"analysis_id": analysis_id, "requested_by": "pytest"})
    md = client.get(f"/api/v1/analysis/{analysis_id}/report", params={"format": "md"}).text
    assert "The latest investigation of this analysis is still running" in md
