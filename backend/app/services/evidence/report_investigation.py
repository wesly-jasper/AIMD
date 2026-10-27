"""
The investigation's part of a served report.

The forensic report is written once, when the analysis finishes, and is never
rewritten: it records what the analysis found. An investigation runs later, on
request, so it was never in the report at all. It is added when the report is
served -- the latest investigation, labelled INFERENCE with its grounding and
the tool calls it cites -- so every download carries it without the stored
forensic record changing after the fact.
"""
from __future__ import annotations

from typing import Optional

from app.services.agent.nemotron_agent import SUMMARY_NOTE, serialise_investigation

_FLAG = {
    "unknown_citation": "a citation of a tool call that does not exist",
    "unsupported_url": "a link no tool returned",
    "unsupported_number": "a number no tool returned",
    "unsupported_timestamp": "a date no tool returned",
}


def investigation_json(record) -> Optional[dict]:
    """The investigation as the report's JSON carries it, or None if there is none."""
    if record is None:
        return None
    view = serialise_investigation(record)
    # The trace's full tool results are already the report's own evidence.
    view["tool_calls"] = [
        {k: c.get(k) for k in ("id", "name", "arguments", "category", "error", "latency_ms")}
        for c in view["tool_calls"]
    ]
    return view


def investigation_markdown(record) -> str:
    """The investigation section appended to the Markdown (and so HTML) report."""
    if record is None:
        return ""
    lines = ["", "---", "", "## Investigation [INFERENCE]", ""]
    if record.status != "COMPLETED" or not record.summary:
        state = {"RUNNING": "is still running", "FAILED": "failed",
                 "UNAVAILABLE": "was unavailable"}.get(record.status, record.status.lower())
        lines.append(f"The latest investigation of this analysis {state}; it produced no summary.")
        return "\n".join(lines) + "\n"

    view = serialise_investigation(record)
    removed = [f for f in view["flags"] if f.get("type") != "uncited"]
    when = view["completed_at"] or view["created_at"] or "unknown time"
    lines += [
        f"- **Model:** `{record.model or 'unknown'}`",
        f"- **Completed:** `{when}`",
        "- **Grounding:** "
        + ("every claim traced to a tool call" if view["grounded"]
           else f"{len(removed)} unsupported claim(s) removed: "
                + "; ".join(_FLAG.get(f["type"], f["type"]) for f in removed)),
        "",
        f"> **NOTICE [INFERENCE]:** {SUMMARY_NOTE}",
        "",
    ]
    lines += [line.rstrip() for line in record.summary.strip().splitlines()]
    calls = view["tool_calls"]
    if calls:
        lines += ["", "### Tool calls cited", ""]
        for c in calls:
            label = {"FACT": "FACT", "OBSERVATION": "OBSERVATION"}.get(c.get("category"), "")
            error = " (error)" if c.get("error") else ""
            lines.append(f"- `{c.get('id')}` `{c.get('name')}`{' -- ' + label if label else ''}{error}")
    return "\n".join(lines) + "\n"
