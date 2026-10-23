"""
The investigation agent's tools: the only things the model can do.

Eight tools, all read-only views of ONE analysis and the media it ran on,
bound when the investigation starts. No tool takes an analysis or media id, a
file path, a shell command or a free URL: the model cannot point a tool at
anything else. Ownership is re-checked on every call with the same
``get_owned_analysis`` the API uses, so a tool can never return what the API
would refuse the caller.

The one tool that reaches the network, ``verify_source_match(url)``, fetches
through the SSRF guard (``GuardedFetcher`` via ``UrlVerificationSource``), only
for a URL that already appeared in an earlier tool result or that the user
supplied, and at most ``agent_max_url_checks`` times per investigation. Its
outcome goes into the investigation's trace; it never adds a source to the
analysis.

Each result is labelled FACT (what AIMD has on record: metadata, sources and
their citations, the timeline rebuilt from them) or OBSERVATION (what a
detector or a verification measured). The model's summary built on them is
INFERENCE.

GPS fields are withheld from the metadata tool: a location is personal data the
model provider has no need to see, and it cannot be grounded any better than
the rest of the metadata.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy.orm import Session

from app.db import repository as repo
from app.db.models import AnalysisRecord
from app.services.fingerprinting.fingerprint import EMBEDDING_ALGORITHM
from app.services.provenance.timeline import build_timeline
from app.services.sources.connectors import SourceQuery
from app.services.sources.url_verification import UrlVerificationSource

FACT = "FACT"
OBSERVATION = "OBSERVATION"

#: The serialised result handed to the model is cut at this many characters.
MAX_RESULT_CHARS = 16_000
_MAX_ITEMS = 25
_MAX_TEXT = 300

_URL = re.compile(r"https?://[^\s\"'<>\\]+")


# ── Arguments ─────────────────────────────────────────────────────────────────

class NoArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VerifySourceMatchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str = Field(min_length=10, max_length=2048, pattern=r"^https?://\S+$")


# ── Context ───────────────────────────────────────────────────────────────────

@dataclass
class ToolContext:
    """What every tool call is bound to. Nothing here comes from the model."""
    db: Session
    analysis_id: str
    owner: str
    url_checks_left: int = 2
    url_source_factory: Callable[[], UrlVerificationSource] = UrlVerificationSource
    # URLs the model may hand to verify_source_match: those seen in earlier
    # tool results, and the one the user supplied.
    known_urls: set[str] = field(default_factory=set)


class ToolError(Exception):
    """A refusal or failure reported back to the model as the tool's result."""


def _text(value: Any) -> Any:
    if isinstance(value, str) and len(value) > _MAX_TEXT:
        return value[:_MAX_TEXT] + "..."
    return value


def _iso(value) -> Optional[str]:
    return value.isoformat() if value else None


# ── Tools ─────────────────────────────────────────────────────────────────────

def get_analysis_overview(ctx: ToolContext, analysis: AnalysisRecord, args: NoArguments) -> dict:
    media = analysis.media
    meta = media.metadata_record
    record = repo.get_assessment_for_analysis(ctx.db, analysis.id)
    return {
        "analysis": {
            "status": analysis.status,
            "assessment": analysis.assessment,
            "evidence_strength": analysis.evidence_strength,
            "created_at": _iso(analysis.created_at),
            "completed_at": _iso(analysis.completed_at),
            "error_message": _text(analysis.error_message),
        },
        "media": {
            "media_type": media.media_type,
            "content_type": media.content_type,
            "size_bytes": media.size_bytes,
            "sha256": media.sha256,
            "width": meta.width if meta else None,
            "height": meta.height if meta else None,
        },
        "assessment_detail": (
            {
                "category": record.category,
                "evidence_strength": record.evidence_strength,
                "rule": record.rule,
                "rationale": record.rationale,
                "coverage_gaps": record.coverage_gaps,
            }
            if record else None
        ),
        # Reported beside the manipulation assessment, never folded into it.
        "ai_generation": record.ai_generation if record else None,
        "source_searches": analysis.source_searches,
    }


def list_detections(ctx: ToolContext, analysis: AnalysisRecord, args: NoArguments) -> dict:
    detections = repo.get_detections_for_analysis(ctx.db, analysis.id)
    return {
        "count": len(detections),
        "detections": [
            {
                "id": d.id,
                "detector": d.detector,
                "detector_version": d.detector_version,
                "availability": d.availability,
                "execution_status": d.execution_status,
                "status_reason": _text(d.status_reason),
                "assessment": d.assessment,
                "confidence": d.confidence,
                "score_semantics": d.score_semantics,
                "manipulation_type": d.manipulation_type,
                "regions": len(d.regions),
            }
            for d in detections[:_MAX_ITEMS]
        ],
    }


def get_observations_and_findings(ctx: ToolContext, analysis: AnalysisRecord, args: NoArguments) -> dict:
    observations = repo.get_observations_for_analysis(ctx.db, analysis.id)
    findings = repo.get_findings_for_analysis(ctx.db, analysis.id)
    return {
        "observation_count": len(observations),
        "observations": [
            {
                "id": o.id,
                "detector": o.detector,
                "signal_family": o.signal_family,
                "polarity": o.polarity,
                "score": o.score,
                "score_semantics": o.score_semantics,
                "localised": o.localised,
                "description": _text(o.description),
            }
            for o in observations[:_MAX_ITEMS]
        ],
        "finding_count": len(findings),
        "findings": [
            {
                "id": f.id,
                "statement": _text(f.statement),
                "supporting_observation_ids": [l.observation_id for l in f.links if l.relation == "SUPPORTS"],
                "contradicting_observation_ids": [l.observation_id for l in f.links if l.relation == "CONTRADICTS"],
            }
            for f in findings[:_MAX_ITEMS]
        ],
    }


def get_similarity_matches(ctx: ToolContext, analysis: AnalysisRecord, args: NoArguments) -> dict:
    matches = repo.get_similarity_matches_for_analysis(ctx.db, analysis.id)
    return {
        "count": len(matches),
        "matches": [
            {
                "match_type": m.match_type,
                "external_url": m.external_url,
                "similarity": m.similarity,
                "method": m.method,
                "rank": m.rank,
                "verification_status": m.verification_status,
                "verification": m.verification,
            }
            for m in matches[:10]
        ],
    }


def get_sources(ctx: ToolContext, analysis: AnalysisRecord, args: NoArguments) -> dict:
    sources = repo.get_sources_for_analysis(ctx.db, analysis.id)
    return {
        # Which connectors ran: no sources means "not found where searched"
        # only for a connector that COMPLETED.
        "source_searches": analysis.source_searches,
        "count": len(sources),
        "sources": [
            {
                "url": s.url,
                "title": _text(s.title),
                "provider": s.provider,
                "published_at": _iso(s.source_timestamp),
                "is_earliest_known": s.is_earliest_known,
                "verification_status": s.verification_status,
                "verification": s.verification,
                "citation": s.citation,
            }
            for s in sources[:10]
        ],
    }


def _metadata_parts(analysis: AnalysisRecord) -> tuple[Optional[dict], Optional[dict]]:
    meta = analysis.media.metadata_record
    if meta is None:
        return None, None
    return meta.exif_data, (meta.raw_metadata or {}).get("c2pa")


def get_provenance_timeline(ctx: ToolContext, analysis: AnalysisRecord, args: NoArguments) -> dict:
    exif, c2pa = _metadata_parts(analysis)
    timeline = build_timeline(repo.get_sources_for_analysis(ctx.db, analysis.id), exif, c2pa)
    timeline["events"] = timeline.get("events", [])[:_MAX_ITEMS]
    return timeline


def get_metadata(ctx: ToolContext, analysis: AnalysisRecord, args: NoArguments) -> dict:
    meta = analysis.media.metadata_record
    exif, c2pa = _metadata_parts(analysis)
    withheld = sorted(k for k in (exif or {}) if str(k).upper().startswith("GPS"))
    fields = {
        str(k): _text(v if isinstance(v, (int, float, bool)) or v is None else str(v))
        for k, v in sorted((exif or {}).items(), key=lambda kv: str(kv[0]))
        if k not in withheld
    }
    return {
        "format": meta.format if meta else None,
        "width": meta.width if meta else None,
        "height": meta.height if meta else None,
        "exif_field_count": len(fields),
        "exif": dict(list(fields.items())[:40]),
        "gps_fields_withheld": withheld,
        # NULL for media analysed before Content Credentials were checked.
        "content_credentials": c2pa,
    }


def verify_source_match(ctx: ToolContext, analysis: AnalysisRecord, args: VerifySourceMatchArguments) -> dict:
    if args.url not in ctx.known_urls:
        raise ToolError(
            "url refused: verify_source_match only checks a URL that appeared in an "
            "earlier tool result or that the user supplied"
        )
    if ctx.url_checks_left <= 0:
        raise ToolError("url check budget for this investigation is spent")
    ctx.url_checks_left -= 1

    media = analysis.media
    fingerprints = repo.get_fingerprints_for_media(ctx.db, media.id)
    query = SourceQuery(
        file_path=media.file_path,
        fingerprints={fp.algorithm: fp.value for fp in fingerprints},
        embedding_version=next(
            (fp.algorithm_version for fp in fingerprints if fp.algorithm == EMBEDDING_ALGORITHM), None
        ),
        claimed_url=args.url,
    )
    result = ctx.url_source_factory().run(query, ctx.db)  # never raises
    return {
        "url": args.url,
        "status": result.status.value,
        "reason": _text(result.reason),
        "hits": [
            {k: v for k, v in hit.to_dict().items() if k in
             ("url", "verification_status", "verification", "similarity", "citation")}
            for hit in result.hits
        ],
        "note": "Checked for this investigation only; not added to the analysis's sources.",
    }


# ── Registry ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    category: str
    arguments: type[BaseModel]
    run: Callable[[ToolContext, AnalysisRecord, Any], dict]

    def spec(self) -> dict:
        """The OpenAI-format function definition, with a strict JSON schema."""
        if self.arguments is NoArguments:
            parameters = {"type": "object", "properties": {}, "required": [],
                          "additionalProperties": False}
        else:
            parameters = {
                "type": "object",
                "properties": {"url": {
                    "type": "string",
                    "description": "An http(s) URL that appeared in an earlier tool result.",
                    "minLength": 10, "maxLength": 2048,
                }},
                "required": ["url"],
                "additionalProperties": False,
            }
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": parameters, "strict": True,
        }}


TOOLS: dict[str, Tool] = {t.name: t for t in (
    Tool("get_analysis_overview",
         "The analysis status, its manipulation assessment with rationale and coverage gaps, "
         "the separate AI-generation assessment, the media type/size/hash, and which source "
         "searches ran.", OBSERVATION, NoArguments, get_analysis_overview),
    Tool("list_detections",
         "Every detector that was asked: whether it was available and ran, its assessment, "
         "its score and what the score means.", OBSERVATION, NoArguments, list_detections),
    Tool("get_observations_and_findings",
         "The observations the detectors produced and the findings built from them, with "
         "which observations support or contradict each finding.",
         OBSERVATION, NoArguments, get_observations_and_findings),
    Tool("get_similarity_matches",
         "Near-duplicate matches from the caller's own media and the reference corpus, with "
         "their verification status.", OBSERVATION, NoArguments, get_similarity_matches),
    Tool("get_sources",
         "Where the image was found published, each with its verification status and the "
         "source's own citation, and which source connectors ran.", FACT, NoArguments, get_sources),
    Tool("get_provenance_timeline",
         "Dated, cited events in the image's known life (publication, archive capture, "
         "retrieval, metadata dates) and the earliest known appearance.",
         FACT, NoArguments, get_provenance_timeline),
    Tool("get_metadata",
         "File format, dimensions, EXIF fields (GPS withheld) and Content Credentials (C2PA).",
         FACT, NoArguments, get_metadata),
    Tool("verify_source_match",
         "Fetch a page and check whether it holds this image, using the same verification "
         "as a source match. Only for a URL that appeared in an earlier tool result. "
         "Limited to a few calls.", OBSERVATION, VerifySourceMatchArguments, verify_source_match),
)}


def tool_specs() -> list[dict]:
    return [tool.spec() for tool in TOOLS.values()]


def serialise(payload: Any) -> str:
    text = json.dumps(payload, default=str, ensure_ascii=False, sort_keys=True)
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS] + "...[truncated]"
    return text


def execute(ctx: ToolContext, name: str, raw_arguments: Optional[str]) -> dict:
    """
    Run one tool call and describe it as a trace entry. Never raises.

    The entry has ``name``, ``arguments``, ``category`` and either ``result`` or
    ``error``. A name outside the registry, arguments that do not validate, or an
    analysis the caller does not own are all errors reported to the model.
    """
    tool = TOOLS.get(name)
    if tool is None:
        return {"name": name, "arguments": raw_arguments, "category": None,
                "error": f"unknown tool {name!r}; only these exist: {sorted(TOOLS)}"}
    try:
        parsed = json.loads(raw_arguments) if raw_arguments and raw_arguments.strip() else {}
        args = tool.arguments.model_validate(parsed)
    except (json.JSONDecodeError, ValidationError) as exc:
        return {"name": name, "arguments": raw_arguments, "category": tool.category,
                "error": f"invalid arguments: {str(exc)[:500]}"}

    entry = {"name": name, "arguments": args.model_dump(), "category": tool.category}
    analysis = repo.get_owned_analysis(ctx.db, ctx.analysis_id, ctx.owner)
    if analysis is None:
        return {**entry, "error": "analysis not found"}
    try:
        result = tool.run(ctx, analysis, args)
    except ToolError as exc:
        return {**entry, "error": str(exc)}
    except Exception as exc:  # a broken tool is reported, never fatal to the run
        return {**entry, "error": f"tool failed: {type(exc).__name__}"}
    ctx.known_urls.update(u.rstrip(".,;:") for u in _URL.findall(serialise(result)))
    return {**entry, "result": result}
