"""
Provenance timeline: dated, cited events in the known life of an image.

Built from what is already stored (verified sources and their citations, the
upload's own metadata), so it can always be rebuilt and never drifts from the
records it cites. Event types:

  PUBLISHED  a source published the image at this time (e.g. a Commons upload)
  ARCHIVED   an archive captured a page holding it at this time
  RETRIEVED  AIMD itself fetched the source's copy at this time
  METADATA   a date written in metadata (EXIF, the uploader's description, or
             a Content Credentials signature): a claim, not an appearance

Every event cites where it came from. Only VERIFIED_MATCH sources contribute:
a timeline of a lookalike is not a timeline of this image.

The earliest PUBLISHED or ARCHIVED event is labelled the earliest known
appearance -- the earliest AIMD found in the indexes it searched. It is never
called the original: the image may have appeared earlier somewhere unindexed.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

VERIFIED_MATCH = "VERIFIED_MATCH"
APPEARANCE_TYPES = ("PUBLISHED", "ARCHIVED")
ARCHIVE_PROVIDERS = {"wayback"}

NOTE = (
    "INFERENCE: 'Earliest known appearance' is the earliest publication or archive "
    "capture AIMD found in the sources it searched. It is not a claim that the image "
    "originated there or then; it may have appeared earlier somewhere unindexed. "
    "METADATA dates are claims written into the file or its description, and can be "
    "wrong or edited."
)

_EXIF = re.compile(r"^(\d{4}):(\d{2}):(\d{2})[ T](\d{2}):(\d{2}):(\d{2})")
_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")


def _utc(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def parse_stated_date(value: object) -> tuple[Optional[str], Optional[str]]:
    """
    ``(iso, precision)`` for a date as metadata writes it, or ``(None, None)``.

    EXIF and Commons dates carry no timezone, so a datetime is returned without
    one (camera-local time) and must not be compared with UTC timestamps.
    """
    text = str(value or "").strip()
    if m := _EXIF.match(text):
        return datetime(*map(int, m.groups())).isoformat(), "second"
    if m := _DATE.match(text):
        return datetime(*map(int, m.groups())).date().isoformat(), "day"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None, None
    return parsed.replace(tzinfo=None).isoformat(), "second"


def _event(type_: str, timestamp: Optional[str], description: str, citation: dict,
           *, basis: str, precision: Optional[str] = "second", timezone_note: Optional[str] = None,
           stated_value: Optional[str] = None) -> dict[str, Any]:
    event = {
        "type": type_,
        "timestamp": timestamp,
        "precision": precision,
        "description": description,
        "basis": basis,
        "citation": citation,
        "earliest_known_appearance": False,
    }
    if timezone_note:
        event["timezone"] = timezone_note
    if stated_value is not None:
        event["stated_value"] = stated_value
    return event


def _source_events(source) -> list[dict[str, Any]]:
    citation = source.citation or {}
    cite = {"provider": source.provider, "url": source.url, "source_id": source.id}
    events = []
    published_at = _utc(source.source_timestamp)
    if published_at is not None:
        archived = source.provider in ARCHIVE_PROVIDERS
        who = f" by {citation['uploader']}" if citation.get("uploader") else ""
        events.append(_event(
            "ARCHIVED" if archived else "PUBLISHED",
            published_at.isoformat(),
            (f"Captured by {source.provider}" if archived
             else f"Published on {source.domain or source.provider}{who}"),
            {**cite, "meaning": citation.get("uploaded_at_meaning")},
            basis="source_record",
        ))
    if citation.get("retrieved_at"):
        events.append(_event(
            "RETRIEVED",
            _utc(datetime.fromisoformat(citation["retrieved_at"])).isoformat(),
            "AIMD retrieved this source's copy for its index",
            {**cite, "url": citation.get("file_url") or source.url},
            basis="aimd_record",
        ))
    if stated := citation.get("date_time_original"):
        iso, precision = parse_stated_date(stated)
        events.append(_event(
            "METADATA", iso,
            f"Date the photo was taken, as stated on {source.domain or source.provider}",
            cite, basis="uploader_statement", precision=precision,
            timezone_note="not stated (local time)" if precision == "second" else None,
            stated_value=str(stated),
        ))
    return events


def _upload_events(exif: Optional[dict]) -> list[dict[str, Any]]:
    events = []
    for tag in ("DateTimeOriginal", "DateTimeDigitized"):
        if value := (exif or {}).get(tag):
            iso, precision = parse_stated_date(value)
            events.append(_event(
                "METADATA", iso,
                f"EXIF {tag} of the uploaded file",
                {"provider": "uploaded_file", "field": f"EXIF {tag}"},
                basis="file_metadata", precision=precision,
                timezone_note="not stated (camera local time)",
                stated_value=str(value),
            ))
            break  # Digitized only when Original is missing
    return events


def _c2pa_events(c2pa: Optional[dict]) -> list[dict[str, Any]]:
    """The signing time of intact Content Credentials, as the signer states it."""
    if not c2pa or c2pa.get("status") not in ("VALID", "TRUSTED"):
        return []  # an INVALID manifest no longer describes these bytes
    signed_at = (c2pa.get("signer") or {}).get("signed_at")
    if not signed_at:
        return []
    signer = (c2pa.get("signer") or {}).get("common_name") or "an unnamed signer"
    trust = "a trusted" if c2pa.get("trusted_signer") else "an untrusted"
    try:
        iso, precision = _utc(datetime.fromisoformat(str(signed_at).replace("Z", "+00:00"))).isoformat(), "second"
    except ValueError:
        iso, precision = None, None
    return [_event(
        "METADATA", iso,
        f"Content Credentials signed by {signer} ({trust} certificate)",
        {"provider": "c2pa_manifest", "field": "signature_info.time",
         "manifest_label": c2pa.get("manifest_label")},
        basis="c2pa_manifest", precision=precision, stated_value=str(signed_at),
    )]


def build_timeline(sources: Iterable, exif: Optional[dict] = None,
                   c2pa: Optional[dict] = None) -> dict[str, Any]:
    """
    The timeline for one analysis: ``{events, earliest_known_appearance, note}``.

    ``sources`` are SourceRecords (only VERIFIED_MATCH ones are used); ``exif`` is
    the upload's EXIF and ``c2pa`` its Content Credentials result, as extracted
    at ingestion.
    """
    events = _upload_events(exif) + _c2pa_events(c2pa)
    for source in sources:
        if source.verification_status == VERIFIED_MATCH:
            events.extend(_source_events(source))

    def sort_key(e):
        # Undated claims last. Naive local times sort among UTC ones by their
        # face value; they are never candidates for the earliest appearance.
        return (e["timestamp"] is None, (e["timestamp"] or "")[:19])

    events.sort(key=sort_key)
    appearances = [e for e in events if e["type"] in APPEARANCE_TYPES and e["timestamp"]]
    earliest = min(appearances, key=lambda e: e["timestamp"]) if appearances else None
    if earliest is not None:
        earliest["earliest_known_appearance"] = True
    return {"events": events, "earliest_known_appearance": earliest, "note": NOTE}
