"""
C2PA (Content Credentials) verification of an uploaded file.

What a result means, and does not:

  NO_MANIFEST  the file carries no Content Credentials. Most images do not; this
               says nothing about the image either way.
  INVALID      a manifest is present but fails validation: the file was changed
               after signing, or the manifest is malformed. The signer's claims
               no longer describe these bytes.
  VALID        the manifest is intact and its signature verifies, but the signer
               is not on a trust list. Anyone can create such a certificate, so
               the claims are only as good as an unknown signer.
  TRUSTED      intact, and signed by a certificate that chains to the configured
               trust list (by default the official C2PA Trust List, installed
               by scripts/fetch_c2pa_trust_list.py). With no list installed
               nothing can be TRUSTED, and the result says so.
  UNAVAILABLE  the C2PA library is not installed.
  FAILED       the library could not read the file.

Even TRUSTED says who signed and what they asserted (e.g. "captured by a
camera", or a ``digitalSourceType`` of trainedAlgorithmicMedia for AI output),
not that the content is true. Declared AI generation is reported as the
signer's declaration, beside -- never inside -- the detector verdicts.
"""
from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

#: IPTC digital source types that declare the content as generated or
#: composited by an AI model.
AI_SOURCE_TYPES = {
    "trainedAlgorithmicMedia",
    "compositeWithTrainedAlgorithmicMedia",
    "algorithmicallyEnhanced",
}

_STATE = {"Invalid": "INVALID", "Valid": "VALID", "Trusted": "TRUSTED"}


def _source_type_name(uri: Optional[str]) -> Optional[str]:
    return uri.rstrip("/").rsplit("/", 1)[-1] if uri else None


def _actions(manifest: dict) -> list[dict[str, Any]]:
    out = []
    for assertion in manifest.get("assertions") or []:
        if not str(assertion.get("label", "")).startswith("c2pa.actions"):
            continue
        for action in (assertion.get("data") or {}).get("actions") or []:
            out.append({
                "action": action.get("action"),
                "when": action.get("when"),
                "software_agent": (action.get("softwareAgent") or {}).get("name")
                if isinstance(action.get("softwareAgent"), dict) else action.get("softwareAgent"),
                "digital_source_type": _source_type_name(action.get("digitalSourceType")),
            })
    return out


@lru_cache(maxsize=4)
def _trust(path: Optional[str]) -> tuple[Optional[str], dict[str, Any]]:
    """The trust anchors PEM at ``path`` and a description of it for the record."""
    if not path or not Path(path).is_file():
        return None, {"loaded": False}
    pem = Path(path).read_text(encoding="utf-8")
    return pem, {
        "loaded": True,
        "name": Path(path).stem,
        "anchors": pem.count("BEGIN CERTIFICATE"),
        "sha256": hashlib.sha256(pem.encode()).hexdigest(),
    }


def verify_c2pa(file_path: Path | str, trust_anchors: Optional[Path | str] = None) -> dict[str, Any]:
    """
    The C2PA outcome for one file; never raises.

    ``trust_anchors`` is a PEM bundle of root certificates; by default the
    configured C2PA trust list. A signer chaining to one of them is TRUSTED.
    """
    try:
        import c2pa
    except ImportError:
        return {"status": "UNAVAILABLE", "reason": "c2pa-python is not installed"}

    anchors = trust_anchors if trust_anchors is not None else settings.c2pa_trust_anchors_path
    pem, trust_list = _trust(str(anchors) if anchors else None)
    base = {"sdk_version": c2pa.sdk_version(), "trust_list": trust_list}
    try:
        context = None
        if pem:
            context = c2pa.Context(c2pa.Settings.from_dict(
                {"trust": {"trust_anchors": pem}, "verify": {"verify_trust": True}}
            ))
        reader = c2pa.Reader.try_create(str(file_path), context=context)
    except Exception as exc:
        # A manifest that cannot even be parsed is a broken manifest, not none.
        logger.info("C2PA read failed for %s: %s", file_path, exc)
        return {**base, "status": "FAILED", "reason": f"{type(exc).__name__}: {exc}"}
    if reader is None:
        return {**base, "status": "NO_MANIFEST"}

    try:
        state = reader.get_validation_state()
        results = reader.get_validation_results() or {}
        manifest = reader.get_active_manifest() or {}
        embedded = reader.is_embedded()
    finally:
        reader.close()

    active = results.get("activeManifest") or {}
    failures = [
        {"code": f.get("code"), "explanation": f.get("explanation")}
        for f in active.get("failure") or []
    ]
    signature = manifest.get("signature_info") or {}
    actions = _actions(manifest)
    source_types = sorted({a["digital_source_type"] for a in actions if a["digital_source_type"]})
    generator = manifest.get("claim_generator_info") or []
    status = _STATE.get(str(state), "INVALID")
    return {
        **base,
        "status": status,
        "validation_state": str(state),
        "trusted_signer": status == "TRUSTED",
        "failures": failures,
        "signer": {
            "common_name": signature.get("common_name"),
            "issuer": signature.get("issuer"),
            "algorithm": signature.get("alg"),
            "signed_at": signature.get("time"),
        },
        "claim_generator": manifest.get("claim_generator")
        or ", ".join(g.get("name", "") for g in generator if isinstance(g, dict)) or None,
        "title": manifest.get("title"),
        "manifest_label": manifest.get("label"),
        "embedded": embedded,
        "actions": actions,
        "digital_source_types": source_types,
        # The signer's declaration, only meaningful while the manifest is intact.
        "ai_generation_declared": status != "INVALID" and bool(AI_SOURCE_TYPES & set(source_types)),
        "ingredients": len(manifest.get("ingredients") or []),
    }
