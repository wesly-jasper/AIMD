"""
Grounding guard: nothing in the model's summary that no tool said.

The system prompt asks the model to end every claim with the ID of the tool
result it rests on, e.g. ``[T2]`` or ``[T1, T3]``. After generation the
summary is checked against the tool results the model actually received:

  * a citation to an ID that was never issued is removed and flagged;
  * every URL, timestamp and number must appear in some tool result. One that
    does not is replaced with ``[unverified]`` and flagged. Numbers match within
    the precision the summary wrote them at (0.87 matches 0.8734), and a
    percentage matches its fraction (87% matches 0.8734); list lengths count as
    numbers a tool returned;
  * a sentence with no citation is kept and flagged ``uncited``.

This is a lexical check. It proves every URL, number and date was copied from
a tool result, not that the sentence around it reads the result correctly --
which is why the summary stays labelled INFERENCE.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

UNVERIFIED = "[unverified]"

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
# Nemotron often cites in CJK brackets (【T1】) rather than [T1]; both count, and
# the cleaned summary always uses [T1].
_ID_SEPARATOR = r"[,;，、]"
_CITATION = re.compile(
    r"[\[【［〔]\s*(T\d+(?:\s*" + _ID_SEPARATOR + r"\s*T\d+)*)\s*[\]】］〕]"
)
_URL = re.compile(r"https?://[^\s<>\"'\[\]\)【】［］〔〕]+")
_TIMESTAMP = re.compile(
    r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?"
    r"|\b\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}"
)
# Not inside an identifier: "T3", "v2", a hash or a uuid fragment is not a number.
_NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?!\w)(%?)")
_ENUMERATOR = re.compile(r"^\s*(?:[-*•]\s*)?\d+[.)](?=\s)")
_SENTENCE_BREAK =re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9*\-•\"(])")


@dataclass
class GroundedSummary:
    text: str
    citations: list[dict] = field(default_factory=list)
    flags: list[dict] = field(default_factory=list)

    @property
    def grounded(self) -> bool:
        """True when nothing had to be removed (uncited sentences aside)."""
        return not any(f["type"] != "uncited" for f in self.flags)


def _strip_url(url: str) -> str:
    return url.rstrip(".,;:!?")


def _numbers_in(text: str) -> list[float]:
    out = []
    for whole, frac, _ in _NUMBER.findall(text):
        out.append(float(whole.replace(",", "") + (frac or "")))
    return out


def _lengths(value: Any) -> Iterable[int]:
    if isinstance(value, dict):
        yield len(value)
        for v in value.values():
            yield from _lengths(v)
    elif isinstance(value, list):
        yield len(value)
        for v in value:
            yield from _lengths(v)


class Evidence:
    """Everything the tools returned, in the forms the summary is checked against."""

    def __init__(self, results: dict[str, Any]):
        self.ids = set(results)
        self.texts = {
            cid: r if isinstance(r, str) else json.dumps(r, default=str, ensure_ascii=False)
            for cid, r in results.items()
        }
        blob = "\n".join(self.texts.values())
        self.blob = blob
        self.urls = {_strip_url(u).rstrip("/") for u in _URL.findall(blob)}
        self.numbers = set(_numbers_in(blob))
        for r in results.values():
            parsed = r
            if isinstance(r, str):
                try:
                    parsed = json.loads(r)
                except ValueError:
                    parsed = None
            self.numbers.update(_lengths(parsed))

    def has_url(self, url: str) -> bool:
        return _strip_url(url).rstrip("/") in self.urls

    def has_timestamp(self, ts: str) -> bool:
        return ts in self.blob or ts.replace(" ", "T") in self.blob

    def has_number(self, token: str) -> bool:
        m = _NUMBER.fullmatch(token)
        if not m:
            return False
        whole, frac, pct = m.groups()
        value = float(whole.replace(",", "") + (frac or ""))
        decimals = len(frac) - 1 if frac else 0
        tolerance = 0.5 * 10 ** -decimals + 1e-9
        if any(abs(n - value) <= tolerance for n in self.numbers):
            return True
        return bool(pct) and any(abs(n * 100 - value) <= tolerance for n in self.numbers)


def _check_sentence(sentence: str, evidence: Evidence, flags: list[dict]) -> tuple[str, list[str]]:
    cited: list[str] = []

    def citation(m: re.Match) -> str:
        ids = [i.strip() for i in re.split(_ID_SEPARATOR, m.group(1))]
        known = list(dict.fromkeys(i for i in ids if i in evidence.ids))
        for i in ids:
            if i not in evidence.ids:
                flags.append({"type": "unknown_citation", "value": i, "sentence": sentence})
        cited.extend(i for i in known if i not in cited)
        return f"[{', '.join(known)}]" if known else ""

    text = _CITATION.sub(citation, sentence)

    def url(m: re.Match) -> str:
        raw = m.group(0)
        value = _strip_url(raw)
        trailing = raw[len(value):]
        if evidence.has_url(value):
            return raw
        flags.append({"type": "unsupported_url", "value": value, "sentence": sentence})
        return UNVERIFIED + trailing

    text = _URL.sub(url, text)

    def timestamp(m: re.Match) -> str:
        if evidence.has_timestamp(m.group(0)):
            return m.group(0)
        flags.append({"type": "unsupported_timestamp", "value": m.group(0), "sentence": sentence})
        return UNVERIFIED

    # Numbers inside URLs, timestamps and citations were checked as part of
    # them; mask those spans so their digits are not checked twice.
    masked: list[tuple[int, int]] = []
    pieces, last = [], 0
    for m in _TIMESTAMP.finditer(text):
        pieces.append(text[last:m.start()])
        pieces.append(timestamp(m))
        last = m.end()
    pieces.append(text[last:])
    text = "".join(pieces)
    for pattern in (_URL, _TIMESTAMP, _CITATION):
        masked.extend(m.span() for m in pattern.finditer(text))
    masked.extend((m.start(), m.end()) for m in re.finditer(re.escape(UNVERIFIED), text))
    masked.extend(m.span() for m in _ENUMERATOR.finditer(text))  # "2. " opening a list item

    out, last = [], 0
    for m in _NUMBER.finditer(text):
        if any(s <= m.start() < e for s, e in masked):
            continue
        out.append(text[last:m.start()])
        if evidence.has_number(m.group(0)):
            out.append(m.group(0))
        else:
            flags.append({"type": "unsupported_number", "value": m.group(0), "sentence": sentence})
            out.append(UNVERIFIED)
        last = m.end()
    out.append(text[last:])
    return re.sub(r"[ \t]{2,}", " ", "".join(out)).strip(), cited


def _is_claim(sentence: str) -> bool:
    """Headings, list labels and bare punctuation make no claim to cite."""
    if re.fullmatch(r"#+\s.*|\*\*[^*]+\*\*:?", sentence.strip()):
        return False
    s = sentence.strip().lstrip("#*->• ").strip()
    return bool(re.search(r"[A-Za-z]{3}", s)) and not s.endswith(":")


def ground(summary: str, results: dict[str, Any]) -> GroundedSummary:
    """
    Check ``summary`` against ``results`` ({tool_call_id: result as sent}).

    Returns the cleaned text, the citations of each claim, and what was removed.
    """
    evidence = Evidence(results)
    flags: list[dict] = []
    citations: list[dict] = []
    lines_out = []
    for line in _THINK.sub("", summary or "").strip().splitlines():
        sentences_out = []
        for sentence in _SENTENCE_BREAK.split(line):
            if not sentence.strip():
                continue
            cleaned, cited = _check_sentence(sentence, evidence, flags)
            if cited:
                citations.append({"sentence": cleaned, "tool_call_ids": cited})
            elif _is_claim(cleaned):
                flags.append({"type": "uncited", "value": None, "sentence": cleaned})
            sentences_out.append(cleaned)
        lines_out.append(" ".join(sentences_out))
    return GroundedSummary("\n".join(lines_out).strip(), citations, flags)
