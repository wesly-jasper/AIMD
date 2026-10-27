"""
Evaluation datasets: what each file is, which split it belongs to, and the
guarantee that no source image appears in more than one split.

A manifest is a JSON list (or JSON Lines) of items:

    {"file": "cam_03_splice.jpg",       # relative to the manifest, or files/
     "label": "manipulated",            # real | manipulated | ai_generated
     "group": "cam_03",                 # the source image; shared by its edits
     "kind": "splice",                  # manipulation kind, or the generator
     "generator": "Midjourney",         # optional, for AI-generated items
     "mask": "cam_03.png",              # optional pixel mask (masks/)
     "split": "test",                   # train | val | test (see assign_splits)
     "tags": ["jpeg_q75"]}              # optional robustness conditions

Splitting is by ``group``, never by file: an edited photo and its original in
different splits would let a calibration fitted on one be tested on what is
nearly the same image. Items without a group are their own group. AI-generated
items are additionally grouped by generator when ``by_generator`` is set, so a
generator can be held out entirely.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

LABELS = ("real", "manipulated", "ai_generated")
SPLITS = ("train", "val", "test")


@dataclass
class EvalItem:
    id: str
    path: Path
    label: str
    group: str
    kind: Optional[str] = None
    generator: Optional[str] = None
    mask_path: Optional[Path] = None
    split: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    #: Fields from the manifest this loader does not interpret, kept verbatim.
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def manipulation_target(self) -> Optional[bool]:
        """Ground truth for the manipulation branch; None = not part of it."""
        if self.label == "manipulated":
            return True
        if self.label == "real":
            return False
        # A wholly generated image was not edited after capture: it is neither
        # a positive nor a fair negative for an editing detector.
        return None

    @property
    def ai_generation_target(self) -> Optional[bool]:
        """Ground truth for the AI-generation branch; None = not part of it."""
        if self.label == "ai_generated":
            return True
        if self.label == "real":
            return False
        if self.label == "manipulated":
            # Generative inpainting is partly generated: ambiguous, so excluded.
            # A conventional edit of a camera photo is a hard negative -- the
            # baseline detector flagged 30% of them as AI-generated.
            kind = (self.kind or "").lower()
            return None if ("inpaint" in kind and "opencv" not in kind) else False
        return None


def _resolve(root: Path, name: str, subdir: str) -> Path:
    for candidate in (root / name, root / subdir / name):
        if candidate.is_file():
            return candidate
    return root / subdir / name  # reported missing by validate()


def load_manifest(path: str | Path) -> list[EvalItem]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    raw: list[dict[str, Any]]
    if path.suffix == ".jsonl":
        raw = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        raw = json.loads(text)
    root = path.parent
    known = {"file", "label", "group", "kind", "generator", "mask", "split", "tags", "id"}
    items: list[EvalItem] = []
    for entry in raw:
        name = entry["file"]
        label = entry["label"]
        if label not in LABELS:
            raise ValueError(f"{name}: label must be one of {LABELS}, not {label!r}")
        kind = entry.get("kind")
        generator = entry.get("generator") or (kind if label == "ai_generated" else None)
        items.append(EvalItem(
            id=entry.get("id") or name,
            path=_resolve(root, name, "files"),
            label=label,
            group=str(entry.get("group") or name),
            kind=kind,
            generator=generator,
            mask_path=_resolve(root, entry["mask"], "masks") if entry.get("mask") else None,
            split=entry.get("split"),
            tags=list(entry.get("tags") or []),
            extra={k: v for k, v in entry.items() if k not in known},
        ))
    return items


def manifest_sha256(items: Iterable[EvalItem]) -> str:
    """Identity of a dataset version: ids, labels, groups and splits."""
    rows = sorted((i.id, i.label, i.group, i.split or "") for i in items)
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def validate(items: list[EvalItem]) -> list[str]:
    """Problems that would make metrics wrong. Empty when the dataset is usable."""
    problems: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item.id in seen:
            problems.append(f"duplicate id {item.id}")
        seen.add(item.id)
        if not item.path.is_file():
            problems.append(f"{item.id}: file missing")
        if item.mask_path is not None and not item.mask_path.is_file():
            problems.append(f"{item.id}: mask missing")
        if item.split is not None and item.split not in SPLITS:
            problems.append(f"{item.id}: unknown split {item.split!r}")
    problems.extend(split_leakage(items))
    return problems


def split_leakage(items: list[EvalItem]) -> list[str]:
    """Groups (source images) that appear in more than one split."""
    splits_by_group: dict[str, set[str]] = {}
    for item in items:
        if item.split:
            splits_by_group.setdefault(item.group, set()).add(item.split)
    return [
        f"group {g} is in several splits: {sorted(s)}"
        for g, s in sorted(splits_by_group.items()) if len(s) > 1
    ]


def _bucket(key: str, seed: str) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64


def assign_splits(
    items: list[EvalItem],
    fractions: dict[str, float],
    *,
    seed: str = "aimd-eval-v1",
    by_generator: bool = False,
    overwrite: bool = False,
) -> list[EvalItem]:
    """
    Deterministic split by group: the same manifest and seed always give the
    same split, and every item of a group lands in the same split.

    ``fractions`` maps split name to share, e.g. ``{"val": 0.4, "test": 0.6}``.
    With ``by_generator``, AI-generated items are split by generator instead,
    so each generator is wholly in one split (a held-out-generator test).
    """
    total = sum(fractions.values())
    if total <= 0 or any(name not in SPLITS for name in fractions):
        raise ValueError(f"fractions must name splits from {SPLITS} with a positive total")
    edges: list[tuple[float, str]] = []
    acc = 0.0
    for name, share in fractions.items():
        acc += share / total
        edges.append((acc, name))

    for item in items:
        if item.split and not overwrite:
            continue
        key = item.group
        if by_generator and item.label == "ai_generated" and item.generator:
            key = f"generator:{item.generator}"
        u = _bucket(key, seed)
        item.split = next(name for edge, name in edges if u < edge or edge == edges[-1][0])
    return items


def write_manifest(items: list[EvalItem], path: str | Path) -> None:
    path = Path(path)
    root = path.parent

    def rel(p: Optional[Path]) -> Optional[str]:
        if p is None:
            return None
        try:
            return p.relative_to(root).as_posix()
        except ValueError:
            return str(p)

    rows = []
    for i in items:
        row = {
            "id": i.id, "file": rel(i.path), "label": i.label, "group": i.group,
            "kind": i.kind, "generator": i.generator, "mask": rel(i.mask_path),
            "split": i.split, "tags": i.tags or None, **i.extra,
        }
        rows.append({k: v for k, v in row.items() if v is not None})
    path.write_text(json.dumps(rows, indent=1), encoding="utf-8")
