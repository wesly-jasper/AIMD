"""Fetch an evaluation-only sample of OpenFake's test split (ComplexDataLab/OpenFake).

OpenFake is CC-BY-NC-4.0, and its proprietary-generator subsets are released
under the providers' non-compete terms. It is used here for **measurement only**:
the files stay in the local evaluation folder, are never redistributed, and
nothing is fitted on them (every item is in the ``test`` split of its own set,
and calibrations are fitted on Commons data only). Each manifest entry carries
that licence note.

The test split holds generators that are not in OpenFake's training split,
including GPT Image 1.5 and 2, Flux.2, Grok (Aurora) and Nano Banana Pro.

Images are read from the original parquet files, never from the dataset viewer,
which serves re-encoded JPEGs: every kept image's bytes are checked against the
dataset's own ``image_sha256``.

Usage (from backend/): python scripts/fetch_openfake.py [--out DIR]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import os
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = "datasets/ComplexDataLab/OpenFake/core"
LICENCE = "CC-BY-NC-4.0 (OpenFake, ComplexDataLab); evaluation only, not redistributed"
#: OpenFake model name -> (generator family used in our reports, quota)
QUOTAS: dict[str, tuple[str | None, int]] = {
    "gpt-image-1.5": ("GPT-image", 250),
    "gpt-image-2": ("GPT-image", 150),
    "flux.2-klein-9b": ("Flux", 250),
    "aurora-20-1-25": ("Grok", 100),
    "nano-banana-pro": ("Imagen / Gemini", 100),
    "midjourney-7": ("Midjourney", 100),
    "docci": (None, 300),
    "imagenet": (None, 150),
}
#: At most this many rows of one model from one row group, so the sample is
#: spread across the split rather than drawn from a few neighbouring rows.
PER_GROUP = 60
DEFAULT_OUT = Path(os.environ.get("LOCALAPPDATA", BACKEND)) / "Temp" / "aimd-reliability" / "sets" / "openfake"
_EXT = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}


def plan(survey: dict) -> list[tuple[str, int]]:
    """Row groups to read, chosen greedily by wanted rows per byte."""
    groups = [(f, i, g) for f, gs in survey.items() for i, g in enumerate(gs)]
    have = collections.Counter()
    chosen = []

    def gain(g):
        return sum(min(g["models"].get(m, 0), PER_GROUP, q - have[m]) for m, (_, q) in QUOTAS.items())

    while any(have[m] < q for m, (_, q) in QUOTAS.items()):
        best = max(groups, key=lambda r: gain(r[2]) / r[2]["bytes"])
        if gain(best[2]) == 0:
            break
        for m, (_, q) in QUOTAS.items():
            have[m] += min(best[2]["models"].get(m, 0), PER_GROUP, q - have[m])
        chosen.append((best[0], best[1]))
        groups.remove(best)
    return chosen


def main() -> int:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    from PIL import Image

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--survey", type=Path, required=True,
                        help="row-group survey: {file: [{rows, bytes, models}]} (model column only)")
    args = parser.parse_args()

    out: Path = args.out
    (out / "files").mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    items = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    have = collections.Counter(i["openfake_model"] for i in items)
    seen = {i["sha256"] for i in items}
    done_groups = {tuple(g) for g in json.loads((out / "groups_done.json").read_text())} \
        if (out / "groups_done.json").exists() else set()

    fs = HfFileSystem()
    for path, g in plan(json.loads(args.survey.read_text(encoding="utf-8"))):
        if (path, g) in done_groups:
            continue
        with fs.open(path, "rb", block_size=8 * 2 ** 20) as fh:
            table = pq.ParquetFile(fh).read_row_group(
                g, columns=["image", "model", "label", "image_sha256", "row_id", "type", "release_date"])
        taken = collections.Counter()
        for row in table.to_pylist():
            model = row["model"]
            if model not in QUOTAS or have[model] >= QUOTAS[model][1] or taken[model] >= PER_GROUP:
                continue
            data = (row["image"] or {}).get("bytes")
            if not data or hashlib.sha256(data).hexdigest() != row["image_sha256"] or row["image_sha256"] in seen:
                continue
            with Image.open(io.BytesIO(data)) as im:
                fmt, size = im.format, im.size
            if fmt not in _EXT or min(size) < 256:
                continue
            family, _ = QUOTAS[model]
            name = f"of_{model.replace('.', '').replace('-', '')[:14]}_{have[model] + 1:04d}{_EXT[fmt]}"
            (out / "files" / name).write_bytes(data)
            items.append({
                "id": name, "file": f"files/{name}",
                "label": "real" if row["label"] == "real" else "ai_generated",
                "generator": family, "openfake_model": model, "openfake_type": row["type"],
                # Every image is its own group: OpenFake records no uploader, and
                # near-duplicates are checked separately by perceptual hash.
                "group": f"openfake:{row['row_id']}", "split": "test", "set": "openfake",
                "tags": ["original"], "sha256": row["image_sha256"], "source": f"openfake:{row['row_id']}",
                "licence": LICENCE, "format": fmt, "size": list(size),
            })
            seen.add(row["image_sha256"])
            have[model] += 1
            taken[model] += 1
        done_groups.add((path, g))
        manifest_path.write_text(json.dumps(items, indent=1), encoding="utf-8")
        (out / "groups_done.json").write_text(json.dumps(sorted(done_groups)))
        print(f"{path.rsplit('/', 1)[1]} rg {g}: +{sum(taken.values())}  total {dict(have)}", flush=True)
    print(json.dumps(dict(have), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
