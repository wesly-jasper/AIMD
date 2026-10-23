"""Index Wikimedia Commons images as the reference corpus for source discovery.

Each file directly in the chosen categories is downloaded as a scaled rendition
(--thumb-width), stored under REFERENCE_DIR (default backend/storage/reference),
and recorded with its page URL, file URL, uploader, upload time and licence,
plus the fingerprints an upload gets. Reruns skip files already indexed.

Usage (from backend/, with the venv python):

    python scripts/build_reference_corpus.py --contact you@example.com
    python scripts/build_reference_corpus.py --contact https://example.com/about \\
        --category "Quality images of buildings" --per-category 50 --limit 200

Wikimedia requires a contact (email address or URL) in the User-Agent; pass
--contact or set WIKIMEDIA_CONTACT. Requests are paced to one per second, so
500 images take roughly 10-15 minutes.

The CLIP embedding is what retrieves a cropped copy, so the build stops if the
backbone is not installed (python scripts/fetch_univfd.py) unless you pass
--no-embeddings.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.core.config import settings  # noqa: E402
from app.db import repository as repo  # noqa: E402
from app.db.database import SessionLocal, init_db  # noqa: E402
from app.services.fingerprinting.fingerprint import (  # noqa: E402
    FingerprintService,
    embedding_version,
)
from app.services.sources.commons_client import CORPUS, CommonsClient, CommonsError  # noqa: E402
from app.services.sources.reference_corpus import build_corpus  # noqa: E402

#: News photography and landmarks: what a misattributed image usually is. Only
#: files directly in a category are listed, so these were picked for holding
#: hundreds directly (events and architecture keep most in subcategories).
DEFAULT_CATEGORIES = (
    "Images from Voice of America",
    "Quality images of events",
    "Quality images of buildings",
    "Quality images of architecture",
    "Quality images of monuments and memorials",
    "Quality images of cityscapes",
)


def _embedder(args):
    if args.no_embeddings:
        return None
    model_path = args.model_path or settings.univfd_model_path
    if not model_path:
        raise SystemExit(
            "No CLIP backbone: set UNIVFD_MODEL_PATH or pass --model-path "
            "(install it with scripts/fetch_univfd.py), or pass --no-embeddings."
        )
    from app.services.detection.models.clip_backbone import get_clip_backbone

    try:
        return get_clip_backbone(model_path, device=settings.device)
    except Exception as exc:
        raise SystemExit(f"CLIP backbone failed to load from {model_path}: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--contact", default=settings.wikimedia_contact,
                        help="email address or URL for the User-Agent (default: WIKIMEDIA_CONTACT)")
    parser.add_argument("--category", action="append", dest="categories",
                        help="Commons category, repeatable (default: news photos and landmarks)")
    parser.add_argument("--per-category", type=int, default=125)
    parser.add_argument("--limit", type=int, default=500, help="images in total")
    parser.add_argument("--thumb-width", type=int, default=1280,
                        help="width of the rendition downloaded (default 1280)")
    parser.add_argument("--min-side", type=int, default=600,
                        help="skip originals whose shorter side is below this")
    parser.add_argument("--root", type=Path, default=settings.reference_dir)
    parser.add_argument("--model-path", type=Path, help="CLIP backbone (default: UNIVFD_MODEL_PATH)")
    parser.add_argument("--no-embeddings", action="store_true",
                        help="index hashes only; crops will not be retrieved")
    args = parser.parse_args(argv)
    # Commons titles are any script; a redirected Windows console is cp1252.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    if not args.contact:
        parser.error("--contact (or WIKIMEDIA_CONTACT) is required by Wikimedia's User-Agent policy")
    embedder = _embedder(args)
    fingerprinter = FingerprintService(embedder=embedder, default_embedder=False)

    init_db()
    categories = args.categories or list(DEFAULT_CATEGORIES)
    with CommonsClient(args.contact) as client, SessionLocal() as db:
        try:
            stats = build_corpus(
                db, client, categories,
                root=args.root.resolve(),
                fingerprinter=fingerprinter,
                embedding_version=embedding_version(embedder) if embedder else None,
                per_category=args.per_category,
                limit=args.limit,
                thumb_width=args.thumb_width,
                min_side=args.min_side,
                progress=print,
            )
        except CommonsError as exc:
            print(f"Stopped: {exc}", file=sys.stderr)
            return 2
        total = repo.count_reference_images(db, CORPUS)

    print(f"\nindexed {stats.indexed}, refreshed {stats.refreshed}, unchanged {stats.unchanged}, "
          f"failed {stats.failed}; {total} images in the {CORPUS} corpus")
    for reason, count in stats.skipped.most_common():
        print(f"  skipped {count}: {reason}")
    return 1 if stats.failed and not stats.in_corpus else 0


if __name__ == "__main__":
    sys.exit(main())
