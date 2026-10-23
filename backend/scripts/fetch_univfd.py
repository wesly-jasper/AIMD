"""Download and verify the CLIP ViT-L/14 backbone and UnivFD head into backend/models/univfd.

Both files are checked against the SHA-256 pinned in
``app/services/detection/models/univfd_release.py``; a file that does not match
is deleted, not installed.

Usage (from backend/, with the venv python):

    python scripts/fetch_univfd.py                              # download both
    python scripts/fetch_univfd.py --clip-pt D:/Downloads/ViT-L-14.pt

The backbone is 890 MB. On a slow connection, download it in a browser from

    https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt

and pass it with ``--clip-pt``: it is verified against the same hash, so where
it came from does not matter. An interrupted download is resumed on the next run.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from fetch_models import BACKEND, VerificationError, download, file_digests  # noqa: E402

from app.services.detection.models import univfd_release as release  # noqa: E402

DEFAULT_ROOT = BACKEND / "models" / "univfd"


def install_file(target: Path, expected_sha256: str, *, url: str | None = None,
                 local: Path | None = None, show_progress: bool = False) -> str:
    """Install one pinned file from ``local`` or ``url``; returns its SHA-256."""
    if target.is_file():
        if file_digests(target)[0] == expected_sha256:
            print(f"{target.name} already installed and verified")
            return expected_sha256
        target.unlink()  # stale or tampered: replace it, never keep it

    target.parent.mkdir(parents=True, exist_ok=True)
    if local is not None:
        partial = target.with_name(target.name + ".part")
        shutil.copyfile(local, partial)
        os.replace(partial, target)
    else:
        print(f"Downloading {url}")
        download(url, target, show_progress=show_progress)

    actual = file_digests(target)[0]
    if actual != expected_sha256:
        target.unlink()
        raise VerificationError(f"{target.name}: SHA-256 {actual} != pinned {expected_sha256}")
    print(f"Verified {target.name} (SHA-256 {actual[:16]}...)")
    return actual


def write_manifest(root: Path, manifest: dict) -> None:
    manifest.update({
        "schema_version": 1,
        "upstream": {"repo": release.UPSTREAM_REPO, "commit": release.UPSTREAM_COMMIT},
        "licence": release.LICENCE_NOTE,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    path = root / release.MANIFEST_NAME
    partial = path.with_name(path.name + ".part")
    partial.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(partial, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="install directory (default: backend/models/univfd)")
    parser.add_argument("--clip-pt", type=Path,
                        help="use an already downloaded ViT-L-14.pt instead of downloading it")
    parser.add_argument("--skip-clip", action="store_true",
                        help="install the 4 KB UnivFD head only")
    args = parser.parse_args(argv)

    root: Path = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / release.MANIFEST_NAME
    manifest: dict = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    )
    exit_code = 0
    try:
        manifest["head"] = {
            "path": release.HEAD_FILE,
            "sha256": install_file(root / release.HEAD_FILE, release.HEAD_SHA256,
                                   url=release.HEAD_URL),
        }
        if not args.skip_clip:
            try:
                manifest["clip"] = {
                    "path": release.CLIP_FILE,
                    "sha256": install_file(root / release.CLIP_FILE, release.CLIP_SHA256,
                                           url=release.CLIP_URL, local=args.clip_pt,
                                           show_progress=True),
                }
            except (URLError, HTTPError, TimeoutError) as exc:
                print(f"CLIP download failed: {type(exc).__name__}: {exc}\n"
                      "Download ViT-L-14.pt in a browser (see --help) and pass --clip-pt.",
                      file=sys.stderr)
                exit_code = 2
    except VerificationError as exc:
        print(f"Verification failed: {exc}", file=sys.stderr)
        exit_code = 1
    except (URLError, HTTPError, TimeoutError) as exc:
        print(f"Download failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        exit_code = 2
    finally:
        write_manifest(root, manifest)

    if exit_code == 0 and (root / release.CLIP_FILE).is_file():
        print(f"UnivFD installed at {root}")
        print("Enable it in .env with UNIVFD_ENABLED=true, SIMILARITY_EMBEDDING_ENABLED=true "
              f"and UNIVFD_MODEL_PATH={root}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
