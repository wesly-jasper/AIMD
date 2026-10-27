"""Download and verify the B-Free AI-generation detector into backend/models/bfree.

The archive is checked against the authors' published MD5 and the checkpoint
inside it against the SHA-256 pinned in
``app/services/detection/models/bfree_release.py``; anything that does not match
is deleted, not installed.

B-Free is licensed for informational and non-profit use only
(https://www.grip.unina.it/download/LICENSE_OPEN.txt). Downloading it means
accepting those terms.

Usage (from backend/, with the venv python):

    python scripts/fetch_bfree.py
    python scripts/fetch_bfree.py --zip D:/Downloads/BFREE_dino2reg4.zip

Then enable it in .env with BFREE_ENABLED=true and BFREE_MODEL_PATH=models/bfree.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from fetch_models import BACKEND, VerificationError, download, file_digests  # noqa: E402

from app.services.detection.models import bfree_release as release  # noqa: E402

DEFAULT_ROOT = BACKEND / "models" / "bfree"


def install(root: Path, local_zip: Path | None) -> str:
    target = root / release.MODEL_DIR / release.WEIGHTS_FILE
    if target.is_file() and file_digests(target)[0] == release.WEIGHTS_SHA256:
        print(f"{target.name} already installed and verified")
        return release.WEIGHTS_SHA256

    archive = root / "BFREE_dino2reg4.zip"
    if local_zip is not None:
        shutil.copyfile(local_zip, archive)
    elif not archive.is_file():
        print(f"Downloading {release.ARCHIVE_URL}")
        download(release.ARCHIVE_URL, archive, show_progress=True)
    md5 = file_digests(archive)[1]
    if md5 != release.ARCHIVE_MD5:
        archive.unlink()
        raise VerificationError(f"archive MD5 {md5} != published {release.ARCHIVE_MD5}")

    wanted = {f"{release.MODEL_DIR}/{release.WEIGHTS_FILE}", f"{release.MODEL_DIR}/config.yaml"}
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        if not wanted <= names:
            raise VerificationError(f"archive lacks {sorted(wanted - names)}")
        for name in wanted:  # only the known members: never extract arbitrary paths
            out = root / name
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
    actual = file_digests(target)[0]
    if actual != release.WEIGHTS_SHA256:
        target.unlink()
        raise VerificationError(f"checkpoint SHA-256 {actual} != pinned {release.WEIGHTS_SHA256}")
    archive.unlink()
    print(f"Verified {target.name} (SHA-256 {actual[:16]}...)")
    return actual


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--zip", type=Path, help="an already downloaded BFREE_dino2reg4.zip")
    args = parser.parse_args(argv)
    root: Path = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    try:
        sha = install(root, args.zip)
    except VerificationError as exc:
        print(f"Verification failed: {exc}", file=sys.stderr)
        return 1
    except (URLError, HTTPError, TimeoutError) as exc:
        print(f"Download failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    manifest = {
        "schema_version": 1,
        "weights": {"path": f"{release.MODEL_DIR}/{release.WEIGHTS_FILE}", "sha256": sha,
                    "archive_md5": release.ARCHIVE_MD5},
        "upstream": {"repo": release.UPSTREAM_REPO, "commit": release.UPSTREAM_COMMIT},
        "licence": release.LICENCE_NOTE,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = root / release.MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"B-Free installed at {root}. Enable with BFREE_ENABLED=true and BFREE_MODEL_PATH={root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
