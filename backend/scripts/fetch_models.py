"""Download and verify the TruFor code and weights into backend/models/trufor.

TruFor is licensed for research and non-profit use only, so it is fetched from
its authors at setup time and never committed. Every file is checked against the
hashes pinned in ``app/services/detection/models/trufor_release.py``; a file
that does not match is deleted, not installed.

Usage (from backend/, with the venv python):

    python scripts/fetch_models.py                      # code + official weights
    python scripts/fetch_models.py --samples            # also the two demo images
    python scripts/fetch_models.py --weights-zip D:/Downloads/TruFor_weights.zip
    python scripts/fetch_models.py --cocoglide-zip D:/Downloads/CocoGlide.zip

The official weights host (www.grip.unina.it) has been unreachable since
mid-2026 (grip-unina/TruFor issues #34, #35). When the download fails, get
``TruFor_weights.zip`` from a mirror and pass it with ``--weights-zip``: it is
still verified against the authors' published MD5, so a mirror cannot
substitute different weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.services.detection.models import trufor_release as release  # noqa: E402

DEFAULT_ROOT = BACKEND / "models" / "trufor"
CHUNK = 1024 * 1024
TIMEOUT_S = 60


class VerificationError(RuntimeError):
    pass


# ── Hashing and download ─────────────────────────────────────────────────────

def file_digests(path: Path) -> tuple[str, str]:
    """(sha256, md5) of a file, streamed."""
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            sha.update(chunk)
            md5.update(chunk)
    return sha.hexdigest(), md5.hexdigest()


def download(url: str, destination: Path, *, show_progress: bool = False) -> None:
    """
    Stream ``url`` to ``destination`` through a ``.part`` file.

    A ``.part`` left by an interrupted run is resumed with an HTTP Range request
    when the server supports it, so a slow connection does not have to start a
    280 MB download over. Verification happens on the finished file, never here.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0

    request = Request(url, headers={"User-Agent": "aimd-fetch-models/1"})
    if offset:
        request.add_header("Range", f"bytes={offset}-")

    with urlopen(request, timeout=TIMEOUT_S) as response:
        resumed = offset and response.status == 206
        mode = "ab" if resumed else "wb"
        total = response.headers.get("Content-Length")
        total = int(total) + (offset if resumed else 0) if total else None
        written = offset if resumed else 0
        with partial.open(mode) as output:
            while chunk := response.read(CHUNK):
                output.write(chunk)
                written += len(chunk)
                if show_progress and total:
                    print(f"\r  {written / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
        if show_progress:
            print()
    os.replace(partial, destination)


def fetch_pinned(relpath: str, expected_sha256: str, root: Path) -> str:
    """Install one pinned upstream file below ``root``; returns its SHA-256."""
    target = root / relpath
    if target.is_file():
        if file_digests(target)[0] == expected_sha256:
            return expected_sha256
        target.unlink()  # stale or tampered: replace it, never keep it

    download(release.UPSTREAM_RAW_BASE + relpath, target)
    actual = file_digests(target)[0]
    if actual != expected_sha256:
        target.unlink()
        raise VerificationError(f"{relpath}: SHA-256 {actual} != pinned {expected_sha256}")
    return actual


# ── Archives ─────────────────────────────────────────────────────────────────

def _safe_member_path(dest: Path, name: str) -> Path | None:
    """Where an archive member may be written, or None if it would escape ``dest``."""
    member = PurePosixPath(name.replace("\\", "/"))
    if not member.parts or member.is_absolute() or ".." in member.parts:
        return None
    if ":" in member.parts[0]:  # a Windows drive letter
        return None
    target = (dest / Path(*member.parts)).resolve()
    try:
        target.relative_to(dest.resolve())
    except ValueError:
        return None
    return target


def extract_checkpoint(archive: Path, destination: Path) -> str:
    """Extract ``trufor.pth.tar`` from the weights archive; returns its SHA-256."""
    with zipfile.ZipFile(archive) as bundle:
        members = [
            m for m in bundle.infolist()
            if not m.is_dir() and PurePosixPath(m.filename).name == release.CHECKPOINT_NAME
        ]
        if len(members) != 1:
            raise VerificationError(
                f"expected exactly one {release.CHECKPOINT_NAME} in the archive, "
                f"found {len(members)}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".part")
        digest = hashlib.sha256()
        with bundle.open(members[0]) as source, partial.open("wb") as output:
            while chunk := source.read(CHUNK):
                output.write(chunk)
                digest.update(chunk)
    os.replace(partial, destination)
    return digest.hexdigest()


def extract_all(archive: Path, dest: Path) -> int:
    """Extract every member, refusing any path that escapes ``dest``."""
    count = 0
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = _safe_member_path(dest, member.filename)
            if target is None:
                raise VerificationError(f"archive member escapes the destination: {member.filename!r}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, CHUNK)
            count += 1
    return count


# ── Steps ────────────────────────────────────────────────────────────────────

def install_code(root: Path, manifest: dict) -> None:
    manifest["code"] = {
        relpath: fetch_pinned(relpath, sha, root)
        for relpath, sha in release.CODE_FILES.items()
    }
    print(f"Verified {len(release.CODE_FILES)} TruFor code files at commit {release.UPSTREAM_COMMIT[:12]}")


def install_samples(root: Path, manifest: dict) -> None:
    samples = {}
    for upstream, sha in release.SAMPLE_FILES.items():
        local = f"{release.SAMPLES_RELPATH}/{PurePosixPath(upstream).name}"
        target = root / local
        if not (target.is_file() and file_digests(target)[0] == sha):
            download(release.UPSTREAM_RAW_BASE + upstream, target)
            if file_digests(target)[0] != sha:
                target.unlink()
                raise VerificationError(f"{upstream}: SHA-256 mismatch")
        samples[local] = sha
    manifest["samples"] = samples
    print(f"Verified {len(samples)} sample images")


def install_weights_from_archive(root: Path, archive: Path, source: str, manifest: dict) -> None:
    zip_sha, zip_md5 = file_digests(archive)
    if zip_md5 != release.WEIGHTS_ZIP_MD5:
        raise VerificationError(
            f"weights archive MD5 {zip_md5} != official {release.WEIGHTS_ZIP_MD5}; "
            "this is not the authors' TruFor_weights.zip"
        )
    checkpoint = root / release.WEIGHTS_RELPATH
    ckpt_sha = extract_checkpoint(archive, checkpoint)
    manifest["weights"] = {
        "path": release.WEIGHTS_RELPATH,
        "sha256": ckpt_sha,
        "size_bytes": checkpoint.stat().st_size,
        "verification": "official_archive_md5",
        "archive": {"source": source, "md5": zip_md5, "sha256": zip_sha},
    }
    print(f"Verified weights archive (MD5 {zip_md5}); checkpoint SHA-256 {ckpt_sha}")


def install_weights_from_checkpoint(root: Path, checkpoint_src: Path, manifest: dict) -> None:
    """
    A bare checkpoint has no published hash to check against. It is installed
    only on explicit request and recorded as unverified, and the runner then
    refuses to unpickle anything beyond plain tensors from it.
    """
    checkpoint = root / release.WEIGHTS_RELPATH
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(checkpoint_src, checkpoint)
    ckpt_sha = file_digests(checkpoint)[0]
    manifest["weights"] = {
        "path": release.WEIGHTS_RELPATH,
        "sha256": ckpt_sha,
        "size_bytes": checkpoint.stat().st_size,
        "verification": "unverified",
        "archive": None,
    }
    print(f"Installed UNVERIFIED checkpoint; SHA-256 {ckpt_sha}")


def install_cocoglide(root: Path, archive: Path, source: str, manifest: dict) -> None:
    # No hash is published for CocoGlide; record what was installed so an
    # evaluation can say which copy it ran on.
    zip_sha, zip_md5 = file_digests(archive)
    dest = root / release.COCOGLIDE_RELPATH
    count = extract_all(archive, dest)
    manifest["cocoglide"] = {
        "path": release.COCOGLIDE_RELPATH,
        "files": count,
        "verification": "unverified",
        "archive": {"source": source, "md5": zip_md5, "sha256": zip_sha},
    }
    print(f"Extracted CocoGlide ({count} files, archive SHA-256 {zip_sha})")


def _download_archive(url: str, root: Path, name: str) -> Path:
    target = root / "downloads" / name
    print(f"Downloading {url}")
    download(url, target, show_progress=True)
    return target


def _weights_help(root: Path) -> str:
    return (
        "\nThe TruFor weights could not be downloaded. The official host has been\n"
        "unreachable since mid-2026 (github.com/grip-unina/TruFor/issues/35).\n"
        "Download TruFor_weights.zip from a mirror in your browser, then run:\n\n"
        "    python scripts/fetch_models.py --weights-zip <path-to>/TruFor_weights.zip\n\n"
        f"It is checked against the official MD5 {release.WEIGHTS_ZIP_MD5} and\n"
        f"installed to {root / release.WEIGHTS_RELPATH}.\n"
    )


# ── Manifest ─────────────────────────────────────────────────────────────────

def load_manifest(root: Path) -> dict:
    path = root / release.MANIFEST_NAME
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


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


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="TruFor install directory (default: backend/models/trufor)")
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument("--weights-zip", type=Path,
                         help="use an already downloaded TruFor_weights.zip")
    weights.add_argument("--weights-url", default=release.WEIGHTS_URL,
                         help="download the weights archive from this URL instead")
    weights.add_argument("--checkpoint", type=Path,
                         help="install a bare trufor.pth.tar (needs --accept-unverified)")
    weights.add_argument("--skip-weights", action="store_true", help="install the code only")
    parser.add_argument("--accept-unverified", action="store_true",
                        help="allow --checkpoint, which has no published hash")
    parser.add_argument("--samples", action="store_true", help="also fetch the upstream demo images")
    coco = parser.add_mutually_exclusive_group()
    coco.add_argument("--cocoglide", action="store_true", help="download the CocoGlide set")
    coco.add_argument("--cocoglide-zip", type=Path, help="use an already downloaded CocoGlide.zip")
    args = parser.parse_args(argv)

    root: Path = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(root)
    exit_code = 0

    try:
        install_code(root, manifest)
        if args.samples:
            install_samples(root, manifest)

        if args.checkpoint:
            if not args.accept_unverified:
                parser.error("--checkpoint has no published hash; add --accept-unverified")
            install_weights_from_checkpoint(root, args.checkpoint, manifest)
        elif args.weights_zip:
            install_weights_from_archive(root, args.weights_zip, "local archive", manifest)
        elif not args.skip_weights:
            checkpoint = root / release.WEIGHTS_RELPATH
            recorded = (manifest.get("weights") or {}).get("sha256")
            if checkpoint.is_file() and recorded and file_digests(checkpoint)[0] == recorded:
                print("Weights already installed and match the manifest")
            else:
                try:
                    archive = _download_archive(args.weights_url, root, "TruFor_weights.zip")
                except (URLError, HTTPError, TimeoutError, OSError) as exc:
                    print(f"Weights download failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                    print(_weights_help(root), file=sys.stderr)
                    exit_code = 2
                else:
                    install_weights_from_archive(root, archive, args.weights_url, manifest)
                    archive.unlink()

        if args.cocoglide_zip:
            install_cocoglide(root, args.cocoglide_zip, "local archive", manifest)
        elif args.cocoglide:
            try:
                archive = _download_archive(release.COCOGLIDE_URL, root, "CocoGlide.zip")
            except (URLError, HTTPError, TimeoutError, OSError) as exc:
                print(f"CocoGlide download failed: {type(exc).__name__}: {exc}\n"
                      "Download CocoGlide.zip from a mirror and pass --cocoglide-zip.",
                      file=sys.stderr)
                exit_code = 2
            else:
                install_cocoglide(root, archive, release.COCOGLIDE_URL, manifest)
                archive.unlink()
    except VerificationError as exc:
        print(f"Verification failed: {exc}", file=sys.stderr)
        exit_code = 1
    finally:
        write_manifest(root, manifest)

    if exit_code == 0:
        if (root / release.WEIGHTS_RELPATH).is_file():
            print(f"TruFor installed at {root}")
            print(f"Enable it in .env with TRUFOR_ENABLED=true and TRUFOR_MODEL_PATH={root}")
        else:
            print(f"TruFor code installed at {root}; the weights are NOT installed, "
                  "so the detector will report UNAVAILABLE.")
            print("Install them with --weights-zip <path-to>/TruFor_weights.zip")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
