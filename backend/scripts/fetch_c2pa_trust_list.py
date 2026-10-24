"""Install the official C2PA Trust List into backend/models/c2pa.

Content Credentials whose signing certificate chains to a root on this list are
reported TRUSTED; without it an intact manifest can only be VALID. The list is
published by the C2PA Conformance Program (CC-BY-4.0) and changes as members
are admitted, so it is fetched rather than pinned: re-run this to update it.
The commit it came from and its SHA-256 are recorded in manifest.json.

Usage (from backend/, with the venv python):

    python scripts/fetch_c2pa_trust_list.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = "c2pa-org/conformance-public"
FILES = ("C2PA-TRUST-LIST.pem", "C2PA-TRUST-LIST.json")
ROOT = Path(__file__).resolve().parents[1] / "models" / "c2pa"
UA = "AIMD-c2pa-trust-list/1.0"


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main() -> int:
    commit = json.loads(get(f"https://api.github.com/repos/{REPO}/commits/main"))["sha"]
    ROOT.mkdir(parents=True, exist_ok=True)
    record = {"source": f"https://github.com/{REPO}/tree/{commit}/trust-list", "commit": commit,
              "licence": "CC-BY-4.0", "fetched_at": datetime.now(timezone.utc).isoformat(), "files": {}}
    for name in FILES:
        data = get(f"https://raw.githubusercontent.com/{REPO}/{commit}/trust-list/{name}")
        if name.endswith(".pem") and b"BEGIN CERTIFICATE" not in data:
            print(f"{name}: not a PEM bundle; nothing installed", file=sys.stderr)
            return 1
        (ROOT / name).write_bytes(data)
        record["files"][name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    (ROOT / "manifest.json").write_text(json.dumps(record, indent=2))
    anchors = (ROOT / FILES[0]).read_text(encoding="utf-8").count("BEGIN CERTIFICATE")
    print(f"Installed the C2PA Trust List ({anchors} root certificates) from {REPO}@{commit[:7]} into {ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
