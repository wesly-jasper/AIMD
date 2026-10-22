"""
Write the API's OpenAPI spec to docs/openapi.json.

Usage (from backend/, with the venv python):
    python scripts/generate_openapi.py
"""
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.main import app  # noqa: E402

out = BACKEND.parent / "docs" / "openapi.json"
out.write_text(json.dumps(app.openapi(), indent=2), encoding="utf-8")
print(f"Wrote {out}")
