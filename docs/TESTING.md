# Testing

## Running

From `backend/`, with the repository-root virtualenv:

```bash
../venv/Scripts/python.exe -m pytest -q -p no:cacheprovider
```

Last full run (2026-10-20): **310 passed, 1 skipped**, ~95 s on CPU. The skip is
`test_face_region_analyzer.py`'s real-inference test, which needs
`FACE_LANDMARKER_TEST_IMAGE` and the MediaPipe model asset (see below).

Subsets:

```bash
../venv/Scripts/python.exe -m pytest tests/unit -q
../venv/Scripts/python.exe -m pytest tests/integration -q
```

A number above is only valid for the date given. Re-run before quoting it.

## Isolation from real evidence

`tests/conftest.py` sets `DATABASE_URL` and every storage directory to a fresh
temp directory **before** `app` is imported, then a session fixture aborts the
whole run if the database or any storage directory still resolves to the real
store. The suite therefore never writes into `backend/aimd.db`, `uploads/`,
`processed/` or `reports/`. Provider API keys are blanked, so no test can reach
an external service.

Do not import `app` from a module that pytest loads before `conftest.py`.

## Markers and fixtures that change behaviour

| Marker | Default for unmarked tests | With the marker |
|---|---|---|
| `real_auth` | An autouse fixture overrides `get_current_user` with an **investigator** identity, so tests exercise their subject rather than asserting 401. | The real dependency runs. Use for anything asserting anonymous / wrong-role behaviour. |
| `rate_limited` | The limiter is **disabled** (every TestClient request shares one client address, so the suite would otherwise trip its own limits). | The limiter is on and its counters are reset before and after the test. |
| `slow` | — | Informational. |

Because of the auth override, ordinary integration tests cannot notice a router
losing its auth dependency. `tests/integration/test_route_authentication.py`
covers that by walking the real route table with no token.

## What the suite guards (by area)

| Area | Files |
|---|---|
| Authentication, roles, key handling, bootstrap admin | `integration/test_auth.py`, `integration/test_route_authentication.py` |
| Rate limits, 429 envelope, fail-closed config | `integration/test_rate_limits.py` |
| Pagination bounds | `integration/test_pagination_bounds.py` |
| Error envelope, request ids, no exception text leakage | `integration/test_error_envelope.py` |
| Path confinement of stored artefacts | `integration/test_security_path_confinement.py` |
| Ingestion: magic bytes, hashing | `unit/test_media_ingestion.py`, `unit/test_hash_generator.py` |
| Detector availability vs execution status; no fabricated availability in `/health`; no numeric overall confidence; YOLO cannot drive an assessment | `unit/test_detector_availability_semantics.py`, `unit/test_signal_family_map.py` |
| Individual detectors | `unit/test_*_detector.py`, `unit/test_face_region_analyzer.py`, `unit/test_manipulation_mask_generator.py` |
| TruFor runner (missing checkpoint ⇒ UNAVAILABLE, never simulated) | `unit/test_trufor_runner.py`, `unit/test_trufor_detector.py` |
| Pipeline and API end-to-end (synthetic fixtures, sandboxed DB) | `integration/test_analysis_pipeline.py`, `integration/test_live_api_e2e.py`, `integration/test_media_api.py` |
| Accounts, ownership isolation, sign-up, quota, delete | `integration/test_auth.py`, `integration/test_ownership.py` |
| Migrations, audit chain, forensic record | `integration/test_migrations.py`, `unit/test_audit_log.py`, `integration/test_forensic_record.py` |
| Reports, evidence, provenance, similarity, source tracing | `unit/test_report_generator.py`, `unit/test_evidence_service.py`, `unit/test_provenance_graph.py`, `unit/test_similarity_engine.py`, `unit/test_source_tracer.py` |

## Optional real-model tests

Skipped unless the asset is present; a skipped test is not a passed one.

- MediaPipe face landmarker: set `FACE_LANDMARKER_TEST_IMAGE` to an image with a
  face and install the model asset.

## Known warnings

- `slowapi` uses `asyncio.iscoroutinefunction` (deprecated, removed in Python
  3.16). Harmless on 3.14; must be resolved before upgrading.
