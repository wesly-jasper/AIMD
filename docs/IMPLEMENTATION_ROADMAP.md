# AIMD Sentinel — Implementation Roadmap

**Living checklist.** Updated at the end of every phase. Every item is either `[ ]` not done, `[~]` in progress, or `[x]` done **and verified by an executed check whose output is recorded in the phase's verification note**.

Baseline recorded 2026-10-20: **132 passed, 5 failed, 1 skipped**. The 5 failures are pre-existing (detection router commented out of `app/main.py`). Every phase below must report its own test delta against this.

Hackathon deadline: **2026-10-30**. Priority rule: *one reliable end-to-end image investigation* beats many partial modalities.

---

## Phase 0 — Audit & baseline  ✅ COMPLETE

- [x] Environment inventory (Python 3.14.2, torch 2.9.1+cu126, CUDA True, RTX 3050) — executed
- [x] Dependency gap analysis; installed `reportlab`, `openai`, `exifread`, `pytest-timeout` — executed
- [x] Baseline test suite recorded: 132 / 5 / 1 — executed
- [x] Checkpoint inventory: only `face_landmarker.task` + `yolov8n.pt` — executed
- [x] Live inference smoke test: YOLOv8n predicts; MediaPipe FaceLandmarker loads — executed
- [x] Confirmed **no** working learned forgery/AI-generation detector exists (TruFor disabled, no checkpoint)
- [x] Confirmed **no** Nebius credential configured anywhere
- [x] Critical vulnerability confirmed **by live exploit**: unauthenticated arbitrary file read
- [x] Confirmed the hardcoded `sha256` in `investigation.py:129` is `sha256(b"")`
- [x] `docs/PROJECT_AUDIT.md` written
- [x] `docs/IMPLEMENTATION_ROADMAP.md` written (this file)
- [x] Test artifacts created during the exploit proof removed from `aimd.db` and temp — executed

**Verification note:** see `docs/PROJECT_AUDIT.md` §1 for recorded command output.

---

## Phase 1 — P0: Stop-and-fix security & integrity  ✅ COMPLETE

Rationale for ordering: a forensics product that persists a fabricated cryptographic hash and permits unauthenticated arbitrary file read cannot earn a feature until both are closed.

**Architectural decision taken here, which later phases depend on: evidence is addressed by identity (`media_id`), never by location (`file_path`).** No API endpoint accepts a filesystem path. Every path that reaches the filesystem is derived from the stored media record and passed through `resolve_evidence_path()`, which confines it to the configured evidence roots after symlink resolution.

### 1a — Arbitrary file read (SEC-1)  ✅
- [x] Confinement primitive implemented and enforced — `app/core/security.py:resolve_evidence_path` + `evidence_roots` [EXECUTED]
- [x] `POST /api/v1/source/trace` no longer accepts a client-supplied filesystem path — `file_path` removed from the schema
- [x] `POST /api/v1/fingerprint/generate` and `/similarity/search` same
- [x] `GET /api/v1/media/{id}/download` re-validates containment before serving
- [x] Regression test: path traversal attempt is rejected (`../../../../etc/passwd`, `..\..\..\..\Windows\win.ini`)
- [x] Regression test: absolute path outside the evidence roots is rejected (`C:/Windows/win.ini`, `/etc/passwd`)
- [x] Rejection message does not echo the probed path back to the client
- [x] Re-ran the original exploit against a **real uvicorn server** — recorded below

A third instance of this vulnerability class was found during the fix: `POST /api/v1/detection/analyze` also took `request.file_path`. Mounting the commented-out router to fix the baseline failures would therefore have *introduced* a new arbitrary-path primitive. Its contract was corrected before the router was mounted.

### 1b — Fabricated SHA-256 (SEC-2)  ✅
- [x] Hardcoded empty-string digest `e3b0c442...7852b855` removed from `investigation.py`
- [x] The whole `repo.create_media` / `repo.create_analysis` fabrication block deleted — unknown evidence now 404s instead of being invented
- [x] No endpoint may invent a hash; hashes come only from a real digest of real bytes
- [x] Repo-wide check: the constant survives only in a test fixture and a build artefact, not in any production path [EXECUTED]
- [x] Test asserts the empty-string digest never appears in a persisted record

### 1c — Detection router honesty  ✅
- [x] `detection_router` mounted (`main.py:19,81`) — no commented-out surface
- [x] 5 baseline failures resolved **legitimately**: by mounting the router and correcting its contract. Three of those tests asserted the old path-based contract and were rewritten to the identity-based contract rather than loosened.
- [x] `analyze_media` no longer mints a `uuid4` `analysis_id` implying a persisted run that never existed; it reports `"persisted": false`
- [x] `media_type` is read from the stored record (determined from magic bytes at ingest), not accepted from the client

### 1d — Error envelope  ✅
- [x] `AIMDException` and catch-all handlers attached (`main.py`) — they were defined but never registered
- [x] Stopped returning raw exception text / absolute paths to clients (`str(exc)` → a fixed string)
- [x] `HTTPException` and `RequestValidationError` handlers attached, completing the unification that 1d deferred. Registered on `StarletteHTTPException` so FastAPI's subclass is covered. The API now has exactly one error shape; `test_detection_api.py` was updated off `{"detail": ...}`.
- [x] `validation_exception_handler` preserves per-field errors under `error.fields` as `{location, message, type}` and **drops Pydantic's `input` value**, which echoes caller-supplied content back
- [x] **`request_id` was a fabricated correlation handle.** Each handler did `getattr(request.state, "request_id", str(uuid.uuid4())[:8])` while nothing in the application ever set it — `set_request_context` had no caller. Every log line read `req=-` and every error body carried a fresh id present in no log line, sending an operator to search for an id that was never written. Fixed with `RequestIdMiddleware`; `_request_id()` returns `"unavailable"` rather than inventing one.
- [x] An inbound `X-Request-ID` is ignored deliberately — honouring it would let a caller choose the key its own requests are filed under in the audit log, including reusing another caller's id
- [x] The logging ContextVar is **not** cleared on the way out: isolation already comes from the ASGI server's task-per-request context copy, and clearing would blank the id while an exception is still propagating to the 500 handler
- [x] Tests (`tests/integration/test_error_envelope.py`, 6): the 500 body leaks none of `C:\Users`, `Administrator`, `id_rsa`, `postgresql://`, the password, the host IP, `RuntimeError`, `Traceback`, `site-packages`; the body's `request_id` equals both the `X-Request-ID` header **and** a real log record's `request_id` attribute, while the withheld detail *is* in the log; a 422 names the field without echoing `'; DROP TABLE--`

### 1e — Fabricated detector availability in `/health`  ✅
- [x] Removed the hardcoded map (`"image_detector": "available"`, `"object_detector": "unavailable"`, …) that was never consulted from the registry
- [x] Replaced with a **derived** list of actually-constructed detectors, plus an explicit `"availability_introspection": "NOT_IMPLEMENTED"` rather than invented per-detector availability
- [x] Collapsed the two independent copies of the health logic (`main.py` and the unmounted `endpoints/health.py`) into one source of truth, `app/core/health.py` — the duplication is why the fabrication survived a prior fix
- [x] Two test assertions that checked the *fabricated* values were deleted, not restored

**Acceptance — met:**

| Check | Baseline | After |
|---|---|---|
| Test suite | 132 passed, **5 failed**, 1 skipped | **162 passed, 0 failed, 1 skipped** (`PYTEST_EXIT=0`) [EXECUTED] |
| Exploit chain | canary exfiltrated | all 4 steps rejected [EXECUTED] |

Live exploit re-run — `python -I scripts/verify_sec1_fix.py` [EXECUTED]:

```
[1] POST /api/v1/source/trace            -> 404
[2] GET  /api/v1/media/{id}/download     -> 404
[3] GET  /api/v1/media/{id}              -> 404
[4] POST /api/v1/detection/analyze       -> 404

SEC-1 VERIFIED CLOSED: exploit chain failed at every step; canary not exfiltrated.
VERIFY_EXIT=0
```

Confirmed by grep that zero `request.file_path` / `request.media_type` usages remain in `app/`; the only filesystem reads are four `resolve_evidence_path(media.file_path)` call sites [EXECUTED].

---

## Phase 2 — P1: Core contracts  ✅

- [x] `DetectorAvailability` enum: `AVAILABLE` / `UNAVAILABLE`
- [x] `DetectorExecutionStatus` enum: `COMPLETED` / `INCONCLUSIVE` / `FAILED` / `SKIPPED`
- [x] These are **independent fields** — a detector may be `AVAILABLE` + `FAILED`, or `COMPLETED` + `INCONCLUSIVE`
- [x] Neither is ever collapsed into a fake/real label
- [x] `BaseDetector` contract gains `detector_id`, `version`, `supported_media_types`, `availability()`
- [x] Four distinct non-result situations, each a named method rather than one overloaded return: `unavailable()` (capability absent → UNAVAILABLE+SKIPPED), `failed()` (capability present, run broke or input unprocessable → AVAILABLE+FAILED), `skipped()` (working but not attempted: unsupported media type, disabled by config → AVAILABLE+SKIPPED), `inconclusive()` (ran, no determination → AVAILABLE+INCONCLUSIVE, regions preserved)
- [x] `produced_observations` deliberately **includes** INCONCLUSIVE; `supports_determination` is the stricter property (AVAILABLE + COMPLETED only)
- [x] `score_semantics` field so a raw model score is never read as a probability: `RAW_MODEL_SCORE` / `CALIBRATED_PROBABILITY` / `HEURISTIC_SEVERITY` / `NOT_APPLICABLE`. Only TruFor is entitled to `RAW_MODEL_SCORE`; **nothing** is entitled to `CALIBRATED_PROBABILITY` without a recorded calibration procedure. A non-result carries `confidence=0.0` with `NOT_APPLICABLE` so the zero cannot be read as "measured very little anomaly".
- [x] Assessment categories fixed to: `NO_SIGNIFICANT_ANOMALIES_DETECTED`, `SUSPICIOUS`, `LIKELY_MANIPULATED`, `LIKELY_AI_GENERATED`, `INCONCLUSIVE`
- [x] **Removed the blended `overall_confidence`** produced by averaging unrelated detectors — invalid on three independent grounds (incomparable scales, non-independence, and absence counted as evidence, since a detector that never ran contributed 0.0 and could push the result below the "CLEAN" threshold)
- [x] Replaced with per-detector attributed scores plus an ordinal `evidence_strength`, and an assessment derived from the *structure* of the observations
- [x] **Signal families** (`_SIGNAL_FAMILY`): detectors sharing a physical basis are not independent corroboration, so corroboration is counted across families and never within one. The `non_evidential` family keeps object detectors out of assessments in either direction.
- [x] Fix: detector identity is never swapped on fallback — **four** silent-substitution sites corrected (returning detector B's result under detector A's identity)
- [x] Fix: TruFor weight-key mismatch (`"TruForDetector"` vs `"TruFor"`)
- [x] Fabricated capability claims removed from names and docstrings — e.g. `NeuralSpeechDetector` → `SpectralSpeechHeuristicDetector` (it runs no network)
- [x] `trufor_runner.predict()` no longer emits `'assessment': 'CLEAN'` from a sub-threshold raw score. A model wrapper cannot see which other detectors ran or whether any failed, so it reports a score and the threshold; labelling happens once, downstream, from the status fields. Its `metadata['status'] = 'MODEL_AVAILABLE'` third status vocabulary was removed, as was the dead `MODEL_AVAILABLE`/`MODEL_UNAVAILABLE`/`INTERNET_SEARCH_UNAVAILABLE` block in `constants.py` (imported by nothing, and `MODEL_AVAILABLE`'s value `"available"` did not even match the literal the runner emitted).
- [x] `metadata['model_path']` (an absolute host path, passed through verbatim into API responses and reports) replaced with `model_file`, the checkpoint filename — which is the provenance an examiner actually needs
- [x] **Video engine:** `NO_SIGNIFICANT_ANOMALIES_DETECTED` is now coverage-based, not threshold-based — it requires that every sampled frame was analysed by every frame detector with no failures. The old `video_confidence < 0.20 → CLEAN` branch and the `tracked_confidence_boost` are gone.
- [x] Asymmetry of missing detectors made explicit and deliberate: a positive observation stands even when other detectors failed (missing data cannot retract a measured anomaly), but an *absence* of positive observations does **not** stand when coverage was incomplete. One override: when >50% of frame-detector invocations failed, even a positive proportion is not representative → INCONCLUSIVE.
- [x] Dead `VideoDetectionEngine._inconclusive` deleted — it had no caller, and a method whose name advertises a status that path never emits invites a future caller to use it. Its bug rationale was preserved on `_non_result`, where the mechanism that prevents the bug now lives.
- [x] Tests for every state combination (`test_detector_availability_semantics.py`, `test_detection_schema.py`, `test_signal_family_map.py`, `test_video_detector.py`)
- [x] `test_signal_family_map.py` pins `_SIGNAL_FAMILY` against the detector list `detection_container` really builds, so an image detector added without a family assignment fails a test instead of only emitting a log warning that nobody reads

**Acceptance — met:** no averaged cross-detector confidence anywhere; availability and execution independently representable and tested.

**Scope note:** `_SIGNAL_FAMILY` covers image detectors only. `BaselineVideoDetector` and `BaselineAudioAnalyzer` are deliberately absent — neither is ever passed to `ImageDetectionEngine`, and `VideoDetectionEngine` aggregates its own frame detectors by coverage rather than by family. An unlisted detector is not silently trusted: `_family` returns `"unclassified"`, which shares one family with every other unlisted detector and so cannot corroborate independently.

---

## Phase 3 — P1: Authentication & authorisation  🟡 PARTIAL

**Audit finding that changed the scope:** `get_current_user` was consumed by exactly **one** route, `/auth/me`. No other route was protected at all. The fail-open fallback was therefore a secondary problem — upload, evidence download, analysis and all four report formats answered requests with no credentials whatsoever.

- [x] Removed fail-open `anonymous_analyst` fallback. `get_current_user` now returns `User`, not `Optional[User]` — receiving a value from it means an authenticated identity exists.
- [x] `auto_error=True`; missing token → 401 raised by the scheme itself
- [x] Signing key read from `AIMD_JWT_SECRET` only; **no committed default**. Both committed copies removed (`auth.py:SECRET_KEY` and an unused `settings.secret_key` that a deployment could override to no effect).
- [x] The committed placeholder is **refused by value** — it is permanently in git history, so it is treated as compromised forever even if a `.env` still carries it. Keys under 32 characters are refused.
- [x] Startup refuses to run with an absent key outside explicit dev mode. In development an ephemeral per-process key is generated, with the consequence (tokens die on restart; workers disagree) logged rather than hidden.
- [x] **The role is read from the user store on every request, never from the token's `role` claim.** Previously a role could not be revoked before the token's 24-hour expiry. Deactivating an account now invalidates its live tokens on the next request.
- [x] Closed open self-registration: `/register` was unauthenticated and took `role` from the body, so one POST created an administrator. It now requires `require_role("admin")`, validates `role` against `ROLES`, enforces a 12-character minimum, and returns 409 rather than overwriting an existing account.
- [x] **Published credentials removed.** `admin`/`admin123` and `investigator`/`investigator123` were seeded in committed source. The store now seeds only from `AIMD_BOOTSTRAP_ADMIN_USERNAME` / `AIMD_BOOTSTRAP_ADMIN_PASSWORD`; with neither set it is empty and every login is refused.
- [x] Login does not enumerate usernames: wrong password, unknown user and deactivated account return one message, and all three cost one bcrypt verification (`_DUMMY_HASH`) so they are indistinguishable in latency too
- [x] `require_role()` returns **403** for an authenticated caller with the wrong role, distinct from 401 for no identity — the caller needs to know whether to re-authenticate or stop
- [x] Auth applied **per router** in `main.py` (`dependencies=authenticated`), so routes added later are protected by default and exempting one requires an explicit decision. 23 routes protected; public: `/health`, `/api/v1/health`, `/api/v1/auth/login`, and FastAPI's schema pages.
- [x] Tests: `test_auth.py` (26) and `test_route_authentication.py` (50). The latter **walks the real route table** with no token and with a forged token, and checks the public-path exemption list against reality so a stale entry cannot silently excuse a future route of the same name.
- [x] `conftest.authenticated_by_default` supplies an **investigator** (not admin) identity to tests that are not about auth, so a test that strays onto an admin route fails rather than passing with rights it would not have in production. Files about auth opt out with `@pytest.mark.real_auth`.
- [x] Persist users in the DB — done in Phase 4 (`users` table).
- [x] Activate rate limiting (`app/core/rate_limit.py`). The limiter had been constructed without `SlowAPIMiddleware` and no route was decorated, so nothing was limited. Now: default `RATE_LIMIT_DEFAULT` on every route via the middleware; `RATE_LIMIT_LOGIN` on login, `RATE_LIMIT_UPLOAD` on upload, `RATE_LIMIT_ANALYSIS` on analysis/detection/source-trace. Keyed per (client address, **endpoint**) — slowapi's default per-URL keying let a caller vary a path parameter for a fresh budget. **Moving window** — the default fixed window admits 2× the limit across a boundary. 429 uses the standard envelope with `Retry-After`, sits inside CORS (readable by the browser). An unparseable limit string is fatal at import (slowapi would log it and install no limit). Tests: `tests/integration/test_rate_limits.py` (`rate_limited` marker; the limiter is off for all other tests).
  - Caveats (documented in the module): in-memory per-worker counters (N workers ⇒ N× ceiling; Redis for shared storage); behind a proxy run uvicorn with `--proxy-headers --forwarded-allow-ips=<proxy>` — `X-Forwarded-For` is deliberately not trusted; no per-account lockout (it would let anyone lock an investigator out); a request body is read before the limit is checked, and malformed (422) bodies are not counted; slowapi 0.1.9 uses `asyncio.iscoroutinefunction`, removed in Python 3.16.
- [x] Bound all pagination — `/analysis/`, `/jobs/`, `/media/` take `limit` 1..200, `offset` ≥ 0 (`tests/integration/test_pagination_bounds.py`)
- [x] `/health` hardening (found while load-testing the default limit): it rebuilt the whole detector container (~0.7 s, MediaPipe) on every anonymous request — now computed once per process; and it returned the raw database exception text to anonymous callers — now `"error"`, details logged.
- [ ] Per-case authorisation (which investigator may see which case) — needs the `Case` entity from Phase 4
- [x] The frontend sends `Authorization: Bearer …` — the new `web/` app (Phase 11); token in sessionStorage, identity from `/auth/me`

---

## Phase 4 — P2: Forensic data model & migrations  ✅ COMPLETE

Recorded 2026-10-22. Suite before: **310 passed, 1 skipped**. After: **352 passed, 0 failed, 1 skipped** (`EXIT=0`) [EXECUTED].

- [x] `Case` table; `media.case_id` FK with **RESTRICT** (deleting a case can never delete its evidence — tested). `POST/GET /api/v1/cases`, `GET /api/v1/cases/{id}`; upload takes an optional `case_id` form field (unknown case → 404).
- [x] `Observation` as a first-class entity: one per detector result that actually observed something, with `polarity` `ANOMALY_OBSERVED | NO_ANOMALY_OBSERVED | INDETERMINATE`, `signal_family`, `score_semantics`. **UNAVAILABLE / FAILED / SKIPPED produce no observation** — they are coverage gaps on the assessment, never negatives.
- [x] `Finding` with explicit links via `finding_observations` (`SUPPORTS | CONTRADICTS`). An anomaly finding links the completed no-anomaly observations as **CONTRADICTS**. A finding cannot cite another analysis's observations (`ValueError`). Inconclusive analyses state no finding. Non-evidential (object) observations are never linked.
- [x] `Assessment` entity: `category`, `evidence_strength`, `rule` (stable id of the decision branch: `no_detections`, `non_evidential_only`, `multi_family_localised`, `multi_family`, `single_family`, `incomplete_coverage`, `full_coverage_no_anomaly`), `rationale`, `coverage_gaps`. The decision logic is unchanged — `_decide` is the old `_compute_final_assessment` body with the branch named; the existing assessment tests still pin it.
- [x] `AuditEvent`, hash-linked (`app/services/audit/audit_log.py`). sha256 over canonical JSON incl. `previous_event_hash`; fixed genesis hash so a second genesis is refused; unique `sequence` + `previous_event_hash` make a racing writer fail and retry instead of forking (tested by forcing the race). `occurred_at` stored as the exact hashed string (a DateTime column does not round-trip on SQLite). Subject ids are deliberately not FKs so a record outlives its subject. Emitted for `USER_CREATED`, `CASE_CREATED`, `EVIDENCE_INGESTED` (only when persisted), `ANALYSIS_STARTED`, `ANALYSIS_FINISHED`. `GET /api/v1/audit/events`, `GET /api/v1/audit/verify` (admin).
  - Limit, stated in the API response and SECURITY.md: tamper-evident, not tamper-proof — a DB writer can recompute the chain, and tail truncation is internally undetectable (tested, to pin the documented limit). Anchor `head_hash` externally.
- [x] `User` table; `UserStore` is DB-backed. `get()` returns a snapshot (mutating it changes nothing — tested); `set_active()` for deactivation. Bootstrap moved from import time to app startup, after migrations, and never overwrites an existing account.
- [x] `AnalysisRun` (`analyses`) gains `requested_by`, `software_version`, `detector_set_version` (sha256 of the (detector, version) pairs the run actually invoked), `config_snapshot` (**allow-list** of analysis settings — a new credential setting cannot leak into it by omission), `duration_ms`. Not backfilled on legacy rows: the values then in force are unknown.
- [x] `DetectorResult` separate availability/execution columns — already present since `a1b2c3d4e5f6`; now also returned by `GET /analysis/{id}` (previously persisted but never exposed, so no client could show them as distinct states).
- [x] `analyses.overall_confidence` **dropped** (`b7e1f0c2d9a4`).
- [x] **Real migrations.** `04e00c1a868b` was `pass`; it now creates the initial schema, idempotently per table (safe to rewrite *because* the old body did nothing). New `b7e1f0c2d9a4` adds the Phase 4 model, idempotent per table/column.
- [x] Fixed a latent defect in `a1b2c3d4e5f6`: its backfill ran unconditionally and would overwrite **recorded** statuses with values guessed from the legacy label on any unstamped `create_all` database. Now backfills only when it added the columns (tested).
- [x] **SQLite rebuild hazard closed.** Dropping a column rebuilds `analyses`; with FKs enforced the DROP cascades. `env.py` disables enforcement for the migration and runs `PRAGMA foreign_key_check` after. Mutation-tested: flipping the pragma to ON makes `test_legacy_evidence_survives_the_table_rebuild` fail with `analyses: 0 == 1` [EXECUTED].
- [x] `Base.metadata.create_all` no longer used at startup: `init_db()` = `alembic upgrade head` (programmatic config — reading alembic.ini would let `fileConfig` replace the app's logging). The test suite now builds its DB through the migrations too.
- [x] `alembic.ini` placeholder `driver://user:pass@...` removed; env.py falls back to `DATABASE_URL`.
- [x] Verified: upgrade from empty → `compare_metadata` reports **0 diffs** vs the models (`test_upgrade_from_empty_matches_the_models_exactly`) [EXECUTED]
- [x] Verified: legacy data survives — a DB at `04e00c1a868b` seeded with media/analysis/detections/regions/fingerprints/evidence/report rows upgrades with every row intact and an empty `foreign_key_check` (`tests/integration/test_migrations.py`, 9 tests) [EXECUTED]
- [x] Verified on a **copy** of the live `aimd.db` (stamped `a1b2c3d4e5f6`, 0 rows): upgraded to `b7e1f0c2d9a4`, all 21 tables, FK check clean; original's sha256 unchanged before/after (`f76dd087…`) [EXECUTED]. **The live DB itself is not yet migrated — the app will upgrade it on next startup.**
- [x] Startup smoke on a scratch DB with a real token, two consecutive startups: migrations ran once; bootstrap admin created then "already exists; unchanged"; login survived the restart; 401 without token on `/cases`; audit chain intact and growing across restarts [EXECUTED].

Also fixed while here (leaks of the Phase 1d class): `POST /media/upload` returned `f"Ingestion error: {exc}"` and `POST /analysis` returned `str(exc)` on 404/500 — the latter includes the absolute evidence path. Both now return fixed strings and log the detail.

**Not done here / carried forward:**
- Per-case authorisation (Phase 3) — the `Case` entity now exists, but there is no membership model; every authenticated user sees every case.
- Evidence strength `CONTRADICTORY` is never emitted: contradicting observations are now *linked*, but the decision rule does not yet weigh them (Phase 7).
- Report rendering from `AuditEvent` rows (Phase 10); no report-generation event is emitted yet.
- The upload response still includes `file_path` (absolute storage path) — Phase 5.

## Phase 5 — P2: Evidence preservation

- [ ] Original evidence stored read-only, separate from derived artifacts
- [ ] Analysis operates on a working copy; originals never written
- [ ] Stored hash re-verified before each analysis run; mismatch → hard stop
- [ ] Reconcile the two independent SHA-256 computations
- [ ] Magic bytes as a real gate; cross-check against `Content-Type`; extension allow-list
- [ ] Size limit enforced **before** buffering the body
- [ ] Duplicate detection via `get_media_by_sha256`
- [ ] Every evidence mutation emits an `AuditEvent`
- [ ] Tests: original unmodified after analysis; tampered file rejected; oversized upload rejected pre-buffer

---

## Phase 6 — P2: Honest image analysis

- [ ] Fix silent fusion truncation (7 heatmaps / 6 weights) in `manipulation_mask_generator.py`
- [ ] Fix structural positive bias in `face_region_analyzer.py`
- [ ] Remove or gate the min-max renormalisation that destroys calibration
- [ ] Remove the `_refine_with_unet` stub and the empty `DeepfakeSegmentationModel`
- [ ] Wire `YOLOv8ObjectDetector` — **regions only**, explicitly never evidence of forgery
- [ ] Replace or document every hardcoded threshold with its basis
- [ ] Exception paths must return `FAILED`, never a literal score like `0.1`
- [ ] Make `NO_SIGNIFICANT_ANOMALIES_DETECTED` reachable
- [ ] Honour the config switches currently ignored by `detector_factory.py`
- [ ] Guard the top-level `import torch` so a torch failure doesn't kill all detection
- [ ] Resolve open question: 468 vs 478 landmarks in the shipped `.task`
- [ ] Tests per detector for each execution status

---

## Phase 7 — P3: Evidence fusion

- [x] Findings reference specific observations by id (Phase 4)
- [x] Contradicting observations recorded, not discarded — linked as `CONTRADICTS` (Phase 4); not yet weighed by the decision rule
- [ ] Uncertainty retained — `INCONCLUSIVE` never reported as negative
- [ ] `NO_SIGNIFICANT_ANOMALIES_DETECTED` documented as *not* a guarantee of authenticity
- [ ] No detector-independence assumption
- [ ] Remove the unconditional TruFor-limitation boilerplate
- [ ] Tests: contradiction surfaced; inconclusive stays inconclusive

---

## Phase 8 — P3: Background jobs

- [ ] Real async execution — analysis no longer blocks the request
- [ ] Observable `QUEUED` → `PROCESSING` → terminal
- [ ] Genuine progress (remove the literal `50`)
- [ ] `PARTIAL` / `INCONCLUSIVE` no longer flattened to `COMPLETED`
- [ ] Timeout honoured; stuck-run reaper
- [ ] Tests: status transitions observable; timeout produces `FAILED`

---

## Phase 9 — P4: AI Investigator (Nebius + NVIDIA Nemotron)

Per the audit: **no credential is configured.** Build to contract, test with mocks, document config, claim nothing live.

- [x] Backend-only Nebius client (OpenAI-compatible); key never reaches the frontend
- [x] NVIDIA Nemotron as the model (hackathon requirement)
- [x] **Bounded** tool registry — no shell, no arbitrary code, no arbitrary file access, no arbitrary network
- [x] LLM output labelled as interpretation, **never** independent forensic proof
- [x] Graceful `UNAVAILABLE` when no credential
- [x] `openai` + `mediapipe` added to `requirements.txt`
- [x] Mocked tests; documented `.env.example` keys
- [x] Record honestly whether live inference was ever executed — **not executed.** No `NEBIUS_API_KEY` has been available. The recorded-response test (`tests/fixtures/nemotron_investigation_recorded.json`) is hand-written in Token Factory's response format, not captured from a call. `test_live_token_factory_investigation` (marker `live`) runs the real request once the key is exported, and skips until then.

---

## Phase 10 — P4: Reports & evidence graph

- [ ] **Delete the fabricated chain of custody**; render from real `AuditEvent` rows
- [ ] Never fabricate signatures, credentials, certifications, or examiner approval
- [ ] HTML escaped (close the injection vector)
- [ ] PDF written to a per-request path, cleaned up
- [ ] No claim of legal admissibility
- [ ] Evidence graph FK-backed with the required typed nodes/edges
- [ ] Tests: report contains only real audit events; escaping verified

---

## Phase 11 — P4: New frontend  🟡 MOSTLY COMPLETE (2026-10-22)

New app in `web/` (React 19 + TS + Vite, no UI library), built with the `design-taste-frontend` skill, app style *achromatic-ledger* (seed 1786698071). `frontend/` and `frontend-new/` are untouched and no longer used.

- [x] Case → evidence → analysis → findings flow against the real API (login, cases, case + upload, analysis, evidence, audit trail)
- [x] Availability and execution status shown as **distinct** columns; did-not-run is hatched in the coverage ledger, never "no anomaly"
- [x] Observations traceable to findings: supports / contradicts lists; pointing at one highlights its detector's regions on the evidence
- [x] Uncertainty displayed: rule id, rationale, coverage gaps with reasons, score semantics on every score, non-result scores print "—"
- [x] Only anomaly regions take the data hue; regions from detectors that found nothing are context, shown dashed on hover only (found and fixed during review: face boxes were initially drawn as anomalies)
- [x] Disabled with a stated reason: report export (fabricated chain of custody, Phase 10); chain verification for non-admins
- [x] No credentials in frontend code
- [x] Backend additions it needed: `GET /analysis/?media_id=`, `evidence_count` on `/cases/`, `case_id`/`uploaded_by` on media; covered in `test_forensic_record.py` [EXECUTED]
- [x] Gate: tasteskill `lint` 0 FAIL; `inspect --app` 0 FAIL on all 6 screens at 1440/768/390 against a scratch backend with real CC0 evidence run through the real pipeline [EXECUTED]
- [ ] Interactions verified by script but not by clicking in a browser: upload, run analysis, open case, admin chain verification
- [x] `frontend/` and `frontend-new/` moved to `_to_delete/` (2026-10-22) for the owner to delete

Finding from the demo data, for Phase 6: on a deliberately copy-moved image (a Ferris wheel cloned into empty sky) `CopyMoveDetector` returned INCONCLUSIVE ("insufficient reliable matches"), and the doctored and untouched copies received **identical** results from every detector — both SUSPICIOUS/WEAK on error-level analysis alone.

## Phase 11b — Self-service SaaS (2026-10-22)  ✅ STEP 1 COMPLETE

Product direction changed from an investigator case tool to a self-service app: sign up, drop a file, read a plain result.

- [x] **Ownership.** Media is owned by `uploaded_by`, folders by `created_by`, analyses by their media. Every route scopes to the caller; foreign ids return 404 (`repository.get_owned_*`). Unscoped `list_media` / `list_analyses` deleted so nothing can call them by accident.
- [x] **Cross-account leak closed in similarity search**: it compared each upload against every account's fingerprints, exposing other accounts' media ids via matches and the provenance graph. Now owner-scoped; unowned media gets no local search.
- [x] Public `POST /auth/signup` (email + 12-char password; always role `user`; 5/hour per address; audited). `/auth/register` removed. Roles are `admin` | `user` (migration `c9d2e4f1a7b3` renames `investigator`).
- [x] `DELETE /media/{id}`: media, analyses, derived rows, stored bytes and frames; the audit trail keeps `EVIDENCE_DELETED`.
- [x] Daily per-account upload quota (`DAILY_UPLOAD_LIMIT`, default 50, counted from the DB) + `GET /media/usage`.
- [x] Upload response no longer discloses the absolute storage path; a failed DB write removes the orphaned bytes and returns 500 instead of a success with `db_error`.
- [x] **Removed old surfaces:** `/detection/*`, `/fingerprint`, `/similarity`, `/source`, `/provenance`, `/evidence`, `/report/*`, `/jobs/*` and their tests. The services remain (the pipeline still writes provenance and report rows); report export returns with Phase 10.
- [x] New dark UI in `web/` (tasteskill *true-black-flat*, seed 753521769): sign in / create account, Check media (drop zone, upload+analyse in one action), Result (plain verdict, marked areas, "What we checked", collapsible Technical details), History (filters, search, delete), Folders.
- [x] Tests: `test_ownership.py` (17) — every id route 404s across accounts, lists hold only the caller's rows, byte-identical upload by another account never matches, delete, quota, usage. Backend **324 passed, 0 failed** [EXECUTED]. Gate: lint 0 FAIL, inspect 0 FAIL on 6 screens × 3 widths [EXECUTED].
- [ ] Step 2: detection quality (Phase 6) — **highest risk**; background jobs (Phase 8)
- [ ] Step 3: password reset / email verification, report export (Phase 10), billing, Postgres + object storage
- [x] `backend/scripts/verify_sec1_fix.py` retired to `_to_delete/` (2026-10-22); path confinement stays covered by `test_security_path_confinement.py`.

## Phase 12 — P5: Video / audio

Only where dependencies and models can actually be validated.

- [ ] Fix `media_detection_router` rejecting audio
- [ ] **Rename or remove `NeuralSpeechDetector`** — it performs no neural inference
- [ ] Remove `BaselineAudioAnalyzer`'s hardcoded verdict
- [ ] Fix `BaselineVideoDetector` discarding all but the first detection
- [ ] Or: mark both modalities `UNAVAILABLE` honestly and ship image end-to-end

---

## Phase 13 — Release

- [ ] Remaining docs: `CLAUDE.md`, `ARCHITECTURE.md`, `API.md`, `DETECTORS.md`, `SECURITY.md`, `TESTING.md`, `DEMO_SCRIPT.md`
- [ ] `.env.example` — placeholders only, never a real value
- [ ] Open-source `LICENSE`
- [ ] `backend/.env` untracked (it is currently **tracked by git**)
- [ ] Repo contains no secrets, personal evidence, private datasets, or non-redistributable model files
- [ ] `requirements.txt` header corrected (claims 3.11.9; venv is 3.14.2)
- [ ] Full suite green; coverage recorded
- [ ] Demo runs on Nebius
- [ ] Every remaining limitation stated in the README

---

## Standing rules

1. Never claim a command succeeded unless its output confirms it. Beware piped exit codes — use `PIPESTATUS`.
2. Never replace real inference with mock data in a production path.
3. Never represent an unavailable detector as having run.
4. Never represent an inconclusive result as a negative result.
5. Preserve originals; analyse copies.
6. No destructive git/filesystem/database operation without explicit authorisation.
7. Credentials stay on the backend and out of logs.
8. A hash proves identity relative to the recorded digest — not that the content was truthful before acquisition.
