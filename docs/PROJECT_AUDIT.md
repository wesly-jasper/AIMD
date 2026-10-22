# AIMD Sentinel — Project Audit

**Audit date:** 2026-10-20
**Audited tree:** `D:\projects\AIMD-2` (working tree, *not* HEAD — 25 files were modified and uncommitted at audit time)
**Auditor method:** source reading + executed verification. Every claim below is labelled with how it was established.

---

## 0. Evidence standard used in this document

| Tag | Meaning |
|---|---|
| **[EXECUTED]** | I ran it and observed the result. Highest confidence. |
| **[READ]** | Established by reading the code at a cited `file:line`. |
| **[UNVERIFIED]** | Plausible but not confirmed. Treated as an open question, never as fact. |

Detector capability states use the vocabulary required by the project spec:
`IMPLEMENTED_AND_VERIFIED` / `IMPLEMENTED_BUT_UNVERIFIED` / `PARTIALLY_IMPLEMENTED` / `UNAVAILABLE` / `PLANNED`.

---

## 1. Environment baseline [EXECUTED]

| Item | Value |
|---|---|
| Python | 3.14.2 (`venv/`) — note `requirements.txt:3` claims 3.11.9 |
| torch | 2.9.1+cu126, **CUDA available: True** |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| FastAPI / SQLAlchemy / Alembic | 0.135.3 / 2.0.49 / 1.18.4 |
| OpenCV / Pillow / NumPy | 5.0.0 / 12.1.0 / 2.4.1 |
| ultralytics / mediapipe / librosa | 8.4.60 / 0.10.35 / 1.0.0 |
| Database | SQLite, live at `backend/aimd.db` (~2.4 MB) |

**Installed during audit** (were declared-but-absent or undeclared): `reportlab 5.0.1`, `openai 3.35.1`, `exifread 3.5.1`, `pytest-timeout`.
`openai` is required for Nebius (OpenAI-compatible API) and was **not** in `requirements.txt`.
`mediapipe` is **installed but not declared** in `requirements.txt` — a clean `pip install -r` would silently disable face-region analysis.

### Baseline test suite [EXECUTED]

```
132 passed, 5 failed, 1 skipped  (97.50s)
```

All 5 failures are **pre-existing**, present before any change by me, and share one root cause:

- `tests/integration/test_detection_api.py::test_detection_api_missing_file` — `assert 404 == 400`
- `…::test_detection_api_missing_media_type` — `assert 404 == 400`
- `…::test_detection_api_invalid_request` — `assert 404 == 422`
- `tests/integration/test_analysis_pipeline.py::test_full_pipeline_image` — `assert 404 == 200`
- `tests/integration/test_live_api_e2e.py::test_e2e_image_investigation`

**Root cause [READ]:** the detection router is commented out of the app.
`app/main.py:14` `# from app.api.v1.endpoints.detection import router as detection_router`
`app/main.py:63` `# app.include_router(detection_router, prefix="/api/v1")`
So every `/api/v1/detection/*` path 404s even though `endpoints/detection.py` defines handlers.

---

## 2. CRITICAL SECURITY FINDINGS

### SEC-1 — Unauthenticated arbitrary file read [EXECUTED — exploit confirmed]

**Severity: Critical.** Confirmed by running the exploit against the real app, not inferred.

Two-step chain:
1. `POST /api/v1/source/trace` accepts a client-controlled `file_path` and writes it verbatim into the `media` table — `app/api/v1/endpoints/investigation.py:125` `"file_path": str(request.file_path)`. No containment check, no existence requirement, no authentication.
2. `GET /api/v1/media/{id}/download` serves that path via `FileResponse` — `app/api/v1/endpoints/media.py:95-103`.

Proof of exploitation (executed): wrote a canary file outside `uploads/`, then

```
POST /api/v1/source/trace {"media_id":"pentest-1","analysis_id":"pentest-a1","file_path":"<canary>"}  -> 200
GET  /api/v1/media/pentest-1/download                                                                 -> 200
body: TOP-SECRET-CANARY-VALUE
```

Any file readable by the server process is exfiltratable, **including `backend/aimd.db` and `backend/.env`**. A sanitising helper `ensure_safe_path` exists at `app/core/security.py:31` but is **imported by nothing** [READ].
*Test artifacts created during this proof were deleted from `aimd.db` and temp.*

### SEC-2 — Fabricated cryptographic hash written into the evidence table [EXECUTED]

`investigation.py:129` hardcodes
`"sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"`.
I computed `sha256(b"")` and it is **exactly that value** — the digest of the empty byte string.

A fabricated hash is persisted into the forensic record and later re-served as `"evidence_category": "FACT"` (`services/evidence/evidence_service.py:76-77`). This is the single most serious *integrity* defect in the system: it violates the project's own stated rule at `evidence_service.py:4` ("must NEVER use hardcoded or mock values") and would invalidate any custody claim.

### SEC-3 — Fail-open authentication [READ]

`app/core/auth.py:13` sets `auto_error=False`; `app/api/v1/endpoints/auth.py:73-76` returns a synthetic user when no token is supplied:

```python
if token is None:
    return User(username="anonymous_analyst", role="investigator", is_active=True)
```

A tokenless `GET /api/v1/auth/me` returns 200 with an `investigator` role.

### SEC-4 — Authentication not applied to any functional route [READ]

`get_current_user` is referenced by exactly one endpoint (`auth.py:90`). All other routes — upload, download, analysis, evidence, reports, jobs — are anonymous. No router-level or global dependency.

### SEC-5 — Privilege escalation via open self-registration [READ]

`endpoints/auth.py:36` takes `role` from the request body and `:73` writes it through. Anyone may register as `admin`. No password policy, no invite, no rate limit.

### SEC-6 — Hardcoded signing key, unrotatable [READ]

`app/core/auth.py:9` `SECRET_KEY = "aimd_forensic_platform_secret_key_change_in_production"`, duplicated at `app/core/config.py:93`. Token signing uses the **module constant**, not `settings.secret_key`, so setting the env var does not change signing behaviour. Anyone with the repo can forge a token.

### SEC-7 — Other confirmed issues [READ]

| ID | Issue | Location |
|---|---|---|
| SEC-7a | Arbitrary-path hashing / existence oracle | `investigation.py:40-44` (`/fingerprint/generate`), `:83` (`/similarity/search`) |
| SEC-7b | Whole upload buffered in RAM *before* the size check | `media_ingestion.py:128-134` (`await file.read()` then `len(data) >`) |
| SEC-7c | Rate limiting configured but inert — `SlowAPIMiddleware` never added, no `@limiter.limit` anywhere | `main.py:24,48-49` |
| SEC-7d | Raw exception text returned to clients (leaks absolute paths) | `media.py:32`, `analysis.py:55`, `investigation.py:234,282,335` |
| SEC-7e | Unescaped HTML report → injection via attacker-controlled filename | `investigation.py:296-309` |
| SEC-7f | Forensic PDF written to shared temp with predictable name, never deleted | `investigation.py:327` |
| SEC-7g | Unbounded pagination (`?limit=1000000` accepted) | `media.py:108`, `analysis.py:129`, `jobs.py:57` |
| SEC-7h | Registered error handlers never attached, so the documented error envelope is never produced | `core/exceptions.py` unused by `main.py` |

**No SQL injection found** [READ] — all DB access uses SQLAlchemy bound constructs.

---

## 3. Detector capability matrix

Checkpoint inventory on disk [EXECUTED] — the *only* two model files in the repo:

```
backend/models/face_landmarker.task   3,758,596 B
backend/yolov8n.pt                    6,549,796 B
```

Live inference smoke test [EXECUTED]: `YOLO('yolov8n.pt').predict(...)` → ran, returned boxes. MediaPipe `FaceLandmarker.create_from_options(...)` → loaded successfully.

| Detector | Wired? | State | Basis |
|---|---|---|---|
| **TruForDetector** | **No** — never constructed | **UNAVAILABLE** | `backend/.env`: `TRUFOR_ENABLED=false`, `TRUFOR_MODEL_PATH=` empty; `detector_factory.py:25-26` returns `None` [EXECUTED+READ] |
| ForensicImageDetector | Yes | PARTIALLY_IMPLEMENTED | Real ELA/FFT/noise DSP, but hardcoded thresholds; returns literal `0.1` on every exception path [READ] |
| CopyMoveDetector | Yes | PARTIALLY_IMPLEMENTED | Real SIFT + RANSAC; confidence floor hardcoded `0.3 + 0.6*ratio` [READ] |
| FaceRegionAnalyzer | Yes | PARTIALLY_IMPLEMENTED | Real MediaPipe landmarks; heuristic scoring incl. a literal `0.3` appended per Hough circle [READ] |
| ManipulationMaskGenerator | Yes | PARTIALLY_IMPLEMENTED | Real multi-scale DSP; UNet path is a confirmed `# TODO` stub; **silent fusion bug** (below) [READ] |
| BaselineObjectDetector | Yes | UNAVAILABLE *by design* | Returns a constant `UNAVAILABLE`; honest, weight 0.00 [READ] |
| YOLOv8ObjectDetector | **No** | IMPLEMENTED_AND_VERIFIED (unwired) | Real inference confirmed [EXECUTED]; weights present; never added to the container |
| BaselineFaceDetector | Yes | PARTIALLY_IMPLEMENTED | Real Haar + Laplacian/FFT, but score can only take 4 discrete values [READ] |
| DeepLearningFaceDetector (MTCNN) | **No** | IMPLEMENTED_BUT_UNVERIFIED (unwired) | MTCNN weights ship inside `facenet_pytorch` [READ] |
| BaselineAudioAnalyzer | **No** — unreachable | PARTIALLY_IMPLEMENTED | Real librosa features, but **verdict hardcoded** `manipulation_detected=False`, `INCONCLUSIVE` regardless of score [READ] |
| NeuralSpeechDetector | **No** — unreachable | **Misleadingly named** | **Contains no neural inference at all** — two `if` statements on mel-spectrogram stats; imports `torch` and never uses it, yet reports `status="active_neural_speech_detector"`, version `"2.0-neural-acoustic"` [READ] |
| BaselineVideoDetector | **No** | PARTIALLY_IMPLEMENTED | Pass-through that keeps only `detections[0]`, discarding the rest [READ] |

### Headline consequence

**There is no working learned forgery/AI-generation detector in this system.** The only neural forensic detector (TruFor) is disabled and has no checkpoint. Everything that actually runs is classical signal processing with hand-tuned magic numbers and no stated calibration. Per the project spec, these must be reported as **heuristic observations, not probabilities**, and the system must not present them as proof.

### Confirmed detector-layer bugs [READ]

1. **Silent fusion truncation** — `manipulation_mask_generator.py:149-170` builds **7** heatmaps but supplies **6** weights; `zip()` truncates, so the colour cue is computed expensively and then **discarded**, and the gradient cue receives the weight intended for colour.
2. **TruFor weight-key mismatch** — the weight table keys `"TruForDetector"` (`image_detection_engine.py:30`) while the detector emits `detector="TruFor"` (`trufor_detector.py:78`). If ever enabled it would silently get the 0.30 default instead of 0.70.
3. **Structural positive bias** — `face_region_analyzer.py:279` keeps only regions scoring `>= 0.3`, then averages over *only the survivors* (`:172-176`), so the face score can never fall below 0.3 once any region is emitted.
4. **Min-max renormalisation destroys calibration** — `manipulation_mask_generator.py:344` normalises each heatmap so its peak is always exactly 1.0, meaning a clean image and a forged image both peak at 1.0.
5. **`CLEAN` is unreachable** — the wired set always contains the permanently-`UNAVAILABLE` `BaselineObjectDetector`, so `has_unavailable` is always true and `image_detection_engine.py:136-143` can never return `CLEAN`.
6. **Identity swap on fallback** — YOLO/MTCNN/NeuralSpeech failures return results relabelled as the baseline detector, so downstream cannot tell "ran" from "failed and silently substituted".
7. **Audio is dead code** — `media_detection_router.py:44-46` raises `ValueError` for anything but image/video, so any audio upload fails inside the pipeline.
8. **Config switches ignored** — `copymove_enabled`, `manipulation_mask_enabled`, `face_region_analyzer_enabled` etc. (`config.py:74-79`) are never read; `detector_factory.py:66-75` hardcodes construction. `ai_image_model_path`, `deepfake_model_path`, `audio_model_path` are declared and never referenced.

---

## 4. Forensic data model gap analysis [READ]

Existing tables (`app/db/models.py`): `media`, `media_metadata`, `analyses`, `frames`, `detections`, `detection_regions`, `fingerprints`, `similarity_matches`, `sources`, `provenance_nodes`, `provenance_edges`, `evidence_items`, `reports`.

| Required entity | Status | Notes |
|---|---|---|
| **Case** | **MISSING** | No table. Synthesised per-request and never stored: `evidence_service.py:61` `f"AIMD-{analysis_id[:8].upper()}"` |
| Evidence | **PARTIAL** | `media` is closest; no case linkage, no custodian, no acquisition actor, no immutability marker |
| AnalysisRun | EXISTS | `analyses`; lacks software version, detector-set version, configuration snapshot, duration |
| DetectorResult | EXISTS | `detections`; lacks `availability_status` separate from `execution_status`, and `score_semantics` |
| Observation | **PARTIAL / conflated** | Split across `detection_regions` and `evidence_items` rows whose free-text `category == "OBSERVATION"` |
| Finding | **MISSING** | `evidence_items` serves as Observation + Finding + Uncertainty simultaneously |
| Assessment | **MISSING** as an entity | Collapsed to two columns on `analyses`: `assessment`, `overall_confidence`. No rationale, no supporting/contradicting links |
| **AuditEvent** | **MISSING entirely** | No audit table. The PDF's "Chain of Custody" is fabricated at render time from the analysis id and the wall clock (`pdf_report.py:308-314`), and `:106` hardcodes `["Analyst:", "AIMD Automated Pipeline"]` |
| Report | EXISTS | `reports` |
| User/principal | **MISSING** | Accounts live in a module-level dict `DEMO_USERS_DB` (`core/auth.py:44`), lost on restart |

**Migrations are a no-op** [READ]: the only migration, `alembic/versions/04e00c1a868b_initial_schema.py:21-25`, is `def upgrade(): pass`. The schema is created at runtime by `Base.metadata.create_all()` (`db/database.py:68`). `alembic upgrade head` against a fresh database therefore produces **nothing**, and there is no migration path for future changes. `alembic.ini:89` still holds the placeholder URL.

No column uses a DB-level enum — every status/verdict is a bare `String` (`models.py:13` imports `Enum as SAEnum` and never uses it), so there is no value integrity at the storage layer.

---

## 5. Assessment & fusion: spec violations [READ/EXECUTED]

Observed live during the baseline test run [EXECUTED]:

```
Pipeline complete -- status=PARTIAL assessment=SUSPICIOUS confidence=0.636
```

This single line embodies three prohibitions from the project spec:

1. **A blended numeric confidence.** `image_detection_engine.py:120-130` takes a weighted mean across detectors measuring unrelated phenomena (ELA vs. face geometry vs. SIFT self-matching), with weights that are unexplained literals. The spec forbids "a numeric overall confidence by averaging unrelated detector scores" and forbids assuming detector independence.
2. **Uncalibrated scores presented as confidence.** None of these detectors has a validation set or calibration curve in-repo, so the outputs are *model scores*, not probabilities.
3. **`dominant_signal` ignores weights** — it is the argmax of raw confidence (`:124-126`), so a weight-0.20 detector can be reported as dominant.

There is no contradiction detection, no observation-level provenance for findings, and no record of unresolved questions.

---

## 6. Mock / fabricated data in operational routes [READ]

| Route | Fabrication | Location |
|---|---|---|
| `GET /health`, `/api/v1/health` | Six of seven model statuses are hardcoded strings never consulted from the registry (`"image_detector": "available"` …) | `main.py:94-102`, duplicated `health.py:36-44` |
| `GET /api/v1/jobs/{id}` | `progress` is a literal — `100 if terminal else 50`. An in-flight job always claims exactly 50% | `jobs.py:52` |
| `GET /api/v1/media/{id}` | `"status": "STORED"` — not a column, never derived | `media.py:76` |
| `POST /api/v1/source/trace` | Fabricated SHA-256 (SEC-2), fabricated analysis row, content type guessed from suffix | `investigation.py:121-136` |
| Evidence package | Boilerplate "No validated AI detection model configured" limitation appended **unconditionally**, even when TruFor is enabled | `evidence_service.py:266-270` |
| Chain of custody (PDF) | Fabricated from analysis id + current clock; analyst name hardcoded | `pdf_report.py:106,308-314` |

`SourceTracer` is permanently inert [READ] — `source_tracer.py:67` defaults to `DisabledSourceSearchProvider()` and is always constructed with no argument, so `/source/trace` and pipeline stage 6 can never return a result. `google_vision_api_key` / `tineye_api_key` are never read.

### Dead code presenting as implemented surface [READ]

`app/core/constants.py`, `app/core/security.py`, `app/core/exceptions.py` handlers, `app/schemas/media.py` (empty), `schemas/{evidence,provenance,report,similarity}.py`, `repository.get_media_by_sha256` (so **no duplicate-upload detection**), `endpoints/health.py` (never mounted), `endpoints/detection.py` (commented out), a duplicate `GET /evidence/{id}` at `investigation.py:338`.

---

## 7. Architecture & operations [READ]

- **No background job mechanism at all.** Repo-wide search for `BackgroundTasks|celery|asyncio.create_task|ThreadPool|add_task` across `app/` returns zero matches. `POST /api/v1/analysis/` runs the full pipeline inline (`analysis.py:51`), pinning a thread-pool worker for minutes. `processing_timeout_seconds` is never read. A client disconnect leaves the row stuck at `PROCESSING` with no reaper.
- **Polling exists but is near-useless** — the only way to learn the id to poll is the response of the blocking call itself.
- **`QUEUED` is unobservable** — records are created already `PROCESSING` (`analysis_pipeline.py:88`).
- **Status mapping hides failure** — `jobs.py:37-38` reports `PARTIAL` and `INCONCLUSIVE` both as `"COMPLETED"`.
- **Original evidence is not preserved immutably** — no `chmod`, no read-only flag, no quarantine/working-copy split; downstream stages re-read the same path, and the stored hash is never re-verified before analysis.
- **Two independent SHA-256 computations are never compared** — `media_ingestion.py:159` (over in-memory bytes) and `hash_generator.py:49-53` (streamed from disk).
- **Magic-byte sniffing is not a gate** — `media_ingestion.py:76-84` falls back to the client-supplied `Content-Type`, and the two are never cross-checked. No extension allow-list (`constants.py:61-65` never imported).
- **Unguarded top-level `import torch`** in `trufor_runner.py:7`, pulled in at container-import time by `detector_factory.py:7-9`, so a broken torch breaks all detection rather than just TruFor.
- **Full-table-scan similarity search** on an anonymous, unthrottled route (`similarity_engine.py:135,152,170`).

---

## 8. What is genuinely good and must be preserved

Being fair to the existing code — these are real strengths to build on, not rewrite:

1. **`TruForRunner` is the most honest module in the tree.** It raises rather than guesses: `RuntimeError('TruFor model is not loaded')`, `ValueError("…does not contain a supported anomaly map or score")`. This is exactly the required pattern.
2. **`TruForDetector` and `FaceRegionAnalyzer` already separate availability from execution**, deciding availability at construction via `initialization_error` and distinguishing `UNAVAILABLE` / `INCONCLUSIVE` / success at call time.
3. **`FaceRegionAnalyzer` distinguishes "ran fine, nothing to analyse"** (`status="no_faces_detected"`, `INCONCLUSIVE`) from failure — a genuinely useful forensic distinction.
4. **`BaselineObjectDetector` is honest by design**, returning `UNAVAILABLE` with an explanatory note rather than a fake score.
5. **`YOLOv8ObjectDetector` refuses to overclaim** — `manipulation_detected=False,  # Object presence != manipulation`, assessment `INCONCLUSIVE`. It reports regions only. This correctly anticipates the spec's prohibition on using object detection as proof of forgery.
6. **`AnalysisPipeline.run(media_id, db)` takes no request objects**, so it can move behind a queue with no pipeline change.
7. **Upload storage names are UUID-derived**, not user-controlled — the strongest existing control in the upload path.
8. **The test suite is substantial** (138 tests) and already includes `test_detector_availability_semantics.py`.
9. **`.gitignore` correctly excludes** `.env`, `*.db`, `uploads/`, `backend/models/`.

---

## 9. Capability summary

| Capability | State |
|---|---|
| File upload + SHA-256 + persistence | IMPLEMENTED_AND_VERIFIED (but original not immutable) |
| Metadata / structural extraction | IMPLEMENTED_BUT_UNVERIFIED |
| ELA / JPEG / FFT heuristics | PARTIALLY_IMPLEMENTED (uncalibrated, mislabelled as confidence) |
| Copy-move detection | PARTIALLY_IMPLEMENTED |
| Face region analysis | PARTIALLY_IMPLEMENTED (real landmarks, heuristic scores, positive bias) |
| Manipulation localisation | PARTIALLY_IMPLEMENTED (silent fusion bug; UNet stub) |
| AI-generation detection (learned) | **UNAVAILABLE** (no checkpoint, TruFor disabled) |
| Object detection | IMPLEMENTED_AND_VERIFIED but **unwired** |
| Video analysis | PARTIALLY_IMPLEMENTED |
| Audio analysis | **UNAVAILABLE** (unreachable via router; verdict hardcoded) |
| Provenance / C2PA credentials | PLANNED |
| Case management | **MISSING** |
| Observations / Findings / Assessment entities | **MISSING** |
| Audit trail | **MISSING** |
| Evidence graph (persisted) | PARTIAL (`provenance_nodes/edges` exist, string-keyed, not FK-backed) |
| Report generation | IMPLEMENTED_BUT_UNVERIFIED (fabricated custody section) |
| **Nebius / NVIDIA Nemotron AI Investigator** | **PLANNED — no integration exists anywhere in the repo** (verified: no non-vendored match for `nebius`/`nemotron`) |
| Background jobs | **MISSING** |
| AuthN/AuthZ | PARTIALLY_IMPLEMENTED and **fail-open** |

---

## 10. Credential status

`backend/.env` contains only `TRUFOR_ENABLED=false` and `TRUFOR_MODEL_PATH=` (empty). Root `.env` contains only `VITE_API_BASE_URL`.

**No Nebius API key is configured.** Per the project spec, the Nemotron integration will therefore be built against the documented API contract and tested with **mocked** responses, with the exact required configuration documented. No live inference will be claimed until a credential is supplied.

---

## 11. Prioritised remediation order

Rationale: a confirmed unauthenticated arbitrary file read and a fabricated hash in the evidence table outrank feature work in a forensics product.

1. **P0 — SEC-1** arbitrary file read; **SEC-2** fabricated SHA-256. Stop-and-fix.
2. **P0** Re-mount or remove the detection router (fixes all 5 baseline failures honestly).
3. **P1 — SEC-3/4/5/6** auth fail-open, unprotected routes, role escalation, hardcoded key.
4. **P1** Core contracts: `DetectorAvailability` / `DetectorExecutionStatus` enums; stop emitting blended `confidence`.
5. **P2** Case / Observation / Finding / Assessment / AuditEvent tables + real migrations.
6. **P2** Immutable original preservation + hash re-verification.
7. **P2** Fix confirmed detector bugs (fusion truncation, weight key, positive bias, identity swap).
8. **P3** Background jobs + genuine progress.
9. **P3** Evidence fusion with observation-linked findings and contradiction detection.
10. **P4** Nebius/Nemotron AI Investigator (mocked tests; documented config).
11. **P4** New frontend against the real contract.
12. **P5** Video/audio only where dependencies and models can be validated.

---

## 12. Open questions [UNVERIFIED]

1. Does the shipped `face_landmarker.task` yield 468 or 478 landmarks? `FACE_REGIONS` indexes up to 477; if 468, `FaceRegionAnalyzer` silently degrades to `UNAVAILABLE` for every image.
2. Would a real TruFor release load? `trufor_runner.py:78-81` requires a pickled `nn.Module` or ONNX export — a plain `state_dict` is rejected and **no TruFor architecture is defined in this repo**.
3. False-positive rates for the hardcoded thresholds (0.50 / 0.55 / 0.45 / 0.35) are unknown; no labelled validation set exists in-repo.
4. Wall-clock cost of `FaceRegionAnalyzer._compute_lbp` — a pure-Python per-pixel triple loop over 11 regions per face.
