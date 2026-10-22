# AIMD Sentinel — Project Audit (Phase 0, re-run)

**Audit date:** 2026-10-22
**Audited tree:** `D:\projects\AIMD-2` at commit `6e6ddea` (clean working tree apart from untracked `web/.vite/`)
**Previous audit:** 2026-10-20, against an older, uncommitted tree. This document replaces it; every finding from that audit is carried forward below with its current status.
**Method:** source reading plus executed verification (FastAPI `TestClient` against throwaway SQLite databases and temp storage; `backend/aimd.db` and `backend/uploads/` were not touched).

---

## 0. Evidence standard

| Tag | Meaning |
|---|---|
| **[EXECUTED]** | Ran it and observed the result. |
| **[READ]** | Established by reading the code at the cited `file:line`. |
| **[UNVERIFIED]** | Plausible but not confirmed. |

Status vocabulary for previous findings: **FIXED** / **PARTIALLY FIXED** / **STILL OPEN** / **N/A** (the code was removed).
Detector capability states: `IMPLEMENTED_AND_VERIFIED` / `IMPLEMENTED_BUT_UNVERIFIED` / `PARTIALLY_IMPLEMENTED` / `UNAVAILABLE` / `PLANNED`.

All paths below are relative to `backend/` unless they start with `web/` or `docs/`.

---

## 1. Headline

**Fixed since 2026-10-20:** every critical and high security finding, the 5 failing tests, the no-op migrations, fail-open auth, the fabricated SHA-256, the missing Case / Observation / Finding / Assessment / AuditEvent / User tables, the blended weighted-mean confidence in the image engine, and the frontend's mock data.

**Still open:** there is **no learned forensic detector** (no TruFor weights, no AI-generation model), **no Nebius/Nemotron integration**, **no external source search**, and analysis still runs inline in the HTTP request. In addition, executed tests show the image pipeline **flags every unmanipulated image tested as SUSPICIOUS** (§4.3). That has to be fixed before any demo result means anything.

---

## 2. Baseline [EXECUTED]

| Item | 2026-10-20 | 2026-10-22 |
|---|---|---|
| Backend tests | 132 passed, 5 failed, 1 skipped | **311 passed, 0 failed, 1 skipped** (93.8 s) |
| Frontend build (`web/`, `tsc -b && vite build`) | not run | **passes** (263 kB JS / 24 kB CSS) |
| Python in `venv/` | 3.14.2 | 3.14.2 (still differs from the declared 3.11.9 and Dockerfile `python:3.11-slim`) |
| GPU | RTX 3050 Laptop, CUDA available | unchanged |
| Database | SQLite, `create_all()` | SQLite, built by Alembic `upgrade head` |

Test-run warning worth fixing: `SAWarning: DELETE statement on table 'observations' expected to delete 5 row(s); 0 were matched` (`app/db/repository.py:137`, from `test_ownership.py`). The ORM and the DB-level cascade both try to delete the same rows.

---

## 3. Security

### 3.1 Previous findings

| ID | Finding (2026-10-20) | Status | Evidence |
|---|---|---|---|
| SEC-1 | Unauthenticated arbitrary file read via persisted `file_path` | **FIXED** | Download resolves through `resolve_evidence_path` (`app/api/v1/endpoints/media.py:208`, `app/core/security.py:71-111`); stored path is UUID-derived (`app/services/ingestion/media_ingestion.py:158-164`) [READ]. Traversal filename + injected `file_path` stored inside the upload dir; a tampered out-of-root DB path returns 400 [EXECUTED] |
| SEC-2 | Fabricated SHA-256 (`sha256(b"")`) written as FACT | **FIXED** | No 64-hex literal in `app/`; hash computed at `media_ingestion.py:111,167` [READ] |
| SEC-3 | Fail-open auth (anonymous investigator) | **FIXED** | `app/core/auth.py:82` `auto_error=True`, no anonymous branch (`:358-389`) [READ]; tokenless `/auth/me` → 401 [EXECUTED] |
| SEC-4 | Auth missing on functional routes | **FIXED** | Router-level dependency (`app/main.py:129-135`); all 14 protected routes → 401 without a token; public: login, signup, `/health`, `/api/v1/health` [EXECUTED] |
| SEC-5 | Self-registration as admin | **FIXED** | Role forced to `user` (`app/api/v1/endpoints/auth.py:106-108`) [READ]; signup with `role=admin` gets `user`, `/audit/verify` → 403 [EXECUTED] |
| SEC-6 | Hardcoded, unrotatable JWT key | **FIXED** | Key from env only; old key rejected; ≥32 chars; refuses to start outside dev without one (`app/core/auth.py:89-145`) [READ]. Old-key and `alg=none` tokens → 401 [EXECUTED] |
| SEC-7a | Arbitrary-path hashing / existence oracle | **N/A** | Routes removed; no route accepts a filesystem path [EXECUTED] |
| SEC-7b | Whole upload buffered before size check | **STILL OPEN** | `await file.read()` then size check (`media_ingestion.py:136-142`); no request-body cap [READ+EXECUTED] |
| SEC-7c | Rate limiting inert | **FIXED** | Middleware + `@limiter.limit` on login, signup, upload, analysis (`app/main.py:59-60,84`); 429s observed [EXECUTED]. Limits are per-IP, in-memory. |
| SEC-7d | Raw exception text returned to clients | **PARTIALLY FIXED** | Generic 500 handler returns a fixed message (`app/core/exceptions.py:189-201`) [EXECUTED]. Still leaks via `error_message = str(exc)` (`app/services/orchestration/analysis_pipeline.py:251`, returned at `app/api/v1/endpoints/analysis.py:140`) and per-stage `f"{stage}: {exc}"` messages (`analysis.py:99`) [READ] |
| SEC-7e | Unescaped HTML report | **N/A** | No HTML generation remains; reports are Markdown/JSON [READ] |
| SEC-7f | Predictable temp PDF never deleted | **N/A** | No PDF code remains [READ] |
| SEC-7g | Unbounded pagination | **FIXED** | Capped at 200 (`media.py:140`, `analysis.py:220`, `cases.py:62`) and 500 (`audit.py:25`); `limit=100000` → 422 [EXECUTED] |
| SEC-7h | Error handlers never attached | **FIXED** | `app/main.py:72-75`; envelopes observed on 404/500 [EXECUTED] — see NEW-S1 |
| — | Magic bytes not a gate; no extension allow-list | **STILL OPEN** | Falls back to client `Content-Type` (`media_ingestion.py:76-83,104`), keeps any extension (`:155`). HTML bytes sent as `image/png` named `x.html` stored as `<uuid>.html`; JPEG sent as `audio/wav` named `.exe` accepted [EXECUTED] |
| — | Cross-user access | **FIXED** | Owner-scoped repository helpers (`app/db/repository.py:76-131,411-424`); user B gets 404 on all of user A's media, cases and analyses [EXECUTED] |
| — | CORS | **OK** | Fixed localhost allow-list (`app/main.py:87-93`); foreign origin rejected [EXECUTED] |
| — | Secrets in repo | **OK** | Only `.env.example` tracked; the previously committed `backend/.env` held no secrets and was removed in `341cfc5` [EXECUTED] |

### 3.2 New security / robustness issues

| ID | Severity | Issue | Evidence |
|---|---|---|---|
| NEW-S1 | Medium | Every 422 response carries a wrong `Content-Length`: the handler copies the headers (including `content-length`) of a smaller envelope onto a larger body. Observed header 127, body 248 bytes. Under uvicorn/h11 this is likely to abort the response [inferred]. | `app/core/exceptions.py:171-186` [READ+EXECUTED] |
| NEW-S2 | Low/Med | Analysis runs synchronously in the request. Rate limit is 10/min/IP but each run can take minutes; signup at 5/h/IP allows multiplying the per-account daily quota. | `app/api/v1/endpoints/analysis.py:69` [READ] |
| NEW-S3 | Low | `/health` is unauthenticated and lists detector class names. `/docs` and `/openapi.json` are probably public (FastAPI default) [UNVERIFIED]. | [EXECUTED] |
| NEW-S4 | Info | Pipeline opens `media.file_path` without re-confining it (server-written value, so defence in depth only); `ensure_safe_path` is unused. | `analysis_pipeline.py:153`, `app/core/security.py:35` [READ] |
| NEW-S5 | Info | 12-hour tokens, no logout or revocation list (deactivating an account does revoke). | `app/core/auth.py:376-381` [READ] |

---

## 4. Detectors

### 4.1 Checkpoints on disk [EXECUTED]

Unchanged: the only model files are `models/face_landmarker.task` (MediaPipe) and `yolov8n.pt` (not wired). There is **no** `.pth`, `.onnx` or `.safetensors` forensic checkpoint, and no `backend/.env`.

### 4.2 Capability matrix

| Detector | Wired? | State | Notes |
|---|---|---|---|
| ForensicImageDetector (ELA/FFT/noise/EXIF) | Yes (image + video) | PARTIALLY_IMPLEMENTED | Literal `0.1` still fed into the composite on every failed sub-analysis (`app/services/detection/detectors/image/forensic_image_detector.py:155,181,235,257,280`); missing camera EXIF scores 0.40 (`:325`) |
| ManipulationMaskGenerator | Yes | PARTIALLY_IMPLEMENTED — **flags every image tested** | See §4.3 |
| CopyMoveDetector (SIFT + RANSAC) | Yes | IMPLEMENTED_BUT_UNVERIFIED | Confidence still `0.3 + 0.6*ratio` (`copy_move_detector.py:265`) |
| FaceRegionAnalyzer (MediaPipe) | Yes | PARTIALLY_IMPLEMENTED | Positive bias still present: 0.3 per Hough circle (`face_region_analyzer.py:546`), keep ≥0.3 (`:327`), average of survivors (`:213-218`) |
| BaselineFaceDetector (Haar) | Yes (image + video) | IMPLEMENTED_BUT_UNVERIFIED (heuristic) | Score 0.15 + 0.40 / + 0.25 steps (`face_detector.py:94-99`) |
| BaselineObjectDetector | Yes | UNAVAILABLE by design | Its presence forces every analysis to PARTIAL/INCONCLUSIVE (§4.4) |
| YOLOv8ObjectDetector | **No** | IMPLEMENTED_BUT_UNVERIFIED | Weights present; never constructed by the factory |
| DeepLearningFaceDetector (MTCNN) | **No** | IMPLEMENTED_BUT_UNVERIFIED | |
| TruForDetector / TruForRunner | **No** (`trufor_enabled=False`, `app/core/config.py:67`) | UNAVAILABLE | No weights; runner rejects a `state_dict` and no TruFor architecture exists in the repo (`trufor_runner.py:71-81`) |
| **AI-generation detector (learned)** | **No** | **PLANNED** | `ai_image_model_path`, `deepfake_model_path`, `audio_model_path` (`config.py:69-71`) have no readers. The only "AI-generated" signal is an EXIF/PNG keyword match plus an FFT threshold. |
| VideoDetectionEngine | Yes | PARTIALLY_IMPLEMENTED | Failure reporting broken (§4.4) |
| BaselineVideoDetector | No | IMPLEMENTED_BUT_UNVERIFIED | Still keeps only `detections[0]` (`video_detector.py:89`), moot while unwired |
| BaselineAudioAnalyzer / SpectralSpeechHeuristicDetector | No (router rejects audio) | IMPLEMENTED_BUT_UNVERIFIED, unreachable | |
| UNet segmentation | No | PLANNED (stub) | `manipulation_mask_generator.py:423,434` |

### 4.3 Previous detector bugs

| # | Bug (2026-10-20) | Status | Evidence |
|---|---|---|---|
| 1 | Mask fusion: 7 heatmaps, 6 weights, `zip()` drops the colour cue | **STILL OPEN** | `manipulation_mask_generator.py:188-196` [READ+EXECUTED] |
| 2 | TruFor weight-key mismatch | **FIXED** | Both use `"TruForDetector"` (`trufor_detector.py:34`, `image_detection_engine.py:85`); moot while disabled |
| 3 | Face-region structural positive bias | **STILL OPEN** | see matrix |
| 4 | Min-max renormalisation makes every heatmap peak at 1.0 | **STILL OPEN** — **consequence confirmed** | `manipulation_mask_generator.py:366`. On 4 unmanipulated synthetic images, `manipulation_detected=True` every time with heatmap max 1.0, and the analysis came out SUSPICIOUS [EXECUTED] |
| 5 | Clean result unreachable | **STILL OPEN** | The always-UNAVAILABLE BaselineObjectDetector lands in `incomplete` (`image_detection_engine.py:187,328`; `analysis_pipeline.py:732-735,780`). Combined with #4, the clean outcome is never reached in practice [EXECUTED] |
| 6 | Failed fallback relabelled as baseline | **FIXED** (unexercised) | YOLO/MTCNN now report themselves UNAVAILABLE (`object_detector.py:137`, `face_detector.py:229`) |
| 7 | Audio unreachable | **STILL OPEN** | `media_detection_router.py:30` raises for audio, although audio uploads are accepted (`config.py:50`) [EXECUTED] |
| 8 | Config enable switches ignored | **STILL OPEN** | Only `trufor_enabled` is read; `create_copy_move_detector` hardcodes `"SIFT"` (`detector_factory.py:67`). Ignored switches are still recorded in the run's config snapshot (`analysis_pipeline.py:63-76`), so run provenance is misleading. |
| 9 | Unguarded top-level `import torch` | **STILL OPEN** | `trufor_runner.py:7` via `detector_factory.py:7` [EXECUTED] |
| 10 | `NeuralSpeechDetector` contains no neural inference | **FIXED** | Renamed `SpectralSpeechHeuristicDetector` (`audio_detector.py:220-248`) |
| 11 | Audio verdict hardcoded | **FIXED** | Now always INCONCLUSIVE (honest); still unreachable |

### 4.4 New detector / pipeline bugs [EXECUTED unless noted]

| ID | Severity | Bug | Evidence |
|---|---|---|---|
| NEW-D1 | **High** | **A video whose frame detectors fail on every frame is reported clean**: frame detectors return `failed()` rather than raising, and only exceptions are counted, so the result is NO_SIGNIFICANT_ANOMALIES_DETECTED / COMPLETED with 0 failures. | `video_detection_engine.py:186-203,342-354` |
| NEW-D2 | **High** | Video failure counters are reset only in `__init__`; the container reuses one engine, so failures leak into later videos (a clean video showed 3 failures and came back INCONCLUSIVE). Shared mutable state is also unsafe under concurrent analyses. | `video_detection_engine.py:83-84,192,203` |
| NEW-D3 | **High** | Every unmanipulated image tested came back SUSPICIOUS (consequence of 4.3 #4: one positive detector is enough). | `manipulation_mask_generator.py:366` |
| NEW-D4 | Medium | Every image analysis ends PARTIAL: the SKIPPED BaselineObjectDetector is added to `stage_failures`. | `analysis_pipeline.py:389-396` |
| NEW-D5 | Medium | Two detectors in the same signal family (ForensicImage vs ManipulationMask, both "compression") are reported as CONTRADICTORY, against the module's own family rule. Engine and pipeline also apply different decision rules (engine CONTRADICTORY vs pipeline WEAK on the same image). | `image_detection_engine.py:203` |
| NEW-D6 | Medium | `dominant_signal` is still the argmax of raw confidence across incomparable score types. | `image_detection_engine.py:252` |
| NEW-D7 | Medium | Video engine still blends 0.5·image + 0.3·flow + 0.2·face into one number. | `video_detection_engine.py:208-213` |

### 4.5 Assessment and output contract

| Requirement | Status |
|---|---|
| No weighted mean of unrelated detector scores (image engine) | **FIXED** (`image_detection_engine.py:205-256`); still present in the video engine (NEW-D7) |
| Separate AI-generation and manipulation branches | **MISSING**: `LIKELY_AI_GENERATED` exists only as an enum value (`app/schemas/detection.py:133`); nothing produces it |
| Detector status AVAILABLE / UNAVAILABLE + execution status | **FIXED** (`schemas/detection.py:196-235`) |
| `score_semantics`, `detector_version` | **FIXED** |
| `limitations`, `execution_time_ms` per detector | **MISSING** |
| Video findings traceable to frame timestamps | **PARTIAL**: `frame_scores` and segments carry timestamps (`video_detection_engine.py:421-430,482`); region objects do not |

---

## 5. Fingerprinting, similarity, source discovery, provenance

| Capability | Status | Evidence |
|---|---|---|
| SHA-256 | IMPLEMENTED_AND_VERIFIED | |
| pHash / dHash | IMPLEMENTED_BUT_UNVERIFIED | `imagehash` in `app/services/similarity/similarity_engine.py` |
| Visual embeddings (CLIP) | **PLANNED** | `open-clip-torch` is a dependency, but no embedding is ever computed |
| FAISS / vector index | **PLANNED** | Each search loads all of the owner's fingerprints with `.all()` (`app/db/repository.py:419-424`) |
| Duplicate detection at upload | **MISSING** | `get_media_by_sha256` (`repository.py:163`) used by tests only |
| Reverse image search (Google Vision / TinEye) | **UNAVAILABLE** | Providers are coded (`internet_search_provider.py`) but never constructed; `SourceTracer()` / `SimilarityEngine()` default to the disabled provider (`analysis_pipeline.py:131-132`); `google_vision_api_key` / `tineye_api_key` (`config.py:117-118`) never read |
| Source timestamps | **BROKEN** | `_run_source_tracing` persists sources without `source_timestamp` (`analysis_pipeline.py:508-517`) |
| Provenance graph | PARTIAL | `provenance_nodes/edges` exist and appear in the evidence package |
| C2PA Content Credentials | **PLANNED** | No code |

---

## 6. Nebius / NVIDIA Nemotron agent

**PLANNED. No integration exists** [EXECUTED grep outside `venv/`]: the only mention is `docs/SECURITY.md:129` ("Nebius key once added"). `openai` (needed for the OpenAI-compatible Nebius endpoint) is installed in the venv but is not in `requirements.txt`. No Nebius API key is configured.

---

## 7. Data model [READ unless noted]

| Entity | 2026-10-20 | Now |
|---|---|---|
| Case | MISSING | **FIXED**: `CaseRecord` (`app/db/models.py:47`); `MediaRecord.case_id` RESTRICT FK (`:70`) |
| Evidence custody | PARTIAL | **PARTIAL**: only `MediaRecord.uploaded_by` (`:73`); no custody/transfer fields |
| AnalysisRun versions / config / duration | MISSING | **FIXED**: `software_version`, `detector_set_version`, `config_snapshot`, `duration_ms` (`:165-171`) |
| DetectorResult availability vs execution, `score_semantics` | MISSING | **FIXED** (`:251-256,270`) |
| Observation / Finding | MISSING / conflated | **FIXED**: `ObservationRecord`, `FindingRecord`, `FindingObservationRecord` (`:466-516`) |
| Assessment with rationale | MISSING | **FIXED**: `AssessmentRecord` with `rule`, `rationale`, `coverage_gaps` (`:521`) |
| AuditEvent | MISSING | **FIXED**: hash-chained `AuditEventRecord` (`:543`); `/audit/events`, `/audit/verify` |
| User | in-memory dict | **FIXED**: `UserRecord` (`:30`) |
| Migrations | no-op | **FIXED**: 4 revisions; `upgrade head` on an empty DB builds all 20 tables, no ORM/schema diff [EXECUTED]; runtime uses `command.upgrade` (`app/db/database.py:79-93`) |
| DB-level enums / CHECK constraints | none | **STILL OPEN**: all status columns plain VARCHAR [EXECUTED] |

---

## 8. Mock / fabricated data and evidence integrity

### 8.1 Previous findings

| Finding | Status | Evidence |
|---|---|---|
| Hardcoded health model statuses | **FIXED** | Derived in `app/core/health.py:48-119` |
| Job progress literal 100/50 | **N/A** | Jobs router removed |
| Media `"status": "STORED"` literal | **FIXED** | Derived from the filesystem (`media.py:179-193`) |
| Fabricated source-trace analysis row | **FIXED** | Route removed; disabled provider returns nothing |
| Unconditional "No validated AI detection model configured" limitation | **STILL OPEN** | Appended even with TruFor enabled (`app/services/evidence/evidence_service.py:265-270`) [EXECUTED] |
| Fabricated PDF chain of custody / analyst name | **N/A** | PDF code removed |
| Case ID synthesised per request | **STILL OPEN** | `case_id = f"AIMD-{analysis_id[:8]}"` (`evidence_service.py:61`) ignores the real `media.case_id` |

### 8.2 New integrity issues

| ID | Severity | Issue | Evidence |
|---|---|---|---|
| NEW-E1 | **High** | **Stored report contradicts the result.** The report is generated in stage 9, *before* the assessment is computed, so the persisted report says assessment `None` and status `PROCESSING` while the analysis is SUSPICIOUS/PARTIAL. | `analysis_pipeline.py:214` vs `:218-231` [READ+EXECUTED] |
| NEW-E2 | **High** | **Every detector is recorded as "returned INCONCLUSIVE"**: `_run_evidence` reads `detector_metadata["assessment"]`, which is always absent, so a SUSPICIOUS mask and a CLEAN detector both become UNCERTAINTY items. `_collect_limitations` has the same bug, so detector limitations never reach the report. | `analysis_pipeline.py:618`, `evidence_service.py:246` [EXECUTED] |
| NEW-E3 | Medium | Stored hash is never re-verified before analysis: with the stored SHA-256 set to zeros, analysis still completed, and the package contained two different hashes that are never compared. | [EXECUTED] |
| NEW-E4 | Medium | Original not preserved immutably (plain `write_bytes`, no read-only flag or working copy). | `media_ingestion.py:164` |
| NEW-E5 | Low | `ANALYSIS_STARTED` audit event is written before the analysis ID exists. | `analysis.py:57-64` |
| NEW-E6 | Low | Config snapshot records `max_image_dimension` / `max_*_duration_seconds`, which are enforced nowhere. | |

### 8.3 Evidence package contents (`evidence_service.py:224-237`)

| Spec item | Present? |
|---|---|
| Case and analysis identifiers | Analysis yes; **case ID fabricated** |
| SHA-256, size, format, metadata | Yes |
| Timestamps, software/model versions | Timestamps and `detector_version`; **no `software_version` / `config_snapshot` / `detector_set_version`** |
| Detector statuses and score meanings | **No**: uses the deprecated `det.assessment` instead of `availability` / `execution_status` / `score_semantics` |
| Localization overlays | Bounding boxes only; no mask overlays |
| Sources with retrieval timestamps | `source_timestamp` field only, and it is never populated (§5) |
| Provenance timeline | Nodes and edges |
| C2PA | No |
| Observations / findings / assessment rationale | **No**, although the tables exist |
| Audit records | **No** |
| Package-integrity hash | **No** |
| FACT / OBSERVATION / INFERENCE / UNCERTAINTY labels | Yes; no detector output is labelled FACT [EXECUTED] |

---

## 9. Operations and architecture

| Finding | Status | Evidence |
|---|---|---|
| No background jobs; pipeline runs inline in `POST /analysis` | **STILL OPEN** | `analysis.py:69` |
| Real progress / observable QUEUED state | **STILL OPEN** | Created directly as `PROCESSING` (`analysis_pipeline.py:163`) |
| PARTIAL/INCONCLUSIVE reported as COMPLETED | **FIXED** | PARTIAL persisted and shown in the UI [EXECUTED] |
| Reaper for stuck `PROCESSING`; `processing_timeout_seconds` | **STILL OPEN** | No reaper; setting has no readers (`config.py:89`) |
| Full-table-scan similarity | **STILL OPEN** (now owner-scoped) | `repository.py:419-424` |

### 9.1 Dead code (unused in `app/`) [EXECUTED grep]

- `app/core/security.py`: `sanitize_filename`, `ensure_safe_path`, `validate_file_size`, `sanitize_ffmpeg_arg`, `get_secret`
- `app/core/exceptions.py`: `InvalidMediaError`, `MediaNotFoundError`, `AnalysisNotFoundError`, `ModelUnavailableError` (never raised)
- `app/schemas`: `DetectionRequest`
- `app/db/repository.py`: `get_media_by_sha256`, `create_frame`, `get_keyframes_for_media`, `create_fingerprint`, `create_similarity_match`, `create_source`, `create_evidence_item`, `get_report_for_analysis` (tests only)
- `EvidenceService.generate_evidence_package` raises `NotImplementedError`

### 9.2 Dependency and deployment drift

| Issue | Evidence |
|---|---|
| Python 3.14.2 in venv vs 3.11.9 declared / `python:3.11-slim` in Dockerfile | [EXECUTED] |
| Imported but not declared: `mediapipe` (clean install silently disables FaceRegionAnalyzer) | `face_region_analyzer.py:103` |
| Declared but never imported: `piexif`, `google-cloud-vision`, `open-clip-torch`, `transformers`, `torchvision`, `aiofiles`, `reportlab`, `matplotlib`, `passlib`, `python-dotenv` | [EXECUTED] |
| Installed but undeclared: `openai`, `exifread`, `psycopg`, `psycopg2-binary`, `torchreid` | [EXECUTED] |
| `docker-compose.yml`: `aimd_db:/app/backend` volume masks the image's code (rebuilds keep stale code); `APP_ENV=production` without `AIMD_JWT_SECRET`, which auth refuses at startup; `aimd_reports` volume unused | [READ] |
| `Dockerfile` copies `web/dist`, but `app/main.py` mounts no static files, so the UI is never served; `libgl1-mesa-glx` likely unavailable on current Debian slim [UNVERIFIED] | [READ] |
| `.env.example` lists keys nothing reads (`REPORT_DIR`, `LOGS_DIR`, `MAX_*_DURATION`, `MAX_IMAGE_DIMENSION`, `PROCESSING_TIMEOUT`, `DEVICE`, `*_MODEL_PATH`, `GOOGLE_VISION_API_KEY`, `TINEYE_API_KEY`), omits `RATE_LIMIT_SIGNUP` / `DAILY_UPLOAD_LIMIT`, and suggests a psycopg2 URL with no driver declared | [READ] |

---

## 10. Frontend (`web/`) [READ]

All requests go through `web/src/api.ts` to `/api/v1`; **no mock or simulated data**, and every endpoint it calls exists in the backend.

| Spec screen | Status |
|---|---|
| Case management | Present ("Folders") |
| Upload with progress / failure states | Present (`Dropzone.tsx`); states only, no real progress |
| Detector results, assessment, rationale | Present (`Result.tsx`) |
| Localization overlay | Bounding boxes (`Evidence.tsx`); no heatmap, no opacity control |
| Original vs analyzed comparison | Missing |
| Video frame timeline | Missing (`suspicious_segments` ignored) |
| Related-source gallery | Missing |
| Provenance timeline | Missing |
| Report viewer / export | Missing; the report is persisted but no route or screen exposes it |
| AI agent summary | Missing (no backend) |

---

## 11. Documentation accuracy [READ]

- `README.md` is mostly accurate but lists a `_to_delete/` folder that does not exist.
- `docs/ARCHITECTURE.md` overclaims: WebSocket updates, a Provenance Graph Inspector and Chain of Custody UI, "Legal-Grade" reporting, 512-d OpenCLIP embeddings, working TruFor, Google Vision/TinEye connectors, and classes named `NeuralSpeechDetector` / `VideoTemporalDetector` (the real classes are `SpectralSpeechHeuristicDetector` / `BaselineVideoDetector`). None of these exist as described.
- Nemotron appears only as an unchecked roadmap item. No accuracy figures are claimed anywhere.

---

## 12. What is good and should be kept

1. **Auth and tenancy are now solid**: env-only key, DB-backed users and roles, router-level auth, owner-scoped queries verified against a second user.
2. **Real migrations and a forensic schema** that already models Case → Media → AnalysisRun → DetectorResult → Observation → Finding → Assessment, plus a hash-chained audit log.
3. **Honest availability semantics**: `TruForRunner` raises rather than guesses; detectors separate availability from execution; `SpectralSpeechHeuristicDetector` is named for what it is; YOLO/MTCNN no longer impersonate baselines.
4. **No blended confidence in the image engine.**
5. **Frontend talks only to the real API.**
6. **311 passing tests**, including availability semantics, path confinement, ownership and rate limits.

---

## 13. Capability summary

| Capability | State |
|---|---|
| Upload + SHA-256 + persistence | IMPLEMENTED_AND_VERIFIED (not immutable; hash not re-verified) |
| Metadata extraction | IMPLEMENTED_BUT_UNVERIFIED |
| Heuristic image forensics (ELA/FFT/noise, copy-move, face regions) | PARTIALLY_IMPLEMENTED (uncalibrated; false positives on clean images) |
| Manipulation localization (learned, pixel-level) | **UNAVAILABLE** (no TruFor weights or architecture) |
| AI-generation detection (learned) | **PLANNED** |
| Video analysis | PARTIALLY_IMPLEMENTED (failure reporting broken) |
| Audio analysis | UNAVAILABLE (router rejects audio) |
| Fingerprints (SHA-256, pHash, dHash) | IMPLEMENTED_BUT_UNVERIFIED |
| Embedding similarity / FAISS | PLANNED |
| External source discovery | UNAVAILABLE (providers never wired) |
| Provenance timeline | PARTIAL |
| C2PA | PLANNED |
| Case management | IMPLEMENTED_AND_VERIFIED |
| Observation / Finding / Assessment entities | IMPLEMENTED_BUT_UNVERIFIED (not yet in the evidence package) |
| Audit trail | IMPLEMENTED_BUT_UNVERIFIED (not in the evidence package) |
| Report generation | PARTIALLY_IMPLEMENTED (stale assessment, mislabelled detectors, not exposed) |
| **Nebius / Nemotron agent** | **PLANNED — no code** |
| Background jobs | MISSING |
| AuthN / AuthZ | IMPLEMENTED_AND_VERIFIED |
| Frontend workflow | PARTIALLY_IMPLEMENTED (upload → result works; sources, provenance, reports, agent missing) |

---

## 14. Prioritised remediation

1. **P0: make results truthful.** Fix min-max renormalisation and the 7-vs-6 fusion (NEW-D3); stop counting the by-design-UNAVAILABLE object detector as a failure/incomplete (NEW-D4, 4.3 #5); fix video failure counting and per-run reset (NEW-D1, NEW-D2); generate the report after the assessment (NEW-E1); fix evidence labelling and limitations (NEW-E2).
2. **P0: fix the 422 `Content-Length` bug** (NEW-S1).
3. **P1: learned detectors.** Add the TruFor architecture and weights (check the licence), and a learned AI-generation detector (AIDE or UnivFD) reported as a separate branch; add `limitations` / `execution_time_ms` to the detector contract.
4. **P1: Nebius/Nemotron agent** with allowlisted tools; add `openai` to requirements.
5. **P2: evidence package completeness**: real case ID, run versions/config, availability/execution/score semantics, observations/findings/assessment, audit records, integrity hash; hash re-verification and immutable originals.
6. **P2: source discovery**: wire a reverse-search provider from config, persist retrieval timestamps, add CLIP embeddings.
7. **P3: background jobs** with QUEUED state, real progress, timeout and reaper.
8. **P3: upload hardening**: streaming size cap, magic-byte gate, extension allow-list; stop leaking exception text.
9. **P3: dependency hygiene**: settle on Python 3.11/3.12, reconcile `requirements.txt`, fix `docker-compose.yml` / `Dockerfile` / `.env.example`, honour or remove ignored config switches.
10. **P4: frontend**: report viewer/export, heatmap overlay with opacity, video timeline, sources, provenance, agent summary.
11. **P4: docs**: correct `docs/ARCHITECTURE.md` overclaims.

---

## 15. Open questions [UNVERIFIED]

1. Will a real TruFor release load once its architecture is vendored, and does it fit in the RTX 3050's 4 GB at useful resolutions?
2. Do `faiss-cpu`, `c2pa-python` and the TruFor/AIDE dependencies have Python 3.14 wheels? If not, the venv must move to 3.11/3.12.
3. False-positive rates of the remaining heuristic thresholds: there is still no labelled validation set in the repo.
4. Does the 422 `Content-Length` mismatch actually break responses under uvicorn in practice (inferred, not observed)?
5. Are `/docs` and `/openapi.json` publicly reachable in production configuration?
