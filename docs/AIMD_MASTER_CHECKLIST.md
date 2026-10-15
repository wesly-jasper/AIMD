# AIMD — Master Project Checklist

This file is the permanent project checklist for the **AIMD (AI Media Detection & Digital Forensics)** platform. It is maintained across development phases and updated after every meaningful implementation.

Legend:
* `[ ]` Not started
* `[~]` In progress
* `[x]` Completed
* `[!]` Blocked

---

## PHASE 1 — INGESTION & PREPROCESSING

### Media ingestion
* [x] Image upload (`POST /api/v1/media/upload`)
* [x] Video upload (`POST /api/v1/media/upload`)
* [x] Audio upload (`POST /api/v1/media/upload`)
* [x] MIME validation via magic bytes (file signature check)
* [x] Extension validation (whitelist enforcement)
* [x] File-size validation (configurable MB limit)
* [x] Corrupt-file detection (OpenCV / PIL / Librosa decode verification)
* [x] Unique media ID (UUIDv4)
* [x] Secure storage (sanitized paths in `uploads/`)
* [x] SHA-256 generation (streaming 64KB block cryptographic hash)
* [x] Metadata extraction
* [x] EXIF extraction (PIL Tag extraction)
* [x] Media dimensions (Width x Height)
* [x] Video FPS
* [x] Video duration
* [x] Audio sample rate
* [x] Codec information (FFprobe/cv2 stream metadata)

### Video preprocessing
* [x] Frame extraction (`FrameExtractor`)
* [x] Configurable frame sampling (`analysis_fps`)
* [x] Keyframe extraction (`KeyframeSelector`)
* [x] Frame numbering
* [x] Frame timestamps
* [x] Keyframe metadata
* [x] Failed-frame handling
* [x] Large-video handling

---

## PHASE 2 — MULTIMODAL DETECTION

### Image
* [x] Error Level Analysis (ELA with compression resave diff)
* [x] 2D-FFT high frequency spectral anomaly analysis
* [x] Noise residual analysis (Laplacian variance)
* [x] Metadata anomaly analysis (software tags, missing timestamps)
* [x] Image manipulation score (weighted composite score)
* [x] Region localization (connected component thresholding)
* [x] Bounding boxes (`[x, y, w, h]` format)
* [x] Confidence values (calibrated 0.0 - 1.0)
* [x] Detector explanations (forensic rationale per finding)

### Face
* [x] Face detection (OpenCV Haar Cascade)
* [x] Face crop extraction
* [x] Face anomaly analysis (boundary gradient sharpness)
* [x] Face localization
* [x] Multiple-face support
* [x] Face manipulation score

### Object
* [x] Object detector interface (`BaseObjectDetector`)
* [x] Object detector placeholder (`BaselineObjectDetector` with explicit `UNAVAILABLE`)
* [ ] Real object detection model (YOLOv8 / Faster R-CNN integration)
* [ ] Object localization
* [ ] Object confidence
* [ ] Object/manipulation relationship
* [ ] Multiple-object support

### Video
* [x] Frame-level analysis
* [x] Temporal consistency (Farnebäck optical flow magnitude)
* [x] Optical-flow analysis
* [x] Face consistency across frame sequences
* [x] Frame anomaly score
* [x] Temporal anomaly score
* [x] Suspicious segments grouping
* [x] Start frame
* [x] End frame
* [x] Start timestamp
* [x] End timestamp

### Audio
* [x] Audio extraction (WAV / MP3 / AAC decoding)
* [x] RMS energy analysis
* [x] Spectral analysis (STFT magnitude)
* [x] MFCC feature extraction
* [x] Spectral flux calculation
* [x] Audio discontinuity detection
* [ ] Synthetic speech model (e.g. RawNet2 / Wav2Vec2)
* [x] Voice manipulation heuristic baseline
* [x] Timestamp localization

---

## PHASE 3 — ADVANCED AI DETECTION

### TruFor
* [x] Verify current TruFor integration structure (`TruForDetector`, `TruForRunner`)
* [ ] Obtain compatible model weights (`trufor.pth`)
* [x] Model loading interface (`TruForRunner.load`)
* [x] Preprocessing interface
* [~] Inference execution (Raises `NotImplementedError` when weights absent)
* [ ] Confidence map extraction
* [ ] Localization map extraction
* [x] Model error handling
* [ ] CPU fallback execution
* [ ] GPU support (CUDA)
* [x] Integration with detection engine
* [x] Integration tests for TruFor disabled and missing weights

### Deep semantic image embeddings
* [x] Model configuration (`settings.py`)
* [x] Embedding provider interface (`BaseEmbeddingProvider`)
* [x] Baseline visual features (`BaselineEmbeddingProvider` 8x8x8 color histogram)
* [x] Explicit labeling of baseline features
* [ ] Deep semantic model loading (CLIP / OpenCLIP / DINOv2)
* [ ] Image preprocessing for neural network
* [ ] Deep embedding generation
* [ ] Embedding normalization (L2 norm)
* [x] Version tracking (`algorithm_name`)
* [x] Model metadata
* [x] Error handling
* [x] Tests

---

## PHASE 4 — FINGERPRINTING & SIMILARITY

### Exact fingerprinting
* [x] SHA-256 implementation
* [x] Algorithm version tracking
* [x] Reproducibility verification

### Perceptual fingerprinting
* [x] pHash (64-bit DCT perceptual hash)
* [x] dHash (64-bit difference hash)
* [x] Additional robust fingerprint (custom DCT fallback if `imagehash` absent)
* [x] Frame-level fingerprints
* [x] Keyframe fingerprints
* [x] Video multi-frame hash (VHash)

### Similarity
* [x] Exact hash matching
* [x] pHash Hamming distance
* [x] dHash Hamming distance
* [x] Embedding cosine similarity
* [x] Weighted composite similarity
* [x] Candidate ranking
* [x] Threshold configuration
* [x] Explain similarity result
* [x] Store similarity result in database
* [x] Test false-positive behavior

---

## PHASE 5 — SOURCE DISCOVERY

### Local source discovery
* [x] Search stored media in database
* [x] Search fingerprints in database
* [x] Search embeddings in database
* [x] Candidate ranking
* [x] Similarity score calculation
* [x] Candidate metadata resolution
* [x] Timestamp information comparison

### External source discovery
* [x] Provider interface (`BaseSourceSearchProvider`)
* [x] Provider configuration (`settings.py`)
* [x] API error handling
* [x] Rate-limit handling design
* [x] Timeout handling
* [x] Result normalization (`SourceSearchResult`)
* [x] Disabled provider returning explicit `UNAVAILABLE` (`DisabledSourceSearchProvider`)
* [ ] Google Cloud Vision API connector
* [ ] TinEye Commercial API connector
* [x] Provider attribution
* [x] Never fabricate source URLs or timestamps

---

## PHASE 6 — PROVENANCE

### Provenance graph
* [x] Provenance entity models (`ProvenanceNodeRecord`, `ProvenanceEdgeRecord`)
* [x] Relationship types (`derived_from`, `visually_similar_to`, `same_hash_as`, `observed_at`)
* [x] Provenance graph creation (`ProvenanceGraphBuilder`)
* [x] Automatic graph population from pipeline results
* [x] Source candidate integration
* [x] Similarity match integration
* [x] Timeline ordering (chronological sorting)
* [x] Confidence calculation
* [x] Evidence references
* [x] Uncertainty representation
* [x] Graph API (`GET /api/v1/provenance/{analysis_id}`)
* [x] Unit and integration tests

---

## PHASE 7 — FORENSIC EVIDENCE

### Evidence classification
* [x] Epistemological classification framework:
  * `[FACT]`: Cryptographic hashes, file sizes, verified timestamps, byte lengths.
  * `[OBSERVATION]`: Algorithmic measurements, optical flow values, ELA residuals, spectral flux.
  * `[INFERENCE]`: Deductions from observations (e.g. "patterns consistent with face swap").
  * `[UNCERTAINTY]`: Known limitations, unconfigured detectors, compression artifacts.
* [x] Evidence objects (`EvidenceItemRecord`)
* [x] Unique evidence IDs (UUIDv4)
* [x] Evidence source tracking
* [x] Evidence timestamping
* [x] Detector attribution
* [x] Model attribution
* [x] Confidence recording
* [x] Supporting media/frame links
* [x] Supporting region bounding boxes
* [x] Human-readable explanations
* [x] Uncertainty documentation
* [x] Chain-of-evidence relationships

---

## PHASE 8 — FORENSIC REPORT

### Report generation
* [x] Investigation ID & Case ID
* [x] Media ID & File information
* [x] Cryptographic SHA-256 verification
* [x] Metadata breakdown
* [x] Multimodal detection summary
* [x] Detector results & model metadata
* [x] Manipulated regions & localized bounding boxes
* [x] Suspicious frames & timestamp intervals
* [x] Similarity results & database matches
* [x] Source candidates & origin platforms
* [x] Provenance timeline
* [x] Classified evidence matrix (FACT / OBSERVATION / INFERENCE / UNCERTAINTY)
* [x] Confidence score & calibration
* [x] Uncertainty & limitations section
* [x] Forensic methodology documentation
* [x] Final assessment (`CLEAN`, `SUSPICIOUS`, `INCONCLUSIVE`, `UNAVAILABLE`)
* [x] 8-section Markdown format generator
* [x] Machine-readable structured JSON format generator

---

## PHASE 9 — DATABASE

### Persistence layer
* [x] Media persistence (`MediaRecord`)
* [x] Analysis persistence (`AnalysisRecord`)
* [x] Frame persistence (`FrameRecord`)
* [x] Detection persistence (`DetectionRecord`, `DetectionRegionRecord`)
* [x] Fingerprint persistence (`FingerprintRecord`)
* [x] Similarity persistence (`SimilarityMatchRecord`)
* [x] Source candidate persistence (`SourceCandidateRecord`)
* [x] Provenance persistence (`ProvenanceNodeRecord`, `ProvenanceEdgeRecord`)
* [x] Evidence persistence (`EvidenceItemRecord`)
* [x] Report persistence (`ReportRecord`)
* [x] Foreign key relationships & cascade deletes
* [x] Database indexes on lookups (`media_id`, `analysis_id`, `sha256`)
* [x] SQLite WAL mode configuration with foreign keys enabled
* [x] PostgreSQL compatibility (SQLAlchemy 2.0 type mapping)
* [ ] Alembic migration environment

---

## PHASE 10 — API

### REST API Endpoints
* [x] Ingestion endpoint (`POST /api/v1/media/upload`)
* [x] Media detail endpoint (`GET /api/v1/media/{media_id}`)
* [x] Media stream/download endpoint (`GET /api/v1/media/{media_id}/download`)
* [x] Media listing endpoint (`GET /api/v1/media/`)
* [x] Analysis execution endpoint (`POST /api/v1/analysis/`)
* [x] Analysis status/detail endpoint (`GET /api/v1/analysis/{analysis_id}`)
* [x] Analysis listing endpoint (`GET /api/v1/analysis/`)
* [x] Detection detail endpoint (`GET /api/v1/detection/{analysis_id}`)
* [x] Standalone detection endpoint (`POST /api/v1/detection/`)
* [x] Provenance graph endpoint (`GET /api/v1/provenance/{analysis_id}`)
* [x] Source discovery endpoint (`GET /api/v1/source/{analysis_id}`)
* [x] Evidence package endpoint (`GET /api/v1/evidence/{analysis_id}`)
* [x] Report generation endpoint (`GET /api/v1/report/{analysis_id}`)
* [x] Health checks (`GET /health`, `GET /api/v1/health`)
* [x] Request validation (Pydantic models)
* [x] Error handling & HTTP status codes (404, 400, 422, 500)

---

## PHASE 11 — INVESTIGATOR FRONTEND

### Dashboard UI
* [x] Top sticky navbar with AIMD crest branding, case ID, investigator badge, and tab controls
* [x] Section 1 header pill (`1. DETECTION & ANALYSIS DASHBOARD`)
* [x] Section 2 header pill (`2. MEDIA PROVENANCE / TRACE TIMELINE`)
* [x] Uploaded media panel with decodable stream, bounding box overlays, scrubber, and danger alert card
* [x] Manipulation analysis panel with multi-signal progress bars (Face, Body, Object, Background, Audio, AI Generated)
* [x] Manipulation heatmap panel with thermal gradient overlay, target face box, vertical High-to-Low scale, and footnote
* [x] Media trace matches panel with `EARLIEST MATCH` badge, source platforms, timestamps, and external links
* [x] Media fingerprint panel with SHA-256, pHash, dHash, VHash, and `Copy Fingerprints` button
* [x] Provenance timeline panel with 5-step chevron flow (`Earliest Known Match` → `Edited / Cropped` → `AI Manipulated` → `Re-uploaded` → `Viral Spread`)
* [x] Quick actions panel (`Generate Forensic Report`, `Export Evidence Package`, `Save Investigation`, `Share with Team`)
* [x] Bottom telemetry strip (`MEDIA TYPE`, `DURATION`, `RESOLUTION`, `FRAMES ANALYZED`, `ANALYSIS TIME`, and Big Red Report CTA)
* [x] Upload modal for dragging and dropping new media files
* [x] Forensic report modal with dual Markdown/JSON views, copy to clipboard, and JSON download
* [x] Investigations history view
* [x] Search & trace DAG provenance viewer
* [x] Evidentiary classification table view (FACT / OBSERVATION / INFERENCE / UNCERTAINTY filter tabs)
* [x] Platform settings view

---

## PHASE 12 — SECURITY

### Security controls
* [x] Magic byte verification before processing
* [x] MIME type & extension whitelist validation
* [x] File-size limit enforcement
* [x] Path traversal protection via `Path.resolve()` checks
* [x] Safe filename generation (`_secure_filename`)
* [x] Strict CORS configuration
* [x] Environment variable configuration for sensitive keys
* [x] Sanitized error messages (no stack trace leakage to client)
* [ ] ClamAV malware scanning
* [ ] Sandboxed ffmpeg container execution
* [ ] Rate limiting on ingestion endpoints (e.g. `slowapi`)

---

## PHASE 13 — PERFORMANCE

### System profiling
* [x] Small image test (<1MB)
* [x] Large image test (>10MB)
* [x] Short video test (12s MP4)
* [x] Audio analysis test (WAV/MP3)
* [x] Database query indexing optimization
* [ ] Asynchronous task queue (Celery + Redis) for background processing
* [ ] Sub-millisecond vector indexing (pgvector / Qdrant)

---

## PHASE 14 — TESTING

### Automated test suite
* [x] 29 Unit test modules in `backend/tests/unit/`
* [x] 5 Integration test modules in `backend/tests/integration/`
* [x] Live E2E test module (`test_live_api_e2e.py`)
* [x] 97 total passing tests (`97 passed in 5.43s`)
* [x] Zero mock hash or dummy confidence tests

---

## PHASE 15 — END-TO-END PIPELINE

### Full verification scenario
* [x] Media Upload → Validate → Store → Extract Metadata → Extract Frames → Select Keyframes → Fingerprint → Detection Router → Multimodal Detectors → Localization → Similarity → Source Discovery → Provenance Graph → Evidence Packaging → Report Generation → Investigator Frontend.
* [x] Verified for Images, Videos, and Audio files.

---

## PHASE 16 — DEPLOYMENT

### Deployment readiness
* [x] Comprehensive `.gitignore`
* [x] `.env.example` with all configuration parameters
* [x] Clean frontend production bundle build (`dist/`)
* [ ] Multi-stage production `Dockerfile`
* [ ] `docker-compose.yml` (FastAPI + PostgreSQL + Redis + React/Vite)
* [ ] Alembic database migration scripts
