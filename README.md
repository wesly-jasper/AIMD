# AIMD — AI Media Detection & Digital Forensics Platform

**AIMD** is an enterprise-ready, modular AI-powered media forensics platform engineered to analyze images, videos, and audio for signs of synthetic generation, manipulation, splicing, and deepfakes. It establishes provenance graphs, locates earliest-known sources, and compiles forensically sound evidence packages with strict evidential classifications.

---

## 🏗️ System Architecture

```
                          ┌──────────────────────────┐
                          │   FastAPI REST API       │
                          │   /api/v1 Endpoints      │
                          └─────────────┬────────────┘
                                        │
           ┌────────────────────────────┼────────────────────────────┐
           ▼                            ▼                            ▼
   ┌───────────────┐            ┌───────────────┐            ┌───────────────┐
   │ Media         │            │ Analysis      │            │ Investigation │
   │ Ingestion     │            │ Pipeline (1-9)│            │ & Evidence    │
   └───────┬───────┘            └───────┬───────┘            └───────┬───────┘
           │                            │                            │
           └────────────────────────────┼────────────────────────────┘
                                        │
                                        ▼
                     ┌─────────────────────────────────────┐
                     │    SQLAlchemy Unified Database      │
                     │  (SQLite / PostgreSQL Persistence)  │
                     └──────────────────┬──────────────────┘
                                        │
           ┌────────────────────────────┴────────────────────────────┐
           ▼                                                         ▼
┌─────────────────────────────────────┐   ┌─────────────────────────────────────┐
│          Multimodal Detectors       │   │        Provenance & Forensics       │
│  - Forensic ELA, Noise & FFT        │   │  - Cryptographic Hashes (SHA-256)   │
│  - TruFor Dense Artifact Runner     │   │  - Perceptual Hashes (pHash, dHash) │
│  - Facial Symmetry & Deepfake       │   │  - Provenance Directed Acyclic Graph│
│  - Video Temporal Optical Flow      │   │  - Epistemic Evidence Classification│
│  - Audio Spectral Flux & Centroid   │   │  - Dual Markdown & JSON Case Reports│
└─────────────────────────────────────┘   └─────────────────────────────────────┘
```

---

## 🔬 9-Stage Forensic Analysis Pipeline

1. **Ingestion & Integrity Guard**: Validates file content using magic byte signatures, sanitizes filenames, enforces file size limits, computes streaming SHA-256 cryptographic hashes, and persists records to the database.
2. **Technical Metadata Extraction**: Extracts EXIF attributes, color profiles, video codecs, framerates, durations, audio sample rates, and channel layouts.
3. **Frame & Keyframe Preprocessing**: Timestamp-aware frame sampling and visual difference (scene-change) keyframe selection.
4. **Multimodal Manipulation Detection**: Evaluates media across independent image, facial, temporal, and acoustic detectors.
5. **Spatial & Temporal Localization**: Identifies exact manipulation bounding boxes (`bbox`) and temporal segments (`start_timestamp`, `end_timestamp`).
6. **Perceptual Fingerprinting**: Generates standardized DCT-based pHash, gradient dHash, and feature vectors.
7. **Similarity Search**: Performs local database perceptual matching and interfaces with external reverse-media search providers.
8. **Source Tracing & Provenance**: Discovers the earliest known online/local occurrences and constructs directed provenance graphs.
9. **Forensic Evidence Package & Case Reporting**: Synthesizes all findings into a structured evidence package and formal case report.

---

## ⚖️ Evidentiary Standards

Forensic credibility requires clear demarcation of certainty:

| Category | What AIMD Can State | What AIMD Cannot Claim |
| :--- | :--- | :--- |
| **`[FACT]`** | Cryptographic integrity (SHA-256 match), exact byte size, container metadata, perceptual hash values. | Cannot claim knowing the subjective intent of the creator. |
| **`[OBSERVATION]`** | Specific signal anomalies, pixel resampling artifacts, ELA discrepancy scores, acoustic discontinuities. | Cannot state an anomaly is deliberate forgery rather than re-compression. |
| **`[INFERENCE]`** | Earliest known online occurrence discovered by AIMD; likelihood of synthetic origin. | Cannot state "this is definitely the absolute origin" if unindexed sources exist. |
| **`[UNCERTAINTY]`** | Explicitly records model limitations, low-resolution warnings, missing audio tracks, or baseline heuristic statuses. | Never fabricates certainty or makes unsupported claims. |

---

## 📡 API Reference

### Health & System
* `GET /health` or `GET /api/v1/health`: System health and database connectivity status.

### Media Ingestion
* `POST /api/v1/media/upload`: Secure multipart upload with magic-byte validation, SHA-256 hashing, and initial preprocessing.
* `GET /api/v1/media/`: List all ingested media records.
* `GET /api/v1/media/{media_id}`: Retrieve stored metadata and fingerprints.

### Forensic Analysis Pipeline
* `POST /api/v1/analysis/`: Trigger the full 9-stage analysis pipeline for a `media_id`.
* `GET /api/v1/analysis/{analysis_id}`: Retrieve analysis status and findings.
* `GET /api/v1/analysis/`: List all stored investigation records.

### Investigation & Forensics
* `POST /api/v1/detection/analyze`: Run multimodal detection standalone.
* `GET /api/v1/detection/{analysis_id}`: Retrieve localized detections.
* `POST /api/v1/fingerprint/generate`: Compute SHA-256, pHash, dHash, and embeddings.
* `POST /api/v1/similarity/search`: Search local and external indexes for similar media.
* `POST /api/v1/source/trace`: Execute source tracing.
* `GET /api/v1/source/{analysis_id}`: Retrieve earliest-known source occurrence.
* `GET /api/v1/provenance/{analysis_id}`: Retrieve provenance graph nodes and edges.
* `GET /api/v1/evidence/{analysis_id}`: Fetch complete machine-readable evidence package.
* `GET /api/v1/report/{analysis_id}`: Fetch dual-format (JSON and Markdown) forensic case reports.

---

## 🚀 Getting Started

### 1. Backend Setup

```bash
cd backend
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Database & Backend API

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive OpenAPI documentation will be accessible at: `http://localhost:8000/docs`

### 3. Run Frontend Dashboard

```bash
cd frontend
npm install
npm run dev
```
The React forensic dashboard will be accessible at: `http://localhost:5173`

### 4. Run Automated Test Suite

```bash
cd backend
python -m pytest tests/ -v
```
Executes all 93 unit and end-to-end integration tests.
