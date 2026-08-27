# AIMD — AI Media Detection & Digital Forensics Platform

**AIMD** is an enterprise-ready, modular AI-powered media forensics platform engineered to analyze images and videos for signs of synthetic generation, manipulation, splicing, and deepfakes. It establishes provenance graphs, locates earliest-known sources across the internet, and compiles forensically sound evidence packages with clear evidentiary classifications.

---

## Architecture Overview

```
                          ┌──────────────────────────┐
                          │       FastAPI API        │
                          │   /api/v1 Endpoints      │
                          └─────────────┬────────────┘
                                        │
             ┌──────────────────────────┼──────────────────────────┐
             ▼                          ▼                          ▼
     ┌───────────────┐          ┌───────────────┐          ┌───────────────┐
     │ Media         │          │ Detection     │          │ Investigation │
     │ Ingestion     │          │ Engine        │          │ Engine        │
     └───────┬───────┘          └───────┬───────┘          └───────┬───────┘
             │                          │                          │
             ▼                          ▼                          ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                          Analysis Layer                                   │
│  - Image (TruFor / Baselines)                                             │
│  - Face (Haar Cascades / Deepfake Heuristics)                             │
│  - Object & Noise Inconsistencies                                         │
│  - Video Temporal Continuity                                              │
│  - Audio Waveform & Synthetic Speech Analysis                             │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                     Fingerprinting & Similarity                           │
│  - SHA-256 Cryptographic Hash                                             │
│  - Perceptual Hashing (pHash)                                             │
│  - Baseline Color / Spatial Feature Embeddings                            │
│  - Internet-Scale Similarity Engine                                       │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                      Source & Provenance Engine                           │
│  - Earliest-Known Matching Occurrence Tracing                             │
│  - Lineage & Relationship Graph (derived_from, visually_similar_to)       │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                      Forensic Evidence Engine                             │
│  - Standardized JSON Evidence Package                                     │
│  - Human-Readable Report [FACT / OBSERVATION / INFERENCE / UNCERTAINTY]   │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Phases

1. **Ingestion & Preprocessing**: Validates media format, extracts rich metadata, computes SHA-256 hashes, extracts video keyframes, and generates perceptual hashes.
2. **Multimodal Manipulation Detection**: Evaluates media across independent image, face, object, video, and audio detectors.
3. **Manipulation Localization**: Highlights spatial regions with bounding boxes (`bbox`) and temporal segments with `start_frame`, `end_frame`, and timestamps.
4. **Fingerprinting & Similarity**: Computes perceptual hashes, embeddings, and conducts similarity search over online reverse-media providers.
5. **Source Tracing**: Identifies the earliest known occurrence by cross-referencing candidate discovery and source timestamps.
6. **Provenance Reconstruction**: Generates directed acyclic provenance graphs detailing lineage between matching assets.
7. **Forensic Evidence Generation**: Synthesizes all findings into structured evidence packages and human-readable forensic reports.

---

## API Reference

### Health & System
* `GET /health` or `GET /api/v1/health`: System health check.

### Ingestion & Media
* `POST /api/v1/media/upload`: Upload image, video, or audio for initial validation, SHA-256 hashing, metadata extraction, and keyframing.
* `GET /api/v1/media/{media_id}`: Retrieve stored metadata and fingerprints for an uploaded media ID.

### Detection & Analysis
* `POST /api/v1/detection/analyze`: Run multimodal detection on a file path. Returns aggregated detections and `analysis_id`.
* `GET /api/v1/detection/{analysis_id}`: Retrieve previously executed detection results.

### Investigation & Provenance
* `POST /api/v1/fingerprint/generate`: Generate SHA-256, pHash, and feature embeddings for any media file.
* `POST /api/v1/similarity/search`: Search internet-scale index for visually or perceptually similar media.
* `POST /api/v1/source/trace`: Execute source tracing to find earliest-known occurrences and candidates.
* `GET /api/v1/source/{analysis_id}`: Retrieve source tracing results for a given analysis.
* `GET /api/v1/provenance/{analysis_id}`: Retrieve graph nodes and edges representing media provenance.
* `GET /api/v1/evidence/{analysis_id}`: Fetch complete, machine-readable forensic evidence package.
* `GET /api/v1/report/{analysis_id}`: Fetch dual-format (JSON and Markdown) forensic reports.

---

## Evidentiary Integrity: What AIMD Can and Cannot Prove

Forensic credibility requires clear demarcation of certainty:

| Category | What AIMD Can Prove | What AIMD Cannot Prove |
| :--- | :--- | :--- |
| **FACT** | Cryptographic integrity (SHA-256 match), exact file size, container metadata, perceptual hash distance. | Cannot prove whether the creator had malicious intent. |
| **OBSERVATION** | Specific signal anomalies, pixel resampling artifacts, detected face regions, acoustic silence patterns. | Cannot prove an anomaly is deliberate manipulation rather than re-compression. |
| **INFERENCE** | Earliest known online occurrence based on crawled timestamps; high likelihood of synthetic origin. | Cannot state "this is definitely the absolute origin" if offline or unindexed sources exist. |
| **UNCERTAINTY** | Explicitly records model limitations, low-resolution warnings, missing audio tracks, or baseline heuristic statuses. | Never makes unsupported definitive claims. |

---

## Running the Platform

### Prerequisites
* Python 3.11+
* OpenCV (`opencv-python` / `opencv-python-headless`)
* FastAPI, Uvicorn, Pydantic, Pillow, imagehash

### Run Backend Server
```powershell
& "<python_path>" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive OpenAPI documentation will be accessible at: `http://localhost:8000/docs`

### Run Complete Test Suite
```powershell
& "<python_path>" -m pytest
```
All 83 automated unit and integration tests will execute and validate the pipeline.
