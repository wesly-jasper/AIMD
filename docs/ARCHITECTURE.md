# AIMD Architecture Documentation

## System Overview

AIMD (AI Media Detection & Digital Forensics) is an end-to-end multi-modal digital forensics platform designed to determine whether images, videos, or audio files are AI-generated, forged, or manipulated.

```
┌─────────────────────────────────────────────────────────────┐
│                    AIMD Frontend (React/Vite)               │
│  - Cyber-Forensic Dashboard    - Real-Time Case Overview    │
│  - Provenance Graph Inspector  - Evidence Chain of Custody  │
└──────────────────────────────┬──────────────────────────────┘
                               │ REST API / WebSocket
┌──────────────────────────────▼──────────────────────────────┐
│                    AIMD FastAPI Backend                     │
│  - Ingestion & Storage Service                              │
│  - Pipeline Orchestrator                                    │
│  - Multi-Modal Forensic Detection Router                    │
│  - Fingerprinting & OpenCLIP Semantic Search                │
│  - Provenance DAG Graph Builder                             │
│  - Evidence & Legal-Grade Reporting Service                 │
└──────────────────────────────┬──────────────────────────────┘
                               │
       ┌───────────────────────┼───────────────────────┐
       ▼                       ▼                       ▼
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│ Image Engine │       │ Video Engine │       │ Audio Engine │
│ - TruFor     │       │ - Temporal   │       │ - Neural     │
│ - ELA / FFT  │       │ - Frame Dets │       │   Speech     │
│ - MTCNN Face │       │ - Keyframe   │       │ - Spectral   │
│ - YOLOv8     │       │   Selector   │       │   Flux/MFCC  │
└──────────────┘       └──────────────┘       └──────────────┘
```

## Core Modules

1. **Ingestion & Preprocessing**: Media format validation, SHA-256 verification, metadata/EXIF extraction, video scene-based keyframe extraction.
2. **Detection Layer**:
   - `TruForDetector` / `TruForRunner`: Deep neural RGB-noise pixel anomaly localization.
   - `ForensicImageDetector`: Error Level Analysis (ELA), 2D-FFT azimuthal frequencies, Laplacian noise variance.
   - `DeepLearningFaceDetector`: MTCNN facial landmarks and boundary gradient forensics.
   - `YOLOv8ObjectDetector`: Semantic object coordinates and classes.
   - `NeuralSpeechDetector`: Mel-spectrogram formant smoothness and voice synthesis detection.
   - `VideoTemporalDetector`: Frame-to-frame inter-frame optical flow & histogram drift.
3. **Fingerprinting & Similarity Engine**:
   - Cryptographic SHA-256
   - Perceptual pHash & dHash
   - 512-dim L2-normalized OpenCLIP ViT-B-32 semantic embedding vectors.
4. **Provenance & Source Tracing**:
   - DAG provenance graphs with relationships (`reposted_from`, `visually_similar_to`).
   - Reverse image search provider connectors (Google Cloud Vision Web Detection, TinEye).
5. **Evidence & Reporting**:
   - Structured Chain of Custody categorizing findings as `FACT`, `OBSERVATION`, `INFERENCE`, `UNCERTAINTY`.
   - Comprehensive Markdown & JSON forensic report generation.
