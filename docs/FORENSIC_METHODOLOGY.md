# AIMD Forensic Methodology & Verification Principles

## Core Principles

1. **No Fabrication**: AIMD never fabricates detection confidence, origin timestamps, reverse image search results, or forensic verdicts.
2. **Four-Tier Evidence Classification**:
   - **FACT**: Cryptographically verified parameters (SHA-256, byte length, MIME type, verified EXIF).
   - **OBSERVATION**: Direct algorithmic detections (ELA compression discrepancies, high-frequency FFT power spikes, MTCNN face boundary gradient variance, YOLOv8 object coordinates).
   - **INFERENCE**: Contextual deductions (Earliest discovered web occurrence, cosine similarity clustering).
   - **UNCERTAINTY / LIMITATION**: Explicit disclosure of missing models, unindexed web regions, or heuristic constraints.
3. **Multi-Detector Consensus**:
   - Individual detector outputs are evaluated collectively.
   - TruFor deep neural model provides localized pixel anomaly heatmaps.
   - Traditional ELA and FFT complement deep learning with compression and sensor frequency traces.
4. **Honest State Reporting**:
   - `MODEL_AVAILABLE` vs `MODEL_UNAVAILABLE`
   - `CONFIGURED` vs `UNAVAILABLE` search providers
