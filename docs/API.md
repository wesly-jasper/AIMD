# AIMD API Documentation

Base URL: `/api/v1`

## Endpoints

### Authentication
- `POST /api/v1/auth/login`: Authenticate with username and password, returns JWT bearer token.
- `POST /api/v1/auth/register`: Register new investigator account.
- `GET /api/v1/auth/me`: Get current authenticated user profile.

### Media Ingestion
- `POST /api/v1/media/upload`: Upload image, video, or audio file.
- `GET /api/v1/media/{media_id}`: Retrieve media record details.
- `GET /api/v1/media/{media_id}/metadata`: Retrieve parsed EXIF/codec properties.

### Analysis & Orchestration
- `POST /api/v1/analysis/run`: Trigger full multi-stage analysis pipeline for media ID.
- `GET /api/v1/analysis/{analysis_id}/status`: Check analysis execution status.

### Investigation & Evidence
- `GET /api/v1/investigation/{analysis_id}/package`: Retrieve complete forensic evidence package.
- `GET /api/v1/investigation/{analysis_id}/report`: Retrieve Markdown and JSON forensic reports.
- `GET /api/v1/investigation/{analysis_id}/provenance`: Retrieve provenance DAG nodes and edges.
- `POST /api/v1/investigation/fingerprint/generate`: Generate SHA-256, pHash, dHash, and OpenCLIP embeddings.
- `POST /api/v1/investigation/similarity/search`: Run similarity search against stored fingerprints.

### System
- `GET /health` or `GET /api/v1/health`: System health and database connectivity.
