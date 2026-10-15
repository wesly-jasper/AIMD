# AIMD Database Schema Documentation

ORM: SQLAlchemy 2.0 (Compatible with SQLite & PostgreSQL)
Migrations: Alembic

## Tables

### `media`
- `id` (VARCHAR PK)
- `filename` (VARCHAR)
- `original_filename` (VARCHAR)
- `file_path` (VARCHAR)
- `media_type` (VARCHAR: image | video | audio)
- `mime_type` (VARCHAR)
- `size_bytes` (INTEGER)
- `sha256` (VARCHAR UNIQUE)
- `created_at` (TIMESTAMP)

### `media_metadata`
- `id` (VARCHAR PK)
- `media_id` (FK `media.id`)
- `format`, `width`, `height`, `duration_seconds`, `fps`, `frame_count`, `codec`
- `exif_data` (JSON), `raw_metadata` (JSON)

### `analyses`
- `id` (VARCHAR PK)
- `media_id` (FK `media.id`)
- `status` (VARCHAR: QUEUED | PROCESSING | COMPLETED | FAILED | PARTIAL)
- `assessment` (VARCHAR: CLEAN | SUSPICIOUS | INCONCLUSIVE | UNAVAILABLE)
- `overall_confidence` (FLOAT)
- `started_at`, `completed_at`

### `detections`
- `id` (VARCHAR PK), `analysis_id` (FK `analyses.id`)
- `detector` (VARCHAR), `media_type` (VARCHAR), `manipulation_type` (VARCHAR)
- `manipulation_detected` (BOOLEAN), `confidence` (FLOAT), `assessment` (VARCHAR)
- `detector_metadata` (JSON)

### `detection_regions`
- `id` (VARCHAR PK), `detection_id` (FK `detections.id`)
- `region_type` (VARCHAR), `confidence` (FLOAT), `bbox` (JSON `[x, y, w, h]`), `mask_path` (VARCHAR)

### `fingerprints`
- `id` (VARCHAR PK), `media_id` (FK), `algorithm` (sha256 | phash | dhash | embedding), `value` (TEXT)

### `provenance_nodes` & `provenance_edges`
- Directed Acyclic Graph representing media relationships, similarity matches, and web sources.

### `evidence_items`
- Structured forensic items categorized as `FACT`, `OBSERVATION`, `INFERENCE`, or `UNCERTAINTY`.

### `reports`
- `id` (VARCHAR PK), `analysis_id` (FK `analyses.id`), `json_report` (JSON), `markdown_report` (TEXT), `conclusion` (TEXT).
