# BACKEND_ANALYSIS.md

## 1. End-to-End Agent Workflow

The AIMD (AI Media Detection) backend implements a complete forensic analysis pipeline that processes media files through multiple stages of ingestion, analysis, detection, similarity matching, provenance tracking, evidence classification, and report generation.

**Complete Flow:**
1. **File Ingestion** (`backend/app/services/ingestion/media_ingestion.py`)
   - File upload via API endpoint
   - Validation of file type and size (max 500MB)
   - SHA-256 cryptographic hash generation
   - Perceptual hashing (pHash, dHash) for similarity detection
   - Metadata extraction (EXIF, container info, dimensions, duration)
   - Frame extraction for video analysis
   - Storage in `uploads/` and `processed/` directories

2. **Database Storage** (`backend/app/db/repository.py`)
   - Media record creation with all ingested metadata
   - Fingerprint storage (SHA-256, pHash, dHash)
   - Initial Analysis record creation

3. **Full Pipeline Analysis** (`backend/app/api/v1/endpoints/analysis.py` POST `/`)
   - Triggers the 9-stage `AnalysisPipeline.run()` synchronously
   - Processes through all detection and analysis stages
   - Returns complete analysis result with evidence package

4. **Detection Engines** (`backend/app/services/detection/`)
   - **ImageDetectionEngine**: Weighted aggregation of multiple image detectors
     - ForensicImageDetector (ELA + FFT + noise analysis)
     - TruForDetector (CNN-based manipulation detection, if enabled)
     - BaselineFaceDetector (facial forensics)
     - BaselineObjectDetector (object inconsistency)
     - CopyMoveDetector (SIFT-based copy-move forgery)
     - ManipulationMaskGenerator (pixel-level heatmaps)
     - FaceRegionAnalyzer (per-facial region deepfake analysis)
   - **VideoDetectionEngine**: Temporal-spatial analysis
     - Frame sampling at configurable FPS
     - Optical flow computation for motion discontinuities
     - Spatial-temporal tracking of suspicious regions
     - Frame-level forensic image analysis
     - Temporal anomaly detection in tracked regions
     - Segment identification (contiguous suspicious frames)

5. **Similarity Search** (`backend/app/services/similarity/similarity_engine.py`)
   - Local search: SHA-256 exact match → pHash → dHash → embedding cosine similarity
   - External search: Google Vision API or TinEye (if configured)
   - Returns ranked matches with similarity scores

6. **Source Tracing** (`backend/app/services/similarity/internet_search_provider.py`)
   - Internet search for earliest known occurrence
   - Provider abstraction layer (Google, TinEye, disabled)

7. **Provenance Tracking** (`backend/app/services/provenance/source_tracer.py`)
   - Graph-based provenance modeling
   - Nodes = media items, Edges = relationships (derivative, copy, etc.)
   - Confidence scoring on relationships

8. **Evidence Classification** (`backend/app/services/evidence/evidence_service.py`)
   - Taxonomy: FACT/OBSERVATION/INFERENCE/UNCERTAINTY
   - FACT: Cryptographic hashes, file properties, EXIF
   - OBSERVATION: Algorithmically extracted signals (ELA, FFT, etc.)
   - INFERENCE: Provenance relationships, earliest source estimates
   - UNCERTAINTY: Detector limitations, inconclusive signals

9. **Report Generation** (`backend/app/services/evidence/report_generator.py`)
   - JSON report (machine-readable evidence package)
   - Markdown report (human-readable forensic report)
   - Structured sections with evidence categorization
   - Verdict: SUSPICIOUS/CLEAN/INCONCLUSIVE with confidence score

## 2. Complete API Endpoint Reference

### Authentication Router (`backend/app/api/v1/endpoints/auth.py`)
- **POST `/auth/register`**
  - Request: `{username, email, password}`
  - Response: `{access_token, token_type, user_info}`
  - Auth: None (public)
  - Creates user with bcrypt-hashed password, returns JWT

- **POST `/auth/login`**
  - Request: `{username, password}`
  - Response: `{access_token, token_type, user_info}`
  - Auth: None (public)
  - Validates credentials, returns JWT (HS256, 24h expiry)

- **GET `/auth/me`**
  - Request: None
  - Response: `{user_info}`
  - Auth: Bearer JWT required
  - Returns current user profile

### Media Router (`backend/app/api/v1/endpoints/media.py`)
- **POST `/media/upload`**
  - Request: multipart/form-data file upload
  - Response: `{media_id, filename, size, upload_timestamp, status}`
  - Auth: Bearer JWT required
  - Streaming: None (complete file upload)
  - Validates file type against allowed lists (images/video/audio)

- **GET `/media/{media_id}`**
  - Request: None
  - Response: Complete media metadata
  - Auth: Bearer JWT required
  - Returns: filename, size, content_type, sha256, metadata, timestamps

- **GET `/media/{media_id}/download`**
  - Request: None
  - Response: file binary stream
  - Auth: Bearer JWT required
  - Streaming: Yes (file download)
  - Streams file from `uploads/` directory

- **GET `/media/`**
  - Request: query params (`skip`, `limit`, `media_type`)
  - Response: paginated list of media objects
  - Auth: Bearer JWT required
  - Supports filtering and pagination

### Analysis Router (`backend/app/api/v1/endpoints/analysis.py`)
- **POST `/analysis/`**
  - Request: `{media_id}`
  - Response: Complete analysis result with evidence package
  - Auth: Bearer JWT required
  - Streaming: None (synchronous 9-stage pipeline)
  - Triggers full forensic pipeline

- **POST `/analysis/async`**
  - Request: `{media_id}`
  - Response: `{job_id, analysis_id, status, message}`
  - Auth: Bearer JWT required
  - Streaming: None (starts background task)
  - Triggers full forensic pipeline asynchronously

- **GET `/analysis/{analysis_id}`**
  - Request: None
  - Response: Analysis result (same as POST response)
  - Auth: Bearer JWT required
  - Retrieves stored analysis

- **GET `/analysis/`**
  - Request: query params (`skip`, `limit`, `media_id`)
  - Response: paginated list of analyses
  - Auth: Bearer JWT required
  - Supports filtering by media_id

- **GET `/analysis/{analysis_id}/events`**
  - Request: None
  - Response: Server-Sent Events stream of pipeline events
  - Auth: Bearer JWT required (or token query param)
  - Streaming: Yes (Server-Sent Events)

- **GET `/analysis/{id}/heatmap`**
  - Request: None
  - Response: PNG image or base64-encoded manipulation mask
  - Auth: Bearer JWT required

- **GET `/analysis/{id}/regions`**
  - Request: None
  - Response: JSON array of detection regions with boxes/polygons and scores
  - Auth: Bearer JWT required

- **GET `/analysis/{id}/timeline`**
  - Request: None
  - Response: JSON object with frame-level scores for video analysis
  - Auth: Bearer JWT required

- **GET `/analysis/{id}/graph`**
  - Request: None
  - Response: JSON object with provenance nodes and edges
  - Auth: Bearer JWT required

### Detection Router (`backend/app/api/v1/endpoints/detection.py`)
- **POST `/detection/analyze`**
  - Request: `{media_id}`
  - Response: Detection results only (no similarity/provenance)
  - Auth: Bearer JWT required
  - Streaming: None
  - Runs detection engines directly (bypasses full pipeline)

- **GET `/detection/{analysis_id}`**
  - Request: None
  - Response: Detection results for analysis
  - Auth: Bearer JWT required

### Investigation Router (`backend/app/api/v1/endpoints/investigation.py`)
- **POST `/investigation/fingerprint/generate`**
  - Request: `{media_id}`
  - Response: `{fingerprint_evidence}`
  - Auth: Bearer JWT required
  - (Re)generates perceptual hashes for media

- **POST `/investigation/similarity/search`**
  - Request: `{media_id, include_external}`
  - Response: `{local_matches, external_matches, provider_status, total_matches}`
  - Auth: Bearer JWT required
  - Triggers similarity search (local first, then external if configured)

- **POST `/investigation/source/trace`**
  - Request: `{media_id}`
  - Response: `{earliest_known_occurrence, provider_status}`
  - Auth: Bearer JWT required
  - Traces to earliest known occurrence via internet search

- **GET `/investigation/source/{source_id}`**
  - Request: None
  - Response: Source tracing details
  - Auth: Bearer JWT required

- **GET `/investigation/provenance/{provenance_id}`**
  - Request: None
  - Response: Provenance graph (nodes and edges)
  - Auth: Bearer JWT required

- **GET `/investigation/evidence/{evidence_id}`**
  - Request: None
  - Response: Evidence item details
  - Auth: Bearer JWT required

- **GET `/investigation/report/{report_id}`**
  - Request: query param (`format`: json|html|pdf, default: json)
  - Response: Report in requested format
  - Auth: Bearer JWT required
  - Streaming: None for JSON/HTML, Yes for PDF generation
  - Generates forensic report from evidence package

### Jobs Router (`backend/app/api/v1/endpoints/jobs.py`)
- **GET `/jobs/{job_id}`**
  - Request: None
  - Response: `{job_id, status, progress, result, error, events}`
  - Auth: Bearer JWT required
  - Polling endpoint for async job status (returns same data as SSE for fallback)

- **GET `/jobs/`**
  - Request: query params (`skip`, `limit`, `status`)
  - Response: paginated list of jobs
  - Auth: Bearer JWT required
  - Lists background jobs

### Health Check (`backend/app/main.py`)
- **GET `/health`**
  - Request: None
  - Response: `{status, version, models_loaded, uptime}`
  - Auth: None (public)
  - Liveness/readiness check
  - Shows which detection models are loaded/available

## 3. Internal Stages and UI-Visualizable Components

The backend implements a 9-stage synchronous pipeline in `backend/app/services/orchestration/analysis_pipeline.py`:

### Stage 1: `_run_ingestion` (`backend/app/services/ingestion/media_ingestion.py`)
- **UI-visualizable**: Upload progress, file validation results
- **Data**: File metadata, SHA-256, perceptual hashes, extracted metadata
- **Visualization**: File info panel, hash display, metadata viewer

### Stage 2: `_run_fingerprinting` (`backend/app/services/fingerprinting/`)
- **UI-visualizable**: Hash generation progress
- **Data**: pHash, dHash values
- **Visualization**: Hash display, similarity search readiness indicator

### Stage 3: `_run_detection` (`backend/app/services/detection/`)
- **UI-visualizable**: Real-time detection progress, detector status
- **Data**: Per-detector confidence scores, assessment, localization regions
- **Visualization**: 
  - Detector status cards (running/complete/failed)
  - Confidence gauges per detector
  - Heatmap overlays for manipulation masks
  - Bounding box visualization for detection regions
  - Face region analysis results
  - Copy-move match visualization

### Stage 4: `_run_similarity` (`backend/app/services/similarity/similarity_engine.py`)
- **UI-visualizable**: Search progress, match quality indicators
- **Data**: Ranked similarity matches with scores and methods
- **Visualization**: 
  - Similarity matrix/thumbnails
  - Match confidence bars
  - Method icons (SHA-256, pHash, dHash, embedding, internet)
  - External source URLs with preview

### Stage 5: `_run_source_tracing` (`backend/app/services/similarity/internet_search_provider.py`)
- **UI-visualizable**: Search progress, discovery timeline
- **Data**: Earliest known occurrence URL, timestamp, similarity
- **Visualization**: 
  - Timeline view of occurrences
  - Map view if geolocation available
  - Source credibility indicators

### Stage 6: `_run_provenance` (`backend/app/services/provenance/source_tracer.py`)
- **UI-visualizable**: Graph construction progress
- **Data**: Nodes (media items), Edges = relationships (derivative, copy, etc.)
- **Visualization**: 
  - Interactive force-directed graph
  - Node sizing by confidence/importance
  - Edge labeling by relationship type
  - Confidence-based edge styling

### Stage 7: `_run_evidence_classification` (`backend/app/services/evidence/evidence_service.py`)
- **UI-visualizable**: Classification progress
- **Data**: Evidence items categorized by FACT/OBSERVATION/INFERENCE/UNCERTAINTY
- **Visualization**: 
  - Tabbed evidence explorer by category
  - Color-coded evidence tags
  - Expandable evidence details with source tracking

### Stage 8: `_run_report_generation` (`backend/app/services/evidence/report_generator.py`)
- **UI-visualizable**: Report generation progress
- **Data**: Complete evidence package
- **Visualization**: 
  - Report preview (Markdown/HTML)
  - Download buttons for formats
  - Section-by-section report generation status

### Stage 9: `_run_persistence` (`backend/app/db/repository.py`)
- **UI-visualizable**: Save progress indicator
- **Data**: All analysis results persisted to database
- **Visualization**: 
  - Success/error indicators
  - "Analysis complete" notification
  - Refresh UI with final results

**Real-time UI Capabilities:**
- Detection stage progress (per-detector completion)
- Similarity search progress (local/external phases)
- Frame-by-frame video analysis (showing processed frames)
- Optical flow visualization (motion vectors)
- Spatial-temporal tracking (region persistence across frames)
- Provenance graph animation (node/edge addition)
- Evidence classification streaming (items appearing by category)

## 4. Data Worth Visualizing

### Forensic Evidence Data
1. **Hash Values & Metadata**
   - SHA-256 cryptographic hash (data integrity)
   - pHash/dHash perceptual hashes (similarity basis)
   - File metadata (dimensions, duration, codec, EXIF)
   - *Visualization*: Hash display panels, metadata tables

2. **Detection Results**
   - Per-detector confidence scores (0-100%)
   - Binary manipulation detection flags
   - Localization regions (bounding boxes, polygons)
   - Assessment levels (SUSPICIOUS/CLEAN/INCONCLUSIVE)
   - *Visualization*: 
     - Overlay heatmaps on media
     - Bounding box annotations
     - Detector confidence gauges
     - Region timelines (for video)

3. **Similarity & Provenance**
   - Similarity match scores (0-100%)
   - Match methods (exact, perceptual, embedding, internet)
   - Provenance graph nodes/edges with confidence
   - Earliest known occurrence data
   - *Visualization*:
     - Similarity thumbnails with score overlays
     - Network graph visualization
     - Timeline of media appearances
     - Geographic distribution maps

4. **Temporal Analysis (Video)**
   - Frame-level anomaly scores over time
   - Suspicious segments (start/end timestamps)
   - Optical flow magnitude fields
   - Tracked region persistence across frames
   - *Visualization*:
     - Anomaly score timeline/heatmap
     - Segment highlighting on timeline
     - Flow field visualization
     - Region tracking trails

5. **Evidence Classification**
   - FACT: Cryptographic proofs, file properties
   - OBSERVATION: Algorithmic signals (ELA, FFT, noise)
   - INFERENCE: Provenance relationships, source estimates
   - UNCERTAINTY: Limitations, conflicting signals
   - *Visualization*:
     - Categorized evidence panels
     - Evidence strength meters
     - Conflict resolution views
     - Limitations disclosure

6. **Final Assessment**
   - Composite confidence score
   - Verdict (SUSPICIOUS/CLEAN/INCONCLUSIVE)
   - Dominant signal detector
   - Overall manipulation type
   - *Visualization*:
     - Verdict badge with confidence
     - Gauge/meter visualization
     - Signal source attribution
     - Recommendation engine

## 5. Error Cases, Performance, Security, and CORS

### Error Handling Patterns
- **File Validation Errors** (`backend/app/services/ingestion/media_ingestion.py`)
  - Invalid file types: 400 Bad Request with supported types list
  - File size exceeded: 413 Payload Too Large (500MB limit)
  - Corrupted/unreadable files: Specific exceptions with file path info

- **Database Errors** (`backend/app/db/repository.py`)
  - Constraint violations: 409 Conflict (duplicates, FK violations)
  - Connection failures: 503 Service Unavailable with retry-after
  - Transaction rollbacks: Automatic on exception with logging

- **Detection Engine Errors** (various detector files)
  - Model loading failures: Graceful degradation to INCONCLUSIVE
  - Processing exceptions: Caught, logged, return default results
  - Resource exhaustion: Timeout handling with partial results

- **External Service Errors** (`backend/app/services/similarity/internet_search_provider.py`)
  - API failures: Fallback to local-only search
  - Rate limiting: Exponential backoff, cached results
  - Network timeouts: Configurable timeouts with retry logic

### Performance Characteristics & Bottlenecks
- **CPU-Bound Operations**:
  - TruForDetector CNN inference (GPU acceleration when available)
  - Optical flow computation (OpenCV Farnebäck)
  - Perceptual hash generation (block-based DCT)
  - Feature detection (SIFT/ORB in copy-move)
  - *Mitigation*: Async processing, model caching, GPU utilization

- **I/O-Bound Operations**:
  - File reading/writing (uploads/processed directories)
  - Database operations (SQLite default, PostgreSQL option)
  - External API calls (search providers)
  - *Mitigation*: Connection pooling, async file ops, caching

- **Memory-Intensive Operations**:
  - Video frame buffering (limited by MAX_ANALYSIS_FRAMES=120)
  - Embedding vector storage (CLIP features)
  - Similarity search indices
  - *Mitigation*: Frame sampling, streaming processing, LRU caches

- **Timeout Controls** (`backend/app/core/config.py`):
  - `processing_timeout_seconds`: 600s (10 min) global limit
  - `max_video_duration_seconds`: 600s (10 min) input limit
  - `max_audio_duration_seconds`: 600s (10 min) input limit
  - Frame analysis capped at MAX_ANALYSIS_FRAMES=120

### Authentication & Authorization
- **JWT Authentication** (`backend/app/core/auth.py`):
  - Algorithm: HS256 (symmetric)
  - Expiry: 24 hours (configurable via SECRET_KEY)
  - Token endpoint: POST `/auth/login`
  - Refresh mechanism: Not implemented (short-lived tokens)
  - Password hashing: bcrypt with salt
  - Demo users: Pre-configured in DEMO_USERS_DB for testing

- **Authorization Model**:
  - All endpoints except `/auth/*` and `/health` require Bearer JWT
  - No role-based access control (all authenticated users equal)
  - Resource ownership: Implicit via media_id/analysis_id scoping
  - No admin/user distinction in current implementation

### CORS Configuration
- **CORS Middleware** (`backend/app/main.py`):
  - Configured origins: 
    - http://localhost:5173 (Vite dev server)
    - http://localhost:3000 (React dev server)
    - http://localhost:8080 (alternative dev)
    - http://localhost:3001 (additional frontend dev)
  - Methods: GET, POST, PUT, DELETE, OPTIONS
  - Headers: Authorization, Content-Type, Accept
  - Credentials: Not enabled (stateless JWT)
  - Exposed headers: Custom headers if needed
  - SSE Support: Authorization header works with EventSource API

### Rate Limiting & Security
- **SlowAPI Integration** (`backend/app/main.py`):
  - Global rate limiting: 100 requests/minute per IP
  - Per-endpoint limits: Customizable via decorators
  - Storage: In-memory (Redis option for production)
  - Exceeded limit: 429 Too Many Requests with retry-after

- **Security Headers** (missing - potential enhancement):
  - No security middleware currently implemented
  - Recommendation: Add helmet.js equivalent for:
    - XSS protection
    - Content Security Policy
    - HSTS (when HTTPS deployed)
    - Referrer policy

### Deployment & Configuration
- **Environment Configuration** (`backend/app/core/config.py`):
  - SQLite default: `aimd.db` in project root
  - PostgreSQL option: Set `DATABASE_URL` env var
  - Model paths: Configurable via env vars (optional models)
  - Device selection: auto/cpu/cuda for torch models
  - External API keys: Google Vision, TinEye (optional)

- **Directory Structure**:
  - `uploads/` - Original uploaded files
  - `processed/` - Frames, keyframes, intermediate data
  - `reports/` - Generated forensic reports
  - `logs/` - Application logs
  - Model directories: Configurable paths

## 6. Missing API Features for Frontend Development

### Real-Time Communication
- **Missing**: WebSocket endpoints for live progress updates
- **Needed**: 
  - `/ws/analysis/{analysis_id}` - Real-time pipeline stage updates
  - `/ws/detection/{media_id}` - Live detector results streaming
  - `/ws/similarity/{media_id}` - Search progress with match updates
- **Implementation**: Use FastAPI WebSocket support with JSON messages

### Job Management & Async Processing
- **Missing**: Full asynchronous job queue system
- **Current**: Synchronous pipeline blocks HTTP request (with async option now available)
- **Needed**:
  - POST `/analysis/async` - Returns job_id immediately (NOW IMPLEMENTED)
  - Enhanced `/jobs/{job_id}` with detailed progress (NOW IMPLEMENTED)
  - WebSocket job progress channels
  - Job cancellation endpoint: DELETE `/jobs/{job_id}`
- **Integration**: Celery/RQ with Redis backend for task queuing

### Advanced Filtering & Search
- **Missing**: Sophisticated query capabilities
- **Needed**:
  - GET `/analysis/?verdict=SUSPICIOUS&confidence_min=0.8`
  - GET `/media/?has_detections=true&media_type=video`
  - GET `/investigation/evidence?category=OBSERVATION&detector=TruFor`
  - Range queries on timestamps, confidence scores
- **Implementation**: SQLAlchemy filtering with OpenAPI schema generation

### Export & Bulk Operations
- **Missing**: Bulk export and batch processing
- **Needed**:
  - POST `/analysis/batch` - Analyze multiple media items
  - GET `/exports/evidence/{analysis_id}` - ZIP with all evidence
  - GET `/exports/report/{analysis_id}` - Multiple formats bundle
  - POST `/media/batch-upload` - Multiple file upload with metadata
- **Implementation**: Background job processing for bulk operations

### Collaboration & Annotation
- **Missing**: User collaboration features
- **Needed**:
  - POST `/analysis/{id}/annotations` - User comments/tags
  - GET `/analysis/{id}/annotations` - Retrieve annotations
  - POST `/analysis/{id}/bookmark` - User bookmarks/favorites
  - GET `/users/{id}/activity` - User action history
- **Implementation**: Annotation tables linked to analyses/users

### Advanced Visualization Data
- **Missing**: Specialized endpoints for visualization
- **Needed**: 
  - GET `/analysis/{id}/heatmap` - PNG or base64 mask (NOW IMPLEMENTED)
  - GET `/analysis/{id}/regions` - boxes/polygons with scores (NOW IMPLEMENTED)
  - GET `/analysis/{id}/timeline` - frame-level scores for video (NOW IMPLEMENTED)
  - GET `/analysis/{id}/graph` - provenance nodes and edges (NOW IMPLEMENTED)
- **Implementation**: Pre-computed views optimized for frontend consumption

### Health & Monitoring
- **Missing**: Detailed system metrics
- **Needed**:
  - GET `/metrics` - Prometheus-format metrics
  - GET `/status/detailed` - Component health (DB, models, external)
  - GET `/logs/recent` - Structured log streaming
  - GET `/usage/stats` - API usage statistics
- **Implementation**: Prometheus client, structured logging, log aggregation

### Configuration & Feature Flags
- **Missing**: Runtime configuration access
- **Needed**:
  - GET `/config/public` - Non-sensitive config (version, features)
  - POST `/config/features` - Toggle experimental features (admin)
  - GET `/models/status` - Detailed model availability/performance
- **Implementation**: Feature flag system with admin controls

### Internationalization & Localization
- **Missing**: i18n support
- **Needed**:
  - Accept-Language header processing
  - Localized error messages and validation
  - Date/number formatting per locale
  - *Note*: Current system uses English-only messages

### API Versioning
- **Missing**: Explicit API versioning
- **Needed**:
  - `/api/v2/*` endpoints for breaking changes
  - Version negotiation via headers
  - Deprecation warnings for v1 endpoints
- **Implementation**: Version prefix in URL or Accept header

### Documentation & Discovery
- **Missing**: Interactive API documentation enhancement
- **Current**: Basic OpenAPI/Swagger at `/docs`
- **Needed**:
  - Example requests/responses for all endpoints
  - Interactive Try-It-Out with authentication
  - SDK generation clients (Python, JS/TS)
  - Postman collection export
- **Implementation**: Enhanced OpenAPI schema with examples

These missing features would significantly enhance the frontend development experience by providing real-time updates, better performance through async processing, advanced data filtering, and specialized endpoints optimized for visualization components.

## 7. Observable Analysis Pipeline Implementation

### New Features Added (as requested in the special requirements):

#### POST /analysis/async
- Creates a background job that runs the AnalysisPipeline
- Returns immediately with `{job_id, analysis_id}`
- Uses FastAPI BackgroundTasks for non-blocking execution
- Job status can be polled via `/jobs/{job_id}` or streamed via SSE

#### Pipeline Event Emission
- Modified `AnalysisPipeline.run()` to emit events after each stage
- Events include: `{stage, status, progress, message, partial_result, timestamp}`
- Per-detector events in stage 3 (detection) for each detector
- Events stored in database table `pipeline_events`
- Event statuses: pending → running → done/failed

#### GET /analysis/{analysis_id}/events (Server-Sent Events)
- Returns stream of pipeline events as Server-Sent Events
- Format: `event: stage_update\n data: {json}\n\n`
- Supports Authorization header for authentication
- Fallback query param token for proxies that strip headers
- Proper Content-Type: `text/event-stream`
- Cache-Control: `no-cache`
- Connection: `keep-alive`

#### GET /jobs/{job_id} (Enhanced)
- Returns job status including progress and recent events
- Same data structure as SSE for polling fallback
- Includes: job_id, analysis_id, status, progress, result, events array
- Events array contains last 50 events for the job

#### Visualization Endpoints
- **Heatmap**: Returns manipulation mask as PNG or base64
- **Regions**: Returns detection regions as JSON with bounding boxes/polygons
- **Timeline**: Returns frame-level anomaly scores over time (video only)
- **Graph**: Returns provenance graph as JSON nodes and edges

#### CORS Updates
- Added common frontend dev origins to allowed list
- Configured CORS to allow Authorization header for SSE
- Preflight OPTIONS requests handled properly

All changes maintain backward compatibility - existing synchronous endpoint works unchanged.