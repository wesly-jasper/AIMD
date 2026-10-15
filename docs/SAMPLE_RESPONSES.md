# Sample API Responses for AIMD Forensic Analysis Platform

This document provides example responses for key API endpoints to help frontend developers understand the data structures they'll be working with.

## 1. Authentication Endpoints

### Login Request
**POST /api/v1/auth/login**

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyMW ExMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
  "token_type": "bearer",
  "expires_in": 86400,
  "user": {
    "id": "user1234567890",
    "email": "user@example.com",
    "full_name": "John Doe",
    "is_active": true,
    "role": "analyst"
  }
}
```

### Login Error Response
```json
{
  "detail": "Incorrect email or password"
}
```

## 2. Media Management Endpoints

### Media Upload Response
**POST /api/v1/media/**

```json
{
  "media_id": "media_550e8400-e29b-41d4-a716-446655440000",
  "filename": "evidence_photo.jpg",
  "content_type": "image/jpeg",
  "size": 2457600,
  "uploaded_at": "2026-10-13T14:30:00Z",
  "storage_path": "/uploads/2026/10/13/media_550e8400-e29b-41d4-a716-446655440000.jpg",
  "metadata": {
    "width": 1920,
    "height": 1080,
    "format": "JPEG",
    "mode": "RGB",
    "has_exif": true
  }
}
```

### Media List Response (Paginated)
**GET /api/v1/media/**

```json
{
  "items": [
    {
      "media_id": "media_550e8400-e29b-41d4-a716-446655440000",
      "filename": "evidence_photo.jpg",
      "content_type": "image/jpeg",
      "size": 2457600,
      "uploaded_at": "2026-10-13T14:30:00Z",
      "metadata": {
        "width": 1920,
        "height": 1080,
        "format": "JPEG"
      }
    },
    {
      "media_id": "media_660e8400-e29b-41d4-a716-446655440001",
      "filename": "suspicious_video.mp4",
      "content_type": "video/mp4",
      "size": 45000000,
      "uploaded_at": "2026-10-13T13:15:00Z",
      "metadata": {
        "width": 1280,
        "height": 720,
        "duration": 30.5,
        "format": "MP4"
      }
    }
  ],
  "total": 2,
  "page": 1,
  "per_page": 10,
  "pages": 1
}
```

### Media Detail Response
**GET /api/v1/media/{media_id}**

```json
{
  "media_id": "media_550e8400-e29b-41d4-a716-446655440000",
  "filename": "evidence_photo.jpg",
  "content_type": "image/jpeg",
  "size": 2457600,
  "uploaded_at": "2026-10-13T14:30:00Z",
  "storage_path": "/uploads/2026/10/13/media_550e8400-e29b-41d4-a716-446655440000.jpg",
  "metadata": {
    "width": 1920,
    "height": 1080,
    "format": "JPEG",
    "mode": "RGB",
    "has_exif": true,
    "exif_data": {
      "Make": "Canon",
      "Model": "EOS R5",
      "DateTime": "2026:10:13 14:29:45",
      "GPSLatitude": "40.7128 N",
      "GPSLongitude": "74.0060 W"
    }
  }
}
```

## 3. Analysis Endpoints

### Synchronous Analysis Response
**POST /api/v1/analysis/**

```json
{
  "analysis_id": "analysis_770e8400-e29b-41d4-a716-446655440000",
  "media_id": "media_550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "created_at": "2026-10-13T14:30:05Z",
  "completed_at": "2026-10-13T14:30:15Z",
  "manipulation_detected": true,
  "confidence": 0.87,
  "manipulation_type": "face",
  "assessment": "SUSPICIOUS",
  "regions": [
    {
      "type": "face_region",
      "confidence": 0.92,
      "bbox": [450.5, 200.0, 180.0, 220.0],
      "extra": {
        "analysis_types": ["boundary", "frequency", "texture"]
      }
    }
  ],
  "metadata": {
    "processing_time_ms": 1247,
    "stages_completed": [
      "metadata_extraction",
      "frame_extraction",
      "detection",
      "fingerprinting",
      "similarity_search",
      "source_tracing",
      "provenance_building",
      "evidence_categorization",
      "report_generation"
    ]
  }
}
```

### Asynchronous Analysis Job Submission
**POST /api/v1/analysis/async**

```json
{
  "job_id": "job_880e8400-e29b-41d4-a716-446655440000",
  "analysis_id": "analysis_770e8400-e29b-41d4-a716-446655440000",
  "status": "submitted",
  "submitted_at": "2026-10-13T14:30:05Z",
  "estimated_completion": "2026-10-13T14:35:00Z"
}
```

### Job Status Response (Polling)
**GET /api/v1/jobs/{job_id}**

```json
{
  "job_id": "job_880e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "progress": 60,
  "current_stage": "similarity_search",
  "stages_completed": [
    "metadata_extraction",
    "frame_extraction",
    "detection",
    "fingerprinting"
  ],
  "stages_remaining": [
    "similarity_search",
    "source_tracing",
    "provenance_building",
    "evidence_categorization",
    "report_generation"
  ],
  "analysis_id": "analysis_770e8400-e29b-41d4-a716-446655440000",
  "media_id": "media_550e8400-e29b-41d4-a716-446655440000",
  "submitted_at": "2026-10-13T14:30:05Z",
  "updated_at": "2026-10-13T14:32:30Z",
  "estimated_completion": "2026-10-13T14:35:00Z"
}
```

### Job Completed Response
**GET /api/v1/jobs/{job_id}** (when completed)

```json
{
  "job_id": "job_880e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "progress": 100,
  "current_stage": "report_generation",
  "stages_completed": [
    "metadata_extraction",
    "frame_extraction",
    "detection",
    "fingerprinting",
    "similarity_search",
    "source_tracing",
    "provenance_building",
    "evidence_categorization",
    "report_generation"
  ],
  "stages_remaining": [],
  "analysis_id": "analysis_770e8400-e29b-41d4-a716-446655440000",
  "media_id": "media_550e8400-e29b-41d4-a716-446655440000",
  "submitted_at": "2026-10-13T14:30:05Z",
  "completed_at": "2026-10-13T14:34:15Z",
  "updated_at": "2026-10-13T14:34:15Z"
}
```

## 4. Real-time Events (Server-Sent Events)

### Events Stream Format
**GET /api/v1/analysis/{analysis_id}/events**

Each event is sent as:
```
data: {"event_type": "pipeline_stage_completed", "stage": "detection", "timestamp": "2026-10-13T14:30:20Z", "data": {...}}

data: {"event_type": "detector_completed", "detector": "BaselineFaceDetector", "confidence": 0.75, "regions_count": 1, "timestamp": "2026-10-13T14:30:25Z"}

data: {"event_type": "pipeline_stage_completed", "stage": "fingerprinting", "timestamp": "2026-10-13T14:30:30Z", "data": {...}}

data: {"event_type": "analysis_completed", "analysis_id": "analysis_770e8400-e29b-41d4-a716-446655440000", "status": "completed", "timestamp": "2026-10-13T14:34:15Z"}
```

## 5. Visualization Endpoints

### Heatmap Response
**GET /api/v1/analysis/{analysis_id}/heatmap**

```json
{
  "heatmap": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
  "content_type": "image/png",
  "width": 1920,
  "height": 1080,
  "analysis_id": "analysis_770e8400-e29b-41d4-a716-446655440000",
  "generated_at": "2026-10-13T14:34:10Z"
}
```

### Regions Response
**GET /api/v1/analysis/{analysis_id}/regions**

```json
{
  "analysis_id": "analysis_770e8400-e29b-41d4-a716-446655440000",
  "regions": [
    {
      "type": "face_region",
      "confidence": 0.92,
      "bbox": [450.5, 200.0, 180.0, 220.0],
      "extra": {
        "analysis_types": ["boundary", "frequency", "texture"],
        "boundary_gradient_variance": 2.45,
        "frequency_spectrum_anomaly": 1.8,
        "texture_lbp_uniformity": 0.3
      }
    },
    {
      "type": "eye_region",
      "confidence": 0.78,
      "bbox": [480.0, 220.0, 40.0, 30.0],
      "extra": {
        "analysis_types": ["eye_specific"],
        "reflection_consistency": 0.82,
        "circularity_score": 0.75
      }
    }
  ],
  "total_regions": 2,
  "analysis_id": "analysis_770e8400-e29b-41d4-a716-446655440000"
}
```

### Timeline Response (Video Analysis)
**GET /api/v1/analysis/{analysis_id}/timeline**

```json
{
  "analysis_id": "analysis_990e8400-e29b-41d4-a716-446655440000",
  "media_id": "media_660e8400-e29b-41d4-a716-446655440001",
  "media_type": "video",
  "duration": 30.5,
  "fps": 25.0,
  "total_frames": 762,
  "frame_scores": [
    {
      "frame_number": 0,
      "timestamp": 0.0,
      "manipulation_detected": false,
      "confidence": 0.05,
      "regions_count": 0
    },
    {
      "frame_number": 25,
      "timestamp": 1.0,
      "manipulation_detected": true,
      "confidence": 0.65,
      "regions_count": 2,
      "regions": [
        {
          "type": "face_region",
          "confidence": 0.65,
          "bbox": [300.5, 150.0, 200.0, 250.0]
        }
      ]
    },
    {
      "frame_number": 50,
      "timestamp": 2.0,
      "manipulation_detected": true,
      "confidence": 0.82,
      "regions_count": 3
    }
    // ... more frames
  ],
  "summary": {
    "frames_analyzed": 762,
    "frames_with_manipulation": 420,
    "max_confidence": 0.91,
    "average_confidence": 0.38
  }
}
```

### Provenance Graph Response
**GET /api/v1/analysis/{analysis_id}/graph**

```json
{
  "analysis_id": "analysis_770e8400-e29b-41d4-a716-446655440000",
  "nodes": [
    {
      "id": "prov_001",
      "type": "source",
      "label": "Original Source",
      "confidence": 0.95,
      "properties": {
        "url": "https://example.com/original.jpg",
        "retrieved_at": "2026-10-13T14:25:00Z",
        "content_type": "image/jpeg"
      }
    },
    {
      "id": "prov_002",
      "type": "transformation",
      "label": "Face Swap Detected",
      "confidence": 0.87,
      "properties": {
        "technique": "deepfake",
        "model_estimate": "DFake_v2.1",
        "transformation_time": "2026-10-12T03:15:00Z"
      }
    },
    {
      "id": "prov_003",
      "type": "artifact",
      "label": "Analyzed Media",
      "confidence": 0.90,
      "properties": {
        "filename": "evidence_photo.jpg",
        "analysis_complete": true
      }
    }
  ],
  "edges": [
    {
      "id": "edge_001",
      "source": "prov_001",
      "target": "prov_002",
      "label": "derived_from",
      "confidence": 0.88
    },
    {
      "id": "edge_002",
      "source": "prov_002",
      "target": "prov_003",
      "label": "derived_from",
      "confidence": 0.85
    }
  ],
  "graph_id": "prov_graph_770e8400-e29b-41d4-a716-446655440000",
  "generated_at": "2026-10-13T14:34:12Z"
}
```

## 6. Similarity Search Endpoints

### Similarity Search Response
**GET /api/v1/similarity/search/{media_id}**

```json
{
  "media_id": "media_550e8400-e29b-41d4-a716-446655440000",
  "search_type": "hybrid",
  "query_hash": "a1b2c3d4e5f6789012345678901234567890abcd",
  "matches": [
    {
      "media_id": "media_110e8400-e29b-41d4-a716-446655440002",
      "filename": "similar_photo_1.jpg",
      "similarity_score": 0.92,
      "match_type": "perceptual_hash",
      "hash_distance": 5,
      "thumbnail_url": "/thumbnails/media_110e8400-e29b-41d4-a716-446655440002.jpg",
      "first_seen": "2026-09-15T08:30:00Z",
      "source_platform": "twitter"
    },
    {
      "media_id": "media_220e8400-e29b-41d4-a716-446655440003",
      "filename": "similar_photo_2.jpg",
      "similarity_score": 0.78,
      "match_type": "clip_embedding",
      "embedding_distance": 0.23,
      "thumbnail_url": "/thumbnails/media_220e8400-e29b-41d4-a716-446655440003.jpg",
      "first_seen": "2026-09-20T14:15:00Z",
      "source_platform": "facebook"
    },
    {
      "media_id": "media_330e8400-e29b-41d4-a716-446655440004",
      "filename": "potential_original.jpg",
      "similarity_score": 0.89,
      "match_type": "hybrid",
      "hash_distance": 8,
      "embedding_distance": 0.18,
      "combined_score": 0.89,
      "thumbnail_url": "/thumbnails/media_330e8400-e29b-41d4-a716-446655440004.jpg",
      "first_seen": "2026-09-10T12:00:00Z",
      "source_platform": "instagram",
      "note": "High similarity suggests this may be the source material"
    }
  ],
  "total_matches": 3,
  "search_completed_at": "2026-10-13T14:31:45Z",
  "search_duration_ms": 1240
}
```

## 7. Evidence Endpoints

### Evidence List Response
**GET /api/v1/evidence/{analysis_id}**

```json
{
  "analysis_id": "analysis_770e8400-e29b-41d4-a716-446655440000",
  "evidence_items": [
    {
      "evidence_id": "evid_001",
      "category": "FACT",
      "description": "Media file is a JPEG image with dimensions 1920x1080 pixels",
      "confidence": 0.99,
      "data_type": "metadata",
      "location": {
        "bbox": null,
        "timestamp": null
      },
      "timestamp": "2026-10-13T14:30:07Z"
    },
    {
      "evidence_id": "evid_002",
      "category": "OBSERVATION",
      "description": "Face region shows inconsistent boundary gradient analysis",
      "confidence": 0.87,
      "data_type": "detection",
      "location": {
        "bbox": [450.5, 200.0, 180.0, 220.0],
        "timestamp": null
      },
      "timestamp": "2026-10-13T14:30:22Z",
      "related_regions": [
        {
          "type": "face_region",
          "confidence": 0.92,
          "bbox": [450.5, 200.0, 180.0, 220.0]
        }
      ]
    },
    {
      "evidence_id": "evid_003",
      "category": "INFERENCE",
      "description": "Likely face swap deepfake based on boundary artifacts and frequency analysis",
      "confidence": 0.76,
      "data_type": "inference",
      "location": {
        "bbox": [450.5, 200.0, 180.0, 220.0],
        "timestamp": null
      },
      "timestamp": "2026-10-13T14:30:25Z",
      "reasoning": [
        "Boundary gradient variance exceeds normal thresholds",
        "Frequency spectrum shows unnatural patterns consistent with GAN generation",
        "Texture analysis indicates synthetic skin properties"
      ]
    },
    {
      "evidence_id": "evid_004",
      "category": "UNCERTAINTY",
      "description": "Eye region analysis inconclusive due to low image resolution in that area",
      "confidence": 0.45,
      "data_type": "detection",
      "location": {
        "bbox": [480.0, 220.0, 40.0, 30.0],
        "timestamp": null
      },
      "timestamp": "2026-10-13T14:30:28Z"
    }
  ],
  "summary": {
    "total_evidence": 4,
    "by_category": {
      "FACT": 1,
      "OBSERVATION": 1,
      "INFERENCE": 1,
      "UNCERTAINTY": 1
    },
    "highest_confidence": 0.99,
    "assessment_override": "SUSPICIOUS"
  }
}
```

## 8. Report Generation Endpoints

### Report Generation Response
**POST /api/v1/reports/generate**

```json
{
  "report_id": "report_440e8400-e29b-41d4-a716-446655440000",
  "analysis_id": "analysis_770e8400-e29b-41d4-a716-446655440000",
  "format": "pdf",
  "status": "completed",
  "generated_at": "2026-10-13T14:35:00Z",
  "download_url": "/downloads/reports/report_440e8400-e29b-41d4-a716-446655440000.pdf",
  "expires_at": "2026-10-20T14:35:00Z",
  "file_size": 245760,
  "page_count": 8,
  "checksum": "sha256:a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456"
}
```

### Report Generation Accepted (Async Processing)
```json
{
  "report_id": "report_550e8400-e29b-41d4-a716-446655440000",
  "analysis_id": "analysis_770e8400-e29b-41d4-a716-446655440000",
  "format": "pdf",
  "status": "processing",
  "accepted_at": "2026-10-13T14:35:00Z",
  "estimated_completion": "2026-10-13T14:36:00Z",
  "message": "Report generation started. Use GET /api/v1/reports/{report_id} to check status."
}
```

## 9. Error Responses

### Failed Analysis Response
**POST /api/v1/analysis/** (when analysis fails)

```json
{
  "analysis_id": "analysis_660e8400-e29b-41d4-a716-446655440000",
  "media_id": "media_550e8400-e29b-41d4-a716-446655440000",
  "status": "failed",
  "created_at": "2026-10-13T14:30:05Z",
  "failed_at": "2026-10-13T14:30:08Z",
  "manipulation_detected": false,
  "confidence": 0.0,
  "manipulation_type": "unknown",
  "assessment": "INCONCLUSIVE",
  "regions": [],
  "metadata": {
    "error": "Failed to load media file: corrupted or unsupported format",
    "error_code": "MEDIA_LOAD_ERROR",
    "stages_completed": [
      "metadata_extraction"
    ],
    "failed_at_stage": "frame_extraction"
  }
}
```

### Validation Error Response
**POST /api/v1/media/** (invalid file type)

```json
{
  "detail": "Invalid file type. Supported types: JPEG, PNG, BMP, TIFF, MP4, AVI, MOV",
  "error_code": "INVALID_FILE_TYPE",
  "supported_types": ["image/jpeg", "image/png", "image/bmp", "image/tiff", "video/mp4", "video/x-msvideo", "video/quicktime"]
}
```

### Authentication Error Response
**POST /api/v1/auth/login** (invalid credentials)

```json
{
  "detail": "Incorrect email or password",
  "error_code": "INVALID_CREDENTIALS"
}
```

### Rate Limit Error Response
**Any endpoint** (when rate limit exceeded)

```json
{
  "detail": "Rate limit exceeded. Maximum 10 requests per minute.",
  "error_code": "RATE_LIMIT_EXCEEDED",
  "reset_at": "2026-10-13T14:35:00Z",
  "retry_after": 30
}
```

## Notes on Usage

1. **Timestamps**: All timestamps are in ISO 8601 format with timezone indicator (Z for UTC)
2. **Confidence Scores**: Range from 0.0 to 1.0, where higher values indicate higher confidence
3. **Bounding Boxes**: Format is [x, y, width, height] in pixel coordinates from top-left corner
4. **Analysis Categories**:
   - FACT: Verifiable, objective information
   - OBSERVATION: Directly observable characteristics
   - INFERENCE: Logical conclusions based on evidence
   - UNCERTAINTY: Information that is unclear or conflicting
5. **Manipulation Types**: Can include "face", "audio", "video", "deepfake", "splice", "copy_move", "unknown"
6. **Assessment Values**: "CLEAN", "SUSPICIOUS", "INCONCLUSIVE" (based on confidence thresholds and evidence)