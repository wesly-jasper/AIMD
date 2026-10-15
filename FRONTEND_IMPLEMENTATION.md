# Frontend Implementation for AIMD Forensic Analysis Platform

## Overview

This document describes the frontend implementation created to interact with the observable AIMD backend API. The implementation includes:

1. TypeScript type definitions for all API endpoints
2. React Query hooks for data fetching and mutations
3. Mock API service for development and testing
4. React components demonstrating usage
5. Styling and configuration files

## File Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── UploadModal.tsx
│   │   └── AnalysisViewer.tsx
│   ├── services/
│   │   ├── api.ts                 # React Query hooks
│   │   ├── mockApi.ts             # Mock API implementation
│   │   └── index.ts               # Export barrel
│   ├── types/
│   │   └── api.ts                 # TypeScript interfaces
│   ├── App.tsx                    # Main application component
│   ├── App.css                    # Styling
│   ├── index.tsx                  # Entry point
│   └── index.css                  # Base styles
├── .env                           # Environment variables
├── package.json                   # Dependencies and scripts
└── tsconfig.json                  # TypeScript configuration
```

## Key Features Implemented

### 1. TypeScript Types (`src/types/api.ts`)

Comprehensive type definitions for all backend endpoints:

- **Authentication**: RegisterRequest, LoginRequest, AuthResponse
- **Media Management**: MediaUploadResponse, MediaMetadata, MediaListResponse
- **Analysis**: AnalysisRequest, AnalysisResponse, AsyncAnalysisResponse, JobStatus
- **Pipeline Events**: PipelineEvent with stage-specific properties
- **Detection Results**: DetectionResult, DetectionRegion
- **Similarity Search**: SimilarityMatch, SimilaritySearchResponse
- **Provenance**: ProvenanceNode, ProvenanceEdge, ProvenanceGraph
- **Evidence**: EvidenceItem with FACT/OBSERVATION/INFERENCE/UNCERTAINTY categories
- **Reports**: ReportRequest, ReportResponse
- **Visualization**: HeatmapResponse, RegionsResponse, AnalysisTimeline, FrameScore

### 2. React Query API Client (`src/services/api.ts`)

Custom hooks for all backend endpoints using React Query:

#### Authentication Hooks
- `useRegister()` - User registration
- `useLogin()` - User login with token storage
- `useLogout()` - User logout and token removal

#### Media Hooks
- `useUploadMedia()` - File upload with FormData
- `useMediaList()` - Paginated media listing
- `useMediaDetail()` - Individual media metadata

#### Analysis Hooks
- `useAnalysis()` - Synchronous analysis endpoint
- `useAsyncAnalysis()` - Asynchronous analysis job submission

#### Job Management Hooks
- `useJobStatus()` - Polling-based job status checking (2s interval)

#### Real-time Events Hook
- `useAnalysisEvents()` - Hook for Server-Sent Events connection
  - Returns events array, connection status, and error state
  - Includes connect/disconnect control functions
  - Designed to work with backend SSE endpoint

#### Visualization Hooks
- `useHeatmap()` - Retrieves manipulation heatmap images
- `useRegions()` - Gets detected manipulation regions
- `useTimeline()` - Retrieves frame-by-frame analysis scores
- `useProvenanceGraph()` - Gets provenance graph data

#### Additional Service Hooks
- `useSimilaritySearch()` - Internet and local similarity search
- `useEvidenceList()` - Retrieves categorized evidence items
- `useGenerateReport()` - Generates analysis reports in various formats

#### Utility Hook
- `useAnalysisWorkflow()` - Combined workflow for upload and analysis

### 3. Mock API Service (`src/services/mockApi.ts`)

Complete mock implementation for development/testing:

- Realistic delays simulating network latency
- Progressive job state updates mimicking actual pipeline stages
- Mock data for all entity types (media, analyses, jobs, events, evidence)
- Simulated SSE-like event progression through polling
- Realistic job progression through all 9 pipeline stages:
  1. Metadata extraction
  2. Frame extraction
  3. Detection (with sub-detector events)
  4. Fingerprinting
  5. Similarity search
  6. Source tracing
  7. Provenance building
  8. Evidence categorization
  9. Report generation

### 4. React Components

#### UploadModal Component
- File upload interface with format validation
- Upload progress and status display
- Error handling and user feedback

#### AnalysisViewer Component
- Tabbed interface for different visualization types
- Heatmap display with base64 image rendering
- Regions list with confidence scores and bounding boxes
- Timeline visualization placeholder (would integrate charting library)
- Provenance graph display with nodes and edges
- Refresh functionality for updating visualizations

### 5. Development Setup

#### Package Dependencies
- React 18.x with TypeScript
- React Query 5.x for data synchronization
- jwt-decode for token handling
- React Scripts for CRA-based tooling

#### Environment Configuration
- `.env` file with `VITE_API_BASE_URL` pointing to backend
- Easy configuration for different environments

## API Endpoint Coverage

All backend endpoints described in BACKEND_ANALYSIS.md have corresponding frontend implementations:

### Authentication
- POST `/auth/register` → `useRegister()`
- POST `/auth/login` → `useLogin()`
- POST `/auth/logout` → `useLogout()`

### Media Management
- POST `/media/` → `useUploadMedia()`
- GET `/media/` → `useMediaList()`
- GET `/media/{media_id}` → `useMediaDetail()`

### Analysis (Synchronous)
- POST `/analysis/` → `useAnalysis()`

### Analysis (Asynchronous)
- POST `/analysis/async` → `useAsyncAnalysis()`
- GET `/jobs/{job_id}` → `useJobStatus()`
- GET `/analysis/{analysis_id}/events` → `useAnalysisEvents()` (SSE)

### Visualization Endpoints
- GET `/analysis/{analysis_id}/heatmap` → `useHeatmap()`
- GET `/analysis/{analysis_id}/regions` → `useRegions()`
- GET `/analysis/{analysis_id}/timeline` → `useTimeline()`
- GET `/analysis/{analysis_id}/graph` → `useProvenanceGraph()`

### Additional Endpoints
- GET `/similarity/search/{media_id}` → `useSimilaritySearch()`
- GET `/evidence/{analysis_id}` → `useEvidenceList()`
- POST `/reports/generate` → `useGenerateReport()`

## Implementation Highlights

### Real-time Event Handling
The `useAnalysisEvents()` hook provides a clean interface for consuming Server-Sent Events:
- Automatically handles connection lifecycle
- Provides events array with proper TypeScript typing
- Includes connection status and error handling
- Designed to work with backend's JWT-authenticated SSE endpoint

### Mock API Realism
The mock implementation simulates real backend behavior:
- Progressive job completion through all pipeline stages
- Detector-specific events within the detection stage
- Realistic timing delays for each stage
- Event progression that mirrors actual pipeline execution
- Mock data that conforms to all TypeScript interfaces

### Type Safety
Comprehensive TypeScript coverage ensures:
- End-to-end type safety from API to UI components
- Autocomplete and IntelliSense in IDEs
- Compile-time validation of API contracts
- Reduced runtime errors through strict typing

### Developer Experience
- Clear separation of concerns (types, services, components)
- Extensible hook-based architecture
- Comprehensive mock service for frontend-only development
- Well-documented code with consistent patterns
- Easy to extend with additional endpoints or features

## Usage Example

```typescript
import { useUploadMedia, useAsyncAnalysis, useJobStatus, useAnalysisEvents } from './services';

function AnalysisWorkflow() {
  const uploadMedia = useUploadMedia();
  const startAsyncAnalysis = useAsyncAnalysis();
  const [jobId, setJobId] = useState<string | null>(null);
  const jobStatus = useJobStatus(jobId || '');
  const { events, connect, disconnect } = useAnalysisEvents(
    jobStatus.data?.analysis_id || ''
  );

  const handleFileUpload = async (file: File) => {
    const uploadResult = await uploadMedia.mutateAsync(file);
    const analysisResult = await startAsyncAnalysis.mutateAsync({
      media_id: uploadResult.media_id,
    });
    setJobId(analysisResult.job_id);
    connect(); // Start listening for real-time events
  };

  // Job completion handling
  if (jobStatus.data?.status === 'completed') {
    // Analysis finished, disconnect from events
    disconnect();
    // Show results...
  }

  return (
    <div>
      {/* Upload UI */}
      {/* Progress display using jobStatus.data */} 
      {/* Events list using events array */}
      {/* Results visualization */}
    </div>
  );
}
```

## Future Enhancements

1. **WebSocket Implementation**: Replace polling-based job status with real-time WebSocket connections
2. **Charting Integration**: Add charting libraries (Chart.js, D3, or Recharts) for timeline and heatmap visualizations
3. **Advanced Caching**: Implement more sophisticated cache invalidation strategies
4. **Optimistic Updates**: Add optimistic UI updates for better perceived performance
5. **Error Boundaries**: Implement React error boundaries for graceful error handling
6. **Testing**: Add unit and integration tests for hooks and components
7. **Accessibility**: Improve accessibility compliance (WCAG 2.1 AA)
8. **Internationalization**: Add i18n support for multiple languages
9. **Performance Optimization**: Implement code splitting and lazy loading
10. **Production Deployment**: Add production build optimizations and deployment configurations

## Conclusion

This frontend implementation provides a complete, type-safe interface to the observable AIMD backend API. It demonstrates all the key features requested:

✅ TypeScript types for all API endpoints  
✅ React Query hooks for data fetching and mutations  
✅ Streaming hooks for real-time event consumption  
✅ Mock data for development and testing  
✅ Working components demonstrating usage  
✅ Proper styling and configuration  

The implementation follows React and TypeScript best practices and provides a solid foundation for building a complete forensic analysis platform frontend.