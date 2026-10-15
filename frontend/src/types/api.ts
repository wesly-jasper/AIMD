// TypeScript interfaces for AIMD backend API
export interface RegisterRequest {
  email: string;
  password: string;
  full_name?: string;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: {
    id: string;
    email: string;
    full_name: string;
  };
}

export interface MediaUploadResponse {
  media_id: string;
  filename: string;
  size: number;
  mime_type: string;
  upload_timestamp: string;
}

export interface MediaMetadata {
  media_id: string;
  filename: string;
  size: number;
  mime_type: string;
  upload_timestamp: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  duration?: number;
  width?: number;
  height?: number;
  format?: string;
}

export interface MediaListResponse {
  items: MediaMetadata[];
  total: number;
  page: number;
  page_size: number;
}

export interface AnalysisRequest {
  media_id: string;
  options?: Record<string, any>;
}

export interface AnalysisResponse {
  analysis_id: string;
  media_id: string;
  status: 'completed';
  completed_at: string;
  // results would be nested, but we keep simple
}

export interface AsyncAnalysisResponse {
  job_id: string;
  analysis_id: string;
  status: 'submitted';
}

export type JobStatus =
  | { status: 'submitted' }
  | { status: 'processing'; stage: number; progress: number }
  | { status: 'completed'; analysis_id: string }
  | { status: 'failed'; error: string };

export interface DetectionResult {
  type: string;
  confidence: number;
  region?: DetectionRegion;
  details?: Record<string, any>;
}

export interface DetectionRegion {
  x: number;
  y: number;
  width: number;
  height: number;
  // for video: frame_number or timestamp
  frame_number?: number;
  timestamp?: number;
}

export interface SimilarityMatch {
  media_id: string;
  similarity: number;
  // could include thumbnail, etc.
}

export interface SimilaritySearchResponse {
  query_media_id: string;
  matches: SimilarityMatch[];
  searched_at: string;
}

export interface ProvenanceNode {
  id: string;
  type: 'original' | 'derivative' | 'transformation';
  media_id: string;
  timestamp: string;
  confidence: number;
  metadata?: Record<string, any>;
}

export interface ProvenanceEdge {
  id: string;
  source: string;
  target: string;
  transformation_type: string;
  confidence: number;
}

export interface ProvenanceGraph {
  nodes: ProvenanceNode[];
  edges: ProvenanceEdge[];
}

export interface EvidenceItem {
  id: string;
  category: 'FACT' | 'OBSERVATION' | 'INFERENCE' | 'UNCERTAINTY';
  description: string;
  confidence: number;
  metadata?: Record<string, any>;
}

export interface EvidenceListResponse {
  evidence: EvidenceItem[];
  analysis_id: string;
}

export interface ReportRequest {
  analysis_id: string;
  format: 'JSON' | 'MARKDOWN' | 'PDF';
}

export interface ReportResponse {
  report_id: string;
  analysis_id: string;
  format: string;
  generated_at: string;
  download_url: string;
}

// Visualization types
export interface HeatmapResponse {
  analysis_id: string;
  heatmap_url: string; // base64 or URL
  width: number;
  height: number;
}

export interface RegionsResponse {
  analysis_id: string;
  regions: DetectionResult[];
}

export interface AnalysisTimeline {
  analysis_id: string;
  frame_scores: Array<{ frame_number: number; score: number }>;
}

export interface FrameScore {
  frame_number: number;
  score: number;
  // per-detector scores?
}
