// Shapes returned by the AIMD backend (/api/v1). Kept to the fields the screens
// read; the backend may return more.

export type Role = "admin" | "user";

export interface User {
  username: string;
  email: string | null;
  role: Role;
  is_active: boolean;
}

/** A folder. The API still calls it a case. */
export interface Folder {
  case_id: string;
  title: string;
  description: string | null;
  status: string;
  created_by: string | null;
  created_at: string | null;
  evidence_count?: number;
}

export type MediaType = "image" | "video" | "audio";

export type Category =
  | "NO_SIGNIFICANT_ANOMALIES_DETECTED"
  | "SUSPICIOUS"
  | "LIKELY_MANIPULATED"
  | "LIKELY_AI_GENERATED"
  | "INCONCLUSIVE";

export type Strength = "NONE" | "WEAK" | "MODERATE" | "STRONG" | "CONTRADICTORY";
export type Availability = "AVAILABLE" | "UNAVAILABLE";
export type Execution = "COMPLETED" | "INCONCLUSIVE" | "FAILED" | "SKIPPED";
export type Polarity = "ANOMALY_OBSERVED" | "NO_ANOMALY_OBSERVED" | "INDETERMINATE";
export type ScoreSemantics =
  | "RAW_MODEL_SCORE"
  | "CALIBRATED_PROBABILITY"
  | "HEURISTIC_SEVERITY"
  | "NOT_APPLICABLE";

export interface LatestAnalysis {
  analysis_id: string;
  status: string;
  assessment: Category | null;
  evidence_strength: Strength | null;
  created_at: string | null;
}

export interface MediaItem {
  media_id: string;
  original_filename: string;
  media_type: MediaType;
  content_type: string;
  size_bytes: number;
  sha256: string;
  case_id: string | null;
  uploaded_by: string | null;
  created_at: string | null;
  latest_analysis?: LatestAnalysis | null;
}

export interface MediaDetail extends MediaItem {
  metadata: {
    width?: number | null;
    height?: number | null;
    fps?: number | null;
    duration_seconds?: number | null;
    codec?: string | null;
  };
  storage_status: "STORED" | "BYTES_MISSING";
}

export interface Usage {
  used: number;
  limit: number;
  window_hours: number;
}

export interface Region {
  type: string;
  confidence: number;
  /** [x, y, width, height] in the media's own pixels. */
  bbox: [number, number, number, number] | null;
  start_timestamp: number | null;
  end_timestamp: number | null;
}

export interface Detection {
  id: string;
  detector: string;
  detector_version: string | null;
  media_type: string;
  manipulation_type: string;
  availability: Availability;
  execution_status: Execution;
  status_reason: string | null;
  score_semantics: ScoreSemantics;
  assessment: string;
  confidence: number;
  regions: Region[];
  /** Detector-specific detail; read through the narrow shapes below. */
  metadata: Record<string, unknown>;
}

/** One evidence file a detector wrote, e.g. a TruFor heatmap. */
export interface Artefact {
  path: string;
  sha256: string;
  media_type: string;
}

export interface FrameScore {
  frame_index: number;
  timestamp_seconds: number;
  score: number;
}

export interface Segment {
  start_frame: number;
  end_frame: number;
  start_timestamp: number;
  end_timestamp: number;
  peak_score: number;
}

export type AiGenerationStatus = "AI_GENERATION_INDICATED" | "NOT_INDICATED" | "UNAVAILABLE";

/** The AI-generation branch: decided apart from the manipulation verdict. */
export interface AiGeneration {
  status: AiGenerationStatus;
  detector: string | null;
  detector_version?: string | null;
  score_semantics?: ScoreSemantics;
  semantics?: string | null;
  limitations?: string[];
  score?: number;
  logit?: number | null;
  threshold?: number | null;
  rationale: string;
  reason?: string | null;
}

export type VerificationStatus = "VERIFIED_MATCH" | "SIMILAR_UNVERIFIED" | "NO_MATCH" | string;

export interface Source {
  id: string;
  url: string;
  title: string | null;
  domain: string | null;
  provider: string | null;
  published_at: string | null;
  similarity: number | null;
  is_earliest_known: boolean;
  verification_status: VerificationStatus;
  verification: Record<string, unknown> | null;
  citation: Record<string, unknown> | null;
}

export interface SourceSearch {
  status: "COMPLETED" | "FAILED" | "UNAVAILABLE";
  reason: string | null;
  hits: number;
  verified: number;
  duration_ms: number | null;
  [detail: string]: unknown;
}

export interface TimelineEvent {
  type: "PUBLISHED" | "ARCHIVED" | "RETRIEVED" | "METADATA" | string;
  timestamp: string;
  precision: string | null;
  description: string;
  basis: string | null;
  citation: Record<string, unknown> | null;
  earliest_known_appearance: boolean;
}

export interface Timeline {
  events: TimelineEvent[];
  earliest_known_appearance: TimelineEvent | null;
  note: string | null;
}

export interface ContentCredentials {
  status: "TRUSTED" | "VALID" | "INVALID" | "NO_MANIFEST" | "UNAVAILABLE" | "FAILED" | string;
  reason?: string;
  signer?: { common_name: string | null; issuer: string | null; signed_at: string | null };
  claim_generator?: string | null;
  digital_source_types?: string[];
  failures?: { code: string | null; explanation: string | null }[];
}

export interface SimilarityMatch {
  id: string;
  match_type: string;
  source_media_id: string | null;
  external_url: string | null;
  similarity: number | null;
  method: string | null;
  algorithm: string | null;
  rank: number | null;
  verification_status: VerificationStatus;
}

export type InvestigationStatus = "RUNNING" | "COMPLETED" | "UNAVAILABLE" | "FAILED";

export interface ToolCall {
  id: string;
  step: number;
  name: string;
  arguments: string | Record<string, unknown> | null;
  category: "FACT" | "OBSERVATION" | null;
  result?: unknown;
  error?: string;
  latency_ms: number;
}

export interface Investigation {
  investigation_id: string;
  analysis_id: string;
  status: InvestigationStatus;
  model: string | null;
  summary: { text: string; category: string; note: string } | null;
  citations: { sentence: string; tool_call_ids: string[] }[];
  flags: { type: string; value: string | null; sentence: string }[];
  grounded: boolean;
  tool_calls: ToolCall[];
  steps: number | null;
  usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number } | null;
  duration_ms: number | null;
  error_message: string | null;
  created_at: string | null;
  completed_at: string | null;
}

export interface Observation {
  id: string;
  detection_id: string;
  detector: string;
  signal_family: string;
  polarity: Polarity;
  score: number | null;
  score_semantics: ScoreSemantics;
  localised: boolean;
  description: string;
}

export interface Finding {
  id: string;
  statement: string;
  supporting_observation_ids: string[];
  contradicting_observation_ids: string[];
}

export interface AnalysisDetail {
  analysis_id: string;
  media_id: string;
  media_type: MediaType;
  status: string;
  assessment: Category | null;
  evidence_strength: Strength | null;
  created_at: string | null;
  completed_at: string | null;
  error_message: string | null;
  run: {
    requested_by: string | null;
    software_version: string | null;
    detector_set_version: string | null;
    config_snapshot: Record<string, unknown> | null;
    duration_ms: number | null;
  };
  assessment_detail: {
    category: Category;
    evidence_strength: Strength;
    rule: string;
    rationale: string;
    coverage_gaps: string[];
  } | null;
  /** NULL for analyses recorded before the branch existed. */
  ai_generation: AiGeneration | null;
  suspicious_segments: Segment[];
  similarity_matches: SimilarityMatch[];
  /** NULL before source search existed; keyed by connector name. */
  source_searches: Record<string, SourceSearch> | null;
  sources: Source[];
  provenance_timeline: Timeline;
  content_credentials: ContentCredentials | null;
  observations: Observation[];
  findings: Finding[];
  detections: Detection[];
}

export interface Page<T> {
  items: T[];
  count: number;
}
