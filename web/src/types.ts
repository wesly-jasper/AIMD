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
  observations: Observation[];
  findings: Finding[];
  detections: Detection[];
}

export interface Page<T> {
  items: T[];
  count: number;
}
