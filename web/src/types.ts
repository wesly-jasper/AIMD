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

export type AiGenerationStatus =
  | "AI_GENERATION_INDICATED"
  | "NOT_INDICATED"
  | "CONFLICTING"   // two generation detectors disagree; reported, not resolved
  | "INCONCLUSIVE"  // ran, could not support a determination (e.g. video keyframes)
  | "UNAVAILABLE";

/** How a calibrated score was made: fitted on a validation split, recorded. */
export interface ScoreCalibration {
  applied: boolean;
  probability?: number;
  raw_score?: number;
  threshold?: number;
  target_fpr?: number;
  method?: string;
  reason?: string;
  fitted_on?: { positives?: number; negatives?: number; split?: string; dataset_sha256?: string };
  validation?: { fpr?: number; recall?: number };
}

/** One AI-generation detector's own result, when several ran. */
export interface AiDetectorResult {
  detector: string;
  status: AiGenerationStatus;
  score: number | null;
  threshold: number | null;
  score_semantics: ScoreSemantics;
  reason: string | null;
  /** false: shown for reference, outweighed by a more accurate detector. */
  voting?: boolean;
}

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
  calibration?: ScoreCalibration | null;
  detectors?: AiDetectorResult[];
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

export interface C2paAction {
  action: string | null;
  when: string | null;
  software_agent: string | null;
  digital_source_type: string | null;
}

export interface ContentCredentials {
  status: "TRUSTED" | "VALID" | "INVALID" | "NO_MANIFEST" | "UNAVAILABLE" | "FAILED" | string;
  reason?: string;
  sdk_version?: string;
  /** The root certificates a signer was checked against; absent on older records. */
  trust_list?: { loaded: boolean; name?: string; anchors?: number; sha256?: string };
  trusted_signer?: boolean;
  signer?: { common_name: string | null; issuer: string | null; algorithm?: string | null; signed_at: string | null };
  claim_generator?: string | null;
  title?: string | null;
  embedded?: boolean;
  actions?: C2paAction[];
  digital_source_types?: string[];
  /** The signer's declaration of AI generation, while the manifest is intact. */
  ai_generation_declared?: boolean;
  ingredients?: number;
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

export type FusionStatus = "SUPPORTS" | "AGAINST" | "LIMITED" | "CONFLICTING" | "NOT_ASSESSED";

export interface FusionContribution {
  detector: string;
  family: string;
  status: "observed" | "no_observation" | "not_run";
  advisory: boolean;
  direction?: 1 | -1;
  calibrated?: boolean;
  log10_lr?: number | null;
  capped?: boolean;
  counted?: boolean;
}

/** One branch's evidence as a likelihood-ratio range; never changes the verdicts. */
export interface FusionBranch {
  status: FusionStatus;
  statement: string;
  log10_lr_range?: [number, number];
  strength?: string;
  any_capped?: boolean;
  conflicts: { for: string; against: string }[];
  uncalibrated: string[];
  contributions: FusionContribution[];
}

export interface EvidenceFusion {
  method: string;
  version: number;
  manipulation: FusionBranch;
  ai_generation: FusionBranch;
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
  /** NULL for analyses recorded before evidence fusion existed. */
  evidence_fusion?: EvidenceFusion | null;
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
