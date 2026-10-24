// Forensic categories in everyday words.
//
// This is the only place a category becomes a sentence, so the honesty rules
// live here:
//   * "No signs found" never becomes "authentic", "original" or "genuine".
//   * Weak evidence says it is weak, and names its most common innocent cause.
//   * A check that did not run is "not available", never "found nothing".

import type { Category, Detection, Observation, Strength } from "./types";

export type Tone = "flag" | "clear" | "unsure";

export interface Verdict {
  headline: string;
  explanation: string;
  tone: Tone;
}

export function verdict(category: Category | null | undefined, strength: Strength | null | undefined): Verdict {
  switch (category) {
    case "LIKELY_MANIPULATED":
      return {
        headline: "Strong signs of editing",
        explanation: "Independent kinds of check agree, and they point to specific areas of the file.",
        tone: "flag",
      };
    case "LIKELY_AI_GENERATED":
      return {
        headline: "Signs of AI generation",
        explanation: "The checks found patterns typical of AI-generated media.",
        tone: "flag",
      };
    case "SUSPICIOUS":
      return strength === "MODERATE" || strength === "STRONG"
        ? {
            headline: "Possible signs of editing",
            explanation: "More than one independent kind of check found something unusual.",
            tone: "flag",
          }
        : {
            headline: "Possible signs of editing",
            explanation:
              "One kind of check found something unusual. On its own this is weak evidence: " +
              "saving a photo again, or editing its brightness, can leave the same traces.",
            tone: "flag",
          };
    case "NO_SIGNIFICANT_ANOMALIES_DETECTED":
      return {
        headline: "No signs of editing found",
        explanation:
          "Every check ran and none found anything unusual. This does not prove the file is original, " +
          "only that these checks found nothing.",
        tone: "clear",
      };
    case "INCONCLUSIVE":
      return {
        headline: "We couldn't reach a conclusion",
        explanation: "Not every check could run on this file, so the absence of signs can't be reported.",
        tone: "unsure",
      };
    default:
      return { headline: "No result yet", explanation: "This file hasn't been checked.", tone: "unsure" };
  }
}

const RUNNING = new Set(["QUEUED", "PROCESSING"]);

/**
 * The headline for an analysis: its run status first, then its category. A
 * failed or unfinished run has no category, and used to fall through to
 * "This file hasn't been checked" -- untrue for a check that ran and failed.
 */
export function analysisVerdict(
  status: string | null | undefined, category: Category | null | undefined, strength: Strength | null | undefined,
): Verdict {
  if (status === "FAILED") {
    return {
      headline: "The check failed",
      explanation: "Something went wrong while checking this file, so there is no result. Nothing was concluded about the file.",
      tone: "unsure",
    };
  }
  if (status && RUNNING.has(status)) {
    return {
      headline: "Checking this file…",
      explanation: "The checks are running. A large image can take a minute or two; this page updates when they finish.",
      tone: "unsure",
    };
  }
  return verdict(category, strength);
}

/** Short form for lists. */
export function shortVerdict(category: Category | null | undefined, status?: string | null): string {
  if (status === "FAILED") return "Check failed";
  if (status && RUNNING.has(status)) return "Checking…";
  switch (category) {
    case "LIKELY_MANIPULATED": return "Strong signs of editing";
    case "LIKELY_AI_GENERATED": return "Signs of AI generation";
    case "SUSPICIOUS": return "Possible signs of editing";
    case "NO_SIGNIFICANT_ANOMALIES_DETECTED": return "No signs found";
    case "INCONCLUSIVE": return "No conclusion";
    default: return "Not checked";
  }
}

export const toneOf = (category: Category | null | undefined): Tone => verdict(category, null).tone;

/** What each check looks at, in words a non-specialist can follow. */
const CHECKS: Record<string, { name: string; what: string }> = {
  ForensicImageDetector: {
    name: "Compression traces",
    what: "Areas that were saved a different number of times from the rest of the image",
  },
  ManipulationMaskGenerator: {
    name: "Compression map",
    what: "A second reading of compression differences across the image",
  },
  CopyMoveDetector: {
    name: "Copied areas",
    what: "Parts of the image copied and pasted elsewhere in the same image",
  },
  FaceRegionAnalyzer: {
    name: "Face edits",
    what: "Blending and lighting inconsistencies around faces",
  },
  BaselineFaceDetector: { name: "Face finding", what: "Locates faces so face checks can run" },
  DeepLearningFaceDetector: { name: "Face finding", what: "Locates faces so face checks can run" },
  TruForDetector: { name: "Forgery model", what: "A trained model that looks for image forgeries" },
  UnivFDDetector: { name: "AI-generation model", what: "A trained model that scores whether a model made the image" },
  VideoTemporalAnalysisEngine: { name: "Frame-to-frame changes", what: "Sudden changes between sampled video frames" },
  YOLOv8ObjectDetector: { name: "Object recognition", what: "Lists what is in the picture; not evidence of editing" },
  BaselineObjectDetector: { name: "Object recognition", what: "Lists what is in the picture; not evidence of editing" },
  BaselineVideoDetector: { name: "Frame checks", what: "Image checks run across sampled video frames" },
  BaselineAudioAnalyzer: { name: "Audio checks", what: "Spectral checks on the soundtrack" },
  SpectralSpeechHeuristicDetector: { name: "Speech checks", what: "Spectral checks for synthetic speech" },
};

export function checkName(detector: string): string {
  // Unknown names: "UnivFDDetector" -> "Univ FD Detector", not "Univ FDDetector".
  return CHECKS[detector]?.name ?? detector.replace(/([a-z])([A-Z])|([A-Z])([A-Z][a-z])/g, "$1$3 $2$4");
}

export function checkWhat(detector: string): string {
  return CHECKS[detector]?.what ?? "";
}

export type CheckOutcome = "found" | "nothing" | "undecided" | "context" | "unavailable";

export const OUTCOME_LABEL: Record<CheckOutcome, string> = {
  found: "Found signs",
  nothing: "Found nothing",
  undecided: "Couldn't decide",
  context: "Context only",
  unavailable: "Not available",
};

/** Checks that describe the picture and never weigh on whether it was edited. */
const CONTEXT_ONLY = new Set(["YOLOv8ObjectDetector", "BaselineObjectDetector"]);

export function outcomeOf(d: Detection, observations: Observation[]): CheckOutcome {
  const o = observations.find((x) => x.detection_id === d.id);
  if (!o) return "unavailable";
  if (CONTEXT_ONLY.has(d.detector)) return "context";
  if (o.polarity === "ANOMALY_OBSERVED") return "found";
  if (o.polarity === "NO_ANOMALY_OBSERVED") return "nothing";
  return "undecided";
}

/** What a context-only check saw: "person, 2 dogs" from object regions. */
export function contextSummary(d: Detection): string {
  const counts = new Map<string, number>();
  for (const r of d.regions) {
    const name = r.type.replace(/^object_/, "");
    counts.set(name, (counts.get(name) ?? 0) + 1);
  }
  if (counts.size === 0) return "Nothing recognised in the picture";
  return "Saw " + [...counts].map(([name, n]) => (n === 1 ? name : `${n} × ${name}`)).join(", ");
}

/** Why a check is "not available", in plain words. */
export function unavailableReason(d: Detection): string {
  if (d.availability === "UNAVAILABLE") return "Not set up on this server";
  if (d.execution_status === "FAILED") return "Failed while running";
  return d.status_reason ?? "Did not run";
}
