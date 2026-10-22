import type { Availability, Execution, Polarity, ScoreSemantics } from "./types";

/** The backend writes UTC; SQLite returns it without an offset. */
function parse(iso: string): Date {
  return new Date(/[zZ]|[+-]\d\d:?\d\d$/.test(iso) ? iso : iso + "Z");
}

const pad = (n: number) => String(n).padStart(2, "0");

/** ISO date and time in the viewer's zone, for rows that are compared. */
export function dateTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = parse(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

export function dateOnly(iso: string | null | undefined): string {
  return dateTime(iso).split(" ")[0];
}

export function bytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

export function duration(ms: number | null | undefined): string {
  if (ms == null) return "—";
  if (ms < 1000) return `${ms} ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)} s`;
  return `${Math.floor(s / 60)} m ${Math.round(s % 60)} s`;
}

export function count(n: number, one: string, many = one + "s"): string {
  return `${n.toLocaleString("en-GB")} ${n === 1 ? one : many}`;
}

export const AVAILABILITY: Record<Availability, string> = {
  AVAILABLE: "Available",
  UNAVAILABLE: "Not installed",
};

export const EXECUTION: Record<Execution, string> = {
  COMPLETED: "Completed",
  INCONCLUSIVE: "Inconclusive",
  FAILED: "Failed",
  SKIPPED: "Did not run",
};

export const POLARITY: Record<Polarity, string> = {
  ANOMALY_OBSERVED: "Anomaly observed",
  NO_ANOMALY_OBSERVED: "No anomaly",
  INDETERMINATE: "Indeterminate",
};

export const SEMANTICS: Record<ScoreSemantics, string> = {
  RAW_MODEL_SCORE: "raw model score",
  CALIBRATED_PROBABILITY: "calibrated probability",
  HEURISTIC_SEVERITY: "heuristic severity",
  NOT_APPLICABLE: "not applicable",
};

/** A non-result's 0.0 is not a measurement, so it prints as a dash. */
export function score(value: number | null | undefined, semantics: ScoreSemantics): string {
  if (value == null || semantics === "NOT_APPLICABLE") return "—";
  return value.toFixed(3);
}

const FAMILY: Record<string, string> = {
  compression: "Compression",
  geometric: "Geometric",
  face: "Face",
  learned: "Learned model",
  non_evidential: "Not evidential",
  unclassified: "Unclassified",
};
export const familyLabel = (f: string) => FAMILY[f] ?? f;
