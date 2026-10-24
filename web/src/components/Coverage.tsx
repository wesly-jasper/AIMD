import type { Detection, Observation } from "../types";
import {
  OUTCOME_LABEL, checkName, checkWhat, contextSummary, outcomeOf, unavailableReason, type CheckOutcome,
} from "../plain";

const ORDER: CheckOutcome[] = ["found", "nothing", "undecided", "context", "unavailable"];

function detail(d: Detection, outcome: CheckOutcome): string {
  if (outcome === "unavailable") return unavailableReason(d);
  if (outcome === "context") return `${contextSummary(d)} · not evidence of editing`;
  return checkWhat(d.detector);
}

/**
 * What we checked: one segment per check that was invoked on this file.
 *
 * A check that did not run is hatched and listed as "Not available" with its
 * reason. It is never drawn or worded as "found nothing": an absent check must
 * not read as a clean result.
 */
export function WhatWeChecked({
  detections, observations, onPoint,
}: {
  detections: Detection[];
  observations: Observation[];
  onPoint?: (detectionId: string | null) => void;
}) {
  if (detections.length === 0) return <p className="muted">No checks ran on this file.</p>;

  const rows = detections
    .map((d) => ({ d, outcome: outcomeOf(d, observations) }))
    .sort((a, b) => ORDER.indexOf(a.outcome) - ORDER.indexOf(b.outcome));
  const ran = rows.filter((r) => r.outcome !== "unavailable").length;

  return (
    <div className="checks">
      <p className="checks-head">
        <span className="num checks-figure">{ran}</span>
        <span className="muted"> of {detections.length} checks ran on this file</span>
      </p>
      <div className="bar" role="img" aria-label={rows.map((r) => `${checkName(r.d.detector)}: ${OUTCOME_LABEL[r.outcome]}`).join("; ")}>
        {rows.map(({ d, outcome }) => (
          <span key={d.id} className={"seg " + outcome} />
        ))}
      </div>
      <ul className="check-list">
        {rows.map(({ d, outcome }) => (
          <li
            key={d.id}
            tabIndex={0}
            onMouseEnter={() => onPoint?.(d.id)}
            onMouseLeave={() => onPoint?.(null)}
            onFocus={() => onPoint?.(d.id)}
            onBlur={() => onPoint?.(null)}
          >
            <span className={"swatch " + outcome} aria-hidden="true" />
            <span className="check-text">
              <span className="check-name">{checkName(d.detector)}</span>
              <span className="sub">{detail(d, outcome)}</span>
            </span>
            <span className={"outcome " + outcome}>{OUTCOME_LABEL[outcome]}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
