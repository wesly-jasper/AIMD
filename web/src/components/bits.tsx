import { useState, type ReactNode } from "react";
import type { Availability, Category, Execution } from "../types";
import { AVAILABILITY, EXECUTION } from "../format";
import { shortVerdict, toneOf } from "../plain";

export function TitleRow({ title, sub, actions }: { title: string; sub?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="title-row">
      <div className="title-text">
        <h1>{title}</h1>
        {sub && <p className="sub">{sub}</p>}
      </div>
      {actions && <div className="title-actions">{actions}</div>}
    </div>
  );
}

/** The result in a list row: a small mark and the short wording. */
export function VerdictTag({ category, status }: { category: Category | null | undefined; status?: string | null }) {
  const tone = category ? toneOf(category) : "none";
  return (
    <span className={"verdict-tag " + tone}>
      <span className="dot" aria-hidden="true" />
      {shortVerdict(category, status)}
    </span>
  );
}

export function AvailabilityMark({ value }: { value: Availability }) {
  return <span className={"state " + (value === "AVAILABLE" ? "on" : "off")}>{AVAILABILITY[value]}</span>;
}

export function ExecutionPill({ value }: { value: Execution }) {
  const tone = { COMPLETED: "", INCONCLUSIVE: "warn", FAILED: "neg", SKIPPED: "hollow" }[value];
  return <span className={"pill " + tone}>{EXECUTION[value]}</span>;
}

export function Hash({ value, n = 12, label }: { value: string | null | undefined; n?: number; label?: string }) {
  const [copied, setCopied] = useState(false);
  if (!value) return <span className="dash">—</span>;
  return (
    <span className="hash">
      <code title={value}>{value.slice(0, n)}…</code>
      <button
        className="copy"
        aria-label={`Copy ${label ?? "hash"}`}
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          navigator.clipboard?.writeText(value).then(() => {
            setCopied(true);
            setTimeout(() => setCopied(false), 1200);
          }).catch(() => {});
        }}
      >
        {copied ? "Copied" : "Copy"}
      </button>
    </span>
  );
}

export function Skeleton({ w, h = 14 }: { w: number | string; h?: number }) {
  return <span className="skeleton" style={{ width: w, height: h, display: "inline-block" }} />;
}

export function ErrorBlock({ what, error, onRetry }: { what: string; error: Error; onRetry?: () => void }) {
  return (
    <div className="error-block" role="alert">
      <span className="error-dot" aria-hidden="true" />
      <span>Couldn’t load {what}. {error.message}</span>
      {onRetry && <button className="text-btn" onClick={onRetry}>Retry</button>}
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}

/**
 * Delete with an inline confirmation that names the file and the consequence.
 * No modal: the question appears where the button was.
 */
export function DeleteControl({ name, onConfirm }: { name: string; onConfirm: () => Promise<void> }) {
  const [asking, setAsking] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!asking) {
    return <button className="text-btn danger" onClick={() => setAsking(true)}>Delete</button>;
  }
  return (
    <span className="confirm" role="group" aria-label={`Delete ${name}`}>
      <span className="confirm-q">Delete {name} and its results?</span>
      <button
        className="btn danger-btn"
        disabled={busy}
        onClick={async () => {
          setBusy(true);
          setError(null);
          try {
            await onConfirm();
          } catch (err) {
            setError(err instanceof Error ? err.message : "Couldn’t delete");
            setBusy(false);
          }
        }}
      >
        {busy ? "Deleting…" : "Delete"}
      </button>
      <button className="text-btn" onClick={() => setAsking(false)} disabled={busy}>Cancel</button>
      {error && <span className="form-error">{error}</span>}
    </span>
  );
}
