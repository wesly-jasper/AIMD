import { useEffect, useRef, useState } from "react";
import { ApiError, getJson, postJson } from "../api";
import { dateTime, duration } from "../format";
import type { Investigation, ToolCall } from "../types";
import { Unavailable } from "./Panels";

const STATUS: Record<Investigation["status"], { label: string; tone: string }> = {
  RUNNING: { label: "Running", tone: "" },
  COMPLETED: { label: "Completed", tone: "" },
  UNAVAILABLE: { label: "Unavailable", tone: "hollow" },
  FAILED: { label: "Failed", tone: "neg" },
};

const FLAG: Record<string, string> = {
  uncited: "Cites no tool call",
  unknown_citation: "Cites a tool call that does not exist (removed)",
  unsupported_url: "A link no tool returned (removed)",
  unsupported_number: "A number no tool returned (removed)",
  unsupported_timestamp: "A date no tool returned (removed)",
};

const TOOL: Record<string, string> = {
  get_analysis_overview: "Read the result overview",
  list_detections: "List every check and its outcome",
  get_observations_and_findings: "Read the observations and finding",
  get_similarity_matches: "Read the similarity matches",
  get_sources: "Read the sources found",
  get_provenance_timeline: "Read the provenance timeline",
  get_metadata: "Read the file’s metadata",
  verify_source_match: "Fetch and compare a URL",
};

const CITE = /\[(T\d+(?:\s*,\s*T\d+)*)\]|【(T\d+(?:\s*,\s*T\d+)*)】/g;

/** The summary with each [T#] turned into a link to its entry in the trace. */
function CitedText({ text, known, onCite }: { text: string; known: Set<string>; onCite: (id: string) => void }) {
  const parts: React.ReactNode[] = [];
  let last = 0;
  for (const m of text.matchAll(CITE)) {
    parts.push(text.slice(last, m.index));
    const ids = (m[1] ?? m[2]).split(/\s*,\s*/);
    parts.push(
      <span key={m.index} className="cites">
        {ids.map((id) => known.has(id)
          ? <button key={id} type="button" className="cite" aria-controls={`trace-${id}`}
              aria-label={`Show tool call ${id} in the trace`} onClick={() => onCite(id)}>{id}</button>
          : <span key={id} className="cite unknown">{id}</span>)}
      </span>,
    );
    last = (m.index ?? 0) + m[0].length;
  }
  parts.push(text.slice(last));
  return <p className="summary-text">{parts}</p>;
}

function args(a: ToolCall["arguments"]): string {
  if (a == null) return "{}";
  if (typeof a === "string") {
    try { return JSON.stringify(JSON.parse(a)); } catch { return a; }
  }
  return JSON.stringify(a);
}

function resultText(r: unknown): string {
  const s = JSON.stringify(r, null, 2) ?? "null";
  return s.length > 6000 ? s.slice(0, 6000) + "\n… (truncated for display)" : s;
}

function Plan({ calls }: { calls: ToolCall[] }) {
  const steps = [...new Set(calls.map((c) => c.step))].sort((a, b) => a - b);
  return (
    <ol className="plan">
      {steps.map((s) => (
        <li key={s}>
          <span className="plan-step num">{s === 0 ? "First" : `Step ${s}`}</span>
          <span>{calls.filter((c) => c.step === s).map((c) => TOOL[c.name] ?? c.name).join(" · ")}</span>
        </li>
      ))}
    </ol>
  );
}

function Trace({ calls, open, focused, setOpen }: { calls: ToolCall[]; open: Set<string>; focused: string | null; setOpen: (id: string) => void }) {
  return (
    <ul className="trace">
      {calls.map((c) => (
        <li key={c.id} id={`trace-${c.id}`} className={focused === c.id ? "focused" : undefined}>
          <details open={open.has(c.id)} onToggle={(e) => { if (e.currentTarget.open) setOpen(c.id); }}>
            <summary>
              <span className="cite static">{c.id}</span>
              <code>{c.name}</code>
              {c.category && <span className={"pill " + (c.category === "FACT" ? "" : "hollow")}>{c.category === "FACT" ? "Fact" : "Observation"}</span>}
              {c.error && <span className="pill neg">Error</span>}
              <span className="sub-inline num trace-latency">{duration(c.latency_ms)}</span>
            </summary>
            <div className="trace-body">
              <p className="sub">Arguments</p>
              <pre>{args(c.arguments)}</pre>
              <p className="sub">{c.error ? "Error" : "Returned"}</p>
              <pre>{c.error ?? resultText(c.result)}</pre>
            </div>
          </details>
        </li>
      ))}
    </ul>
  );
}

/**
 * The Nemotron investigation: an agent reads this analysis through AIMD's own
 * tools and writes a summary in which every sentence cites the calls it rests on.
 * It never changes the analysis; its summary is an interpretation, labelled so.
 */
export function InvestigationCard({ analysisId, finished, onFinished }: {
  analysisId: string; finished: boolean; onFinished?: () => void;
}) {
  const [inv, setInv] = useState<Investigation | null>(null);
  const [loadError, setLoadError] = useState<Error | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);
  const [sourceUrl, setSourceUrl] = useState("");
  const [open, setOpen] = useState<Set<string>>(new Set());
  const [tick, setTick] = useState(0);

  useEffect(() => {
    let live = true;
    getJson<Investigation>(`/analysis/${analysisId}/investigation`)
      .then((r) => { if (live) { setInv(r); setLoadError(null); } })
      .catch((e: Error) => {
        if (!live) return;
        if (e instanceof ApiError && e.status === 404) { setInv(null); setLoadError(null); }
        else setLoadError(e);
      })
      .finally(() => { if (live) setLoaded(true); });
    return () => { live = false; };
  }, [analysisId, tick]);

  // A run that was RUNNING has finished: the served report now includes it.
  const lastStatus = useRef<string | undefined>(undefined);
  useEffect(() => {
    if (lastStatus.current === "RUNNING" && inv && inv.status !== "RUNNING") onFinished?.();
    lastStatus.current = inv?.status;
  }, [inv, onFinished]);

  // Poll while the agent works; it runs in the background on the server.
  useEffect(() => {
    if (inv?.status !== "RUNNING") return;
    const t = window.setTimeout(() => setTick((n) => n + 1), 2000);
    return () => window.clearTimeout(t);
  }, [inv, tick]);

  async function start() {
    setStarting(true);
    setStartError(null);
    try {
      const body = sourceUrl.trim() ? { source_url: sourceUrl.trim() } : {};
      await postJson(`/analysis/${analysisId}/investigate`, body);
      setOpen(new Set());
      setTick((n) => n + 1);
    } catch (e) {
      setStartError(e instanceof Error ? e.message : "The investigation could not be started.");
    } finally {
      setStarting(false);
    }
  }

  const [focused, setFocused] = useState<string | null>(null);
  const expand = (id: string) => setOpen((s) => new Set(s).add(id));
  // A citation opens its trace entry and scrolls to it. Not a #fragment link:
  // the app routes on the hash, so one would navigate away from this page.
  const cite = (id: string) => {
    expand(id);
    setFocused(id);
    window.requestAnimationFrame(() =>
      document.getElementById(`trace-${id}`)?.scrollIntoView({ behavior: "smooth", block: "center" }));
  };
  const running = inv?.status === "RUNNING";
  const known = new Set(inv?.tool_calls.map((c) => c.id) ?? []);
  const removed = inv?.flags.filter((f) => f.type !== "uncited") ?? [];
  const uncited = inv?.flags.filter((f) => f.type === "uncited") ?? [];

  return (
    <section className="card investigation" aria-labelledby="inv-h" aria-busy={running}>
      <div className="card-head">
        <h2 id="inv-h">Investigation</h2>
        {inv && <span className={"pill " + STATUS[inv.status].tone}>{STATUS[inv.status].label}</span>}
      </div>
      <p className="sub">
        An AI agent ({inv?.model ?? "Nemotron"}) reads this result through AIMD’s own tools and writes a summary in which every
        sentence cites the tool calls it rests on. It cannot change the result.
      </p>

      <form className="inv-start" onSubmit={(e) => { e.preventDefault(); void start(); }}>
        <input className="input" type="url" inputMode="url" placeholder="Optional: a URL where you saw this image"
          value={sourceUrl} onChange={(e) => setSourceUrl(e.target.value)} aria-label="Source URL to verify (optional)"
          disabled={!finished || running || starting} />
        <button className="btn primary" type="submit" disabled={!finished || running || starting}>
          {starting ? "Starting…" : running ? "Investigating…" : inv ? "Run again" : "Run investigation"}
        </button>
      </form>
      {!finished && <p className="sub">The analysis must finish before it can be investigated.</p>}
      {startError && <p className="form-error" role="alert">{startError}</p>}

      {!loaded ? (
        <div className="inv-loading"><span className="spinner" aria-hidden="true" /> <span className="sub-inline">Loading…</span></div>
      ) : loadError ? (
        <div className="error-block" role="alert">
          <span className="error-dot" aria-hidden="true" />
          <span>The investigation could not be loaded: {loadError.message}</span>
          <button className="text-btn" onClick={() => setTick((n) => n + 1)}>Try again</button>
        </div>
      ) : !inv ? (
        <p className="empty">No investigation has been run for this result yet.</p>
      ) : (
        <>
          {running && (
            <div className="inv-loading" role="status">
              <span className="spinner" aria-hidden="true" />
              <span className="sub-inline">The agent is calling tools. This usually takes under a minute.</span>
            </div>
          )}
          {(inv.status === "UNAVAILABLE" || inv.status === "FAILED") && (
            <Unavailable>{inv.error_message ?? "The agent did not produce a summary."}</Unavailable>
          )}

          {inv.summary && (
            <div className="summary">
              <div className="summary-head">
                <span className="pill hollow">Inference</span>
                {inv.grounded
                  ? <span className="pill">Grounded: every claim traced to a tool</span>
                  : <span className="pill warn">{removed.length} unsupported {removed.length === 1 ? "claim" : "claims"} removed</span>}
              </div>
              <CitedText text={inv.summary.text} known={known} onCite={cite} />
              <p className="sub">{inv.summary.note}</p>
            </div>
          )}

          {inv.flags.length > 0 && (
            <div className="flags">
              <h3>Grounding check</h3>
              <ul>
                {[...removed, ...uncited].map((f, i) => (
                  <li key={i}>
                    <span className={"pill " + (f.type === "uncited" ? "hollow" : "warn")}>{FLAG[f.type] ?? f.type}</span>
                    {f.value && <code>{f.value}</code>}
                    <span className="sub">“{f.sentence}”</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {inv.tool_calls.length > 0 && (
            <div className="inv-cols">
              <div>
                <h3>Plan</h3>
                <p className="sub">The tools the agent chose, in the order it called them. A URL you supply is checked first, by AIMD itself.</p>
                <Plan calls={inv.tool_calls} />
              </div>
              <div>
                <h3>Tool trace</h3>
                <p className="sub">What each call was asked and what it returned. Facts come from the file or a fetched source; observations from AIMD’s checks.</p>
                <Trace calls={inv.tool_calls} open={open} focused={focused} setOpen={expand} />
              </div>
            </div>
          )}

          <dl className="run-strip inv-run">
            <div><dt>Started</dt><dd className="num">{dateTime(inv.created_at)}</dd></div>
            <div><dt>Duration</dt><dd className="num">{duration(inv.duration_ms)}</dd></div>
            <div><dt>Model turns</dt><dd className="num">{inv.steps ?? "—"}</dd></div>
            <div><dt>Tool calls</dt><dd className="num">{inv.tool_calls.length}</dd></div>
            <div><dt>Tokens</dt><dd className="num">{inv.usage ? inv.usage.total_tokens.toLocaleString() : "—"}</dd></div>
          </dl>
        </>
      )}
    </section>
  );
}
