import { useEffect, useRef, useState } from "react";
import { ApiError, getBlobUrl, getJson } from "../api";

type View = "html" | "json";

const NOT_WRITTEN = "No report has been written for this analysis yet.";

async function download(path: string, filename: string): Promise<void> {
  const url = await getBlobUrl(path);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/**
 * The stored report, read back from the API. The HTML form is rendered in a
 * sandboxed frame (no scripts) and is what "Save as PDF" prints.
 */
export function ReportCard({ analysisId }: { analysisId: string }) {
  const base = `/analysis/${analysisId}/report`;
  const [view, setView] = useState<View>("html");
  const [html, setHtml] = useState<string | null>(null);
  const [json, setJson] = useState<unknown>(null);
  const [error, setError] = useState<Error | null>(null);
  const [tick, setTick] = useState(0);
  const [busy, setBusy] = useState<string | null>(null);
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const frame = useRef<HTMLIFrameElement>(null);

  useEffect(() => {
    let live = true;
    let made: string | null = null;
    setError(null);
    const load = view === "html"
      ? getBlobUrl(`${base}?format=html`).then((u) => { made = u; if (live) setHtml(u); else URL.revokeObjectURL(u); })
      : getJson<unknown>(base).then((j) => { if (live) setJson(j); });
    load.catch((e: Error) => { if (live) setError(e); });
    return () => { live = false; if (made) URL.revokeObjectURL(made); };
  }, [base, view, tick]);

  async function save(format: "json" | "html" | "md") {
    setBusy(format);
    setDownloadError(null);
    try {
      await download(`${base}?format=${format}`, `aimd-report-${analysisId.slice(0, 8)}.${format}`);
    } catch (e) {
      setDownloadError(e instanceof Error ? e.message : "The download failed.");
    } finally {
      setBusy(null);
    }
  }

  const missing = error instanceof ApiError && error.status === 404;
  const ready = !error && (view === "html" ? html !== null : json !== null);

  return (
    <section className="card report" aria-labelledby="report-h">
      <div className="card-head">
        <h2 id="report-h">Report</h2>
        <div className="segmented compact" role="tablist" aria-label="Report view">
          <button role="tab" aria-selected={view === "html"} onClick={() => setView("html")}>Document</button>
          <button role="tab" aria-selected={view === "json"} onClick={() => setView("json")}>JSON</button>
        </div>
      </div>

      {missing ? (
        <p className="empty">{NOT_WRITTEN}</p>
      ) : error ? (
        <div className="error-block" role="alert">
          <span className="error-dot" aria-hidden="true" />
          <span>The report could not be loaded: {error.message}</span>
          <button className="text-btn" onClick={() => setTick((n) => n + 1)}>Try again</button>
        </div>
      ) : !ready ? (
        <div className="report-frame loading" aria-busy="true" />
      ) : view === "html" ? (
        <iframe ref={frame} className="report-frame" src={html!} title="Forensic report"
          sandbox="allow-same-origin allow-modals" />
      ) : (
        <pre className="report-json">{JSON.stringify(json, null, 2)}</pre>
      )}

      {!missing && (
        <div className="report-actions">
          <button className="btn" disabled={view !== "html" || !ready}
            onClick={() => frame.current?.contentWindow?.print()}>Print / Save as PDF</button>
          <button className="btn" disabled={busy !== null} onClick={() => void save("json")}>{busy === "json" ? "Downloading…" : "Download JSON"}</button>
          <button className="btn" disabled={busy !== null} onClick={() => void save("html")}>{busy === "html" ? "Downloading…" : "Download HTML"}</button>
          <button className="btn" disabled={busy !== null} onClick={() => void save("md")}>{busy === "md" ? "Downloading…" : "Download Markdown"}</button>
        </div>
      )}
      {downloadError && <p className="form-error" role="alert">{downloadError}</p>}
    </section>
  );
}
