import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { deleteMedia, postJson } from "../api";
import { bytes, dateTime } from "../format";
import type { MediaItem } from "../types";
import { DeleteControl, VerdictTag } from "./bits";
import { Thumb } from "./Evidence";

/** One file the user has checked: what it is, what we found, and what to do. */
export function MediaRow({
  item, folderName, onDeleted,
}: {
  item: MediaItem;
  folderName?: string | null;
  onDeleted?: () => void;
}) {
  const latest = item.latest_analysis;
  const navigate = useNavigate();
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function runCheck() {
    setRunning(true);
    setError(null);
    try {
      const r = await postJson<{ analysis_id: string }>("/analysis/", { media_id: item.media_id });
      navigate(`/results/${r.analysis_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "The check could not be run.");
      setRunning(false);
    }
  }
  const meta = [
    item.media_type.charAt(0).toUpperCase() + item.media_type.slice(1),
    bytes(item.size_bytes),
    folderName ? `in ${folderName}` : null,
  ].filter(Boolean).join(" · ");

  return (
    <li className="media-row">
      <Thumb mediaId={item.media_id} mediaType={item.media_type} />
      <div className="media-id">
        {latest ? (
          <Link className="name" to={`/results/${latest.analysis_id}`}>{item.original_filename}</Link>
        ) : (
          <span className="name">{item.original_filename}</span>
        )}
        <span className="sub">{meta}</span>
      </div>
      <div className="media-result">
        <VerdictTag category={latest?.assessment} />
        <span className="sub num">{dateTime(latest?.created_at ?? item.created_at)}</span>
      </div>
      <div className="media-actions">
        {latest
          ? <Link className="btn" to={`/results/${latest.analysis_id}`}>View result</Link>
          : <button className="btn" onClick={runCheck} disabled={running}>{running ? "Checking…" : "Run check"}</button>}
        <DeleteControl
          name={item.original_filename}
          onConfirm={async () => { await deleteMedia(item.media_id); onDeleted?.(); }}
        />
      </div>
      {error && <p className="row-note form-error" role="alert">{error}</p>}
    </li>
  );
}
