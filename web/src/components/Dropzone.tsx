import { useRef, useState, type DragEvent } from "react";
import { useNavigate } from "react-router-dom";
import { postForm, postJson } from "../api";
import type { MediaItem } from "../types";

const ACCEPT = "image/jpeg,image/png,image/webp,image/tiff,video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,audio/wav,audio/mpeg,audio/mp4";

type Stage =
  | { kind: "idle" }
  | { kind: "uploading"; name: string }
  | { kind: "analysing"; name: string; video: boolean }
  | { kind: "error"; message: string };

/**
 * Drop a file and get a result: upload, then run the checks, then open the
 * result. The two steps are separate requests, and each one's real state is
 * shown -- there is no invented progress percentage.
 */
export function Dropzone({
  folderId, compact = false, onDone, remaining,
}: {
  folderId?: string;
  compact?: boolean;
  onDone?: () => void;
  remaining?: number | null;
}) {
  const navigate = useNavigate();
  const input = useRef<HTMLInputElement>(null);
  const [stage, setStage] = useState<Stage>({ kind: "idle" });
  const [over, setOver] = useState(false);
  const busy = stage.kind === "uploading" || stage.kind === "analysing";
  const exhausted = remaining === 0;

  async function check(file: File) {
    setStage({ kind: "uploading", name: file.name });
    try {
      const form = new FormData();
      form.append("file", file);
      if (folderId) form.append("case_id", folderId);
      const media = await postForm<MediaItem>("/media/upload", form);
      setStage({ kind: "analysing", name: file.name, video: media.media_type === "video" });
      const run = await postJson<{ analysis_id: string }>("/analysis/", { media_id: media.media_id });
      onDone?.();
      navigate(`/results/${run.analysis_id}`);
    } catch (err) {
      setStage({ kind: "error", message: err instanceof Error ? err.message : "The check could not be completed." });
      onDone?.();
    } finally {
      if (input.current) input.current.value = "";
    }
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    setOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file && !busy && !exhausted) check(file);
  }

  return (
    <div
      className={"drop" + (compact ? " compact" : "") + (over ? " over" : "") + (busy ? " busy" : "")}
      onDragOver={(e) => { e.preventDefault(); if (!busy && !exhausted) setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={onDrop}
    >
      <input ref={input} type="file" accept={ACCEPT} hidden onChange={(e) => e.target.files?.[0] && check(e.target.files[0])} />
      {stage.kind === "uploading" && (
        <div className="drop-state" role="status">
          <span className="spinner" aria-hidden="true" />
          <p className="drop-title">Uploading {stage.name}</p>
          <p className="sub">Storing the file and recording its fingerprint</p>
        </div>
      )}
      {stage.kind === "analysing" && (
        <div className="drop-state" role="status">
          <span className="spinner" aria-hidden="true" />
          <p className="drop-title">Checking {stage.name}</p>
          <p className="sub">{stage.video ? "Video is checked frame by frame and can take a few minutes" : "This usually takes about ten seconds"}</p>
        </div>
      )}
      {(stage.kind === "idle" || stage.kind === "error") && (
        <div className="drop-state">
          <p className="drop-title">{compact ? "Add a file to this folder" : "Drop a photo or video here"}</p>
          <p className="sub">JPEG, PNG, WebP, TIFF, MP4, MOV, MKV, WAV or MP3</p>
          <button className="btn primary" onClick={() => input.current?.click()} disabled={exhausted}>
            Choose a file
          </button>
          {exhausted && <p className="form-error">You've reached today's upload limit. It resets on a rolling 24 hours.</p>}
          {stage.kind === "error" && <p className="form-error" role="alert">{stage.message}</p>}
        </div>
      )}
    </div>
  );
}
