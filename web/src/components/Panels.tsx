import { useEffect, useState, type RefObject } from "react";
import { useBlobUrl } from "../hooks";
import { SEMANTICS, count, dateTime, duration, score } from "../format";
import { checkName, outcomeOf, shortVerdict, unavailableReason } from "../plain";
import { DeclaredOrigin, credentialsSummary } from "./Credentials";
import type { AiGeneration, AnalysisDetail, Artefact, Detection, FrameScore, Observation, Segment, Source, SourceSearch, TimelineEvent, EvidenceFusion, FusionBranch, FusionContribution, FusionStatus } from "../types";

/** Not assessed, said in so many words: an absent result never reads as a clean one. */
export function Unavailable({ children }: { children: React.ReactNode }) {
  return (
    <div className="unavailable-block" role="status">
      <span className="hatch" aria-hidden="true" />
      <span><strong>Unavailable.</strong> {children}</span>
    </div>
  );
}

// ── The two branches ──────────────────────────────────────────────────────────

const AI_STATUS: Record<AiGeneration["status"], { label: string; tone: string }> = {
  AI_GENERATION_INDICATED: { label: "Indicated", tone: "warn" },
  NOT_INDICATED: { label: "Not indicated", tone: "" },
  CONFLICTING: { label: "Detectors disagree", tone: "warn" },
  INCONCLUSIVE: { label: "Inconclusive", tone: "hollow" },
  UNAVAILABLE: { label: "Unavailable", tone: "hollow" },
};

/** "Calibrated: 1.2% at a 5% FPR target, fitted on 40 + 60 validation images." */
function CalibrationNote({ ai }: { ai: AiGeneration }) {
  const c = ai.calibration;
  if (!c) return null;
  if (!c.applied) return <p className="sub">Not calibrated for this model version: {c.reason}</p>;
  const fitted = c.fitted_on ?? {};
  return (
    <p className="sub">
      Calibrated on {fitted.positives ?? "?"} generated and {fitted.negatives ?? "?"} other images
      ({fitted.split ?? "validation"} split); the threshold holds the false-positive rate at or
      under {c.target_fpr != null ? `${Math.round(c.target_fpr * 100)}%` : "its target"} on that set.
      Raw score {c.raw_score != null ? c.raw_score.toFixed(3) : "—"}.
    </p>
  );
}

function Limitations({ items }: { items: string[] }) {
  if (items.length === 0) return null;
  return (
    <div className="branch-limits">
      <h4>Limitations</h4>
      <ul>{items.map((l, i) => <li key={i}>{l}</li>)}</ul>
    </div>
  );
}

/** Detection ids in the AI-generation branch, so the manipulation card leaves them out. */
export function aiDetectionIds(d: AnalysisDetail): Set<string> {
  const ids = new Set(d.observations.filter((o) => o.signal_family === "ai_generation").map((o) => o.detection_id));
  const names = new Set([
    d.ai_generation?.detector,
    ...(d.ai_generation?.detectors ?? []).map((x) => x.detector),
  ].filter(Boolean));
  d.detections.filter((x) => names.has(x.detector)).forEach((x) => ids.add(x.id));
  return ids;
}

/**
 * Manipulation and AI generation, side by side and never merged: an edited
 * camera photograph and a wholly generated image are different findings.
 */
export function BranchCards({ d }: { d: AnalysisDetail }) {
  const ai = d.ai_generation;
  const aiIds = aiDetectionIds(d);
  const manip = d.detections.filter((x) => !aiIds.has(x.id));
  const ran = manip.filter((x) => x.execution_status === "COMPLETED" || x.execution_status === "INCONCLUSIVE");
  const missing = manip.filter((x) => !ran.includes(x));
  const detail = d.assessment_detail;
  // coverage_gaps can hold detection ids; those checks are listed by name from
  // ``missing`` below, so an id is never shown as text.
  const ids = new Set(d.detections.map((x) => x.id));
  const manipLimits = [
    ...(detail?.coverage_gaps ?? []).filter((g) => !ids.has(g)),
    ...missing.map((x) => `${checkName(x.detector)}: ${unavailableReason(x)}`),
  ];

  return (
    <div className="grid branches">
      <section className="card span-6 branch" aria-labelledby="manip-h">
        <div className="card-head">
          <h2 id="manip-h">Manipulation</h2>
          {detail
            ? <span className="pill">{shortVerdict(detail.category)}</span>
            : <span className="pill hollow">Unavailable</span>}
        </div>
        <p className="sub">Was this file edited after capture? Decided by the forensic checks below.</p>
        {detail ? (
          <>
            <p className="branch-rationale">{detail.rationale}</p>
            <dl className="branch-facts">
              <div><dt>Evidence strength</dt><dd>{detail.evidence_strength.toLowerCase()}</dd></div>
              <div><dt>Decision rule</dt><dd><code>{detail.rule}</code></dd></div>
              <div><dt>Checks that ran</dt><dd className="num">{ran.length} of {manip.length}</dd></div>
            </dl>
            <h4 className="branch-sub">Score semantics</h4>
            <ul className="semantics-list">
              {ran.map((x) => (
                <li key={x.id}>
                  <span>{checkName(x.detector)}</span>
                  <span className="num">{score(x.confidence, x.score_semantics)}</span>
                  <span className="sub-inline">{SEMANTICS[x.score_semantics]}</span>
                </li>
              ))}
              {ran.length === 0 && <li className="muted">No manipulation check produced a score.</li>}
            </ul>
            <p className="sub">Scores from different checks are on different scales and are never added together.</p>
          </>
        ) : <Unavailable>This analysis has no recorded manipulation assessment.</Unavailable>}
        <Limitations items={manipLimits} />
      </section>

      <section className="card span-6 branch" aria-labelledby="ai-h">
        <div className="card-head">
          <h2 id="ai-h">AI generation</h2>
          <span className={"pill " + (ai ? AI_STATUS[ai.status].tone : "hollow")}>
            {ai?.reason === "image_only" ? "Images only" : ai ? AI_STATUS[ai.status].label : "Unavailable"}
          </span>
        </div>
        <p className="sub">Was this {d.media_type === "video" ? "video" : "image"} generated by a model? Its own question, its own detector.</p>
        <DeclaredOrigin c={d.content_credentials} />
        {!ai ? (
          <Unavailable>This analysis was recorded before the AI-generation check existed.</Unavailable>
        ) : ai.status === "UNAVAILABLE" ? (
          <Unavailable>{ai.rationale}{ai.reason && ai.reason !== "image_only" ? ` Reason: ${ai.reason}` : ""}</Unavailable>
        ) : ai.status === "INCONCLUSIVE" ? (
          <p className="branch-rationale">{ai.rationale}</p>
        ) : (
          <>
            <p className="branch-rationale">{ai.rationale}</p>
            <dl className="branch-facts">
              <div><dt>Score</dt><dd className="num">{ai.score != null ? ai.score.toFixed(3) : "—"}</dd></div>
              <div><dt>Threshold</dt><dd className="num">{ai.threshold ?? "—"}</dd></div>
              <div><dt>Detector</dt><dd>{ai.detector}{ai.detector_version ? ` v${ai.detector_version}` : ""}</dd></div>
            </dl>
            {ai.score != null && ai.threshold != null && (
              <div className="meter" role="img" aria-label={`Score ${ai.score.toFixed(3)} against threshold ${ai.threshold}`}>
                <span className="meter-fill" style={{ width: `${Math.min(100, Math.max(0, ai.score * 100))}%` }} />
                <span className="meter-mark" style={{ left: `${Math.min(100, Math.max(0, ai.threshold * 100))}%` }} />
              </div>
            )}
            <h4 className="branch-sub">Score semantics</h4>
            <p className="sub">
              {ai.score_semantics ? SEMANTICS[ai.score_semantics] : "not stated"}
              {ai.semantics ? ` — ${ai.semantics}` : ""}
            </p>
            <CalibrationNote ai={ai} />
          </>
        )}
        {ai?.detectors && ai.detectors.length > 1 && (
          <>
            <h4 className="branch-sub">Each detector</h4>
            <ul className="semantics-list">
              {ai.detectors.map((x) => (
                <li key={x.detector}>
                  <span>{checkName(x.detector)}</span>
                  <span className="num">{x.score != null ? score(x.score, x.score_semantics) : "—"}</span>
                  <span className="sub-inline">{AI_STATUS[x.status].label}{x.reason ? ` — ${x.reason}` : ""}{x.voting === false ? " · for reference, does not count" : ""}</span>
                </li>
              ))}
            </ul>
          </>
        )}
        <Limitations items={ai?.limitations ?? []} />
      </section>
    </div>
  );
}

// ── Strength of evidence ──────────────────────────────────────────────────────

const FUSION_LABEL: Record<FusionStatus, string> = {
  SUPPORTS: "Supports",
  AGAINST: "Points against",
  LIMITED: "Limited either way",
  CONFLICTING: "Checks disagree",
  NOT_ASSESSED: "Not quantified",
};

function lrText(v: number): string {
  const x = 10 ** Math.abs(v);
  return (v < 0 ? "1/" : "") + (x >= 100 ? Math.round(x).toLocaleString() : x.toPrecision(2));
}

function FusionPart({ title, part }: { title: string; part: FusionBranch }) {
  const range = part.log10_lr_range;
  const direction = (c: FusionContribution) =>
    c.direction == null ? "" : c.direction > 0 ? "points towards" : "points against";
  return (
    <div className="fusion-part">
      <div className="card-head">
        <h3>{title}</h3>
        <span className={`pill${part.status === "NOT_ASSESSED" ? " hollow" : ""}`}>{FUSION_LABEL[part.status]}</span>
      </div>
      {range && (
        <p className="num">
          Likelihood ratio {lrText(range[0])}
          {lrText(range[1]) !== lrText(range[0]) ? ` to ${lrText(range[1])}` : ""}
          {part.strength ? ` · ${part.strength}` : ""}
        </p>
      )}
      <p className="sub">{part.statement}</p>
      {part.conflicts.length > 0 && (
        <ul className="semantics-list">
          {part.conflicts.map((c) => (
            <li key={`${c.for}-${c.against}`}>
              <span>Disagreement</span>
              <span className="sub-inline">{checkName(c.for)} points towards, {checkName(c.against)} against</span>
            </li>
          ))}
        </ul>
      )}
      <ul className="semantics-list">
        {part.contributions.filter((c) => c.status === "observed").map((c) => (
          <li key={c.detector} title={c.calibrated === false && c.reason ? c.reason : undefined}>
            <span>{checkName(c.detector)}</span>
            <span className="num">{c.log10_lr != null ? `LR ${lrText(c.log10_lr)}${c.capped ? " (capped)" : ""}` : "—"}</span>
            <span className="sub-inline">
              {direction(c)}
              {c.calibrated === false ? " · not calibrated, direction only" : ""}
              {c.advisory ? " · for reference, does not count" : ""}
              {c.calibrated && c.counted === false ? " · same family as a stronger check, not counted again" : ""}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Likelihood-ratio ranges for both questions, beside the verdicts and never changing them. */
export function EvidenceStrengthCard({ fusion }: { fusion: EvidenceFusion | null | undefined }) {
  if (!fusion) return null;
  return (
    <section className="card" aria-labelledby="fusion-h">
      <div className="card-head"><h2 id="fusion-h">Strength of evidence</h2></div>
      <p className="sub">
        How much more probable the check results are if the answer is yes than if it is no. Shown as a range:
        from treating the kinds of check as fully dependent to treating them as independent. This does not
        change the verdicts above.
      </p>
      {fusion.calibration && (fusion.calibration.state === "missing" || fusion.calibration.state === "unreadable"
        || fusion.calibration.checksum === "mismatch") && (
        <p className="sub" role="alert">
          The configured calibration file is {fusion.calibration.checksum === "mismatch" ? "not the recorded version" : fusion.calibration.state},
          so no check is given a number here.
        </p>
      )}
      <div className="grid">
        <div className="span-6"><FusionPart title="Edited after capture?" part={fusion.manipulation} /></div>
        <div className="span-6"><FusionPart title="Generated by an AI model?" part={fusion.ai_generation} /></div>
      </div>
    </section>
  );
}

// ── Manipulation mask ─────────────────────────────────────────────────────────

function artefactsOf(d: Detection): Record<string, Artefact> {
  const a = d.metadata?.artefacts;
  return a && typeof a === "object" ? (a as Record<string, Artefact>) : {};
}

export function maskDetection(detections: Detection[]): Detection | undefined {
  return detections.find((d) => artefactsOf(d).mask || artefactsOf(d).localization_map)
    ?? detections.find((d) => d.detector === "TruForDetector" || "artefacts_error" in (d.metadata ?? {}));
}

interface MaskLayers { fill: string | null; edge: string | null; fraction: number | null; error: boolean }

/**
 * Pixels at or above `cut` (0-255) become the mask: a flat amber fill and a solid
 * outline, as two layers so the fill can be faded while the outline stays. The
 * outline is about 1/300 of the long side thick, so it survives being shown small.
 */
function useMaskLayers(src: string | null, cut: number): MaskLayers {
  const [out, setOut] = useState<MaskLayers>({ fill: null, edge: null, fraction: null, error: false });
  useEffect(() => {
    if (!src) return;
    let live = true;
    const made: string[] = [];
    const img = new Image();
    img.onload = () => {
      const w = img.naturalWidth, h = img.naturalHeight;
      const c = document.createElement("canvas");
      c.width = w;
      c.height = h;
      const ctx = c.getContext("2d", { willReadFrequently: true });
      if (!ctx) { if (live) setOut((o) => ({ ...o, error: true })); return; }
      ctx.drawImage(img, 0, 0);
      const src8 = ctx.getImageData(0, 0, w, h).data;
      const m = new Uint8Array(w * h);
      let on = 0;
      for (let i = 0; i < m.length; i++) if (src8[i * 4] >= cut) { m[i] = 1; on++; }

      // Boundary pixels, then widened by t with two running-window passes.
      const edge = new Uint8Array(w * h);
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const i = y * w + x;
        if (m[i] && (x === 0 || y === 0 || x === w - 1 || y === h - 1
          || !m[i - 1] || !m[i + 1] || !m[i - w] || !m[i + w])) edge[i] = 1;
      }
      const t = Math.max(1, Math.round(Math.max(w, h) / 300));
      const wide = new Uint8Array(w * h);
      for (let y = 0; y < h; y++) {
        let n = 0;
        for (let x = 0; x < Math.min(t, w); x++) n += edge[y * w + x];
        for (let x = 0; x < w; x++) {
          if (x + t < w) n += edge[y * w + x + t];
          if (x - t - 1 >= 0) n -= edge[y * w + x - t - 1];
          wide[y * w + x] = n > 0 ? 1 : 0;
        }
      }
      for (let x = 0; x < w; x++) {
        let n = 0;
        for (let y = 0; y < Math.min(t, h); y++) n += wide[y * w + x];
        for (let y = 0; y < h; y++) {
          if (y + t < h) n += wide[(y + t) * w + x];
          if (y - t - 1 >= 0) n -= wide[(y - t - 1) * w + x];
          edge[y * w + x] = n > 0 ? 1 : 0;
        }
      }

      const paint = (on: Uint8Array): Promise<string | null> => {
        const px = ctx.createImageData(w, h);
        const p = px.data;
        for (let i = 0; i < on.length; i++) {
          if (!on[i]) continue;
          p[i * 4] = 242; p[i * 4 + 1] = 179; p[i * 4 + 2] = 61; p[i * 4 + 3] = 255;
        }
        ctx.putImageData(px, 0, 0);
        return new Promise((resolve) => c.toBlob((b) => resolve(b ? URL.createObjectURL(b) : null), "image/png"));
      };
      (async () => {
        const fill = await paint(m);
        const edgeUrl = await paint(edge);
        for (const u of [fill, edgeUrl]) if (u) made.push(u);
        if (!live) return;
        setOut({ fill, edge: edgeUrl, fraction: on / m.length, error: !fill || !edgeUrl });
      })();
    };
    img.onerror = () => { if (live) setOut((o) => ({ ...o, error: true })); };
    img.src = src;
    return () => { live = false; made.forEach((u) => URL.revokeObjectURL(u)); };
  }, [src, cut]);
  return out;
}

function pct(f: number): string {
  if (f === 0) return "0%";
  if (f < 0.001) return "under 0.1%";
  return `${(f * 100).toFixed(f < 0.1 ? 1 : 0)}%`;
}

/**
 * Where the manipulation-localisation model marked the image: its map cut at the
 * recorded threshold. Manipulation only: AI generation is a separate question and
 * a generated image need not be masked anywhere.
 */
export function MaskCard({ analysisId, mediaId, detections, observations }: {
  analysisId: string; mediaId: string; detections: Detection[]; observations: Observation[];
}) {
  const det = maskDetection(detections);
  // The model's own image-level call. A map can pass the pixel cut in places
  // on an image the model did not flag; that is shown on request, labelled as
  // a local response, and never presented as a marked area.
  const flagged = det ? outcomeOf(det, observations) === "found" : false;
  const art = det ? artefactsOf(det) : {};
  // The stored mask when the run wrote one. Runs from before masks were stored
  // only have the localisation map; it is cut here at the runner's threshold,
  // without the speckle removal the stored mask has, and the card says so.
  const stored = art.mask;
  const kind = stored ? "mask" : art.localization_map ? "localization_map" : null;
  const threshold = stored?.threshold ?? (typeof det?.metadata?.threshold === "number" ? det.metadata.threshold as number : 0.5);
  const filePath = det && kind ? `/analysis/${analysisId}/detections/${det.id}/artefacts/${kind}` : null;
  const original = useBlobUrl(filePath ? `/media/${mediaId}/download` : null);
  const file = useBlobUrl(filePath);
  const layers = useMaskLayers(file.url, stored ? 128 : Math.round(threshold * 255));
  const [fill, setFill] = useState(55);
  const [view, setView] = useState<"mask" | "original" | null>(null);

  const fraction = stored?.masked_fraction ?? layers.fraction;
  const empty = fraction === 0;
  let body: React.ReactNode;
  if (!det) {
    body = <Unavailable>No manipulation-localisation model ran on this image, so nothing can be masked.</Unavailable>;
  } else if (det.execution_status !== "COMPLETED") {
    body = <Unavailable>{checkName(det.detector)}: {unavailableReason(det)}.</Unavailable>;
  } else if (!filePath) {
    const err = det.metadata?.artefacts_error;
    body = <Unavailable>{err ? `The mask could not be saved during analysis (${String(err)}).` : "This run recorded no mask."}</Unavailable>;
  } else if (file.error || layers.error || original.error) {
    body = <Unavailable>The mask file could not be retrieved: it is missing from storage or no longer matches the hash recorded at analysis.</Unavailable>;
  } else if (!original.url || !layers.fill || !layers.edge) {
    body = <div className="evidence-frame loading" aria-busy="true" />;
  } else {
    const shown = (view ?? (flagged ? "mask" : "original")) === "mask" && !empty;
    body = (
      <>
        <div className="evidence-frame">
          <span className="evidence-inner">
            <img src={original.url} alt={shown ? "The checked image with the masked areas covered" : "The checked image"} />
            <img className="mask-layer" src={layers.fill} alt="" aria-hidden="true"
              style={{ opacity: shown ? fill / 100 : 0 }} />
            <img className="mask-layer" src={layers.edge} alt="" aria-hidden="true"
              style={{ opacity: shown ? 1 : 0 }} />
          </span>
        </div>
        <div className="mask-controls">
          <div className="segmented compact" role="tablist" aria-label="Before and after">
            <button role="tab" aria-selected={!shown} onClick={() => setView("original")}>Original</button>
            <button role="tab" aria-selected={shown} onClick={() => setView("mask")} disabled={empty}>Masked</button>
          </div>
          <label className="slider">
            <span className="sub-inline">Fill</span>
            <input type="range" min={0} max={100} value={fill} disabled={!shown}
              onChange={(e) => setFill(Number(e.target.value))} aria-valuetext={`${fill}%`} />
            <span className="num sub-inline">{fill}%</span>
          </label>
        </div>
        <p className="mask-summary">
          {empty
            ? <>No area reached the threshold, so nothing is masked. That is not a finding that the image is unedited.</>
            : !flagged
            ? <>
                {checkName(det.detector)} did not flag this image.{" "}
                {stored?.areas != null ? count(stored.areas, "small area") : "Part"} of its map still reached the threshold
                ({fraction != null ? pct(fraction) : "—"} of the image): a local response below the model’s own decision,
                not a marked area. Choose Masked to see it.
              </>
            : <>
                <span className="mask-swatch" aria-hidden="true" />
                <span>
                  {stored?.areas != null ? <>{count(stored.areas, "area")} masked, </> : null}
                  covering <span className="num">{fraction != null ? pct(fraction) : "—"}</span> of the image.
                </span>
              </>}
        </p>
        <p className="sub">
          Masked where {checkName(det.detector)}’s per-pixel localisation output ({SEMANTICS[det.score_semantics]}) is
          at least <span className="num">{threshold.toFixed(2)}</span>.{" "}
          {stored ? "Specks smaller than 0.1% of the image are left out, so every masked area is one of the areas listed for this check." :
            "This run predates stored masks: the mask is cut from its localisation map in the browser, without speckle removal."}{" "}
          A mask shows where the model responded, not proof of editing, and it says nothing about AI generation.
        </p>
      </>
    );
  }

  return (
    <section className="card span-7" aria-labelledby="mask-h">
      <div className="card-head"><h2 id="mask-h">Manipulation mask</h2></div>
      {body}
    </section>
  );
}

// ── Video keyframes ───────────────────────────────────────────────────────────

export function VideoTimeline({ detections, videoRef }: {
  detections: Detection[]; videoRef: RefObject<HTMLVideoElement | null>;
}) {
  const det = detections.find((d) => Array.isArray(d.metadata?.frame_scores));
  const frames = (det?.metadata?.frame_scores as FrameScore[] | undefined) ?? [];
  const segments = (det?.metadata?.suspicious_segments as Segment[] | undefined) ?? [];
  const seek = (t: number) => {
    const v = videoRef.current;
    if (v) { v.currentTime = t; v.scrollIntoView({ block: "nearest", behavior: "smooth" }); }
  };

  let body: React.ReactNode;
  if (!det) {
    const failed = detections.find((d) => d.media_type === "video");
    body = <Unavailable>{failed ? `${checkName(failed.detector)}: ${unavailableReason(failed)}.` : "No frame-level check ran on this video."}</Unavailable>;
  } else if (frames.length === 0) {
    body = <Unavailable>The frame check ran but scored no frames.</Unavailable>;
  } else {
    const end = Math.max(...frames.map((f) => f.timestamp_seconds), ...segments.map((s) => s.end_timestamp), 0.001);
    const peak = Math.max(...frames.map((f) => f.score), 0.001);
    const W = 1000, H = 120;
    const x = (t: number) => (t / end) * W;
    body = (
      <>
        <svg className="frame-chart" viewBox={`0 0 ${W} ${H + 18}`} preserveAspectRatio="none" role="img"
          aria-label={`${frames.length} sampled frames over ${end.toFixed(1)} s; ${segments.length} suspicious segments`}>
          <line x1={0} x2={W} y1={H} y2={H} className="axis" />
          {segments.map((s, i) => (
            <rect key={i} className="seg-band" x={x(s.start_timestamp)} y={0}
              width={Math.max(4, x(s.end_timestamp) - x(s.start_timestamp))} height={H} />
          ))}
          {frames.map((f) => {
            const h = (f.score / peak) * (H - 6);
            const hot = segments.some((s) => f.frame_index >= s.start_frame && f.frame_index <= s.end_frame);
            return (
              <rect key={f.frame_index} className={"frame-bar" + (hot ? " hot" : "")}
                x={x(f.timestamp_seconds) - 2} y={H - h} width={4} height={Math.max(1, h)}
                onClick={() => seek(f.timestamp_seconds)}>
                <title>{`Frame ${f.frame_index} at ${f.timestamp_seconds.toFixed(2)} s: ${f.score.toFixed(3)}`}</title>
              </rect>
            );
          })}
        </svg>
        <div className="chart-axis sub-inline"><span className="num">0 s</span><span className="num">{end.toFixed(1)} s</span></div>
        <p className="sub">
          {frames.length} sampled keyframes; bar height is each frame’s combined check score ({SEMANTICS[det.score_semantics]},
          hand-weighted, not calibrated), scaled to the peak of <span className="num">{peak.toFixed(3)}</span>. Select a bar to play from that frame.
        </p>
        {segments.length > 0 ? (
          <ul className="segment-list">
            {segments.map((s, i) => (
              <li key={i}>
                <span className="area-mark" aria-hidden="true" />
                <span className="num">{s.start_timestamp.toFixed(2)}–{s.end_timestamp.toFixed(2)} s</span>
                <span className="sub-inline">frames {s.start_frame}–{s.end_frame} · peak <span className="num">{s.peak_score.toFixed(3)}</span></span>
                <button className="text-btn" onClick={() => seek(s.start_timestamp)}>Play from here</button>
              </li>
            ))}
          </ul>
        ) : <p className="muted">No segment of consecutive frames crossed the suspicion threshold.</p>}
      </>
    );
  }

  return (
    <section className="card" aria-labelledby="frames-h">
      <div className="card-head"><h2 id="frames-h">Keyframe timeline</h2></div>
      {body}
    </section>
  );
}

// ── Sources ───────────────────────────────────────────────────────────────────

const VERIFICATION: Record<string, { label: string; tone: string; what: string }> = {
  VERIFIED_MATCH: { label: "Verified match", tone: "ok", what: "Fetched and compared: the same image" },
  SIMILAR_UNVERIFIED: { label: "Similar, unverified", tone: "warn", what: "Close in one measure, not confirmed by the others" },
  NO_MATCH: { label: "Not a match", tone: "hollow", what: "Compared and rejected" },
};

// Keyed by connector (search strip) and by hit provider (source cards).
const PROVIDER: Record<string, string> = {
  // A local corpus of pre-fetched Commons images, not a search of Commons.
  wikimedia_commons: "Commons reference set",
  wikimedia_commons_live: "Wikimedia Commons (exact copies)",
  wayback: "Internet Archive",
  url_verification: "URL you supplied",
  user_supplied_url: "URL you supplied",
};

/**
 * A search that did not complete, in the reader's terms. The backend's reason
 * is written for whoever runs the server ("set WIKIMEDIA_CONTACT ..."), so it
 * stays in the tooltip rather than the sentence.
 */
function searchNote(name: string, s: SourceSearch): string {
  if (s.status === "UNAVAILABLE" && (name === "url_verification" || name === "user_supplied_url")) {
    return "not used: no link was given with this check";
  }
  if (s.status === "UNAVAILABLE") return "not set up on this server";
  if (s.status === "FAILED") return "could not be reached this time";
  return s.status.toLowerCase();
}

function SearchStrip({ searches }: { searches: Record<string, SourceSearch> }) {
  return (
    <ul className="search-strip">
      {Object.entries(searches).map(([name, s]) => (
        <li key={name} title={s.status !== "COMPLETED" && s.reason ? s.reason : undefined}>
          <span className={"state " + (s.status === "COMPLETED" ? "on" : "off")}>{PROVIDER[name] ?? name}</span>
          <span className="sub-inline">
            {s.status === "COMPLETED"
              ? <><span className="num">{s.hits}</span> found · <span className="num">{s.verified}</span> verified{s.duration_ms != null ? ` · ${duration(s.duration_ms)}` : ""}</>
              : searchNote(name, s)}
          </span>
        </li>
      ))}
    </ul>
  );
}

function safeHref(url: unknown): string | null {
  return typeof url === "string" && /^https?:\/\//i.test(url) ? url : null;
}

function SourceCard({ s }: { s: Source }) {
  const v = VERIFICATION[s.verification_status] ?? { label: s.verification_status, tone: "hollow", what: "" };
  const href = safeHref(s.url);
  const cited = Object.entries(s.citation ?? {}).filter(([, val]) => typeof val === "string" || typeof val === "number").slice(0, 4);
  return (
    <li className="source-card">
      <div className="source-badges">
        <span className={"pill " + v.tone} title={v.what}>{v.label}</span>
        {s.is_earliest_known && <span className="pill earliest">Earliest known</span>}
      </div>
      <p className="source-title">{s.title || s.domain || "Untitled source"}</p>
      <p className="sub">{PROVIDER[s.provider ?? ""] ?? s.provider ?? "—"}{s.published_at ? <> · published <span className="num">{dateTime(s.published_at)}</span></> : null}</p>
      {s.similarity != null && <p className="sub">Similarity <span className="num">{s.similarity.toFixed(3)}</span></p>}
      {cited.length > 0 && (
        <dl className="citation">
          {cited.map(([k, val]) => (
            <div key={k}><dt>{k.replace(/_/g, " ")}</dt><dd>{safeHref(val) ? <a className="text-link" href={String(val)} target="_blank" rel="noopener noreferrer nofollow">{String(val)}</a> : String(val)}</dd></div>
          ))}
        </dl>
      )}
      {href && <a className="text-link source-link" href={href} target="_blank" rel="noopener noreferrer nofollow">{s.domain ?? "Open source"} ↗</a>}
    </li>
  );
}

export function SourcesCard({ d }: { d: AnalysisDetail }) {
  const searches = d.source_searches;
  const completed = searches ? Object.values(searches).filter((s) => s.status === "COMPLETED").length : 0;
  const sorted = [...d.sources].sort((a, b) =>
    Number(b.verification_status === "VERIFIED_MATCH") - Number(a.verification_status === "VERIFIED_MATCH")
    || Number(b.is_earliest_known) - Number(a.is_earliest_known));
  const local = d.similarity_matches.filter((m) => m.match_type === "local");

  let body: React.ReactNode;
  if (!searches) {
    body = <Unavailable>Source search did not run for this analysis.</Unavailable>;
  } else if (sorted.length === 0) {
    body = completed === 0
      ? <Unavailable>No source search completed, so an empty list says nothing about where the image has appeared.</Unavailable>
      : <p className="empty">Not found in the {completed === 1 ? "index" : "indexes"} searched. That is not evidence the image is new: most of the web is not indexed here.</p>;
  } else {
    body = <ul className="source-grid">{sorted.map((s) => <SourceCard key={s.id} s={s} />)}</ul>;
  }

  return (
    <section className="card" aria-labelledby="sources-h">
      <div className="card-head">
        <h2 id="sources-h">Sources</h2>
        <span className="sub-inline num">{d.sources.length} found</span>
      </div>
      {searches && <SearchStrip searches={searches} />}
      {body}
      {local.length > 0 && (
        <p className="sub local-matches">
          Also similar to <span className="num">{local.length}</span> of your own earlier uploads
          ({local.filter((m) => m.verification_status === "VERIFIED_MATCH").length} verified).
        </p>
      )}
    </section>
  );
}

// ── Provenance ────────────────────────────────────────────────────────────────

const EVENT: Record<string, string> = {
  PUBLISHED: "Published",
  ARCHIVED: "Archived",
  RETRIEVED: "Retrieved by AIMD",
  METADATA: "Date in metadata",
};

function EventRow({ e }: { e: TimelineEvent }) {
  const page = safeHref(e.citation?.page_url ?? e.citation?.url);
  return (
    <li className={"event" + (e.earliest_known_appearance ? " earliest" : "") + (e.type === "METADATA" ? " claim" : "")}>
      <span className="event-dot" aria-hidden="true" />
      <div>
        <p className="event-head">
          <span className="num">{e.precision === "day" ? e.timestamp.slice(0, 10) : dateTime(e.timestamp)}</span>
          <span className="event-type">{EVENT[e.type] ?? e.type}</span>
          {e.earliest_known_appearance && <span className="pill earliest">Earliest known appearance</span>}
        </p>
        <p className="event-desc">{e.description}</p>
        {(e.basis || page) && (
          <p className="sub">
            {e.basis}
            {page && <> · <a className="text-link" href={page} target="_blank" rel="noopener noreferrer nofollow">cited page ↗</a></>}
          </p>
        )}
      </div>
    </li>
  );
}

export function ProvenanceCard({ d }: { d: AnalysisDetail }) {
  const t = d.provenance_timeline;
  // The full record is the Content Credentials card; here, one line pointing to it.
  const c = credentialsSummary(d.content_credentials);
  return (
    <section className="card" aria-labelledby="prov-h">
      <div className="card-head"><h2 id="prov-h">Provenance timeline</h2></div>
      <p className="content-creds">
        <span className="sub-inline">Content Credentials</span>{" "}
        <a className={"state text-link " + (c.tone === "ok" ? "on" : "off")} href="#creds-h"
          onClick={(e) => { e.preventDefault(); document.getElementById("creds-h")?.scrollIntoView({ behavior: "smooth", block: "start" }); }}>
          {c.label}
        </a>
      </p>
      {t.events.length === 0
        ? <p className="empty">No dated events: no verified source gave a date and the file’s metadata holds none.</p>
        : <ol className="timeline">{t.events.map((e, i) => <EventRow key={i} e={e} />)}</ol>}
      {t.note && <p className="sub timeline-note">{t.note}</p>}
    </section>
  );
}
