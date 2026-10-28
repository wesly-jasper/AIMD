import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { deleteMedia, postJson } from "../api";
import { AvailabilityMark, DeleteControl, ErrorBlock, ExecutionPill, Hash, Skeleton, TitleRow } from "../components/bits";
import { WhatWeChecked } from "../components/Coverage";
import { EvidenceView, regionsOf } from "../components/Evidence";
import { ContentCredentialsCard } from "../components/Credentials";
import { InvestigationCard } from "../components/Investigation";
import { BranchCards, EvidenceStrengthCard, MaskCard, ProvenanceCard, SourcesCard, VideoTimeline } from "../components/Panels";
import { ReportCard } from "../components/Report";
import { Shell } from "../components/Shell";
import { POLARITY, SEMANTICS, count, dateTime, duration, familyLabel, score } from "../format";
import { useResource } from "../hooks";
import { analysisVerdict, checkName } from "../plain";
import type { AnalysisDetail, Detection, MediaDetail, Observation } from "../types";

function ObservationList({ ids, byId, onPoint, empty }: {
  ids: string[]; byId: Map<string, Observation>; onPoint: (id: string | null) => void; empty: string;
}) {
  const rows = ids.map((id) => byId.get(id)).filter((o): o is Observation => Boolean(o));
  if (rows.length === 0) return <p className="muted">{empty}</p>;
  return (
    <ul className="obs-list">
      {rows.map((o) => (
        <li key={o.id} tabIndex={0} onMouseEnter={() => onPoint(o.detection_id)} onMouseLeave={() => onPoint(null)}
          onFocus={() => onPoint(o.detection_id)} onBlur={() => onPoint(null)}>
          <span className="check-name">{checkName(o.detector)}</span>
          <span className="sub">{familyLabel(o.signal_family)} · {POLARITY[o.polarity]}{o.localised ? " · localised" : ""}</span>
          <span className="obs-score num">{score(o.score, o.score_semantics)} <span className="sub-inline">{SEMANTICS[o.score_semantics]}</span></span>
        </li>
      ))}
    </ul>
  );
}

function DetectorTable({ detections }: { detections: Detection[] }) {
  return (
    <div className="table-wrap">
      <table className="detectors">
        <thead>
          <tr><th>Check</th><th>Capability</th><th>Run</th><th className="n">Score</th><th className="n">Areas</th><th>Note</th></tr>
        </thead>
        <tbody>
          {detections.map((d) => (
            <tr key={d.id}>
              <td><span className="check-name">{checkName(d.detector)}</span><span className="sub">{d.detector}{d.detector_version ? ` v${d.detector_version}` : ""}</span></td>
              <td className="cap"><AvailabilityMark value={d.availability} /></td>
              <td className="run"><ExecutionPill value={d.execution_status} /></td>
              <td className="n" data-label="Score"><span className="num">{score(d.confidence, d.score_semantics)}</span><span className="sub">{SEMANTICS[d.score_semantics]}</span></td>
              <td className="n num" data-label="Areas">{d.regions.length}</td>
              <td className="note" data-label="Note">{d.status_reason ?? <span className="dash">—</span>}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Result() {
  const { analysisId = "" } = useParams();
  const navigate = useNavigate();
  const a = useResource<AnalysisDetail>(`/analysis/${analysisId}`);
  const media = useResource<MediaDetail>(a.data ? `/media/${a.data.media_id}` : null);
  const [point, setPoint] = useState<string | null>(null);
  const [rerunning, setRerunning] = useState(false);
  const [rerunError, setRerunError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  // Bumped when an investigation finishes, so the report is fetched again.
  const [reportKey, setReportKey] = useState(0);

  const d = a.data;
  const byId = useMemo(() => new Map((d?.observations ?? []).map((o) => [o.id, o])), [d]);
  const regions = useMemo(() => regionsOf(
    d?.detections ?? [],
    new Set((d?.observations ?? []).filter((o) => o.polarity === "ANOMALY_OBSERVED").map((o) => o.detection_id)),
  ), [d]);
  const name = media.data?.original_filename ?? "this file";
  const failed = d?.status === "FAILED";
  const running = d?.status === "QUEUED" || d?.status === "PROCESSING";
  // Only a finished run has evidence to show; a failed or unfinished one would
  // fill every panel with "Unavailable".
  const finished = d != null && !failed && !running;
  const bumpReport = useCallback(() => setReportKey((k) => k + 1), []);

  // Re-read while the run is still going.
  const { reload } = a;
  useEffect(() => {
    if (!running) return;
    const t = window.setTimeout(reload, 2500);
    return () => window.clearTimeout(t);
  }, [running, d, reload]);

  if (a.error) {
    return (
      <Shell>
        <TitleRow title="Result" />
        <ErrorBlock what="this result" error={a.error} onRetry={a.reload} />
      </Shell>
    );
  }

  const v = analysisVerdict(d?.status, d?.assessment_detail?.category ?? d?.assessment, d?.evidence_strength);
  const flagged = regions.filter((r) => r.anomaly);
  const finding = d?.findings[0];
  const dims = media.data?.metadata.width && media.data.metadata.height
    ? `${media.data.metadata.width} × ${media.data.metadata.height} px` : null;

  async function rerun() {
    if (!d) return;
    setRerunning(true);
    setRerunError(null);
    try {
      const r = await postJson<{ analysis_id: string }>("/analysis/", { media_id: d.media_id });
      setRerunning(false);
      navigate(`/results/${r.analysis_id}`);
    } catch (err) {
      setRerunError(err instanceof Error ? err.message : "The check could not be run again.");
      setRerunning(false);
    }
  }

  return (
    <Shell>
      <nav className="crumbs" aria-label="Breadcrumb"><Link to="/history">History</Link><span aria-hidden="true"> / </span></nav>
      <TitleRow
        title={d ? name : "Loading result…"}
        sub={d ? (
          <>{failed ? "Check failed" : running ? "Started" : "Checked"} <span className="num">{dateTime(d.created_at)}</span>{dims ? <> · <span className="num">{dims}</span></> : null}{d.status === "PARTIAL" ? " · some steps did not finish" : ""}</>
        ) : undefined}
        actions={d && (
          <>
            <button className="btn" onClick={rerun} disabled={rerunning}>{rerunning ? "Checking…" : "Check again"}</button>
            <Link className="btn primary" to="/">Check another file</Link>
          </>
        )}
      />
      {rerunError && <p className="form-error" role="alert">{rerunError}</p>}

      <div className="grid">
        <section className="card span-7 media-card" aria-label="The file">
          {d ? (
            <EvidenceView mediaId={d.media_id} mediaType={d.media_type} regions={regions}
              highlight={point ? new Set([point]) : null} alt={`The checked file: ${name}`} videoRef={videoRef} />
          ) : <div className="evidence-frame loading" />}
          {finished && (
            <p className="media-caption">
              {flagged.length > 0 && <span className="area-mark" aria-hidden="true" />}
              <span>
                {flagged.length
                  ? `${count(flagged.length, "area")} marked where a check found signs`
                  : d.media_type === "image" ? "No area was marked" : "Areas are marked on images only"}
              </span>
            </p>
          )}
        </section>

        <section className={"card span-5 verdict-card " + v.tone} aria-labelledby="verdict-h">
          {!d ? <><Skeleton w="70%" h={26} /><p><Skeleton w="90%" /></p></> : (
            <>
              {finished && (d.verdict_state?.state ?? "ADVISORY") === "ADVISORY" && (
                <span className="pill advisory">Advisory · not yet independently validated</span>
              )}
              <h2 id="verdict-h" className="verdict">{v.headline}</h2>
              <p className="explanation">{v.explanation}</p>
              {finished && (
                <p className="advisory-note">
                  {d.verdict_state?.statement ?? "Experimental, advisory result. AIMD's verdicts have not yet passed independent evaluation, so this result supports an examiner's review and must not be relied on as a forensic conclusion on its own."}
                </p>
              )}
              {failed ? (
                <>
                  <div className="error-block run-failed" role="alert">
                    <span className="error-dot" aria-hidden="true" />
                    <span>{d.error_message ? <>Reason: {d.error_message}</> : "No reason was recorded."}</span>
                  </div>
                  <button className="btn primary" onClick={rerun} disabled={rerunning}>{rerunning ? "Checking…" : "Try again"}</button>
                </>
              ) : running ? (
                <div className="inv-loading" role="status">
                  <span className="spinner" aria-hidden="true" />
                  <span className="sub-inline">Running the checks…</span>
                </div>
              ) : (
                <WhatWeChecked detections={d.detections} observations={d.observations} onPoint={setPoint} />
              )}
            </>
          )}
        </section>
      </div>

      {finished && (
        <>
          <BranchCards d={d} />
          <EvidenceStrengthCard fusion={d.evidence_fusion} />
          {d.media_type === "image" ? (
            <div className="grid">
              <MaskCard analysisId={d.analysis_id} mediaId={d.media_id} detections={d.detections} observations={d.observations} />
              <div className="span-5"><ProvenanceCard d={d} /></div>
            </div>
          ) : (
            <>
              {d.media_type === "video" && <VideoTimeline detections={d.detections} videoRef={videoRef} />}
              <ProvenanceCard d={d} />
            </>
          )}
          <ContentCredentialsCard d={d} />
          <SourcesCard d={d} />
          <InvestigationCard analysisId={d.analysis_id} finished onFinished={bumpReport} />
          <ReportCard key={reportKey} analysisId={d.analysis_id} />
        </>
      )}

      {finished && (
        <details className="card technical">
          <summary>
            <span className="tech-title">Technical details</span>
            <span className="sub">The evidence behind this result, for anyone who needs to verify it</span>
          </summary>

          <div className="tech-body">
            <section aria-labelledby="finding-h">
              <h3 id="finding-h">Finding</h3>
              {finding ? (
                <>
                  <p className="statement">{finding.statement}</p>
                  <div className="links">
                    <div>
                      <h4>Supports <span className="num muted">{finding.supporting_observation_ids.length}</span></h4>
                      <ObservationList ids={finding.supporting_observation_ids} byId={byId} onPoint={setPoint} empty="None" />
                    </div>
                    <div>
                      <h4>Contradicts <span className="num muted">{finding.contradicting_observation_ids.length}</span></h4>
                      <ObservationList ids={finding.contradicting_observation_ids} byId={byId} onPoint={setPoint} empty="Nothing contradicts it" />
                    </div>
                  </div>
                </>
              ) : <p className="muted">No finding is stated for an inconclusive result.</p>}
              {d.assessment_detail && (
                <p className="sub rule-line">Decision rule <code>{d.assessment_detail.rule}</code>: {d.assessment_detail.rationale}</p>
              )}
            </section>

            <section aria-labelledby="checks-h">
              <h3 id="checks-h">Every check</h3>
              <p className="sub">Whether a check is installed and whether it ran are recorded separately: a check can be installed and still fail.</p>
              <DetectorTable detections={d.detections} />
            </section>

            <section aria-labelledby="run-h">
              <h3 id="run-h">This run</h3>
              <dl className="run-strip">
                <div><dt>Duration</dt><dd className="num">{duration(d.run.duration_ms)}</dd></div>
                <div><dt>Status</dt><dd>{d.status.charAt(0) + d.status.slice(1).toLowerCase()}</dd></div>
                <div><dt>Software</dt><dd className="num">{d.run.software_version ?? "—"}</dd></div>
                <div><dt>Check set</dt><dd><Hash value={d.run.detector_set_version} n={8} label="check set version" /></dd></div>
                <div><dt>File SHA-256</dt><dd><Hash value={media.data?.sha256} n={12} label="SHA-256" /></dd></div>
              </dl>
            </section>
          </div>
        </details>
      )}

      {d && media.data && (
        <div className="danger-zone">
          <DeleteControl
            name={name}
            onConfirm={async () => { await deleteMedia(d.media_id); navigate("/history"); }}
          />
          <span className="sub">Removes the file and all of its results from your account.</span>
        </div>
      )}
    </Shell>
  );
}
