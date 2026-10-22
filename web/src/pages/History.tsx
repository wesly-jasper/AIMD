import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Empty, ErrorBlock, Skeleton, TitleRow } from "../components/bits";
import { MediaRow } from "../components/MediaRow";
import { Shell } from "../components/Shell";
import { count } from "../format";
import { useResource } from "../hooks";
import { toneOf } from "../plain";
import type { Folder, MediaItem, Page } from "../types";

type Filter = "all" | "flag" | "clear" | "unsure";
const FILTERS: { id: Filter; label: string }[] = [
  { id: "all", label: "All" },
  { id: "flag", label: "Signs found" },
  { id: "clear", label: "No signs" },
  { id: "unsure", label: "Unclear" },
];

export function History() {
  const media = useResource<Page<MediaItem>>("/media/?limit=200");
  const folders = useResource<Page<Folder>>("/cases/?limit=200");
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<Filter>("all");

  const names = useMemo(() => new Map((folders.data?.items ?? []).map((f) => [f.case_id, f.title])), [folders.data]);
  const items = media.data?.items ?? [];
  const toneFor = (m: MediaItem) => (m.latest_analysis ? toneOf(m.latest_analysis.assessment) : "unsure");
  // Counts follow the view: each tab counts the rows it would show.
  const counts = useMemo(() => ({
    all: items.length,
    flag: items.filter((m) => toneFor(m) === "flag").length,
    clear: items.filter((m) => toneFor(m) === "clear").length,
    unsure: items.filter((m) => toneFor(m) === "unsure").length,
  }), [items]);
  const shown = items.filter((m) =>
    (filter === "all" || toneFor(m) === filter) &&
    m.original_filename.toLowerCase().includes(query.trim().toLowerCase()));

  return (
    <Shell>
      <TitleRow
        title="History"
        sub={media.data ? `${count(items.length, "file")} checked` : "Every file you have checked"}
        actions={<Link className="btn primary" to="/">Check a file</Link>}
      />

      <div className="tabs" role="tablist" aria-label="Filter by result">
        {FILTERS.map((f) => (
          <button key={f.id} role="tab" aria-selected={filter === f.id} onClick={() => setFilter(f.id)}>
            {f.label} <span className="num muted">{counts[f.id]}</span>
          </button>
        ))}
      </div>

      <div className="toolbar">
        <input className="input search" type="search" placeholder="Search by file name" aria-label="Search by file name"
          value={query} onChange={(e) => setQuery(e.target.value)} />
        <span className="muted num">{shown.length === items.length ? count(items.length, "file") : `${shown.length} of ${items.length}`}</span>
      </div>

      {media.error && <ErrorBlock what="your files" error={media.error} onRetry={media.reload} />}

      <section className="card" aria-label="Files">
        <ul className="media-list">
          {media.loading && !media.data && [0, 1, 2, 3].map((i) => (
            <li key={i} className="media-row">
              <span className="thumb" aria-hidden="true" />
              <div className="media-id"><Skeleton w="40%" /><span className="sub"><Skeleton w="25%" h={12} /></span></div>
            </li>
          ))}
          {shown.map((m) => (
            <MediaRow key={m.media_id} item={m} folderName={m.case_id ? names.get(m.case_id) : null} onDeleted={media.reload} />
          ))}
        </ul>
        {media.data && shown.length === 0 && (
          <Empty>
            {items.length === 0
              ? <>You haven't checked any files yet. <Link className="text-link" to="/">Check your first file</Link></>
              : "No file matches this filter."}
          </Empty>
        )}
      </section>
    </Shell>
  );
}
