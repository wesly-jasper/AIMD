import { Link } from "react-router-dom";
import { Empty, ErrorBlock, Skeleton, TitleRow } from "../components/bits";
import { Dropzone } from "../components/Dropzone";
import { MediaRow } from "../components/MediaRow";
import { Shell } from "../components/Shell";
import { count } from "../format";
import { useResource } from "../hooks";
import type { MediaItem, Page, Usage } from "../types";

export function Home() {
  const recent = useResource<Page<MediaItem>>("/media/?limit=5");
  const usage = useResource<Usage>("/media/usage");
  const remaining = usage.data ? Math.max(0, usage.data.limit - usage.data.used) : null;
  const refresh = () => { recent.reload(); usage.reload(); };

  return (
    <Shell>
      <TitleRow
        title="Check media"
        sub={usage.data
          ? <>{count(usage.data.used, "upload")} of {usage.data.limit} used in the last 24 hours</>
          : "Find out whether a photo or video shows signs of editing"}
      />

      <Dropzone remaining={remaining} onDone={refresh} />

      <section className="card recent" aria-labelledby="recent-h">
        <header className="card-head">
          <h2 id="recent-h">Recent checks</h2>
          <Link className="text-link" to="/history">All files</Link>
        </header>
        {recent.error && <ErrorBlock what="your recent checks" error={recent.error} onRetry={recent.reload} />}
        <ul className="media-list">
          {recent.loading && !recent.data && [0, 1, 2].map((i) => (
            <li key={i} className="media-row">
              <span className="thumb" aria-hidden="true" />
              <div className="media-id"><Skeleton w="40%" /><span className="sub"><Skeleton w="25%" h={12} /></span></div>
            </li>
          ))}
          {recent.data?.items.map((m) => <MediaRow key={m.media_id} item={m} onDeleted={refresh} />)}
        </ul>
        {recent.data && recent.data.items.length === 0 && (
          <Empty>Nothing checked yet. Your results will be listed here.</Empty>
        )}
      </section>
    </Shell>
  );
}
