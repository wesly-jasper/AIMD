import { useState, type FormEvent } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { postJson } from "../api";
import { Empty, ErrorBlock, Skeleton, TitleRow } from "../components/bits";
import { Dropzone } from "../components/Dropzone";
import { MediaRow } from "../components/MediaRow";
import { Shell } from "../components/Shell";
import { count, dateOnly } from "../format";
import { useResource } from "../hooks";
import type { Folder, MediaItem, Page } from "../types";

function NewFolder({ onCreated }: { onCreated: (id: string) => void }) {
  const [title, setTitle] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const f = await postJson<Folder>("/cases/", { title: title.trim() });
      onCreated(f.case_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "The folder could not be created.");
      setBusy(false);
    }
  }

  return (
    <form className="new-folder" onSubmit={submit}>
      <label htmlFor="folder-name" className="visually-hidden">Folder name</label>
      <input id="folder-name" className="input" placeholder="New folder name" maxLength={300}
        value={title} onChange={(e) => setTitle(e.target.value)} />
      <button className="btn" disabled={busy || !title.trim()}>{busy ? "Creating…" : "Create folder"}</button>
      {error && <span className="form-error">{error}</span>}
    </form>
  );
}

export function Folders() {
  const navigate = useNavigate();
  const folders = useResource<Page<Folder>>("/cases/?limit=200");
  const items = folders.data?.items ?? [];

  return (
    <Shell>
      <TitleRow
        title="Folders"
        sub="Group related files, such as every version of one photo"
        actions={<NewFolder onCreated={(id) => navigate(`/folders/${id}`)} />}
      />
      {folders.error && <ErrorBlock what="your folders" error={folders.error} onRetry={folders.reload} />}
      <section className="card" aria-label="Folders">
        <ul className="folder-list">
          {folders.loading && !folders.data && [0, 1].map((i) => <li key={i}><Skeleton w="30%" /></li>)}
          {items.map((f) => (
            <li key={f.case_id}>
              <Link className="folder-link" to={`/folders/${f.case_id}`}>
                <span className="check-name">{f.title}</span>
                <span className="sub">Created <span className="num">{dateOnly(f.created_at)}</span></span>
              </Link>
              <span className="num folder-count">{count(f.evidence_count ?? 0, "file")}</span>
            </li>
          ))}
        </ul>
        {folders.data && items.length === 0 && (
          <Empty>No folders yet. Folders are optional: every file you check is also listed in History.</Empty>
        )}
      </section>
    </Shell>
  );
}

export function FolderDetail() {
  const { folderId = "" } = useParams();
  const folder = useResource<Folder>(`/cases/${folderId}`);
  const media = useResource<Page<MediaItem>>(`/media/?case_id=${folderId}&limit=200`);
  const items = media.data?.items ?? [];

  if (folder.error) {
    return (
      <Shell>
        <TitleRow title="Folder" />
        <ErrorBlock what="this folder" error={folder.error} onRetry={folder.reload} />
      </Shell>
    );
  }

  return (
    <Shell>
      <nav className="crumbs" aria-label="Breadcrumb"><Link to="/folders">Folders</Link><span aria-hidden="true"> / </span></nav>
      <TitleRow
        title={folder.data?.title ?? "Loading folder…"}
        sub={media.data ? `${count(items.length, "file")}` : undefined}
      />
      <div className="grid">
        <section className="card span-8" aria-label="Files in this folder">
          <ul className="media-list">
            {items.map((m) => <MediaRow key={m.media_id} item={m} onDeleted={media.reload} />)}
          </ul>
          {media.data && items.length === 0 && <Empty>This folder is empty. Add a file to check it here.</Empty>}
        </section>
        <div className="span-4">
          <Dropzone folderId={folderId} compact onDone={media.reload} />
        </div>
      </div>
    </Shell>
  );
}
