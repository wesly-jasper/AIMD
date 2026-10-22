import { useCallback, useEffect, useRef, useState } from "react";
import { getBlobUrl, getJson } from "./api";

export interface Resource<T> {
  data: T | null;
  error: Error | null;
  loading: boolean;
  reload: () => void;
}

/**
 * GET a path and keep its result. A reload keeps the previous data on screen
 * until the new data lands, so the layout does not jump.
 */
export function useResource<T>(path: string | null): Resource<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState<boolean>(path !== null);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    if (path === null) return;
    let live = true;
    setLoading(true);
    getJson<T>(path)
      .then((d) => { if (live) { setData(d); setError(null); } })
      .catch((e: Error) => { if (live) setError(e); })
      .finally(() => { if (live) setLoading(false); });
    return () => { live = false; };
  }, [path, tick]);

  const reload = useCallback(() => setTick((t) => t + 1), []);
  return { data, error, loading, reload };
}

/** An authenticated object URL for evidence bytes, revoked when no longer shown. */
export function useBlobUrl(path: string | null): { url: string | null; error: boolean } {
  const [url, setUrl] = useState<string | null>(null);
  const [error, setError] = useState(false);
  const current = useRef<string | null>(null);

  useEffect(() => {
    if (!path) return;
    let live = true;
    setError(false);
    getBlobUrl(path)
      .then((u) => {
        if (!live) { URL.revokeObjectURL(u); return; }
        current.current = u;
        setUrl(u);
      })
      .catch(() => { if (live) setError(true); });
    return () => {
      live = false;
      if (current.current) URL.revokeObjectURL(current.current);
      current.current = null;
    };
  }, [path]);

  return { url, error };
}
