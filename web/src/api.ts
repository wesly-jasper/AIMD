// The one place the app talks to the backend.
//
// The access token lives in sessionStorage: it survives a reload but not the
// tab, and it is never written anywhere else. Every request carries it as a
// Bearer header; a 401 clears it and returns the app to the sign-in screen.

const BASE = "/api/v1";
const TOKEN_KEY = "aimd.session";

export class ApiError extends Error {
  readonly status: number;
  readonly requestId: string | null;
  constructor(status: number, message: string, requestId: string | null) {
    super(message);
    this.status = status;
    this.requestId = requestId;
  }
}

type Session = { token: string };

export function readSession(): Session | null {
  try {
    const raw = sessionStorage.getItem(TOKEN_KEY);
    return raw ? (JSON.parse(raw) as Session) : null;
  } catch {
    return null;
  }
}

export function writeSession(session: Session | null): void {
  try {
    if (session) sessionStorage.setItem(TOKEN_KEY, JSON.stringify(session));
    else sessionStorage.removeItem(TOKEN_KEY);
  } catch {
    // Storage blocked: the session lasts only as long as this page.
  }
}

const unauthorisedListeners = new Set<() => void>();
export function onUnauthorised(fn: () => void): () => void {
  unauthorisedListeners.add(fn);
  return () => unauthorisedListeners.delete(fn);
}

/** The backend's single error envelope: { error: { message, ... }, request_id }. */
async function errorFrom(res: Response): Promise<ApiError> {
  let message = `Request failed (${res.status})`;
  let requestId = res.headers.get("x-request-id");
  try {
    const body = await res.json();
    const err = body?.error;
    if (err?.message) message = err.message;
    if (Array.isArray(err?.fields) && err.fields.length) {
      message = err.fields.map((f: { message: string }) => f.message).join("; ");
    }
    requestId = body?.request_id ?? requestId;
  } catch {
    // Not JSON: keep the status line.
  }
  return new ApiError(res.status, message, requestId);
}

async function send(path: string, init: RequestInit = {}): Promise<Response> {
  const session = readSession();
  const headers = new Headers(init.headers);
  if (session) headers.set("Authorization", `Bearer ${session.token}`);
  let res: Response;
  try {
    res = await fetch(BASE + path, { ...init, headers });
  } catch {
    throw new ApiError(0, "The service could not be reached.", null);
  }
  if (res.status === 401 && session) {
    writeSession(null);
    unauthorisedListeners.forEach((fn) => fn());
  }
  if (!res.ok) throw await errorFrom(res);
  return res;
}

export async function getJson<T>(path: string): Promise<T> {
  return (await send(path)).json() as Promise<T>;
}

export async function postJson<T>(path: string, body: unknown): Promise<T> {
  const res = await send(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json() as Promise<T>;
}

export async function postForm<T>(path: string, form: FormData): Promise<T> {
  return (await send(path, { method: "POST", body: form })).json() as Promise<T>;
}

/** Evidence bytes need the Bearer header, so they cannot be a plain <img src>. */
export async function getBlobUrl(path: string): Promise<string> {
  const blob = await (await send(path)).blob();
  return URL.createObjectURL(blob);
}

async function tokenFrom(path: string, body: unknown): Promise<string> {
  const res = await fetch(BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).catch(() => {
    throw new ApiError(0, "The service could not be reached.", null);
  });
  if (!res.ok) throw await errorFrom(res);
  return ((await res.json()) as { access_token: string }).access_token;
}

export const login = (email: string, password: string) =>
  tokenFrom("/auth/login", { username: email, password });

export const signup = (email: string, password: string) =>
  tokenFrom("/auth/signup", { email, password });

export async function deleteMedia(mediaId: string): Promise<void> {
  await send(`/media/${mediaId}`, { method: "DELETE" });
}
