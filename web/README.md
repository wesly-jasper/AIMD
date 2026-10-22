# AIMD Sentinel — web

A self-service app: sign up, drop a photo or video, read in plain words whether
it shows signs of editing. React 19 + TypeScript + Vite, no UI library, dark.

## Run

```bash
# 1. Backend (from backend/)
uvicorn app.main:app --port 8000
#    Optional: AIMD_JWT_SECRET (32+ chars) so sign-ins survive restarts.
#    Optional: DAILY_UPLOAD_LIMIT (default 50 per account per 24 h).

# 2. This app
npm install
npm run dev          # http://localhost:5173, proxies /api to 127.0.0.1:8000
```

Create an account on the sign-in screen. `AIMD_API=http://host:port npm run dev`
points the proxy elsewhere; `npm run build` type-checks and writes `dist/`.

## Screens

| Route | Answers |
| --- | --- |
| sign in | Create an account or sign in with email and password. |
| `#/` Check media | What do I want to check? Drop a file: it uploads, runs every check, opens the result. Shows today's usage against the daily limit. |
| `#/results/:id` | Does this show signs of editing, and how sure is that? Plain verdict, marked areas on the image, "What we checked", and Technical details (finding with supporting / contradicting evidence, every check's capability and run state, run provenance). |
| `#/history` | What have I checked? Filter by result, search, delete. |
| `#/folders` | Optional grouping of files. |

## Rules the interface keeps

- **Amber means one thing:** a check found signs here. It never decorates chrome.
- **"No signs found" is never "authentic".** The caveat is part of the verdict.
- **Weak evidence says it's weak**, and names the innocent cause (re-saving).
- **A check that did not run is "Not available"**, hatched, never "found nothing".
- **Every account sees only its own files** — enforced by the API, not the UI.
- The session token lives in `sessionStorage`; identity comes from `/auth/me`.

Wording lives in one file, `src/plain.ts`, so the honesty rules are in one place.
Design: tasteskill app style *true-black-flat* (seed 753521769), amber data hue.
