# API

Base URL `/api/v1`. Every route except the public ones needs
`Authorization: Bearer <token>`. Each account sees only its own data: a file,
result or folder that belongs to someone else answers **404**, not 403.

The full machine-readable spec is `docs/openapi.json`
(regenerate: `python scripts/generate_openapi.py` from `backend/`).

## Account (public except `/me`)

| Method | Path | Does |
|---|---|---|
| POST | `/auth/signup` | Create an account (email, password ≥ 12 chars). Returns a token. Rate-limited. |
| POST | `/auth/login` | Sign in. Returns a token. Rate-limited. |
| GET | `/auth/me` | The signed-in account. |

## Files

| Method | Path | Does |
|---|---|---|
| POST | `/media/upload` | Upload an image, video or audio file (optional `case_id`). Daily per-account limit. |
| GET | `/media/` | Your files, newest first, each with its latest result. `case_id` filters by folder. |
| GET | `/media/usage` | Uploads used today against the daily limit. |
| GET | `/media/{id}` | One file's details. |
| GET | `/media/{id}/download` | The original bytes. |
| DELETE | `/media/{id}` | Delete the file, its results and its derived frames. |

## Results

| Method | Path | Does |
|---|---|---|
| POST | `/analysis/` | Run every check on a file (`media_id`). |
| GET | `/analysis/` | Your results. `media_id` filters by file. |
| GET | `/analysis/{id}` | One result: assessment, findings, observations, every check's availability and run state, run provenance. |

## Folders

| Method | Path | Does |
|---|---|---|
| POST | `/cases/` | Create a folder. |
| GET | `/cases/` | Your folders, with file counts. |
| GET | `/cases/{id}` | One folder. |

## Audit

| Method | Path | Does |
|---|---|---|
| GET | `/audit/events` | Your own audit events. |
| GET | `/audit/verify` | Verify the hash chain. Admin only. |

## System (public)

`GET /health` and `GET /api/v1/health`: status and database connectivity.
