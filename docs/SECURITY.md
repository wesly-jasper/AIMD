# Security

What the backend enforces today, what it deliberately does not, and the gaps
that are known and scheduled. Each claim below points at the code or test that
backs it. Anything not listed here should be assumed absent.

Last reviewed: 2026-10-22.

---

## 1. Authentication

| Aspect | Behaviour | Where |
|---|---|---|
| Scheme | Bearer JWT, HS256, 12 h expiry, no refresh token | `app/core/auth.py` |
| Signing key | `AIMD_JWT_SECRET` from the environment only — no default in source. Fewer than 32 characters is refused. The key that was once committed to this repository (and a few other well-known placeholders) is refused by value, because it is permanently in git history. | `_load_signing_key` |
| Missing key | `APP_ENV` = development/dev/local/test: an ephemeral per-process key is generated and the consequences are logged (restart invalidates tokens; workers reject each other's tokens). Any other `APP_ENV`: startup fails. | `_load_signing_key` |
| Passwords | bcrypt. 12-character minimum at account creation. | `UserStore.create` |
| Accounts | None in source. The first administrator comes from `AIMD_BOOTSTRAP_ADMIN_USERNAME` / `AIMD_BOOTSTRAP_ADMIN_PASSWORD`; with those unset there is no admin. Everyone else creates a basic `user` account via `POST /api/v1/auth/signup`; no route grants `admin`. | `UserStore`, `endpoints/auth.py` |
| Login | Same 401 and message for an unknown user and a wrong password; an unknown user still costs one bcrypt comparison against a dummy hash so response time does not reveal which accounts exist. | `authenticate_user` |
| Identity per request | The token's subject is looked up in the store on every request. The role comes from the store, never from the token, so demotion and deactivation (`is_active=False`) take effect on the next request rather than at token expiry. | `get_current_user` |
| Failure codes | 401 = not authenticated (missing/invalid/expired token, unknown or inactive subject). 403 = authenticated but wrong role. | `get_current_user`, `require_role` |

### Route protection

Authentication is attached **per router** in `app/main.py`
(`dependencies=[Depends(get_current_user)]`), not per endpoint, so a new endpoint
on an existing router cannot be added unprotected by omission.

Public routes, by design: `GET /health`, `GET /api/v1/health`,
`POST /api/v1/auth/login`, `POST /api/v1/auth/signup`, and the OpenAPI/docs pages. `/health` returns no case
data, no identifiers and no exception text.

`tests/integration/test_route_authentication.py` walks the **real route table**
with no token and fails if any route outside that public allow-list answers with
anything but 401. A newly added router without auth fails this test.

### Known gaps

- ~~Users are in memory.~~ Fixed in Phase 4: users persist in the `users`
  table. The bootstrap admin is created once from the environment and never
  re-applied over an existing account (that would undo a password change,
  demotion or deactivation).
- ~~No per-case authorisation.~~ Fixed by the move to a self-service product:
  every route reads only the caller's own media, analyses, folders and audit
  events, and returns 404 (not 403) for anything else. Similarity search was
  also comparing uploads against every account's fingerprints; it is now
  scoped to the owner. Guarded by `tests/integration/test_ownership.py`.
- **Sign-up reveals whether an email has an account** (409 on a taken address),
  bounded by `RATE_LIMIT_SIGNUP` (5/hour per address). No email verification or
  password reset yet: both need an email provider.
- **Audit trail is tamper-evident, not tamper-proof.** `GET /api/v1/audit/verify`
  detects edits, deletions and reordering, but someone with database write
  access can rewrite the whole chain consistently. Truncating the newest events
  is also undetectable internally. Record `head_hash` outside the database to
  close both.
- **No token revocation list.** Deactivating the user is the revocation
  mechanism; individual tokens cannot be revoked.

---

## 2. Rate limiting

`app/core/rate_limit.py`; tests in `tests/integration/test_rate_limits.py`.

| Route | Limit (default) | Setting |
|---|---|---|
| Everything not listed below | 120/minute | `RATE_LIMIT_DEFAULT` |
| `POST /api/v1/auth/login` | 10/minute | `RATE_LIMIT_LOGIN` |
| `POST /api/v1/media/upload` | 30/minute | `RATE_LIMIT_UPLOAD` |
| `POST /api/v1/auth/signup` | 5/hour | `RATE_LIMIT_SIGNUP` |
| `POST /api/v1/analysis/` | 10/minute | `RATE_LIMIT_ANALYSIS` |

- Counted per (client address, **endpoint**) over a **moving window**. Path
  parameters do not create new budgets, and there is no boundary at which a
  second full burst is admitted.
- A route's own limit replaces the default for that route.
- Exceeding a limit returns `429` in the standard error envelope
  (`code: RATE_LIMITED`) with a `Retry-After` header. It is readable by the
  browser (the limiter sits inside CORS).
- A malformed limit string **refuses to start** the application. The underlying
  library would otherwise log it and enforce nothing.

### Operator caveats

1. **Reverse proxies.** Behind a proxy every client has the proxy's address and
   shares one budget. `X-Forwarded-For` is not trusted by the application
   (doing so would let any caller pick its own key). Run uvicorn with
   `--proxy-headers --forwarded-allow-ips=<proxy address>`.
2. **Per-worker, in-memory counters.** With N workers the effective ceiling is
   N × the configured one; a restart resets all counters. Use a shared store
   (Redis) if that matters for the deployment.
3. **Not an account lockout.** A per-address login limit slows guessing from one
   address, not a distributed guess against one account. Account lockout is not
   implemented on purpose: it would let any anonymous caller lock a named
   investigator out. Strong, unpublished passwords are the real defence.
4. **Checked after the body is read.** A large upload is received before a 429
   can be returned, and requests rejected as malformed (422) are not counted.
5. slowapi 0.1.9 calls `asyncio.iscoroutinefunction`, removed in Python 3.16.
   Pin or replace before moving to 3.16.

---

## 3. Evidence handling

| Control | State |
|---|---|
| Uploaded files are never executed | ✅ No code path executes, imports or shells out on upload content. |
| Type check by content, not by name/header | ✅ Magic-byte check in `services/ingestion/media_ingestion.py`; MIME allow-lists in `core/config.py`. |
| SHA-256 recorded at ingest | ✅ |
| Size limit | ⚠️ `MAX_FILE_SIZE_MB` is checked **after** the upload is buffered in memory. A transport-level cap is Phase 5. |
| Originals separate from derived artefacts, read-only, re-verified by hash before analysis | ❌ Phase 5. |
| Test suite isolated from real evidence | ✅ `tests/conftest.py` redirects the database and every storage directory to a temp sandbox before `app` is imported, and aborts the session if they still point at the real store. |

A hash establishes that a file is identical to the one recorded at ingest. It
says nothing about whether the content was truthful or unmanipulated before
acquisition.

---

## 4. Secrets

- No secret has a default value in source. `.env.example` contains placeholders
  only.
- `backend/.env` is untracked and ignored. Its historical copy in commit
  `8161ff0` contained no live secret; the JWT key and demo passwords that *were*
  committed in source are refused by the application (see §1).
- Provider credentials (`GOOGLE_VISION_API_KEY`, `TINEYE_API_KEY`, and the
  Nebius key once added) are read by the backend only and never returned in a
  response. The test suite blanks them so no test can reach a provider.
- Exception handlers return a request id, not exception text; details go to the
  server log.

---

## 5. Not claimed

None of the above makes the platform's output legally admissible, and nothing
in it constitutes an examiner's certification or signature. Those are
procedural questions outside the software.

## Reporting a vulnerability

Open a private security advisory on the repository rather than a public issue.
