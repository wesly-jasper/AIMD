"""
Request rate limiting.

WHAT WAS THERE
--------------
``app.main`` constructed a ``Limiter`` with ``default_limits=["120/minute"]`` and
registered its exception handler, but never installed ``SlowAPIMiddleware`` and
decorated no route. slowapi's default limits are enforced *only* by that
middleware, so nothing was limited at all -- including ``/auth/login``, which
could be driven at whatever rate bcrypt allowed.

WHAT IS ENFORCED NOW
--------------------
* Every route: ``settings.rate_limit_default`` via ``SlowAPIMiddleware``.
* Login, upload, analysis, detection and source tracing carry their own,
  tighter limit via ``@limiter.limit``. A decorated route is skipped by the
  middleware, so its own limit replaces the default rather than adding to it.

Counters are keyed on (client address, endpoint) -- the endpoint function, not
the concrete URL, so path parameters cannot be varied to get a fresh budget --
and counted over a moving window. Two consequences an operator
must know about:

1. **Behind a reverse proxy every client shares the proxy's address**, and so
   one bucket. ``X-Forwarded-For`` is deliberately not trusted here: honouring it
   unconditionally lets any caller choose its own key and escape the limit. Run
   uvicorn with ``--proxy-headers --forwarded-allow-ips=<proxy>`` so that the
   proxy -- and only the proxy -- sets the client address.
2. **Storage is in process memory.** Each worker counts separately, and a restart
   resets every counter. With N workers the effective ceiling is N times the
   configured one. A shared store (``storage_uri="redis://..."``) fixes this when
   the deployment needs it.

WHAT IT IS NOT
--------------
A per-address login limit slows password guessing from one address; it does not
stop a distributed guess against one account. Per-account lockout is not added,
because it hands any anonymous caller a way to lock a named investigator out of
their cases. Long, unpublished passwords (enforced at account creation) remain
the actual defence.

The limit is checked after FastAPI has read the request body. For a large upload
that means the body is received before a 429 can be returned -- the same is true
of authentication. A transport-level size cap is Phase 5 work.
"""
import logging
import math
import time

from fastapi import Request
from limits import parse_many
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.core.config import settings
from app.core.exceptions import _envelope, _request_id

logger = logging.getLogger(__name__)


def _validated(name: str, value: str) -> str:
    """
    Refuse an unparseable limit at import.

    slowapi catches the ``ValueError`` from a bad limit string, logs it, and
    installs no limit for that route -- it fails open. A typo in an environment
    variable would silently remove brute-force protection from login.
    """
    try:
        parsed = parse_many(value)
    except ValueError as exc:
        raise RuntimeError(f"{name}={value!r} is not a valid rate limit: {exc}") from exc
    if not parsed:
        raise RuntimeError(f"{name} is empty; a route with no limit must be exempted explicitly")
    return value


DEFAULT_LIMIT = _validated("RATE_LIMIT_DEFAULT", settings.rate_limit_default)
LOGIN_LIMIT = _validated("RATE_LIMIT_LOGIN", settings.rate_limit_login)
SIGNUP_LIMIT = _validated("RATE_LIMIT_SIGNUP", settings.rate_limit_signup)
UPLOAD_LIMIT = _validated("RATE_LIMIT_UPLOAD", settings.rate_limit_upload)
ANALYSIS_LIMIT = _validated("RATE_LIMIT_ANALYSIS", settings.rate_limit_analysis)

# key_style="endpoint": slowapi's default keys counters on the concrete URL, so
# ``/jobs/a`` and ``/jobs/b`` had separate budgets and varying a path parameter
# (or appending junk to it) reset the limit at will. Keying on the endpoint
# makes every path-parameter value share one bucket.
#
# strategy="moving-window": the default fixed window resets on the minute
# boundary, so a caller could spend the full allowance at :59 and again at :00
# -- twice the configured rate in two seconds. The moving window never admits
# more than the configured amount in any interval of the configured length.
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[DEFAULT_LIMIT],
    key_style="endpoint",
    strategy="moving-window",
)


def _retry_after_seconds(request: Request) -> int:
    """
    Seconds until the exhausted window resets, from the limiter's own counters.

    Falls back to 60 if the window cannot be read: a client told to come back in
    a minute is worse served than one told the truth, but far better served than
    one given no ``Retry-After`` and left to retry in a tight loop.
    """
    current = getattr(request.state, "view_rate_limit", None)
    if current is None:
        return 60
    try:
        reset_at, _remaining = limiter.limiter.get_window_stats(current[0], *current[1])
    except Exception:  # storage failure must not turn a 429 into a 500
        logger.warning("Could not read rate-limit window; using fallback Retry-After")
        return 60
    return max(1, math.ceil(reset_at - time.time()))


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """
    429 on the standard error envelope, with ``Retry-After``.

    Synchronous on purpose. ``SlowAPIMiddleware`` looks the handler up by exact
    exception type and, if it finds a coroutine, discards it in favour of
    slowapi's own handler -- whose ``{"error": "..."}`` body is a third error
    shape the client would have to parse.

    The client's address is not echoed: it is the limiter's key, and the caller
    has no use for it.
    """
    retry_after = _retry_after_seconds(request)
    logger.warning(
        "Rate limit %s exceeded on %s %s", exc.detail, request.method, request.url.path
    )
    response = _envelope(
        429,
        "RATE_LIMITED",
        f"Too many requests. Retry after {retry_after} seconds.",
        _request_id(request),
    )
    response.headers["Retry-After"] = str(retry_after)
    return response
