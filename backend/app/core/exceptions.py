"""
Custom exceptions and global handlers for AIMD.

Ensures consistent error responses:
{
  "success": false,
  "data": null,
  "error": {"code": "...", "message": "..."},
  "request_id": "..."
}

WHAT ``request_id`` IS FOR, AND WHAT IT WAS
-------------------------------------------
It exists so an operator handed an error response can find the matching log
lines. Each handler used to do ``getattr(request.state, "request_id", str(
uuid.uuid4())[:8])`` while *nothing in the application ever set*
``request.state.request_id`` -- ``set_request_context`` had no caller, so every
log line carried ``req=-`` and every error body carried a freshly minted id that
appeared in no log line at all. The field looked like a correlation handle and
could not be used as one, which is worse than omitting it: an operator would
search the logs for an id that was never written.

``RequestIdMiddleware`` now sets it on both ``request.state`` and the logging
context var, so the id in the body is the id in the logs. The handlers below read
it without a fallback id for exactly that reason: if the middleware is not
installed, the body reports ``"unavailable"`` rather than inventing a handle that
leads nowhere.
"""
import logging

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging import set_request_context

logger = logging.getLogger(__name__)

#: Header the id is echoed on, so an operator can quote it from a failed call.
REQUEST_ID_HEADER = "X-Request-ID"


class RequestIdMiddleware(BaseHTTPMiddleware):
    """
    Assign one id per request, visible to the handlers, the logs and the client.

    The id is generated server-side and an inbound ``X-Request-ID`` is ignored
    deliberately: honouring a client-supplied value would let a caller choose the
    key its own requests are filed under in the audit log, including reusing
    another caller's id.

    The logging context is not cleared on the way out. Isolation between requests
    already comes from the ASGI server running each request in its own task,
    which takes a copy of the context; clearing here would instead blank the id
    while an exception is still propagating outward to the 500 handler -- exactly
    the case where the log line needs it.
    """

    async def dispatch(self, request: Request, call_next):
        request_id = set_request_context()
        # Written into ``scope["state"]``, which the whole middleware stack
        # shares -- including ServerErrorMiddleware, which sits *outside* this
        # middleware and is where ``generic_exception_handler`` runs.
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response


def _request_id(request: Request) -> str:
    """The id this request was logged under, or an explicit admission of none."""
    return getattr(request.state, "request_id", None) or "unavailable"


def _envelope(status_code: int, code: str, message: str, request_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "data": None,
            "error": {"code": code, "message": message},
            "request_id": request_id,
        },
        headers={REQUEST_ID_HEADER: request_id},
    )


class AIMDException(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400):
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class InvalidMediaError(AIMDException):
    def __init__(self, message: str):
        super().__init__("INVALID_MEDIA", message, 400)


class MediaNotFoundError(AIMDException):
    def __init__(self, media_id: str):
        super().__init__("MEDIA_NOT_FOUND", f"Media not found: {media_id}", 404)


class AnalysisNotFoundError(AIMDException):
    def __init__(self, analysis_id: str):
        super().__init__("ANALYSIS_NOT_FOUND", f"Analysis not found: {analysis_id}", 404)


class ModelUnavailableError(AIMDException):
    def __init__(self, model_name: str, reason: str = ""):
        super().__init__(
            "MODEL_UNAVAILABLE",
            f"Model {model_name} unavailable: {reason}",
            503,
        )


async def aimd_exception_handler(request: Request, exc: AIMDException):
    """A condition this application raised deliberately; its message is ours."""
    request_id = _request_id(request)
    logger.warning("AIMD error %s: %s", exc.code, exc.message)
    return _envelope(exc.status_code, exc.code, exc.message, request_id)


async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Unify ``HTTPException`` onto the same envelope as everything else.

    FastAPI's default is ``{"detail": ...}``, so until this was attached the API
    returned two different error shapes depending on which exception type a route
    happened to raise, and a client had to handle both.

    ``exc.detail`` is passed through: these details are written by this codebase
    (see ``app.core.security``, which is careful to name no path) and a 4xx needs
    to say what the caller did wrong. That places a standing requirement on
    route code -- a detail string must not interpolate a filesystem path, a query
    or a credential. ``test_error_envelope.py`` guards the 500 case, where the
    message comes from an exception we did not write and is therefore replaced
    wholesale rather than filtered.
    """
    request_id = _request_id(request)
    logger.warning("HTTP %s on %s", exc.status_code, request.url.path)
    return _envelope(
        exc.status_code, f"HTTP_{exc.status_code}", str(exc.detail), request_id
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Request validation failures, on the same envelope.

    The per-field errors are preserved under ``error.fields`` because a caller
    cannot fix a malformed request without knowing which field was wrong. Pydantic
    includes the offending ``input`` value in each error; it is dropped here,
    since for an upload or a path parameter that value is caller-supplied content
    that has no business being reflected back in an error body.
    """
    request_id = _request_id(request)
    fields = [
        {
            "location": list(err.get("loc", [])),
            "message": err.get("msg", ""),
            "type": err.get("type", ""),
        }
        for err in exc.errors()
    ]
    logger.warning("Validation failed on %s: %d field error(s)", request.url.path, len(fields))
    body = {
        "success": False,
        "data": None,
        "error": {
            "code": "VALIDATION_ERROR",
            "message": "Request validation failed",
            "fields": fields,
        },
        "request_id": request_id,
    }
    # Only the request ID header. Copying another response's headers carried its
    # content-length, which no longer matched this (larger) body.
    return JSONResponse(
        status_code=422, content=body, headers={REQUEST_ID_HEADER: request_id}
    )


async def generic_exception_handler(request: Request, exc: Exception):
    """
    Anything unanticipated.

    The message is a fixed string, never ``str(exc)``: an arbitrary exception's
    text routinely carries absolute filesystem paths, SQL, and occasionally
    connection strings, and this is the one handler whose exception was not
    written by this codebase. The detail goes to the log, under the same
    ``request_id`` the client is given.
    """
    request_id = _request_id(request)
    logger.error("Unhandled %s: %s", type(exc).__name__, exc, exc_info=True)
    return _envelope(500, "INTERNAL_ERROR", "Internal server error", request_id)
