"""
Custom exceptions and global handlers for AIMD.

Ensures consistent error responses:
{
  "success": false,
  "data": null,
  "error": {"code": "...", "message": "..."},
  "request_id": "..."
}
"""
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import uuid
import logging

logger = logging.getLogger(__name__)


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
    request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])
    logger.warning("AIMD error %s: %s (req=%s)", exc.code, exc.message, request_id)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "data": None,
            "error": {"code": exc.code, "message": exc.message},
            "request_id": request_id,
        },
    )


async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "data": None,
            "error": {"code": f"HTTP_{exc.status_code}", "message": str(exc.detail)},
            "request_id": request_id,
        },
    )


async def generic_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])
    logger.error("Unhandled error (req=%s): %s", request_id, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "data": None,
            "error": {"code": "INTERNAL_ERROR", "message": "Internal server error"},
            "request_id": request_id,
        },
    )
