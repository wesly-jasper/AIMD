"""
Every error leaves the API in one shape, and a 500 reveals nothing about the host.

Two defects are guarded here:

1. The API returned two different error bodies depending on which exception type
   a route happened to raise -- the envelope for ``AIMDException``, FastAPI's
   ``{"detail": ...}`` for ``HTTPException`` and validation errors, because only
   two of the four handlers were attached.

2. ``request_id`` looked like a handle for finding the matching log lines while
   nothing in the application ever set ``request.state.request_id``, so each
   handler minted a fresh random id that was written to no log line. A
   correlation key that leads nowhere is worse than none: it sends an operator
   searching for an id that does not exist.
"""
import logging

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.core.exceptions import REQUEST_ID_HEADER, MediaNotFoundError
from app.main import app

# raise_server_exceptions=False is required to exercise the 500 handler: with the
# default the TestClient re-raises instead of returning the response a real
# client would receive, so the body under test is never produced.
client = TestClient(app, raise_server_exceptions=False)


@pytest.fixture(scope="module", autouse=True)
def _error_routes():
    """Routes that raise each error class, registered only for this module."""

    @app.get("/__test__/unhandled")
    def _unhandled():
        # A message of exactly the kind that must not reach a client: an absolute
        # path, a connection string and a credential-shaped token.
        raise RuntimeError(
            "could not open C:\\Users\\Administrator\\.ssh\\id_rsa while "
            "connecting to postgresql://aimd:s3cr3t-p4ssw0rd@10.0.0.4:5432/aimd"
        )

    @app.get("/__test__/http-error")
    def _http_error():
        raise HTTPException(status_code=403, detail="Evidence is sealed")

    @app.get("/__test__/aimd-error")
    def _aimd_error():
        raise MediaNotFoundError("abc123")

    @app.get("/__test__/validated")
    def _validated(count: int):
        return {"count": count}

    yield

    app.router.routes = [
        r for r in app.router.routes
        if not getattr(r, "path", "").startswith("/__test__/")
    ]
    app.openapi_schema = None


def _assert_envelope(body: dict, code: str) -> None:
    assert body["success"] is False
    assert body["data"] is None
    assert body["error"]["code"] == code
    assert isinstance(body["error"]["message"], str) and body["error"]["message"]
    assert body["request_id"] and body["request_id"] != "unavailable"


def test_unhandled_exception_returns_the_envelope_and_leaks_nothing():
    response = client.get("/__test__/unhandled")

    assert response.status_code == 500
    body = response.json()
    _assert_envelope(body, "INTERNAL_ERROR")

    # The message is a fixed string, not str(exc). This is the one handler whose
    # exception text was not written by this codebase, so it is replaced wholesale
    # rather than filtered.
    assert body["error"]["message"] == "Internal server error"

    serialised = response.text
    for leak in (
        "C:\\Users",
        "Administrator",
        "id_rsa",
        "postgresql://",
        "s3cr3t-p4ssw0rd",
        "10.0.0.4",
        "RuntimeError",
        "Traceback",
        "site-packages",
        "/app/",
    ):
        assert leak not in serialised, f"500 body leaked {leak!r}"

    # No stack frames and no local variables under any key.
    assert "traceback" not in {k.lower() for k in body}
    assert set(body) == {"success", "data", "error", "request_id"}


def test_500_request_id_matches_the_logged_one(caplog):
    """The id handed to the client is the id written to the log."""
    with caplog.at_level(logging.ERROR, logger="app.core.exceptions"):
        response = client.get("/__test__/unhandled")

    request_id = response.json()["request_id"]
    assert response.headers[REQUEST_ID_HEADER] == request_id

    records = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert records, "the 500 was not logged at all"
    # The ContextFilter stamps the id the request was logged under; it must be the
    # same value the client was given, or the handle is useless.
    assert any(getattr(r, "request_id", None) == request_id for r in records), (
        f"no log record carried request_id={request_id}; "
        f"got {[getattr(r, 'request_id', None) for r in records]}"
    )
    # The detail that was withheld from the client is present in the log.
    assert any("id_rsa" in r.getMessage() for r in records)


def test_http_exception_uses_the_envelope_not_fastapi_detail():
    response = client.get("/__test__/http-error")

    assert response.status_code == 403
    body = response.json()
    _assert_envelope(body, "HTTP_403")
    assert body["error"]["message"] == "Evidence is sealed"
    # FastAPI's default shape must be gone, not merely supplemented.
    assert "detail" not in body


def test_aimd_exception_uses_the_envelope():
    response = client.get("/__test__/aimd-error")

    assert response.status_code == 404
    body = response.json()
    _assert_envelope(body, "MEDIA_NOT_FOUND")
    assert "abc123" in body["error"]["message"]


def test_validation_error_names_the_field_without_echoing_the_value():
    response = client.get("/__test__/validated", params={"count": "'; DROP TABLE--"})

    assert response.status_code == 422
    body = response.json()
    _assert_envelope(body, "VALIDATION_ERROR")
    assert "detail" not in body

    fields = body["error"]["fields"]
    assert any("count" in field["location"] for field in fields)
    # Pydantic puts the offending value in each error under "input". For an upload
    # or a path parameter that is caller-supplied content, so it is dropped rather
    # than reflected back.
    assert "DROP TABLE" not in response.text
    assert all(set(field) == {"location", "message", "type"} for field in fields)


def test_validation_error_content_length_matches_the_body():
    """
    Regression: the 422 copied another response's headers, content-length
    included, so it declared fewer bytes than it sent. The test client reads the
    whole body regardless; a real server aborts the response mid-write.
    """
    response = client.get("/__test__/validated", params={"count": "x"})

    assert response.status_code == 422
    assert int(response.headers["content-length"]) == len(response.content)
    assert response.headers[REQUEST_ID_HEADER] == response.json()["request_id"]


def test_every_response_carries_a_correlation_header():
    ok = client.get("/health")
    assert ok.headers[REQUEST_ID_HEADER]

    err = client.get("/__test__/http-error")
    assert err.headers[REQUEST_ID_HEADER] == err.json()["request_id"]

    # Distinct requests get distinct ids, or the handle cannot identify one call.
    assert ok.headers[REQUEST_ID_HEADER] != err.headers[REQUEST_ID_HEADER]
