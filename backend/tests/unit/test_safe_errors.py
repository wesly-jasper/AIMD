"""safe_reason / scrub: what a failure may say to the client."""
from sqlalchemy.exc import IntegrityError, PendingRollbackError

from app.core.safe_errors import DATABASE_ERROR, safe_reason, scrub


def test_windows_and_posix_paths_are_replaced():
    assert scrub(r"Unable to read image: D:\projects\AIMD-2\backend\uploads\abc.jpg") == "Unable to read image: <file>"
    assert scrub("cannot open /srv/aimd/uploads/abc.jpg for reading") == "cannot open <file> for reading"
    assert scrub(r"missing \\fileserver\share\evidence.png") == "missing <file>"


def test_paths_with_spaces_are_replaced_whole():
    raw = r"Unable to read image: C:\Users\Jane Doe\AppData\Local\Temp\uploads\abc.jpg was empty"
    assert scrub(raw) == "Unable to read image: <file> was empty"
    assert scrub("cannot open /home/jane doe/uploads/abc.jpg now") == "cannot open <file> now"
    # In a list of failures, the quote ends the path.
    assert scrub(r"['metadata: Unable to read image: D:\a b\uploads\x.jpg', 'detection: ok']") == (
        "['metadata: Unable to read image: <file>', 'detection: ok']"
    )


def test_urls_and_plain_reasons_are_kept():
    text = "fetch of https://upload.wikimedia.org/a/b/c.jpg failed: 404"
    assert scrub(text) == text
    assert scrub("no source URL was supplied") == "no source URL was supplied"
    assert scrub("ratio 1/2 of tiles") == "ratio 1/2 of tiles"


def test_sql_and_parameters_are_cut():
    raw = (
        "(builtins.TypeError) Object of type IFDRational is not JSON serializable\n"
        "[SQL: INSERT INTO media_metadata (id, exif_data) VALUES (?, ?)]\n"
        "[parameters: [{'width': 4680, 'raw_metadata': {...}}]]"
    )
    out = scrub(raw)
    assert "INSERT" not in out and "parameters" not in out
    assert out == "(builtins.TypeError) Object of type IFDRational is not JSON serializable"


def test_a_rolled_back_session_is_reported_as_a_database_error():
    raw = "This Session's transaction has been rolled back due to a previous exception during flush."
    assert scrub(raw) == DATABASE_ERROR


def test_database_exceptions_never_expose_their_text():
    exc = IntegrityError("INSERT INTO media (id) VALUES (?)", {"id": "x"}, Exception("UNIQUE failed"))
    assert safe_reason(exc) == DATABASE_ERROR
    assert safe_reason(PendingRollbackError("rolled back")) == DATABASE_ERROR


def test_long_text_is_one_line_and_bounded():
    out = scrub("line one\nline two " + "x" * 500)
    assert "\n" not in out and len(out) <= 240 and out.endswith("…")


def test_empty_and_missing_text_pass_through():
    assert scrub(None) is None and scrub("") == ""
    assert safe_reason(ValueError()) == "ValueError"
