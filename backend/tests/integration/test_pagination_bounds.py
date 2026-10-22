"""
List endpoints refuse unbounded or nonsensical pages.

``/analysis/`` and the since-removed ``/jobs/`` took ``limit: int = 20`` with no bound, so a single
request with ``limit=1000000`` asked the database for every row; ``/media/``
was already bounded and is held to the same rule here so the three cannot drift.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

LIST_ROUTES = ["/api/v1/analysis/", "/api/v1/cases/", "/api/v1/media/"]


@pytest.mark.parametrize("path", LIST_ROUTES)
@pytest.mark.parametrize(
    "query",
    ["limit=1000000", "limit=201", "limit=0", "limit=-1", "offset=-1"],
)
def test_out_of_range_paging_is_refused(path, query):
    response = client.get(f"{path}?{query}")
    assert response.status_code == 422, f"{path}?{query} -> {response.status_code}"
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.parametrize("path", LIST_ROUTES)
def test_in_range_paging_is_accepted(path):
    response = client.get(f"{path}?limit=200&offset=0")
    assert response.status_code == 200, response.text
    assert len(response.json()["items"]) <= 200
