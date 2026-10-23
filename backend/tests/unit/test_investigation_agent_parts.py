"""
The investigation agent's parts: the tool registry, the grounding guard and the
loop's bounds.
"""
import json
from types import SimpleNamespace

import pytest

from app.core.config import settings
from app.db.database import SessionLocal
from app.services.agent import tools as agent_tools
from app.services.agent.grounding import UNVERIFIED, ground
from app.services.agent.nemotron_agent import NemotronAgent
from tests.conftest import TEST_USERNAME
from tests.integration.test_investigation_agent import SOURCE_URL, RecordedClient, _Broken, seed_analysis


@pytest.fixture
def db():
    with SessionLocal() as session:
        yield session


def _ctx(db, analysis_id, owner=TEST_USERNAME, **kwargs):
    return agent_tools.ToolContext(db=db, analysis_id=analysis_id, owner=owner, **kwargs)


# ── Registry ──────────────────────────────────────────────────────────────────

def test_eight_tools_with_strict_schemas_and_one_argument_in_all():
    specs = agent_tools.tool_specs()
    assert len(specs) == 8
    for spec in specs:
        fn = spec["function"]
        assert spec["type"] == "function" and fn["strict"] is True
        assert fn["parameters"]["type"] == "object"
        assert fn["parameters"]["additionalProperties"] is False
    properties = {s["function"]["name"]: set(s["function"]["parameters"]["properties"]) for s in specs}
    # No ids, paths, commands: the only argument any tool takes is one URL.
    assert {n: p for n, p in properties.items() if p} == {"verify_source_match": {"url"}}


def test_every_tool_runs_on_an_owned_analysis_and_labels_its_result(db):
    analysis_id = seed_analysis()
    ctx = _ctx(db, analysis_id)
    for name in agent_tools.TOOLS:
        if name == "verify_source_match":
            continue
        entry = agent_tools.execute(ctx, name, "{}")
        assert "error" not in entry, (name, entry)
        assert entry["category"] in ("FACT", "OBSERVATION")
    assert SOURCE_URL in ctx.known_urls


def test_metadata_withholds_gps_and_detections_withhold_detector_metadata(db):
    ctx = _ctx(db, seed_analysis())
    meta = agent_tools.execute(ctx, "get_metadata", "{}")["result"]
    assert "GPSLatitude" not in meta["exif"] and meta["gps_fields_withheld"] == ["GPSLatitude"]
    assert meta["exif"]["Make"] == "Canon"
    detections = agent_tools.execute(ctx, "list_detections", "{}")["result"]
    assert detections["count"] == 2
    assert "not for the model" not in agent_tools.serialise(detections)


def test_another_users_analysis_is_not_found(db):
    theirs = seed_analysis(owner="someone-else@example.invalid")
    entry = agent_tools.execute(_ctx(db, theirs), "get_analysis_overview", "{}")
    assert entry["error"] == "analysis not found" and "result" not in entry


@pytest.mark.parametrize("name, arguments, expected", [
    ("run_shell", "{}", "unknown tool"),
    ("get_metadata", '{"path": "/etc/passwd"}', "invalid arguments"),
    ("get_metadata", "not json", "invalid arguments"),
    ("verify_source_match", '{"url": "file:///etc/passwd"}', "invalid arguments"),
    ("verify_source_match", "{}", "invalid arguments"),
])
def test_bad_calls_are_reported_to_the_model_not_raised(db, name, arguments, expected):
    entry = agent_tools.execute(_ctx(db, seed_analysis()), name, arguments)
    assert entry["error"].startswith(expected)


class _UrlSource:
    def __init__(self):
        self.queries = []

    def __call__(self):
        return self

    def run(self, query, db):
        self.queries.append(query)
        hit = SimpleNamespace(to_dict=lambda: {
            "url": query.claimed_url, "verification_status": "VERIFIED_MATCH",
            "similarity": 0.97, "verification": {"inliers": 41}, "raw_html": "<html>",
        })
        return SimpleNamespace(status=SimpleNamespace(value="COMPLETED"), reason=None, hits=[hit])


def test_verify_source_match_only_checks_known_urls_within_its_budget(db):
    source = _UrlSource()
    ctx = _ctx(db, seed_analysis(), url_checks_left=1, url_source_factory=source)
    args = json.dumps({"url": SOURCE_URL})

    refused = agent_tools.execute(ctx, "verify_source_match", args)
    assert "url refused" in refused["error"] and source.queries == []

    agent_tools.execute(ctx, "get_sources", "{}")       # now the URL has been seen
    checked = agent_tools.execute(ctx, "verify_source_match", args)
    assert checked["result"]["status"] == "COMPLETED"
    assert checked["result"]["hits"][0]["verification_status"] == "VERIFIED_MATCH"
    assert "raw_html" not in checked["result"]["hits"][0]
    assert source.queries[0].claimed_url == SOURCE_URL

    spent = agent_tools.execute(ctx, "verify_source_match", args)
    assert "budget" in spent["error"] and len(source.queries) == 1


def test_results_are_truncated_for_the_model():
    text = agent_tools.serialise({"x": "y" * (agent_tools.MAX_RESULT_CHARS * 2)})
    assert len(text) <= agent_tools.MAX_RESULT_CHARS + len("...[truncated]")
    assert text.endswith("...[truncated]")


# ── Grounding guard ───────────────────────────────────────────────────────────

RESULTS = {
    "T1": json.dumps({"score": 0.8734, "url": "https://example.org/a", "items": [1, 2, 3],
                      "published_at": "2019-05-04T10:00:00+00:00", "exif": "2019:05:03 18:20:11"}),
}


@pytest.mark.parametrize("claim", [
    "The score was 0.87 [T1].",
    "The score was 87% [T1].",
    "It was published at https://example.org/a on 2019-05-04 [T1].",
    "It was published at 2019-05-04T10:00:00+00:00 [T1].",
    "The camera clock read 2019:05:03 18:20:11 [T1].",
    "There were 3 items [T1].",
])
def test_supported_claims_pass_untouched(claim):
    out = ground(claim, RESULTS)
    assert out.text == claim and out.flags == [] and out.grounded
    assert out.citations == [{"sentence": claim, "tool_call_ids": ["T1"]}]


@pytest.mark.parametrize("claim, flag, value", [
    ("The score was 0.95 [T1].", "unsupported_number", "0.95"),
    ("It came from https://evil.example/x [T1].", "unsupported_url", "https://evil.example/x"),
    ("It was taken on 2017-02-01 [T1].", "unsupported_timestamp", "2017-02-01"),
    ("It is 12 years old [T1].", "unsupported_number", "12"),
])
def test_unsupported_values_are_replaced_and_flagged(claim, flag, value):
    out = ground(claim, RESULTS)
    assert value not in out.text and UNVERIFIED in out.text
    assert {"type": flag, "value": value, "sentence": claim} in out.flags
    assert not out.grounded


def test_unknown_citations_are_removed_and_uncited_claims_flagged():
    out = ground("Score 0.87 [T1, T7]. The image looks authentic.", RESULTS)
    assert "T7" not in out.text and "[T1]" in out.text
    types = [f["type"] for f in out.flags]
    assert types == ["unknown_citation", "uncited"]
    assert not out.grounded


def test_cjk_bracket_citations_count_and_are_normalised():
    # As the live model wrote them, URL and all.
    out = ground("The score was 0.87【T1】. See https://example.org/a【T1、T1】.", RESULTS)
    assert out.text == "The score was 0.87[T1]. See https://example.org/a[T1]."
    assert out.flags == [] and [c["tool_call_ids"] for c in out.citations] == [["T1"], ["T1"]]


def test_think_blocks_headings_and_list_numbers_are_not_claims():
    summary = "<think>secret 42</think>\n**Findings**\n## Sources\n1. The score was 0.87 [T1]."
    out = ground(summary, RESULTS)
    assert "secret" not in out.text and "42" not in out.text
    assert out.flags == []
    assert "1. The score was 0.87 [T1]." in out.text


# ── Loop bounds ───────────────────────────────────────────────────────────────

def _tool_call_response(i):
    return {
        "id": f"r{i}", "object": "chat.completion", "created": 1, "model": "m",
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": f"c{i}", "type": "function",
                            "function": {"name": "get_metadata", "arguments": "{}"}}],
        }}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
    }


def test_the_loop_stops_at_max_steps_and_forbids_tools_on_the_last(db, monkeypatch):
    monkeypatch.setattr(settings, "agent_max_steps", 3)
    fake = RecordedClient([_tool_call_response(i) for i in range(3)])
    outcome = NemotronAgent(client_factory=lambda: fake).investigate(db, seed_analysis(), TEST_USERNAME)
    assert len(fake.requests) == 3
    assert [r["tool_choice"] for r in fake.requests] == ["auto", "auto", "none"]
    assert outcome.status == "UNAVAILABLE" and "no summary within 3 steps" in outcome.error
    assert outcome.usage["total_tokens"] == 33 and len(outcome.tool_calls) == 2


def test_the_loop_times_out(db, monkeypatch):
    monkeypatch.setattr(settings, "agent_timeout_seconds", 60.0)
    now = [0.0]

    def clock():
        now[0] += 25.0  # every reading of the clock is 25 s later
        return now[0]

    fake = RecordedClient([_tool_call_response(i) for i in range(8)])
    outcome = NemotronAgent(client_factory=lambda: fake, clock=clock).investigate(
        db, seed_analysis(), TEST_USERNAME)
    assert outcome.status == "UNAVAILABLE" and "timed out" in outcome.error
    assert len(fake.requests) < 8


def test_a_client_error_is_unavailable_and_an_aimd_bug_is_failed(db, monkeypatch):
    analysis_id = seed_analysis()
    down = NemotronAgent(client_factory=lambda: _Broken(TimeoutError("read timed out")))
    outcome = down.investigate(db, analysis_id, TEST_USERNAME)
    assert outcome.status == "UNAVAILABLE" and "TimeoutError" in outcome.error

    def broken_specs():
        raise KeyError("registry")
    monkeypatch.setattr(agent_tools, "tool_specs", broken_specs)
    outcome = NemotronAgent(client_factory=lambda: RecordedClient([])).investigate(
        db, analysis_id, TEST_USERNAME)
    assert outcome.status == "FAILED" and "KeyError" in outcome.error


def test_disabled_or_keyless_agent_never_builds_a_client(db, monkeypatch):
    def no_client():
        raise AssertionError("client built")

    assert "NEBIUS_API_KEY" in NemotronAgent().unavailable_reason()
    monkeypatch.setattr(settings, "agent_enabled", False)
    outcome = NemotronAgent(client_factory=no_client).investigate(db, seed_analysis(), TEST_USERNAME)
    assert outcome.status == "UNAVAILABLE" and "disabled" in outcome.error
