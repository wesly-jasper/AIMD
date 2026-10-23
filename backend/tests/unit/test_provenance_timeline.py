"""The provenance timeline, built from stored sources and the upload's EXIF."""
from datetime import datetime, timezone
from types import SimpleNamespace

from app.services.provenance.timeline import build_timeline, parse_stated_date


def source(status="VERIFIED_MATCH", published=datetime(2007, 12, 10, 0, 2, 15),
           provider="wikimedia_commons", url="https://commons.wikimedia.org/wiki/File:A.jpg",
           **citation):
    return SimpleNamespace(
        id="src-1", url=url, provider=provider, domain="commons.wikimedia.org",
        verification_status=status, source_timestamp=published,
        citation={
            "uploader": "Lucag",
            "file_url": "https://upload.wikimedia.org/wikipedia/commons/a/ab/A.jpg",
            "uploaded_at_meaning": "upload of the current file version on Wikimedia Commons",
            "retrieved_at": "2026-10-23T03:50:00+00:00",
            "date_time_original": "2007-07-21",
            **citation,
        },
    )


def test_a_verified_commons_source_gives_three_cited_events():
    timeline = build_timeline([source()])
    by_type = {e["type"]: e for e in timeline["events"]}
    assert set(by_type) == {"PUBLISHED", "RETRIEVED", "METADATA"}

    published = by_type["PUBLISHED"]
    assert published["timestamp"] == "2007-12-10T00:02:15+00:00"
    assert published["citation"]["url"] == "https://commons.wikimedia.org/wiki/File:A.jpg"
    assert "Lucag" in published["description"]
    assert published["earliest_known_appearance"] is True

    assert by_type["RETRIEVED"]["citation"]["url"].startswith("https://upload.wikimedia.org/")
    assert by_type["RETRIEVED"]["basis"] == "aimd_record"
    assert by_type["METADATA"]["timestamp"] == "2007-07-21"
    assert by_type["METADATA"]["precision"] == "day"
    assert by_type["METADATA"]["basis"] == "uploader_statement"
    assert all(e["citation"] for e in timeline["events"])


def test_events_are_in_time_order():
    stamps = [e["timestamp"] for e in build_timeline([source()])["events"]]
    assert stamps == sorted(stamps)


def test_the_earliest_appearance_is_never_called_the_original():
    timeline = build_timeline([source()])
    assert timeline["earliest_known_appearance"]["type"] == "PUBLISHED"
    assert "original" not in timeline["earliest_known_appearance"]["description"].lower()
    assert "not a claim that the image originated" in timeline["note"]


def test_unverified_sources_contribute_nothing():
    timeline = build_timeline([source(status="SIMILAR_UNVERIFIED")])
    assert timeline["events"] == [] and timeline["earliest_known_appearance"] is None


def test_a_metadata_date_is_never_the_earliest_appearance():
    # The EXIF date is years earlier, but it is a claim about capture, not appearance.
    timeline = build_timeline([source()], exif={"DateTimeOriginal": "2001:05:06 10:11:12"})
    first = timeline["events"][0]
    assert first["type"] == "METADATA" and first["citation"]["field"] == "EXIF DateTimeOriginal"
    assert first["timezone"] == "not stated (camera local time)"
    assert timeline["earliest_known_appearance"]["type"] == "PUBLISHED"


def test_an_archive_capture_can_be_the_earliest_appearance():
    archived = source(provider="wayback", published=datetime(2005, 1, 1),
                      url="https://web.archive.org/web/20050101000000/https://example.org/a",
                      retrieved_at=None, date_time_original=None)
    timeline = build_timeline([source(), archived])
    earliest = timeline["earliest_known_appearance"]
    assert earliest["type"] == "ARCHIVED"
    assert earliest["citation"]["url"].startswith("https://web.archive.org/")


def test_exif_alone_gives_a_timeline_with_no_appearance():
    timeline = build_timeline([], exif={"DateTimeDigitized": "2016:07:21 20:11:03"})
    [event] = timeline["events"]
    assert event["type"] == "METADATA" and event["timestamp"] == "2016-07-21T20:11:03"
    assert timeline["earliest_known_appearance"] is None


def test_an_unparseable_stated_date_is_kept_undated_and_last():
    timeline = build_timeline([source(date_time_original="circa 1900")])
    last = timeline["events"][-1]
    assert last["type"] == "METADATA" and last["timestamp"] is None
    assert last["stated_value"] == "circa 1900"


def test_parse_stated_date_forms():
    assert parse_stated_date("2016:07:21 20:11:03") == ("2016-07-21T20:11:03", "second")
    assert parse_stated_date("2007-07-21") == ("2007-07-21", "day")
    assert parse_stated_date("2016-07-21 20:11") == ("2016-07-21T20:11:00", "second")
    assert parse_stated_date("July 2016") == (None, None)


def test_a_timezone_aware_publication_time_is_normalised_to_utc():
    from datetime import timedelta

    aware = datetime(2007, 12, 10, 2, 2, 15, tzinfo=timezone(timedelta(hours=2)))
    published = build_timeline([source(published=aware)])["earliest_known_appearance"]
    assert published["timestamp"] == "2007-12-10T00:02:15+00:00"
