"""Source discovery: connectors that look for where an image was published."""


def default_source_connectors() -> list:
    """Every connector, enabled or not: a disabled one reports UNAVAILABLE."""
    from app.services.sources.url_verification import UrlVerificationSource
    from app.services.sources.wayback import WaybackSource
    from app.services.sources.commons_live import CommonsSha1Source
    from app.services.sources.wikimedia_corpus import WikimediaCorpusSource

    # Wayback last: it looks up the URLs the others verified.
    return [WikimediaCorpusSource(), CommonsSha1Source(), UrlVerificationSource(), WaybackSource()]
