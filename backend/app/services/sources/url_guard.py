"""
Fetching a URL a user supplied, without letting it reach anything private.

A user-supplied URL is an SSRF vector: ``http://169.254.169.254/`` (cloud
metadata), ``http://localhost:8000/admin``, or a public hostname whose DNS
answer is ``10.0.0.5``. ``GuardedFetcher`` therefore:

  * accepts only http(s) on ports 80/443, with no credentials in the URL;
  * resolves the host and refuses it unless EVERY address is globally routable
    (so a name that also resolves to a private address is refused);
  * checks the address the connection actually reached before reading the body,
    which closes the gap between that check and the connect (DNS rebinding);
  * follows redirects itself, re-checking every hop, at most ``max_redirects``;
  * caps the body at ``max_bytes`` and the whole fetch at ``deadline`` seconds.
"""
from __future__ import annotations

import ipaddress
import socket
import time
from dataclasses import dataclass
from typing import Callable, Optional
from urllib.parse import urljoin, urlsplit

import httpx

ALLOWED_SCHEMES = {"http", "https"}
ALLOWED_PORTS = {80, 443}
USER_AGENT = "AIMD-source-verification/1.0 (+https://github.com/wesly-jasper/AIMD)"


class BlockedURL(ValueError):
    """The URL, or where it leads, is not allowed to be fetched."""


class FetchError(RuntimeError):
    """The fetch was allowed but did not succeed (status, size, time, type)."""


@dataclass
class Fetched:
    url: str              # after redirects
    status: int
    content_type: str
    body: bytes
    peer_ip: Optional[str]
    redirects: list[str]


Resolver = Callable[[str, int], list[str]]


def _system_resolver(host: str, port: int) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise FetchError(f"cannot resolve {host}: {exc}") from exc
    return sorted({info[4][0] for info in infos})


def is_public_address(ip: str) -> bool:
    """Globally routable unicast only; an IPv4-mapped IPv6 address is judged as IPv4."""
    addr = ipaddress.ip_address(ip.split("%", 1)[0])
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        addr = addr.ipv4_mapped
    return addr.is_global and not (addr.is_multicast or addr.is_reserved)


def check_url(url: str, resolver: Resolver = _system_resolver) -> tuple[str, int, list[str]]:
    """``(host, port, addresses)`` for an allowed URL; raises ``BlockedURL`` otherwise."""
    if not isinstance(url, str) or len(url) > 2048:
        raise BlockedURL("URL missing or longer than 2048 characters")
    parts = urlsplit(url.strip())
    if parts.scheme.lower() not in ALLOWED_SCHEMES:
        raise BlockedURL(f"scheme {parts.scheme or '(none)'!r} is not http(s)")
    if parts.username or parts.password:
        raise BlockedURL("credentials in the URL are not allowed")
    host = parts.hostname
    if not host:
        raise BlockedURL("URL has no host")
    try:
        port = parts.port or (443 if parts.scheme.lower() == "https" else 80)
    except ValueError as exc:
        raise BlockedURL(f"invalid port: {exc}") from exc
    if port not in ALLOWED_PORTS:
        raise BlockedURL(f"port {port} is not allowed")
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    addresses = [str(literal)] if literal else resolver(host, port)
    if not addresses:
        raise FetchError(f"{host} has no addresses")
    private = [a for a in addresses if not is_public_address(a)]
    if private:
        raise BlockedURL(f"{host} resolves to a non-public address ({', '.join(private)})")
    return host, port, addresses


class GuardedFetcher:
    """``transport`` and ``resolver`` exist for tests; real fetches use neither."""

    def __init__(
        self,
        *,
        max_bytes: int = 15 * 1024 * 1024,
        deadline: float = 10.0,
        max_redirects: int = 3,
        resolver: Resolver = _system_resolver,
        transport: Optional[httpx.BaseTransport] = None,
    ):
        self.max_bytes = max_bytes
        self.deadline = deadline
        self.max_redirects = max_redirects
        self._resolver = resolver
        self._injected_transport = transport is not None
        self._client = httpx.Client(
            headers={"User-Agent": USER_AGENT},
            timeout=httpx.Timeout(deadline, connect=min(5.0, deadline)),
            follow_redirects=False,
            transport=transport,
            trust_env=False,  # never route through a proxy from the environment
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "GuardedFetcher":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _peer_ip(self, response: httpx.Response) -> Optional[str]:
        stream = response.extensions.get("network_stream")
        addr = stream.get_extra_info("server_addr") if stream is not None else None
        return addr[0] if addr else None

    def fetch(self, url: str, *, accept: tuple[str, ...] = ("image/", "text/html")) -> Fetched:
        started = time.monotonic()
        redirects: list[str] = []
        current = url
        for _hop in range(self.max_redirects + 1):
            _, _, addresses = check_url(current, self._resolver)
            with self._client.stream("GET", current) as response:
                peer = self._peer_ip(response)
                if peer is None and not self._injected_transport:
                    raise BlockedURL("could not confirm the address connected to")
                if peer is not None and not is_public_address(peer):
                    raise BlockedURL(f"connected to a non-public address ({peer})")
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise FetchError(f"HTTP {response.status_code} without a Location")
                    redirects.append(current)
                    current = urljoin(current, location)
                    continue
                if response.status_code != 200:
                    raise FetchError(f"HTTP {response.status_code}")
                content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
                if not content_type.startswith(accept):
                    raise FetchError(f"unexpected content type {content_type or '(none)'}")
                declared = response.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > self.max_bytes:
                    raise FetchError(f"body of {declared} bytes exceeds {self.max_bytes}")
                chunks, size = [], 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > self.max_bytes:
                        raise FetchError(f"body exceeds {self.max_bytes} bytes")
                    if time.monotonic() - started > self.deadline:
                        raise FetchError(f"fetch exceeded {self.deadline:.0f} s")
                    chunks.append(chunk)
                return Fetched(current, response.status_code, content_type, b"".join(chunks),
                               peer or addresses[0], redirects)
        raise FetchError(f"more than {self.max_redirects} redirects")
