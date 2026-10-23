"""
The stored Markdown report as a standalone HTML document.

Only the Markdown the report generator writes is supported: headings, bullet
lists, blockquotes, rules, pipe tables, paragraphs and the inline forms
``**bold**``, ``*italic*`` and ``code``. Every character is HTML-escaped before
any markup is added, so a filename, a source title or an EXIF value can never
become markup -- the report quotes attacker-controlled text by design.

The document carries its own print stylesheet: printing it (or "Save as PDF")
is how the PDF form of the report is produced.
"""
from __future__ import annotations

import html
import re

_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])")
_BULLET = re.compile(r"^\s*[-*]\s+")
_TABLE_RULE = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?$")

STYLE = """
body { font: 14px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; color: #111;
       max-width: 860px; margin: 32px auto; padding: 0 24px; }
h1 { font-size: 22px; } h2 { font-size: 17px; margin-top: 28px; } h3 { font-size: 15px; }
code { font: 12.5px ui-monospace, Consolas, monospace; background: #f1f1f1; padding: 1px 4px;
       border-radius: 3px; overflow-wrap: anywhere; }
blockquote { margin: 12px 0; padding: 8px 14px; border-left: 3px solid #b8860b; background: #faf6ec; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 13px; }
th, td { border: 1px solid #ddd; padding: 5px 8px; text-align: left; vertical-align: top; }
th { background: #f6f6f6; }
hr { border: 0; border-top: 1px solid #ddd; margin: 24px 0; }
@media print { body { margin: 0; max-width: none; } h2 { break-after: avoid; } tr { break-inside: avoid; } }
"""


def _inline(text: str) -> str:
    out = html.escape(text, quote=True)
    # Code first, so emphasis markers inside code stay literal.
    codes: list[str] = []

    def stash(m: re.Match) -> str:
        codes.append(f"<code>{m.group(1)}</code>")
        return f"\x00{len(codes) - 1}\x00"

    out = _CODE.sub(stash, out)
    out = _BOLD.sub(r"<strong>\1</strong>", out)
    out = _ITALIC.sub(r"<em>\1</em>", out)
    return re.sub(r"\x00(\d+)\x00", lambda m: codes[int(m.group(1))], out)


def _cells(row: str) -> list[str]:
    return [c.strip() for c in row.strip().strip("|").split("|")]


def render_report_html(markdown: str, title: str = "AIMD forensic analysis report") -> str:
    lines = (markdown or "").splitlines()
    body: list[str] = []
    paragraph: list[str] = []
    i = 0

    def flush() -> None:
        # Consecutive lines (the header's "  " line breaks) stay one paragraph.
        if paragraph:
            body.append("<p>" + "<br>".join(_inline(p.rstrip()) for p in paragraph) + "</p>")
            paragraph.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            flush()
            i += 1
        elif m := re.match(r"^(#{1,6})\s+(.*)$", stripped):
            flush()
            level = len(m.group(1))
            body.append(f"<h{level}>{_inline(m.group(2))}</h{level}>")
            i += 1
        elif re.fullmatch(r"-{3,}|\*{3,}", stripped):
            flush()
            body.append("<hr>")
            i += 1
        elif stripped.startswith("|") and i + 1 < len(lines) and _TABLE_RULE.match(lines[i + 1].strip()):
            flush()
            head = "".join(f"<th>{_inline(c)}</th>" for c in _cells(stripped))
            rows = []
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in _cells(lines[i])) + "</tr>")
                i += 1
            body.append(f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>")
        elif re.match(r"^[-*]\s+", stripped):
            flush()
            items = []
            while i < len(lines) and _BULLET.match(lines[i]):
                items.append(f"<li>{_inline(_BULLET.sub('', lines[i]).rstrip())}</li>")
                i += 1
            body.append(f"<ul>{''.join(items)}</ul>")
        elif stripped.startswith(">"):
            flush()
            quoted = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quoted.append(_inline(lines[i].strip()[1:].strip()))
                i += 1
            body.append(f"<blockquote>{'<br>'.join(quoted)}</blockquote>")
        else:
            paragraph.append(stripped)
            i += 1
    flush()

    return (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<title>{html.escape(title)}</title><style>{STYLE}</style></head>"
        f"<body>{''.join(body)}</body></html>\n"
    )
