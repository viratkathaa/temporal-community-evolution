"""Build the final PDF report from report/final_report.md.

Pipeline:
    markdown -> HTML (via `markdown` lib, tables + fenced code extensions)
                -> PDF (via `xhtml2pdf.pisa`, pure Python, no LaTeX required)

Run:
    python scripts/build_pdf.py
Output:
    report/final_report.pdf
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import markdown
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "report" / "final_report.md"
PDF_PATH = ROOT / "report" / "final_report.pdf"
REPORT_DIR = MD_PATH.parent


CSS = """
@page { size: A4; margin: 2cm; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 11pt; color: #222; line-height: 1.45; }
h1 { font-size: 20pt; color: #1a365d; margin-top: 18pt; }
h2 { font-size: 15pt; color: #2c5282; border-bottom: 1px solid #ddd; padding-bottom: 3pt; margin-top: 16pt; }
h3 { font-size: 12pt; color: #2a4365; margin-top: 12pt; }
p  { margin: 0 0 8pt 0; text-align: justify; }
ul, ol { margin: 0 0 8pt 18pt; }
li { margin-bottom: 3pt; }
code { font-family: "Courier New", monospace; background: #f2f2f2; padding: 1pt 3pt; font-size: 9.5pt; }
pre  { background: #f7f7f7; border: 1px solid #e2e2e2; padding: 8pt; font-size: 9pt; white-space: pre-wrap; }
table { border-collapse: collapse; margin: 10pt 0; font-size: 10pt; width: auto; }
th, td { border: 1px solid #bbb; padding: 4pt 8pt; text-align: left; }
th { background: #eef2f7; }
img { max-width: 16cm; margin: 10pt 0; display: block; }
hr { border: 0; border-top: 1px solid #ccc; margin: 14pt 0; }
a { color: #2b6cb0; text-decoration: none; }
"""

HTML_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"/><style>{css}</style></head>
<body>{body}</body></html>
"""


def _md_to_html(md_text: str) -> str:
    """Convert markdown to HTML with GFM-ish extensions."""
    return markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
        output_format="html5",
    )


def _rewrite_image_paths(html: str) -> str:
    """Make relative image paths resolvable by xhtml2pdf.

    xhtml2pdf opens images via the filesystem. Relative paths in our markdown
    like '../results/figures/foo.png' are relative to the .md file, but the
    library resolves them relative to its own CWD unless we absolutize them
    (or supply a link_callback). Absolutizing is simpler and deterministic.
    """
    def repl(m: re.Match) -> str:
        prefix, src, suffix = m.group(1), m.group(2), m.group(3)
        # Leave absolute / URL sources alone.
        if src.startswith(("http://", "https://", "file:", "/")):
            return m.group(0)
        abs_path = (REPORT_DIR / src).resolve()
        return f'{prefix}{abs_path.as_posix()}{suffix}'

    return re.sub(r'(<img[^>]*?src=")([^"]+)(")', repl, html)


def main() -> int:
    if not MD_PATH.exists():
        print(f"missing {MD_PATH}", file=sys.stderr)
        return 1
    md_text = MD_PATH.read_text(encoding="utf-8")
    html_body = _md_to_html(md_text)
    html_body = _rewrite_image_paths(html_body)
    html = HTML_TEMPLATE.format(css=CSS, body=html_body)

    with open(PDF_PATH, "wb") as f:
        result = pisa.CreatePDF(src=html, dest=f, encoding="utf-8")

    if result.err:
        print(f"PDF build reported {result.err} error(s)", file=sys.stderr)
        return 1
    size_kb = PDF_PATH.stat().st_size / 1024
    print(f"Wrote {PDF_PATH.relative_to(ROOT)}  ({size_kb:,.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
