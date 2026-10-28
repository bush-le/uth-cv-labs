"""Render the Lab04 Markdown report as standalone HTML with embedded images."""

import base64
import re
from pathlib import Path

import markdown


def main() -> None:
    """Convert the saved Markdown; the Markdown remains the editable report source."""
    root = Path(__file__).resolve().parents[2]
    report = root / "docs/labs/lab04/LAB_04_REPORT.md"
    converter = markdown.Markdown(
        extensions=["fenced_code", "toc"],
        extension_configs={"toc": {"toc_depth": "2-3"}},
    )
    body = converter.convert(report.read_text(encoding="utf-8"))

    def embed_image(match: re.Match[str]) -> str:
        path = (report.parent / match.group(1)).resolve()
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'src="data:image/png;base64,{data}"'

    body = re.sub(r'src="([^"]+\.png)"', embed_image, body)
    css = Path(__file__).with_name("report.css").read_text(encoding="utf-8")
    html = (
        '<!doctype html><html lang="vi"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Lab04 — Wavelet Hash</title><style>{css}</style></head>"
        '<body><div class="layout"><nav aria-label="Mục lục"><strong>MỤC LỤC</strong>'
        f"{converter.toc}</nav><main>{body}</main></div></body></html>"
    )
    report.with_suffix(".html").write_text(html, encoding="utf-8")
    print("Lab04 Markdown rendered to standalone HTML.")


if __name__ == "__main__":
    main()
