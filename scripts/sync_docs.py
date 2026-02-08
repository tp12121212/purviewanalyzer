#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Optional

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

SOURCES = {
    "tutorial": {
        "url": "https://microsoft.github.io/presidio/tutorial/",
        "output": "content/docs/tutorial.md",
    },
    "installation": {
        "url": "https://microsoft.github.io/presidio/installation/",
        "output": "content/docs/installation.md",
    },
    "faq": {
        "url": "https://microsoft.github.io/presidio/faq/",
        "output": "content/docs/faq.md",
    },
}

LOCAL_LINKS = {
    "https://microsoft.github.io/presidio/tutorial/": "tutorial.md",
    "https://microsoft.github.io/presidio/installation/": "installation.md",
    "https://microsoft.github.io/presidio/faq/": "faq.md",
}


def _fetch(url: str) -> str:
    response = requests.get(url, timeout=30, headers={"User-Agent": "presidio-demo-sync"})
    response.raise_for_status()
    return response.text


def _extract_main_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    main = (
        soup.find("article", class_="md-content__inner")
        or soup.find("div", class_="md-content")
        or soup.find("article", class_="markdown-body")
        or soup.find("main")
    )
    if main is None:
        return html

    for selector in [
        "nav",
        "header",
        "footer",
        "div.md-sidebar",
        "div.md-header",
        "div.md-footer",
        "div.md-search",
        "a.headerlink",
        "div.md-content__button",
    ]:
        for node in main.select(selector):
            node.decompose()

    return str(main)


def _convert_to_markdown(html: str) -> str:
    markdown = md(html, heading_style="ATX")
    markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()
    return markdown


def _rewrite_links(markdown: str) -> str:
    for src, dst in LOCAL_LINKS.items():
        markdown = markdown.replace(f"({src})", f"({dst})")
    return markdown


def _write_markdown(output_path: Path, source_url: str, content: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    header = f"<!-- Source: {source_url} | Last synced: {timestamp} -->\n\n"
    output_path.write_text(header + content + "\n", encoding="utf-8")


def sync_page(source: dict) -> None:
    html = _fetch(source["url"])
    main_html = _extract_main_html(html)
    markdown = _convert_to_markdown(main_html)

    markdown = _rewrite_links(markdown)
    _write_markdown(Path(source["output"]), source["url"], markdown)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync local docs content.")
    parser.add_argument(
        "--page",
        choices=list(SOURCES.keys()),
        help="Sync a single page",
    )
    args = parser.parse_args()

    if args.page:
        sync_page(SOURCES[args.page])
    else:
        for source in SOURCES.values():
            sync_page(source)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
